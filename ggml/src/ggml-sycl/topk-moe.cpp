#അന്തര്ഭാവ <cfloat>
#അന്തര്ഭാവ <initializer_list>
#അന്തര്ഭാവ <vector>

#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

ഘടന ggml_sycl_topk_moe_args {
    തർക്കം sigmoid{};
    തർക്കം softmax{};
    തർക്കം delayed_softmax{};
    തർക്കം prob_bias{};
    തർക്കം norm{};
    തർക്കം scale{};
};

ഘടന topk_moe_config {
    തർക്കം use_sigmoid;
    തർക്കം with_norm;
    തർക്കം delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <പൂർണ്ണസംഖ്യ experts_per_thread, തർക്കം use_limit>
സ്ഥിതം വരിയിൽ ശൂന്യം softmax_warp_inplace(ദശാംശം (&vals)[experts_per_thread], സ്ഥിരം പൂർണ്ണസംഖ്യ limit, സ്ഥിരം പൂർണ്ണസംഖ്യ lane) {
    ദശാംശം max_val = -INFINITY;
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < experts_per_thread; i++) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ  idx    = lane + i * WARP_SIZE;
        സ്ഥിരം തർക്കം active = !use_limit || (idx < limit);
        എങ്കിൽ (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    ദശാംശം sum = 0.f;
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < experts_per_thread; i++) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ  idx    = lane + i * WARP_SIZE;
        സ്ഥിരം തർക്കം active = !use_limit || (idx < limit);
        എങ്കിൽ (active) {
            സ്ഥിരം ദശാംശം val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } അല്ലെങ്കിൽ {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    സ്ഥിരം ദശാംശം inv_sum = 1.0f / sum;
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < experts_per_thread; i++) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ idx = lane + i * WARP_SIZE;
        എങ്കിൽ (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <പൂർണ്ണസംഖ്യ experts_per_thread, തർക്കം use_limit>
സ്ഥിതം വരിയിൽ ശൂന്യം sigmoid_warp_inplace(ദശാംശം (&vals)[experts_per_thread], സ്ഥിരം പൂർണ്ണസംഖ്യ limit, സ്ഥിരം പൂർണ്ണസംഖ്യ lane) {
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < experts_per_thread; i++) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ  idx    = lane + i * WARP_SIZE;
        സ്ഥിരം തർക്കം active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + sycl::exp(-vals[i])) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax/sigmoid over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as a fusion of the softmax->top-k->get_rows pipeline for MoE models.
    One sub-group handles one row/token, mirroring topk_moe_cuda's one-warp-per-row layout.
*/
template <പൂർണ്ണസംഖ്യ n_experts>
സ്ഥിതം ശൂന്യം topk_moe_kernel(സ്ഥിരം ദശാംശം * __restrict__ logits,
                            ദശാംശം * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            സ്ഥിരം പൂർണ്ണസംഖ്യ                  n_rows,
                            സ്ഥിരം പൂർണ്ണസംഖ്യ                  n_expert_used,
                            സ്ഥിരം ദശാംശം                clamp_val,
                            സ്ഥിരം ദശാംശം                scale_val,
                            സ്ഥിരം topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    സ്ഥിരം പൂർണ്ണസംഖ്യ row       = item_ct1.get_group(0);
    എങ്കിൽ (row >= n_rows) {
        തിരികെ;
    }
    സ്ഥിരം പൂർണ്ണസംഖ്യ lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr പൂർണ്ണസംഖ്യ experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    ദശാംശം wt[experts_per_thread];
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_experts; i += WARP_SIZE) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    എങ്കിൽ (!config.delayed_softmax) {
        എങ്കിൽ (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, അസത്യം>(wt, n_experts, lane);
        } അല്ലെങ്കിൽ {
            softmax_warp_inplace<experts_per_thread, അസത്യം>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < experts_per_thread; i++) {
        എങ്കിൽ (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    ദശാംശം wt_sum = 0.f;
    ദശാംശം output_weights[experts_per_thread];
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    സ്ഥിരം sycl::sub_group sg = item_ct1.get_sub_group();

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < n_expert_used; k++) {
        ദശാംശം max_val    = wt[0];
        പൂർണ്ണസംഖ്യ   max_expert = lane;
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < experts_per_thread; i++) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ expert = lane + i * WARP_SIZE;
            എങ്കിൽ ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            സ്ഥിരം ദശാംശം val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            സ്ഥിരം പൂർണ്ണസംഖ്യ   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            എങ്കിൽ (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        എങ്കിൽ ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        എങ്കിൽ ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        എങ്കിൽ ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            എങ്കിൽ (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    എങ്കിൽ (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        സ്ഥിരം ദശാംശം inv = 1.0f / wt_sum;
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    എങ്കിൽ (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, സത്യം>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < experts_per_thread; i++) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ idx = i * WARP_SIZE + lane;
        എങ്കിൽ (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <പൂർണ്ണസംഖ്യ n_experts>
സ്ഥിതം ശൂന്യം launch_topk_moe(queue_ptr stream, സ്ഥിരം ദശാംശം * logits, ദശാംശം * weights, int32_t * ids, പൂർണ്ണസംഖ്യ n_rows,
                            പൂർണ്ണസംഖ്യ n_expert_used, ദശാംശം clamp_val, ദശാംശം scale_val, സ്ഥിരം topk_moe_config & config) {
    സ്ഥിരം sycl::range<1> block_dims(WARP_SIZE);
    സ്ഥിരം sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

സ്ഥിതം ശൂന്യം ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  സ്ഥിരം ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  സ്ഥിരം ggml_tensor *             clamp,
                                  സ്ഥിരം ggml_tensor *             scale,
                                  സ്ഥിരം ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_experts     = logits->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_rows        = logits->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    സ്ഥിരം ദശാംശം * logits_d  = (സ്ഥിരം ദശാംശം *) logits->data;
    ദശാംശം *       weights_d = (ദശാംശം *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    സ്ഥിരം തർക്കം  with_norm = clamp != nullptr;
    സ്ഥിരം ദശാംശം clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    സ്ഥിരം ദശാംശം scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    തിരഞ്ഞെടുപ്പ് (n_experts) {
        അവസ്ഥ 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            നിർത്തുക;
        അവസ്ഥ 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            നിർത്തുക;
        അവസ്ഥ 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            നിർത്തുക;
        അവസ്ഥ 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            നിർത്തുക;
        അവസ്ഥ 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            നിർത്തുക;
        അവസ്ഥ 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            നിർത്തുക;
        അവസ്ഥ 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            നിർത്തുക;
        അവസ്ഥ 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            നിർത്തുക;
        അവസ്ഥ 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            നിർത്തുക;
        അവസ്ഥ 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ASSERT(അസത്യം && "fatal error");
    }
}

സ്ഥിതം തർക്കം ggml_sycl_should_use_topk_moe(സ്ഥിരം ggml_tensor * gating_op, സ്ഥിരം ggml_tensor * weights,
                                          സ്ഥിരം ggml_tensor * logits, സ്ഥിരം ggml_tensor * ids) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_expert = ids->nb[1] / ids->nb[0];
    എങ്കിൽ ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (gating_op->op == GGML_OP_SOFT_MAX) {
        ദശാംശം scale    = 1.0f;
        ദശാംശം max_bias = 0.0f;

        memcpy(&scale, (സ്ഥിരം ദശാംശം *) gating_op->op_params + 0, വലുപ്പം(ദശാംശം));
        memcpy(&max_bias, (സ്ഥിരം ദശാംശം *) gating_op->op_params + 1, വലുപ്പം(ദശാംശം));

        എങ്കിൽ (!ggml_is_contiguous(gating_op->src[0])) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (scale != 1.0f || max_bias != 0.0f) {
            തിരികെ അസത്യം;
        }
        // don't fuse when masks or sinks are present
        എങ്കിൽ (gating_op->src[1] || gating_op->src[2]) {
            തിരികെ അസത്യം;
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (gating_op->op == GGML_OP_UNARY) {
        എങ്കിൽ (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
സ്ഥിതം തർക്കം ggml_sycl_topk_moe_fusion(സ്ഥിരം ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    സ്ഥിരം പൂർണ്ണസംഖ്യ      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    എങ്കിൽ (nodes[കണ്ണി_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = സത്യം;
    }

    എങ്കിൽ (nodes[കണ്ണി_idx]->op == GGML_OP_UNARY) {
        എങ്കിൽ (ggml_get_unary_op(nodes[കണ്ണി_idx]) != GGML_UNARY_OP_SIGMOID) {
            തിരികെ അസത്യം;
        }
        args.sigmoid = സത്യം;
    }

    എങ്കിൽ (nodes[കണ്ണി_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = സത്യം;
    }

    കണ്ണി_idx++;

    എങ്കിൽ (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        എങ്കിൽ (കണ്ണി_idx >= n_nodes || nodes[കണ്ണി_idx]->op != GGML_OP_RESHAPE ||
            nodes[കണ്ണി_idx]->src[0] != nodes[കണ്ണി_idx - 1]) {
            തിരികെ അസത്യം;
        }
        ggml_tensor * probs_reshaped = nodes[കണ്ണി_idx];
        കണ്ണി_idx++;

        എങ്കിൽ (കണ്ണി_idx >= n_nodes) {
            തിരികെ അസത്യം;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        എങ്കിൽ (nodes[കണ്ണി_idx]->op == GGML_OP_ADD && nodes[കണ്ണി_idx]->src[0] == nodes[കണ്ണി_idx - 2]) {
            args.prob_bias = സത്യം;
            കണ്ണി_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        എങ്കിൽ (കണ്ണി_idx >= n_nodes || nodes[കണ്ണി_idx]->op != GGML_OP_ARGSORT) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (args.prob_bias && nodes[കണ്ണി_idx]->src[0] != nodes[കണ്ണി_idx - 1]) {
            തിരികെ അസത്യം;
        } അല്ലെങ്കിൽ എങ്കിൽ (!args.prob_bias && nodes[കണ്ണി_idx]->src[0] != nodes[കണ്ണി_idx - 2]) {
            തിരികെ അസത്യം;
        }

        കണ്ണി_idx++;

        // ARGSORT -> VIEW
        എങ്കിൽ (കണ്ണി_idx >= n_nodes || nodes[കണ്ണി_idx]->op != GGML_OP_VIEW ||
            nodes[കണ്ണി_idx]->src[0] != nodes[കണ്ണി_idx - 1]) {
            തിരികെ അസത്യം;
        }
        കണ്ണി_idx++;

        എങ്കിൽ (കണ്ണി_idx >= n_nodes || nodes[കണ്ണി_idx]->op != GGML_OP_GET_ROWS) {
            തിരികെ അസത്യം;
        }

        // GET_ROWS
        എങ്കിൽ (nodes[കണ്ണി_idx]->src[0] != probs_reshaped || nodes[കണ്ണി_idx]->src[1] != nodes[കണ്ണി_idx - 1]) {
            തിരികെ അസത്യം;
        }
        കണ്ണി_idx++;
    } അല്ലെങ്കിൽ എങ്കിൽ (args.delayed_softmax) {
        എങ്കിൽ (കണ്ണി_idx - 2 < 0) {
            തിരികെ അസത്യം;
        }
        ggml_tensor * probs_reshaped = nodes[കണ്ണി_idx - 2];

        // VIEW -> ARGSORT
        എങ്കിൽ (കണ്ണി_idx >= n_nodes || nodes[കണ്ണി_idx]->op != GGML_OP_VIEW ||
            nodes[കണ്ണി_idx]->src[0] != nodes[കണ്ണി_idx - 1]) {
            തിരികെ അസത്യം;
        }
        കണ്ണി_idx++;

        // GET_ROWS
        എങ്കിൽ (കണ്ണി_idx >= n_nodes || nodes[കണ്ണി_idx]->src[1] != nodes[കണ്ണി_idx - 1] ||
            nodes[കണ്ണി_idx]->src[0] != probs_reshaped) {
            തിരികെ അസത്യം;
        }
        കണ്ണി_idx++;

        സ്ഥിതം സ്ഥിരം std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        വേണ്ടി (സ്ഥിരം ggml_op op : remaining_ops) {
            എങ്കിൽ (കണ്ണി_idx >= n_nodes || nodes[കണ്ണി_idx]->op != op || nodes[കണ്ണി_idx]->src[0] != nodes[കണ്ണി_idx - 1]) {
                തിരികെ അസത്യം;
            }
            കണ്ണി_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    എങ്കിൽ (കണ്ണി_idx >= n_nodes) {
        തിരികെ സത്യം;
    }

    എങ്കിൽ (nodes[കണ്ണി_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        സ്ഥിതം സ്ഥിരം std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = സത്യം;
        വേണ്ടി (സ്ഥിരം ggml_op op : norm_ops) {
            എങ്കിൽ (nodes[കണ്ണി_idx]->op == op && nodes[കണ്ണി_idx]->src[0] == nodes[കണ്ണി_idx - 1]) {
                കണ്ണി_idx++;
            } അല്ലെങ്കിൽ {
                args.norm = അസത്യം;
                തിരികെ സത്യം;
            }
        }

        // DIV <- CLAMP, RESHAPE
        എങ്കിൽ (nodes[കണ്ണി_idx]->op != GGML_OP_DIV || nodes[കണ്ണി_idx]->src[1] != nodes[കണ്ണി_idx - 1] ||
            nodes[കണ്ണി_idx]->src[0] != nodes[കണ്ണി_idx - 3]) {
            args.norm = അസത്യം;
            തിരികെ സത്യം;
        }
        കണ്ണി_idx++;

        എങ്കിൽ (nodes[കണ്ണി_idx]->op != GGML_OP_RESHAPE || nodes[കണ്ണി_idx]->src[0] != nodes[കണ്ണി_idx - 1]) {
            args.norm = അസത്യം;
            തിരികെ സത്യം;
        }
        കണ്ണി_idx++;
    }

    എങ്കിൽ (nodes[കണ്ണി_idx]->op == GGML_OP_SCALE && nodes[കണ്ണി_idx]->src[0] == nodes[കണ്ണി_idx - 1]) {
        args.scale = സത്യം;
    }

    തിരികെ സത്യം;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
സ്ഥിതം തർക്കം ggml_sycl_check_fusion_memory_ranges(സ്ഥിരം ggml_cgraph * cgraph, സ്ഥിരം പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                 സ്ഥിരം പൂർണ്ണസംഖ്യ കണ്ണി_എണ്ണം, സ്ഥിരം പൂർണ്ണസംഖ്യ * out_nodes, സ്ഥിരം പൂർണ്ണസംഖ്യ out_എണ്ണം,
                                                 സ്ഥിരം തർക്കം is_topk_moe = അസത്യം) {
    auto nodes_overlap = [&](സ്ഥിരം ggml_tensor * a, സ്ഥിരം ggml_tensor * b) {
        സ്ഥിരം int64_t a_start = (int64_t) a->data;
        സ്ഥിരം int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        സ്ഥിരം int64_t b_start = (int64_t) b->data;
        സ്ഥിരം int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        എങ്കിൽ ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            തിരികെ സത്യം;
        }

        തിരികെ അസത്യം;
    };

    തർക്കം is_ok = സത്യം;
    // exception for topk-moe, as each row is read entirely before writing
    എങ്കിൽ (ggml_nrows(cgraph->nodes[കണ്ണി_idx]) == 1 && is_topk_moe) {
        തിരികെ സത്യം;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < out_എണ്ണം; ++i) {
        സ്ഥിരം ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = കണ്ണി_idx; j < കണ്ണി_idx + കണ്ണി_എണ്ണം; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            വേണ്ടി (പൂർണ്ണസംഖ്യ src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                സ്ഥിരം ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                എങ്കിൽ (!src || src->op == GGML_OP_NONE) {
                    തുടരുക;
                }

                എങ്കിൽ (nodes_overlap(dst, src)) {
                    തർക്കം found = അസത്യം;

                    വേണ്ടി (പൂർണ്ണസംഖ്യ k = കണ്ണി_idx; k < j; ++k) {
                        എങ്കിൽ (cgraph->nodes[k] == src) {
                            found = സത്യം;
                            നിർത്തുക;
                        }
                    }

                    എങ്കിൽ (!found) {
                        is_ok = അസത്യം;
                        നിർത്തുക;
                    }
                }
            }
        }
    }

    തിരികെ is_ok;
}

പൂർണ്ണസംഖ്യ ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ i) {
    എങ്കിൽ (!g_ggml_sycl_enable_fusion) {
        തിരികെ 0;
    }

    തിരികെ ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

പൂർണ്ണസംഖ്യ ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ i) {
    ggml_tensor * കണ്ണി = cgraph->nodes[i];

    എങ്കിൽ (കണ്ണി->op != GGML_OP_UNARY && കണ്ണി->op != GGML_OP_SOFT_MAX && കണ്ണി->op != GGML_OP_ARGSORT) {
        തിരികെ 0;
    }

    ggml_sycl_topk_moe_args args;
    എങ്കിൽ (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        തിരികെ 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    എങ്കിൽ (args.prob_bias) {
        തിരികെ 0;
    }

    സ്ഥിരം ggml_tensor * logits  = കണ്ണി->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    സ്ഥിരം ggml_tensor * clamp   = nullptr;
    സ്ഥിരം ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    പൂർണ്ണസംഖ്യ                  out_nodes[2];

    എങ്കിൽ (!args.delayed_softmax) {
        സ്ഥിരം ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        എങ്കിൽ (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (പൂർണ്ണസംഖ്യ) ops.size() - 3];
        }
        എങ്കിൽ (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (പൂർണ്ണസംഖ്യ) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (പൂർണ്ണസംഖ്യ) ops.size() - 1];
        out_nodes[1] = i + (പൂർണ്ണസംഖ്യ) ops.size() - 1;

        എങ്കിൽ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(കണ്ണി, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (പൂർണ്ണസംഖ്യ) ops.size(), out_nodes, 2, /*is_topk_moe=*/സത്യം)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            തിരികെ (പൂർണ്ണസംഖ്യ) ops.size() - 1;
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        സ്ഥിരം ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        എങ്കിൽ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (പൂർണ്ണസംഖ്യ) ops.size(), out_nodes, 2, /*is_topk_moe=*/സത്യം)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            തിരികെ (പൂർണ്ണസംഖ്യ) ops.size() - 1;
        }
    }

    തിരികെ 0;
}
