#ଅନ୍ତର୍ଭାଵ <cfloat>
#ଅନ୍ତର୍ଭାଵ <initializer_list>
#ଅନ୍ତର୍ଭାଵ <vector>

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

ସଂରଚନା ggml_sycl_topk_moe_args {
    ତାର୍କିକ sigmoid{};
    ତାର୍କିକ softmax{};
    ତାର୍କିକ delayed_softmax{};
    ତାର୍କିକ prob_bias{};
    ତାର୍କିକ norm{};
    ତାର୍କିକ scale{};
};

ସଂରଚନା topk_moe_config {
    ତାର୍କିକ use_sigmoid;
    ତାର୍କିକ with_norm;
    ତାର୍କିକ delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା experts_per_thread, ତାର୍କିକ use_limit>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ softmax_warp_inplace(ଦଶମିକ (&vals)[experts_per_thread], ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା limit, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane) {
    ଦଶମିକ max_val = -INFINITY;
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < experts_per_thread; i++) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  idx    = lane + i * WARP_SIZE;
        ଅଚଳ ତାର୍କିକ active = !use_limit || (idx < limit);
        ଯଦି (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    ଦଶମିକ sum = 0.f;
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < experts_per_thread; i++) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  idx    = lane + i * WARP_SIZE;
        ଅଚଳ ତାର୍କିକ active = !use_limit || (idx < limit);
        ଯଦି (active) {
            ଅଚଳ ଦଶମିକ val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } ଅନ୍ୟଥା {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    ଅଚଳ ଦଶମିକ inv_sum = 1.0f / sum;
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < experts_per_thread; i++) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = lane + i * WARP_SIZE;
        ଯଦି (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା experts_per_thread, ତାର୍କିକ use_limit>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ sigmoid_warp_inplace(ଦଶମିକ (&vals)[experts_per_thread], ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା limit, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane) {
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < experts_per_thread; i++) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  idx    = lane + i * WARP_SIZE;
        ଅଚଳ ତାର୍କିକ active = !use_limit || (idx < limit);
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
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_experts>
ସ୍ଥିର ଶୂନ୍ୟ topk_moe_kernel(ଅଚଳ ଦଶମିକ * __restrict__ logits,
                            ଦଶମିକ * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  n_rows,
                            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  n_expert_used,
                            ଅଚଳ ଦଶମିକ                clamp_val,
                            ଅଚଳ ଦଶମିକ                scale_val,
                            ଅଚଳ topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row       = item_ct1.get_group(0);
    ଯଦି (row >= n_rows) {
        ଫେରନ୍ତୁ;
    }
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    ଦଶମିକ wt[experts_per_thread];
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_experts; i += WARP_SIZE) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    ଯଦି (!config.delayed_softmax) {
        ଯଦି (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, ମିଥ୍ୟା>(wt, n_experts, lane);
        } ଅନ୍ୟଥା {
            softmax_warp_inplace<experts_per_thread, ମିଥ୍ୟା>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < experts_per_thread; i++) {
        ଯଦି (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    ଦଶମିକ wt_sum = 0.f;
    ଦଶମିକ output_weights[experts_per_thread];
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ଅଚଳ sycl::sub_group sg = item_ct1.get_sub_group();

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < n_expert_used; k++) {
        ଦଶମିକ max_val    = wt[0];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା   max_expert = lane;
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < experts_per_thread; i++) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା expert = lane + i * WARP_SIZE;
            ଯଦି ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            ଅଚଳ ଦଶମିକ val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            ଯଦି (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        ଯଦି ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        ଯଦି ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        ଯଦି ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            ଯଦି (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    ଯଦି (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        ଅଚଳ ଦଶମିକ inv = 1.0f / wt_sum;
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    ଯଦି (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, ସତ୍ୟ>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < experts_per_thread; i++) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = i * WARP_SIZE + lane;
        ଯଦି (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_experts>
ସ୍ଥିର ଶୂନ୍ୟ launch_topk_moe(queue_ptr stream, ଅଚଳ ଦଶମିକ * logits, ଦଶମିକ * weights, int32_t * ids, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows,
                            ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_expert_used, ଦଶମିକ clamp_val, ଦଶମିକ scale_val, ଅଚଳ topk_moe_config & config) {
    ଅଚଳ sycl::range<1> block_dims(WARP_SIZE);
    ଅଚଳ sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ଅଚଳ ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ଅଚଳ ggml_tensor *             clamp,
                                  ଅଚଳ ggml_tensor *             scale,
                                  ଅଚଳ ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_experts     = logits->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows        = logits->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    ଅଚଳ ଦଶମିକ * logits_d  = (ଅଚଳ ଦଶମିକ *) logits->data;
    ଦଶମିକ *       weights_d = (ଦଶମିକ *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    ଅଚଳ ତାର୍କିକ  with_norm = clamp != nullptr;
    ଅଚଳ ଦଶମିକ clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ଅଚଳ ଦଶମିକ scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    ବାଛ (n_experts) {
        ସ୍ଥିତି 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ASSERT(ମିଥ୍ୟା && "fatal error");
    }
}

ସ୍ଥିର ତାର୍କିକ ggml_sycl_should_use_topk_moe(ଅଚଳ ggml_tensor * gating_op, ଅଚଳ ggml_tensor * weights,
                                          ଅଚଳ ggml_tensor * logits, ଅଚଳ ggml_tensor * ids) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_expert = ids->nb[1] / ids->nb[0];
    ଯଦି ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (gating_op->op == GGML_OP_SOFT_MAX) {
        ଦଶମିକ scale    = 1.0f;
        ଦଶମିକ max_bias = 0.0f;

        memcpy(&scale, (ଅଚଳ ଦଶମିକ *) gating_op->op_params + 0, ଆକାର(ଦଶମିକ));
        memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) gating_op->op_params + 1, ଆକାର(ଦଶମିକ));

        ଯଦି (!ggml_is_contiguous(gating_op->src[0])) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (scale != 1.0f || max_bias != 0.0f) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        // don't fuse when masks or sinks are present
        ଯଦି (gating_op->src[1] || gating_op->src[2]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    } ଅନ୍ୟଥା ଯଦି (gating_op->op == GGML_OP_UNARY) {
        ଯଦି (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ସ୍ଥିର ତାର୍କିକ ggml_sycl_topk_moe_fusion(ଅଚଳ ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    ଯଦି (nodes[ଗଣ୍ଠି_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = ସତ୍ୟ;
    }

    ଯଦି (nodes[ଗଣ୍ଠି_idx]->op == GGML_OP_UNARY) {
        ଯଦି (ggml_get_unary_op(nodes[ଗଣ୍ଠି_idx]) != GGML_UNARY_OP_SIGMOID) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        args.sigmoid = ସତ୍ୟ;
    }

    ଯଦି (nodes[ଗଣ୍ଠି_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = ସତ୍ୟ;
    }

    ଗଣ୍ଠି_idx++;

    ଯଦି (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        ଯଦି (ଗଣ୍ଠି_idx >= n_nodes || nodes[ଗଣ୍ଠି_idx]->op != GGML_OP_RESHAPE ||
            nodes[ଗଣ୍ଠି_idx]->src[0] != nodes[ଗଣ୍ଠି_idx - 1]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ggml_tensor * probs_reshaped = nodes[ଗଣ୍ଠି_idx];
        ଗଣ୍ଠି_idx++;

        ଯଦି (ଗଣ୍ଠି_idx >= n_nodes) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        ଯଦି (nodes[ଗଣ୍ଠି_idx]->op == GGML_OP_ADD && nodes[ଗଣ୍ଠି_idx]->src[0] == nodes[ଗଣ୍ଠି_idx - 2]) {
            args.prob_bias = ସତ୍ୟ;
            ଗଣ୍ଠି_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        ଯଦି (ଗଣ୍ଠି_idx >= n_nodes || nodes[ଗଣ୍ଠି_idx]->op != GGML_OP_ARGSORT) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (args.prob_bias && nodes[ଗଣ୍ଠି_idx]->src[0] != nodes[ଗଣ୍ଠି_idx - 1]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        } ଅନ୍ୟଥା ଯଦି (!args.prob_bias && nodes[ଗଣ୍ଠି_idx]->src[0] != nodes[ଗଣ୍ଠି_idx - 2]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଗଣ୍ଠି_idx++;

        // ARGSORT -> VIEW
        ଯଦି (ଗଣ୍ଠି_idx >= n_nodes || nodes[ଗଣ୍ଠି_idx]->op != GGML_OP_VIEW ||
            nodes[ଗଣ୍ଠି_idx]->src[0] != nodes[ଗଣ୍ଠି_idx - 1]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଗଣ୍ଠି_idx++;

        ଯଦି (ଗଣ୍ଠି_idx >= n_nodes || nodes[ଗଣ୍ଠି_idx]->op != GGML_OP_GET_ROWS) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // GET_ROWS
        ଯଦି (nodes[ଗଣ୍ଠି_idx]->src[0] != probs_reshaped || nodes[ଗଣ୍ଠି_idx]->src[1] != nodes[ଗଣ୍ଠି_idx - 1]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଗଣ୍ଠି_idx++;
    } ଅନ୍ୟଥା ଯଦି (args.delayed_softmax) {
        ଯଦି (ଗଣ୍ଠି_idx - 2 < 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ggml_tensor * probs_reshaped = nodes[ଗଣ୍ଠି_idx - 2];

        // VIEW -> ARGSORT
        ଯଦି (ଗଣ୍ଠି_idx >= n_nodes || nodes[ଗଣ୍ଠି_idx]->op != GGML_OP_VIEW ||
            nodes[ଗଣ୍ଠି_idx]->src[0] != nodes[ଗଣ୍ଠି_idx - 1]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଗଣ୍ଠି_idx++;

        // GET_ROWS
        ଯଦି (ଗଣ୍ଠି_idx >= n_nodes || nodes[ଗଣ୍ଠି_idx]->src[1] != nodes[ଗଣ୍ଠି_idx - 1] ||
            nodes[ଗଣ୍ଠି_idx]->src[0] != probs_reshaped) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଗଣ୍ଠି_idx++;

        ସ୍ଥିର ଅଚଳ std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        ପାଇଁ (ଅଚଳ ggml_op op : remaining_ops) {
            ଯଦି (ଗଣ୍ଠି_idx >= n_nodes || nodes[ଗଣ୍ଠି_idx]->op != op || nodes[ଗଣ୍ଠି_idx]->src[0] != nodes[ଗଣ୍ଠି_idx - 1]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଗଣ୍ଠି_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    ଯଦି (ଗଣ୍ଠି_idx >= n_nodes) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (nodes[ଗଣ୍ଠି_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ସ୍ଥିର ଅଚଳ std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = ସତ୍ୟ;
        ପାଇଁ (ଅଚଳ ggml_op op : norm_ops) {
            ଯଦି (nodes[ଗଣ୍ଠି_idx]->op == op && nodes[ଗଣ୍ଠି_idx]->src[0] == nodes[ଗଣ୍ଠି_idx - 1]) {
                ଗଣ୍ଠି_idx++;
            } ଅନ୍ୟଥା {
                args.norm = ମିଥ୍ୟା;
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }

        // DIV <- CLAMP, RESHAPE
        ଯଦି (nodes[ଗଣ୍ଠି_idx]->op != GGML_OP_DIV || nodes[ଗଣ୍ଠି_idx]->src[1] != nodes[ଗଣ୍ଠି_idx - 1] ||
            nodes[ଗଣ୍ଠି_idx]->src[0] != nodes[ଗଣ୍ଠି_idx - 3]) {
            args.norm = ମିଥ୍ୟା;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଗଣ୍ଠି_idx++;

        ଯଦି (nodes[ଗଣ୍ଠି_idx]->op != GGML_OP_RESHAPE || nodes[ଗଣ୍ଠି_idx]->src[0] != nodes[ଗଣ୍ଠି_idx - 1]) {
            args.norm = ମିଥ୍ୟା;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଗଣ୍ଠି_idx++;
    }

    ଯଦି (nodes[ଗଣ୍ଠି_idx]->op == GGML_OP_SCALE && nodes[ଗଣ୍ଠି_idx]->src[0] == nodes[ଗଣ୍ଠି_idx - 1]) {
        args.scale = ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ସ୍ଥିର ତାର୍କିକ ggml_sycl_check_fusion_memory_ranges(ଅଚଳ ggml_cgraph * cgraph, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                 ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_ଗଣନା, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * out_nodes, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_ଗଣନା,
                                                 ଅଚଳ ତାର୍କିକ is_topk_moe = ମିଥ୍ୟା) {
    auto nodes_overlap = [&](ଅଚଳ ggml_tensor * a, ଅଚଳ ggml_tensor * b) {
        ଅଚଳ int64_t a_start = (int64_t) a->data;
        ଅଚଳ int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        ଅଚଳ int64_t b_start = (int64_t) b->data;
        ଅଚଳ int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        ଯଦି ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    };

    ତାର୍କିକ is_ok = ସତ୍ୟ;
    // exception for topk-moe, as each row is read entirely before writing
    ଯଦି (ggml_nrows(cgraph->nodes[ଗଣ୍ଠି_idx]) == 1 && is_topk_moe) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < out_ଗଣନା; ++i) {
        ଅଚଳ ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ଗଣ୍ଠି_idx; j < ଗଣ୍ଠି_idx + ଗଣ୍ଠି_ଗଣନା; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ଅଚଳ ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                ଯଦି (!src || src->op == GGML_OP_NONE) {
                    ଜାରି;
                }

                ଯଦି (nodes_overlap(dst, src)) {
                    ତାର୍କିକ found = ମିଥ୍ୟା;

                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = ଗଣ୍ଠି_idx; k < j; ++k) {
                        ଯଦି (cgraph->nodes[k] == src) {
                            found = ସତ୍ୟ;
                            ଭାଙ୍ଗ;
                        }
                    }

                    ଯଦି (!found) {
                        is_ok = ମିଥ୍ୟା;
                        ଭାଙ୍ଗ;
                    }
                }
            }
        }
    }

    ଫେରନ୍ତୁ is_ok;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) {
    ଯଦି (!g_ggml_sycl_enable_fusion) {
        ଫେରନ୍ତୁ 0;
    }

    ଫେରନ୍ତୁ ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) {
    ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];

    ଯଦି (ଗଣ୍ଠି->op != GGML_OP_UNARY && ଗଣ୍ଠି->op != GGML_OP_SOFT_MAX && ଗଣ୍ଠି->op != GGML_OP_ARGSORT) {
        ଫେରନ୍ତୁ 0;
    }

    ggml_sycl_topk_moe_args args;
    ଯଦି (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        ଫେରନ୍ତୁ 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    ଯଦି (args.prob_bias) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ ggml_tensor * logits  = ଗଣ୍ଠି->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    ଅଚଳ ggml_tensor * clamp   = nullptr;
    ଅଚଳ ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                  out_nodes[2];

    ଯଦି (!args.delayed_softmax) {
        ଅଚଳ ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        ଯଦି (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size() - 3];
        }
        ଯଦି (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size() - 1];
        out_nodes[1] = i + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size() - 1;

        ଯଦି (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(ଗଣ୍ଠି, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size(), out_nodes, 2, /*is_topk_moe=*/ସତ୍ୟ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            ଫେରନ୍ତୁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size() - 1;
        }
    } ଅନ୍ୟଥା ଯଦି (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ଅଚଳ ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        ଯଦି (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size(), out_nodes, 2, /*is_topk_moe=*/ସତ୍ୟ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            ଫେରନ୍ତୁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size() - 1;
        }
    }

    ଫେରନ୍ତୁ 0;
}
