#अन्तर्भाव <cfloat>
#अन्तर्भाव <initializer_list>
#अन्तर्भाव <vector>

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

संरचना ggml_sycl_topk_moe_args {
    तार्किक sigmoid{};
    तार्किक softmax{};
    तार्किक delayed_softmax{};
    तार्किक prob_bias{};
    तार्किक norm{};
    तार्किक scale{};
};

संरचना topk_moe_config {
    तार्किक use_sigmoid;
    तार्किक with_norm;
    तार्किक delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <पूर्णाङ्क experts_per_thread, तार्किक use_limit>
स्थिर अन्तःस्थ शून्य softmax_warp_inplace(भिन्न (&vals)[experts_per_thread], ध्रुव पूर्णाङ्क limit, ध्रुव पूर्णाङ्क lane) {
    भिन्न max_val = -INFINITY;
#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < experts_per_thread; i++) {
        ध्रुव पूर्णाङ्क  idx    = lane + i * WARP_SIZE;
        ध्रुव तार्किक active = !use_limit || (idx < limit);
        सचे (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    भिन्न sum = 0.f;
#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < experts_per_thread; i++) {
        ध्रुव पूर्णाङ्क  idx    = lane + i * WARP_SIZE;
        ध्रुव तार्किक active = !use_limit || (idx < limit);
        सचे (active) {
            ध्रुव भिन्न val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } अञ्ञथा {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    ध्रुव भिन्न inv_sum = 1.0f / sum;
#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < experts_per_thread; i++) {
        ध्रुव पूर्णाङ्क idx = lane + i * WARP_SIZE;
        सचे (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <पूर्णाङ्क experts_per_thread, तार्किक use_limit>
स्थिर अन्तःस्थ शून्य sigmoid_warp_inplace(भिन्न (&vals)[experts_per_thread], ध्रुव पूर्णाङ्क limit, ध्रुव पूर्णाङ्क lane) {
#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < experts_per_thread; i++) {
        ध्रुव पूर्णाङ्क  idx    = lane + i * WARP_SIZE;
        ध्रुव तार्किक active = !use_limit || (idx < limit);
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
template <पूर्णाङ्क n_experts>
स्थिर शून्य topk_moe_kernel(ध्रुव भिन्न * __restrict__ logits,
                            भिन्न * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ध्रुव पूर्णाङ्क                  n_rows,
                            ध्रुव पूर्णाङ्क                  n_expert_used,
                            ध्रुव भिन्न                clamp_val,
                            ध्रुव भिन्न                scale_val,
                            ध्रुव topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    ध्रुव पूर्णाङ्क row       = item_ct1.get_group(0);
    सचे (row >= n_rows) {
        पटिनिवत्त;
    }
    ध्रुव पूर्णाङ्क lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr पूर्णाङ्क experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    भिन्न wt[experts_per_thread];
#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < n_experts; i += WARP_SIZE) {
        ध्रुव पूर्णाङ्क expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    सचे (!config.delayed_softmax) {
        सचे (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, मिच्छा>(wt, n_experts, lane);
        } अञ्ञथा {
            softmax_warp_inplace<experts_per_thread, मिच्छा>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < experts_per_thread; i++) {
        सचे (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    भिन्न wt_sum = 0.f;
    भिन्न output_weights[experts_per_thread];
#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ध्रुव sycl::sub_group sg = item_ct1.get_sub_group();

    अत्थाय (पूर्णाङ्क k = 0; k < n_expert_used; k++) {
        भिन्न max_val    = wt[0];
        पूर्णाङ्क   max_expert = lane;
#pragma unroll
        अत्थाय (पूर्णाङ्क i = 1; i < experts_per_thread; i++) {
            ध्रुव पूर्णाङ्क expert = lane + i * WARP_SIZE;
            सचे ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        अत्थाय (पूर्णाङ्क mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            ध्रुव भिन्न val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            ध्रुव पूर्णाङ्क   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            सचे (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        सचे ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        सचे ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        सचे ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            सचे (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    सचे (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        ध्रुव भिन्न inv = 1.0f / wt_sum;
#pragma unroll
        अत्थाय (पूर्णाङ्क i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    सचे (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, सच्च>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    अत्थाय (पूर्णाङ्क i = 0; i < experts_per_thread; i++) {
        ध्रुव पूर्णाङ्क idx = i * WARP_SIZE + lane;
        सचे (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <पूर्णाङ्क n_experts>
स्थिर शून्य launch_topk_moe(queue_ptr stream, ध्रुव भिन्न * logits, भिन्न * weights, int32_t * ids, पूर्णाङ्क n_rows,
                            पूर्णाङ्क n_expert_used, भिन्न clamp_val, भिन्न scale_val, ध्रुव topk_moe_config & config) {
    ध्रुव sycl::range<1> block_dims(WARP_SIZE);
    ध्रुव sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

स्थिर शून्य ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ध्रुव ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ध्रुव ggml_tensor *             clamp,
                                  ध्रुव ggml_tensor *             scale,
                                  ध्रुव ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ध्रुव पूर्णाङ्क n_experts     = logits->ne[0];
    ध्रुव पूर्णाङ्क n_rows        = logits->ne[1];
    ध्रुव पूर्णाङ्क n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    ध्रुव भिन्न * logits_d  = (ध्रुव भिन्न *) logits->data;
    भिन्न *       weights_d = (भिन्न *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    ध्रुव तार्किक  with_norm = clamp != nullptr;
    ध्रुव भिन्न clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ध्रुव भिन्न scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    विकल्प (n_experts) {
        पक्ष 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            भङ्ग;
        पक्ष 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            भङ्ग;
        पक्ष 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            भङ्ग;
        पक्ष 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            भङ्ग;
        पक्ष 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            भङ्ग;
        पक्ष 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            भङ्ग;
        पक्ष 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            भङ्ग;
        पक्ष 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            भङ्ग;
        पक्ष 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            भङ्ग;
        पक्ष 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            भङ्ग;
        स्वभाव:
            GGML_ASSERT(मिच्छा && "fatal error");
    }
}

स्थिर तार्किक ggml_sycl_should_use_topk_moe(ध्रुव ggml_tensor * gating_op, ध्रुव ggml_tensor * weights,
                                          ध्रुव ggml_tensor * logits, ध्रुव ggml_tensor * ids) {
    ध्रुव पूर्णाङ्क n_expert = ids->nb[1] / ids->nb[0];
    सचे ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        पटिनिवत्त मिच्छा;
    }

    सचे (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        पटिनिवत्त मिच्छा;
    }

    सचे (gating_op->op == GGML_OP_SOFT_MAX) {
        भिन्न scale    = 1.0f;
        भिन्न max_bias = 0.0f;

        memcpy(&scale, (ध्रुव भिन्न *) gating_op->op_params + 0, परिमाण(भिन्न));
        memcpy(&max_bias, (ध्रुव भिन्न *) gating_op->op_params + 1, परिमाण(भिन्न));

        सचे (!ggml_is_contiguous(gating_op->src[0])) {
            पटिनिवत्त मिच्छा;
        }
        सचे (scale != 1.0f || max_bias != 0.0f) {
            पटिनिवत्त मिच्छा;
        }
        // don't fuse when masks or sinks are present
        सचे (gating_op->src[1] || gating_op->src[2]) {
            पटिनिवत्त मिच्छा;
        }
    } अञ्ञथा सचे (gating_op->op == GGML_OP_UNARY) {
        सचे (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            पटिनिवत्त मिच्छा;
        }
    }

    पटिनिवत्त सच्च;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
स्थिर तार्किक ggml_sycl_topk_moe_fusion(ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    ध्रुव पूर्णाङ्क      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    सचे (nodes[ग्रन्थिः_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = सच्च;
    }

    सचे (nodes[ग्रन्थिः_idx]->op == GGML_OP_UNARY) {
        सचे (ggml_get_unary_op(nodes[ग्रन्थिः_idx]) != GGML_UNARY_OP_SIGMOID) {
            पटिनिवत्त मिच्छा;
        }
        args.sigmoid = सच्च;
    }

    सचे (nodes[ग्रन्थिः_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = सच्च;
    }

    ग्रन्थिः_idx++;

    सचे (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        सचे (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_RESHAPE ||
            nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            पटिनिवत्त मिच्छा;
        }
        ggml_tensor * probs_reshaped = nodes[ग्रन्थिः_idx];
        ग्रन्थिः_idx++;

        सचे (ग्रन्थिः_idx >= n_nodes) {
            पटिनिवत्त मिच्छा;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        सचे (nodes[ग्रन्थिः_idx]->op == GGML_OP_ADD && nodes[ग्रन्थिः_idx]->src[0] == nodes[ग्रन्थिः_idx - 2]) {
            args.prob_bias = सच्च;
            ग्रन्थिः_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        सचे (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_ARGSORT) {
            पटिनिवत्त मिच्छा;
        }

        सचे (args.prob_bias && nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            पटिनिवत्त मिच्छा;
        } अञ्ञथा सचे (!args.prob_bias && nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 2]) {
            पटिनिवत्त मिच्छा;
        }

        ग्रन्थिः_idx++;

        // ARGSORT -> VIEW
        सचे (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_VIEW ||
            nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            पटिनिवत्त मिच्छा;
        }
        ग्रन्थिः_idx++;

        सचे (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_GET_ROWS) {
            पटिनिवत्त मिच्छा;
        }

        // GET_ROWS
        सचे (nodes[ग्रन्थिः_idx]->src[0] != probs_reshaped || nodes[ग्रन्थिः_idx]->src[1] != nodes[ग्रन्थिः_idx - 1]) {
            पटिनिवत्त मिच्छा;
        }
        ग्रन्थिः_idx++;
    } अञ्ञथा सचे (args.delayed_softmax) {
        सचे (ग्रन्थिः_idx - 2 < 0) {
            पटिनिवत्त मिच्छा;
        }
        ggml_tensor * probs_reshaped = nodes[ग्रन्थिः_idx - 2];

        // VIEW -> ARGSORT
        सचे (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_VIEW ||
            nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            पटिनिवत्त मिच्छा;
        }
        ग्रन्थिः_idx++;

        // GET_ROWS
        सचे (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->src[1] != nodes[ग्रन्थिः_idx - 1] ||
            nodes[ग्रन्थिः_idx]->src[0] != probs_reshaped) {
            पटिनिवत्त मिच्छा;
        }
        ग्रन्थिः_idx++;

        स्थिर ध्रुव std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        अत्थाय (ध्रुव ggml_op op : remaining_ops) {
            सचे (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != op || nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
                पटिनिवत्त मिच्छा;
            }
            ग्रन्थिः_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    सचे (ग्रन्थिः_idx >= n_nodes) {
        पटिनिवत्त सच्च;
    }

    सचे (nodes[ग्रन्थिः_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        स्थिर ध्रुव std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = सच्च;
        अत्थाय (ध्रुव ggml_op op : norm_ops) {
            सचे (nodes[ग्रन्थिः_idx]->op == op && nodes[ग्रन्थिः_idx]->src[0] == nodes[ग्रन्थिः_idx - 1]) {
                ग्रन्थिः_idx++;
            } अञ्ञथा {
                args.norm = मिच्छा;
                पटिनिवत्त सच्च;
            }
        }

        // DIV <- CLAMP, RESHAPE
        सचे (nodes[ग्रन्थिः_idx]->op != GGML_OP_DIV || nodes[ग्रन्थिः_idx]->src[1] != nodes[ग्रन्थिः_idx - 1] ||
            nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 3]) {
            args.norm = मिच्छा;
            पटिनिवत्त सच्च;
        }
        ग्रन्थिः_idx++;

        सचे (nodes[ग्रन्थिः_idx]->op != GGML_OP_RESHAPE || nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            args.norm = मिच्छा;
            पटिनिवत्त सच्च;
        }
        ग्रन्थिः_idx++;
    }

    सचे (nodes[ग्रन्थिः_idx]->op == GGML_OP_SCALE && nodes[ग्रन्थिः_idx]->src[0] == nodes[ग्रन्थिः_idx - 1]) {
        args.scale = सच्च;
    }

    पटिनिवत्त सच्च;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
स्थिर तार्किक ggml_sycl_check_fusion_memory_ranges(ध्रुव ggml_cgraph * cgraph, ध्रुव पूर्णाङ्क ग्रन्थिः_idx,
                                                 ध्रुव पूर्णाङ्क ग्रन्थिः_गणनम्, ध्रुव पूर्णाङ्क * out_nodes, ध्रुव पूर्णाङ्क out_गणनम्,
                                                 ध्रुव तार्किक is_topk_moe = मिच्छा) {
    auto nodes_overlap = [&](ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
        ध्रुव int64_t a_start = (int64_t) a->data;
        ध्रुव int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        ध्रुव int64_t b_start = (int64_t) b->data;
        ध्रुव int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        सचे ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            पटिनिवत्त सच्च;
        }

        पटिनिवत्त मिच्छा;
    };

    तार्किक is_ok = सच्च;
    // exception for topk-moe, as each row is read entirely before writing
    सचे (ggml_nrows(cgraph->nodes[ग्रन्थिः_idx]) == 1 && is_topk_moe) {
        पटिनिवत्त सच्च;
    }

    अत्थाय (पूर्णाङ्क i = 0; i < out_गणनम्; ++i) {
        ध्रुव ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        अत्थाय (पूर्णाङ्क j = ग्रन्थिः_idx; j < ग्रन्थिः_idx + ग्रन्थिः_गणनम्; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            अत्थाय (पूर्णाङ्क src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ध्रुव ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                सचे (!src || src->op == GGML_OP_NONE) {
                    अनुवर्त;
                }

                सचे (nodes_overlap(dst, src)) {
                    तार्किक found = मिच्छा;

                    अत्थाय (पूर्णाङ्क k = ग्रन्थिः_idx; k < j; ++k) {
                        सचे (cgraph->nodes[k] == src) {
                            found = सच्च;
                            भङ्ग;
                        }
                    }

                    सचे (!found) {
                        is_ok = मिच्छा;
                        भङ्ग;
                    }
                }
            }
        }
    }

    पटिनिवत्त is_ok;
}

पूर्णाङ्क ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णाङ्क i) {
    सचे (!g_ggml_sycl_enable_fusion) {
        पटिनिवत्त 0;
    }

    पटिनिवत्त ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

पूर्णाङ्क ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णाङ्क i) {
    ggml_tensor * ग्रन्थिः = cgraph->nodes[i];

    सचे (ग्रन्थिः->op != GGML_OP_UNARY && ग्रन्थिः->op != GGML_OP_SOFT_MAX && ग्रन्थिः->op != GGML_OP_ARGSORT) {
        पटिनिवत्त 0;
    }

    ggml_sycl_topk_moe_args args;
    सचे (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        पटिनिवत्त 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    सचे (args.prob_bias) {
        पटिनिवत्त 0;
    }

    ध्रुव ggml_tensor * logits  = ग्रन्थिः->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    ध्रुव ggml_tensor * clamp   = nullptr;
    ध्रुव ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    पूर्णाङ्क                  out_nodes[2];

    सचे (!args.delayed_softmax) {
        ध्रुव ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        सचे (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (पूर्णाङ्क) ops.size() - 3];
        }
        सचे (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (पूर्णाङ्क) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (पूर्णाङ्क) ops.size() - 1];
        out_nodes[1] = i + (पूर्णाङ्क) ops.size() - 1;

        सचे (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(ग्रन्थिः, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (पूर्णाङ्क) ops.size(), out_nodes, 2, /*is_topk_moe=*/सच्च)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            पटिनिवत्त (पूर्णाङ्क) ops.size() - 1;
        }
    } अञ्ञथा सचे (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ध्रुव ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        सचे (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (पूर्णाङ्क) ops.size(), out_nodes, 2, /*is_topk_moe=*/सच्च)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            पटिनिवत्त (पूर्णाङ्क) ops.size() - 1;
        }
    }

    पटिनिवत्त 0;
}
