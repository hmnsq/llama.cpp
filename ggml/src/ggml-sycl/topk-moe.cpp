#ಅನ್ತರ್ಭಾವ <cfloat>
#ಅನ್ತರ್ಭಾವ <initializer_list>
#ಅನ್ತರ್ಭಾವ <vector>

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

ಸಂರಚನಾ ggml_sycl_topk_moe_args {
    ತಾರ್ಕಿಕ sigmoid{};
    ತಾರ್ಕಿಕ softmax{};
    ತಾರ್ಕಿಕ delayed_softmax{};
    ತಾರ್ಕಿಕ prob_bias{};
    ತಾರ್ಕಿಕ norm{};
    ತಾರ್ಕಿಕ scale{};
};

ಸಂರಚನಾ topk_moe_config {
    ತಾರ್ಕಿಕ use_sigmoid;
    ತಾರ್ಕಿಕ with_norm;
    ತಾರ್ಕಿಕ delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <ಪೂರ್ಣಾಙ್ಕ experts_per_thread, ತಾರ್ಕಿಕ use_limit>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ softmax_warp_inplace(ಭಿನ್ನ (&vals)[experts_per_thread], ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ limit, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane) {
    ಭಿನ್ನ max_val = -INFINITY;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < experts_per_thread; i++) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ  idx    = lane + i * WARP_SIZE;
        ಧ್ರುವ ತಾರ್ಕಿಕ active = !use_limit || (idx < limit);
        ಯದಿ (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    ಭಿನ್ನ sum = 0.f;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < experts_per_thread; i++) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ  idx    = lane + i * WARP_SIZE;
        ಧ್ರುವ ತಾರ್ಕಿಕ active = !use_limit || (idx < limit);
        ಯದಿ (active) {
            ಧ್ರುವ ಭಿನ್ನ val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } ಅನ್ಯಥಾ {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    ಧ್ರುವ ಭಿನ್ನ inv_sum = 1.0f / sum;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < experts_per_thread; i++) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = lane + i * WARP_SIZE;
        ಯದಿ (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <ಪೂರ್ಣಾಙ್ಕ experts_per_thread, ತಾರ್ಕಿಕ use_limit>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ sigmoid_warp_inplace(ಭಿನ್ನ (&vals)[experts_per_thread], ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ limit, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane) {
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < experts_per_thread; i++) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ  idx    = lane + i * WARP_SIZE;
        ಧ್ರುವ ತಾರ್ಕಿಕ active = !use_limit || (idx < limit);
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
template <ಪೂರ್ಣಾಙ್ಕ n_experts>
ಸ್ಥಿರ ಶೂನ್ಯ topk_moe_kernel(ಧ್ರುವ ಭಿನ್ನ * __restrict__ logits,
                            ಭಿನ್ನ * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                  n_rows,
                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                  n_expert_used,
                            ಧ್ರುವ ಭಿನ್ನ                clamp_val,
                            ಧ್ರುವ ಭಿನ್ನ                scale_val,
                            ಧ್ರುವ topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ row       = item_ct1.get_group(0);
    ಯದಿ (row >= n_rows) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr ಪೂರ್ಣಾಙ್ಕ experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    ಭಿನ್ನ wt[experts_per_thread];
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_experts; i += WARP_SIZE) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    ಯದಿ (!config.delayed_softmax) {
        ಯದಿ (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, ಅಸತ್ಯ>(wt, n_experts, lane);
        } ಅನ್ಯಥಾ {
            softmax_warp_inplace<experts_per_thread, ಅಸತ್ಯ>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < experts_per_thread; i++) {
        ಯದಿ (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    ಭಿನ್ನ wt_sum = 0.f;
    ಭಿನ್ನ output_weights[experts_per_thread];
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ಧ್ರುವ sycl::sub_group sg = item_ct1.get_sub_group();

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < n_expert_used; k++) {
        ಭಿನ್ನ max_val    = wt[0];
        ಪೂರ್ಣಾಙ್ಕ   max_expert = lane;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < experts_per_thread; i++) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ expert = lane + i * WARP_SIZE;
            ಯದಿ ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            ಧ್ರುವ ಭಿನ್ನ val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            ಯದಿ (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        ಯದಿ ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        ಯದಿ ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        ಯದಿ ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            ಯದಿ (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    ಯದಿ (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        ಧ್ರುವ ಭಿನ್ನ inv = 1.0f / wt_sum;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    ಯದಿ (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, ಸತ್ಯ>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < experts_per_thread; i++) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = i * WARP_SIZE + lane;
        ಯದಿ (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <ಪೂರ್ಣಾಙ್ಕ n_experts>
ಸ್ಥಿರ ಶೂನ್ಯ launch_topk_moe(queue_ptr stream, ಧ್ರುವ ಭಿನ್ನ * logits, ಭಿನ್ನ * weights, int32_t * ids, ಪೂರ್ಣಾಙ್ಕ n_rows,
                            ಪೂರ್ಣಾಙ್ಕ n_expert_used, ಭಿನ್ನ clamp_val, ಭಿನ್ನ scale_val, ಧ್ರುವ topk_moe_config & config) {
    ಧ್ರುವ sycl::range<1> block_dims(WARP_SIZE);
    ಧ್ರುವ sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ಧ್ರುವ ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ಧ್ರುವ ggml_tensor *             clamp,
                                  ಧ್ರುವ ggml_tensor *             scale,
                                  ಧ್ರುವ ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_experts     = logits->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_rows        = logits->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    ಧ್ರುವ ಭಿನ್ನ * logits_d  = (ಧ್ರುವ ಭಿನ್ನ *) logits->data;
    ಭಿನ್ನ *       weights_d = (ಭಿನ್ನ *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    ಧ್ರುವ ತಾರ್ಕಿಕ  with_norm = clamp != nullptr;
    ಧ್ರುವ ಭಿನ್ನ clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ಧ್ರುವ ಭಿನ್ನ scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    ವಿಕಲ್ಪ (n_experts) {
        ಪಕ್ಷ 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ಭಙ್ಗ;
        ಪಕ್ಷ 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ಭಙ್ಗ;
        ಪಕ್ಷ 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ಭಙ್ಗ;
        ಪಕ್ಷ 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ಭಙ್ಗ;
        ಪಕ್ಷ 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            ಭಙ್ಗ;
        ಪಕ್ಷ 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            ಭಙ್ಗ;
        ಪಕ್ಷ 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            ಭಙ್ಗ;
        ಪಕ್ಷ 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            ಭಙ್ಗ;
        ಪಕ್ಷ 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            ಭಙ್ಗ;
        ಪಕ್ಷ 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ASSERT(ಅಸತ್ಯ && "fatal error");
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_should_use_topk_moe(ಧ್ರುವ ggml_tensor * gating_op, ಧ್ರುವ ggml_tensor * weights,
                                          ಧ್ರುವ ggml_tensor * logits, ಧ್ರುವ ggml_tensor * ids) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_expert = ids->nb[1] / ids->nb[0];
    ಯದಿ ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (gating_op->op == GGML_OP_SOFT_MAX) {
        ಭಿನ್ನ scale    = 1.0f;
        ಭಿನ್ನ max_bias = 0.0f;

        memcpy(&scale, (ಧ್ರುವ ಭಿನ್ನ *) gating_op->op_params + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
        memcpy(&max_bias, (ಧ್ರುವ ಭಿನ್ನ *) gating_op->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

        ಯದಿ (!ggml_is_contiguous(gating_op->src[0])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (scale != 1.0f || max_bias != 0.0f) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // don't fuse when masks or sinks are present
        ಯದಿ (gating_op->src[1] || gating_op->src[2]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ ಯದಿ (gating_op->op == GGML_OP_UNARY) {
        ಯದಿ (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_topk_moe_fusion(ಧ್ರುವ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    ಯದಿ (nodes[ಗಂಟು_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = ಸತ್ಯ;
    }

    ಯದಿ (nodes[ಗಂಟು_idx]->op == GGML_OP_UNARY) {
        ಯದಿ (ggml_get_unary_op(nodes[ಗಂಟು_idx]) != GGML_UNARY_OP_SIGMOID) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        args.sigmoid = ಸತ್ಯ;
    }

    ಯದಿ (nodes[ಗಂಟು_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = ಸತ್ಯ;
    }

    ಗಂಟು_idx++;

    ಯದಿ (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        ಯದಿ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_RESHAPE ||
            nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ggml_tensor * probs_reshaped = nodes[ಗಂಟು_idx];
        ಗಂಟು_idx++;

        ಯದಿ (ಗಂಟು_idx >= n_nodes) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        ಯದಿ (nodes[ಗಂಟು_idx]->op == GGML_OP_ADD && nodes[ಗಂಟು_idx]->src[0] == nodes[ಗಂಟು_idx - 2]) {
            args.prob_bias = ಸತ್ಯ;
            ಗಂಟು_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        ಯದಿ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_ARGSORT) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (args.prob_bias && nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        } ಅನ್ಯಥಾ ಯದಿ (!args.prob_bias && nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 2]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಗಂಟು_idx++;

        // ARGSORT -> VIEW
        ಯದಿ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_VIEW ||
            nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಗಂಟು_idx++;

        ಯದಿ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_GET_ROWS) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // GET_ROWS
        ಯದಿ (nodes[ಗಂಟು_idx]->src[0] != probs_reshaped || nodes[ಗಂಟು_idx]->src[1] != nodes[ಗಂಟು_idx - 1]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಗಂಟು_idx++;
    } ಅನ್ಯಥಾ ಯದಿ (args.delayed_softmax) {
        ಯದಿ (ಗಂಟು_idx - 2 < 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ggml_tensor * probs_reshaped = nodes[ಗಂಟು_idx - 2];

        // VIEW -> ARGSORT
        ಯದಿ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_VIEW ||
            nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಗಂಟು_idx++;

        // GET_ROWS
        ಯದಿ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->src[1] != nodes[ಗಂಟು_idx - 1] ||
            nodes[ಗಂಟು_idx]->src[0] != probs_reshaped) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಗಂಟು_idx++;

        ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        ಪ್ರತಿ (ಧ್ರುವ ggml_op op : remaining_ops) {
            ಯದಿ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != op || nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಗಂಟು_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    ಯದಿ (ಗಂಟು_idx >= n_nodes) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (nodes[ಗಂಟು_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = ಸತ್ಯ;
        ಪ್ರತಿ (ಧ್ರುವ ggml_op op : norm_ops) {
            ಯದಿ (nodes[ಗಂಟು_idx]->op == op && nodes[ಗಂಟು_idx]->src[0] == nodes[ಗಂಟು_idx - 1]) {
                ಗಂಟು_idx++;
            } ಅನ್ಯಥಾ {
                args.norm = ಅಸತ್ಯ;
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }

        // DIV <- CLAMP, RESHAPE
        ಯದಿ (nodes[ಗಂಟು_idx]->op != GGML_OP_DIV || nodes[ಗಂಟು_idx]->src[1] != nodes[ಗಂಟು_idx - 1] ||
            nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 3]) {
            args.norm = ಅಸತ್ಯ;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಗಂಟು_idx++;

        ಯದಿ (nodes[ಗಂಟು_idx]->op != GGML_OP_RESHAPE || nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            args.norm = ಅಸತ್ಯ;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಗಂಟು_idx++;
    }

    ಯದಿ (nodes[ಗಂಟು_idx]->op == GGML_OP_SCALE && nodes[ಗಂಟು_idx]->src[0] == nodes[ಗಂಟು_idx - 1]) {
        args.scale = ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_check_fusion_memory_ranges(ಧ್ರುವ ggml_cgraph * cgraph, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                 ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_ಎಣಿಕೆ, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * out_nodes, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_ಎಣಿಕೆ,
                                                 ಧ್ರುವ ತಾರ್ಕಿಕ is_topk_moe = ಅಸತ್ಯ) {
    auto nodes_overlap = [&](ಧ್ರುವ ggml_tensor * a, ಧ್ರುವ ggml_tensor * b) {
        ಧ್ರುವ int64_t a_start = (int64_t) a->data;
        ಧ್ರುವ int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        ಧ್ರುವ int64_t b_start = (int64_t) b->data;
        ಧ್ರುವ int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        ಯದಿ ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    };

    ತಾರ್ಕಿಕ is_ok = ಸತ್ಯ;
    // exception for topk-moe, as each row is read entirely before writing
    ಯದಿ (ggml_nrows(cgraph->nodes[ಗಂಟು_idx]) == 1 && is_topk_moe) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < out_ಎಣಿಕೆ; ++i) {
        ಧ್ರುವ ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = ಗಂಟು_idx; j < ಗಂಟು_idx + ಗಂಟು_ಎಣಿಕೆ; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ಧ್ರುವ ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                ಯದಿ (!src || src->op == GGML_OP_NONE) {
                    ಅನುವರ್ತ;
                }

                ಯದಿ (nodes_overlap(dst, src)) {
                    ತಾರ್ಕಿಕ found = ಅಸತ್ಯ;

                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = ಗಂಟು_idx; k < j; ++k) {
                        ಯದಿ (cgraph->nodes[k] == src) {
                            found = ಸತ್ಯ;
                            ಭಙ್ಗ;
                        }
                    }

                    ಯದಿ (!found) {
                        is_ok = ಅಸತ್ಯ;
                        ಭಙ್ಗ;
                    }
                }
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ is_ok;
}

ಪೂರ್ಣಾಙ್ಕ ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ i) {
    ಯದಿ (!g_ggml_sycl_enable_fusion) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

ಪೂರ್ಣಾಙ್ಕ ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ i) {
    ggml_tensor * ಗಂಟು = cgraph->nodes[i];

    ಯದಿ (ಗಂಟು->op != GGML_OP_UNARY && ಗಂಟು->op != GGML_OP_SOFT_MAX && ಗಂಟು->op != GGML_OP_ARGSORT) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ggml_sycl_topk_moe_args args;
    ಯದಿ (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    ಯದಿ (args.prob_bias) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ ggml_tensor * logits  = ಗಂಟು->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    ಧ್ರುವ ggml_tensor * clamp   = nullptr;
    ಧ್ರುವ ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    ಪೂರ್ಣಾಙ್ಕ                  out_nodes[2];

    ಯದಿ (!args.delayed_softmax) {
        ಧ್ರುವ ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        ಯದಿ (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (ಪೂರ್ಣಾಙ್ಕ) ops.size() - 3];
        }
        ಯದಿ (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (ಪೂರ್ಣಾಙ್ಕ) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (ಪೂರ್ಣಾಙ್ಕ) ops.size() - 1];
        out_nodes[1] = i + (ಪೂರ್ಣಾಙ್ಕ) ops.size() - 1;

        ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(ಗಂಟು, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (ಪೂರ್ಣಾಙ್ಕ) ops.size(), out_nodes, 2, /*is_topk_moe=*/ಸತ್ಯ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            ಪ್ರತ್ಯಾಗಮ (ಪೂರ್ಣಾಙ್ಕ) ops.size() - 1;
        }
    } ಅನ್ಯಥಾ ಯದಿ (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ಧ್ರುವ ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        ಯದಿ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (ಪೂರ್ಣಾಙ್ಕ) ops.size(), out_nodes, 2, /*is_topk_moe=*/ಸತ್ಯ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            ಪ್ರತ್ಯಾಗಮ (ಪೂರ್ಣಾಙ್ಕ) ops.size() - 1;
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
