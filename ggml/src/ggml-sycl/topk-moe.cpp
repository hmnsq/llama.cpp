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

ರಚನೆ ggml_sycl_topk_moe_ವಾದಗಳು {
    ತಾರ್ಕಿಕ sigmoid{};
    ತಾರ್ಕಿಕ softmax{};
    ತಾರ್ಕಿಕ delayed_softmax{};
    ತಾರ್ಕಿಕ prob_bias{};
    ತಾರ್ಕಿಕ norm{};
    ತಾರ್ಕಿಕ scale{};
};

ರಚನೆ topk_moe_config {
    ತಾರ್ಕಿಕ use_sigmoid;
    ತಾರ್ಕಿಕ with_norm;
    ತಾರ್ಕಿಕ delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ experts_ಪ್ರತಿ_thread, ತಾರ್ಕಿಕ use_ಎಲ್ಲೆ>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ softmax_warp_inplace(ದಶಮಾಂಶ (&vals)[experts_ಪ್ರತಿ_thread], ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಎಲ್ಲೆ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane) {
    ದಶಮಾಂಶ max_val = -INFINITY;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < experts_ಪ್ರತಿ_thread; i++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  idx    = lane + i * WARP_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಸಕ್ರಿಯ = !use_ಎಲ್ಲೆ || (idx < ಎಲ್ಲೆ);
        ಆದರೆ (ಸಕ್ರಿಯ) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_ಗಾತ್ರ>(max_val);

    ದಶಮಾಂಶ ಮೊತ್ತ = 0.f;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < experts_ಪ್ರತಿ_thread; i++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  idx    = lane + i * WARP_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಸಕ್ರಿಯ = !use_ಎಲ್ಲೆ || (idx < ಎಲ್ಲೆ);
        ಆದರೆ (ಸಕ್ರಿಯ) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val = sycl::ಘಾತಾಂಕ(vals[i] - max_val);
            vals[i]         = val;
            ಮೊತ್ತ += val;
        } ಇಲ್ಲದಿದ್ದರೆ {
            vals[i] = 0.f;
        }
    }
    ಮೊತ್ತ = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(ಮೊತ್ತ);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = 1.0f / ಮೊತ್ತ;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < experts_ಪ್ರತಿ_thread; i++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = lane + i * WARP_ಗಾತ್ರ;
        ಆದರೆ (!use_ಎಲ್ಲೆ || idx < ಎಲ್ಲೆ) {
            vals[i] *= inv_ಮೊತ್ತ;
        }
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ experts_ಪ್ರತಿ_thread, ತಾರ್ಕಿಕ use_ಎಲ್ಲೆ>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ sigmoid_warp_inplace(ದಶಮಾಂಶ (&vals)[experts_ಪ್ರತಿ_thread], ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಎಲ್ಲೆ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane) {
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < experts_ಪ್ರತಿ_thread; i++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  idx    = lane + i * WARP_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಸಕ್ರಿಯ = !use_ಎಲ್ಲೆ || (idx < ಎಲ್ಲೆ);
        vals[i]           = ಸಕ್ರಿಯ ? 1.f / (1.f + sycl::ಘಾತಾಂಕ(-vals[i])) : -INFINITY;
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
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ n_experts>
ಸ್ಥಿರ ಶೂನ್ಯ topk_moe_kernel(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ logits,
                            ದಶಮಾಂಶ * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                  n_ಅಡ್ಡಸಾಲುಗಳು,
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                  n_expert_ಬಳಸಿದ,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ                clamp_val,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ                scale_val,
                            ಸ್ಥಿರಾಂಕ topk_moe_config       config) {
    ಸ್ವಯಂಚಾಲಿತ      ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<1>();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು       = ಅಂಶ_ct1.get_ಗುಂಪು(0);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= n_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane = ಅಂಶ_ct1.get_local_id(0);

    logits  += (size_t) n_experts * ಅಡ್ಡಸಾಲು;
    weights += (size_t) n_expert_ಬಳಸಿದ * ಅಡ್ಡಸಾಲು;
    ids     += (size_t) n_experts * ಅಡ್ಡಸಾಲು;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr ಪೂರ್ಣಾಂಕ experts_ಪ್ರತಿ_thread = (n_experts > WARP_ಗಾತ್ರ) ? n_experts / WARP_ಗಾತ್ರ : 1;

    ದಶಮಾಂಶ wt[experts_ಪ್ರತಿ_thread];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < experts_ಪ್ರತಿ_thread; i++) {
        wt[i] = -INFINITY;
    }
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_experts; i += WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ expert  = i + lane;
        wt[i / WARP_ಗಾತ್ರ] = (n_experts % WARP_ಗಾತ್ರ == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    ಆದರೆ (!config.delayed_softmax) {
        ಆದರೆ (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_ಪ್ರತಿ_thread, ಸುಳ್ಳು>(wt, n_experts, lane);
        } ಇಲ್ಲದಿದ್ದರೆ {
            softmax_warp_inplace<experts_ಪ್ರತಿ_thread, ಸುಳ್ಳು>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < experts_ಪ್ರತಿ_thread; i++) {
        ಆದರೆ (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    ದಶಮಾಂಶ wt_ಮೊತ್ತ = 0.f;
    ದಶಮಾಂಶ output_weights[experts_ಪ್ರತಿ_thread];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < experts_ಪ್ರತಿ_thread; i++) {
        output_weights[i] = 0.f;
    }

    ಸ್ಥಿರಾಂಕ sycl::sub_ಗುಂಪು sg = ಅಂಶ_ct1.get_sub_ಗುಂಪು();

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < n_expert_ಬಳಸಿದ; k++) {
        ದಶಮಾಂಶ max_val    = wt[0];
        ಪೂರ್ಣಾಂಕ   max_expert = lane;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < experts_ಪ್ರತಿ_thread; i++) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ expert = lane + i * WARP_ಗಾತ್ರ;
            ಆದರೆ ((n_experts % WARP_ಗಾತ್ರ == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val    = dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(sg, max_val, ಮುಖವಾಡ);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   expert = dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(sg, max_expert, ಮುಖವಾಡ);
            ಆದರೆ (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        ಆದರೆ ((max_expert & (WARP_ಗಾತ್ರ - 1)) == lane) {
            wt[max_expert / WARP_ಗಾತ್ರ] = -INFINITY;
        }
        ಆದರೆ ((k & (WARP_ಗಾತ್ರ - 1)) == lane) {
            output_weights[k / WARP_ಗಾತ್ರ] = max_val;
        }
        ಆದರೆ ((max_expert & (WARP_ಗಾತ್ರ - 1)) == lane) {
            ids[k] = max_expert;
            ಆದರೆ (config.with_norm) {
                wt_ಮೊತ್ತ += max_val;
            }
        }
    }

    ಆದರೆ (config.with_norm) {
        wt_ಮೊತ್ತ          = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(wt_ಮೊತ್ತ);
        wt_ಮೊತ್ತ          = sycl::fmax(wt_ಮೊತ್ತ, clamp_val);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv = 1.0f / wt_ಮೊತ್ತ;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < experts_ಪ್ರತಿ_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    ಆದರೆ (config.delayed_softmax) {
        softmax_warp_inplace<experts_ಪ್ರತಿ_thread, ನಿಜ>(output_weights, n_expert_ಬಳಸಿದ, lane);
    }

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < experts_ಪ್ರತಿ_thread; i++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = i * WARP_ಗಾತ್ರ + lane;
        ಆದರೆ (idx < n_expert_ಬಳಸಿದ) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ n_experts>
ಸ್ಥಿರ ಶೂನ್ಯ launch_topk_moe(ಸರತಿ_ptr stream, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * logits, ದಶಮಾಂಶ * weights, int32_t * ids, ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು,
                            ಪೂರ್ಣಾಂಕ n_expert_ಬಳಸಿದ, ದಶಮಾಂಶ clamp_val, ದಶಮಾಂಶ scale_val, ಸ್ಥಿರಾಂಕ topk_moe_config & config) {
    ಸ್ಥಿರಾಂಕ sycl::range<1> ಖಂಡ_dims(WARP_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<1> ಖಂಡ_nums(n_ಅಡ್ಡಸಾಲುಗಳು);
    stream->parallel_for(sycl::nd_range<1>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val,
                                                        scale_val, config);
                             GGML_ಬಳಸದ(ಅಂಶ_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ಸ್ಥಿರಾಂಕ ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ಸ್ಥಿರಾಂಕ ggml_tensor *             clamp,
                                  ಸ್ಥಿರಾಂಕ ggml_tensor *             scale,
                                  ಸ್ಥಿರಾಂಕ ggml_sycl_topk_moe_ವಾದಗಳು & ವಾದಗಳು_2) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_experts     = logits->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು        = logits->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_expert_ಬಳಸಿದ = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_ಗಾತ್ರ(ids->type) == (size_t) n_experts);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * logits_d  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) logits->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       weights_d = (ದಶಮಾಂಶ *) weights->ದತ್ತಾಂಶ;
    int32_t *     ids_d     = (int32_t *) ids->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ  with_norm = clamp != ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = ವಾದಗಳು_2.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = ವಾದಗಳು_2.delayed_softmax;

    ಸರತಿ_ptr stream = ctx.stream();
    ggml_sycl_ಗಣ_device(ctx.device);

    ಆಯ್ಕೆ (n_experts) {
        ಸ್ಥಿತಿ 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                               config);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                               config);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                               config);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                               config);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                                config);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                                config);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                                config);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                                 config);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                                 config);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_ಅಡ್ಡಸಾಲುಗಳು, n_expert_ಬಳಸಿದ, clamp_val, scale_val,
                                 config);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ASSERT(ಸುಳ್ಳು && "fatal error");
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_should_use_topk_moe(ಸ್ಥಿರಾಂಕ ggml_tensor * gating_op, ಸ್ಥಿರಾಂಕ ggml_tensor * weights,
                                          ಸ್ಥಿರಾಂಕ ggml_tensor * logits, ಸ್ಥಿರಾಂಕ ggml_tensor * ids) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_expert = ids->nb[1] / ids->nb[0];
    ಆದರೆ ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (gating_op->op == GGML_OP_SOFT_MAX) {
        ದಶಮಾಂಶ scale    = 1.0f;
        ದಶಮಾಂಶ max_bias = 0.0f;

        ಸ್ಮರಣೆ_ನಕಲು(&scale, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) gating_op->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) gating_op->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

        ಆದರೆ (!ggml_is_contiguous(gating_op->src[0])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (scale != 1.0f || max_bias != 0.0f) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // don't fuse when masks or sinks are present
        ಆದರೆ (gating_op->src[1] || gating_op->src[2]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (gating_op->op == GGML_OP_UNARY) {
        ಆದರೆ (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_topk_moe_fusion(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ggml_sycl_topk_moe_ವಾದಗಳು & ವಾದಗಳು_2) {
    ವಾದಗಳು_2 = ggml_sycl_topk_moe_ವಾದಗಳು{};

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    ಆದರೆ (nodes[ಗಂಟು_idx]->op == GGML_OP_SOFT_MAX) {
        ವಾದಗಳು_2.softmax = ನಿಜ;
    }

    ಆದರೆ (nodes[ಗಂಟು_idx]->op == GGML_OP_UNARY) {
        ಆದರೆ (ggml_get_unary_op(nodes[ಗಂಟು_idx]) != GGML_UNARY_OP_SIGMOID) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ವಾದಗಳು_2.sigmoid = ನಿಜ;
    }

    ಆದರೆ (nodes[ಗಂಟು_idx]->op == GGML_OP_ARGSORT) {
        ವಾದಗಳು_2.delayed_softmax = ನಿಜ;
    }

    ಗಂಟು_idx++;

    ಆದರೆ (ವಾದಗಳು_2.sigmoid || ವಾದಗಳು_2.softmax) {
        // SOFTMAX -> RESHAPE
        ಆದರೆ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_RESHAPE ||
            nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ggml_tensor * probs_reshaped = nodes[ಗಂಟು_idx];
        ಗಂಟು_idx++;

        ಆದರೆ (ಗಂಟು_idx >= n_nodes) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        ಆದರೆ (nodes[ಗಂಟು_idx]->op == GGML_OP_ಕೂಡಿಸು && nodes[ಗಂಟು_idx]->src[0] == nodes[ಗಂಟು_idx - 2]) {
            ವಾದಗಳು_2.prob_bias = ನಿಜ;
            ಗಂಟು_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        ಆದರೆ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_ARGSORT) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (ವಾದಗಳು_2.prob_bias && nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ವಾದಗಳು_2.prob_bias && nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 2]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಗಂಟು_idx++;

        // ARGSORT -> VIEW
        ಆದರೆ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_VIEW ||
            nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಂಟು_idx++;

        ಆದರೆ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // GET_ROWS
        ಆದರೆ (nodes[ಗಂಟು_idx]->src[0] != probs_reshaped || nodes[ಗಂಟು_idx]->src[1] != nodes[ಗಂಟು_idx - 1]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಂಟು_idx++;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ವಾದಗಳು_2.delayed_softmax) {
        ಆದರೆ (ಗಂಟು_idx - 2 < 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ggml_tensor * probs_reshaped = nodes[ಗಂಟು_idx - 2];

        // VIEW -> ARGSORT
        ಆದರೆ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != GGML_OP_VIEW ||
            nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಂಟು_idx++;

        // GET_ROWS
        ಆದರೆ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->src[1] != nodes[ಗಂಟು_idx - 1] ||
            nodes[ಗಂಟು_idx]->src[0] != probs_reshaped) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಂಟು_idx++;

        ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ಉಳಿದ_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_op op : ಉಳಿದ_ops) {
            ಆದರೆ (ಗಂಟು_idx >= n_nodes || nodes[ಗಂಟು_idx]->op != op || nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಗಂಟು_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    ಆದರೆ (ಗಂಟು_idx >= n_nodes) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (nodes[ಗಂಟು_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು, GGML_OP_CLAMP };

        ವಾದಗಳು_2.norm = ನಿಜ;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_op op : norm_ops) {
            ಆದರೆ (nodes[ಗಂಟು_idx]->op == op && nodes[ಗಂಟು_idx]->src[0] == nodes[ಗಂಟು_idx - 1]) {
                ಗಂಟು_idx++;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ವಾದಗಳು_2.norm = ಸುಳ್ಳು;
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }

        // DIV <- CLAMP, RESHAPE
        ಆದರೆ (nodes[ಗಂಟು_idx]->op != GGML_OP_DIV || nodes[ಗಂಟು_idx]->src[1] != nodes[ಗಂಟು_idx - 1] ||
            nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 3]) {
            ವಾದಗಳು_2.norm = ಸುಳ್ಳು;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಗಂಟು_idx++;

        ಆದರೆ (nodes[ಗಂಟು_idx]->op != GGML_OP_RESHAPE || nodes[ಗಂಟು_idx]->src[0] != nodes[ಗಂಟು_idx - 1]) {
            ವಾದಗಳು_2.norm = ಸುಳ್ಳು;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಗಂಟು_idx++;
    }

    ಆದರೆ (nodes[ಗಂಟು_idx]->op == GGML_OP_SCALE && nodes[ಗಂಟು_idx]->src[0] == nodes[ಗಂಟು_idx - 1]) {
        ವಾದಗಳು_2.scale = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_check_fusion_ಸ್ಮರಣೆ_ranges(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_ಎಣಿಕೆ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಹೊರಹೋಗು_nodes, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_ಎಣಿಕೆ,
                                                 ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_topk_moe = ಸುಳ್ಳು) {
    ಸ್ವಯಂಚಾಲಿತ nodes_overlap = [&](ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b) {
        ಸ್ಥಿರಾಂಕ int64_t a_ಆರಂಭ = (int64_t) a->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ int64_t a_ಕೊನೆ   = a_ಆರಂಭ + ggml_backend_buft_get_alloc_ಗಾತ್ರ(a->ಬಫರ್->buft, a);

        ಸ್ಥಿರಾಂಕ int64_t b_ಆರಂಭ = (int64_t) b->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ int64_t b_ಕೊನೆ   = b_ಆರಂಭ + ggml_backend_buft_get_alloc_ಗಾತ್ರ(b->ಬಫರ್->buft, b);

        ಆದರೆ ((b_ಆರಂಭ <= a_ಆರಂಭ && a_ಆರಂಭ < b_ಕೊನೆ) || (a_ಆರಂಭ <= b_ಆರಂಭ && b_ಆರಂಭ < a_ಕೊನೆ)) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    };

    ತಾರ್ಕಿಕ is_ok = ನಿಜ;
    // exception for topk-moe, as each row is read entirely before writing
    ಆದರೆ (ggml_nrows(cgraph->nodes[ಗಂಟು_idx]) == 1 && is_topk_moe) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಹೊರಹೋಗು_ಎಣಿಕೆ; ++i) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * dst = cgraph->nodes[ಹೊರಹೋಗು_nodes[i]];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ಗಂಟು_idx; j < ಗಂಟು_idx + ಗಂಟು_ಎಣಿಕೆ; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                ಆದರೆ (!src || src->op == GGML_OP_NONE) {
                    ಮುಂದುವರಿ;
                }

                ಆದರೆ (nodes_overlap(dst, src)) {
                    ತಾರ್ಕಿಕ ಸಿಕ್ಕಿತು = ಸುಳ್ಳು;

                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = ಗಂಟು_idx; k < j; ++k) {
                        ಆದರೆ (cgraph->nodes[k] == src) {
                            ಸಿಕ್ಕಿತು = ನಿಜ;
                            ನಿಲ್ಲಿಸು;
                        }
                    }

                    ಆದರೆ (!ಸಿಕ್ಕಿತು) {
                        is_ok = ಸುಳ್ಳು;
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ is_ok;
}

ಪೂರ್ಣಾಂಕ ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ i) {
    ಆದರೆ (!g_ggml_sycl_enable_fusion) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಹಿಂತಿರುಗಿ ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

ಪೂರ್ಣಾಂಕ ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ i) {
    ggml_tensor * ಗಂಟು = cgraph->nodes[i];

    ಆದರೆ (ಗಂಟು->op != GGML_OP_UNARY && ಗಂಟು->op != GGML_OP_SOFT_MAX && ಗಂಟು->op != GGML_OP_ARGSORT) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ggml_sycl_topk_moe_ವಾದಗಳು ವಾದಗಳು_2;
    ಆದರೆ (!ggml_sycl_topk_moe_fusion(cgraph, i, ವಾದಗಳು_2)) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    ಆದರೆ (ವಾದಗಳು_2.prob_bias) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * logits  = ಗಂಟು->src[0];
    ggml_tensor *       weights = ಶೂನ್ಯಸೂಚಕ;
    ggml_tensor *       ids     = ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ggml_tensor * clamp   = ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ggml_tensor * scale   = ಶೂನ್ಯಸೂಚಕ;

    std::vector<ggml_op> ops;
    ಪೂರ್ಣಾಂಕ                  ಹೊರಹೋಗು_nodes[2];

    ಆದರೆ (!ವಾದಗಳು_2.delayed_softmax) {
        ಸ್ಥಿರಾಂಕ ggml_op gating_op = ವಾದಗಳು_2.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು });
        ಹೊರಹೋಗು_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        ಆದರೆ (ವಾದಗಳು_2.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (ಪೂರ್ಣಾಂಕ) ops.size() - 3];
        }
        ಆದರೆ (ವಾದಗಳು_2.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (ಪೂರ್ಣಾಂಕ) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (ಪೂರ್ಣಾಂಕ) ops.size() - 1];
        ಹೊರಹೋಗು_nodes[1] = i + (ಪೂರ್ಣಾಂಕ) ops.size() - 1;

        ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.ದತ್ತಾಂಶ(), ಹೊರಹೋಗು_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(ಗಂಟು, weights, logits, ids) &&
            ggml_sycl_check_fusion_ಸ್ಮರಣೆ_ranges(cgraph, i, (ಪೂರ್ಣಾಂಕ) ops.size(), ಹೊರಹೋಗು_nodes, 2, /*is_topk_moe=*/ನಿಜ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, ವಾದಗಳು_2);
            ಹಿಂತಿರುಗಿ (ಪೂರ್ಣಾಂಕ) ops.size() - 1;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ವಾದಗಳು_2.norm && !ವಾದಗಳು_2.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * softmax = cgraph->nodes[i + 4];
        ಹೊರಹೋಗು_nodes[0]                = i + 1;
        ಹೊರಹೋಗು_nodes[1]                = i + 5;

        ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.ದತ್ತಾಂಶ(), ಹೊರಹೋಗು_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_ಸ್ಮರಣೆ_ranges(cgraph, i, (ಪೂರ್ಣಾಂಕ) ops.size(), ಹೊರಹೋಗು_nodes, 2, /*is_topk_moe=*/ನಿಜ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, ವಾದಗಳು_2);
            ಹಿಂತಿರುಗಿ (ಪೂರ್ಣಾಂಕ) ops.size() - 1;
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
