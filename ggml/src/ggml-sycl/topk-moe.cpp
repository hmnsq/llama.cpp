#ਅਨ੍ਤਰ੍ਭਾਵ <cfloat>
#ਅਨ੍ਤਰ੍ਭਾਵ <initializer_list>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

ਬਣਤਰ ggml_sycl_topk_moe_ਦਲੀਲਾਂ {
    ਤਾਰਕਿਕ sigmoid{};
    ਤਾਰਕਿਕ softmax{};
    ਤਾਰਕਿਕ delayed_softmax{};
    ਤਾਰਕਿਕ prob_bias{};
    ਤਾਰਕਿਕ norm{};
    ਤਾਰਕਿਕ scale{};
};

ਬਣਤਰ topk_moe_config {
    ਤਾਰਕਿਕ use_sigmoid;
    ਤਾਰਕਿਕ with_norm;
    ਤਾਰਕਿਕ delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
ਸਾਂਚਾ <ਪੂਰਨ experts_ਪ੍ਰਤੀ_thread, ਤਾਰਕਿਕ use_ਹੱਦਬੰਦੀ>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ softmax_warp_inplace(ਦਸ਼ਮਲਵ (&vals)[experts_ਪ੍ਰਤੀ_thread], ਅਚਲ ਪੂਰਨ ਹੱਦਬੰਦੀ, ਅਚਲ ਪੂਰਨ lane) {
    ਦਸ਼ਮਲਵ max_val = -INFINITY;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < experts_ਪ੍ਰਤੀ_thread; i++) {
        ਅਚਲ ਪੂਰਨ  idx    = lane + i * WARP_ਮਾਪ;
        ਅਚਲ ਤਾਰਕਿਕ ਸਰਗਰਮ = !use_ਹੱਦਬੰਦੀ || (idx < ਹੱਦਬੰਦੀ);
        ਜੇ (ਸਰਗਰਮ) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_ਮਾਪ>(max_val);

    ਦਸ਼ਮਲਵ ਜੋੜ = 0.f;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < experts_ਪ੍ਰਤੀ_thread; i++) {
        ਅਚਲ ਪੂਰਨ  idx    = lane + i * WARP_ਮਾਪ;
        ਅਚਲ ਤਾਰਕਿਕ ਸਰਗਰਮ = !use_ਹੱਦਬੰਦੀ || (idx < ਹੱਦਬੰਦੀ);
        ਜੇ (ਸਰਗਰਮ) {
            ਅਚਲ ਦਸ਼ਮਲਵ val = sycl::ਘਾਤਾਂਕ(vals[i] - max_val);
            vals[i]         = val;
            ਜੋੜ += val;
        } ਨਹੀਂਤਾਂ {
            vals[i] = 0.f;
        }
    }
    ਜੋੜ = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(ਜੋੜ);

    ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = 1.0f / ਜੋੜ;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < experts_ਪ੍ਰਤੀ_thread; i++) {
        ਅਚਲ ਪੂਰਨ idx = lane + i * WARP_ਮਾਪ;
        ਜੇ (!use_ਹੱਦਬੰਦੀ || idx < ਹੱਦਬੰਦੀ) {
            vals[i] *= inv_ਜੋੜ;
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ experts_ਪ੍ਰਤੀ_thread, ਤਾਰਕਿਕ use_ਹੱਦਬੰਦੀ>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ sigmoid_warp_inplace(ਦਸ਼ਮਲਵ (&vals)[experts_ਪ੍ਰਤੀ_thread], ਅਚਲ ਪੂਰਨ ਹੱਦਬੰਦੀ, ਅਚਲ ਪੂਰਨ lane) {
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < experts_ਪ੍ਰਤੀ_thread; i++) {
        ਅਚਲ ਪੂਰਨ  idx    = lane + i * WARP_ਮਾਪ;
        ਅਚਲ ਤਾਰਕਿਕ ਸਰਗਰਮ = !use_ਹੱਦਬੰਦੀ || (idx < ਹੱਦਬੰਦੀ);
        vals[i]           = ਸਰਗਰਮ ? 1.f / (1.f + sycl::ਘਾਤਾਂਕ(-vals[i])) : -INFINITY;
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
ਸਾਂਚਾ <ਪੂਰਨ n_experts>
ਸਥਿਰ ਖਾਲੀ topk_moe_kernel(ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ logits,
                            ਦਸ਼ਮਲਵ * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ਅਚਲ ਪੂਰਨ                  n_ਪੰਕਤੀਆਂ,
                            ਅਚਲ ਪੂਰਨ                  n_expert_ਵਰਤਿਆ,
                            ਅਚਲ ਦਸ਼ਮਲਵ                clamp_val,
                            ਅਚਲ ਦਸ਼ਮਲਵ                scale_val,
                            ਅਚਲ topk_moe_config       config) {
    ਸਵੈਚਾਲਿਤ      ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<1>();
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ       = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
    ਜੇ (ਪੰਕਤੀ >= n_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ;
    }
    ਅਚਲ ਪੂਰਨ lane = ਵਸਤੂ_ct1.get_local_id(0);

    logits  += (size_t) n_experts * ਪੰਕਤੀ;
    weights += (size_t) n_expert_ਵਰਤਿਆ * ਪੰਕਤੀ;
    ids     += (size_t) n_experts * ਪੰਕਤੀ;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr ਪੂਰਨ experts_ਪ੍ਰਤੀ_thread = (n_experts > WARP_ਮਾਪ) ? n_experts / WARP_ਮਾਪ : 1;

    ਦਸ਼ਮਲਵ wt[experts_ਪ੍ਰਤੀ_thread];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < experts_ਪ੍ਰਤੀ_thread; i++) {
        wt[i] = -INFINITY;
    }
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < n_experts; i += WARP_ਮਾਪ) {
        ਅਚਲ ਪੂਰਨ expert  = i + lane;
        wt[i / WARP_ਮਾਪ] = (n_experts % WARP_ਮਾਪ == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    ਜੇ (!config.delayed_softmax) {
        ਜੇ (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_ਪ੍ਰਤੀ_thread, ਝੂਠ>(wt, n_experts, lane);
        } ਨਹੀਂਤਾਂ {
            softmax_warp_inplace<experts_ਪ੍ਰਤੀ_thread, ਝੂਠ>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < experts_ਪ੍ਰਤੀ_thread; i++) {
        ਜੇ (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    ਦਸ਼ਮਲਵ wt_ਜੋੜ = 0.f;
    ਦਸ਼ਮਲਵ output_weights[experts_ਪ੍ਰਤੀ_thread];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < experts_ਪ੍ਰਤੀ_thread; i++) {
        output_weights[i] = 0.f;
    }

    ਅਚਲ sycl::sub_ਸਮੂਹ sg = ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ();

    ਲਈ (ਪੂਰਨ k = 0; k < n_expert_ਵਰਤਿਆ; k++) {
        ਦਸ਼ਮਲਵ max_val    = wt[0];
        ਪੂਰਨ   max_expert = lane;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 1; i < experts_ਪ੍ਰਤੀ_thread; i++) {
            ਅਚਲ ਪੂਰਨ expert = lane + i * WARP_ਮਾਪ;
            ਜੇ ((n_experts % WARP_ਮਾਪ == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
            ਅਚਲ ਦਸ਼ਮਲਵ val    = dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(sg, max_val, ਨਕਾਬ);
            ਅਚਲ ਪੂਰਨ   expert = dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(sg, max_expert, ਨਕਾਬ);
            ਜੇ (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        ਜੇ ((max_expert & (WARP_ਮਾਪ - 1)) == lane) {
            wt[max_expert / WARP_ਮਾਪ] = -INFINITY;
        }
        ਜੇ ((k & (WARP_ਮਾਪ - 1)) == lane) {
            output_weights[k / WARP_ਮਾਪ] = max_val;
        }
        ਜੇ ((max_expert & (WARP_ਮਾਪ - 1)) == lane) {
            ids[k] = max_expert;
            ਜੇ (config.with_norm) {
                wt_ਜੋੜ += max_val;
            }
        }
    }

    ਜੇ (config.with_norm) {
        wt_ਜੋੜ          = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(wt_ਜੋੜ);
        wt_ਜੋੜ          = sycl::fmax(wt_ਜੋੜ, clamp_val);
        ਅਚਲ ਦਸ਼ਮਲਵ inv = 1.0f / wt_ਜੋੜ;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 0; i < experts_ਪ੍ਰਤੀ_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    ਜੇ (config.delayed_softmax) {
        softmax_warp_inplace<experts_ਪ੍ਰਤੀ_thread, ਸੱਚ>(output_weights, n_expert_ਵਰਤਿਆ, lane);
    }

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < experts_ਪ੍ਰਤੀ_thread; i++) {
        ਅਚਲ ਪੂਰਨ idx = i * WARP_ਮਾਪ + lane;
        ਜੇ (idx < n_expert_ਵਰਤਿਆ) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ n_experts>
ਸਥਿਰ ਖਾਲੀ launch_topk_moe(ਕਤਾਰ_ptr stream, ਅਚਲ ਦਸ਼ਮਲਵ * logits, ਦਸ਼ਮਲਵ * weights, int32_t * ids, ਪੂਰਨ n_ਪੰਕਤੀਆਂ,
                            ਪੂਰਨ n_expert_ਵਰਤਿਆ, ਦਸ਼ਮਲਵ clamp_val, ਦਸ਼ਮਲਵ scale_val, ਅਚਲ topk_moe_config & config) {
    ਅਚਲ sycl::range<1> ਖੰਡ_dims(WARP_ਮਾਪ);
    ਅਚਲ sycl::range<1> ਖੰਡ_nums(n_ਪੰਕਤੀਆਂ);
    stream->parallel_for(sycl::nd_range<1>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val,
                                                        scale_val, config);
                             GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ਅਚਲ ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ਅਚਲ ggml_tensor *             clamp,
                                  ਅਚਲ ggml_tensor *             scale,
                                  ਅਚਲ ggml_sycl_topk_moe_ਦਲੀਲਾਂ & ਦਲੀਲਾਂ_2) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ਅਚਲ ਪੂਰਨ n_experts     = logits->ne[0];
    ਅਚਲ ਪੂਰਨ n_ਪੰਕਤੀਆਂ        = logits->ne[1];
    ਅਚਲ ਪੂਰਨ n_expert_ਵਰਤਿਆ = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_ਮਾਪ(ids->type) == (size_t) n_experts);

    ਅਚਲ ਦਸ਼ਮਲਵ * logits_d  = (ਅਚਲ ਦਸ਼ਮਲਵ *) logits->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       weights_d = (ਦਸ਼ਮਲਵ *) weights->ਅੰਕੜੇ;
    int32_t *     ids_d     = (int32_t *) ids->ਅੰਕੜੇ;

    ਅਚਲ ਤਾਰਕਿਕ  with_norm = clamp != ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ਦਸ਼ਮਲਵ clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ਅਚਲ ਦਸ਼ਮਲਵ scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = ਦਲੀਲਾਂ_2.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = ਦਲੀਲਾਂ_2.delayed_softmax;

    ਕਤਾਰ_ptr stream = ctx.stream();
    ggml_sycl_ਸੈੱਟ_device(ctx.device);

    ਚੋਣ (n_experts) {
        ਹਾਲਤ 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                               config);
            ਰੋਕੋ;
        ਹਾਲਤ 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                               config);
            ਰੋਕੋ;
        ਹਾਲਤ 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                               config);
            ਰੋਕੋ;
        ਹਾਲਤ 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                               config);
            ਰੋਕੋ;
        ਹਾਲਤ 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                                config);
            ਰੋਕੋ;
        ਹਾਲਤ 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                                config);
            ਰੋਕੋ;
        ਹਾਲਤ 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                                config);
            ਰੋਕੋ;
        ਹਾਲਤ 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                                 config);
            ਰੋਕੋ;
        ਹਾਲਤ 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                                 config);
            ਰੋਕੋ;
        ਹਾਲਤ 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_ਪੰਕਤੀਆਂ, n_expert_ਵਰਤਿਆ, clamp_val, scale_val,
                                 config);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ASSERT(ਝੂਠ && "fatal error");
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_should_use_topk_moe(ਅਚਲ ggml_tensor * gating_op, ਅਚਲ ggml_tensor * weights,
                                          ਅਚਲ ggml_tensor * logits, ਅਚਲ ggml_tensor * ids) {
    ਅਚਲ ਪੂਰਨ n_expert = ids->nb[1] / ids->nb[0];
    ਜੇ ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (gating_op->op == GGML_OP_SOFT_MAX) {
        ਦਸ਼ਮਲਵ scale    = 1.0f;
        ਦਸ਼ਮਲਵ max_bias = 0.0f;

        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਅਚਲ ਦਸ਼ਮਲਵ *) gating_op->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) gating_op->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        ਜੇ (!ggml_is_contiguous(gating_op->src[0])) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (scale != 1.0f || max_bias != 0.0f) {
            ਵਾਪਸ ਝੂਠ;
        }
        // don't fuse when masks or sinks are present
        ਜੇ (gating_op->src[1] || gating_op->src[2]) {
            ਵਾਪਸ ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (gating_op->op == GGML_OP_UNARY) {
        ਜੇ (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_topk_moe_fusion(ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ggml_sycl_topk_moe_ਦਲੀਲਾਂ & ਦਲੀਲਾਂ_2) {
    ਦਲੀਲਾਂ_2 = ggml_sycl_topk_moe_ਦਲੀਲਾਂ{};

    ਅਚਲ ਪੂਰਨ      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    ਜੇ (nodes[ਗੰਢ_idx]->op == GGML_OP_SOFT_MAX) {
        ਦਲੀਲਾਂ_2.softmax = ਸੱਚ;
    }

    ਜੇ (nodes[ਗੰਢ_idx]->op == GGML_OP_UNARY) {
        ਜੇ (ggml_get_unary_op(nodes[ਗੰਢ_idx]) != GGML_UNARY_OP_SIGMOID) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਦਲੀਲਾਂ_2.sigmoid = ਸੱਚ;
    }

    ਜੇ (nodes[ਗੰਢ_idx]->op == GGML_OP_ARGSORT) {
        ਦਲੀਲਾਂ_2.delayed_softmax = ਸੱਚ;
    }

    ਗੰਢ_idx++;

    ਜੇ (ਦਲੀਲਾਂ_2.sigmoid || ਦਲੀਲਾਂ_2.softmax) {
        // SOFTMAX -> RESHAPE
        ਜੇ (ਗੰਢ_idx >= n_nodes || nodes[ਗੰਢ_idx]->op != GGML_OP_RESHAPE ||
            nodes[ਗੰਢ_idx]->src[0] != nodes[ਗੰਢ_idx - 1]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ggml_tensor * probs_reshaped = nodes[ਗੰਢ_idx];
        ਗੰਢ_idx++;

        ਜੇ (ਗੰਢ_idx >= n_nodes) {
            ਵਾਪਸ ਝੂਠ;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        ਜੇ (nodes[ਗੰਢ_idx]->op == GGML_OP_ਜੋੜੋ && nodes[ਗੰਢ_idx]->src[0] == nodes[ਗੰਢ_idx - 2]) {
            ਦਲੀਲਾਂ_2.prob_bias = ਸੱਚ;
            ਗੰਢ_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        ਜੇ (ਗੰਢ_idx >= n_nodes || nodes[ਗੰਢ_idx]->op != GGML_OP_ARGSORT) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (ਦਲੀਲਾਂ_2.prob_bias && nodes[ਗੰਢ_idx]->src[0] != nodes[ਗੰਢ_idx - 1]) {
            ਵਾਪਸ ਝੂਠ;
        } ਨਹੀਂਤਾਂ ਜੇ (!ਦਲੀਲਾਂ_2.prob_bias && nodes[ਗੰਢ_idx]->src[0] != nodes[ਗੰਢ_idx - 2]) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਗੰਢ_idx++;

        // ARGSORT -> VIEW
        ਜੇ (ਗੰਢ_idx >= n_nodes || nodes[ਗੰਢ_idx]->op != GGML_OP_VIEW ||
            nodes[ਗੰਢ_idx]->src[0] != nodes[ਗੰਢ_idx - 1]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਗੰਢ_idx++;

        ਜੇ (ਗੰਢ_idx >= n_nodes || nodes[ਗੰਢ_idx]->op != GGML_OP_GET_ਪੰਕਤੀਆਂ) {
            ਵਾਪਸ ਝੂਠ;
        }

        // GET_ROWS
        ਜੇ (nodes[ਗੰਢ_idx]->src[0] != probs_reshaped || nodes[ਗੰਢ_idx]->src[1] != nodes[ਗੰਢ_idx - 1]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਗੰਢ_idx++;
    } ਨਹੀਂਤਾਂ ਜੇ (ਦਲੀਲਾਂ_2.delayed_softmax) {
        ਜੇ (ਗੰਢ_idx - 2 < 0) {
            ਵਾਪਸ ਝੂਠ;
        }
        ggml_tensor * probs_reshaped = nodes[ਗੰਢ_idx - 2];

        // VIEW -> ARGSORT
        ਜੇ (ਗੰਢ_idx >= n_nodes || nodes[ਗੰਢ_idx]->op != GGML_OP_VIEW ||
            nodes[ਗੰਢ_idx]->src[0] != nodes[ਗੰਢ_idx - 1]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਗੰਢ_idx++;

        // GET_ROWS
        ਜੇ (ਗੰਢ_idx >= n_nodes || nodes[ਗੰਢ_idx]->src[1] != nodes[ਗੰਢ_idx - 1] ||
            nodes[ਗੰਢ_idx]->src[0] != probs_reshaped) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਗੰਢ_idx++;

        ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ਬਚਿਆ_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        ਲਈ (ਅਚਲ ggml_op op : ਬਚਿਆ_ops) {
            ਜੇ (ਗੰਢ_idx >= n_nodes || nodes[ਗੰਢ_idx]->op != op || nodes[ਗੰਢ_idx]->src[0] != nodes[ਗੰਢ_idx - 1]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਗੰਢ_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    ਜੇ (ਗੰਢ_idx >= n_nodes) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (nodes[ਗੰਢ_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ਸਥਿਰ ਅਚਲ std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ, GGML_OP_CLAMP };

        ਦਲੀਲਾਂ_2.norm = ਸੱਚ;
        ਲਈ (ਅਚਲ ggml_op op : norm_ops) {
            ਜੇ (nodes[ਗੰਢ_idx]->op == op && nodes[ਗੰਢ_idx]->src[0] == nodes[ਗੰਢ_idx - 1]) {
                ਗੰਢ_idx++;
            } ਨਹੀਂਤਾਂ {
                ਦਲੀਲਾਂ_2.norm = ਝੂਠ;
                ਵਾਪਸ ਸੱਚ;
            }
        }

        // DIV <- CLAMP, RESHAPE
        ਜੇ (nodes[ਗੰਢ_idx]->op != GGML_OP_DIV || nodes[ਗੰਢ_idx]->src[1] != nodes[ਗੰਢ_idx - 1] ||
            nodes[ਗੰਢ_idx]->src[0] != nodes[ਗੰਢ_idx - 3]) {
            ਦਲੀਲਾਂ_2.norm = ਝੂਠ;
            ਵਾਪਸ ਸੱਚ;
        }
        ਗੰਢ_idx++;

        ਜੇ (nodes[ਗੰਢ_idx]->op != GGML_OP_RESHAPE || nodes[ਗੰਢ_idx]->src[0] != nodes[ਗੰਢ_idx - 1]) {
            ਦਲੀਲਾਂ_2.norm = ਝੂਠ;
            ਵਾਪਸ ਸੱਚ;
        }
        ਗੰਢ_idx++;
    }

    ਜੇ (nodes[ਗੰਢ_idx]->op == GGML_OP_SCALE && nodes[ਗੰਢ_idx]->src[0] == nodes[ਗੰਢ_idx - 1]) {
        ਦਲੀਲਾਂ_2.scale = ਸੱਚ;
    }

    ਵਾਪਸ ਸੱਚ;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_check_fusion_ਸਮ੍ਰਿਤੀ_ranges(ਅਚਲ ggml_cgraph * cgraph, ਅਚਲ ਪੂਰਨ ਗੰਢ_idx,
                                                 ਅਚਲ ਪੂਰਨ ਗੰਢ_ਗਿਣਤੀ, ਅਚਲ ਪੂਰਨ * ਬਾਹਰਲਾ_nodes, ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_ਗਿਣਤੀ,
                                                 ਅਚਲ ਤਾਰਕਿਕ is_topk_moe = ਝੂਠ) {
    ਸਵੈਚਾਲਿਤ nodes_overlap = [&](ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b) {
        ਅਚਲ int64_t a_ਸ਼ੁਰੂ = (int64_t) a->ਅੰਕੜੇ;
        ਅਚਲ int64_t a_ਸਿਰਾ   = a_ਸ਼ੁਰੂ + ggml_backend_buft_get_alloc_ਮਾਪ(a->ਬਫਰ->buft, a);

        ਅਚਲ int64_t b_ਸ਼ੁਰੂ = (int64_t) b->ਅੰਕੜੇ;
        ਅਚਲ int64_t b_ਸਿਰਾ   = b_ਸ਼ੁਰੂ + ggml_backend_buft_get_alloc_ਮਾਪ(b->ਬਫਰ->buft, b);

        ਜੇ ((b_ਸ਼ੁਰੂ <= a_ਸ਼ੁਰੂ && a_ਸ਼ੁਰੂ < b_ਸਿਰਾ) || (a_ਸ਼ੁਰੂ <= b_ਸ਼ੁਰੂ && b_ਸ਼ੁਰੂ < a_ਸਿਰਾ)) {
            ਵਾਪਸ ਸੱਚ;
        }

        ਵਾਪਸ ਝੂਠ;
    };

    ਤਾਰਕਿਕ is_ok = ਸੱਚ;
    // exception for topk-moe, as each row is read entirely before writing
    ਜੇ (ggml_nrows(cgraph->nodes[ਗੰਢ_idx]) == 1 && is_topk_moe) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < ਬਾਹਰਲਾ_ਗਿਣਤੀ; ++i) {
        ਅਚਲ ggml_tensor * dst = cgraph->nodes[ਬਾਹਰਲਾ_nodes[i]];

        ਲਈ (ਪੂਰਨ j = ਗੰਢ_idx; j < ਗੰਢ_idx + ਗੰਢ_ਗਿਣਤੀ; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            ਲਈ (ਪੂਰਨ src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ਅਚਲ ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                ਜੇ (!src || src->op == GGML_OP_NONE) {
                    ਜਾਰੀ;
                }

                ਜੇ (nodes_overlap(dst, src)) {
                    ਤਾਰਕਿਕ ਮਿਲਿਆ = ਝੂਠ;

                    ਲਈ (ਪੂਰਨ k = ਗੰਢ_idx; k < j; ++k) {
                        ਜੇ (cgraph->nodes[k] == src) {
                            ਮਿਲਿਆ = ਸੱਚ;
                            ਰੋਕੋ;
                        }
                    }

                    ਜੇ (!ਮਿਲਿਆ) {
                        is_ok = ਝੂਠ;
                        ਰੋਕੋ;
                    }
                }
            }
        }
    }

    ਵਾਪਸ is_ok;
}

ਪੂਰਨ ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ਪੂਰਨ i) {
    ਜੇ (!g_ggml_sycl_enable_fusion) {
        ਵਾਪਸ 0;
    }

    ਵਾਪਸ ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

ਪੂਰਨ ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ਪੂਰਨ i) {
    ggml_tensor * ਗੰਢ = cgraph->nodes[i];

    ਜੇ (ਗੰਢ->op != GGML_OP_UNARY && ਗੰਢ->op != GGML_OP_SOFT_MAX && ਗੰਢ->op != GGML_OP_ARGSORT) {
        ਵਾਪਸ 0;
    }

    ggml_sycl_topk_moe_ਦਲੀਲਾਂ ਦਲੀਲਾਂ_2;
    ਜੇ (!ggml_sycl_topk_moe_fusion(cgraph, i, ਦਲੀਲਾਂ_2)) {
        ਵਾਪਸ 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    ਜੇ (ਦਲੀਲਾਂ_2.prob_bias) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ggml_tensor * logits  = ਗੰਢ->src[0];
    ggml_tensor *       weights = ਖਾਲੀ_ਸੂਚਕ;
    ggml_tensor *       ids     = ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ggml_tensor * clamp   = ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ggml_tensor * scale   = ਖਾਲੀ_ਸੂਚਕ;

    std::vector<ggml_op> ops;
    ਪੂਰਨ                  ਬਾਹਰਲਾ_nodes[2];

    ਜੇ (!ਦਲੀਲਾਂ_2.delayed_softmax) {
        ਅਚਲ ggml_op gating_op = ਦਲੀਲਾਂ_2.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ਪੰਕਤੀਆਂ });
        ਬਾਹਰਲਾ_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        ਜੇ (ਦਲੀਲਾਂ_2.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (ਪੂਰਨ) ops.size() - 3];
        }
        ਜੇ (ਦਲੀਲਾਂ_2.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (ਪੂਰਨ) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (ਪੂਰਨ) ops.size() - 1];
        ਬਾਹਰਲਾ_nodes[1] = i + (ਪੂਰਨ) ops.size() - 1;

        ਜੇ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.ਅੰਕੜੇ(), ਬਾਹਰਲਾ_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(ਗੰਢ, weights, logits, ids) &&
            ggml_sycl_check_fusion_ਸਮ੍ਰਿਤੀ_ranges(cgraph, i, (ਪੂਰਨ) ops.size(), ਬਾਹਰਲਾ_nodes, 2, /*is_topk_moe=*/ਸੱਚ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, ਦਲੀਲਾਂ_2);
            ਵਾਪਸ (ਪੂਰਨ) ops.size() - 1;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (!ਦਲੀਲਾਂ_2.norm && !ਦਲੀਲਾਂ_2.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ਪੰਕਤੀਆਂ, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ਅਚਲ ggml_tensor * softmax = cgraph->nodes[i + 4];
        ਬਾਹਰਲਾ_nodes[0]                = i + 1;
        ਬਾਹਰਲਾ_nodes[1]                = i + 5;

        ਜੇ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.ਅੰਕੜੇ(), ਬਾਹਰਲਾ_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_ਸਮ੍ਰਿਤੀ_ranges(cgraph, i, (ਪੂਰਨ) ops.size(), ਬਾਹਰਲਾ_nodes, 2, /*is_topk_moe=*/ਸੱਚ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, ਦਲੀਲਾਂ_2);
            ਵਾਪਸ (ਪੂਰਨ) ops.size() - 1;
        }
    }

    ਵਾਪਸ 0;
}
