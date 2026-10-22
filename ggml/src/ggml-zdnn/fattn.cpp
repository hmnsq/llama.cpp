#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "fattn.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_zdnn_fattn_ಮುಖವಾಡ_ಮೌಲ್ಯ(ggml_fp16_t m) {
    ಹಿಂತಿರುಗಿ std::max(GGML_FP16_ಗಮ್ಯ_FP32(m), -65504.0f);
}

ತಾರ್ಕಿಕ ggml_zdnn_supports_flash_attn_ext(ಸ್ಥಿರಾಂಕ ggml_backend_zdnn_device_context * ctx_dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * q    = op->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k    = op->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * v    = op->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = op->src[3];

    ದಶಮಾಂಶ max_bias      = 0.0f;
    ದಶಮಾಂಶ logit_softcap = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias,      (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) op->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) op->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    ಆದರೆ (op->src[4] != ಶೂನ್ಯಸೂಚಕ || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ && (ಮುಖವಾಡ->type != GGML_TYPE_F16 || ಮುಖವಾಡ->ne[2] != 1 || ಮುಖವಾಡ->ne[3] != 1)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    ಆದರೆ (q->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || k->nb[0] != ಗಾತ್ರ(ggml_fp16_t) || v->nb[0] != ಗಾತ್ರ(ggml_fp16_t)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t max_dim = ctx_dev->max_size;
    ಸ್ಥಿರಾಂಕ int64_t ಗುಂಪು   = q->ne[2] / k->ne[2];

    ಹಿಂತಿರುಗಿ q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && ಗುಂಪು * q->ne[1] <= max_dim;
}

ಶೂನ್ಯ ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * q    = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k    = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * v    = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = dst->src[3];

    ಸ್ಥಿರಾಂಕ int64_t d_k       = q->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_q       = q->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ತಲೆ    = q->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_kv      = k->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ತಲೆ_kv = k->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t d_v       = v->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ಗುಂಪು     = n_ತಲೆ / n_ತಲೆ_kv;

    ದಶಮಾಂಶ scale = 1.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ int32_t n_ಎಳೆಗಳು = n_q*n_kv >= 65536 ? ctx->n_ಎಳೆಗಳು : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ single = n_q == 1;

    ಸ್ಥಿರಾಂಕ int64_t n_ಮುಖವಾಡ = ಮುಖವಾಡ ? (single ? n_ತಲೆ_kv*n_kv : n_q*n_kv) : 0;

    ggml_zdnn_pool_alloc<ದಶಮಾಂಶ>       q_host_alloc(ctx->pool, n_ತಲೆ*n_q*d_k);
    ggml_zdnn_pool_alloc<ದಶಮಾಂಶ>       o_host_alloc(ctx->pool, n_ತಲೆ*n_q*d_v);
    ggml_zdnn_pool_alloc<ದಶಮಾಂಶ>       ಮುಖವಾಡ_host_alloc(ctx->pool, n_ಮುಖವಾಡ);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_ತಲೆ_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_ತಲೆ_kv*n_kv*d_v);

    ದಶಮಾಂಶ       * q_host    = q_host_alloc.get();
    ದಶಮಾಂಶ       * o_host    = o_host_alloc.get();
    ದಶಮಾಂಶ       * ಮುಖವಾಡ_host = ಮುಖವಾಡ_host_alloc.get();
    ggml_fp16_t * k_host    = k_host_alloc.get();
    ggml_fp16_t * v_host    = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_ಎಳೆಗಳು, n_ತಲೆ, [&](int64_t h_ಆರಂಭ, int64_t h_ಕೊನೆ) {
        ಗಾಗಿ (int64_t h = h_ಆರಂಭ; h < h_ಕೊನೆ; h++) {
            ಗಾಗಿ (int64_t iq = 0; iq < n_q; iq++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ + iq*q->nb[1] + h*q->nb[2]);
                ದಶಮಾಂಶ       * ಅಡ್ಡಸಾಲು = q_host + (h*n_q + iq)*d_k;
                ಗಾಗಿ (int64_t i0 = 0; i0 < d_k; i0++) {
                    ಅಡ್ಡಸಾಲು[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_ಎಳೆಗಳು, n_ತಲೆ_kv*n_kv, [&](int64_t r_ಆರಂಭ, int64_t r_ಕೊನೆ) {
        ಗಾಗಿ (int64_t r = r_ಆರಂಭ; r < r_ಕೊನೆ; r++) {
            ಸ್ಥಿರಾಂಕ int64_t hk = r / n_kv;
            ಸ್ಥಿರಾಂಕ int64_t ik = r % n_kv;
            ಸ್ಮರಣೆ_ನಕಲು(k_host + r*d_k, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ + ik*k->nb[1] + hk*k->nb[2], d_k*ಗಾತ್ರ(ggml_fp16_t));
            ಸ್ಮರಣೆ_ನಕಲು(v_host + r*d_v, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ + ik*v->nb[1] + hk*v->nb[2], d_v*ಗಾತ್ರ(ggml_fp16_t));
        }
    });

    zdnn_tensor_desc q_pre_tfm_desc, q_tfm_desc;
    zdnn_tensor_desc k_pre_tfm_desc, k_tfm_desc;
    zdnn_tensor_desc v_pre_tfm_desc, v_tfm_desc;
    zdnn_tensor_desc o_pre_tfm_desc, o_tfm_desc;
    zdnn_ztensor     q_ztensor, k_ztensor, v_ztensor, o_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> q_ztensor_alloc, k_ztensor_alloc, v_ztensor_alloc, o_ztensor_alloc;

    // a single query keeps the heads of a kv head in one stack, which is the same memory order as one head per stack
    ಆದರೆ (single) {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_ತಲೆ_kv, ಗುಂಪು, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_ತಲೆ_kv, ಗುಂಪು, d_v);
    } ಇಲ್ಲದಿದ್ದರೆ {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_ತಲೆ, n_q, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_ತಲೆ, n_q, d_v);
    }
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &k_pre_tfm_desc, n_ತಲೆ_kv, n_kv, d_k);
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &v_pre_tfm_desc, n_ತಲೆ_kv, n_kv, d_v);

    ZDNN_CHECK(zdnn_generate_transformed_desc(&q_pre_tfm_desc, &q_tfm_desc));
    ZDNN_CHECK(zdnn_generate_transformed_desc(&k_pre_tfm_desc, &k_tfm_desc));
    ZDNN_CHECK(zdnn_generate_transformed_desc(&v_pre_tfm_desc, &v_tfm_desc));
    ZDNN_CHECK(zdnn_generate_transformed_desc(&o_pre_tfm_desc, &o_tfm_desc));

    ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &q_pre_tfm_desc, &q_tfm_desc, &q_ztensor, q_ztensor_alloc);
    ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &k_pre_tfm_desc, &k_tfm_desc, &k_ztensor, k_ztensor_alloc);
    ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &v_pre_tfm_desc, &v_tfm_desc, &v_ztensor, v_ztensor_alloc);
    ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &o_pre_tfm_desc, &o_tfm_desc, &o_ztensor, o_ztensor_alloc);

    ZDNN_CHECK(zdnn_transform_ztensor(&q_ztensor, q_host));
    ZDNN_CHECK(zdnn_transform_ztensor(&k_ztensor, k_host));
    ZDNN_CHECK(zdnn_transform_ztensor(&v_ztensor, v_host));

    ggml_zdnn_pool_alloc<uint8_t> softmax_save_area(ctx->pool, ZDNN_SOFTMAX_SAVEAREA_ಗಾತ್ರ);

    zdnn_tensor_desc s_pre_tfm_desc, s_tfm_desc;
    zdnn_ztensor     s_ztensor, s2_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> s_ztensor_alloc, s2_ztensor_alloc;

    ಆದರೆ (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        zdnn_tensor_desc bias_pre_tfm_desc, bias_tfm_desc;
        zdnn_ztensor     bias_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> bias_ztensor_alloc;

        ಸ್ಥಿರಾಂಕ zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_ತಲೆ_kv, n_kv);
        ಆದರೆ (ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ) {
            ಸ್ಥಿರಾಂಕ ggml_fp16_t * ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ggml_fp16_t *) ಮುಖವಾಡ->ದತ್ತಾಂಶ;
            ಗಾಗಿ (int64_t hk = 0; hk < n_ತಲೆ_kv; hk++) {
                ಗಾಗಿ (int64_t ik = 0; ik < n_kv; ik++) {
                    ಮುಖವಾಡ_host[hk*n_kv + ik] = ggml_zdnn_fattn_ಮುಖವಾಡ_ಮೌಲ್ಯ(ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು[ik]);
                }
            }

            zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias_pre_tfm_desc, n_ತಲೆ_kv, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&bias_pre_tfm_desc, &bias_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &bias_pre_tfm_desc, &bias_tfm_desc, &bias_ztensor, bias_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&bias_ztensor, ಮುಖವಾಡ_host));
            bias = &bias_ztensor;
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, n_ತಲೆ_kv, ಗುಂಪು, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, ಸುಳ್ಳು, ನಿಜ, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_ತಲೆ_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } ಇಲ್ಲದಿದ್ದರೆ {
        zdnn_tensor_desc ಮುಖವಾಡ_pre_tfm_desc, ಮುಖವಾಡ_tfm_desc;
        zdnn_ztensor     ಮುಖವಾಡ_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> ಮುಖವಾಡ_ztensor_alloc;

        ಆದರೆ (ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ) {
            ggml_zdnn_parallel_for(n_ಎಳೆಗಳು, n_q, [&](int64_t iq_ಆರಂಭ, int64_t iq_ಕೊನೆ) {
                ಗಾಗಿ (int64_t iq = iq_ಆರಂಭ; iq < iq_ಕೊನೆ; iq++) {
                    ಸ್ಥಿರಾಂಕ ggml_fp16_t * ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ + iq*ಮುಖವಾಡ->nb[1]);
                    ಗಾಗಿ (int64_t ik = 0; ik < n_kv; ik++) {
                        ಮುಖವಾಡ_host[iq*n_kv + ik] = ggml_zdnn_fattn_ಮುಖವಾಡ_ಮೌಲ್ಯ(ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು[ik]);
                    }
                }
            });

            zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &ಮುಖವಾಡ_pre_tfm_desc, 1, n_q, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&ಮುಖವಾಡ_pre_tfm_desc, &ಮುಖವಾಡ_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &ಮುಖವಾಡ_pre_tfm_desc, &ಮುಖವಾಡ_tfm_desc, &ಮುಖವಾಡ_ztensor, ಮುಖವಾಡ_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&ಮುಖವಾಡ_ztensor, ಮುಖವಾಡ_host));
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, 1, n_q, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ಸ್ಥಿರಾಂಕ zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        ಸ್ಥಿರಾಂಕ zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        ಗಾಗಿ (int64_t h = 0; h < n_ತಲೆ; h++) {
            ಸ್ಥಿರಾಂಕ zdnn_ztensor q_view = ggml_zdnn_ರಾಶಿ_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_ತಲೆ,    h,         1);
            ಸ್ಥಿರಾಂಕ zdnn_ztensor k_view = ggml_zdnn_ರಾಶಿ_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_ತಲೆ_kv, h / ಗುಂಪು, 1);
            ಸ್ಥಿರಾಂಕ zdnn_ztensor v_view = ggml_zdnn_ರಾಶಿ_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_ತಲೆ_kv, h / ಗುಂಪು, 1);
            zdnn_ztensor       o_view = ggml_zdnn_ರಾಶಿ_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_ತಲೆ,    h,         1);

            s_ztensor.is_transformed  = ಸುಳ್ಳು;
            s2_ztensor.is_transformed = ಸುಳ್ಳು;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, ಸುಳ್ಳು, ನಿಜ, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            ಆದರೆ (ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ) {
                ZDNN_CHECK(zdnn_ಕೂಡಿಸು(&s_ztensor, &ಮುಖವಾಡ_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = ಸುಳ್ಳು;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = ನಿಜ;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    ಗಾಗಿ (int64_t iq = 0; iq < n_q; iq++) {
        ಗಾಗಿ (int64_t h = 0; h < n_ತಲೆ; h++) {
            ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*ಗಾತ್ರ(ದಶಮಾಂಶ));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_ಬಫರ್ * dst_extra = (ggml_backend_zdnn_ಬಫರ್ *) dst->extra;
    ಆದರೆ (dst_extra != ಶೂನ್ಯಸೂಚಕ) {
        dst_extra->ztensor.is_transformed = ಸುಳ್ಳು;
    }
}
