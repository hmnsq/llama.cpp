#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_zdnn_fattn_ਨਕਾਬ_ਮੁੱਲ(ggml_fp16_t m) {
    ਵਾਪਸ std::max(GGML_FP16_ਮੰਜ਼ਿਲ_FP32(m), -65504.0f);
}

ਤਾਰਕਿਕ ggml_zdnn_supports_flash_attn_ext(ਅਚਲ ggml_backend_zdnn_device_context * ctx_dev, ਅਚਲ ggml_tensor * op) {
    ਅਚਲ ggml_tensor * q    = op->src[0];
    ਅਚਲ ggml_tensor * k    = op->src[1];
    ਅਚਲ ggml_tensor * v    = op->src[2];
    ਅਚਲ ggml_tensor * ਨਕਾਬ = op->src[3];

    ਦਸ਼ਮਲਵ max_bias      = 0.0f;
    ਦਸ਼ਮਲਵ logit_softcap = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias,      (ਅਚਲ ਦਸ਼ਮਲਵ *) op->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) op->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    ਜੇ (op->src[4] != ਖਾਲੀ_ਸੂਚਕ || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ && (ਨਕਾਬ->type != GGML_TYPE_F16 || ਨਕਾਬ->ne[2] != 1 || ਨਕਾਬ->ne[3] != 1)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    ਜੇ (q->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || k->nb[0] != ਆਕਾਰ(ggml_fp16_t) || v->nb[0] != ਆਕਾਰ(ggml_fp16_t)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t max_dim = ctx_dev->max_size;
    ਅਚਲ int64_t ਸਮੂਹ   = q->ne[2] / k->ne[2];

    ਵਾਪਸ q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && ਸਮੂਹ * q->ne[1] <= max_dim;
}

ਖਾਲੀ ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * q    = dst->src[0];
    ਅਚਲ ggml_tensor * k    = dst->src[1];
    ਅਚਲ ggml_tensor * v    = dst->src[2];
    ਅਚਲ ggml_tensor * ਨਕਾਬ = dst->src[3];

    ਅਚਲ int64_t d_k       = q->ne[0];
    ਅਚਲ int64_t n_q       = q->ne[1];
    ਅਚਲ int64_t n_ਸਿਰ    = q->ne[2];
    ਅਚਲ int64_t n_kv      = k->ne[1];
    ਅਚਲ int64_t n_ਸਿਰ_kv = k->ne[2];
    ਅਚਲ int64_t d_v       = v->ne[0];
    ਅਚਲ int64_t ਸਮੂਹ     = n_ਸਿਰ / n_ਸਿਰ_kv;

    ਦਸ਼ਮਲਵ scale = 1.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ int32_t n_ਧਾਗੇ = n_q*n_kv >= 65536 ? ctx->n_ਧਾਗੇ : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    ਅਚਲ ਤਾਰਕਿਕ single = n_q == 1;

    ਅਚਲ int64_t n_ਨਕਾਬ = ਨਕਾਬ ? (single ? n_ਸਿਰ_kv*n_kv : n_q*n_kv) : 0;

    ggml_zdnn_pool_alloc<ਦਸ਼ਮਲਵ>       q_host_alloc(ctx->pool, n_ਸਿਰ*n_q*d_k);
    ggml_zdnn_pool_alloc<ਦਸ਼ਮਲਵ>       o_host_alloc(ctx->pool, n_ਸਿਰ*n_q*d_v);
    ggml_zdnn_pool_alloc<ਦਸ਼ਮਲਵ>       ਨਕਾਬ_host_alloc(ctx->pool, n_ਨਕਾਬ);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_ਸਿਰ_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_ਸਿਰ_kv*n_kv*d_v);

    ਦਸ਼ਮਲਵ       * q_host    = q_host_alloc.get();
    ਦਸ਼ਮਲਵ       * o_host    = o_host_alloc.get();
    ਦਸ਼ਮਲਵ       * ਨਕਾਬ_host = ਨਕਾਬ_host_alloc.get();
    ggml_fp16_t * k_host    = k_host_alloc.get();
    ggml_fp16_t * v_host    = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_ਧਾਗੇ, n_ਸਿਰ, [&](int64_t h_ਸ਼ੁਰੂ, int64_t h_ਸਿਰਾ) {
        ਲਈ (int64_t h = h_ਸ਼ੁਰੂ; h < h_ਸਿਰਾ; h++) {
            ਲਈ (int64_t iq = 0; iq < n_q; iq++) {
                ਅਚਲ ਦਸ਼ਮਲਵ * src = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *) q->ਅੰਕੜੇ + iq*q->nb[1] + h*q->nb[2]);
                ਦਸ਼ਮਲਵ       * ਪੰਕਤੀ = q_host + (h*n_q + iq)*d_k;
                ਲਈ (int64_t i0 = 0; i0 < d_k; i0++) {
                    ਪੰਕਤੀ[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_ਧਾਗੇ, n_ਸਿਰ_kv*n_kv, [&](int64_t r_ਸ਼ੁਰੂ, int64_t r_ਸਿਰਾ) {
        ਲਈ (int64_t r = r_ਸ਼ੁਰੂ; r < r_ਸਿਰਾ; r++) {
            ਅਚਲ int64_t hk = r / n_kv;
            ਅਚਲ int64_t ik = r % n_kv;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(k_host + r*d_k, (ਅਚਲ ਅੱਖਰ *) k->ਅੰਕੜੇ + ik*k->nb[1] + hk*k->nb[2], d_k*ਆਕਾਰ(ggml_fp16_t));
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(v_host + r*d_v, (ਅਚਲ ਅੱਖਰ *) v->ਅੰਕੜੇ + ik*v->nb[1] + hk*v->nb[2], d_v*ਆਕਾਰ(ggml_fp16_t));
        }
    });

    zdnn_tensor_desc q_pre_tfm_desc, q_tfm_desc;
    zdnn_tensor_desc k_pre_tfm_desc, k_tfm_desc;
    zdnn_tensor_desc v_pre_tfm_desc, v_tfm_desc;
    zdnn_tensor_desc o_pre_tfm_desc, o_tfm_desc;
    zdnn_ztensor     q_ztensor, k_ztensor, v_ztensor, o_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> q_ztensor_alloc, k_ztensor_alloc, v_ztensor_alloc, o_ztensor_alloc;

    // a single query keeps the heads of a kv head in one stack, which is the same memory order as one head per stack
    ਜੇ (single) {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_ਸਿਰ_kv, ਸਮੂਹ, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_ਸਿਰ_kv, ਸਮੂਹ, d_v);
    } ਨਹੀਂਤਾਂ {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_ਸਿਰ, n_q, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_ਸਿਰ, n_q, d_v);
    }
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &k_pre_tfm_desc, n_ਸਿਰ_kv, n_kv, d_k);
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &v_pre_tfm_desc, n_ਸਿਰ_kv, n_kv, d_v);

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

    ggml_zdnn_pool_alloc<uint8_t> softmax_save_area(ctx->pool, ZDNN_SOFTMAX_SAVEAREA_ਮਾਪ);

    zdnn_tensor_desc s_pre_tfm_desc, s_tfm_desc;
    zdnn_ztensor     s_ztensor, s2_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> s_ztensor_alloc, s2_ztensor_alloc;

    ਜੇ (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        zdnn_tensor_desc bias_pre_tfm_desc, bias_tfm_desc;
        zdnn_ztensor     bias_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> bias_ztensor_alloc;

        ਅਚਲ zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_ਸਿਰ_kv, n_kv);
        ਜੇ (ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ) {
            ਅਚਲ ggml_fp16_t * ਨਕਾਬ_ਪੰਕਤੀ = (ਅਚਲ ggml_fp16_t *) ਨਕਾਬ->ਅੰਕੜੇ;
            ਲਈ (int64_t hk = 0; hk < n_ਸਿਰ_kv; hk++) {
                ਲਈ (int64_t ik = 0; ik < n_kv; ik++) {
                    ਨਕਾਬ_host[hk*n_kv + ik] = ggml_zdnn_fattn_ਨਕਾਬ_ਮੁੱਲ(ਨਕਾਬ_ਪੰਕਤੀ[ik]);
                }
            }

            zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias_pre_tfm_desc, n_ਸਿਰ_kv, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&bias_pre_tfm_desc, &bias_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &bias_pre_tfm_desc, &bias_tfm_desc, &bias_ztensor, bias_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&bias_ztensor, ਨਕਾਬ_host));
            bias = &bias_ztensor;
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, n_ਸਿਰ_kv, ਸਮੂਹ, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, ਝੂਠ, ਸੱਚ, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_ਸਿਰ_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } ਨਹੀਂਤਾਂ {
        zdnn_tensor_desc ਨਕਾਬ_pre_tfm_desc, ਨਕਾਬ_tfm_desc;
        zdnn_ztensor     ਨਕਾਬ_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> ਨਕਾਬ_ztensor_alloc;

        ਜੇ (ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ) {
            ggml_zdnn_parallel_for(n_ਧਾਗੇ, n_q, [&](int64_t iq_ਸ਼ੁਰੂ, int64_t iq_ਸਿਰਾ) {
                ਲਈ (int64_t iq = iq_ਸ਼ੁਰੂ; iq < iq_ਸਿਰਾ; iq++) {
                    ਅਚਲ ggml_fp16_t * ਨਕਾਬ_ਪੰਕਤੀ = (ਅਚਲ ggml_fp16_t *)((ਅਚਲ ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ + iq*ਨਕਾਬ->nb[1]);
                    ਲਈ (int64_t ik = 0; ik < n_kv; ik++) {
                        ਨਕਾਬ_host[iq*n_kv + ik] = ggml_zdnn_fattn_ਨਕਾਬ_ਮੁੱਲ(ਨਕਾਬ_ਪੰਕਤੀ[ik]);
                    }
                }
            });

            zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &ਨਕਾਬ_pre_tfm_desc, 1, n_q, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&ਨਕਾਬ_pre_tfm_desc, &ਨਕਾਬ_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &ਨਕਾਬ_pre_tfm_desc, &ਨਕਾਬ_tfm_desc, &ਨਕਾਬ_ztensor, ਨਕਾਬ_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&ਨਕਾਬ_ztensor, ਨਕਾਬ_host));
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, 1, n_q, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ਅਚਲ zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        ਅਚਲ zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        ਲਈ (int64_t h = 0; h < n_ਸਿਰ; h++) {
            ਅਚਲ zdnn_ztensor q_view = ggml_zdnn_ਢੇਰ_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_ਸਿਰ,    h,         1);
            ਅਚਲ zdnn_ztensor k_view = ggml_zdnn_ਢੇਰ_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_ਸਿਰ_kv, h / ਸਮੂਹ, 1);
            ਅਚਲ zdnn_ztensor v_view = ggml_zdnn_ਢੇਰ_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_ਸਿਰ_kv, h / ਸਮੂਹ, 1);
            zdnn_ztensor       o_view = ggml_zdnn_ਢੇਰ_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_ਸਿਰ,    h,         1);

            s_ztensor.is_transformed  = ਝੂਠ;
            s2_ztensor.is_transformed = ਝੂਠ;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, ਝੂਠ, ਸੱਚ, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            ਜੇ (ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ) {
                ZDNN_CHECK(zdnn_ਜੋੜੋ(&s_ztensor, &ਨਕਾਬ_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = ਝੂਠ;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = ਸੱਚ;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    ਲਈ (int64_t iq = 0; iq < n_q; iq++) {
        ਲਈ (int64_t h = 0; h < n_ਸਿਰ; h++) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) dst->ਅੰਕੜੇ + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_ਬਫਰ * dst_extra = (ggml_backend_zdnn_ਬਫਰ *) dst->extra;
    ਜੇ (dst_extra != ਖਾਲੀ_ਸੂਚਕ) {
        dst_extra->ztensor.is_transformed = ਝੂਠ;
    }
}
