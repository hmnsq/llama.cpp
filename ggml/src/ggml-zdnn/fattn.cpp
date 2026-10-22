#شامل "ggml.h"
#شامل "fattn.hpp"

#شامل <algorithm>
#شامل <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
ساکن اندرونی اعشاری ggml_zdnn_fattn_نقاب_قدر(ggml_fp16_t m) {
    واپس std::max(GGML_FP16_مقصد_FP32(m), -65504.0f);
}

منطقی ggml_zdnn_supports_flash_attn_ext(ثابت ggml_backend_zdnn_device_context * ctx_dev, ثابت ggml_tensor * op) {
    ثابت ggml_tensor * q    = op->src[0];
    ثابت ggml_tensor * k    = op->src[1];
    ثابت ggml_tensor * v    = op->src[2];
    ثابت ggml_tensor * نقاب = op->src[3];

    اعشاری max_bias      = 0.0f;
    اعشاری logit_softcap = 0.0f;
    حافظہ_نقل(&max_bias,      (ثابت اعشاری *) op->op_params + 1, حجم(اعشاری));
    حافظہ_نقل(&logit_softcap, (ثابت اعشاری *) op->op_params + 2, حجم(اعشاری));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    اگر (op->src[4] != خالی_اشاریہ || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        واپس جھوٹ;
    }

    اگر (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        واپس جھوٹ;
    }

    اگر (نقاب != خالی_اشاریہ && (نقاب->type != GGML_TYPE_F16 || نقاب->ne[2] != 1 || نقاب->ne[3] != 1)) {
        واپس جھوٹ;
    }

    اگر (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        واپس جھوٹ;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    اگر (q->nb[0] != حجم(اعشاری) || k->nb[0] != حجم(ggml_fp16_t) || v->nb[0] != حجم(ggml_fp16_t)) {
        واپس جھوٹ;
    }

    ثابت int64_t max_dim = ctx_dev->max_size;
    ثابت int64_t گروہ   = q->ne[2] / k->ne[2];

    واپس q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && گروہ * q->ne[1] <= max_dim;
}

خالی ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * q    = dst->src[0];
    ثابت ggml_tensor * k    = dst->src[1];
    ثابت ggml_tensor * v    = dst->src[2];
    ثابت ggml_tensor * نقاب = dst->src[3];

    ثابت int64_t d_k       = q->ne[0];
    ثابت int64_t n_q       = q->ne[1];
    ثابت int64_t n_سر    = q->ne[2];
    ثابت int64_t n_kv      = k->ne[1];
    ثابت int64_t n_سر_kv = k->ne[2];
    ثابت int64_t d_v       = v->ne[0];
    ثابت int64_t گروہ     = n_سر / n_سر_kv;

    اعشاری scale = 1.0f;
    حافظہ_نقل(&scale, (ثابت اعشاری *) dst->op_params + 0, حجم(اعشاری));

    ثابت int32_t n_دھاگے = n_q*n_kv >= 65536 ? ctx->n_دھاگے : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    ثابت منطقی single = n_q == 1;

    ثابت int64_t n_نقاب = نقاب ? (single ? n_سر_kv*n_kv : n_q*n_kv) : 0;

    ggml_zdnn_pool_alloc<اعشاری>       q_host_alloc(ctx->pool, n_سر*n_q*d_k);
    ggml_zdnn_pool_alloc<اعشاری>       o_host_alloc(ctx->pool, n_سر*n_q*d_v);
    ggml_zdnn_pool_alloc<اعشاری>       نقاب_host_alloc(ctx->pool, n_نقاب);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_سر_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_سر_kv*n_kv*d_v);

    اعشاری       * q_host    = q_host_alloc.get();
    اعشاری       * o_host    = o_host_alloc.get();
    اعشاری       * نقاب_host = نقاب_host_alloc.get();
    ggml_fp16_t * k_host    = k_host_alloc.get();
    ggml_fp16_t * v_host    = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_دھاگے, n_سر, [&](int64_t h_آغاز, int64_t h_انتہا) {
        برائے (int64_t h = h_آغاز; h < h_انتہا; h++) {
            برائے (int64_t iq = 0; iq < n_q; iq++) {
                ثابت اعشاری * src = (ثابت اعشاری *)((ثابت حرف *) q->ڈیٹا + iq*q->nb[1] + h*q->nb[2]);
                اعشاری       * صف = q_host + (h*n_q + iq)*d_k;
                برائے (int64_t i0 = 0; i0 < d_k; i0++) {
                    صف[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_دھاگے, n_سر_kv*n_kv, [&](int64_t r_آغاز, int64_t r_انتہا) {
        برائے (int64_t r = r_آغاز; r < r_انتہا; r++) {
            ثابت int64_t hk = r / n_kv;
            ثابت int64_t ik = r % n_kv;
            حافظہ_نقل(k_host + r*d_k, (ثابت حرف *) k->ڈیٹا + ik*k->nb[1] + hk*k->nb[2], d_k*حجم(ggml_fp16_t));
            حافظہ_نقل(v_host + r*d_v, (ثابت حرف *) v->ڈیٹا + ik*v->nb[1] + hk*v->nb[2], d_v*حجم(ggml_fp16_t));
        }
    });

    zdnn_tensor_desc q_pre_tfm_desc, q_tfm_desc;
    zdnn_tensor_desc k_pre_tfm_desc, k_tfm_desc;
    zdnn_tensor_desc v_pre_tfm_desc, v_tfm_desc;
    zdnn_tensor_desc o_pre_tfm_desc, o_tfm_desc;
    zdnn_ztensor     q_ztensor, k_ztensor, v_ztensor, o_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> q_ztensor_alloc, k_ztensor_alloc, v_ztensor_alloc, o_ztensor_alloc;

    // a single query keeps the heads of a kv head in one stack, which is the same memory order as one head per stack
    اگر (single) {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_سر_kv, گروہ, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_سر_kv, گروہ, d_v);
    } ورنہ {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_سر, n_q, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_سر, n_q, d_v);
    }
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &k_pre_tfm_desc, n_سر_kv, n_kv, d_k);
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &v_pre_tfm_desc, n_سر_kv, n_kv, d_v);

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

    ggml_zdnn_pool_alloc<uint8_t> softmax_save_area(ctx->pool, ZDNN_SOFTMAX_SAVEAREA_سائز);

    zdnn_tensor_desc s_pre_tfm_desc, s_tfm_desc;
    zdnn_ztensor     s_ztensor, s2_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> s_ztensor_alloc, s2_ztensor_alloc;

    اگر (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        zdnn_tensor_desc bias_pre_tfm_desc, bias_tfm_desc;
        zdnn_ztensor     bias_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> bias_ztensor_alloc;

        ثابت zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_سر_kv, n_kv);
        اگر (نقاب != خالی_اشاریہ) {
            ثابت ggml_fp16_t * نقاب_صف = (ثابت ggml_fp16_t *) نقاب->ڈیٹا;
            برائے (int64_t hk = 0; hk < n_سر_kv; hk++) {
                برائے (int64_t ik = 0; ik < n_kv; ik++) {
                    نقاب_host[hk*n_kv + ik] = ggml_zdnn_fattn_نقاب_قدر(نقاب_صف[ik]);
                }
            }

            zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias_pre_tfm_desc, n_سر_kv, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&bias_pre_tfm_desc, &bias_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &bias_pre_tfm_desc, &bias_tfm_desc, &bias_ztensor, bias_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&bias_ztensor, نقاب_host));
            bias = &bias_ztensor;
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, n_سر_kv, گروہ, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, جھوٹ, سچ, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_سر_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } ورنہ {
        zdnn_tensor_desc نقاب_pre_tfm_desc, نقاب_tfm_desc;
        zdnn_ztensor     نقاب_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> نقاب_ztensor_alloc;

        اگر (نقاب != خالی_اشاریہ) {
            ggml_zdnn_parallel_for(n_دھاگے, n_q, [&](int64_t iq_آغاز, int64_t iq_انتہا) {
                برائے (int64_t iq = iq_آغاز; iq < iq_انتہا; iq++) {
                    ثابت ggml_fp16_t * نقاب_صف = (ثابت ggml_fp16_t *)((ثابت حرف *) نقاب->ڈیٹا + iq*نقاب->nb[1]);
                    برائے (int64_t ik = 0; ik < n_kv; ik++) {
                        نقاب_host[iq*n_kv + ik] = ggml_zdnn_fattn_نقاب_قدر(نقاب_صف[ik]);
                    }
                }
            });

            zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &نقاب_pre_tfm_desc, 1, n_q, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&نقاب_pre_tfm_desc, &نقاب_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &نقاب_pre_tfm_desc, &نقاب_tfm_desc, &نقاب_ztensor, نقاب_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&نقاب_ztensor, نقاب_host));
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, 1, n_q, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ثابت zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        ثابت zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        برائے (int64_t h = 0; h < n_سر; h++) {
            ثابت zdnn_ztensor q_view = ggml_zdnn_ڈھیر_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_سر,    h,         1);
            ثابت zdnn_ztensor k_view = ggml_zdnn_ڈھیر_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_سر_kv, h / گروہ, 1);
            ثابت zdnn_ztensor v_view = ggml_zdnn_ڈھیر_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_سر_kv, h / گروہ, 1);
            zdnn_ztensor       o_view = ggml_zdnn_ڈھیر_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_سر,    h,         1);

            s_ztensor.is_transformed  = جھوٹ;
            s2_ztensor.is_transformed = جھوٹ;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, جھوٹ, سچ, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            اگر (نقاب != خالی_اشاریہ) {
                ZDNN_CHECK(zdnn_جوڑو(&s_ztensor, &نقاب_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = جھوٹ;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = سچ;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    برائے (int64_t iq = 0; iq < n_q; iq++) {
        برائے (int64_t h = 0; h < n_سر; h++) {
            حافظہ_نقل((حرف *) dst->ڈیٹا + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*حجم(اعشاری));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_بفر * dst_extra = (ggml_backend_zdnn_بفر *) dst->extra;
    اگر (dst_extra != خالی_اشاریہ) {
        dst_extra->ztensor.is_transformed = جھوٹ;
    }
}
