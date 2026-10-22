#ضمن "ggml.h"
#ضمن "fattn.hpp"

#ضمن <algorithm>
#ضمن <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
ساكن مضمن عائم ggml_zdnn_fattn_ماسک_ارزښت(ggml_fp16_t m) {
    أرجع std::max(GGML_FP16_منزل_FP32(m), -65504.0f);
}

منطقي ggml_zdnn_supports_flash_attn_ext(ثابت ggml_backend_zdnn_device_context * ctx_dev, ثابت ggml_tensor * op) {
    ثابت ggml_tensor * q    = op->src[0];
    ثابت ggml_tensor * k    = op->src[1];
    ثابت ggml_tensor * v    = op->src[2];
    ثابت ggml_tensor * ماسک = op->src[3];

    عائم max_bias      = 0.0f;
    عائم logit_softcap = 0.0f;
    حافظه_نقل(&max_bias,      (ثابت عائم *) op->op_params + 1, حجم(عائم));
    حافظه_نقل(&logit_softcap, (ثابت عائم *) op->op_params + 2, حجم(عائم));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    إذا (op->src[4] != تش_اشاره || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        أرجع خطأ;
    }

    إذا (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }

    إذا (ماسک != تش_اشاره && (ماسک->type != GGML_TYPE_F16 || ماسک->ne[2] != 1 || ماسک->ne[3] != 1)) {
        أرجع خطأ;
    }

    إذا (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        أرجع خطأ;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    إذا (q->nb[0] != حجم(عائم) || k->nb[0] != حجم(ggml_fp16_t) || v->nb[0] != حجم(ggml_fp16_t)) {
        أرجع خطأ;
    }

    ثابت int64_t max_dim = ctx_dev->max_size;
    ثابت int64_t ډله   = q->ne[2] / k->ne[2];

    أرجع q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && ډله * q->ne[1] <= max_dim;
}

فراغ ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * q    = dst->src[0];
    ثابت ggml_tensor * k    = dst->src[1];
    ثابت ggml_tensor * v    = dst->src[2];
    ثابت ggml_tensor * ماسک = dst->src[3];

    ثابت int64_t d_k       = q->ne[0];
    ثابت int64_t n_q       = q->ne[1];
    ثابت int64_t n_سر    = q->ne[2];
    ثابت int64_t n_kv      = k->ne[1];
    ثابت int64_t n_سر_kv = k->ne[2];
    ثابت int64_t d_v       = v->ne[0];
    ثابت int64_t ډله     = n_سر / n_سر_kv;

    عائم scale = 1.0f;
    حافظه_نقل(&scale, (ثابت عائم *) dst->op_params + 0, حجم(عائم));

    ثابت int32_t n_تارونه = n_q*n_kv >= 65536 ? ctx->n_تارونه : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    ثابت منطقي single = n_q == 1;

    ثابت int64_t n_ماسک = ماسک ? (single ? n_سر_kv*n_kv : n_q*n_kv) : 0;

    ggml_zdnn_pool_alloc<عائم>       q_host_alloc(ctx->pool, n_سر*n_q*d_k);
    ggml_zdnn_pool_alloc<عائم>       o_host_alloc(ctx->pool, n_سر*n_q*d_v);
    ggml_zdnn_pool_alloc<عائم>       ماسک_host_alloc(ctx->pool, n_ماسک);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_سر_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_سر_kv*n_kv*d_v);

    عائم       * q_host    = q_host_alloc.get();
    عائم       * o_host    = o_host_alloc.get();
    عائم       * ماسک_host = ماسک_host_alloc.get();
    ggml_fp16_t * k_host    = k_host_alloc.get();
    ggml_fp16_t * v_host    = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_تارونه, n_سر, [&](int64_t h_پیل, int64_t h_پای_ټکی) {
        لكل (int64_t h = h_پیل; h < h_پای_ټکی; h++) {
            لكل (int64_t iq = 0; iq < n_q; iq++) {
                ثابت عائم * src = (ثابت عائم *)((ثابت حرف *) q->معلومات + iq*q->nb[1] + h*q->nb[2]);
                عائم       * لیکه = q_host + (h*n_q + iq)*d_k;
                لكل (int64_t i0 = 0; i0 < d_k; i0++) {
                    لیکه[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_تارونه, n_سر_kv*n_kv, [&](int64_t r_پیل, int64_t r_پای_ټکی) {
        لكل (int64_t r = r_پیل; r < r_پای_ټکی; r++) {
            ثابت int64_t hk = r / n_kv;
            ثابت int64_t ik = r % n_kv;
            حافظه_نقل(k_host + r*d_k, (ثابت حرف *) k->معلومات + ik*k->nb[1] + hk*k->nb[2], d_k*حجم(ggml_fp16_t));
            حافظه_نقل(v_host + r*d_v, (ثابت حرف *) v->معلومات + ik*v->nb[1] + hk*v->nb[2], d_v*حجم(ggml_fp16_t));
        }
    });

    zdnn_tensor_desc q_pre_tfm_desc, q_tfm_desc;
    zdnn_tensor_desc k_pre_tfm_desc, k_tfm_desc;
    zdnn_tensor_desc v_pre_tfm_desc, v_tfm_desc;
    zdnn_tensor_desc o_pre_tfm_desc, o_tfm_desc;
    zdnn_ztensor     q_ztensor, k_ztensor, v_ztensor, o_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> q_ztensor_alloc, k_ztensor_alloc, v_ztensor_alloc, o_ztensor_alloc;

    // a single query keeps the heads of a kv head in one stack, which is the same memory order as one head per stack
    إذا (single) {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_سر_kv, ډله, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_سر_kv, ډله, d_v);
    } وإلا {
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

    ggml_zdnn_pool_alloc<uint8_t> softmax_save_area(ctx->pool, ZDNN_SOFTMAX_SAVEAREA_اندازه);

    zdnn_tensor_desc s_pre_tfm_desc, s_tfm_desc;
    zdnn_ztensor     s_ztensor, s2_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> s_ztensor_alloc, s2_ztensor_alloc;

    إذا (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        zdnn_tensor_desc bias_pre_tfm_desc, bias_tfm_desc;
        zdnn_ztensor     bias_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> bias_ztensor_alloc;

        ثابت zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_سر_kv, n_kv);
        إذا (ماسک != تش_اشاره) {
            ثابت ggml_fp16_t * ماسک_لیکه = (ثابت ggml_fp16_t *) ماسک->معلومات;
            لكل (int64_t hk = 0; hk < n_سر_kv; hk++) {
                لكل (int64_t ik = 0; ik < n_kv; ik++) {
                    ماسک_host[hk*n_kv + ik] = ggml_zdnn_fattn_ماسک_ارزښت(ماسک_لیکه[ik]);
                }
            }

            zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias_pre_tfm_desc, n_سر_kv, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&bias_pre_tfm_desc, &bias_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &bias_pre_tfm_desc, &bias_tfm_desc, &bias_ztensor, bias_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&bias_ztensor, ماسک_host));
            bias = &bias_ztensor;
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, n_سر_kv, ډله, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, خطأ, ریښتیا, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_سر_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } وإلا {
        zdnn_tensor_desc ماسک_pre_tfm_desc, ماسک_tfm_desc;
        zdnn_ztensor     ماسک_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> ماسک_ztensor_alloc;

        إذا (ماسک != تش_اشاره) {
            ggml_zdnn_parallel_for(n_تارونه, n_q, [&](int64_t iq_پیل, int64_t iq_پای_ټکی) {
                لكل (int64_t iq = iq_پیل; iq < iq_پای_ټکی; iq++) {
                    ثابت ggml_fp16_t * ماسک_لیکه = (ثابت ggml_fp16_t *)((ثابت حرف *) ماسک->معلومات + iq*ماسک->nb[1]);
                    لكل (int64_t ik = 0; ik < n_kv; ik++) {
                        ماسک_host[iq*n_kv + ik] = ggml_zdnn_fattn_ماسک_ارزښت(ماسک_لیکه[ik]);
                    }
                }
            });

            zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &ماسک_pre_tfm_desc, 1, n_q, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&ماسک_pre_tfm_desc, &ماسک_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &ماسک_pre_tfm_desc, &ماسک_tfm_desc, &ماسک_ztensor, ماسک_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&ماسک_ztensor, ماسک_host));
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, 1, n_q, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ثابت zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        ثابت zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        لكل (int64_t h = 0; h < n_سر; h++) {
            ثابت zdnn_ztensor q_view = ggml_zdnn_ډېری_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_سر,    h,         1);
            ثابت zdnn_ztensor k_view = ggml_zdnn_ډېری_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_سر_kv, h / ډله, 1);
            ثابت zdnn_ztensor v_view = ggml_zdnn_ډېری_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_سر_kv, h / ډله, 1);
            zdnn_ztensor       o_view = ggml_zdnn_ډېری_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_سر,    h,         1);

            s_ztensor.is_transformed  = خطأ;
            s2_ztensor.is_transformed = خطأ;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, خطأ, ریښتیا, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            إذا (ماسک != تش_اشاره) {
                ZDNN_CHECK(zdnn_جمع_کړه(&s_ztensor, &ماسک_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = خطأ;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = ریښتیا;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    لكل (int64_t iq = 0; iq < n_q; iq++) {
        لكل (int64_t h = 0; h < n_سر; h++) {
            حافظه_نقل((حرف *) dst->معلومات + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*حجم(عائم));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_بفر * dst_extra = (ggml_backend_zdnn_بفر *) dst->extra;
    إذا (dst_extra != تش_اشاره) {
        dst_extra->ztensor.is_transformed = خطأ;
    }
}
