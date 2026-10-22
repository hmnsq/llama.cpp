#అన్తర్భావ "ggml.h"
#అన్తర్భావ "fattn.hpp"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
స్థిర అంతఃవరుస దశాంశం ggml_zdnn_fattn_ముసుగు_విలువ(ggml_fp16_t m) {
    తిరిగి std::max(GGML_FP16_గమ్యం_FP32(m), -65504.0f);
}

తార్కిక ggml_zdnn_supports_flash_attn_ext(స్థిరరాశి ggml_backend_zdnn_device_context * ctx_dev, స్థిరరాశి ggml_tensor * op) {
    స్థిరరాశి ggml_tensor * q    = op->src[0];
    స్థిరరాశి ggml_tensor * k    = op->src[1];
    స్థిరరాశి ggml_tensor * v    = op->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు = op->src[3];

    దశాంశం max_bias      = 0.0f;
    దశాంశం logit_softcap = 0.0f;
    జ్ఞాపకం_నకలు(&max_bias,      (స్థిరరాశి దశాంశం *) op->op_params + 1, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) op->op_params + 2, పరిమాణం(దశాంశం));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    ఒకవేళ (op->src[4] != శూన్యసూచిక || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (ముసుగు != శూన్యసూచిక && (ముసుగు->type != GGML_TYPE_F16 || ముసుగు->ne[2] != 1 || ముసుగు->ne[3] != 1)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        తిరిగి అబద్ధం;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    ఒకవేళ (q->nb[0] != పరిమాణం(దశాంశం) || k->nb[0] != పరిమాణం(ggml_fp16_t) || v->nb[0] != పరిమాణం(ggml_fp16_t)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t max_dim = ctx_dev->max_size;
    స్థిరరాశి int64_t సమూహం   = q->ne[2] / k->ne[2];

    తిరిగి q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && సమూహం * q->ne[1] <= max_dim;
}

శూన్య ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * q    = dst->src[0];
    స్థిరరాశి ggml_tensor * k    = dst->src[1];
    స్థిరరాశి ggml_tensor * v    = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు = dst->src[3];

    స్థిరరాశి int64_t d_k       = q->ne[0];
    స్థిరరాశి int64_t n_q       = q->ne[1];
    స్థిరరాశి int64_t n_తల    = q->ne[2];
    స్థిరరాశి int64_t n_kv      = k->ne[1];
    స్థిరరాశి int64_t n_తల_kv = k->ne[2];
    స్థిరరాశి int64_t d_v       = v->ne[0];
    స్థిరరాశి int64_t సమూహం     = n_తల / n_తల_kv;

    దశాంశం scale = 1.0f;
    జ్ఞాపకం_నకలు(&scale, (స్థిరరాశి దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));

    స్థిరరాశి int32_t n_దారాలు = n_q*n_kv >= 65536 ? ctx->n_దారాలు : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    స్థిరరాశి తార్కిక single = n_q == 1;

    స్థిరరాశి int64_t n_ముసుగు = ముసుగు ? (single ? n_తల_kv*n_kv : n_q*n_kv) : 0;

    ggml_zdnn_pool_alloc<దశాంశం>       q_host_alloc(ctx->pool, n_తల*n_q*d_k);
    ggml_zdnn_pool_alloc<దశాంశం>       o_host_alloc(ctx->pool, n_తల*n_q*d_v);
    ggml_zdnn_pool_alloc<దశాంశం>       ముసుగు_host_alloc(ctx->pool, n_ముసుగు);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_తల_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_తల_kv*n_kv*d_v);

    దశాంశం       * q_host    = q_host_alloc.get();
    దశాంశం       * o_host    = o_host_alloc.get();
    దశాంశం       * ముసుగు_host = ముసుగు_host_alloc.get();
    ggml_fp16_t * k_host    = k_host_alloc.get();
    ggml_fp16_t * v_host    = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_దారాలు, n_తల, [&](int64_t h_ప్రారంభం, int64_t h_చివర) {
        కొరకు (int64_t h = h_ప్రారంభం; h < h_చివర; h++) {
            కొరకు (int64_t iq = 0; iq < n_q; iq++) {
                స్థిరరాశి దశాంశం * src = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *) q->దత్తాంశం + iq*q->nb[1] + h*q->nb[2]);
                దశాంశం       * అడ్డువరుస = q_host + (h*n_q + iq)*d_k;
                కొరకు (int64_t i0 = 0; i0 < d_k; i0++) {
                    అడ్డువరుస[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_దారాలు, n_తల_kv*n_kv, [&](int64_t r_ప్రారంభం, int64_t r_చివర) {
        కొరకు (int64_t r = r_ప్రారంభం; r < r_చివర; r++) {
            స్థిరరాశి int64_t hk = r / n_kv;
            స్థిరరాశి int64_t ik = r % n_kv;
            జ్ఞాపకం_నకలు(k_host + r*d_k, (స్థిరరాశి అక్షరం *) k->దత్తాంశం + ik*k->nb[1] + hk*k->nb[2], d_k*పరిమాణం(ggml_fp16_t));
            జ్ఞాపకం_నకలు(v_host + r*d_v, (స్థిరరాశి అక్షరం *) v->దత్తాంశం + ik*v->nb[1] + hk*v->nb[2], d_v*పరిమాణం(ggml_fp16_t));
        }
    });

    zdnn_tensor_desc q_pre_tfm_desc, q_tfm_desc;
    zdnn_tensor_desc k_pre_tfm_desc, k_tfm_desc;
    zdnn_tensor_desc v_pre_tfm_desc, v_tfm_desc;
    zdnn_tensor_desc o_pre_tfm_desc, o_tfm_desc;
    zdnn_ztensor     q_ztensor, k_ztensor, v_ztensor, o_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> q_ztensor_alloc, k_ztensor_alloc, v_ztensor_alloc, o_ztensor_alloc;

    // a single query keeps the heads of a kv head in one stack, which is the same memory order as one head per stack
    ఒకవేళ (single) {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_తల_kv, సమూహం, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_తల_kv, సమూహం, d_v);
    } లేకపోతే {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_తల, n_q, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_తల, n_q, d_v);
    }
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &k_pre_tfm_desc, n_తల_kv, n_kv, d_k);
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &v_pre_tfm_desc, n_తల_kv, n_kv, d_v);

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

    ggml_zdnn_pool_alloc<uint8_t> softmax_save_area(ctx->pool, ZDNN_SOFTMAX_SAVEAREA_కొలత);

    zdnn_tensor_desc s_pre_tfm_desc, s_tfm_desc;
    zdnn_ztensor     s_ztensor, s2_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> s_ztensor_alloc, s2_ztensor_alloc;

    ఒకవేళ (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        zdnn_tensor_desc bias_pre_tfm_desc, bias_tfm_desc;
        zdnn_ztensor     bias_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> bias_ztensor_alloc;

        స్థిరరాశి zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_తల_kv, n_kv);
        ఒకవేళ (ముసుగు != శూన్యసూచిక) {
            స్థిరరాశి ggml_fp16_t * ముసుగు_అడ్డువరుస = (స్థిరరాశి ggml_fp16_t *) ముసుగు->దత్తాంశం;
            కొరకు (int64_t hk = 0; hk < n_తల_kv; hk++) {
                కొరకు (int64_t ik = 0; ik < n_kv; ik++) {
                    ముసుగు_host[hk*n_kv + ik] = ggml_zdnn_fattn_ముసుగు_విలువ(ముసుగు_అడ్డువరుస[ik]);
                }
            }

            zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias_pre_tfm_desc, n_తల_kv, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&bias_pre_tfm_desc, &bias_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &bias_pre_tfm_desc, &bias_tfm_desc, &bias_ztensor, bias_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&bias_ztensor, ముసుగు_host));
            bias = &bias_ztensor;
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, n_తల_kv, సమూహం, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, అబద్ధం, నిజం, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_తల_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } లేకపోతే {
        zdnn_tensor_desc ముసుగు_pre_tfm_desc, ముసుగు_tfm_desc;
        zdnn_ztensor     ముసుగు_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> ముసుగు_ztensor_alloc;

        ఒకవేళ (ముసుగు != శూన్యసూచిక) {
            ggml_zdnn_parallel_for(n_దారాలు, n_q, [&](int64_t iq_ప్రారంభం, int64_t iq_చివర) {
                కొరకు (int64_t iq = iq_ప్రారంభం; iq < iq_చివర; iq++) {
                    స్థిరరాశి ggml_fp16_t * ముసుగు_అడ్డువరుస = (స్థిరరాశి ggml_fp16_t *)((స్థిరరాశి అక్షరం *) ముసుగు->దత్తాంశం + iq*ముసుగు->nb[1]);
                    కొరకు (int64_t ik = 0; ik < n_kv; ik++) {
                        ముసుగు_host[iq*n_kv + ik] = ggml_zdnn_fattn_ముసుగు_విలువ(ముసుగు_అడ్డువరుస[ik]);
                    }
                }
            });

            zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &ముసుగు_pre_tfm_desc, 1, n_q, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&ముసుగు_pre_tfm_desc, &ముసుగు_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &ముసుగు_pre_tfm_desc, &ముసుగు_tfm_desc, &ముసుగు_ztensor, ముసుగు_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&ముసుగు_ztensor, ముసుగు_host));
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, 1, n_q, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        స్థిరరాశి zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        స్థిరరాశి zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        కొరకు (int64_t h = 0; h < n_తల; h++) {
            స్థిరరాశి zdnn_ztensor q_view = ggml_zdnn_దొంతర_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_తల,    h,         1);
            స్థిరరాశి zdnn_ztensor k_view = ggml_zdnn_దొంతర_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_తల_kv, h / సమూహం, 1);
            స్థిరరాశి zdnn_ztensor v_view = ggml_zdnn_దొంతర_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_తల_kv, h / సమూహం, 1);
            zdnn_ztensor       o_view = ggml_zdnn_దొంతర_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_తల,    h,         1);

            s_ztensor.is_transformed  = అబద్ధం;
            s2_ztensor.is_transformed = అబద్ధం;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, అబద్ధం, నిజం, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            ఒకవేళ (ముసుగు != శూన్యసూచిక) {
                ZDNN_CHECK(zdnn_కూడు(&s_ztensor, &ముసుగు_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = అబద్ధం;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = నిజం;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    కొరకు (int64_t iq = 0; iq < n_q; iq++) {
        కొరకు (int64_t h = 0; h < n_తల; h++) {
            జ్ఞాపకం_నకలు((అక్షరం *) dst->దత్తాంశం + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*పరిమాణం(దశాంశం));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_బఫరు * dst_extra = (ggml_backend_zdnn_బఫరు *) dst->extra;
    ఒకవేళ (dst_extra != శూన్యసూచిక) {
        dst_extra->ztensor.is_transformed = అబద్ధం;
    }
}
