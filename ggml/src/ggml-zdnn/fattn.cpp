#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "fattn.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_zdnn_fattn_mask_ಮೌಲ್ಯ(ggml_fp16_t m) {
    ಪ್ರತ್ಯಾಗಮ std::max(GGML_FP16_TO_FP32(m), -65504.0f);
}

ತಾರ್ಕಿಕ ggml_zdnn_supports_flash_attn_ext(ಧ್ರುವ ggml_backend_zdnn_device_context * ctx_dev, ಧ್ರುವ ggml_tensor * op) {
    ಧ್ರುವ ggml_tensor * q    = op->src[0];
    ಧ್ರುವ ggml_tensor * k    = op->src[1];
    ಧ್ರುವ ggml_tensor * v    = op->src[2];
    ಧ್ರುವ ggml_tensor * mask = op->src[3];

    ಭಿನ್ನ max_bias      = 0.0f;
    ಭಿನ್ನ logit_softcap = 0.0f;
    memcpy(&max_bias,      (ಧ್ರುವ ಭಿನ್ನ *) op->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&logit_softcap, (ಧ್ರುವ ಭಿನ್ನ *) op->op_params + 2, ಪರಿಮಾಣ(ಭಿನ್ನ));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    ಯದಿ (op->src[4] != nullptr || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    ಯದಿ (q->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || k->nb[0] != ಪರಿಮಾಣ(ggml_fp16_t) || v->nb[0] != ಪರಿಮಾಣ(ggml_fp16_t)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t max_dim = ctx_dev->max_size;
    ಧ್ರುವ int64_t group   = q->ne[2] / k->ne[2];

    ಪ್ರತ್ಯಾಗಮ q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && group * q->ne[1] <= max_dim;
}

ಶೂನ್ಯ ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * q    = dst->src[0];
    ಧ್ರುವ ggml_tensor * k    = dst->src[1];
    ಧ್ರುವ ggml_tensor * v    = dst->src[2];
    ಧ್ರುವ ggml_tensor * mask = dst->src[3];

    ಧ್ರುವ int64_t d_k       = q->ne[0];
    ಧ್ರುವ int64_t n_q       = q->ne[1];
    ಧ್ರುವ int64_t n_head    = q->ne[2];
    ಧ್ರುವ int64_t n_kv      = k->ne[1];
    ಧ್ರುವ int64_t n_head_kv = k->ne[2];
    ಧ್ರುವ int64_t d_v       = v->ne[0];
    ಧ್ರುವ int64_t group     = n_head / n_head_kv;

    ಭಿನ್ನ scale = 1.0f;
    memcpy(&scale, (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಧ್ರುವ int32_t n_threads = n_q*n_kv >= 65536 ? ctx->n_threads : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    ಧ್ರುವ ತಾರ್ಕಿಕ single = n_q == 1;

    ಧ್ರುವ int64_t n_mask = mask ? (single ? n_head_kv*n_kv : n_q*n_kv) : 0;

    ggml_zdnn_pool_alloc<ಭಿನ್ನ>       q_host_alloc(ctx->pool, n_head*n_q*d_k);
    ggml_zdnn_pool_alloc<ಭಿನ್ನ>       o_host_alloc(ctx->pool, n_head*n_q*d_v);
    ggml_zdnn_pool_alloc<ಭಿನ್ನ>       mask_host_alloc(ctx->pool, n_mask);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_head_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_head_kv*n_kv*d_v);

    ಭಿನ್ನ       * q_host    = q_host_alloc.get();
    ಭಿನ್ನ       * o_host    = o_host_alloc.get();
    ಭಿನ್ನ       * mask_host = mask_host_alloc.get();
    ggml_fp16_t * k_host    = k_host_alloc.get();
    ggml_fp16_t * v_host    = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_threads, n_head, [&](int64_t h_start, int64_t h_end) {
        ಪ್ರತಿ (int64_t h = h_start; h < h_end; h++) {
            ಪ್ರತಿ (int64_t iq = 0; iq < n_q; iq++) {
                ಧ್ರುವ ಭಿನ್ನ * src = (ಧ್ರುವ ಭಿನ್ನ *)((ಧ್ರುವ ವರ್ಣ *) q->data + iq*q->nb[1] + h*q->nb[2]);
                ಭಿನ್ನ       * row = q_host + (h*n_q + iq)*d_k;
                ಪ್ರತಿ (int64_t i0 = 0; i0 < d_k; i0++) {
                    row[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_threads, n_head_kv*n_kv, [&](int64_t r_start, int64_t r_end) {
        ಪ್ರತಿ (int64_t r = r_start; r < r_end; r++) {
            ಧ್ರುವ int64_t hk = r / n_kv;
            ಧ್ರುವ int64_t ik = r % n_kv;
            memcpy(k_host + r*d_k, (ಧ್ರುವ ವರ್ಣ *) k->data + ik*k->nb[1] + hk*k->nb[2], d_k*ಪರಿಮಾಣ(ggml_fp16_t));
            memcpy(v_host + r*d_v, (ಧ್ರುವ ವರ್ಣ *) v->data + ik*v->nb[1] + hk*v->nb[2], d_v*ಪರಿಮಾಣ(ggml_fp16_t));
        }
    });

    zdnn_tensor_desc q_pre_tfm_desc, q_tfm_desc;
    zdnn_tensor_desc k_pre_tfm_desc, k_tfm_desc;
    zdnn_tensor_desc v_pre_tfm_desc, v_tfm_desc;
    zdnn_tensor_desc o_pre_tfm_desc, o_tfm_desc;
    zdnn_ztensor     q_ztensor, k_ztensor, v_ztensor, o_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> q_ztensor_alloc, k_ztensor_alloc, v_ztensor_alloc, o_ztensor_alloc;

    // a single query keeps the heads of a kv head in one stack, which is the same memory order as one head per stack
    ಯದಿ (single) {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_head_kv, group, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_head_kv, group, d_v);
    } ಅನ್ಯಥಾ {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_head, n_q, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_head, n_q, d_v);
    }
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &k_pre_tfm_desc, n_head_kv, n_kv, d_k);
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &v_pre_tfm_desc, n_head_kv, n_kv, d_v);

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

    ggml_zdnn_pool_alloc<uint8_t> softmax_save_area(ctx->pool, ZDNN_SOFTMAX_SAVEAREA_SIZE);

    zdnn_tensor_desc s_pre_tfm_desc, s_tfm_desc;
    zdnn_ztensor     s_ztensor, s2_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> s_ztensor_alloc, s2_ztensor_alloc;

    ಯದಿ (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        zdnn_tensor_desc bias_pre_tfm_desc, bias_tfm_desc;
        zdnn_ztensor     bias_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> bias_ztensor_alloc;

        ಧ್ರುವ zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_head_kv, n_kv);
        ಯದಿ (mask != nullptr) {
            ಧ್ರುವ ggml_fp16_t * mask_row = (ಧ್ರುವ ggml_fp16_t *) mask->data;
            ಪ್ರತಿ (int64_t hk = 0; hk < n_head_kv; hk++) {
                ಪ್ರತಿ (int64_t ik = 0; ik < n_kv; ik++) {
                    mask_host[hk*n_kv + ik] = ggml_zdnn_fattn_mask_ಮೌಲ್ಯ(mask_row[ik]);
                }
            }

            zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias_pre_tfm_desc, n_head_kv, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&bias_pre_tfm_desc, &bias_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &bias_pre_tfm_desc, &bias_tfm_desc, &bias_ztensor, bias_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&bias_ztensor, mask_host));
            bias = &bias_ztensor;
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, n_head_kv, group, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, ಅಸತ್ಯ, ಸತ್ಯ, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_head_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } ಅನ್ಯಥಾ {
        zdnn_tensor_desc mask_pre_tfm_desc, mask_tfm_desc;
        zdnn_ztensor     mask_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> mask_ztensor_alloc;

        ಯದಿ (mask != nullptr) {
            ggml_zdnn_parallel_for(n_threads, n_q, [&](int64_t iq_start, int64_t iq_end) {
                ಪ್ರತಿ (int64_t iq = iq_start; iq < iq_end; iq++) {
                    ಧ್ರುವ ggml_fp16_t * mask_row = (ಧ್ರುವ ggml_fp16_t *)((ಧ್ರುವ ವರ್ಣ *) mask->data + iq*mask->nb[1]);
                    ಪ್ರತಿ (int64_t ik = 0; ik < n_kv; ik++) {
                        mask_host[iq*n_kv + ik] = ggml_zdnn_fattn_mask_ಮೌಲ್ಯ(mask_row[ik]);
                    }
                }
            });

            zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &mask_pre_tfm_desc, 1, n_q, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&mask_pre_tfm_desc, &mask_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &mask_pre_tfm_desc, &mask_tfm_desc, &mask_ztensor, mask_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&mask_ztensor, mask_host));
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, 1, n_q, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ಧ್ರುವ zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        ಧ್ರುವ zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        ಪ್ರತಿ (int64_t h = 0; h < n_head; h++) {
            ಧ್ರುವ zdnn_ztensor q_view = ggml_zdnn_ರಾಶಿ_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_head,    h,         1);
            ಧ್ರುವ zdnn_ztensor k_view = ggml_zdnn_ರಾಶಿ_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_head_kv, h / group, 1);
            ಧ್ರುವ zdnn_ztensor v_view = ggml_zdnn_ರಾಶಿ_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_head_kv, h / group, 1);
            zdnn_ztensor       o_view = ggml_zdnn_ರಾಶಿ_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_head,    h,         1);

            s_ztensor.is_transformed  = ಅಸತ್ಯ;
            s2_ztensor.is_transformed = ಅಸತ್ಯ;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, ಅಸತ್ಯ, ಸತ್ಯ, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            ಯದಿ (mask != nullptr) {
                ZDNN_CHECK(zdnn_add(&s_ztensor, &mask_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = ಅಸತ್ಯ;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = ಸತ್ಯ;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    ಪ್ರತಿ (int64_t iq = 0; iq < n_q; iq++) {
        ಪ್ರತಿ (int64_t h = 0; h < n_head; h++) {
            memcpy((ವರ್ಣ *) dst->data + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*ಪರಿಮಾಣ(ಭಿನ್ನ));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_buffer * dst_extra = (ggml_backend_zdnn_buffer *) dst->extra;
    ಯದಿ (dst_extra != nullptr) {
        dst_extra->ztensor.is_transformed = ಅಸತ್ಯ;
    }
}
