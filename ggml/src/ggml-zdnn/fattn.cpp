#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "fattn.hpp"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
സ്ഥിതം വരിയിൽ ദശാംശം ggml_zdnn_fattn_mask_മൂല്യം(ggml_fp16_t m) {
    തിരികെ std::max(GGML_FP16_TO_FP32(m), -65504.0f);
}

തർക്കം ggml_zdnn_supports_flash_attn_ext(സ്ഥിരം ggml_backend_zdnn_device_context * ctx_dev, സ്ഥിരം ggml_tensor * op) {
    സ്ഥിരം ggml_tensor * q    = op->src[0];
    സ്ഥിരം ggml_tensor * k    = op->src[1];
    സ്ഥിരം ggml_tensor * v    = op->src[2];
    സ്ഥിരം ggml_tensor * mask = op->src[3];

    ദശാംശം max_bias      = 0.0f;
    ദശാംശം logit_softcap = 0.0f;
    memcpy(&max_bias,      (സ്ഥിരം ദശാംശം *) op->op_params + 1, വലുപ്പം(ദശാംശം));
    memcpy(&logit_softcap, (സ്ഥിരം ദശാംശം *) op->op_params + 2, വലുപ്പം(ദശാംശം));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    എങ്കിൽ (op->src[4] != nullptr || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        തിരികെ അസത്യം;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    എങ്കിൽ (q->nb[0] != വലുപ്പം(ദശാംശം) || k->nb[0] != വലുപ്പം(ggml_fp16_t) || v->nb[0] != വലുപ്പം(ggml_fp16_t)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t max_dim = ctx_dev->max_size;
    സ്ഥിരം int64_t group   = q->ne[2] / k->ne[2];

    തിരികെ q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && group * q->ne[1] <= max_dim;
}

ശൂന്യം ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * q    = dst->src[0];
    സ്ഥിരം ggml_tensor * k    = dst->src[1];
    സ്ഥിരം ggml_tensor * v    = dst->src[2];
    സ്ഥിരം ggml_tensor * mask = dst->src[3];

    സ്ഥിരം int64_t d_k       = q->ne[0];
    സ്ഥിരം int64_t n_q       = q->ne[1];
    സ്ഥിരം int64_t n_head    = q->ne[2];
    സ്ഥിരം int64_t n_kv      = k->ne[1];
    സ്ഥിരം int64_t n_head_kv = k->ne[2];
    സ്ഥിരം int64_t d_v       = v->ne[0];
    സ്ഥിരം int64_t group     = n_head / n_head_kv;

    ദശാംശം scale = 1.0f;
    memcpy(&scale, (സ്ഥിരം ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));

    സ്ഥിരം int32_t n_threads = n_q*n_kv >= 65536 ? ctx->n_threads : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    സ്ഥിരം തർക്കം single = n_q == 1;

    സ്ഥിരം int64_t n_mask = mask ? (single ? n_head_kv*n_kv : n_q*n_kv) : 0;

    ggml_zdnn_pool_alloc<ദശാംശം>       q_host_alloc(ctx->pool, n_head*n_q*d_k);
    ggml_zdnn_pool_alloc<ദശാംശം>       o_host_alloc(ctx->pool, n_head*n_q*d_v);
    ggml_zdnn_pool_alloc<ദശാംശം>       mask_host_alloc(ctx->pool, n_mask);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_head_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_head_kv*n_kv*d_v);

    ദശാംശം       * q_host    = q_host_alloc.get();
    ദശാംശം       * o_host    = o_host_alloc.get();
    ദശാംശം       * mask_host = mask_host_alloc.get();
    ggml_fp16_t * k_host    = k_host_alloc.get();
    ggml_fp16_t * v_host    = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_threads, n_head, [&](int64_t h_start, int64_t h_end) {
        വേണ്ടി (int64_t h = h_start; h < h_end; h++) {
            വേണ്ടി (int64_t iq = 0; iq < n_q; iq++) {
                സ്ഥിരം ദശാംശം * src = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *) q->data + iq*q->nb[1] + h*q->nb[2]);
                ദശാംശം       * row = q_host + (h*n_q + iq)*d_k;
                വേണ്ടി (int64_t i0 = 0; i0 < d_k; i0++) {
                    row[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_threads, n_head_kv*n_kv, [&](int64_t r_start, int64_t r_end) {
        വേണ്ടി (int64_t r = r_start; r < r_end; r++) {
            സ്ഥിരം int64_t hk = r / n_kv;
            സ്ഥിരം int64_t ik = r % n_kv;
            memcpy(k_host + r*d_k, (സ്ഥിരം അക്ഷരം *) k->data + ik*k->nb[1] + hk*k->nb[2], d_k*വലുപ്പം(ggml_fp16_t));
            memcpy(v_host + r*d_v, (സ്ഥിരം അക്ഷരം *) v->data + ik*v->nb[1] + hk*v->nb[2], d_v*വലുപ്പം(ggml_fp16_t));
        }
    });

    zdnn_tensor_desc q_pre_tfm_desc, q_tfm_desc;
    zdnn_tensor_desc k_pre_tfm_desc, k_tfm_desc;
    zdnn_tensor_desc v_pre_tfm_desc, v_tfm_desc;
    zdnn_tensor_desc o_pre_tfm_desc, o_tfm_desc;
    zdnn_ztensor     q_ztensor, k_ztensor, v_ztensor, o_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> q_ztensor_alloc, k_ztensor_alloc, v_ztensor_alloc, o_ztensor_alloc;

    // a single query keeps the heads of a kv head in one stack, which is the same memory order as one head per stack
    എങ്കിൽ (single) {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_head_kv, group, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_head_kv, group, d_v);
    } അല്ലെങ്കിൽ {
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

    എങ്കിൽ (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        zdnn_tensor_desc bias_pre_tfm_desc, bias_tfm_desc;
        zdnn_ztensor     bias_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> bias_ztensor_alloc;

        സ്ഥിരം zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_head_kv, n_kv);
        എങ്കിൽ (mask != nullptr) {
            സ്ഥിരം ggml_fp16_t * mask_row = (സ്ഥിരം ggml_fp16_t *) mask->data;
            വേണ്ടി (int64_t hk = 0; hk < n_head_kv; hk++) {
                വേണ്ടി (int64_t ik = 0; ik < n_kv; ik++) {
                    mask_host[hk*n_kv + ik] = ggml_zdnn_fattn_mask_മൂല്യം(mask_row[ik]);
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

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, അസത്യം, സത്യം, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_head_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } അല്ലെങ്കിൽ {
        zdnn_tensor_desc mask_pre_tfm_desc, mask_tfm_desc;
        zdnn_ztensor     mask_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> mask_ztensor_alloc;

        എങ്കിൽ (mask != nullptr) {
            ggml_zdnn_parallel_for(n_threads, n_q, [&](int64_t iq_start, int64_t iq_end) {
                വേണ്ടി (int64_t iq = iq_start; iq < iq_end; iq++) {
                    സ്ഥിരം ggml_fp16_t * mask_row = (സ്ഥിരം ggml_fp16_t *)((സ്ഥിരം അക്ഷരം *) mask->data + iq*mask->nb[1]);
                    വേണ്ടി (int64_t ik = 0; ik < n_kv; ik++) {
                        mask_host[iq*n_kv + ik] = ggml_zdnn_fattn_mask_മൂല്യം(mask_row[ik]);
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

        സ്ഥിരം zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        സ്ഥിരം zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        വേണ്ടി (int64_t h = 0; h < n_head; h++) {
            സ്ഥിരം zdnn_ztensor q_view = ggml_zdnn_അട്ടി_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_head,    h,         1);
            സ്ഥിരം zdnn_ztensor k_view = ggml_zdnn_അട്ടി_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_head_kv, h / group, 1);
            സ്ഥിരം zdnn_ztensor v_view = ggml_zdnn_അട്ടി_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_head_kv, h / group, 1);
            zdnn_ztensor       o_view = ggml_zdnn_അട്ടി_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_head,    h,         1);

            s_ztensor.is_transformed  = അസത്യം;
            s2_ztensor.is_transformed = അസത്യം;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, അസത്യം, സത്യം, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            എങ്കിൽ (mask != nullptr) {
                ZDNN_CHECK(zdnn_add(&s_ztensor, &mask_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = അസത്യം;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = സത്യം;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    വേണ്ടി (int64_t iq = 0; iq < n_q; iq++) {
        വേണ്ടി (int64_t h = 0; h < n_head; h++) {
            memcpy((അക്ഷരം *) dst->data + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*വലുപ്പം(ദശാംശം));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_buffer * dst_extra = (ggml_backend_zdnn_buffer *) dst->extra;
    എങ്കിൽ (dst_extra != nullptr) {
        dst_extra->ztensor.is_transformed = അസത്യം;
    }
}
