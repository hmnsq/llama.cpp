#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "fattn.hpp"

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
স্থির অন্তঃস্থ ভিন্ন ggml_zdnn_fattn_mask_মান(ggml_fp16_t m) {
    ঘূৰাই std::max(GGML_FP16_TO_FP32(m), -65504.0f);
}

তার্কিক ggml_zdnn_supports_flash_attn_ext(ধ্রুৱ ggml_backend_zdnn_device_context * ctx_dev, ধ্রুৱ ggml_tensor * op) {
    ধ্রুৱ ggml_tensor * q    = op->src[0];
    ধ্রুৱ ggml_tensor * k    = op->src[1];
    ধ্রুৱ ggml_tensor * v    = op->src[2];
    ধ্রুৱ ggml_tensor * mask = op->src[3];

    ভিন্ন max_bias      = 0.0f;
    ভিন্ন logit_softcap = 0.0f;
    memcpy(&max_bias,      (ধ্রুৱ ভিন্ন *) op->op_params + 1, পরিমাণ(ভিন্ন));
    memcpy(&logit_softcap, (ধ্রুৱ ভিন্ন *) op->op_params + 2, পরিমাণ(ভিন্ন));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    যদি (op->src[4] != nullptr || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        ঘূৰাই মিছা;
    }

    যদি (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        ঘূৰাই মিছা;
    }

    যদি (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1)) {
        ঘূৰাই মিছা;
    }

    যদি (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        ঘূৰাই মিছা;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    যদি (q->nb[0] != পরিমাণ(ভিন্ন) || k->nb[0] != পরিমাণ(ggml_fp16_t) || v->nb[0] != পরিমাণ(ggml_fp16_t)) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ int64_t max_dim = ctx_dev->max_size;
    ধ্রুৱ int64_t group   = q->ne[2] / k->ne[2];

    ঘূৰাই q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && group * q->ne[1] <= max_dim;
}

শূন্য ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * q    = dst->src[0];
    ধ্রুৱ ggml_tensor * k    = dst->src[1];
    ধ্রুৱ ggml_tensor * v    = dst->src[2];
    ধ্রুৱ ggml_tensor * mask = dst->src[3];

    ধ্রুৱ int64_t d_k       = q->ne[0];
    ধ্রুৱ int64_t n_q       = q->ne[1];
    ধ্রুৱ int64_t n_head    = q->ne[2];
    ধ্রুৱ int64_t n_kv      = k->ne[1];
    ধ্রুৱ int64_t n_head_kv = k->ne[2];
    ধ্রুৱ int64_t d_v       = v->ne[0];
    ধ্রুৱ int64_t group     = n_head / n_head_kv;

    ভিন্ন scale = 1.0f;
    memcpy(&scale, (ধ্রুৱ ভিন্ন *) dst->op_params + 0, পরিমাণ(ভিন্ন));

    ধ্রুৱ int32_t n_threads = n_q*n_kv >= 65536 ? ctx->n_threads : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    ধ্রুৱ তার্কিক single = n_q == 1;

    ধ্রুৱ int64_t n_mask = mask ? (single ? n_head_kv*n_kv : n_q*n_kv) : 0;

    ggml_zdnn_pool_alloc<ভিন্ন>       q_host_alloc(ctx->pool, n_head*n_q*d_k);
    ggml_zdnn_pool_alloc<ভিন্ন>       o_host_alloc(ctx->pool, n_head*n_q*d_v);
    ggml_zdnn_pool_alloc<ভিন্ন>       mask_host_alloc(ctx->pool, n_mask);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_head_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_head_kv*n_kv*d_v);

    ভিন্ন       * q_host    = q_host_alloc.get();
    ভিন্ন       * o_host    = o_host_alloc.get();
    ভিন্ন       * mask_host = mask_host_alloc.get();
    ggml_fp16_t * k_host    = k_host_alloc.get();
    ggml_fp16_t * v_host    = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_threads, n_head, [&](int64_t h_start, int64_t h_end) {
        বাবে (int64_t h = h_start; h < h_end; h++) {
            বাবে (int64_t iq = 0; iq < n_q; iq++) {
                ধ্রুৱ ভিন্ন * src = (ধ্রুৱ ভিন্ন *)((ধ্রুৱ ৱর্ণ *) q->data + iq*q->nb[1] + h*q->nb[2]);
                ভিন্ন       * row = q_host + (h*n_q + iq)*d_k;
                বাবে (int64_t i0 = 0; i0 < d_k; i0++) {
                    row[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_threads, n_head_kv*n_kv, [&](int64_t r_start, int64_t r_end) {
        বাবে (int64_t r = r_start; r < r_end; r++) {
            ধ্রুৱ int64_t hk = r / n_kv;
            ধ্রুৱ int64_t ik = r % n_kv;
            memcpy(k_host + r*d_k, (ধ্রুৱ ৱর্ণ *) k->data + ik*k->nb[1] + hk*k->nb[2], d_k*পরিমাণ(ggml_fp16_t));
            memcpy(v_host + r*d_v, (ধ্রুৱ ৱর্ণ *) v->data + ik*v->nb[1] + hk*v->nb[2], d_v*পরিমাণ(ggml_fp16_t));
        }
    });

    zdnn_tensor_desc q_pre_tfm_desc, q_tfm_desc;
    zdnn_tensor_desc k_pre_tfm_desc, k_tfm_desc;
    zdnn_tensor_desc v_pre_tfm_desc, v_tfm_desc;
    zdnn_tensor_desc o_pre_tfm_desc, o_tfm_desc;
    zdnn_ztensor     q_ztensor, k_ztensor, v_ztensor, o_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> q_ztensor_alloc, k_ztensor_alloc, v_ztensor_alloc, o_ztensor_alloc;

    // a single query keeps the heads of a kv head in one stack, which is the same memory order as one head per stack
    যদি (single) {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_head_kv, group, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_head_kv, group, d_v);
    } অন্যথা {
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

    যদি (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        zdnn_tensor_desc bias_pre_tfm_desc, bias_tfm_desc;
        zdnn_ztensor     bias_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> bias_ztensor_alloc;

        ধ্রুৱ zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_head_kv, n_kv);
        যদি (mask != nullptr) {
            ধ্রুৱ ggml_fp16_t * mask_row = (ধ্রুৱ ggml_fp16_t *) mask->data;
            বাবে (int64_t hk = 0; hk < n_head_kv; hk++) {
                বাবে (int64_t ik = 0; ik < n_kv; ik++) {
                    mask_host[hk*n_kv + ik] = ggml_zdnn_fattn_mask_মান(mask_row[ik]);
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

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, মিছা, সঁচা, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_head_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } অন্যথা {
        zdnn_tensor_desc mask_pre_tfm_desc, mask_tfm_desc;
        zdnn_ztensor     mask_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> mask_ztensor_alloc;

        যদি (mask != nullptr) {
            ggml_zdnn_parallel_for(n_threads, n_q, [&](int64_t iq_start, int64_t iq_end) {
                বাবে (int64_t iq = iq_start; iq < iq_end; iq++) {
                    ধ্রুৱ ggml_fp16_t * mask_row = (ধ্রুৱ ggml_fp16_t *)((ধ্রুৱ ৱর্ণ *) mask->data + iq*mask->nb[1]);
                    বাবে (int64_t ik = 0; ik < n_kv; ik++) {
                        mask_host[iq*n_kv + ik] = ggml_zdnn_fattn_mask_মান(mask_row[ik]);
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

        ধ্রুৱ zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        ধ্রুৱ zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        বাবে (int64_t h = 0; h < n_head; h++) {
            ধ্রুৱ zdnn_ztensor q_view = ggml_zdnn_স্তূপ_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_head,    h,         1);
            ধ্রুৱ zdnn_ztensor k_view = ggml_zdnn_স্তূপ_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_head_kv, h / group, 1);
            ধ্রুৱ zdnn_ztensor v_view = ggml_zdnn_স্তূপ_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_head_kv, h / group, 1);
            zdnn_ztensor       o_view = ggml_zdnn_স্তূপ_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_head,    h,         1);

            s_ztensor.is_transformed  = মিছা;
            s2_ztensor.is_transformed = মিছা;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, মিছা, সঁচা, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            যদি (mask != nullptr) {
                ZDNN_CHECK(zdnn_add(&s_ztensor, &mask_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = মিছা;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = সঁচা;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    বাবে (int64_t iq = 0; iq < n_q; iq++) {
        বাবে (int64_t h = 0; h < n_head; h++) {
            memcpy((ৱর্ণ *) dst->data + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*পরিমাণ(ভিন্ন));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_buffer * dst_extra = (ggml_backend_zdnn_buffer *) dst->extra;
    যদি (dst_extra != nullptr) {
        dst_extra->ztensor.is_transformed = মিছা;
    }
}
