#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "fattn.hpp"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cstring>

// DLFLOAT16 has no infinity, so masked scores use the lowest FP16 value, exp of which is still 0
স্থির অন্তঃরেখ দশমিক ggml_zdnn_fattn_mask_মান(ggml_fp16_t m) {
    ফেরত std::max(GGML_FP16_TO_FP32(m), -65504.0f);
}

যৌক্তিক ggml_zdnn_supports_flash_attn_ext(ধ্রুবক ggml_backend_zdnn_device_context * ctx_dev, ধ্রুবক ggml_tensor * op) {
    ধ্রুবক ggml_tensor * q    = op->src[0];
    ধ্রুবক ggml_tensor * k    = op->src[1];
    ধ্রুবক ggml_tensor * v    = op->src[2];
    ধ্রুবক ggml_tensor * mask = op->src[3];

    দশমিক max_bias      = 0.0f;
    দশমিক logit_softcap = 0.0f;
    memcpy(&max_bias,      (ধ্রুবক দশমিক *) op->op_params + 1, আকার(দশমিক));
    memcpy(&logit_softcap, (ধ্রুবক দশমিক *) op->op_params + 2, আকার(দশমিক));

    // sinks, alibi, softcap and sparse k/v sets are not implemented
    যদি (op->src[4] != nullptr || max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(op, 4) != 0) {
        ফেরত মিথ্যা;
    }

    যদি (op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        ফেরত মিথ্যা;
    }

    যদি (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1)) {
        ফেরত মিথ্যা;
    }

    যদি (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] != 0) {
        ফেরত মিথ্যা;
    }

    // rows are gathered with their strides, but each row has to be contiguous
    যদি (q->nb[0] != আকার(দশমিক) || k->nb[0] != আকার(ggml_fp16_t) || v->nb[0] != আকার(ggml_fp16_t)) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক int64_t max_dim = ctx_dev->max_size;
    ধ্রুবক int64_t group   = q->ne[2] / k->ne[2];

    ফেরত q->ne[0] <= max_dim && v->ne[0] <= max_dim && k->ne[1] <= max_dim &&
           q->ne[2] <= max_dim && group * q->ne[1] <= max_dim;
}

শূন্য ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * q    = dst->src[0];
    ধ্রুবক ggml_tensor * k    = dst->src[1];
    ধ্রুবক ggml_tensor * v    = dst->src[2];
    ধ্রুবক ggml_tensor * mask = dst->src[3];

    ধ্রুবক int64_t d_k       = q->ne[0];
    ধ্রুবক int64_t n_q       = q->ne[1];
    ধ্রুবক int64_t n_শীর্ষ    = q->ne[2];
    ধ্রুবক int64_t n_kv      = k->ne[1];
    ধ্রুবক int64_t n_শীর্ষ_kv = k->ne[2];
    ধ্রুবক int64_t d_v       = v->ne[0];
    ধ্রুবক int64_t group     = n_শীর্ষ / n_শীর্ষ_kv;

    দশমিক scale = 1.0f;
    memcpy(&scale, (ধ্রুবক দশমিক *) dst->op_params + 0, আকার(দশমিক));

    ধ্রুবক int32_t n_threads = n_q*n_kv >= 65536 ? ctx->n_threads : 1;

    // one query runs all heads in one call with the mask row as the K*Q bias, more queries run per head and add the mask
    ধ্রুবক যৌক্তিক single = n_q == 1;

    ধ্রুবক int64_t n_mask = mask ? (single ? n_শীর্ষ_kv*n_kv : n_q*n_kv) : 0;

    ggml_zdnn_pool_alloc<দশমিক>       q_host_alloc(ctx->pool, n_শীর্ষ*n_q*d_k);
    ggml_zdnn_pool_alloc<দশমিক>       o_host_alloc(ctx->pool, n_শীর্ষ*n_q*d_v);
    ggml_zdnn_pool_alloc<দশমিক>       mask_host_alloc(ctx->pool, n_mask);
    ggml_zdnn_pool_alloc<ggml_fp16_t> k_host_alloc(ctx->pool, n_শীর্ষ_kv*n_kv*d_k);
    ggml_zdnn_pool_alloc<ggml_fp16_t> v_host_alloc(ctx->pool, n_শীর্ষ_kv*n_kv*d_v);

    দশমিক       * q_host    = q_host_alloc.get();
    দশমিক       * o_host    = o_host_alloc.get();
    দশমিক       * mask_host = mask_host_alloc.get();
    ggml_fp16_t * k_host    = k_host_alloc.get();
    ggml_fp16_t * v_host    = v_host_alloc.get();

    // q is [n_head][n_q][d_k] with the scale folded in, k and v are [n_head_kv][n_kv][d]
    ggml_zdnn_parallel_for(n_threads, n_শীর্ষ, [&](int64_t h_start, int64_t h_end) {
        জন্য (int64_t h = h_start; h < h_end; h++) {
            জন্য (int64_t iq = 0; iq < n_q; iq++) {
                ধ্রুবক দশমিক * src = (ধ্রুবক দশমিক *)((ধ্রুবক অক্ষর *) q->তথ্য + iq*q->nb[1] + h*q->nb[2]);
                দশমিক       * row = q_host + (h*n_q + iq)*d_k;
                জন্য (int64_t i0 = 0; i0 < d_k; i0++) {
                    row[i0] = src[i0] * scale;
                }
            }
        }
    });

    ggml_zdnn_parallel_for(n_threads, n_শীর্ষ_kv*n_kv, [&](int64_t r_start, int64_t r_end) {
        জন্য (int64_t r = r_start; r < r_end; r++) {
            ধ্রুবক int64_t hk = r / n_kv;
            ধ্রুবক int64_t ik = r % n_kv;
            memcpy(k_host + r*d_k, (ধ্রুবক অক্ষর *) k->তথ্য + ik*k->nb[1] + hk*k->nb[2], d_k*আকার(ggml_fp16_t));
            memcpy(v_host + r*d_v, (ধ্রুবক অক্ষর *) v->তথ্য + ik*v->nb[1] + hk*v->nb[2], d_v*আকার(ggml_fp16_t));
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
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_শীর্ষ_kv, group, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_শীর্ষ_kv, group, d_v);
    } অন্যথা {
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &q_pre_tfm_desc, n_শীর্ষ, n_q, d_k);
        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &o_pre_tfm_desc, n_শীর্ষ, n_q, d_v);
    }
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &k_pre_tfm_desc, n_শীর্ষ_kv, n_kv, d_k);
    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP16, &v_pre_tfm_desc, n_শীর্ষ_kv, n_kv, d_v);

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

    ggml_zdnn_pool_alloc<uint8_t> softmax_save_area(ctx->pool, ZDNN_SOFTMAX_SAVEAREA_আকার);

    zdnn_tensor_desc s_pre_tfm_desc, s_tfm_desc;
    zdnn_ztensor     s_ztensor, s2_ztensor;

    ggml_zdnn_pool_alloc<uint8_t> s_ztensor_alloc, s2_ztensor_alloc;

    যদি (single) {
        // the bias of K*Q is the mask row of the query, repeated for every kv head
        zdnn_tensor_desc bias_pre_tfm_desc, bias_tfm_desc;
        zdnn_ztensor     bias_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> bias_ztensor_alloc;

        ধ্রুবক zdnn_ztensor * bias = &ggml_zdnn_zero_bias(ctx, n_শীর্ষ_kv, n_kv);
        যদি (mask != nullptr) {
            ধ্রুবক ggml_fp16_t * mask_row = (ধ্রুবক ggml_fp16_t *) mask->তথ্য;
            জন্য (int64_t hk = 0; hk < n_শীর্ষ_kv; hk++) {
                জন্য (int64_t ik = 0; ik < n_kv; ik++) {
                    mask_host[hk*n_kv + ik] = ggml_zdnn_fattn_mask_মান(mask_row[ik]);
                }
            }

            zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &bias_pre_tfm_desc, n_শীর্ষ_kv, n_kv);
            ZDNN_CHECK(zdnn_generate_transformed_desc(&bias_pre_tfm_desc, &bias_tfm_desc));
            ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &bias_pre_tfm_desc, &bias_tfm_desc, &bias_ztensor, bias_ztensor_alloc);
            ZDNN_CHECK(zdnn_transform_ztensor(&bias_ztensor, mask_host));
            bias = &bias_ztensor;
        }

        zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &s_pre_tfm_desc, n_শীর্ষ_kv, group, n_kv);
        ZDNN_CHECK(zdnn_generate_transformed_desc(&s_pre_tfm_desc, &s_tfm_desc));
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s_ztensor,  s_ztensor_alloc);
        ggml_zdnn_init_scratch_ztensor_dlf16(ctx, &s_pre_tfm_desc, &s_tfm_desc, &s2_ztensor, s2_ztensor_alloc);

        ZDNN_CHECK(zdnn_matmul_transpose_op(&q_ztensor, &k_ztensor, bias, মিথ্যা, সত্য, MATMUL_OP_ADDITION, &s_ztensor));
        ZDNN_CHECK(zdnn_softmax(&s_ztensor, softmax_save_area.get(), SOFTMAX_ACT_NONE, &s2_ztensor));
        ZDNN_CHECK(zdnn_matmul_op(&s2_ztensor, &v_ztensor, &ggml_zdnn_zero_bias(ctx, n_শীর্ষ_kv, d_v), MATMUL_OP_ADDITION, &o_ztensor));
    } অন্যথা {
        zdnn_tensor_desc mask_pre_tfm_desc, mask_tfm_desc;
        zdnn_ztensor     mask_ztensor;

        ggml_zdnn_pool_alloc<uint8_t> mask_ztensor_alloc;

        যদি (mask != nullptr) {
            ggml_zdnn_parallel_for(n_threads, n_q, [&](int64_t iq_start, int64_t iq_end) {
                জন্য (int64_t iq = iq_start; iq < iq_end; iq++) {
                    ধ্রুবক ggml_fp16_t * mask_row = (ধ্রুবক ggml_fp16_t *)((ধ্রুবক অক্ষর *) mask->তথ্য + iq*mask->nb[1]);
                    জন্য (int64_t ik = 0; ik < n_kv; ik++) {
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

        ধ্রুবক zdnn_ztensor & k_bias = ggml_zdnn_zero_bias(ctx, 1, n_kv);
        ধ্রুবক zdnn_ztensor & v_bias = ggml_zdnn_zero_bias(ctx, 1, d_v);

        zdnn_tensor_desc view_pre_tfm_desc[4];
        zdnn_tensor_desc view_tfm_desc[4];

        জন্য (int64_t h = 0; h < n_শীর্ষ; h++) {
            ধ্রুবক zdnn_ztensor q_view = ggml_zdnn_স্তূপ_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], q_ztensor, n_শীর্ষ,    h,         1);
            ধ্রুবক zdnn_ztensor k_view = ggml_zdnn_স্তূপ_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], k_ztensor, n_শীর্ষ_kv, h / group, 1);
            ধ্রুবক zdnn_ztensor v_view = ggml_zdnn_স্তূপ_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], v_ztensor, n_শীর্ষ_kv, h / group, 1);
            zdnn_ztensor       o_view = ggml_zdnn_স্তূপ_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], o_ztensor, n_শীর্ষ,    h,         1);

            s_ztensor.is_transformed  = মিথ্যা;
            s2_ztensor.is_transformed = মিথ্যা;

            ZDNN_CHECK(zdnn_matmul_transpose_op(&q_view, &k_view, &k_bias, মিথ্যা, সত্য, MATMUL_OP_ADDITION, &s_ztensor));

            zdnn_ztensor * scores = &s_ztensor;
            zdnn_ztensor * probs  = &s2_ztensor;
            যদি (mask != nullptr) {
                ZDNN_CHECK(zdnn_add(&s_ztensor, &mask_ztensor, &s2_ztensor));
                scores = &s2_ztensor;
                probs  = &s_ztensor;
                probs->is_transformed = মিথ্যা;
            }

            ZDNN_CHECK(zdnn_softmax(scores, softmax_save_area.get(), SOFTMAX_ACT_NONE, probs));
            ZDNN_CHECK(zdnn_matmul_op(probs, &v_view, &v_bias, MATMUL_OP_ADDITION, &o_view));
        }

        o_ztensor.is_transformed = সত্য;
    }

    ZDNN_CHECK(zdnn_transform_origtensor(&o_ztensor, o_host));

    // the result is [d_v][n_head][n_q]
    জন্য (int64_t iq = 0; iq < n_q; iq++) {
        জন্য (int64_t h = 0; h < n_শীর্ষ; h++) {
            memcpy((অক্ষর *) dst->তথ্য + h*dst->nb[1] + iq*dst->nb[2], o_host + (h*n_q + iq)*d_v, d_v*আকার(দশমিক));
        }
    }

    // the result is written to dst->data directly, so the dst ztensor is stale
    ggml_backend_zdnn_বাফার * dst_extra = (ggml_backend_zdnn_বাফার *) dst->extra;
    যদি (dst_extra != nullptr) {
        dst_extra->ztensor.is_transformed = মিথ্যা;
    }
}
