//******************************************************************************
// F32 Flash Attention for ET backend
//
// Supports:
// - arbitrary dk/dv (up to 128)
// - GQA (n_head_q can differ from n_head_kv)
// - mask (F16 or F32, causal pattern)
// - F16 or F32 K and V (with non-contiguous strides from KV cache permute)
//
// Limitations:
// - Q and dst must be F32
// - no sinks, ALiBi, logit softcap
//
// Parallelization strategy:
// - flatten [query position, head, outer batch] into independent rows
// - assign rows round-robin across ET threads
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdbool.h>
#شامل <stdint.h>

ساختار ggml_et_flash_attn_ext_params {
    ساختار ggml_tensor src0;      // Q tensor (F32)
    ساختار ggml_tensor src1;      // K tensor (F16 or F32)
    ساختار ggml_tensor src2;      // V tensor (F16 or F32)
    ساختار ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    ساختار ggml_tensor dst;       // Output tensor (F32)
    اعشاری              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
ایستا درونخطی اعشاری read_kv_f32(ثابت نویسه * row_base, int64_t d, int64_t nb0, صحیح type) {
    اگر (type == GGML_TYPE_F32) {
        بازگشت *(ثابت اعشاری *) (row_base + d * nb0);
    }
    // F16
    بازگشت fp16_to_fp32(*(ثابت uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
ایستا درونخطی اعشاری dot_qk(ثابت اعشاری * q, ثابت نویسه * k_row, int64_t dk, int64_t k_nb0, صحیح k_type) {
    اعشاری acc = 0.0f;
    اگر (k_type == GGML_TYPE_F32) {
        ثابت اعشاری * kf = (ثابت اعشاری *) k_row;
        برای (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } وگرنه {
        // F16 stride-aware read
        برای (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(ثابت uint16_t *) (k_row + i * k_nb0));
        }
    }
    بازگشت acc;
}

ایستا درونخطی اعشاری get_mask_val(ثابت ساختار ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    ثابت نویسه * base = (ثابت نویسه *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    اگر (mask->type == GGML_TYPE_F32) {
        بازگشت *(ثابت اعشاری *) (base + ik1 * mask->nb[0]);
    }
    // F16
    بازگشت fp16_to_fp32(*(ثابت uint16_t *) (base + ik1 * mask->nb[0]));
}

صحیح entry_point(ساختار ggml_et_flash_attn_ext_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env || !params) {
        بازگشت -1;
    }

    ثابت صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ثابت صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);
    اگر (thread_id < 0 || رقم_threads <= 0) {
        بازگشت 0;
    }

    ساختار ggml_tensor * q        = &params->src0;
    ساختار ggml_tensor * k        = &params->src1;
    ساختار ggml_tensor * v        = &params->src2;
    ساختار ggml_tensor * dst      = &params->dst;
    ثابت int32_t        has_mask = params->has_mask;
    ساختار ggml_tensor * mask     = has_mask ? &params->mask : (ساختار ggml_tensor *) 0;

    ثابت نویسه * q_data   = (ثابت نویسه *) q->data;
    ثابت نویسه * k_data   = (ثابت نویسه *) k->data;
    ثابت نویسه * v_data   = (ثابت نویسه *) v->data;
    نویسه *       dst_data = (نویسه *) dst->data;

    ثابت صحیح     k_type = k->type;
    ثابت صحیح     v_type = v->type;
    ثابت int64_t k_nb0  = k->nb[0];
    ثابت int64_t v_nb0  = v->nb[0];

    ثابت int64_t dk  = q->ne[0];  // head dim for keys/queries
    ثابت int64_t nq  = q->ne[1];  // number of query positions
    ثابت int64_t nhq = q->ne[2];  // number of query heads
    ثابت int64_t no  = q->ne[3];  // outer batch

    ثابت int64_t nk  = k->ne[1];  // number of key/value positions
    ثابت int64_t nhk = k->ne[2];  // number of kv heads
    ثابت int64_t dv  = v->ne[0];  // head dim for values

    اگر (dv > FA_DV_MAX) {
        بازگشت -1;
    }

    // GQA: query heads per kv head
    ثابت int64_t gqa_ratio = nhq / nhk;

    ثابت int64_t total_rows = nq * nhq * no;
    ثابت اعشاری   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    ثابت صحیح use_fast_store = (dv % 16 == 0);

    برای (int64_t row = thread_id; row < total_rows; row += رقم_threads) {
        ثابت int64_t iq3 = row / (nhq * nq);
        ثابت int64_t rem = row % (nhq * nq);
        ثابت int64_t iq2 = rem / nq;  // query head index
        ثابت int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        ثابت int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        ثابت اعشاری * pq = (ثابت اعشاری *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        اعشاری * out = (اعشاری *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        ثابت int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        ثابت int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        اعشاری acc[FA_DV_MAX];
        برای (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        اعشاری M = -3.402823466e+38f;
        اعشاری S = 0.0f;

        برای (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            اعشاری mask_val = 0.0f;
            اگر (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                اگر (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    ادامه;
                }
            }

            ثابت نویسه * pk = k_data + ik1 * k->nb[1] + kv_base;
            ثابت نویسه * pv = v_data + ik1 * v->nb[1] + vv_base;

            اعشاری       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            ثابت اعشاری Mold = M;

            اعشاری ms = 1.0f;
            اعشاری vs = 1.0f;
            اگر (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                برای (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } وگرنه {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            اگر (v_type == GGML_TYPE_F32) {
                ثابت اعشاری * pvf = (ثابت اعشاری *) pv;
                برای (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } وگرنه {
                برای (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(ثابت uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        ثابت اعشاری S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        اگر (use_fast_store) {
            برای (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } وگرنه {
            برای (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((فرار اعشاری *) &out[d], acc[d] * S_inv);
            }
        }
    }

    بازگشت 0;
}
