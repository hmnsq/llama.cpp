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

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_flash_attn_ext_params {
    কাঠামো ggml_tensor src0;      // Q tensor (F32)
    কাঠামো ggml_tensor src1;      // K tensor (F16 or F32)
    কাঠামো ggml_tensor src2;      // V tensor (F16 or F32)
    কাঠামো ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    কাঠামো ggml_tensor dst;       // Output tensor (F32)
    দশমিক              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
স্থির অন্তঃরেখ দশমিক read_kv_f32(ধ্রুবক অক্ষর * row_base, int64_t d, int64_t nb0, পূর্ণসংখ্যা type) {
    যদি (type == GGML_TYPE_F32) {
        ফেরত *(ধ্রুবক দশমিক *) (row_base + d * nb0);
    }
    // F16
    ফেরত fp16_to_fp32(*(ধ্রুবক uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
স্থির অন্তঃরেখ দশমিক dot_qk(ধ্রুবক দশমিক * q, ধ্রুবক অক্ষর * k_row, int64_t dk, int64_t k_nb0, পূর্ণসংখ্যা k_type) {
    দশমিক acc = 0.0f;
    যদি (k_type == GGML_TYPE_F32) {
        ধ্রুবক দশমিক * kf = (ধ্রুবক দশমিক *) k_row;
        জন্য (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } অন্যথা {
        // F16 stride-aware read
        জন্য (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(ধ্রুবক uint16_t *) (k_row + i * k_nb0));
        }
    }
    ফেরত acc;
}

স্থির অন্তঃরেখ দশমিক get_mask_val(ধ্রুবক কাঠামো ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    ধ্রুবক অক্ষর * base = (ধ্রুবক অক্ষর *) mask->তথ্য + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    যদি (mask->type == GGML_TYPE_F32) {
        ফেরত *(ধ্রুবক দশমিক *) (base + ik1 * mask->nb[0]);
    }
    // F16
    ফেরত fp16_to_fp32(*(ধ্রুবক uint16_t *) (base + ik1 * mask->nb[0]));
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_flash_attn_ext_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env || !params) {
        ফেরত -1;
    }

    ধ্রুবক পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);
    যদি (thread_id < 0 || অঙ্ক_threads <= 0) {
        ফেরত 0;
    }

    কাঠামো ggml_tensor * q        = &params->src0;
    কাঠামো ggml_tensor * k        = &params->src1;
    কাঠামো ggml_tensor * v        = &params->src2;
    কাঠামো ggml_tensor * dst      = &params->dst;
    ধ্রুবক int32_t        has_mask = params->has_mask;
    কাঠামো ggml_tensor * mask     = has_mask ? &params->mask : (কাঠামো ggml_tensor *) 0;

    ধ্রুবক অক্ষর * q_তথ্য   = (ধ্রুবক অক্ষর *) q->তথ্য;
    ধ্রুবক অক্ষর * k_তথ্য   = (ধ্রুবক অক্ষর *) k->তথ্য;
    ধ্রুবক অক্ষর * v_তথ্য   = (ধ্রুবক অক্ষর *) v->তথ্য;
    অক্ষর *       dst_তথ্য = (অক্ষর *) dst->তথ্য;

    ধ্রুবক পূর্ণসংখ্যা     k_type = k->type;
    ধ্রুবক পূর্ণসংখ্যা     v_type = v->type;
    ধ্রুবক int64_t k_nb0  = k->nb[0];
    ধ্রুবক int64_t v_nb0  = v->nb[0];

    ধ্রুবক int64_t dk  = q->ne[0];  // head dim for keys/queries
    ধ্রুবক int64_t nq  = q->ne[1];  // number of query positions
    ধ্রুবক int64_t nhq = q->ne[2];  // number of query heads
    ধ্রুবক int64_t no  = q->ne[3];  // outer batch

    ধ্রুবক int64_t nk  = k->ne[1];  // number of key/value positions
    ধ্রুবক int64_t nhk = k->ne[2];  // number of kv heads
    ধ্রুবক int64_t dv  = v->ne[0];  // head dim for values

    যদি (dv > FA_DV_MAX) {
        ফেরত -1;
    }

    // GQA: query heads per kv head
    ধ্রুবক int64_t gqa_ratio = nhq / nhk;

    ধ্রুবক int64_t যোগ_rows = nq * nhq * no;
    ধ্রুবক দশমিক   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    ধ্রুবক পূর্ণসংখ্যা use_fast_store = (dv % 16 == 0);

    জন্য (int64_t row = thread_id; row < যোগ_rows; row += অঙ্ক_threads) {
        ধ্রুবক int64_t iq3 = row / (nhq * nq);
        ধ্রুবক int64_t rem = row % (nhq * nq);
        ধ্রুবক int64_t iq2 = rem / nq;  // query head index
        ধ্রুবক int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        ধ্রুবক int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        ধ্রুবক দশমিক * pq = (ধ্রুবক দশমিক *) (q_তথ্য + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        দশমিক * out = (দশমিক *) (dst_তথ্য + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        ধ্রুবক int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        ধ্রুবক int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        দশমিক acc[FA_DV_MAX];
        জন্য (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        দশমিক M = -3.402823466e+38f;
        দশমিক S = 0.0f;

        জন্য (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            দশমিক mask_val = 0.0f;
            যদি (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                যদি (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    চালু;
                }
            }

            ধ্রুবক অক্ষর * pk = k_তথ্য + ik1 * k->nb[1] + kv_base;
            ধ্রুবক অক্ষর * pv = v_তথ্য + ik1 * v->nb[1] + vv_base;

            দশমিক       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            ধ্রুবক দশমিক Mold = M;

            দশমিক ms = 1.0f;
            দশমিক vs = 1.0f;
            যদি (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                জন্য (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } অন্যথা {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            যদি (v_type == GGML_TYPE_F32) {
                ধ্রুবক দশমিক * pvf = (ধ্রুবক দশমিক *) pv;
                জন্য (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } অন্যথা {
                জন্য (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(ধ্রুবক uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        ধ্রুবক দশমিক S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        যদি (use_fast_store) {
            জন্য (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } অন্যথা {
            জন্য (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((অস্থির দশমিক *) &out[d], acc[d] * S_inv);
            }
        }
    }

    ফেরত 0;
}
