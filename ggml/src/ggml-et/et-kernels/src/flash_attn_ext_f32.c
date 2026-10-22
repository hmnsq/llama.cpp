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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_flash_attn_ext_params {
    ಸಂರಚನಾ ggml_tensor src0;      // Q tensor (F32)
    ಸಂರಚನಾ ggml_tensor src1;      // K tensor (F16 or F32)
    ಸಂರಚನಾ ggml_tensor src2;      // V tensor (F16 or F32)
    ಸಂರಚನಾ ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    ಸಂರಚನಾ ggml_tensor dst;       // Output tensor (F32)
    ಭಿನ್ನ              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ read_kv_f32(ಧ್ರುವ ವರ್ಣ * row_base, int64_t d, int64_t nb0, ಪೂರ್ಣಾಙ್ಕ type) {
    ಯದಿ (type == GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ *(ಧ್ರುವ ಭಿನ್ನ *) (row_base + d * nb0);
    }
    // F16
    ಪ್ರತ್ಯಾಗಮ fp16_to_fp32(*(ಧ್ರುವ uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ dot_qk(ಧ್ರುವ ಭಿನ್ನ * q, ಧ್ರುವ ವರ್ಣ * k_row, int64_t dk, int64_t k_nb0, ಪೂರ್ಣಾಙ್ಕ k_type) {
    ಭಿನ್ನ acc = 0.0f;
    ಯದಿ (k_type == GGML_TYPE_F32) {
        ಧ್ರುವ ಭಿನ್ನ * kf = (ಧ್ರುವ ಭಿನ್ನ *) k_row;
        ಪ್ರತಿ (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } ಅನ್ಯಥಾ {
        // F16 stride-aware read
        ಪ್ರತಿ (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(ಧ್ರುವ uint16_t *) (k_row + i * k_nb0));
        }
    }
    ಪ್ರತ್ಯಾಗಮ acc;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ get_mask_val(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    ಧ್ರುವ ವರ್ಣ * base = (ಧ್ರುವ ವರ್ಣ *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    ಯದಿ (mask->type == GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ *(ಧ್ರುವ ಭಿನ್ನ *) (base + ik1 * mask->nb[0]);
    }
    // F16
    ಪ್ರತ್ಯಾಗಮ fp16_to_fp32(*(ಧ್ರುವ uint16_t *) (base + ik1 * mask->nb[0]));
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_flash_attn_ext_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env || !params) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);
    ಯದಿ (thread_id < 0 || ಅಂಕೆ_threads <= 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಸಂರಚನಾ ggml_tensor * q        = &params->src0;
    ಸಂರಚನಾ ggml_tensor * k        = &params->src1;
    ಸಂರಚನಾ ggml_tensor * v        = &params->src2;
    ಸಂರಚನಾ ggml_tensor * dst      = &params->dst;
    ಧ್ರುವ int32_t        has_mask = params->has_mask;
    ಸಂರಚನಾ ggml_tensor * mask     = has_mask ? &params->mask : (ಸಂರಚನಾ ggml_tensor *) 0;

    ಧ್ರುವ ವರ್ಣ * q_data   = (ಧ್ರುವ ವರ್ಣ *) q->data;
    ಧ್ರುವ ವರ್ಣ * k_data   = (ಧ್ರುವ ವರ್ಣ *) k->data;
    ಧ್ರುವ ವರ್ಣ * v_data   = (ಧ್ರುವ ವರ್ಣ *) v->data;
    ವರ್ಣ *       dst_data = (ವರ್ಣ *) dst->data;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     k_type = k->type;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     v_type = v->type;
    ಧ್ರುವ int64_t k_nb0  = k->nb[0];
    ಧ್ರುವ int64_t v_nb0  = v->nb[0];

    ಧ್ರುವ int64_t dk  = q->ne[0];  // head dim for keys/queries
    ಧ್ರುವ int64_t nq  = q->ne[1];  // number of query positions
    ಧ್ರುವ int64_t nhq = q->ne[2];  // number of query heads
    ಧ್ರುವ int64_t no  = q->ne[3];  // outer batch

    ಧ್ರುವ int64_t nk  = k->ne[1];  // number of key/value positions
    ಧ್ರುವ int64_t nhk = k->ne[2];  // number of kv heads
    ಧ್ರುವ int64_t dv  = v->ne[0];  // head dim for values

    ಯದಿ (dv > FA_DV_MAX) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    // GQA: query heads per kv head
    ಧ್ರುವ int64_t gqa_ratio = nhq / nhk;

    ಧ್ರುವ int64_t total_rows = nq * nhq * no;
    ಧ್ರುವ ಭಿನ್ನ   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ use_fast_store = (dv % 16 == 0);

    ಪ್ರತಿ (int64_t row = thread_id; row < total_rows; row += ಅಂಕೆ_threads) {
        ಧ್ರುವ int64_t iq3 = row / (nhq * nq);
        ಧ್ರುವ int64_t rem = row % (nhq * nq);
        ಧ್ರುವ int64_t iq2 = rem / nq;  // query head index
        ಧ್ರುವ int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        ಧ್ರುವ int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        ಧ್ರುವ ಭಿನ್ನ * pq = (ಧ್ರುವ ಭಿನ್ನ *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        ಭಿನ್ನ * out = (ಭಿನ್ನ *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        ಧ್ರುವ int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        ಧ್ರುವ int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        ಭಿನ್ನ acc[FA_DV_MAX];
        ಪ್ರತಿ (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        ಭಿನ್ನ M = -3.402823466e+38f;
        ಭಿನ್ನ S = 0.0f;

        ಪ್ರತಿ (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            ಭಿನ್ನ mask_val = 0.0f;
            ಯದಿ (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                ಯದಿ (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    ಅನುವರ್ತ;
                }
            }

            ಧ್ರುವ ವರ್ಣ * pk = k_data + ik1 * k->nb[1] + kv_base;
            ಧ್ರುವ ವರ್ಣ * pv = v_data + ik1 * v->nb[1] + vv_base;

            ಭಿನ್ನ       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            ಧ್ರುವ ಭಿನ್ನ Mold = M;

            ಭಿನ್ನ ms = 1.0f;
            ಭಿನ್ನ vs = 1.0f;
            ಯದಿ (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                ಪ್ರತಿ (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } ಅನ್ಯಥಾ {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            ಯದಿ (v_type == GGML_TYPE_F32) {
                ಧ್ರುವ ಭಿನ್ನ * pvf = (ಧ್ರುವ ಭಿನ್ನ *) pv;
                ಪ್ರತಿ (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } ಅನ್ಯಥಾ {
                ಪ್ರತಿ (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(ಧ್ರುವ uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        ಧ್ರುವ ಭಿನ್ನ S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        ಯದಿ (use_fast_store) {
            ಪ್ರತಿ (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } ಅನ್ಯಥಾ {
            ಪ್ರತಿ (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((ಚಞ್ಚಲ ಭಿನ್ನ *) &out[d], acc[d] * S_inv);
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
