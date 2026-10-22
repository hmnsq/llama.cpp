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

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdbool.h>
#ضمن <stdint.h>

بنية ggml_et_flash_attn_ext_params {
    بنية ggml_tensor src0;      // Q tensor (F32)
    بنية ggml_tensor src1;      // K tensor (F16 or F32)
    بنية ggml_tensor src2;      // V tensor (F16 or F32)
    بنية ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    بنية ggml_tensor dst;       // Output tensor (F32)
    عائم              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
ساكن مضمن عائم read_kv_f32(ثابت حرف * row_base, int64_t d, int64_t nb0, صحيح type) {
    إذا (type == GGML_TYPE_F32) {
        أرجع *(ثابت عائم *) (row_base + d * nb0);
    }
    // F16
    أرجع fp16_to_fp32(*(ثابت uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
ساكن مضمن عائم dot_qk(ثابت عائم * q, ثابت حرف * k_row, int64_t dk, int64_t k_nb0, صحيح k_type) {
    عائم acc = 0.0f;
    إذا (k_type == GGML_TYPE_F32) {
        ثابت عائم * kf = (ثابت عائم *) k_row;
        لكل (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } وإلا {
        // F16 stride-aware read
        لكل (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(ثابت uint16_t *) (k_row + i * k_nb0));
        }
    }
    أرجع acc;
}

ساكن مضمن عائم get_mask_val(ثابت بنية ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    ثابت حرف * base = (ثابت حرف *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    إذا (mask->type == GGML_TYPE_F32) {
        أرجع *(ثابت عائم *) (base + ik1 * mask->nb[0]);
    }
    // F16
    أرجع fp16_to_fp32(*(ثابت uint16_t *) (base + ik1 * mask->nb[0]));
}

صحيح entry_point(بنية ggml_et_flash_attn_ext_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env || !params) {
        أرجع -1;
    }

    ثابت صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ثابت صحيح ژمارە_threads = get_ژمارە_threads(kernel_env->shire_mask);
    إذا (thread_id < 0 || ژمارە_threads <= 0) {
        أرجع 0;
    }

    بنية ggml_tensor * q        = &params->src0;
    بنية ggml_tensor * k        = &params->src1;
    بنية ggml_tensor * v        = &params->src2;
    بنية ggml_tensor * dst      = &params->dst;
    ثابت int32_t        has_mask = params->has_mask;
    بنية ggml_tensor * mask     = has_mask ? &params->mask : (بنية ggml_tensor *) 0;

    ثابت حرف * q_data   = (ثابت حرف *) q->data;
    ثابت حرف * k_data   = (ثابت حرف *) k->data;
    ثابت حرف * v_data   = (ثابت حرف *) v->data;
    حرف *       dst_data = (حرف *) dst->data;

    ثابت صحيح     k_type = k->type;
    ثابت صحيح     v_type = v->type;
    ثابت int64_t k_nb0  = k->nb[0];
    ثابت int64_t v_nb0  = v->nb[0];

    ثابت int64_t dk  = q->ne[0];  // head dim for keys/queries
    ثابت int64_t nq  = q->ne[1];  // number of query positions
    ثابت int64_t nhq = q->ne[2];  // number of query heads
    ثابت int64_t no  = q->ne[3];  // outer batch

    ثابت int64_t nk  = k->ne[1];  // number of key/value positions
    ثابت int64_t nhk = k->ne[2];  // number of kv heads
    ثابت int64_t dv  = v->ne[0];  // head dim for values

    إذا (dv > FA_DV_MAX) {
        أرجع -1;
    }

    // GQA: query heads per kv head
    ثابت int64_t gqa_ratio = nhq / nhk;

    ثابت int64_t total_rows = nq * nhq * no;
    ثابت عائم   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    ثابت صحيح use_fast_store = (dv % 16 == 0);

    لكل (int64_t row = thread_id; row < total_rows; row += ژمارە_threads) {
        ثابت int64_t iq3 = row / (nhq * nq);
        ثابت int64_t rem = row % (nhq * nq);
        ثابت int64_t iq2 = rem / nq;  // query head index
        ثابت int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        ثابت int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        ثابت عائم * pq = (ثابت عائم *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        عائم * out = (عائم *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        ثابت int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        ثابت int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        عائم acc[FA_DV_MAX];
        لكل (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        عائم M = -3.402823466e+38f;
        عائم S = 0.0f;

        لكل (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            عائم mask_val = 0.0f;
            إذا (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                إذا (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    استمر;
                }
            }

            ثابت حرف * pk = k_data + ik1 * k->nb[1] + kv_base;
            ثابت حرف * pv = v_data + ik1 * v->nb[1] + vv_base;

            عائم       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            ثابت عائم Mold = M;

            عائم ms = 1.0f;
            عائم vs = 1.0f;
            إذا (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                لكل (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } وإلا {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            إذا (v_type == GGML_TYPE_F32) {
                ثابت عائم * pvf = (ثابت عائم *) pv;
                لكل (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } وإلا {
                لكل (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(ثابت uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        ثابت عائم S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        إذا (use_fast_store) {
            لكل (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } وإلا {
            لكل (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((متطاير عائم *) &out[d], acc[d] * S_inv);
            }
        }
    }

    أرجع 0;
}
