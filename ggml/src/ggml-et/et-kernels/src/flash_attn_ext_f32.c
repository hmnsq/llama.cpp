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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_flash_attn_ext_params {
    સંરચના ggml_tensor src0;      // Q tensor (F32)
    સંરચના ggml_tensor src1;      // K tensor (F16 or F32)
    સંરચના ggml_tensor src2;      // V tensor (F16 or F32)
    સંરચના ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    સંરચના ggml_tensor dst;       // Output tensor (F32)
    દશાંશ              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
સ્થિર અંતઃરેખ દશાંશ read_kv_f32(અચલ અક્ષર * row_base, int64_t d, int64_t nb0, પૂર્ણાંક type) {
    જો (type == GGML_TYPE_F32) {
        પાછા *(અચલ દશાંશ *) (row_base + d * nb0);
    }
    // F16
    પાછા fp16_to_fp32(*(અચલ uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
સ્થિર અંતઃરેખ દશાંશ dot_qk(અચલ દશાંશ * q, અચલ અક્ષર * k_row, int64_t dk, int64_t k_nb0, પૂર્ણાંક k_type) {
    દશાંશ acc = 0.0f;
    જો (k_type == GGML_TYPE_F32) {
        અચલ દશાંશ * kf = (અચલ દશાંશ *) k_row;
        માટે (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } નહીંતર {
        // F16 stride-aware read
        માટે (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(અચલ uint16_t *) (k_row + i * k_nb0));
        }
    }
    પાછા acc;
}

સ્થિર અંતઃરેખ દશાંશ get_mask_val(અચલ સંરચના ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    અચલ અક્ષર * base = (અચલ અક્ષર *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    જો (mask->type == GGML_TYPE_F32) {
        પાછા *(અચલ દશાંશ *) (base + ik1 * mask->nb[0]);
    }
    // F16
    પાછા fp16_to_fp32(*(અચલ uint16_t *) (base + ik1 * mask->nb[0]));
}

પૂર્ણાંક entry_point(સંરચના ggml_et_flash_attn_ext_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env || !params) {
        પાછા -1;
    }

    અચલ પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    અચલ પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);
    જો (thread_id < 0 || અંક_threads <= 0) {
        પાછા 0;
    }

    સંરચના ggml_tensor * q        = &params->src0;
    સંરચના ggml_tensor * k        = &params->src1;
    સંરચના ggml_tensor * v        = &params->src2;
    સંરચના ggml_tensor * dst      = &params->dst;
    અચલ int32_t        has_mask = params->has_mask;
    સંરચના ggml_tensor * mask     = has_mask ? &params->mask : (સંરચના ggml_tensor *) 0;

    અચલ અક્ષર * q_data   = (અચલ અક્ષર *) q->data;
    અચલ અક્ષર * k_data   = (અચલ અક્ષર *) k->data;
    અચલ અક્ષર * v_data   = (અચલ અક્ષર *) v->data;
    અક્ષર *       dst_data = (અક્ષર *) dst->data;

    અચલ પૂર્ણાંક     k_type = k->type;
    અચલ પૂર્ણાંક     v_type = v->type;
    અચલ int64_t k_nb0  = k->nb[0];
    અચલ int64_t v_nb0  = v->nb[0];

    અચલ int64_t dk  = q->ne[0];  // head dim for keys/queries
    અચલ int64_t nq  = q->ne[1];  // number of query positions
    અચલ int64_t nhq = q->ne[2];  // number of query heads
    અચલ int64_t no  = q->ne[3];  // outer batch

    અચલ int64_t nk  = k->ne[1];  // number of key/value positions
    અચલ int64_t nhk = k->ne[2];  // number of kv heads
    અચલ int64_t dv  = v->ne[0];  // head dim for values

    જો (dv > FA_DV_MAX) {
        પાછા -1;
    }

    // GQA: query heads per kv head
    અચલ int64_t gqa_ratio = nhq / nhk;

    અચલ int64_t total_rows = nq * nhq * no;
    અચલ દશાંશ   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    અચલ પૂર્ણાંક use_fast_store = (dv % 16 == 0);

    માટે (int64_t row = thread_id; row < total_rows; row += અંક_threads) {
        અચલ int64_t iq3 = row / (nhq * nq);
        અચલ int64_t rem = row % (nhq * nq);
        અચલ int64_t iq2 = rem / nq;  // query head index
        અચલ int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        અચલ int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        અચલ દશાંશ * pq = (અચલ દશાંશ *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        દશાંશ * out = (દશાંશ *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        અચલ int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        અચલ int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        દશાંશ acc[FA_DV_MAX];
        માટે (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        દશાંશ M = -3.402823466e+38f;
        દશાંશ S = 0.0f;

        માટે (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            દશાંશ mask_val = 0.0f;
            જો (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                જો (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    ચાલુ;
                }
            }

            અચલ અક્ષર * pk = k_data + ik1 * k->nb[1] + kv_base;
            અચલ અક્ષર * pv = v_data + ik1 * v->nb[1] + vv_base;

            દશાંશ       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            અચલ દશાંશ Mold = M;

            દશાંશ ms = 1.0f;
            દશાંશ vs = 1.0f;
            જો (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                માટે (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } નહીંતર {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            જો (v_type == GGML_TYPE_F32) {
                અચલ દશાંશ * pvf = (અચલ દશાંશ *) pv;
                માટે (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } નહીંતર {
                માટે (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(અચલ uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        અચલ દશાંશ S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        જો (use_fast_store) {
            માટે (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } નહીંતર {
            માટે (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((અસ્થિર દશાંશ *) &out[d], acc[d] * S_inv);
            }
        }
    }

    પાછા 0;
}
