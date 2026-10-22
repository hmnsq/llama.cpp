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

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_flash_attn_ext_params {
    ସଂରଚନା ggml_tensor src0;      // Q tensor (F32)
    ସଂରଚନା ggml_tensor src1;      // K tensor (F16 or F32)
    ସଂରଚନା ggml_tensor src2;      // V tensor (F16 or F32)
    ସଂରଚନା ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    ସଂରଚନା ggml_tensor dst;       // Output tensor (F32)
    ଦଶମିକ              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ read_kv_f32(ଅଚଳ ଅକ୍ଷର * row_base, int64_t d, int64_t nb0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା type) {
    ଯଦି (type == GGML_TYPE_F32) {
        ଫେରନ୍ତୁ *(ଅଚଳ ଦଶମିକ *) (row_base + d * nb0);
    }
    // F16
    ଫେରନ୍ତୁ fp16_to_fp32(*(ଅଚଳ uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ dot_qk(ଅଚଳ ଦଶମିକ * q, ଅଚଳ ଅକ୍ଷର * k_row, int64_t dk, int64_t k_nb0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_type) {
    ଦଶମିକ acc = 0.0f;
    ଯଦି (k_type == GGML_TYPE_F32) {
        ଅଚଳ ଦଶମିକ * kf = (ଅଚଳ ଦଶମିକ *) k_row;
        ପାଇଁ (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } ଅନ୍ୟଥା {
        // F16 stride-aware read
        ପାଇଁ (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(ଅଚଳ uint16_t *) (k_row + i * k_nb0));
        }
    }
    ଫେରନ୍ତୁ acc;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ get_mask_val(ଅଚଳ ସଂରଚନା ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    ଅଚଳ ଅକ୍ଷର * base = (ଅଚଳ ଅକ୍ଷର *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    ଯଦି (mask->type == GGML_TYPE_F32) {
        ଫେରନ୍ତୁ *(ଅଚଳ ଦଶମିକ *) (base + ik1 * mask->nb[0]);
    }
    // F16
    ଫେରନ୍ତୁ fp16_to_fp32(*(ଅଚଳ uint16_t *) (base + ik1 * mask->nb[0]));
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_flash_attn_ext_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env || !params) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);
    ଯଦି (thread_id < 0 || ଅଙ୍କ_threads <= 0) {
        ଫେରନ୍ତୁ 0;
    }

    ସଂରଚନା ggml_tensor * q        = &params->src0;
    ସଂରଚନା ggml_tensor * k        = &params->src1;
    ସଂରଚନା ggml_tensor * v        = &params->src2;
    ସଂରଚନା ggml_tensor * dst      = &params->dst;
    ଅଚଳ int32_t        has_mask = params->has_mask;
    ସଂରଚନା ggml_tensor * mask     = has_mask ? &params->mask : (ସଂରଚନା ggml_tensor *) 0;

    ଅଚଳ ଅକ୍ଷର * q_data   = (ଅଚଳ ଅକ୍ଷର *) q->data;
    ଅଚଳ ଅକ୍ଷର * k_data   = (ଅଚଳ ଅକ୍ଷର *) k->data;
    ଅଚଳ ଅକ୍ଷର * v_data   = (ଅଚଳ ଅକ୍ଷର *) v->data;
    ଅକ୍ଷର *       dst_data = (ଅକ୍ଷର *) dst->data;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     k_type = k->type;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     v_type = v->type;
    ଅଚଳ int64_t k_nb0  = k->nb[0];
    ଅଚଳ int64_t v_nb0  = v->nb[0];

    ଅଚଳ int64_t dk  = q->ne[0];  // head dim for keys/queries
    ଅଚଳ int64_t nq  = q->ne[1];  // number of query positions
    ଅଚଳ int64_t nhq = q->ne[2];  // number of query heads
    ଅଚଳ int64_t no  = q->ne[3];  // outer batch

    ଅଚଳ int64_t nk  = k->ne[1];  // number of key/value positions
    ଅଚଳ int64_t nhk = k->ne[2];  // number of kv heads
    ଅଚଳ int64_t dv  = v->ne[0];  // head dim for values

    ଯଦି (dv > FA_DV_MAX) {
        ଫେରନ୍ତୁ -1;
    }

    // GQA: query heads per kv head
    ଅଚଳ int64_t gqa_ratio = nhq / nhk;

    ଅଚଳ int64_t total_rows = nq * nhq * no;
    ଅଚଳ ଦଶମିକ   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା use_fast_store = (dv % 16 == 0);

    ପାଇଁ (int64_t row = thread_id; row < total_rows; row += ଅଙ୍କ_threads) {
        ଅଚଳ int64_t iq3 = row / (nhq * nq);
        ଅଚଳ int64_t rem = row % (nhq * nq);
        ଅଚଳ int64_t iq2 = rem / nq;  // query head index
        ଅଚଳ int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        ଅଚଳ int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        ଅଚଳ ଦଶମିକ * pq = (ଅଚଳ ଦଶମିକ *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        ଦଶମିକ * out = (ଦଶମିକ *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        ଅଚଳ int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        ଅଚଳ int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        ଦଶମିକ acc[FA_DV_MAX];
        ପାଇଁ (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        ଦଶମିକ M = -3.402823466e+38f;
        ଦଶମିକ S = 0.0f;

        ପାଇଁ (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            ଦଶମିକ mask_val = 0.0f;
            ଯଦି (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                ଯଦି (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    ଜାରି;
                }
            }

            ଅଚଳ ଅକ୍ଷର * pk = k_data + ik1 * k->nb[1] + kv_base;
            ଅଚଳ ଅକ୍ଷର * pv = v_data + ik1 * v->nb[1] + vv_base;

            ଦଶମିକ       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            ଅଚଳ ଦଶମିକ Mold = M;

            ଦଶମିକ ms = 1.0f;
            ଦଶମିକ vs = 1.0f;
            ଯଦି (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                ପାଇଁ (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } ଅନ୍ୟଥା {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            ଯଦି (v_type == GGML_TYPE_F32) {
                ଅଚଳ ଦଶମିକ * pvf = (ଅଚଳ ଦଶମିକ *) pv;
                ପାଇଁ (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } ଅନ୍ୟଥା {
                ପାଇଁ (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(ଅଚଳ uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        ଅଚଳ ଦଶମିକ S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        ଯଦି (use_fast_store) {
            ପାଇଁ (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((ଅସ୍ଥିର ଦଶମିକ *) &out[d], acc[d] * S_inv);
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
