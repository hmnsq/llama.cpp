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

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_flash_attn_ext_params {
    கட்டமைப்பு ggml_tensor src0;      // Q tensor (F32)
    கட்டமைப்பு ggml_tensor src1;      // K tensor (F16 or F32)
    கட்டமைப்பு ggml_tensor src2;      // V tensor (F16 or F32)
    கட்டமைப்பு ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    கட்டமைப்பு ggml_tensor dst;       // Output tensor (F32)
    தசம              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
நிலையான உள்வரி தசம read_kv_f32(மாறிலி எழுத்து * row_base, int64_t d, int64_t nb0, முழு type) {
    எனில் (type == GGML_TYPE_F32) {
        திரும்பு *(மாறிலி தசம *) (row_base + d * nb0);
    }
    // F16
    திரும்பு fp16_to_fp32(*(மாறிலி uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
நிலையான உள்வரி தசம dot_qk(மாறிலி தசம * q, மாறிலி எழுத்து * k_row, int64_t dk, int64_t k_nb0, முழு k_type) {
    தசம acc = 0.0f;
    எனில் (k_type == GGML_TYPE_F32) {
        மாறிலி தசம * kf = (மாறிலி தசம *) k_row;
        க்கு (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } இல்லையெனில் {
        // F16 stride-aware read
        க்கு (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(மாறிலி uint16_t *) (k_row + i * k_nb0));
        }
    }
    திரும்பு acc;
}

நிலையான உள்வரி தசம get_mask_val(மாறிலி கட்டமைப்பு ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    மாறிலி எழுத்து * base = (மாறிலி எழுத்து *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    எனில் (mask->type == GGML_TYPE_F32) {
        திரும்பு *(மாறிலி தசம *) (base + ik1 * mask->nb[0]);
    }
    // F16
    திரும்பு fp16_to_fp32(*(மாறிலி uint16_t *) (base + ik1 * mask->nb[0]));
}

முழு entry_point(கட்டமைப்பு ggml_et_flash_attn_ext_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env || !params) {
        திரும்பு -1;
    }

    மாறிலி முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    மாறிலி முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);
    எனில் (thread_id < 0 || இலக்கம்_threads <= 0) {
        திரும்பு 0;
    }

    கட்டமைப்பு ggml_tensor * q        = &params->src0;
    கட்டமைப்பு ggml_tensor * k        = &params->src1;
    கட்டமைப்பு ggml_tensor * v        = &params->src2;
    கட்டமைப்பு ggml_tensor * dst      = &params->dst;
    மாறிலி int32_t        has_mask = params->has_mask;
    கட்டமைப்பு ggml_tensor * mask     = has_mask ? &params->mask : (கட்டமைப்பு ggml_tensor *) 0;

    மாறிலி எழுத்து * q_data   = (மாறிலி எழுத்து *) q->data;
    மாறிலி எழுத்து * k_data   = (மாறிலி எழுத்து *) k->data;
    மாறிலி எழுத்து * v_data   = (மாறிலி எழுத்து *) v->data;
    எழுத்து *       dst_data = (எழுத்து *) dst->data;

    மாறிலி முழு     k_type = k->type;
    மாறிலி முழு     v_type = v->type;
    மாறிலி int64_t k_nb0  = k->nb[0];
    மாறிலி int64_t v_nb0  = v->nb[0];

    மாறிலி int64_t dk  = q->ne[0];  // head dim for keys/queries
    மாறிலி int64_t nq  = q->ne[1];  // number of query positions
    மாறிலி int64_t nhq = q->ne[2];  // number of query heads
    மாறிலி int64_t no  = q->ne[3];  // outer batch

    மாறிலி int64_t nk  = k->ne[1];  // number of key/value positions
    மாறிலி int64_t nhk = k->ne[2];  // number of kv heads
    மாறிலி int64_t dv  = v->ne[0];  // head dim for values

    எனில் (dv > FA_DV_MAX) {
        திரும்பு -1;
    }

    // GQA: query heads per kv head
    மாறிலி int64_t gqa_ratio = nhq / nhk;

    மாறிலி int64_t total_rows = nq * nhq * no;
    மாறிலி தசம   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    மாறிலி முழு use_fast_store = (dv % 16 == 0);

    க்கு (int64_t row = thread_id; row < total_rows; row += இலக்கம்_threads) {
        மாறிலி int64_t iq3 = row / (nhq * nq);
        மாறிலி int64_t rem = row % (nhq * nq);
        மாறிலி int64_t iq2 = rem / nq;  // query head index
        மாறிலி int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        மாறிலி int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        மாறிலி தசம * pq = (மாறிலி தசம *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        தசம * out = (தசம *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        மாறிலி int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        மாறிலி int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        தசம acc[FA_DV_MAX];
        க்கு (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        தசம M = -3.402823466e+38f;
        தசம S = 0.0f;

        க்கு (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            தசம mask_val = 0.0f;
            எனில் (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                எனில் (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    தொடர்;
                }
            }

            மாறிலி எழுத்து * pk = k_data + ik1 * k->nb[1] + kv_base;
            மாறிலி எழுத்து * pv = v_data + ik1 * v->nb[1] + vv_base;

            தசம       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            மாறிலி தசம Mold = M;

            தசம ms = 1.0f;
            தசம vs = 1.0f;
            எனில் (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                க்கு (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } இல்லையெனில் {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            எனில் (v_type == GGML_TYPE_F32) {
                மாறிலி தசம * pvf = (மாறிலி தசம *) pv;
                க்கு (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } இல்லையெனில் {
                க்கு (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(மாறிலி uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        மாறிலி தசம S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        எனில் (use_fast_store) {
            க்கு (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } இல்லையெனில் {
            க்கு (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((நிலையற்ற தசம *) &out[d], acc[d] * S_inv);
            }
        }
    }

    திரும்பு 0;
}
