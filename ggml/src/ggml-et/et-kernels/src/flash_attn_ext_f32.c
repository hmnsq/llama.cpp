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

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <stdbool.h>
#समावेश <stdint.h>

काष्ठा ggml_et_flash_attn_ext_params {
    काष्ठा ggml_tensor src0;      // Q tensor (F32)
    काष्ठा ggml_tensor src1;      // K tensor (F16 or F32)
    काष्ठा ggml_tensor src2;      // V tensor (F16 or F32)
    काष्ठा ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    काष्ठा ggml_tensor dst;       // Output tensor (F32)
    भग्न              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#घोषणा FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
अटल अंतरभूत भग्न read_kv_f32(स्थिर अक्षर * row_base, int64_t d, int64_t nb0, पूर्णांक type) {
    अगर (type == GGML_TYPE_F32) {
        वापस *(स्थिर भग्न *) (row_base + d * nb0);
    }
    // F16
    वापस fp16_to_fp32(*(स्थिर uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
अटल अंतरभूत भग्न dot_qk(स्थिर भग्न * q, स्थिर अक्षर * k_row, int64_t dk, int64_t k_nb0, पूर्णांक k_type) {
    भग्न acc = 0.0f;
    अगर (k_type == GGML_TYPE_F32) {
        स्थिर भग्न * kf = (स्थिर भग्न *) k_row;
        क्रम (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } अन्यथा {
        // F16 stride-aware read
        क्रम (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(स्थिर uint16_t *) (k_row + i * k_nb0));
        }
    }
    वापस acc;
}

अटल अंतरभूत भग्न get_mask_val(स्थिर काष्ठा ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    स्थिर अक्षर * base = (स्थिर अक्षर *) mask->तथ्य + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    अगर (mask->type == GGML_TYPE_F32) {
        वापस *(स्थिर भग्न *) (base + ik1 * mask->nb[0]);
    }
    // F16
    वापस fp16_to_fp32(*(स्थिर uint16_t *) (base + ik1 * mask->nb[0]));
}

पूर्णांक entry_point(काष्ठा ggml_et_flash_attn_ext_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env || !params) {
        वापस -1;
    }

    स्थिर पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    स्थिर पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);
    अगर (thread_id < 0 || अंक_threads <= 0) {
        वापस 0;
    }

    काष्ठा ggml_tensor * q        = &params->src0;
    काष्ठा ggml_tensor * k        = &params->src1;
    काष्ठा ggml_tensor * v        = &params->src2;
    काष्ठा ggml_tensor * dst      = &params->dst;
    स्थिर int32_t        has_mask = params->has_mask;
    काष्ठा ggml_tensor * mask     = has_mask ? &params->mask : (काष्ठा ggml_tensor *) 0;

    स्थिर अक्षर * q_तथ्य   = (स्थिर अक्षर *) q->तथ्य;
    स्थिर अक्षर * k_तथ्य   = (स्थिर अक्षर *) k->तथ्य;
    स्थिर अक्षर * v_तथ्य   = (स्थिर अक्षर *) v->तथ्य;
    अक्षर *       dst_तथ्य = (अक्षर *) dst->तथ्य;

    स्थिर पूर्णांक     k_type = k->type;
    स्थिर पूर्णांक     v_type = v->type;
    स्थिर int64_t k_nb0  = k->nb[0];
    स्थिर int64_t v_nb0  = v->nb[0];

    स्थिर int64_t dk  = q->ne[0];  // head dim for keys/queries
    स्थिर int64_t nq  = q->ne[1];  // number of query positions
    स्थिर int64_t nhq = q->ne[2];  // number of query heads
    स्थिर int64_t no  = q->ne[3];  // outer batch

    स्थिर int64_t nk  = k->ne[1];  // number of key/value positions
    स्थिर int64_t nhk = k->ne[2];  // number of kv heads
    स्थिर int64_t dv  = v->ne[0];  // head dim for values

    अगर (dv > FA_DV_MAX) {
        वापस -1;
    }

    // GQA: query heads per kv head
    स्थिर int64_t gqa_ratio = nhq / nhk;

    स्थिर int64_t योग_rows = nq * nhq * no;
    स्थिर भग्न   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    स्थिर पूर्णांक use_fast_store = (dv % 16 == 0);

    क्रम (int64_t row = thread_id; row < योग_rows; row += अंक_threads) {
        स्थिर int64_t iq3 = row / (nhq * nq);
        स्थिर int64_t rem = row % (nhq * nq);
        स्थिर int64_t iq2 = rem / nq;  // query head index
        स्थिर int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        स्थिर int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        स्थिर भग्न * pq = (स्थिर भग्न *) (q_तथ्य + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        भग्न * out = (भग्न *) (dst_तथ्य + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        स्थिर int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        स्थिर int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        भग्न acc[FA_DV_MAX];
        क्रम (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        भग्न M = -3.402823466e+38f;
        भग्न S = 0.0f;

        क्रम (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            भग्न mask_val = 0.0f;
            अगर (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                अगर (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    जारी;
                }
            }

            स्थिर अक्षर * pk = k_तथ्य + ik1 * k->nb[1] + kv_base;
            स्थिर अक्षर * pv = v_तथ्य + ik1 * v->nb[1] + vv_base;

            भग्न       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            स्थिर भग्न Mold = M;

            भग्न ms = 1.0f;
            भग्न vs = 1.0f;
            अगर (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                क्रम (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } अन्यथा {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            अगर (v_type == GGML_TYPE_F32) {
                स्थिर भग्न * pvf = (स्थिर भग्न *) pv;
                क्रम (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } अन्यथा {
                क्रम (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(स्थिर uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        स्थिर भग्न S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        अगर (use_fast_store) {
            क्रम (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } अन्यथा {
            क्रम (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((अस्थिर भग्न *) &out[d], acc[d] * S_inv);
            }
        }
    }

    वापस 0;
}
