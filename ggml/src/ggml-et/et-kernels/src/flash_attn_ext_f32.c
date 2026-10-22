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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

संरचना ggml_et_flash_attn_ext_params {
    संरचना ggml_tensor src0;      // Q tensor (F32)
    संरचना ggml_tensor src1;      // K tensor (F16 or F32)
    संरचना ggml_tensor src2;      // V tensor (F16 or F32)
    संरचना ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    संरचना ggml_tensor dst;       // Output tensor (F32)
    दशांश              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
स्थिर ओळीत दशांश read_kv_f32(स्थिरांक अक्षर * row_base, int64_t d, int64_t nb0, पूर्णांक type) {
    जर (type == GGML_TYPE_F32) {
        परत *(स्थिरांक दशांश *) (row_base + d * nb0);
    }
    // F16
    परत fp16_to_fp32(*(स्थिरांक uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
स्थिर ओळीत दशांश dot_qk(स्थिरांक दशांश * q, स्थिरांक अक्षर * k_row, int64_t dk, int64_t k_nb0, पूर्णांक k_type) {
    दशांश acc = 0.0f;
    जर (k_type == GGML_TYPE_F32) {
        स्थिरांक दशांश * kf = (स्थिरांक दशांश *) k_row;
        साठी (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } अन्यथा {
        // F16 stride-aware read
        साठी (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(स्थिरांक uint16_t *) (k_row + i * k_nb0));
        }
    }
    परत acc;
}

स्थिर ओळीत दशांश get_mask_val(स्थिरांक संरचना ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    स्थिरांक अक्षर * base = (स्थिरांक अक्षर *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    जर (mask->type == GGML_TYPE_F32) {
        परत *(स्थिरांक दशांश *) (base + ik1 * mask->nb[0]);
    }
    // F16
    परत fp16_to_fp32(*(स्थिरांक uint16_t *) (base + ik1 * mask->nb[0]));
}

पूर्णांक entry_point(संरचना ggml_et_flash_attn_ext_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env || !params) {
        परत -1;
    }

    स्थिरांक पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    स्थिरांक पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);
    जर (thread_id < 0 || अंक_threads <= 0) {
        परत 0;
    }

    संरचना ggml_tensor * q        = &params->src0;
    संरचना ggml_tensor * k        = &params->src1;
    संरचना ggml_tensor * v        = &params->src2;
    संरचना ggml_tensor * dst      = &params->dst;
    स्थिरांक int32_t        has_mask = params->has_mask;
    संरचना ggml_tensor * mask     = has_mask ? &params->mask : (संरचना ggml_tensor *) 0;

    स्थिरांक अक्षर * q_data   = (स्थिरांक अक्षर *) q->data;
    स्थिरांक अक्षर * k_data   = (स्थिरांक अक्षर *) k->data;
    स्थिरांक अक्षर * v_data   = (स्थिरांक अक्षर *) v->data;
    अक्षर *       dst_data = (अक्षर *) dst->data;

    स्थिरांक पूर्णांक     k_type = k->type;
    स्थिरांक पूर्णांक     v_type = v->type;
    स्थिरांक int64_t k_nb0  = k->nb[0];
    स्थिरांक int64_t v_nb0  = v->nb[0];

    स्थिरांक int64_t dk  = q->ne[0];  // head dim for keys/queries
    स्थिरांक int64_t nq  = q->ne[1];  // number of query positions
    स्थिरांक int64_t nhq = q->ne[2];  // number of query heads
    स्थिरांक int64_t no  = q->ne[3];  // outer batch

    स्थिरांक int64_t nk  = k->ne[1];  // number of key/value positions
    स्थिरांक int64_t nhk = k->ne[2];  // number of kv heads
    स्थिरांक int64_t dv  = v->ne[0];  // head dim for values

    जर (dv > FA_DV_MAX) {
        परत -1;
    }

    // GQA: query heads per kv head
    स्थिरांक int64_t gqa_ratio = nhq / nhk;

    स्थिरांक int64_t total_rows = nq * nhq * no;
    स्थिरांक दशांश   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    स्थिरांक पूर्णांक use_fast_store = (dv % 16 == 0);

    साठी (int64_t row = thread_id; row < total_rows; row += अंक_threads) {
        स्थिरांक int64_t iq3 = row / (nhq * nq);
        स्थिरांक int64_t rem = row % (nhq * nq);
        स्थिरांक int64_t iq2 = rem / nq;  // query head index
        स्थिरांक int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        स्थिरांक int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        स्थिरांक दशांश * pq = (स्थिरांक दशांश *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        दशांश * out = (दशांश *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        स्थिरांक int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        स्थिरांक int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        दशांश acc[FA_DV_MAX];
        साठी (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        दशांश M = -3.402823466e+38f;
        दशांश S = 0.0f;

        साठी (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            दशांश mask_val = 0.0f;
            जर (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                जर (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    सुरूठेवा;
                }
            }

            स्थिरांक अक्षर * pk = k_data + ik1 * k->nb[1] + kv_base;
            स्थिरांक अक्षर * pv = v_data + ik1 * v->nb[1] + vv_base;

            दशांश       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            स्थिरांक दशांश Mold = M;

            दशांश ms = 1.0f;
            दशांश vs = 1.0f;
            जर (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                साठी (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } अन्यथा {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            जर (v_type == GGML_TYPE_F32) {
                स्थिरांक दशांश * pvf = (स्थिरांक दशांश *) pv;
                साठी (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } अन्यथा {
                साठी (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(स्थिरांक uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        स्थिरांक दशांश S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        जर (use_fast_store) {
            साठी (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } अन्यथा {
            साठी (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((अस्थिर दशांश *) &out[d], acc[d] * S_inv);
            }
        }
    }

    परत 0;
}
