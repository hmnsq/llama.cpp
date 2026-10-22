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
    संरचना ggml_tensor आवरणम्;      // mask tensor (F16 or F32), zeroed when absent
    संरचना ggml_tensor dst;       // Output tensor (F32)
    भिन्न              scale;     // Scale factor applied to QK
    int32_t            has_आवरणम्;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#परिभाषय FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
स्थिर अन्तःस्थ भिन्न read_kv_f32(ध्रुव वर्ण * row_आधारः, int64_t d, int64_t nb0, पूर्णाङ्क type) {
    यदि (type == GGML_TYPE_F32) {
        प्रत्यागम *(ध्रुव भिन्न *) (row_आधारः + d * nb0);
    }
    // F16
    प्रत्यागम fp16_गन्तव्यम्_fp32(*(ध्रुव uint16_t *) (row_आधारः + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
स्थिर अन्तःस्थ भिन्न dot_qk(ध्रुव भिन्न * q, ध्रुव वर्ण * k_row, int64_t dk, int64_t k_nb0, पूर्णाङ्क k_type) {
    भिन्न acc = 0.0f;
    यदि (k_type == GGML_TYPE_F32) {
        ध्रुव भिन्न * kf = (ध्रुव भिन्न *) k_row;
        प्रति (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } अन्यथा {
        // F16 stride-aware read
        प्रति (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_गन्तव्यम्_fp32(*(ध्रुव uint16_t *) (k_row + i * k_nb0));
        }
    }
    प्रत्यागम acc;
}

स्थिर अन्तःस्थ भिन्न get_आवरणम्_val(ध्रुव संरचना ggml_tensor * आवरणम्, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    ध्रुव वर्ण * आधारः = (ध्रुव वर्ण *) आवरणम्->दत्त + iq1 * आवरणम्->nb[1] + (iq2 % आवरणम्->ne[2]) * आवरणम्->nb[2] +
                        (iq3 % आवरणम्->ne[3]) * आवरणम्->nb[3];

    यदि (आवरणम्->type == GGML_TYPE_F32) {
        प्रत्यागम *(ध्रुव भिन्न *) (आधारः + ik1 * आवरणम्->nb[0]);
    }
    // F16
    प्रत्यागम fp16_गन्तव्यम्_fp32(*(ध्रुव uint16_t *) (आधारः + ik1 * आवरणम्->nb[0]));
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_flash_attn_ext_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env || !params) {
        प्रत्यागम -1;
    }

    ध्रुव पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    ध्रुव पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);
    यदि (thread_id < 0 || अङ्कः_तन्तवः <= 0) {
        प्रत्यागम 0;
    }

    संरचना ggml_tensor * q        = &params->src0;
    संरचना ggml_tensor * k        = &params->src1;
    संरचना ggml_tensor * v        = &params->src2;
    संरचना ggml_tensor * dst      = &params->dst;
    ध्रुव int32_t        has_आवरणम् = params->has_आवरणम्;
    संरचना ggml_tensor * आवरणम्     = has_आवरणम् ? &params->आवरणम् : (संरचना ggml_tensor *) 0;

    ध्रुव वर्ण * q_दत्त   = (ध्रुव वर्ण *) q->दत्त;
    ध्रुव वर्ण * k_दत्त   = (ध्रुव वर्ण *) k->दत्त;
    ध्रुव वर्ण * v_दत्त   = (ध्रुव वर्ण *) v->दत्त;
    वर्ण *       dst_दत्त = (वर्ण *) dst->दत्त;

    ध्रुव पूर्णाङ्क     k_type = k->type;
    ध्रुव पूर्णाङ्क     v_type = v->type;
    ध्रुव int64_t k_nb0  = k->nb[0];
    ध्रुव int64_t v_nb0  = v->nb[0];

    ध्रुव int64_t dk  = q->ne[0];  // head dim for keys/queries
    ध्रुव int64_t nq  = q->ne[1];  // number of query positions
    ध्रुव int64_t nhq = q->ne[2];  // number of query heads
    ध्रुव int64_t no  = q->ne[3];  // outer batch

    ध्रुव int64_t nk  = k->ne[1];  // number of key/value positions
    ध्रुव int64_t nhk = k->ne[2];  // number of kv heads
    ध्रुव int64_t dv  = v->ne[0];  // head dim for values

    यदि (dv > FA_DV_MAX) {
        प्रत्यागम -1;
    }

    // GQA: query heads per kv head
    ध्रुव int64_t gqa_अनुपातः = nhq / nhk;

    ध्रुव int64_t योग_पङ्क्तिकाः = nq * nhq * no;
    ध्रुव भिन्न   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    ध्रुव पूर्णाङ्क use_fast_store = (dv % 16 == 0);

    प्रति (int64_t row = thread_id; row < योग_पङ्क्तिकाः; row += अङ्कः_तन्तवः) {
        ध्रुव int64_t iq3 = row / (nhq * nq);
        ध्रुव int64_t rem = row % (nhq * nq);
        ध्रुव int64_t iq2 = rem / nq;  // query head index
        ध्रुव int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        ध्रुव int64_t ik2 = iq2 / gqa_अनुपातः;

        // Q is always F32
        ध्रुव भिन्न * pq = (ध्रुव भिन्न *) (q_दत्त + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        भिन्न * बहिर्गमनम् = (भिन्न *) (dst_दत्त + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        ध्रुव int64_t kv_आधारः = ik2 * k->nb[2] + iq3 * k->nb[3];
        ध्रुव int64_t vv_आधारः = ik2 * v->nb[2] + iq3 * v->nb[3];

        भिन्न acc[FA_DV_MAX];
        प्रति (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        भिन्न M = -3.402823466e+38f;
        भिन्न S = 0.0f;

        प्रति (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            भिन्न आवरणम्_val = 0.0f;
            यदि (has_आवरणम्) {
                आवरणम्_val = get_आवरणम्_val(आवरणम्, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                यदि (आवरणम्_val == -3.402823466e+38f || आवरणम्_val != आवरणम्_val) {
                    अनुवर्त;
                }
            }

            ध्रुव वर्ण * pk = k_दत्त + ik1 * k->nb[1] + kv_आधारः;
            ध्रुव वर्ण * pv = v_दत्त + ik1 * v->nb[1] + vv_आधारः;

            भिन्न       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + आवरणम्_val;
            ध्रुव भिन्न Mold = M;

            भिन्न ms = 1.0f;
            भिन्न vs = 1.0f;
            यदि (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                प्रति (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } अन्यथा {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            यदि (v_type == GGML_TYPE_F32) {
                ध्रुव भिन्न * pvf = (ध्रुव भिन्न *) pv;
                प्रति (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } अन्यथा {
                प्रति (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_गन्तव्यम्_fp32(*(ध्रुव uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        ध्रुव भिन्न S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        यदि (use_fast_store) {
            प्रति (int64_t d = 0; d < dv; ++d) {
                बहिर्गमनम्[d] = acc[d] * S_inv;
            }
        } अन्यथा {
            प्रति (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((चञ्चल भिन्न *) &बहिर्गमनम्[d], acc[d] * S_inv);
            }
        }
    }

    प्रत्यागम 0;
}
