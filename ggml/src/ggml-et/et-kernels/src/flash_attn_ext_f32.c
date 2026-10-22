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

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_flash_attn_ext_params {
    ഘടന ggml_tensor src0;      // Q tensor (F32)
    ഘടന ggml_tensor src1;      // K tensor (F16 or F32)
    ഘടന ggml_tensor src2;      // V tensor (F16 or F32)
    ഘടന ggml_tensor mask;      // mask tensor (F16 or F32), zeroed when absent
    ഘടന ggml_tensor dst;       // Output tensor (F32)
    ദശാംശം              scale;     // Scale factor applied to QK
    int32_t            has_mask;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
സ്ഥിതം വരിയിൽ ദശാംശം read_kv_f32(സ്ഥിരം അക്ഷരം * row_base, int64_t d, int64_t nb0, പൂർണ്ണസംഖ്യ type) {
    എങ്കിൽ (type == GGML_TYPE_F32) {
        തിരികെ *(സ്ഥിരം ദശാംശം *) (row_base + d * nb0);
    }
    // F16
    തിരികെ fp16_to_fp32(*(സ്ഥിരം uint16_t *) (row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
സ്ഥിതം വരിയിൽ ദശാംശം dot_qk(സ്ഥിരം ദശാംശം * q, സ്ഥിരം അക്ഷരം * k_row, int64_t dk, int64_t k_nb0, പൂർണ്ണസംഖ്യ k_type) {
    ദശാംശം acc = 0.0f;
    എങ്കിൽ (k_type == GGML_TYPE_F32) {
        സ്ഥിരം ദശാംശം * kf = (സ്ഥിരം ദശാംശം *) k_row;
        വേണ്ടി (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } അല്ലെങ്കിൽ {
        // F16 stride-aware read
        വേണ്ടി (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(സ്ഥിരം uint16_t *) (k_row + i * k_nb0));
        }
    }
    തിരികെ acc;
}

സ്ഥിതം വരിയിൽ ദശാംശം get_mask_val(സ്ഥിരം ഘടന ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    സ്ഥിരം അക്ഷരം * base = (സ്ഥിരം അക്ഷരം *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    എങ്കിൽ (mask->type == GGML_TYPE_F32) {
        തിരികെ *(സ്ഥിരം ദശാംശം *) (base + ik1 * mask->nb[0]);
    }
    // F16
    തിരികെ fp16_to_fp32(*(സ്ഥിരം uint16_t *) (base + ik1 * mask->nb[0]));
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_flash_attn_ext_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env || !params) {
        തിരികെ -1;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);
    എങ്കിൽ (thread_id < 0 || അക്കം_threads <= 0) {
        തിരികെ 0;
    }

    ഘടന ggml_tensor * q        = &params->src0;
    ഘടന ggml_tensor * k        = &params->src1;
    ഘടന ggml_tensor * v        = &params->src2;
    ഘടന ggml_tensor * dst      = &params->dst;
    സ്ഥിരം int32_t        has_mask = params->has_mask;
    ഘടന ggml_tensor * mask     = has_mask ? &params->mask : (ഘടന ggml_tensor *) 0;

    സ്ഥിരം അക്ഷരം * q_data   = (സ്ഥിരം അക്ഷരം *) q->data;
    സ്ഥിരം അക്ഷരം * k_data   = (സ്ഥിരം അക്ഷരം *) k->data;
    സ്ഥിരം അക്ഷരം * v_data   = (സ്ഥിരം അക്ഷരം *) v->data;
    അക്ഷരം *       dst_data = (അക്ഷരം *) dst->data;

    സ്ഥിരം പൂർണ്ണസംഖ്യ     k_type = k->type;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     v_type = v->type;
    സ്ഥിരം int64_t k_nb0  = k->nb[0];
    സ്ഥിരം int64_t v_nb0  = v->nb[0];

    സ്ഥിരം int64_t dk  = q->ne[0];  // head dim for keys/queries
    സ്ഥിരം int64_t nq  = q->ne[1];  // number of query positions
    സ്ഥിരം int64_t nhq = q->ne[2];  // number of query heads
    സ്ഥിരം int64_t no  = q->ne[3];  // outer batch

    സ്ഥിരം int64_t nk  = k->ne[1];  // number of key/value positions
    സ്ഥിരം int64_t nhk = k->ne[2];  // number of kv heads
    സ്ഥിരം int64_t dv  = v->ne[0];  // head dim for values

    എങ്കിൽ (dv > FA_DV_MAX) {
        തിരികെ -1;
    }

    // GQA: query heads per kv head
    സ്ഥിരം int64_t gqa_ratio = nhq / nhk;

    സ്ഥിരം int64_t total_rows = nq * nhq * no;
    സ്ഥിരം ദശാംശം   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    സ്ഥിരം പൂർണ്ണസംഖ്യ use_fast_store = (dv % 16 == 0);

    വേണ്ടി (int64_t row = thread_id; row < total_rows; row += അക്കം_threads) {
        സ്ഥിരം int64_t iq3 = row / (nhq * nq);
        സ്ഥിരം int64_t rem = row % (nhq * nq);
        സ്ഥിരം int64_t iq2 = rem / nq;  // query head index
        സ്ഥിരം int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        സ്ഥിരം int64_t ik2 = iq2 / gqa_ratio;

        // Q is always F32
        സ്ഥിരം ദശാംശം * pq = (സ്ഥിരം ദശാംശം *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        ദശാംശം * out = (ദശാംശം *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        സ്ഥിരം int64_t kv_base = ik2 * k->nb[2] + iq3 * k->nb[3];
        സ്ഥിരം int64_t vv_base = ik2 * v->nb[2] + iq3 * v->nb[3];

        ദശാംശം acc[FA_DV_MAX];
        വേണ്ടി (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        ദശാംശം M = -3.402823466e+38f;
        ദശാംശം S = 0.0f;

        വേണ്ടി (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            ദശാംശം mask_val = 0.0f;
            എങ്കിൽ (has_mask) {
                mask_val = get_mask_val(mask, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                എങ്കിൽ (mask_val == -3.402823466e+38f || mask_val != mask_val) {
                    തുടരുക;
                }
            }

            സ്ഥിരം അക്ഷരം * pk = k_data + ik1 * k->nb[1] + kv_base;
            സ്ഥിരം അക്ഷരം * pv = v_data + ik1 * v->nb[1] + vv_base;

            ദശാംശം       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + mask_val;
            സ്ഥിരം ദശാംശം Mold = M;

            ദശാംശം ms = 1.0f;
            ദശാംശം vs = 1.0f;
            എങ്കിൽ (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                വേണ്ടി (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } അല്ലെങ്കിൽ {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            എങ്കിൽ (v_type == GGML_TYPE_F32) {
                സ്ഥിരം ദശാംശം * pvf = (സ്ഥിരം ദശാംശം *) pv;
                വേണ്ടി (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } അല്ലെങ്കിൽ {
                വേണ്ടി (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_to_fp32(*(സ്ഥിരം uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        സ്ഥിരം ദശാംശം S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        എങ്കിൽ (use_fast_store) {
            വേണ്ടി (int64_t d = 0; d < dv; ++d) {
                out[d] = acc[d] * S_inv;
            }
        } അല്ലെങ്കിൽ {
            വേണ്ടി (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((അസ്ഥിര ദശാംശം *) &out[d], acc[d] * S_inv);
            }
        }
    }

    തിരികെ 0;
}
