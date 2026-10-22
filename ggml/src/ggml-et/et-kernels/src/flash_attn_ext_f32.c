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

ರಚನೆ ggml_et_flash_attn_ext_params {
    ರಚನೆ ggml_tensor src0;      // Q tensor (F32)
    ರಚನೆ ggml_tensor src1;      // K tensor (F16 or F32)
    ರಚನೆ ggml_tensor src2;      // V tensor (F16 or F32)
    ರಚನೆ ggml_tensor ಮುಖವಾಡ;      // mask tensor (F16 or F32), zeroed when absent
    ರಚನೆ ggml_tensor dst;       // Output tensor (F32)
    ದಶಮಾಂಶ              scale;     // Scale factor applied to QK
    int32_t            has_ಮುಖವಾಡ;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#ವ್ಯಾಖ್ಯಾನಿಸು FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ read_kv_f32(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಅಡ್ಡಸಾಲು_ತಳ, int64_t d, int64_t nb0, ಪೂರ್ಣಾಂಕ type) {
    ಆದರೆ (type == GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (ಅಡ್ಡಸಾಲು_ತಳ + d * nb0);
    }
    // F16
    ಹಿಂತಿರುಗಿ fp16_ಗಮ್ಯ_fp32(*(ಸ್ಥಿರಾಂಕ uint16_t *) (ಅಡ್ಡಸಾಲು_ತಳ + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ dot_qk(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ಅಡ್ಡಸಾಲು, int64_t dk, int64_t k_nb0, ಪೂರ್ಣಾಂಕ k_type) {
    ದಶಮಾಂಶ acc = 0.0f;
    ಆದರೆ (k_type == GGML_TYPE_F32) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * kf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) k_ಅಡ್ಡಸಾಲು;
        ಗಾಗಿ (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // F16 stride-aware read
        ಗಾಗಿ (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_ಗಮ್ಯ_fp32(*(ಸ್ಥಿರಾಂಕ uint16_t *) (k_ಅಡ್ಡಸಾಲು + i * k_nb0));
        }
    }
    ಹಿಂತಿರುಗಿ acc;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ get_ಮುಖವಾಡ_val(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಮುಖವಾಡ, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ + iq1 * ಮುಖವಾಡ->nb[1] + (iq2 % ಮುಖವಾಡ->ne[2]) * ಮುಖವಾಡ->nb[2] +
                        (iq3 % ಮುಖವಾಡ->ne[3]) * ಮುಖವಾಡ->nb[3];

    ಆದರೆ (ಮುಖವಾಡ->type == GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (ತಳ + ik1 * ಮುಖವಾಡ->nb[0]);
    }
    // F16
    ಹಿಂತಿರುಗಿ fp16_ಗಮ್ಯ_fp32(*(ಸ್ಥಿರಾಂಕ uint16_t *) (ತಳ + ik1 * ಮುಖವಾಡ->nb[0]));
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_flash_attn_ext_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env || !params) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);
    ಆದರೆ (thread_id < 0 || ಅಂಕೆ_ಎಳೆಗಳು <= 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ರಚನೆ ggml_tensor * q        = &params->src0;
    ರಚನೆ ggml_tensor * k        = &params->src1;
    ರಚನೆ ggml_tensor * v        = &params->src2;
    ರಚನೆ ggml_tensor * dst      = &params->dst;
    ಸ್ಥಿರಾಂಕ int32_t        has_ಮುಖವಾಡ = params->has_ಮುಖವಾಡ;
    ರಚನೆ ggml_tensor * ಮುಖವಾಡ     = has_ಮುಖವಾಡ ? &params->ಮುಖವಾಡ : (ರಚನೆ ggml_tensor *) 0;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * q_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ *       dst_ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     k_type = k->type;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     v_type = v->type;
    ಸ್ಥಿರಾಂಕ int64_t k_nb0  = k->nb[0];
    ಸ್ಥಿರಾಂಕ int64_t v_nb0  = v->nb[0];

    ಸ್ಥಿರಾಂಕ int64_t dk  = q->ne[0];  // head dim for keys/queries
    ಸ್ಥಿರಾಂಕ int64_t nq  = q->ne[1];  // number of query positions
    ಸ್ಥಿರಾಂಕ int64_t nhq = q->ne[2];  // number of query heads
    ಸ್ಥಿರಾಂಕ int64_t no  = q->ne[3];  // outer batch

    ಸ್ಥಿರಾಂಕ int64_t nk  = k->ne[1];  // number of key/value positions
    ಸ್ಥಿರಾಂಕ int64_t nhk = k->ne[2];  // number of kv heads
    ಸ್ಥಿರಾಂಕ int64_t dv  = v->ne[0];  // head dim for values

    ಆದರೆ (dv > FA_DV_MAX) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // GQA: query heads per kv head
    ಸ್ಥಿರಾಂಕ int64_t gqa_ಅನುಪಾತ = nhq / nhk;

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = nq * nhq * no;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ use_fast_store = (dv % 16 == 0);

    ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = thread_id; ಅಡ್ಡಸಾಲು < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int64_t iq3 = ಅಡ್ಡಸಾಲು / (nhq * nq);
        ಸ್ಥಿರಾಂಕ int64_t rem = ಅಡ್ಡಸಾಲು % (nhq * nq);
        ಸ್ಥಿರಾಂಕ int64_t iq2 = rem / nq;  // query head index
        ಸ್ಥಿರಾಂಕ int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        ಸ್ಥಿರಾಂಕ int64_t ik2 = iq2 / gqa_ಅನುಪಾತ;

        // Q is always F32
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * pq = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (q_ದತ್ತಾಂಶ + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        ದಶಮಾಂಶ * ಹೊರಹೋಗು = (ದಶಮಾಂಶ *) (dst_ದತ್ತಾಂಶ + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        ಸ್ಥಿರಾಂಕ int64_t kv_ತಳ = ik2 * k->nb[2] + iq3 * k->nb[3];
        ಸ್ಥಿರಾಂಕ int64_t vv_ತಳ = ik2 * v->nb[2] + iq3 * v->nb[3];

        ದಶಮಾಂಶ acc[FA_DV_MAX];
        ಗಾಗಿ (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        ದಶಮಾಂಶ M = -3.402823466e+38f;
        ದಶಮಾಂಶ S = 0.0f;

        ಗಾಗಿ (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            ದಶಮಾಂಶ ಮುಖವಾಡ_val = 0.0f;
            ಆದರೆ (has_ಮುಖವಾಡ) {
                ಮುಖವಾಡ_val = get_ಮುಖವಾಡ_val(ಮುಖವಾಡ, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                ಆದರೆ (ಮುಖವಾಡ_val == -3.402823466e+38f || ಮುಖವಾಡ_val != ಮುಖವಾಡ_val) {
                    ಮುಂದುವರಿ;
                }
            }

            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * pk = k_ದತ್ತಾಂಶ + ik1 * k->nb[1] + kv_ತಳ;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * pv = v_ದತ್ತಾಂಶ + ik1 * v->nb[1] + vv_ತಳ;

            ದಶಮಾಂಶ       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + ಮುಖವಾಡ_val;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ Mold = M;

            ದಶಮಾಂಶ ms = 1.0f;
            ದಶಮಾಂಶ vs = 1.0f;
            ಆದರೆ (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                ಗಾಗಿ (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            ಆದರೆ (v_type == GGML_TYPE_F32) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * pvf = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) pv;
                ಗಾಗಿ (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾಗಿ (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_ಗಮ್ಯ_fp32(*(ಸ್ಥಿರಾಂಕ uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        ಆದರೆ (use_fast_store) {
            ಗಾಗಿ (int64_t d = 0; d < dv; ++d) {
                ಹೊರಹೋಗು[d] = acc[d] * S_inv;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) &ಹೊರಹೋಗು[d], acc[d] * S_inv);
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
