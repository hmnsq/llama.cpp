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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_flash_attn_ext_params {
    ਬਣਤਰ ggml_tensor src0;      // Q tensor (F32)
    ਬਣਤਰ ggml_tensor src1;      // K tensor (F16 or F32)
    ਬਣਤਰ ggml_tensor src2;      // V tensor (F16 or F32)
    ਬਣਤਰ ggml_tensor ਨਕਾਬ;      // mask tensor (F16 or F32), zeroed when absent
    ਬਣਤਰ ggml_tensor dst;       // Output tensor (F32)
    ਦਸ਼ਮਲਵ              scale;     // Scale factor applied to QK
    int32_t            has_ਨਕਾਬ;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ read_kv_f32(ਅਚਲ ਅੱਖਰ * ਪੰਕਤੀ_ਅਧਾਰ, int64_t d, int64_t nb0, ਪੂਰਨ type) {
    ਜੇ (type == GGML_TYPE_F32) {
        ਵਾਪਸ *(ਅਚਲ ਦਸ਼ਮਲਵ *) (ਪੰਕਤੀ_ਅਧਾਰ + d * nb0);
    }
    // F16
    ਵਾਪਸ fp16_ਮੰਜ਼ਿਲ_fp32(*(ਅਚਲ uint16_t *) (ਪੰਕਤੀ_ਅਧਾਰ + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ dot_qk(ਅਚਲ ਦਸ਼ਮਲਵ * q, ਅਚਲ ਅੱਖਰ * k_ਪੰਕਤੀ, int64_t dk, int64_t k_nb0, ਪੂਰਨ k_type) {
    ਦਸ਼ਮਲਵ acc = 0.0f;
    ਜੇ (k_type == GGML_TYPE_F32) {
        ਅਚਲ ਦਸ਼ਮਲਵ * kf = (ਅਚਲ ਦਸ਼ਮਲਵ *) k_ਪੰਕਤੀ;
        ਲਈ (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } ਨਹੀਂਤਾਂ {
        // F16 stride-aware read
        ਲਈ (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_ਮੰਜ਼ਿਲ_fp32(*(ਅਚਲ uint16_t *) (k_ਪੰਕਤੀ + i * k_nb0));
        }
    }
    ਵਾਪਸ acc;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ get_ਨਕਾਬ_val(ਅਚਲ ਬਣਤਰ ggml_tensor * ਨਕਾਬ, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    ਅਚਲ ਅੱਖਰ * ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ + iq1 * ਨਕਾਬ->nb[1] + (iq2 % ਨਕਾਬ->ne[2]) * ਨਕਾਬ->nb[2] +
                        (iq3 % ਨਕਾਬ->ne[3]) * ਨਕਾਬ->nb[3];

    ਜੇ (ਨਕਾਬ->type == GGML_TYPE_F32) {
        ਵਾਪਸ *(ਅਚਲ ਦਸ਼ਮਲਵ *) (ਅਧਾਰ + ik1 * ਨਕਾਬ->nb[0]);
    }
    // F16
    ਵਾਪਸ fp16_ਮੰਜ਼ਿਲ_fp32(*(ਅਚਲ uint16_t *) (ਅਧਾਰ + ik1 * ਨਕਾਬ->nb[0]));
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_flash_attn_ext_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env || !params) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);
    ਜੇ (thread_id < 0 || ਅੰਕ_ਧਾਗੇ <= 0) {
        ਵਾਪਸ 0;
    }

    ਬਣਤਰ ggml_tensor * q        = &params->src0;
    ਬਣਤਰ ggml_tensor * k        = &params->src1;
    ਬਣਤਰ ggml_tensor * v        = &params->src2;
    ਬਣਤਰ ggml_tensor * dst      = &params->dst;
    ਅਚਲ int32_t        has_ਨਕਾਬ = params->has_ਨਕਾਬ;
    ਬਣਤਰ ggml_tensor * ਨਕਾਬ     = has_ਨਕਾਬ ? &params->ਨਕਾਬ : (ਬਣਤਰ ggml_tensor *) 0;

    ਅਚਲ ਅੱਖਰ * q_ਅੰਕੜੇ   = (ਅਚਲ ਅੱਖਰ *) q->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * k_ਅੰਕੜੇ   = (ਅਚਲ ਅੱਖਰ *) k->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * v_ਅੰਕੜੇ   = (ਅਚਲ ਅੱਖਰ *) v->ਅੰਕੜੇ;
    ਅੱਖਰ *       dst_ਅੰਕੜੇ = (ਅੱਖਰ *) dst->ਅੰਕੜੇ;

    ਅਚਲ ਪੂਰਨ     k_type = k->type;
    ਅਚਲ ਪੂਰਨ     v_type = v->type;
    ਅਚਲ int64_t k_nb0  = k->nb[0];
    ਅਚਲ int64_t v_nb0  = v->nb[0];

    ਅਚਲ int64_t dk  = q->ne[0];  // head dim for keys/queries
    ਅਚਲ int64_t nq  = q->ne[1];  // number of query positions
    ਅਚਲ int64_t nhq = q->ne[2];  // number of query heads
    ਅਚਲ int64_t no  = q->ne[3];  // outer batch

    ਅਚਲ int64_t nk  = k->ne[1];  // number of key/value positions
    ਅਚਲ int64_t nhk = k->ne[2];  // number of kv heads
    ਅਚਲ int64_t dv  = v->ne[0];  // head dim for values

    ਜੇ (dv > FA_DV_MAX) {
        ਵਾਪਸ -1;
    }

    // GQA: query heads per kv head
    ਅਚਲ int64_t gqa_ਅਨੁਪਾਤ = nhq / nhk;

    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = nq * nhq * no;
    ਅਚਲ ਦਸ਼ਮਲਵ   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    ਅਚਲ ਪੂਰਨ use_fast_store = (dv % 16 == 0);

    ਲਈ (int64_t ਪੰਕਤੀ = thread_id; ਪੰਕਤੀ < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int64_t iq3 = ਪੰਕਤੀ / (nhq * nq);
        ਅਚਲ int64_t rem = ਪੰਕਤੀ % (nhq * nq);
        ਅਚਲ int64_t iq2 = rem / nq;  // query head index
        ਅਚਲ int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        ਅਚਲ int64_t ik2 = iq2 / gqa_ਅਨੁਪਾਤ;

        // Q is always F32
        ਅਚਲ ਦਸ਼ਮਲਵ * pq = (ਅਚਲ ਦਸ਼ਮਲਵ *) (q_ਅੰਕੜੇ + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ = (ਦਸ਼ਮਲਵ *) (dst_ਅੰਕੜੇ + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        ਅਚਲ int64_t kv_ਅਧਾਰ = ik2 * k->nb[2] + iq3 * k->nb[3];
        ਅਚਲ int64_t vv_ਅਧਾਰ = ik2 * v->nb[2] + iq3 * v->nb[3];

        ਦਸ਼ਮਲਵ acc[FA_DV_MAX];
        ਲਈ (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        ਦਸ਼ਮਲਵ M = -3.402823466e+38f;
        ਦਸ਼ਮਲਵ S = 0.0f;

        ਲਈ (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            ਦਸ਼ਮਲਵ ਨਕਾਬ_val = 0.0f;
            ਜੇ (has_ਨਕਾਬ) {
                ਨਕਾਬ_val = get_ਨਕਾਬ_val(ਨਕਾਬ, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                ਜੇ (ਨਕਾਬ_val == -3.402823466e+38f || ਨਕਾਬ_val != ਨਕਾਬ_val) {
                    ਜਾਰੀ;
                }
            }

            ਅਚਲ ਅੱਖਰ * pk = k_ਅੰਕੜੇ + ik1 * k->nb[1] + kv_ਅਧਾਰ;
            ਅਚਲ ਅੱਖਰ * pv = v_ਅੰਕੜੇ + ik1 * v->nb[1] + vv_ਅਧਾਰ;

            ਦਸ਼ਮਲਵ       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + ਨਕਾਬ_val;
            ਅਚਲ ਦਸ਼ਮਲਵ Mold = M;

            ਦਸ਼ਮਲਵ ms = 1.0f;
            ਦਸ਼ਮਲਵ vs = 1.0f;
            ਜੇ (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                ਲਈ (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } ਨਹੀਂਤਾਂ {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            ਜੇ (v_type == GGML_TYPE_F32) {
                ਅਚਲ ਦਸ਼ਮਲਵ * pvf = (ਅਚਲ ਦਸ਼ਮਲਵ *) pv;
                ਲਈ (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } ਨਹੀਂਤਾਂ {
                ਲਈ (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_ਮੰਜ਼ਿਲ_fp32(*(ਅਚਲ uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        ਅਚਲ ਦਸ਼ਮਲਵ S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        ਜੇ (use_fast_store) {
            ਲਈ (int64_t d = 0; d < dv; ++d) {
                ਬਾਹਰਲਾ[d] = acc[d] * S_inv;
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) &ਬਾਹਰਲਾ[d], acc[d] * S_inv);
            }
        }
    }

    ਵਾਪਸ 0;
}
