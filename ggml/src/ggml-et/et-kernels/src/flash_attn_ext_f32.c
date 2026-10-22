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

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_flash_attn_ext_params {
    నిర్మాణం ggml_tensor src0;      // Q tensor (F32)
    నిర్మాణం ggml_tensor src1;      // K tensor (F16 or F32)
    నిర్మాణం ggml_tensor src2;      // V tensor (F16 or F32)
    నిర్మాణం ggml_tensor ముసుగు;      // mask tensor (F16 or F32), zeroed when absent
    నిర్మాణం ggml_tensor dst;       // Output tensor (F32)
    దశాంశం              scale;     // Scale factor applied to QK
    int32_t            has_ముసుగు;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#నిర్వచించు FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
స్థిర అంతఃవరుస దశాంశం read_kv_f32(స్థిరరాశి అక్షరం * అడ్డువరుస_పునాది, int64_t d, int64_t nb0, పూర్ణాంకం type) {
    ఒకవేళ (type == GGML_TYPE_F32) {
        తిరిగి *(స్థిరరాశి దశాంశం *) (అడ్డువరుస_పునాది + d * nb0);
    }
    // F16
    తిరిగి fp16_గమ్యం_fp32(*(స్థిరరాశి uint16_t *) (అడ్డువరుస_పునాది + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
స్థిర అంతఃవరుస దశాంశం dot_qk(స్థిరరాశి దశాంశం * q, స్థిరరాశి అక్షరం * k_అడ్డువరుస, int64_t dk, int64_t k_nb0, పూర్ణాంకం k_type) {
    దశాంశం acc = 0.0f;
    ఒకవేళ (k_type == GGML_TYPE_F32) {
        స్థిరరాశి దశాంశం * kf = (స్థిరరాశి దశాంశం *) k_అడ్డువరుస;
        కొరకు (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } లేకపోతే {
        // F16 stride-aware read
        కొరకు (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_గమ్యం_fp32(*(స్థిరరాశి uint16_t *) (k_అడ్డువరుస + i * k_nb0));
        }
    }
    తిరిగి acc;
}

స్థిర అంతఃవరుస దశాంశం get_ముసుగు_val(స్థిరరాశి నిర్మాణం ggml_tensor * ముసుగు, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    స్థిరరాశి అక్షరం * పునాది = (స్థిరరాశి అక్షరం *) ముసుగు->దత్తాంశం + iq1 * ముసుగు->nb[1] + (iq2 % ముసుగు->ne[2]) * ముసుగు->nb[2] +
                        (iq3 % ముసుగు->ne[3]) * ముసుగు->nb[3];

    ఒకవేళ (ముసుగు->type == GGML_TYPE_F32) {
        తిరిగి *(స్థిరరాశి దశాంశం *) (పునాది + ik1 * ముసుగు->nb[0]);
    }
    // F16
    తిరిగి fp16_గమ్యం_fp32(*(స్థిరరాశి uint16_t *) (పునాది + ik1 * ముసుగు->nb[0]));
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_flash_attn_ext_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env || !params) {
        తిరిగి -1;
    }

    స్థిరరాశి పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    స్థిరరాశి పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);
    ఒకవేళ (thread_id < 0 || అంకె_దారాలు <= 0) {
        తిరిగి 0;
    }

    నిర్మాణం ggml_tensor * q        = &params->src0;
    నిర్మాణం ggml_tensor * k        = &params->src1;
    నిర్మాణం ggml_tensor * v        = &params->src2;
    నిర్మాణం ggml_tensor * dst      = &params->dst;
    స్థిరరాశి int32_t        has_ముసుగు = params->has_ముసుగు;
    నిర్మాణం ggml_tensor * ముసుగు     = has_ముసుగు ? &params->ముసుగు : (నిర్మాణం ggml_tensor *) 0;

    స్థిరరాశి అక్షరం * q_దత్తాంశం   = (స్థిరరాశి అక్షరం *) q->దత్తాంశం;
    స్థిరరాశి అక్షరం * k_దత్తాంశం   = (స్థిరరాశి అక్షరం *) k->దత్తాంశం;
    స్థిరరాశి అక్షరం * v_దత్తాంశం   = (స్థిరరాశి అక్షరం *) v->దత్తాంశం;
    అక్షరం *       dst_దత్తాంశం = (అక్షరం *) dst->దత్తాంశం;

    స్థిరరాశి పూర్ణాంకం     k_type = k->type;
    స్థిరరాశి పూర్ణాంకం     v_type = v->type;
    స్థిరరాశి int64_t k_nb0  = k->nb[0];
    స్థిరరాశి int64_t v_nb0  = v->nb[0];

    స్థిరరాశి int64_t dk  = q->ne[0];  // head dim for keys/queries
    స్థిరరాశి int64_t nq  = q->ne[1];  // number of query positions
    స్థిరరాశి int64_t nhq = q->ne[2];  // number of query heads
    స్థిరరాశి int64_t no  = q->ne[3];  // outer batch

    స్థిరరాశి int64_t nk  = k->ne[1];  // number of key/value positions
    స్థిరరాశి int64_t nhk = k->ne[2];  // number of kv heads
    స్థిరరాశి int64_t dv  = v->ne[0];  // head dim for values

    ఒకవేళ (dv > FA_DV_MAX) {
        తిరిగి -1;
    }

    // GQA: query heads per kv head
    స్థిరరాశి int64_t gqa_నిష్పత్తి = nhq / nhk;

    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = nq * nhq * no;
    స్థిరరాశి దశాంశం   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    స్థిరరాశి పూర్ణాంకం use_fast_store = (dv % 16 == 0);

    కొరకు (int64_t అడ్డువరుస = thread_id; అడ్డువరుస < మొత్తం_అడ్డువరుసలు; అడ్డువరుస += అంకె_దారాలు) {
        స్థిరరాశి int64_t iq3 = అడ్డువరుస / (nhq * nq);
        స్థిరరాశి int64_t rem = అడ్డువరుస % (nhq * nq);
        స్థిరరాశి int64_t iq2 = rem / nq;  // query head index
        స్థిరరాశి int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        స్థిరరాశి int64_t ik2 = iq2 / gqa_నిష్పత్తి;

        // Q is always F32
        స్థిరరాశి దశాంశం * pq = (స్థిరరాశి దశాంశం *) (q_దత్తాంశం + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        దశాంశం * బహిర్గతం = (దశాంశం *) (dst_దత్తాంశం + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        స్థిరరాశి int64_t kv_పునాది = ik2 * k->nb[2] + iq3 * k->nb[3];
        స్థిరరాశి int64_t vv_పునాది = ik2 * v->nb[2] + iq3 * v->nb[3];

        దశాంశం acc[FA_DV_MAX];
        కొరకు (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        దశాంశం M = -3.402823466e+38f;
        దశాంశం S = 0.0f;

        కొరకు (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            దశాంశం ముసుగు_val = 0.0f;
            ఒకవేళ (has_ముసుగు) {
                ముసుగు_val = get_ముసుగు_val(ముసుగు, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                ఒకవేళ (ముసుగు_val == -3.402823466e+38f || ముసుగు_val != ముసుగు_val) {
                    కొనసాగు;
                }
            }

            స్థిరరాశి అక్షరం * pk = k_దత్తాంశం + ik1 * k->nb[1] + kv_పునాది;
            స్థిరరాశి అక్షరం * pv = v_దత్తాంశం + ik1 * v->nb[1] + vv_పునాది;

            దశాంశం       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + ముసుగు_val;
            స్థిరరాశి దశాంశం Mold = M;

            దశాంశం ms = 1.0f;
            దశాంశం vs = 1.0f;
            ఒకవేళ (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                కొరకు (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } లేకపోతే {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            ఒకవేళ (v_type == GGML_TYPE_F32) {
                స్థిరరాశి దశాంశం * pvf = (స్థిరరాశి దశాంశం *) pv;
                కొరకు (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } లేకపోతే {
                కొరకు (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_గమ్యం_fp32(*(స్థిరరాశి uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        స్థిరరాశి దశాంశం S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        ఒకవేళ (use_fast_store) {
            కొరకు (int64_t d = 0; d < dv; ++d) {
                బహిర్గతం[d] = acc[d] * S_inv;
            }
        } లేకపోతే {
            కొరకు (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((అస్థిర దశాంశం *) &బహిర్గతం[d], acc[d] * S_inv);
            }
        }
    }

    తిరిగి 0;
}
