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

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdbool.h>
#شامل <stdint.h>

ساخت ggml_et_flash_attn_ext_params {
    ساخت ggml_tensor src0;      // Q tensor (F32)
    ساخت ggml_tensor src1;      // K tensor (F16 or F32)
    ساخت ggml_tensor src2;      // V tensor (F16 or F32)
    ساخت ggml_tensor نقاب;      // mask tensor (F16 or F32), zeroed when absent
    ساخت ggml_tensor dst;       // Output tensor (F32)
    اعشاری              scale;     // Scale factor applied to QK
    int32_t            has_نقاب;  // nonzero if mask is present
};

// Maximum head dimension supported (128 covers all common LLMs).
#تعریف_کرو FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
ساکن اندرونی اعشاری read_kv_f32(ثابت حرف * صف_بنیاد, int64_t d, int64_t nb0, صحیح type) {
    اگر (type == GGML_TYPE_F32) {
        واپس *(ثابت اعشاری *) (صف_بنیاد + d * nb0);
    }
    // F16
    واپس fp16_مقصد_fp32(*(ثابت uint16_t *) (صف_بنیاد + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
ساکن اندرونی اعشاری dot_qk(ثابت اعشاری * q, ثابت حرف * k_صف, int64_t dk, int64_t k_nb0, صحیح k_type) {
    اعشاری acc = 0.0f;
    اگر (k_type == GGML_TYPE_F32) {
        ثابت اعشاری * kf = (ثابت اعشاری *) k_صف;
        برائے (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } ورنہ {
        // F16 stride-aware read
        برائے (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_مقصد_fp32(*(ثابت uint16_t *) (k_صف + i * k_nb0));
        }
    }
    واپس acc;
}

ساکن اندرونی اعشاری get_نقاب_val(ثابت ساخت ggml_tensor * نقاب, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    ثابت حرف * بنیاد = (ثابت حرف *) نقاب->ڈیٹا + iq1 * نقاب->nb[1] + (iq2 % نقاب->ne[2]) * نقاب->nb[2] +
                        (iq3 % نقاب->ne[3]) * نقاب->nb[3];

    اگر (نقاب->type == GGML_TYPE_F32) {
        واپس *(ثابت اعشاری *) (بنیاد + ik1 * نقاب->nb[0]);
    }
    // F16
    واپس fp16_مقصد_fp32(*(ثابت uint16_t *) (بنیاد + ik1 * نقاب->nb[0]));
}

صحیح اندراج_نقطہ(ساخت ggml_et_flash_attn_ext_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env || !params) {
        واپس -1;
    }

    ثابت صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    ثابت صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);
    اگر (thread_id < 0 || عدد_دھاگے <= 0) {
        واپس 0;
    }

    ساخت ggml_tensor * q        = &params->src0;
    ساخت ggml_tensor * k        = &params->src1;
    ساخت ggml_tensor * v        = &params->src2;
    ساخت ggml_tensor * dst      = &params->dst;
    ثابت int32_t        has_نقاب = params->has_نقاب;
    ساخت ggml_tensor * نقاب     = has_نقاب ? &params->نقاب : (ساخت ggml_tensor *) 0;

    ثابت حرف * q_ڈیٹا   = (ثابت حرف *) q->ڈیٹا;
    ثابت حرف * k_ڈیٹا   = (ثابت حرف *) k->ڈیٹا;
    ثابت حرف * v_ڈیٹا   = (ثابت حرف *) v->ڈیٹا;
    حرف *       dst_ڈیٹا = (حرف *) dst->ڈیٹا;

    ثابت صحیح     k_type = k->type;
    ثابت صحیح     v_type = v->type;
    ثابت int64_t k_nb0  = k->nb[0];
    ثابت int64_t v_nb0  = v->nb[0];

    ثابت int64_t dk  = q->ne[0];  // head dim for keys/queries
    ثابت int64_t nq  = q->ne[1];  // number of query positions
    ثابت int64_t nhq = q->ne[2];  // number of query heads
    ثابت int64_t no  = q->ne[3];  // outer batch

    ثابت int64_t nk  = k->ne[1];  // number of key/value positions
    ثابت int64_t nhk = k->ne[2];  // number of kv heads
    ثابت int64_t dv  = v->ne[0];  // head dim for values

    اگر (dv > FA_DV_MAX) {
        واپس -1;
    }

    // GQA: query heads per kv head
    ثابت int64_t gqa_تناسب = nhq / nhk;

    ثابت int64_t کل_صفیں = nq * nhq * no;
    ثابت اعشاری   scale      = params->scale;

    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
    // cache-line aligned and we can use fast normal stores. Otherwise we must
    // use atomic stores to avoid cache-line sharing corruption.
    ثابت صحیح use_fast_store = (dv % 16 == 0);

    برائے (int64_t صف = thread_id; صف < کل_صفیں; صف += عدد_دھاگے) {
        ثابت int64_t iq3 = صف / (nhq * nq);
        ثابت int64_t rem = صف % (nhq * nq);
        ثابت int64_t iq2 = rem / nq;  // query head index
        ثابت int64_t iq1 = rem % nq;  // query position

        // Map query head -> kv head for GQA
        ثابت int64_t ik2 = iq2 / gqa_تناسب;

        // Q is always F32
        ثابت اعشاری * pq = (ثابت اعشاری *) (q_ڈیٹا + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);

        // dst layout: [dv, nhq, nq, no]
        اعشاری * برآمد = (اعشاری *) (dst_ڈیٹا + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        // Base byte offsets for K and V head+batch slice
        ثابت int64_t kv_بنیاد = ik2 * k->nb[2] + iq3 * k->nb[3];
        ثابت int64_t vv_بنیاد = ik2 * v->nb[2] + iq3 * v->nb[3];

        اعشاری acc[FA_DV_MAX];
        برائے (int64_t d = 0; d < dv; ++d) {
            acc[d] = 0.0f;
        }

        اعشاری M = -3.402823466e+38f;
        اعشاری S = 0.0f;

        برائے (int64_t ik1 = 0; ik1 < nk; ++ik1) {
            // If mask is present, check for -inf (skip masked positions)
            اعشاری نقاب_val = 0.0f;
            اگر (has_نقاب) {
                نقاب_val = get_نقاب_val(نقاب, iq1, ik1, iq2, iq3);
                // llama.cpp uses -inf for masked positions
                اگر (نقاب_val == -3.402823466e+38f || نقاب_val != نقاب_val) {
                    جاری;
                }
            }

            ثابت حرف * pk = k_ڈیٹا + ik1 * k->nb[1] + kv_بنیاد;
            ثابت حرف * pv = v_ڈیٹا + ik1 * v->nb[1] + vv_بنیاد;

            اعشاری       s    = dot_qk(pq, pk, dk, k_nb0, k_type) * scale + نقاب_val;
            ثابت اعشاری Mold = M;

            اعشاری ms = 1.0f;
            اعشاری vs = 1.0f;
            اگر (s > M) {
                M  = s;
                ms = et_expf(Mold - M);
                برائے (int64_t d = 0; d < dv; ++d) {
                    acc[d] *= ms;
                }
            } ورنہ {
                vs = et_expf(s - M);
            }

            // Accumulate weighted V
            اگر (v_type == GGML_TYPE_F32) {
                ثابت اعشاری * pvf = (ثابت اعشاری *) pv;
                برائے (int64_t d = 0; d < dv; ++d) {
                    acc[d] += pvf[d] * vs;
                }
            } ورنہ {
                برائے (int64_t d = 0; d < dv; ++d) {
                    acc[d] += fp16_مقصد_fp32(*(ثابت uint16_t *) (pv + d * v_nb0)) * vs;
                }
            }

            S = S * ms + vs;
        }

        ثابت اعشاری S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        اگر (use_fast_store) {
            برائے (int64_t d = 0; d < dv; ++d) {
                برآمد[d] = acc[d] * S_inv;
            }
        } ورنہ {
            برائے (int64_t d = 0; d < dv; ++d) {
                atomic_store_f32((متلون اعشاری *) &برآمد[d], acc[d] * S_inv);
            }
        }
    }

    واپس 0;
}
