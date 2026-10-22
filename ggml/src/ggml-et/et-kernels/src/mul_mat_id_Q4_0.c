//******************************************************************************
// MUL_MAT_ID kernel specialized for Q4_0 weights (Mixture of Experts).
//
// C[m, s, b] = Sum(k=0..K-1) A[k, m, ids[s,b]] * B[k, s % ne11, b]
//   A: Q4_0  [K, M, n_expert]   weights
//   B: F32   [K, n_cols, batch] activations
//   ids: I32 [n_expert_used, batch]
//   C: F32   [M, n_expert_used, batch]
//
// Strategy: All harts active. Flat m-major output partition allows amortized
// expert lookups and 2-row x2 dot products.
//******************************************************************************

#شامل "block_ops.h"
#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"
#شامل "quants.h"

#شامل <stdint.h>

صحیح اندراج_نقطہ(ساخت ggml_et_mul_mat_id_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    اگر (!kernel_env || !params) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);
    اگر (thread_id < 0) {
        واپس 0;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * src1 = &params->src1;
    ساخت ggml_tensor * src2 = &params->src2;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    ثابت خالی *    src0_ڈیٹا = src0->ڈیٹا;
    ثابت اعشاری *   src1_ڈیٹا = (ثابت اعشاری *) src1->ڈیٹا;
    ثابت int32_t * src2_ڈیٹا = (ثابت int32_t *) src2->ڈیٹا;
    اعشاری *         dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;
    اگر (!src0_ڈیٹا || !src1_ڈیٹا || !src2_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t K             = src0->ne[0];
    ثابت int64_t M             = src0->ne[1];
    ثابت int64_t n_expert      = src0->ne[2];
    ثابت int64_t n_expert_مستعمل = src2->ne[0];
    ثابت int64_t batch         = src2->ne[1];
    ثابت int64_t ne11          = src1->ne[1];

    اگر (K % QK4_0 != 0) {
        واپس -1;
    }

    ثابت size_t nb01 = src0->nb[1];  // Q4_0 row stride
    ثابت size_t nb02 = src0->nb[2];  // expert stride
    ثابت size_t nb11 = src1->nb[1];  // activation column stride
    ثابت size_t nb12 = src1->nb[2];  // activation batch stride
    ثابت size_t nb20 = src2->nb[0];
    ثابت size_t nb21 = src2->nb[1];
    ثابت size_t nbd0 = dst->nb[0];
    ثابت size_t nbd1 = dst->nb[1];
    ثابت size_t nbd2 = dst->nb[2];

    اگر (src0->nb[0] != حجم(بلاک_q4_0) || src1->nb[0] != حجم(اعشاری) || src2->nb[0] != حجم(int32_t) ||
        nbd0 != حجم(اعشاری)) {
        واپس -1;
    }

    ثابت int64_t K_بلاکیں = K / QK4_0;
    ثابت صحیح     use_x2   = ((nb01 & 31) == 0);

    ثابت uint64_t کل_outputs = (uint64_t) M * (uint64_t) n_expert_مستعمل * (uint64_t) batch;
    اگر (کل_outputs == 0) {
        واپس 0;
    }

    // Even partition: hart h owns outputs [h*chunk, (h+1)*chunk).
    ثابت uint64_t chunk    = (کل_outputs + (uint64_t) عدد_دھاگے - 1) / (uint64_t) عدد_دھاگے;
    ثابت uint64_t my_آغاز = (uint64_t) thread_id * chunk;
    اگر (my_آغاز >= کل_outputs) {
        واپس 0;
    }
    uint64_t my_انتہا = my_آغاز + chunk;
    اگر (my_انتہا > کل_outputs) {
        my_انتہا = کل_outputs;
    }

    // Save mask register once; full lanes for vector dot.
    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    ثابت uint64_t فی_batch = (uint64_t) M * (uint64_t) n_expert_مستعمل;

    uint64_t idx = my_آغاز;
    جب (idx < my_انتہا) {
        // Decode (m, slot, batch) from the m-major linear index.
        ثابت int64_t  batch_idx = (int64_t) (idx / فی_batch);
        ثابت uint64_t rem       = idx - (uint64_t) batch_idx * فی_batch;
        ثابت int64_t  خانہ_idx  = (int64_t) (rem / (uint64_t) M);
        ثابت int64_t  m0        = (int64_t) (rem - (uint64_t) خانہ_idx * (uint64_t) M);

        // How many outputs left in this (slot, batch) run AND in my range.
        ثابت uint64_t تسلسل_انتہا_global =
            (uint64_t) batch_idx * فی_batch + (uint64_t) خانہ_idx * (uint64_t) M + (uint64_t) M;
        ثابت uint64_t انتہا_اندر_my = (تسلسل_انتہا_global < my_انتہا) ? تسلسل_انتہا_global : my_انتہا;
        int64_t        تسلسل_len   = (int64_t) (انتہا_اندر_my - idx);

        // Resolve expert + B column + dst slot for this run.
        ثابت int32_t expert_id =
            *(ثابت int32_t *) ((ثابت حرف *) src2_ڈیٹا + خانہ_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        حرف * dst_خانہ = (حرف *) dst_ڈیٹا + خانہ_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        اگر (expert_id < 0 || expert_id >= n_expert) {
            // Invalid expert id — zero out this run's outputs.
            int64_t m = m0;
            برائے (int64_t i = 0; i < تسلسل_len; i++, m++) {
                atomic_store_f32((متلون اعشاری *) (dst_خانہ + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) تسلسل_len;
            جاری;
        }

        ثابت int64_t ستون_idx = خانہ_idx % ne11;
        ثابت اعشاری * b_ستون_بنیاد =
            (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + ستون_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ثابت حرف * expert_بنیاد = (ثابت حرف *) src0_ڈیٹا + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t بایاں = تسلسل_len;

        // Paired-row dots: halves B bandwidth for runs >= 2.
        اگر (use_x2) {
            جب (بایاں >= 2) {
                ثابت بلاک_q4_0 * row0 = (ثابت بلاک_q4_0 *) (expert_بنیاد + m * (int64_t) nb01);
                ثابت بلاک_q4_0 * row1 = (ثابت بلاک_q4_0 *) (expert_بنیاد + (m + 1) * (int64_t) nb01);
                اعشاری              s0, s1;
                q4_dot_compute_x2_aligned(row0, row1, b_ستون_بنیاد, K_بلاکیں, &s0, &s1);
                atomic_store_f32((متلون اعشاری *) (dst_خانہ + m * (int64_t) nbd0), s0);
                atomic_store_f32((متلون اعشاری *) (dst_خانہ + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                بایاں -= 2;
            }
        }

        // Tail / non-aligned fallback: single-row dots.
        جب (بایاں > 0) {
            ثابت بلاک_q4_0 * صف = (ثابت بلاک_q4_0 *) (expert_بنیاد + m * (int64_t) nb01);
            اعشاری              s   = q4_dot_compute(صف, b_ستون_بنیاد, K_بلاکیں);
            atomic_store_f32((متلون اعشاری *) (dst_خانہ + m * (int64_t) nbd0), s);
            m++;
            بایاں--;
        }

        idx += (uint64_t) تسلسل_len;
    }

    q4_dot_انتہا(&q4_state);
    واپس 0;
}
