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

#ضمن "block_ops.h"
#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"
#ضمن "quants.h"

#ضمن <stdint.h>

صحيح مدخل_نقطه(بنية ggml_et_mul_mat_id_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    إذا (!kernel_env || !params) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);
    إذا (thread_id < 0) {
        أرجع 0;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * src2 = &params->src2;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت فراغ *    src0_داده = src0->داده;
    ثابت عائم *   src1_داده = (ثابت عائم *) src1->داده;
    ثابت int32_t * src2_داده = (ثابت int32_t *) src2->داده;
    عائم *         dst_داده  = (عائم *) dst->داده;
    إذا (!src0_داده || !src1_داده || !src2_داده || !dst_داده) {
        أرجع -1;
    }

    ثابت int64_t K             = src0->ne[0];
    ثابت int64_t M             = src0->ne[1];
    ثابت int64_t n_expert      = src0->ne[2];
    ثابت int64_t n_expert_استفاده_شده = src2->ne[0];
    ثابت int64_t batch         = src2->ne[1];
    ثابت int64_t ne11          = src1->ne[1];

    إذا (K % QK4_0 != 0) {
        أرجع -1;
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

    إذا (src0->nb[0] != حجم(بلوک_q4_0) || src1->nb[0] != حجم(عائم) || src2->nb[0] != حجم(int32_t) ||
        nbd0 != حجم(عائم)) {
        أرجع -1;
    }

    ثابت int64_t K_بلوکها = K / QK4_0;
    ثابت صحيح     use_x2   = ((nb01 & 31) == 0);

    ثابت uint64_t مجموع_outputs = (uint64_t) M * (uint64_t) n_expert_استفاده_شده * (uint64_t) batch;
    إذا (مجموع_outputs == 0) {
        أرجع 0;
    }

    // Even partition: hart h owns outputs [h*chunk, (h+1)*chunk).
    ثابت uint64_t chunk    = (مجموع_outputs + (uint64_t) رقم_نخها - 1) / (uint64_t) رقم_نخها;
    ثابت uint64_t my_شروع = (uint64_t) thread_id * chunk;
    إذا (my_شروع >= مجموع_outputs) {
        أرجع 0;
    }
    uint64_t my_انتها = my_شروع + chunk;
    إذا (my_انتها > مجموع_outputs) {
        my_انتها = مجموع_outputs;
    }

    // Save mask register once; full lanes for vector dot.
    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    ثابت uint64_t در_هر_batch = (uint64_t) M * (uint64_t) n_expert_استفاده_شده;

    uint64_t idx = my_شروع;
    طالما (idx < my_انتها) {
        // Decode (m, slot, batch) from the m-major linear index.
        ثابت int64_t  batch_idx = (int64_t) (idx / در_هر_batch);
        ثابت uint64_t rem       = idx - (uint64_t) batch_idx * در_هر_batch;
        ثابت int64_t  جایگاه_idx  = (int64_t) (rem / (uint64_t) M);
        ثابت int64_t  m0        = (int64_t) (rem - (uint64_t) جایگاه_idx * (uint64_t) M);

        // How many outputs left in this (slot, batch) run AND in my range.
        ثابت uint64_t پیاپی_انتها_global =
            (uint64_t) batch_idx * در_هر_batch + (uint64_t) جایگاه_idx * (uint64_t) M + (uint64_t) M;
        ثابت uint64_t انتها_درون_my = (پیاپی_انتها_global < my_انتها) ? پیاپی_انتها_global : my_انتها;
        int64_t        پیاپی_len   = (int64_t) (انتها_درون_my - idx);

        // Resolve expert + B column + dst slot for this run.
        ثابت int32_t expert_id =
            *(ثابت int32_t *) ((ثابت حرف *) src2_داده + جایگاه_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        حرف * dst_جایگاه = (حرف *) dst_داده + جایگاه_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        إذا (expert_id < 0 || expert_id >= n_expert) {
            // Invalid expert id — zero out this run's outputs.
            int64_t m = m0;
            لكل (int64_t i = 0; i < پیاپی_len; i++, m++) {
                atomic_store_f32((متطاير عائم *) (dst_جایگاه + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) پیاپی_len;
            استمر;
        }

        ثابت int64_t ستون_idx = جایگاه_idx % ne11;
        ثابت عائم * b_ستون_پایه =
            (ثابت عائم *) ((ثابت حرف *) src1_داده + ستون_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ثابت حرف * expert_پایه = (ثابت حرف *) src0_داده + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t چپ_2 = پیاپی_len;

        // Paired-row dots: halves B bandwidth for runs >= 2.
        إذا (use_x2) {
            طالما (چپ_2 >= 2) {
                ثابت بلوک_q4_0 * row0 = (ثابت بلوک_q4_0 *) (expert_پایه + m * (int64_t) nb01);
                ثابت بلوک_q4_0 * row1 = (ثابت بلوک_q4_0 *) (expert_پایه + (m + 1) * (int64_t) nb01);
                عائم              s0, s1;
                q4_dot_compute_x2_aligned(row0, row1, b_ستون_پایه, K_بلوکها, &s0, &s1);
                atomic_store_f32((متطاير عائم *) (dst_جایگاه + m * (int64_t) nbd0), s0);
                atomic_store_f32((متطاير عائم *) (dst_جایگاه + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                چپ_2 -= 2;
            }
        }

        // Tail / non-aligned fallback: single-row dots.
        طالما (چپ_2 > 0) {
            ثابت بلوک_q4_0 * ردیف = (ثابت بلوک_q4_0 *) (expert_پایه + m * (int64_t) nb01);
            عائم              s   = q4_dot_compute(ردیف, b_ستون_پایه, K_بلوکها);
            atomic_store_f32((متطاير عائم *) (dst_جایگاه + m * (int64_t) nbd0), s);
            m++;
            چپ_2--;
        }

        idx += (uint64_t) پیاپی_len;
    }

    q4_dot_انتها(&q4_state);
    أرجع 0;
}
