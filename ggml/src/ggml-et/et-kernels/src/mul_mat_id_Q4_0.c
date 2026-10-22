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

صحيح داخلا_نقطو(بنية ggml_et_mul_mat_id_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    إذا (!kernel_env || !params) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);
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

    ثابت فراغ *    src0_معلومات = src0->معلومات;
    ثابت عائم *   src1_معلومات = (ثابت عائم *) src1->معلومات;
    ثابت int32_t * src2_معلومات = (ثابت int32_t *) src2->معلومات;
    عائم *         dst_معلومات  = (عائم *) dst->معلومات;
    إذا (!src0_معلومات || !src1_معلومات || !src2_معلومات || !dst_معلومات) {
        أرجع -1;
    }

    ثابت int64_t K             = src0->ne[0];
    ثابت int64_t M             = src0->ne[1];
    ثابت int64_t n_expert      = src0->ne[2];
    ثابت int64_t n_expert_استعمال_ٿيل = src2->ne[0];
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

    إذا (src0->nb[0] != حجم(بلاڪ_q4_0) || src1->nb[0] != حجم(عائم) || src2->nb[0] != حجم(int32_t) ||
        nbd0 != حجم(عائم)) {
        أرجع -1;
    }

    ثابت int64_t K_بلاڪون = K / QK4_0;
    ثابت صحيح     use_x2   = ((nb01 & 31) == 0);

    ثابت uint64_t ڪل_outputs = (uint64_t) M * (uint64_t) n_expert_استعمال_ٿيل * (uint64_t) batch;
    إذا (ڪل_outputs == 0) {
        أرجع 0;
    }

    // Even partition: hart h owns outputs [h*chunk, (h+1)*chunk).
    ثابت uint64_t chunk    = (ڪل_outputs + (uint64_t) انگ_ڌاڳا - 1) / (uint64_t) انگ_ڌاڳا;
    ثابت uint64_t my_شروعات = (uint64_t) thread_id * chunk;
    إذا (my_شروعات >= ڪل_outputs) {
        أرجع 0;
    }
    uint64_t my_پڇاڙي = my_شروعات + chunk;
    إذا (my_پڇاڙي > ڪل_outputs) {
        my_پڇاڙي = ڪل_outputs;
    }

    // Save mask register once; full lanes for vector dot.
    q4_dot_state q4_state;
    q4_dot_begin(&q4_state);

    ثابت uint64_t في_batch = (uint64_t) M * (uint64_t) n_expert_استعمال_ٿيل;

    uint64_t idx = my_شروعات;
    طالما (idx < my_پڇاڙي) {
        // Decode (m, slot, batch) from the m-major linear index.
        ثابت int64_t  batch_idx = (int64_t) (idx / في_batch);
        ثابت uint64_t rem       = idx - (uint64_t) batch_idx * في_batch;
        ثابت int64_t  جڳهه_idx  = (int64_t) (rem / (uint64_t) M);
        ثابت int64_t  m0        = (int64_t) (rem - (uint64_t) جڳهه_idx * (uint64_t) M);

        // How many outputs left in this (slot, batch) run AND in my range.
        ثابت uint64_t لاڳيتو_پڇاڙي_global =
            (uint64_t) batch_idx * في_batch + (uint64_t) جڳهه_idx * (uint64_t) M + (uint64_t) M;
        ثابت uint64_t پڇاڙي_اندريون_my = (لاڳيتو_پڇاڙي_global < my_پڇاڙي) ? لاڳيتو_پڇاڙي_global : my_پڇاڙي;
        int64_t        لاڳيتو_len   = (int64_t) (پڇاڙي_اندريون_my - idx);

        // Resolve expert + B column + dst slot for this run.
        ثابت int32_t expert_id =
            *(ثابت int32_t *) ((ثابت حرف *) src2_معلومات + جڳهه_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        حرف * dst_جڳهه = (حرف *) dst_معلومات + جڳهه_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        إذا (expert_id < 0 || expert_id >= n_expert) {
            // Invalid expert id — zero out this run's outputs.
            int64_t m = m0;
            لكل (int64_t i = 0; i < لاڳيتو_len; i++, m++) {
                atomic_store_f32((متطاير عائم *) (dst_جڳهه + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) لاڳيتو_len;
            استمر;
        }

        ثابت int64_t کانو_idx = جڳهه_idx % ne11;
        ثابت عائم * b_کانو_بنياد =
            (ثابت عائم *) ((ثابت حرف *) src1_معلومات + کانو_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ثابت حرف * expert_بنياد = (ثابت حرف *) src0_معلومات + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t کاٻو = لاڳيتو_len;

        // Paired-row dots: halves B bandwidth for runs >= 2.
        إذا (use_x2) {
            طالما (کاٻو >= 2) {
                ثابت بلاڪ_q4_0 * row0 = (ثابت بلاڪ_q4_0 *) (expert_بنياد + m * (int64_t) nb01);
                ثابت بلاڪ_q4_0 * row1 = (ثابت بلاڪ_q4_0 *) (expert_بنياد + (m + 1) * (int64_t) nb01);
                عائم              s0, s1;
                q4_dot_compute_x2_aligned(row0, row1, b_کانو_بنياد, K_بلاڪون, &s0, &s1);
                atomic_store_f32((متطاير عائم *) (dst_جڳهه + m * (int64_t) nbd0), s0);
                atomic_store_f32((متطاير عائم *) (dst_جڳهه + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                کاٻو -= 2;
            }
        }

        // Tail / non-aligned fallback: single-row dots.
        طالما (کاٻو > 0) {
            ثابت بلاڪ_q4_0 * صف = (ثابت بلاڪ_q4_0 *) (expert_بنياد + m * (int64_t) nb01);
            عائم              s   = q4_dot_compute(صف, b_کانو_بنياد, K_بلاڪون);
            atomic_store_f32((متطاير عائم *) (dst_جڳهه + m * (int64_t) nbd0), s);
            m++;
            کاٻو--;
        }

        idx += (uint64_t) لاڳيتو_len;
    }

    q4_dot_پڇاڙي(&q4_state);
    أرجع 0;
}
