//******************************************************************************
// MUL_MAT_ID kernel specialized for Q8_0 weights (Mixture of Experts).
//
// C[m, s, b] = Sum(k=0..K-1) A[k, m, ids[s,b]] * B[k, s % ne11, b]
//   A: Q8_0  [K, M, n_expert]   weights
//   B: F32   [K, n_cols, batch] activations
//   ids: I32 [n_expert_used, batch]
//   C: F32   [M, n_expert_used, batch]
//
// Strategy mirrors mul_mat_id_Q4_0.c.
//******************************************************************************

#ضمن "block_ops.h"
#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"
#ضمن "quants.h"

#ضمن <stdint.h>

صحيح اندراج_نقطہ(بنية ggml_et_mul_mat_id_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    إذا (!kernel_env || !params) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);
    إذا (thread_id < 0) {
        أرجع 0;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * src2 = &params->src2;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت فراغ *    src0_انکڑے = src0->انکڑے;
    ثابت عائم *   src1_انکڑے = (ثابت عائم *) src1->انکڑے;
    ثابت int32_t * src2_انکڑے = (ثابت int32_t *) src2->انکڑے;
    عائم *         dst_انکڑے  = (عائم *) dst->انکڑے;
    إذا (!src0_انکڑے || !src1_انکڑے || !src2_انکڑے || !dst_انکڑے) {
        أرجع -1;
    }

    ثابت int64_t K             = src0->ne[0];
    ثابت int64_t M             = src0->ne[1];
    ثابت int64_t n_expert      = src0->ne[2];
    ثابت int64_t n_expert_ورتیا = src2->ne[0];
    ثابت int64_t batch         = src2->ne[1];
    ثابت int64_t ne11          = src1->ne[1];

    إذا (K % QK8_0 != 0) {
        أرجع -1;
    }

    ثابت size_t nb01 = src0->nb[1];
    ثابت size_t nb02 = src0->nb[2];
    ثابت size_t nb11 = src1->nb[1];
    ثابت size_t nb12 = src1->nb[2];
    ثابت size_t nb20 = src2->nb[0];
    ثابت size_t nb21 = src2->nb[1];
    ثابت size_t nbd0 = dst->nb[0];
    ثابت size_t nbd1 = dst->nb[1];
    ثابت size_t nbd2 = dst->nb[2];

    إذا (src0->nb[0] != حجم(بلاک_q8_0) || src1->nb[0] != حجم(عائم) || src2->nb[0] != حجم(int32_t) ||
        nbd0 != حجم(عائم)) {
        أرجع -1;
    }

    ثابت int64_t K_بلاکاں = K / QK8_0;
    ثابت صحيح     use_x2   = ((nb01 & 31) == 0);

    ثابت uint64_t کل_outputs = (uint64_t) M * (uint64_t) n_expert_ورتیا * (uint64_t) batch;
    إذا (کل_outputs == 0) {
        أرجع 0;
    }

    ثابت uint64_t chunk    = (کل_outputs + (uint64_t) عدد_دھاگے - 1) / (uint64_t) عدد_دھاگے;
    ثابت uint64_t my_شروع = (uint64_t) thread_id * chunk;
    إذا (my_شروع >= کل_outputs) {
        أرجع 0;
    }
    uint64_t my_سرا = my_شروع + chunk;
    إذا (my_سرا > کل_outputs) {
        my_سرا = کل_outputs;
    }

    q8_dot_state q8_state;
    q8_dot_begin(&q8_state);

    ثابت uint64_t فی_batch = (uint64_t) M * (uint64_t) n_expert_ورتیا;

    uint64_t idx = my_شروع;
    طالما (idx < my_سرا) {
        ثابت int64_t  batch_idx = (int64_t) (idx / فی_batch);
        ثابت uint64_t rem       = idx - (uint64_t) batch_idx * فی_batch;
        ثابت int64_t  تھاں_idx  = (int64_t) (rem / (uint64_t) M);
        ثابت int64_t  m0        = (int64_t) (rem - (uint64_t) تھاں_idx * (uint64_t) M);

        ثابت uint64_t لگاتار_سرا_global =
            (uint64_t) batch_idx * فی_batch + (uint64_t) تھاں_idx * (uint64_t) M + (uint64_t) M;
        ثابت uint64_t سرا_وچ_my = (لگاتار_سرا_global < my_سرا) ? لگاتار_سرا_global : my_سرا;
        int64_t        لگاتار_len   = (int64_t) (سرا_وچ_my - idx);

        ثابت int32_t expert_id =
            *(ثابت int32_t *) ((ثابت حرف *) src2_انکڑے + تھاں_idx * (int64_t) nb20 + batch_idx * (int64_t) nb21);

        حرف * dst_تھاں = (حرف *) dst_انکڑے + تھاں_idx * (int64_t) nbd1 + batch_idx * (int64_t) nbd2;

        إذا (expert_id < 0 || expert_id >= n_expert) {
            int64_t m = m0;
            لكل (int64_t i = 0; i < لگاتار_len; i++, m++) {
                atomic_store_f32((متطاير عائم *) (dst_تھاں + m * (int64_t) nbd0), 0.0f);
            }
            idx += (uint64_t) لگاتار_len;
            استمر;
        }

        ثابت int64_t خانہ_idx = تھاں_idx % ne11;
        ثابت عائم * b_خانہ_ادھار =
            (ثابت عائم *) ((ثابت حرف *) src1_انکڑے + خانہ_idx * (int64_t) nb11 + batch_idx * (int64_t) nb12);
        ثابت حرف * expert_ادھار = (ثابت حرف *) src0_انکڑے + expert_id * (int64_t) nb02;

        int64_t m    = m0;
        int64_t بایاں = لگاتار_len;

        إذا (use_x2) {
            طالما (بایاں >= 2) {
                ثابت بلاک_q8_0 * row0 = (ثابت بلاک_q8_0 *) (expert_ادھار + m * (int64_t) nb01);
                ثابت بلاک_q8_0 * row1 = (ثابت بلاک_q8_0 *) (expert_ادھار + (m + 1) * (int64_t) nb01);
                عائم              s0, s1;
                q8_dot_compute_x2_aligned(row0, row1, b_خانہ_ادھار, K_بلاکاں, &s0, &s1);
                atomic_store_f32((متطاير عائم *) (dst_تھاں + m * (int64_t) nbd0), s0);
                atomic_store_f32((متطاير عائم *) (dst_تھاں + (m + 1) * (int64_t) nbd0), s1);
                m += 2;
                بایاں -= 2;
            }
        }

        طالما (بایاں > 0) {
            ثابت بلاک_q8_0 * صف = (ثابت بلاک_q8_0 *) (expert_ادھار + m * (int64_t) nb01);
            عائم              s   = q8_dot_compute(صف, b_خانہ_ادھار, K_بلاکاں);
            atomic_store_f32((متطاير عائم *) (dst_تھاں + m * (int64_t) nbd0), s);
            m++;
            بایاں--;
        }

        idx += (uint64_t) لگاتار_len;
    }

    q8_dot_سرا(&q8_state);
    أرجع 0;
}
