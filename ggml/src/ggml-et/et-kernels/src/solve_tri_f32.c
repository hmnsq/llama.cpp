//******************************************************************************
// Solve Triangular F32 Kernel
// Forward substitution: solve AX = B where A is lower-triangular.
//
// src0 (A): [n, n, B1, B2]  lower-triangular matrix
// src1 (B): [k, n, B1, B2]  right-hand side
// dst  (X): [k, n, B1, B2]  solution
//
// For each column j (parallelized across threads):
//   For i = 0..n-1:
//     X[i,j] = (B[i,j] - dot(A[i,0..i-1], X[0..i-1,j])) / A[i,i]
//
// Lower-triangular, left-side, non-unit variant implemented.
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_solve_tri_params {
    بنية ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    بنية ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    بنية ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

صحيح داخلا_نقطو(بنية ggml_et_solve_tri_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;  // A
    بنية ggml_tensor * src1 = &params->src1;  // B
    بنية ggml_tensor * dst  = &params->dst;   // X

    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت عائم * A_معلومات = (ثابت عائم *) src0->معلومات;
    ثابت عائم * B_معلومات = (ثابت عائم *) src1->معلومات;
    عائم *       X_معلومات = (عائم *) dst->معلومات;

    إذا (!A_معلومات || !B_معلومات || !X_معلومات) {
        أرجع -1;
    }

    ثابت int64_t n   = src0->ne[1];  // A is n×n
    ثابت int64_t k   = src1->ne[0];  // number of RHS columns
    ثابت int64_t ne2 = src0->ne[2];
    ثابت int64_t ne3 = src0->ne[3];

    // Strides in bytes
    ثابت size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ثابت size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    ثابت int64_t کانن_ڳڻپ_في_cl    = 16;
    ثابت int64_t انگ_کانو_groups = k / کانن_ڳڻپ_في_cl;
    ثابت int64_t ڪل_work     = انگ_کانو_groups * ne2 * ne3;

    لكل (int64_t work = thread_id; work < ڪل_work; work += انگ_ڌاڳا) {
        ثابت int64_t cg = work % انگ_کانو_groups;
        ثابت int64_t i2 = (work / انگ_کانو_groups) % ne2;
        ثابت int64_t i3 = work / (انگ_کانو_groups * ne2);

        ثابت int64_t j_شروعات = cg * کانن_ڳڻپ_في_cl;
        ثابت int64_t j_پڇاڙي   = j_شروعات + کانن_ڳڻپ_في_cl;

        ثابت عائم * A_batch = (ثابت عائم *) ((ثابت حرف *) A_معلومات + i2 * nb02 + i3 * nb03);
        ثابت عائم * B_batch = (ثابت عائم *) ((ثابت حرف *) B_معلومات + i2 * nb12 + i3 * nb13);
        عائم *       X_batch = (عائم *) ((حرف *) X_معلومات + i2 * nb2 + i3 * nb3);

        لكل (int64_t j = j_شروعات; j < j_پڇاڙي; j++) {
            لكل (int64_t i = 0; i < n; i++) {
                ثابت عائم * A_صف = (ثابت عائم *) ((ثابت حرف *) A_batch + i * nb01);
                عائم *       X_صف = (عائم *) ((حرف *) X_batch + i * nb1);
                ثابت عائم * B_صف = (ثابت عائم *) ((ثابت حرف *) B_batch + i * nb11);

                عائم جوڙ = 0.0f;
                لكل (int64_t t = 0; t < i; t++) {
                    ثابت عائم * X_t = (ثابت عائم *) ((ثابت حرف *) X_batch + t * nb1);
                    جوڙ += A_صف[t] * X_t[j];
                }

                X_صف[j] = et_fdiv(B_صف[j] - جوڙ, A_صف[i]);
            }
        }
    }

    أرجع 0;
}
