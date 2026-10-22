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

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_solve_tri_params {
    কাঠামো ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    কাঠামো ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    কাঠামো ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_solve_tri_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;  // A
    কাঠামো ggml_tensor * src1 = &params->src1;  // B
    কাঠামো ggml_tensor * dst  = &params->dst;   // X

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    ধ্রুবক দশমিক * A_তথ্য = (ধ্রুবক দশমিক *) src0->তথ্য;
    ধ্রুবক দশমিক * B_তথ্য = (ধ্রুবক দশমিক *) src1->তথ্য;
    দশমিক *       X_তথ্য = (দশমিক *) dst->তথ্য;

    যদি (!A_তথ্য || !B_তথ্য || !X_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t n   = src0->ne[1];  // A is n×n
    ধ্রুবক int64_t k   = src1->ne[0];  // number of RHS columns
    ধ্রুবক int64_t ne2 = src0->ne[2];
    ধ্রুবক int64_t ne3 = src0->ne[3];

    // Strides in bytes
    ধ্রুবক size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ধ্রুবক size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ধ্রুবক size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    ধ্রুবক int64_t cols_per_cl    = 16;
    ধ্রুবক int64_t অঙ্ক_col_groups = k / cols_per_cl;
    ধ্রুবক int64_t যোগ_work     = অঙ্ক_col_groups * ne2 * ne3;

    জন্য (int64_t work = thread_id; work < যোগ_work; work += অঙ্ক_threads) {
        ধ্রুবক int64_t cg = work % অঙ্ক_col_groups;
        ধ্রুবক int64_t i2 = (work / অঙ্ক_col_groups) % ne2;
        ধ্রুবক int64_t i3 = work / (অঙ্ক_col_groups * ne2);

        ধ্রুবক int64_t j_start = cg * cols_per_cl;
        ধ্রুবক int64_t j_end   = j_start + cols_per_cl;

        ধ্রুবক দশমিক * A_batch = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) A_তথ্য + i2 * nb02 + i3 * nb03);
        ধ্রুবক দশমিক * B_batch = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) B_তথ্য + i2 * nb12 + i3 * nb13);
        দশমিক *       X_batch = (দশমিক *) ((অক্ষর *) X_তথ্য + i2 * nb2 + i3 * nb3);

        জন্য (int64_t j = j_start; j < j_end; j++) {
            জন্য (int64_t i = 0; i < n; i++) {
                ধ্রুবক দশমিক * A_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) A_batch + i * nb01);
                দশমিক *       X_row = (দশমিক *) ((অক্ষর *) X_batch + i * nb1);
                ধ্রুবক দশমিক * B_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) B_batch + i * nb11);

                দশমিক যোগ_2 = 0.0f;
                জন্য (int64_t t = 0; t < i; t++) {
                    ধ্রুবক দশমিক * X_t = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) X_batch + t * nb1);
                    যোগ_2 += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - যোগ_2, A_row[i]);
            }
        }
    }

    ফেরত 0;
}
