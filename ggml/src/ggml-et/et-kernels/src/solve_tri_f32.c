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

صحيح entry_point(بنية ggml_et_solve_tri_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    אם (!kernel_env) {
        החזר -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح num_threads = get_num_threads(kernel_env->shire_mask);

    אם (thread_id < 0) {
        החזר 0;
    }

    אם (params == 0 || ((uint64_t) params & 0x7) != 0) {
        החזר -1;
    }

    بنية ggml_tensor * src0 = &params->src0;  // A
    بنية ggml_tensor * src1 = &params->src1;  // B
    بنية ggml_tensor * dst  = &params->dst;   // X

    אם (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        החזר -1;
    }

    ثابت عائم * A_data = (ثابت عائم *) src0->data;
    ثابت عائم * B_data = (ثابت عائم *) src1->data;
    عائم *       X_data = (عائم *) dst->data;

    אם (!A_data || !B_data || !X_data) {
        החזר -1;
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
    ثابت int64_t cols_per_cl    = 16;
    ثابت int64_t num_col_groups = k / cols_per_cl;
    ثابت int64_t total_work     = num_col_groups * ne2 * ne3;

    עבור (int64_t work = thread_id; work < total_work; work += num_threads) {
        ثابت int64_t cg = work % num_col_groups;
        ثابت int64_t i2 = (work / num_col_groups) % ne2;
        ثابت int64_t i3 = work / (num_col_groups * ne2);

        ثابت int64_t j_start = cg * cols_per_cl;
        ثابت int64_t j_end   = j_start + cols_per_cl;

        ثابت عائم * A_batch = (ثابت عائم *) ((ثابت حرف *) A_data + i2 * nb02 + i3 * nb03);
        ثابت عائم * B_batch = (ثابت عائم *) ((ثابت حرف *) B_data + i2 * nb12 + i3 * nb13);
        عائم *       X_batch = (عائم *) ((حرف *) X_data + i2 * nb2 + i3 * nb3);

        עבור (int64_t j = j_start; j < j_end; j++) {
            עבור (int64_t i = 0; i < n; i++) {
                ثابت عائم * A_row = (ثابت عائم *) ((ثابت حرف *) A_batch + i * nb01);
                عائم *       X_row = (عائم *) ((حرف *) X_batch + i * nb1);
                ثابت عائم * B_row = (ثابت عائم *) ((ثابت حرف *) B_batch + i * nb11);

                عائم sum = 0.0f;
                עבור (int64_t t = 0; t < i; t++) {
                    ثابت عائم * X_t = (ثابت عائم *) ((ثابت حرف *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    החזר 0;
}
