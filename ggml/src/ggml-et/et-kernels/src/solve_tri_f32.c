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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_solve_tri_params {
    ಸಂರಚನಾ ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    ಸಂರಚನಾ ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    ಸಂರಚನಾ ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_solve_tri_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;  // A
    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;  // B
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;   // X

    ಯದಿ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ ಭಿನ್ನ * A_data = (ಧ್ರುವ ಭಿನ್ನ *) src0->data;
    ಧ್ರುವ ಭಿನ್ನ * B_data = (ಧ್ರುವ ಭಿನ್ನ *) src1->data;
    ಭಿನ್ನ *       X_data = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!A_data || !B_data || !X_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t n   = src0->ne[1];  // A is n×n
    ಧ್ರುವ int64_t k   = src1->ne[0];  // number of RHS columns
    ಧ್ರುವ int64_t ne2 = src0->ne[2];
    ಧ್ರುವ int64_t ne3 = src0->ne[3];

    // Strides in bytes
    ಧ್ರುವ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಧ್ರುವ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ಧ್ರುವ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    ಧ್ರುವ int64_t cols_per_cl    = 16;
    ಧ್ರುವ int64_t ಅಂಕೆ_col_groups = k / cols_per_cl;
    ಧ್ರುವ int64_t total_work     = ಅಂಕೆ_col_groups * ne2 * ne3;

    ಪ್ರತಿ (int64_t work = thread_id; work < total_work; work += ಅಂಕೆ_threads) {
        ಧ್ರುವ int64_t cg = work % ಅಂಕೆ_col_groups;
        ಧ್ರುವ int64_t i2 = (work / ಅಂಕೆ_col_groups) % ne2;
        ಧ್ರುವ int64_t i3 = work / (ಅಂಕೆ_col_groups * ne2);

        ಧ್ರುವ int64_t j_start = cg * cols_per_cl;
        ಧ್ರುವ int64_t j_end   = j_start + cols_per_cl;

        ಧ್ರುವ ಭಿನ್ನ * A_batch = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) A_data + i2 * nb02 + i3 * nb03);
        ಧ್ರುವ ಭಿನ್ನ * B_batch = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) B_data + i2 * nb12 + i3 * nb13);
        ಭಿನ್ನ *       X_batch = (ಭಿನ್ನ *) ((ವರ್ಣ *) X_data + i2 * nb2 + i3 * nb3);

        ಪ್ರತಿ (int64_t j = j_start; j < j_end; j++) {
            ಪ್ರತಿ (int64_t i = 0; i < n; i++) {
                ಧ್ರುವ ಭಿನ್ನ * A_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) A_batch + i * nb01);
                ಭಿನ್ನ *       X_row = (ಭಿನ್ನ *) ((ವರ್ಣ *) X_batch + i * nb1);
                ಧ್ರುವ ಭಿನ್ನ * B_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) B_batch + i * nb11);

                ಭಿನ್ನ sum = 0.0f;
                ಪ್ರತಿ (int64_t t = 0; t < i; t++) {
                    ಧ್ರುವ ಭಿನ್ನ * X_t = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
