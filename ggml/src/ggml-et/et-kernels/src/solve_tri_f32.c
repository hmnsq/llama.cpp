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

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_solve_tri_params {
    ساختار ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    ساختار ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    ساختار ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

صحیح entry_point(ساختار ggml_et_solve_tri_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    ساختار ggml_tensor * src0 = &params->src0;  // A
    ساختار ggml_tensor * src1 = &params->src1;  // B
    ساختار ggml_tensor * dst  = &params->dst;   // X

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    ثابت اعشاری * A_data = (ثابت اعشاری *) src0->data;
    ثابت اعشاری * B_data = (ثابت اعشاری *) src1->data;
    اعشاری *       X_data = (اعشاری *) dst->data;

    اگر (!A_data || !B_data || !X_data) {
        بازگشت -1;
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
    ثابت int64_t رقم_col_groups = k / cols_per_cl;
    ثابت int64_t total_work     = رقم_col_groups * ne2 * ne3;

    برای (int64_t work = thread_id; work < total_work; work += رقم_threads) {
        ثابت int64_t cg = work % رقم_col_groups;
        ثابت int64_t i2 = (work / رقم_col_groups) % ne2;
        ثابت int64_t i3 = work / (رقم_col_groups * ne2);

        ثابت int64_t j_start = cg * cols_per_cl;
        ثابت int64_t j_end   = j_start + cols_per_cl;

        ثابت اعشاری * A_batch = (ثابت اعشاری *) ((ثابت نویسه *) A_data + i2 * nb02 + i3 * nb03);
        ثابت اعشاری * B_batch = (ثابت اعشاری *) ((ثابت نویسه *) B_data + i2 * nb12 + i3 * nb13);
        اعشاری *       X_batch = (اعشاری *) ((نویسه *) X_data + i2 * nb2 + i3 * nb3);

        برای (int64_t j = j_start; j < j_end; j++) {
            برای (int64_t i = 0; i < n; i++) {
                ثابت اعشاری * A_row = (ثابت اعشاری *) ((ثابت نویسه *) A_batch + i * nb01);
                اعشاری *       X_row = (اعشاری *) ((نویسه *) X_batch + i * nb1);
                ثابت اعشاری * B_row = (ثابت اعشاری *) ((ثابت نویسه *) B_batch + i * nb11);

                اعشاری sum = 0.0f;
                برای (int64_t t = 0; t < i; t++) {
                    ثابت اعشاری * X_t = (ثابت اعشاری *) ((ثابت نویسه *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    بازگشت 0;
}
