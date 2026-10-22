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

সংরচনা ggml_et_solve_tri_params {
    সংরচনা ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    সংরচনা ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    সংরচনা ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_solve_tri_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        প্রত্যাগম -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        প্রত্যাগম 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        প্রত্যাগম -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;  // A
    সংরচনা ggml_tensor * src1 = &params->src1;  // B
    সংরচনা ggml_tensor * dst  = &params->dst;   // X

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;
    }

    ধ্রুব ভিন্ন * A_data = (ধ্রুব ভিন্ন *) src0->data;
    ধ্রুব ভিন্ন * B_data = (ধ্রুব ভিন্ন *) src1->data;
    ভিন্ন *       X_data = (ভিন্ন *) dst->data;

    যদি (!A_data || !B_data || !X_data) {
        প্রত্যাগম -1;
    }

    ধ্রুব int64_t n   = src0->ne[1];  // A is n×n
    ধ্রুব int64_t k   = src1->ne[0];  // number of RHS columns
    ধ্রুব int64_t ne2 = src0->ne[2];
    ধ্রুব int64_t ne3 = src0->ne[3];

    // Strides in bytes
    ধ্রুব size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ধ্রুব size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ধ্রুব size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    ধ্রুব int64_t cols_per_cl    = 16;
    ধ্রুব int64_t অঙ্ক_col_groups = k / cols_per_cl;
    ধ্রুব int64_t total_work     = অঙ্ক_col_groups * ne2 * ne3;

    প্রতি (int64_t work = thread_id; work < total_work; work += অঙ্ক_threads) {
        ধ্রুব int64_t cg = work % অঙ্ক_col_groups;
        ধ্রুব int64_t i2 = (work / অঙ্ক_col_groups) % ne2;
        ধ্রুব int64_t i3 = work / (অঙ্ক_col_groups * ne2);

        ধ্রুব int64_t j_start = cg * cols_per_cl;
        ধ্রুব int64_t j_end   = j_start + cols_per_cl;

        ধ্রুব ভিন্ন * A_batch = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) A_data + i2 * nb02 + i3 * nb03);
        ধ্রুব ভিন্ন * B_batch = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) B_data + i2 * nb12 + i3 * nb13);
        ভিন্ন *       X_batch = (ভিন্ন *) ((বর্ণ *) X_data + i2 * nb2 + i3 * nb3);

        প্রতি (int64_t j = j_start; j < j_end; j++) {
            প্রতি (int64_t i = 0; i < n; i++) {
                ধ্রুব ভিন্ন * A_row = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) A_batch + i * nb01);
                ভিন্ন *       X_row = (ভিন্ন *) ((বর্ণ *) X_batch + i * nb1);
                ধ্রুব ভিন্ন * B_row = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) B_batch + i * nb11);

                ভিন্ন sum = 0.0f;
                প্রতি (int64_t t = 0; t < i; t++) {
                    ধ্রুব ভিন্ন * X_t = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    প্রত্যাগম 0;
}
