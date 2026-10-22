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

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_solve_tri_params {
    ସଂରଚନା ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    ସଂରଚନା ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    ସଂରଚନା ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_solve_tri_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;  // A
    ସଂରଚନା ggml_tensor * src1 = &params->src1;  // B
    ସଂରଚନା ggml_tensor * dst  = &params->dst;   // X

    ଯଦି (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ ଦଶମିକ * A_data = (ଅଚଳ ଦଶମିକ *) src0->data;
    ଅଚଳ ଦଶମିକ * B_data = (ଅଚଳ ଦଶମିକ *) src1->data;
    ଦଶମିକ *       X_data = (ଦଶମିକ *) dst->data;

    ଯଦି (!A_data || !B_data || !X_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t n   = src0->ne[1];  // A is n×n
    ଅଚଳ int64_t k   = src1->ne[0];  // number of RHS columns
    ଅଚଳ int64_t ne2 = src0->ne[2];
    ଅଚଳ int64_t ne3 = src0->ne[3];

    // Strides in bytes
    ଅଚଳ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ଅଚଳ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ଅଚଳ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    ଅଚଳ int64_t cols_per_cl    = 16;
    ଅଚଳ int64_t ଅଙ୍କ_col_groups = k / cols_per_cl;
    ଅଚଳ int64_t total_work     = ଅଙ୍କ_col_groups * ne2 * ne3;

    ପାଇଁ (int64_t work = thread_id; work < total_work; work += ଅଙ୍କ_threads) {
        ଅଚଳ int64_t cg = work % ଅଙ୍କ_col_groups;
        ଅଚଳ int64_t i2 = (work / ଅଙ୍କ_col_groups) % ne2;
        ଅଚଳ int64_t i3 = work / (ଅଙ୍କ_col_groups * ne2);

        ଅଚଳ int64_t j_start = cg * cols_per_cl;
        ଅଚଳ int64_t j_end   = j_start + cols_per_cl;

        ଅଚଳ ଦଶମିକ * A_batch = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) A_data + i2 * nb02 + i3 * nb03);
        ଅଚଳ ଦଶମିକ * B_batch = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) B_data + i2 * nb12 + i3 * nb13);
        ଦଶମିକ *       X_batch = (ଦଶମିକ *) ((ଅକ୍ଷର *) X_data + i2 * nb2 + i3 * nb3);

        ପାଇଁ (int64_t j = j_start; j < j_end; j++) {
            ପାଇଁ (int64_t i = 0; i < n; i++) {
                ଅଚଳ ଦଶମିକ * A_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) A_batch + i * nb01);
                ଦଶମିକ *       X_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) X_batch + i * nb1);
                ଅଚଳ ଦଶମିକ * B_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) B_batch + i * nb11);

                ଦଶମିକ sum = 0.0f;
                ପାଇଁ (int64_t t = 0; t < i; t++) {
                    ଅଚଳ ଦଶମିକ * X_t = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
