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

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_solve_tri_params {
    கட்டமைப்பு ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    கட்டமைப்பு ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    கட்டமைப்பு ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

முழு entry_point(கட்டமைப்பு ggml_et_solve_tri_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;  // A
    கட்டமைப்பு ggml_tensor * src1 = &params->src1;  // B
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;   // X

    எனில் (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    மாறிலி தசம * A_data = (மாறிலி தசம *) src0->data;
    மாறிலி தசம * B_data = (மாறிலி தசம *) src1->data;
    தசம *       X_data = (தசம *) dst->data;

    எனில் (!A_data || !B_data || !X_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t n   = src0->ne[1];  // A is n×n
    மாறிலி int64_t k   = src1->ne[0];  // number of RHS columns
    மாறிலி int64_t ne2 = src0->ne[2];
    மாறிலி int64_t ne3 = src0->ne[3];

    // Strides in bytes
    மாறிலி size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    மாறிலி size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    மாறிலி size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    மாறிலி int64_t cols_per_cl    = 16;
    மாறிலி int64_t இலக்கம்_col_groups = k / cols_per_cl;
    மாறிலி int64_t total_work     = இலக்கம்_col_groups * ne2 * ne3;

    க்கு (int64_t work = thread_id; work < total_work; work += இலக்கம்_threads) {
        மாறிலி int64_t cg = work % இலக்கம்_col_groups;
        மாறிலி int64_t i2 = (work / இலக்கம்_col_groups) % ne2;
        மாறிலி int64_t i3 = work / (இலக்கம்_col_groups * ne2);

        மாறிலி int64_t j_start = cg * cols_per_cl;
        மாறிலி int64_t j_end   = j_start + cols_per_cl;

        மாறிலி தசம * A_batch = (மாறிலி தசம *) ((மாறிலி எழுத்து *) A_data + i2 * nb02 + i3 * nb03);
        மாறிலி தசம * B_batch = (மாறிலி தசம *) ((மாறிலி எழுத்து *) B_data + i2 * nb12 + i3 * nb13);
        தசம *       X_batch = (தசம *) ((எழுத்து *) X_data + i2 * nb2 + i3 * nb3);

        க்கு (int64_t j = j_start; j < j_end; j++) {
            க்கு (int64_t i = 0; i < n; i++) {
                மாறிலி தசம * A_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) A_batch + i * nb01);
                தசம *       X_row = (தசம *) ((எழுத்து *) X_batch + i * nb1);
                மாறிலி தசம * B_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) B_batch + i * nb11);

                தசம sum = 0.0f;
                க்கு (int64_t t = 0; t < i; t++) {
                    மாறிலி தசம * X_t = (மாறிலி தசம *) ((மாறிலி எழுத்து *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    திரும்பு 0;
}
