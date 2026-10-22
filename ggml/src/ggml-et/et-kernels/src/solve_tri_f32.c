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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_solve_tri_params {
    સંરચના ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    સંરચના ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    સંરચના ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

પૂર્ણાંક entry_point(સંરચના ggml_et_solve_tri_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;  // A
    સંરચના ggml_tensor * src1 = &params->src1;  // B
    સંરચના ggml_tensor * dst  = &params->dst;   // X

    જો (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    અચલ દશાંશ * A_data = (અચલ દશાંશ *) src0->data;
    અચલ દશાંશ * B_data = (અચલ દશાંશ *) src1->data;
    દશાંશ *       X_data = (દશાંશ *) dst->data;

    જો (!A_data || !B_data || !X_data) {
        પાછા -1;
    }

    અચલ int64_t n   = src0->ne[1];  // A is n×n
    અચલ int64_t k   = src1->ne[0];  // number of RHS columns
    અચલ int64_t ne2 = src0->ne[2];
    અચલ int64_t ne3 = src0->ne[3];

    // Strides in bytes
    અચલ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    અચલ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    અચલ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    અચલ int64_t cols_per_cl    = 16;
    અચલ int64_t અંક_col_groups = k / cols_per_cl;
    અચલ int64_t total_work     = અંક_col_groups * ne2 * ne3;

    માટે (int64_t work = thread_id; work < total_work; work += અંક_threads) {
        અચલ int64_t cg = work % અંક_col_groups;
        અચલ int64_t i2 = (work / અંક_col_groups) % ne2;
        અચલ int64_t i3 = work / (અંક_col_groups * ne2);

        અચલ int64_t j_start = cg * cols_per_cl;
        અચલ int64_t j_end   = j_start + cols_per_cl;

        અચલ દશાંશ * A_batch = (અચલ દશાંશ *) ((અચલ અક્ષર *) A_data + i2 * nb02 + i3 * nb03);
        અચલ દશાંશ * B_batch = (અચલ દશાંશ *) ((અચલ અક્ષર *) B_data + i2 * nb12 + i3 * nb13);
        દશાંશ *       X_batch = (દશાંશ *) ((અક્ષર *) X_data + i2 * nb2 + i3 * nb3);

        માટે (int64_t j = j_start; j < j_end; j++) {
            માટે (int64_t i = 0; i < n; i++) {
                અચલ દશાંશ * A_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) A_batch + i * nb01);
                દશાંશ *       X_row = (દશાંશ *) ((અક્ષર *) X_batch + i * nb1);
                અચલ દશાંશ * B_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) B_batch + i * nb11);

                દશાંશ sum = 0.0f;
                માટે (int64_t t = 0; t < i; t++) {
                    અચલ દશાંશ * X_t = (અચલ દશાંશ *) ((અચલ અક્ષર *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    પાછા 0;
}
