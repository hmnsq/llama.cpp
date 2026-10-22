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

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_solve_tri_params {
    ഘടന ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    ഘടന ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    ഘടന ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_solve_tri_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    ഘടന ggml_tensor * src0 = &params->src0;  // A
    ഘടന ggml_tensor * src1 = &params->src1;  // B
    ഘടന ggml_tensor * dst  = &params->dst;   // X

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    സ്ഥിരം ദശാംശം * A_data = (സ്ഥിരം ദശാംശം *) src0->data;
    സ്ഥിരം ദശാംശം * B_data = (സ്ഥിരം ദശാംശം *) src1->data;
    ദശാംശം *       X_data = (ദശാംശം *) dst->data;

    എങ്കിൽ (!A_data || !B_data || !X_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t n   = src0->ne[1];  // A is n×n
    സ്ഥിരം int64_t k   = src1->ne[0];  // number of RHS columns
    സ്ഥിരം int64_t ne2 = src0->ne[2];
    സ്ഥിരം int64_t ne3 = src0->ne[3];

    // Strides in bytes
    സ്ഥിരം size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    സ്ഥിരം size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    സ്ഥിരം size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    സ്ഥിരം int64_t cols_per_cl    = 16;
    സ്ഥിരം int64_t അക്കം_col_groups = k / cols_per_cl;
    സ്ഥിരം int64_t total_work     = അക്കം_col_groups * ne2 * ne3;

    വേണ്ടി (int64_t work = thread_id; work < total_work; work += അക്കം_threads) {
        സ്ഥിരം int64_t cg = work % അക്കം_col_groups;
        സ്ഥിരം int64_t i2 = (work / അക്കം_col_groups) % ne2;
        സ്ഥിരം int64_t i3 = work / (അക്കം_col_groups * ne2);

        സ്ഥിരം int64_t j_start = cg * cols_per_cl;
        സ്ഥിരം int64_t j_end   = j_start + cols_per_cl;

        സ്ഥിരം ദശാംശം * A_batch = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) A_data + i2 * nb02 + i3 * nb03);
        സ്ഥിരം ദശാംശം * B_batch = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) B_data + i2 * nb12 + i3 * nb13);
        ദശാംശം *       X_batch = (ദശാംശം *) ((അക്ഷരം *) X_data + i2 * nb2 + i3 * nb3);

        വേണ്ടി (int64_t j = j_start; j < j_end; j++) {
            വേണ്ടി (int64_t i = 0; i < n; i++) {
                സ്ഥിരം ദശാംശം * A_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) A_batch + i * nb01);
                ദശാംശം *       X_row = (ദശാംശം *) ((അക്ഷരം *) X_batch + i * nb1);
                സ്ഥിരം ദശാംശം * B_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) B_batch + i * nb11);

                ദശാംശം sum = 0.0f;
                വേണ്ടി (int64_t t = 0; t < i; t++) {
                    സ്ഥിരം ദശാംശം * X_t = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    തിരികെ 0;
}
