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

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_solve_tri_params {
    काष्ठा ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    काष्ठा ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    काष्ठा ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

पूर्णांक entry_point(काष्ठा ggml_et_solve_tri_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;  // A
    काष्ठा ggml_tensor * src1 = &params->src1;  // B
    काष्ठा ggml_tensor * dst  = &params->dst;   // X

    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    स्थिर भग्न * A_तथ्य = (स्थिर भग्न *) src0->तथ्य;
    स्थिर भग्न * B_तथ्य = (स्थिर भग्न *) src1->तथ्य;
    भग्न *       X_तथ्य = (भग्न *) dst->तथ्य;

    अगर (!A_तथ्य || !B_तथ्य || !X_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t n   = src0->ne[1];  // A is n×n
    स्थिर int64_t k   = src1->ne[0];  // number of RHS columns
    स्थिर int64_t ne2 = src0->ne[2];
    स्थिर int64_t ne3 = src0->ne[3];

    // Strides in bytes
    स्थिर माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    स्थिर माप_प्रकार nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    स्थिर माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    स्थिर int64_t cols_per_cl    = 16;
    स्थिर int64_t अंक_col_groups = k / cols_per_cl;
    स्थिर int64_t योग_work     = अंक_col_groups * ne2 * ne3;

    क्रम (int64_t work = thread_id; work < योग_work; work += अंक_threads) {
        स्थिर int64_t cg = work % अंक_col_groups;
        स्थिर int64_t i2 = (work / अंक_col_groups) % ne2;
        स्थिर int64_t i3 = work / (अंक_col_groups * ne2);

        स्थिर int64_t j_start = cg * cols_per_cl;
        स्थिर int64_t j_end   = j_start + cols_per_cl;

        स्थिर भग्न * A_batch = (स्थिर भग्न *) ((स्थिर अक्षर *) A_तथ्य + i2 * nb02 + i3 * nb03);
        स्थिर भग्न * B_batch = (स्थिर भग्न *) ((स्थिर अक्षर *) B_तथ्य + i2 * nb12 + i3 * nb13);
        भग्न *       X_batch = (भग्न *) ((अक्षर *) X_तथ्य + i2 * nb2 + i3 * nb3);

        क्रम (int64_t j = j_start; j < j_end; j++) {
            क्रम (int64_t i = 0; i < n; i++) {
                स्थिर भग्न * A_row = (स्थिर भग्न *) ((स्थिर अक्षर *) A_batch + i * nb01);
                भग्न *       X_row = (भग्न *) ((अक्षर *) X_batch + i * nb1);
                स्थिर भग्न * B_row = (स्थिर भग्न *) ((स्थिर अक्षर *) B_batch + i * nb11);

                भग्न योग_2 = 0.0f;
                क्रम (int64_t t = 0; t < i; t++) {
                    स्थिर भग्न * X_t = (स्थिर भग्न *) ((स्थिर अक्षर *) X_batch + t * nb1);
                    योग_2 += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - योग_2, A_row[i]);
            }
        }
    }

    वापस 0;
}
