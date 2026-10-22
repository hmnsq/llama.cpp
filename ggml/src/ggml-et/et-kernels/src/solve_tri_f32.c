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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_solve_tri_params {
    संरचना ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    संरचना ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    संरचना ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

पूर्णाङ्क entry_point(संरचना ggml_et_solve_tri_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        फर्काऊ -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्क_threads = get_अङ्क_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        फर्काऊ 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        फर्काऊ -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;  // A
    संरचना ggml_tensor * src1 = &params->src1;  // B
    संरचना ggml_tensor * dst  = &params->dst;   // X

    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        फर्काऊ -1;
    }

    अचर भिन्न * A_data = (अचर भिन्न *) src0->data;
    अचर भिन्न * B_data = (अचर भिन्न *) src1->data;
    भिन्न *       X_data = (भिन्न *) dst->data;

    यदि (!A_data || !B_data || !X_data) {
        फर्काऊ -1;
    }

    अचर int64_t n   = src0->ne[1];  // A is n×n
    अचर int64_t k   = src1->ne[0];  // number of RHS columns
    अचर int64_t ne2 = src0->ne[2];
    अचर int64_t ne3 = src0->ne[3];

    // Strides in bytes
    अचर माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    अचर माप_प्रकार nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    अचर माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    अचर int64_t cols_per_cl    = 16;
    अचर int64_t अङ्क_col_groups = k / cols_per_cl;
    अचर int64_t total_work     = अङ्क_col_groups * ne2 * ne3;

    लागि (int64_t work = thread_id; work < total_work; work += अङ्क_threads) {
        अचर int64_t cg = work % अङ्क_col_groups;
        अचर int64_t i2 = (work / अङ्क_col_groups) % ne2;
        अचर int64_t i3 = work / (अङ्क_col_groups * ne2);

        अचर int64_t j_start = cg * cols_per_cl;
        अचर int64_t j_end   = j_start + cols_per_cl;

        अचर भिन्न * A_batch = (अचर भिन्न *) ((अचर वर्ण *) A_data + i2 * nb02 + i3 * nb03);
        अचर भिन्न * B_batch = (अचर भिन्न *) ((अचर वर्ण *) B_data + i2 * nb12 + i3 * nb13);
        भिन्न *       X_batch = (भिन्न *) ((वर्ण *) X_data + i2 * nb2 + i3 * nb3);

        लागि (int64_t j = j_start; j < j_end; j++) {
            लागि (int64_t i = 0; i < n; i++) {
                अचर भिन्न * A_row = (अचर भिन्न *) ((अचर वर्ण *) A_batch + i * nb01);
                भिन्न *       X_row = (भिन्न *) ((वर्ण *) X_batch + i * nb1);
                अचर भिन्न * B_row = (अचर भिन्न *) ((अचर वर्ण *) B_batch + i * nb11);

                भिन्न sum = 0.0f;
                लागि (int64_t t = 0; t < i; t++) {
                    अचर भिन्न * X_t = (अचर भिन्न *) ((अचर वर्ण *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    फर्काऊ 0;
}
