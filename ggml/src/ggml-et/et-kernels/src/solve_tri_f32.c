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

पूर्णांक entry_point(संरचना ggml_et_solve_tri_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;  // A
    संरचना ggml_tensor * src1 = &params->src1;  // B
    संरचना ggml_tensor * dst  = &params->dst;   // X

    जर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    स्थिरांक दशांश * A_data = (स्थिरांक दशांश *) src0->data;
    स्थिरांक दशांश * B_data = (स्थिरांक दशांश *) src1->data;
    दशांश *       X_data = (दशांश *) dst->data;

    जर (!A_data || !B_data || !X_data) {
        परत -1;
    }

    स्थिरांक int64_t n   = src0->ne[1];  // A is n×n
    स्थिरांक int64_t k   = src1->ne[0];  // number of RHS columns
    स्थिरांक int64_t ne2 = src0->ne[2];
    स्थिरांक int64_t ne3 = src0->ne[3];

    // Strides in bytes
    स्थिरांक माप_प्रकार nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    स्थिरांक माप_प्रकार nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    स्थिरांक माप_प्रकार nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    स्थिरांक int64_t cols_per_cl    = 16;
    स्थिरांक int64_t अंक_col_groups = k / cols_per_cl;
    स्थिरांक int64_t total_work     = अंक_col_groups * ne2 * ne3;

    साठी (int64_t work = thread_id; work < total_work; work += अंक_threads) {
        स्थिरांक int64_t cg = work % अंक_col_groups;
        स्थिरांक int64_t i2 = (work / अंक_col_groups) % ne2;
        स्थिरांक int64_t i3 = work / (अंक_col_groups * ne2);

        स्थिरांक int64_t j_start = cg * cols_per_cl;
        स्थिरांक int64_t j_end   = j_start + cols_per_cl;

        स्थिरांक दशांश * A_batch = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) A_data + i2 * nb02 + i3 * nb03);
        स्थिरांक दशांश * B_batch = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) B_data + i2 * nb12 + i3 * nb13);
        दशांश *       X_batch = (दशांश *) ((अक्षर *) X_data + i2 * nb2 + i3 * nb3);

        साठी (int64_t j = j_start; j < j_end; j++) {
            साठी (int64_t i = 0; i < n; i++) {
                स्थिरांक दशांश * A_row = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) A_batch + i * nb01);
                दशांश *       X_row = (दशांश *) ((अक्षर *) X_batch + i * nb1);
                स्थिरांक दशांश * B_row = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) B_batch + i * nb11);

                दशांश sum = 0.0f;
                साठी (int64_t t = 0; t < i; t++) {
                    स्थिरांक दशांश * X_t = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) X_batch + t * nb1);
                    sum += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - sum, A_row[i]);
            }
        }
    }

    परत 0;
}
