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

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_solve_tri_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;  // A
    संरचना ggml_tensor * src1 = &params->src1;  // B
    संरचना ggml_tensor * dst  = &params->dst;   // X

    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न * A_दत्त = (ध्रुव भिन्न *) src0->दत्त;
    ध्रुव भिन्न * B_दत्त = (ध्रुव भिन्न *) src1->दत्त;
    भिन्न *       X_दत्त = (भिन्न *) dst->दत्त;

    यदि (!A_दत्त || !B_दत्त || !X_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t n   = src0->ne[1];  // A is n×n
    ध्रुव int64_t k   = src1->ne[0];  // number of RHS columns
    ध्रुव int64_t ne2 = src0->ne[2];
    ध्रुव int64_t ne3 = src0->ne[3];

    // Strides in bytes
    ध्रुव size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ध्रुव size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ध्रुव size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    ध्रुव int64_t स्तम्भसङ्ख्या_प्रत्येकम्_cl    = 16;
    ध्रुव int64_t अङ्कः_col_groups = k / स्तम्भसङ्ख्या_प्रत्येकम्_cl;
    ध्रुव int64_t योग_work     = अङ्कः_col_groups * ne2 * ne3;

    प्रति (int64_t work = thread_id; work < योग_work; work += अङ्कः_तन्तवः) {
        ध्रुव int64_t cg = work % अङ्कः_col_groups;
        ध्रुव int64_t i2 = (work / अङ्कः_col_groups) % ne2;
        ध्रुव int64_t i3 = work / (अङ्कः_col_groups * ne2);

        ध्रुव int64_t j_start = cg * स्तम्भसङ्ख्या_प्रत्येकम्_cl;
        ध्रुव int64_t j_अन्तः   = j_start + स्तम्भसङ्ख्या_प्रत्येकम्_cl;

        ध्रुव भिन्न * A_batch = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) A_दत्त + i2 * nb02 + i3 * nb03);
        ध्रुव भिन्न * B_batch = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) B_दत्त + i2 * nb12 + i3 * nb13);
        भिन्न *       X_batch = (भिन्न *) ((वर्ण *) X_दत्त + i2 * nb2 + i3 * nb3);

        प्रति (int64_t j = j_start; j < j_अन्तः; j++) {
            प्रति (int64_t i = 0; i < n; i++) {
                ध्रुव भिन्न * A_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) A_batch + i * nb01);
                भिन्न *       X_row = (भिन्न *) ((वर्ण *) X_batch + i * nb1);
                ध्रुव भिन्न * B_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) B_batch + i * nb11);

                भिन्न योग_2 = 0.0f;
                प्रति (int64_t t = 0; t < i; t++) {
                    ध्रुव भिन्न * X_t = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) X_batch + t * nb1);
                    योग_2 += A_row[t] * X_t[j];
                }

                X_row[j] = et_fdiv(B_row[j] - योग_2, A_row[i]);
            }
        }
    }

    प्रत्यागम 0;
}
