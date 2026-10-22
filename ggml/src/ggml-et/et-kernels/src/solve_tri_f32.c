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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_solve_tri_params {
    ರಚನೆ ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    ರಚನೆ ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    ರಚನೆ ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_solve_tri_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;  // A
    ರಚನೆ ggml_tensor * src1 = &params->src1;  // B
    ರಚನೆ ggml_tensor * dst  = &params->dst;   // X

    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       X_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!A_ದತ್ತಾಂಶ || !B_ದತ್ತಾಂಶ || !X_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t n   = src0->ne[1];  // A is n×n
    ಸ್ಥಿರಾಂಕ int64_t k   = src1->ne[0];  // number of RHS columns
    ಸ್ಥಿರಾಂಕ int64_t ne2 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = src0->ne[3];

    // Strides in bytes
    ಸ್ಥಿರಾಂಕ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    ಸ್ಥಿರಾಂಕ int64_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_cl    = 16;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಲಂಬಸಾಲು_groups = k / ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_cl;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_work     = ಅಂಕೆ_ಲಂಬಸಾಲು_groups * ne2 * ne3;

    ಗಾಗಿ (int64_t work = thread_id; work < ಒಟ್ಟು_work; work += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int64_t cg = work % ಅಂಕೆ_ಲಂಬಸಾಲು_groups;
        ಸ್ಥಿರಾಂಕ int64_t i2 = (work / ಅಂಕೆ_ಲಂಬಸಾಲು_groups) % ne2;
        ಸ್ಥಿರಾಂಕ int64_t i3 = work / (ಅಂಕೆ_ಲಂಬಸಾಲು_groups * ne2);

        ಸ್ಥಿರಾಂಕ int64_t j_ಆರಂಭ = cg * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_cl;
        ಸ್ಥಿರಾಂಕ int64_t j_ಕೊನೆ   = j_ಆರಂಭ + ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_cl;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_batch = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) A_ದತ್ತಾಂಶ + i2 * nb02 + i3 * nb03);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_batch = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) B_ದತ್ತಾಂಶ + i2 * nb12 + i3 * nb13);
        ದಶಮಾಂಶ *       X_batch = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) X_ದತ್ತಾಂಶ + i2 * nb2 + i3 * nb3);

        ಗಾಗಿ (int64_t j = j_ಆರಂಭ; j < j_ಕೊನೆ; j++) {
            ಗಾಗಿ (int64_t i = 0; i < n; i++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) A_batch + i * nb01);
                ದಶಮಾಂಶ *       X_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) X_batch + i * nb1);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) B_batch + i * nb11);

                ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
                ಗಾಗಿ (int64_t t = 0; t < i; t++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * X_t = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) X_batch + t * nb1);
                    ಮೊತ್ತ += A_ಅಡ್ಡಸಾಲು[t] * X_t[j];
                }

                X_ಅಡ್ಡಸಾಲು[j] = et_fdiv(B_ಅಡ್ಡಸಾಲು[j] - ಮೊತ್ತ, A_ಅಡ್ಡಸಾಲು[i]);
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
