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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_solve_tri_params {
    ਬਣਤਰ ggml_tensor src0;  // A: lower-triangular [n, n, B1, B2]
    ਬਣਤਰ ggml_tensor src1;  // B: RHS [k, n, B1, B2]
    ਬਣਤਰ ggml_tensor dst;   // X: solution [k, n, B1, B2]
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_solve_tri_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;  // A
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;  // B
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;   // X

    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * A_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * B_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       X_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!A_ਅੰਕੜੇ || !B_ਅੰਕੜੇ || !X_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t n   = src0->ne[1];  // A is n×n
    ਅਚਲ int64_t k   = src1->ne[0];  // number of RHS columns
    ਅਚਲ int64_t ne2 = src0->ne[2];
    ਅਚਲ int64_t ne3 = src0->ne[3];

    // Strides in bytes
    ਅਚਲ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ਅਚਲ size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    ਅਚਲ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // k % 16 == 0 guaranteed by supports_op. Rows are cache-line aligned,
    // so column groups of 16 map to exclusive cache lines.
    // TODO: Vectorize the thing
    ਅਚਲ int64_t ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_cl    = 16;
    ਅਚਲ int64_t ਅੰਕ_ਖਾਨਾ_groups = k / ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_cl;
    ਅਚਲ int64_t ਕੁੱਲ_work     = ਅੰਕ_ਖਾਨਾ_groups * ne2 * ne3;

    ਲਈ (int64_t work = thread_id; work < ਕੁੱਲ_work; work += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int64_t cg = work % ਅੰਕ_ਖਾਨਾ_groups;
        ਅਚਲ int64_t i2 = (work / ਅੰਕ_ਖਾਨਾ_groups) % ne2;
        ਅਚਲ int64_t i3 = work / (ਅੰਕ_ਖਾਨਾ_groups * ne2);

        ਅਚਲ int64_t j_ਸ਼ੁਰੂ = cg * ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_cl;
        ਅਚਲ int64_t j_ਸਿਰਾ   = j_ਸ਼ੁਰੂ + ਖਾਨੇ_ਗਿਣਤੀ_ਪ੍ਰਤੀ_cl;

        ਅਚਲ ਦਸ਼ਮਲਵ * A_batch = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) A_ਅੰਕੜੇ + i2 * nb02 + i3 * nb03);
        ਅਚਲ ਦਸ਼ਮਲਵ * B_batch = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) B_ਅੰਕੜੇ + i2 * nb12 + i3 * nb13);
        ਦਸ਼ਮਲਵ *       X_batch = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) X_ਅੰਕੜੇ + i2 * nb2 + i3 * nb3);

        ਲਈ (int64_t j = j_ਸ਼ੁਰੂ; j < j_ਸਿਰਾ; j++) {
            ਲਈ (int64_t i = 0; i < n; i++) {
                ਅਚਲ ਦਸ਼ਮਲਵ * A_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) A_batch + i * nb01);
                ਦਸ਼ਮਲਵ *       X_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) X_batch + i * nb1);
                ਅਚਲ ਦਸ਼ਮਲਵ * B_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) B_batch + i * nb11);

                ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
                ਲਈ (int64_t t = 0; t < i; t++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * X_t = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) X_batch + t * nb1);
                    ਜੋੜ += A_ਪੰਕਤੀ[t] * X_t[j];
                }

                X_ਪੰਕਤੀ[j] = et_fdiv(B_ਪੰਕਤੀ[j] - ਜੋੜ, A_ਪੰਕਤੀ[i]);
            }
        }
    }

    ਵਾਪਸ 0;
}
