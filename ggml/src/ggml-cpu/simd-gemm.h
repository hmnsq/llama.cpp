#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#अन्तर्भाव "simd-mappings.h"

// TODO: add support for sizeless vector types
#सचे defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#सचे defined(__AVX512F__) || defined (__ARM_NEON__)
    स्थिर constexpr पूर्णाङ्क GEMM_RM = 4;
    स्थिर constexpr पूर्णाङ्क GEMM_RN = 4; // 16+4+1 = 25/32
#नोचेद्यदि defined(__AVX2__) || defined(__AVX__)
    स्थिर constexpr पूर्णाङ्क GEMM_RM = 6;
    स्थिर constexpr पूर्णाङ्क GEMM_RN = 2; // 12+2+1 = 15/16
#अञ्ञथा
    स्थिर constexpr पूर्णाङ्क GEMM_RM = 2;
    स्थिर constexpr पूर्णाङ्क GEMM_RN = 2;
#endif

template <पूर्णाङ्क RM, पूर्णाङ्क RN>
स्थिर अन्तःस्थ शून्य simd_gemm_ukernel(
    भिन्न       * GGML_RESTRICT C,
    ध्रुव भिन्न * GGML_RESTRICT A,
    ध्रुव भिन्न * GGML_RESTRICT B,
    पूर्णाङ्क K, पूर्णाङ्क N)
{
    स्थिर constexpr पूर्णाङ्क KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    अत्थाय (int64_t i = 0; i < RM; i++) {
        अत्थाय (पूर्णाङ्क r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    अत्थाय (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        अत्थाय (पूर्णाङ्क r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        अत्थाय (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            अत्थाय (पूर्णाङ्क r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    अत्थाय (int64_t i = 0; i < RM; i++) {
        अत्थाय (पूर्णाङ्क r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

template <पूर्णाङ्क RM>
स्थिर अन्तःस्थ शून्य simd_gemm_ukernel_tail(
    भिन्न       * GGML_RESTRICT C,
    ध्रुव भिन्न * GGML_RESTRICT A,
    ध्रुव भिन्न * GGML_RESTRICT B,
    पूर्णाङ्क K, पूर्णाङ्क N, पूर्णाङ्क cols)
{
#सचे defined(__AVX512F__)
    ध्रुव __mmask16 mask = (1u << cols) - 1;
    __m512 acc[RM];
    अत्थाय (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm512_maskz_loadu_ps(mask, C + i * N);
    }
    अत्थाय (int64_t kk = 0; kk < K; kk++) {
        ध्रुव __m512 b = _mm512_maskz_loadu_ps(mask, B + kk * N);
        अत्थाय (int64_t i = 0; i < RM; i++) {
            acc[i] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(A[i * K + kk]), b, acc[i], mask);
        }
    }
    अत्थाय (int64_t i = 0; i < RM; i++) {
        _mm512_mask_storeu_ps(C + i * N, mask, acc[i]);
    }
#नोचेद्यदि defined(__AVX2__)
    ध्रुव __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(cols), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 acc[RM];
    अत्थाय (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm256_maskload_ps(C + i * N, mask);
    }
    अत्थाय (int64_t kk = 0; kk < K; kk++) {
        ध्रुव __m256 b = _mm256_maskload_ps(B + kk * N, mask);
        अत्थाय (int64_t i = 0; i < RM; i++) {
            acc[i] = GGML_F32_VEC_FMA(acc[i], b, _mm256_set1_ps(A[i * K + kk]));
        }
    }
    अत्थाय (int64_t i = 0; i < RM; i++) {
        _mm256_maskstore_ps(C + i * N, mask, acc[i]);
    }
#अञ्ञथा
    अत्थाय (int64_t j = 0; j < cols; j++) {
        अत्थाय (int64_t i = 0; i < RM; i++) {
            भिन्न a = C[i * N + j];
            अत्थाय (int64_t kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
#endif
}

// C[M x N] += A[M x K] * B[K x N]
स्थिर शून्य simd_gemm(
    भिन्न       * GGML_RESTRICT C,
    ध्रुव भिन्न * GGML_RESTRICT A,
    ध्रुव भिन्न * GGML_RESTRICT B,
    पूर्णाङ्क M, पूर्णाङ्क K, पूर्णाङ्क N)
{
    स्थिर constexpr पूर्णाङ्क KN = GGML_F32_EPR;

    int64_t ii = 0;
    अत्थाय (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        अत्थाय (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        अत्थाय (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        सचे (jj < N) {
            simd_gemm_ukernel_tail<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    अत्थाय (; ii < M; ii++) {
        int64_t jj = 0;
        अत्थाय (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        अत्थाय (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        सचे (jj < N) {
            simd_gemm_ukernel_tail<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}
#नोचेद्यदि defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
template <पूर्णाङ्क RM>
स्थिर अन्तःस्थ शून्य rvv_simd_gemm_ukernel(
    भिन्न       * GGML_RESTRICT C,
    ध्रुव भिन्न * GGML_RESTRICT A,
    ध्रुव भिन्न * GGML_RESTRICT B,
    पूर्णाङ्क K, पूर्णाङ्क N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    सचे constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    सचे constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    सचे constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    सचे constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    सचे constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    सचे constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    अत्थाय (पूर्णाङ्क kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        सचे constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        सचे constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        सचे constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        सचे constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        सचे constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        सचे constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    सचे constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    सचे constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    सचे constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    सचे constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    सचे constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    सचे constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

template <पूर्णाङ्क RM>
स्थिर अन्तःस्थ शून्य rvv_simd_gemm_dispatch_tail(
    भिन्न       * GGML_RESTRICT C,
    ध्रुव भिन्न * GGML_RESTRICT A,
    ध्रुव भिन्न * GGML_RESTRICT B,
    पूर्णाङ्क K, पूर्णाङ्क N, पूर्णाङ्क KN, पूर्णाङ्क remaining_rows)
{
    सचे constexpr (RM > 0) {
        सचे (remaining_rows == RM) {
            int64_t jj = 0;
            अत्थाय (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            सचे (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } अञ्ञथा {
            rvv_simd_gemm_dispatch_tail<RM - 1>(C, A, B, K, N, KN, remaining_rows);
        }
    }
}

स्थिर constexpr पूर्णाङ्क GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
स्थिर शून्य simd_gemm(
    भिन्न       * GGML_RESTRICT C,
    ध्रुव भिन्न * GGML_RESTRICT A,
    ध्रुव भिन्न * GGML_RESTRICT B,
    पूर्णाङ्क M, पूर्णाङ्क K, पूर्णाङ्क N)
{
    ध्रुव पूर्णाङ्क KN = (पूर्णाङ्क)__riscv_vlenb();
    int64_t ii = 0;
    अत्थाय (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        अत्थाय (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        सचे (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    पूर्णाङ्क remaining_rows = M - ii;
    rvv_simd_gemm_dispatch_tail<GEMM_RM - 1>(C, A, B, K, N, KN, remaining_rows);
}

#सचे defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic निष्कासय
#endif

#अञ्ञथा // scalar path

स्थिर शून्य simd_gemm(
    भिन्न       * GGML_RESTRICT C,
    ध्रुव भिन्न * GGML_RESTRICT A,
    ध्रुव भिन्न * GGML_RESTRICT B,
    पूर्णाङ्क M, पूर्णाङ्क K, पूर्णाङ्क N)
{
    अत्थाय (int64_t i = 0; i < M; i++) {
        अत्थाय (int64_t j = 0; j < N; j++) {
            भिन्न sum = C[i * N + j];
            अत्थाय (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD
