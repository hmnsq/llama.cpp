#ಸೂಚನೆ once

// Computes C[M x N] += A[M x K] * B[K x N]

#ಅನ್ತರ್ಭಾವ "simd-mappings.h"

// TODO: add support for sizeless vector types
#ಆದರೆ defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#ಆದರೆ defined(__AVX512F__) || defined (__ARM_NEON__)
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GEMM_RM = 4;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GEMM_RN = 4; // 16+4+1 = 25/32
#ಅಥವಾಆದರೆ defined(__AVX2__) || defined(__AVX__)
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GEMM_RM = 6;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GEMM_RN = 2; // 12+2+1 = 15/16
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GEMM_RM = 2;
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GEMM_RN = 2;
#ಆದರೆ_ಅಂತ್ಯ

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ simd_gemm_ukernel(
    ದಶಮಾಂಶ       * GGML_RESTRICT C,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT A,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT B,
    ಪೂರ್ಣಾಂಕ K, ಪೂರ್ಣಾಂಕ N)
{
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_ತುಂಬು(C + i * N + r * KN);
        }
    }

    ಗಾಗಿ (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_ತುಂಬು(B + kk * N + r * KN);
        }
        ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ simd_gemm_ukernel_ಬಾಲ(
    ದಶಮಾಂಶ       * GGML_RESTRICT C,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT A,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT B,
    ಪೂರ್ಣಾಂಕ K, ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2)
{
#ಆದರೆ defined(__AVX512F__)
    ಸ್ಥಿರಾಂಕ __mmask16 ಮುಖವಾಡ = (1u << ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2) - 1;
    __m512 acc[RM];
    ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm512_maskz_loadu_ps(ಮುಖವಾಡ, C + i * N);
    }
    ಗಾಗಿ (int64_t kk = 0; kk < K; kk++) {
        ಸ್ಥಿರಾಂಕ __m512 b = _mm512_maskz_loadu_ps(ಮುಖವಾಡ, B + kk * N);
        ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
            acc[i] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(A[i * K + kk]), b, acc[i], ಮುಖವಾಡ);
        }
    }
    ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
        mm512_ಮುಖವಾಡ_storeu_ps(C + i * N, ಮುಖವಾಡ, acc[i]);
    }
#ಅಥವಾಆದರೆ defined(__AVX2__)
    ಸ್ಥಿರಾಂಕ __m256i ಮುಖವಾಡ = _mm256_cmpgt_epi32(_mm256_set1_epi32(ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 acc[RM];
    ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm256_maskload_ps(C + i * N, ಮುಖವಾಡ);
    }
    ಗಾಗಿ (int64_t kk = 0; kk < K; kk++) {
        ಸ್ಥಿರಾಂಕ __m256 b = _mm256_maskload_ps(B + kk * N, ಮುಖವಾಡ);
        ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
            acc[i] = GGML_F32_VEC_FMA(acc[i], b, _mm256_set1_ps(A[i * K + kk]));
        }
    }
    ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
        _mm256_maskstore_ps(C + i * N, ಮುಖವಾಡ, acc[i]);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಗಾಗಿ (int64_t j = 0; j < ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2; j++) {
        ಗಾಗಿ (int64_t i = 0; i < RM; i++) {
            ದಶಮಾಂಶ a = C[i * N + j];
            ಗಾಗಿ (int64_t kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
#ಆದರೆ_ಅಂತ್ಯ
}

// C[M x N] += A[M x K] * B[K x N]
ಸ್ಥಿರ ಶೂನ್ಯ simd_gemm(
    ದಶಮಾಂಶ       * GGML_RESTRICT C,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT A,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT B,
    ಪೂರ್ಣಾಂಕ M, ಪೂರ್ಣಾಂಕ K, ಪೂರ್ಣಾಂಕ N)
{
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ KN = GGML_F32_EPR;

    int64_t ii = 0;
    ಗಾಗಿ (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        ಗಾಗಿ (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        ಗಾಗಿ (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        ಆದರೆ (jj < N) {
            simd_gemm_ukernel_ಬಾಲ<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    ಗಾಗಿ (; ii < M; ii++) {
        int64_t jj = 0;
        ಗಾಗಿ (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        ಗಾಗಿ (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        ಆದರೆ (jj < N) {
            simd_gemm_ukernel_ಬಾಲ<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}
#ಅಥವಾಆದರೆ defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_simd_gemm_ukernel(
    ದಶಮಾಂಶ       * GGML_RESTRICT C,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT A,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT B,
    ಪೂರ್ಣಾಂಕ K, ಪೂರ್ಣಾಂಕ N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    ಆದರೆ constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    ಆದರೆ constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    ಆದರೆ constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    ಆದರೆ constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    ಆದರೆ constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    ಆದರೆ constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        ಆದರೆ constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        ಆದರೆ constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        ಆದರೆ constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        ಆದರೆ constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        ಆದರೆ constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        ಆದರೆ constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    ಆದರೆ constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    ಆದರೆ constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    ಆದರೆ constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    ಆದರೆ constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    ಆದರೆ constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    ಆದರೆ constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rvv_simd_gemm_dispatch_ಬಾಲ(
    ದಶಮಾಂಶ       * GGML_RESTRICT C,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT A,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT B,
    ಪೂರ್ಣಾಂಕ K, ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ KN, ಪೂರ್ಣಾಂಕ ಉಳಿದ_ಅಡ್ಡಸಾಲುಗಳು)
{
    ಆದರೆ constexpr (RM > 0) {
        ಆದರೆ (ಉಳಿದ_ಅಡ್ಡಸಾಲುಗಳು == RM) {
            int64_t jj = 0;
            ಗಾಗಿ (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            ಆದರೆ (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            rvv_simd_gemm_dispatch_ಬಾಲ<RM - 1>(C, A, B, K, N, KN, ಉಳಿದ_ಅಡ್ಡಸಾಲುಗಳು);
        }
    }
}

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
ಸ್ಥಿರ ಶೂನ್ಯ simd_gemm(
    ದಶಮಾಂಶ       * GGML_RESTRICT C,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT A,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT B,
    ಪೂರ್ಣಾಂಕ M, ಪೂರ್ಣಾಂಕ K, ಪೂರ್ಣಾಂಕ N)
{
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ KN = (ಪೂರ್ಣಾಂಕ)__riscv_vlenb();
    int64_t ii = 0;
    ಗಾಗಿ (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        ಗಾಗಿ (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        ಆದರೆ (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    ಪೂರ್ಣಾಂಕ ಉಳಿದ_ಅಡ್ಡಸಾಲುಗಳು = M - ii;
    rvv_simd_gemm_dispatch_ಬಾಲ<GEMM_RM - 1>(C, A, B, K, N, KN, ಉಳಿದ_ಅಡ್ಡಸಾಲುಗಳು);
}

#ಆದರೆ defined(__GNUC__) && !defined(__clang__)
#ಸೂಚನೆ GCC diagnostic ತೆಗೆ
#ಆದರೆ_ಅಂತ್ಯ

#ಇಲ್ಲದಿದ್ದರೆ // scalar path

ಸ್ಥಿರ ಶೂನ್ಯ simd_gemm(
    ದಶಮಾಂಶ       * GGML_RESTRICT C,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT A,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT B,
    ಪೂರ್ಣಾಂಕ M, ಪೂರ್ಣಾಂಕ K, ಪೂರ್ಣಾಂಕ N)
{
    ಗಾಗಿ (int64_t i = 0; i < M; i++) {
        ಗಾಗಿ (int64_t j = 0; j < N; j++) {
            ದಶಮಾಂಶ ಮೊತ್ತ = C[i * N + j];
            ಗಾಗಿ (int64_t kk = 0; kk < K; kk++) {
                ಮೊತ್ತ += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = ಮೊತ್ತ;
        }
    }
}

#ಆದರೆ_ಅಂತ್ಯ // GGML_SIMD
