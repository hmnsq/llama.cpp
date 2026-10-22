#ਹਦਾਇਤ once

// Computes C[M x N] += A[M x K] * B[K x N]

#ਅਨ੍ਤਰ੍ਭਾਵ "simd-mappings.h"

// TODO: add support for sizeless vector types
#ਜੇ defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#ਜੇ defined(__AVX512F__) || defined (__ARM_NEON__)
    ਸਥਿਰ constexpr ਪੂਰਨ GEMM_RM = 4;
    ਸਥਿਰ constexpr ਪੂਰਨ GEMM_RN = 4; // 16+4+1 = 25/32
#ਜਾਂਜੇ defined(__AVX2__) || defined(__AVX__)
    ਸਥਿਰ constexpr ਪੂਰਨ GEMM_RM = 6;
    ਸਥਿਰ constexpr ਪੂਰਨ GEMM_RN = 2; // 12+2+1 = 15/16
#ਨਹੀਂਤਾਂ
    ਸਥਿਰ constexpr ਪੂਰਨ GEMM_RM = 2;
    ਸਥਿਰ constexpr ਪੂਰਨ GEMM_RN = 2;
#ਜੇ_ਅੰਤ

ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ simd_gemm_ukernel(
    ਦਸ਼ਮਲਵ       * GGML_RESTRICT C,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT A,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT B,
    ਪੂਰਨ K, ਪੂਰਨ N)
{
    ਸਥਿਰ constexpr ਪੂਰਨ KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    ਲਈ (int64_t i = 0; i < RM; i++) {
        ਲਈ (ਪੂਰਨ r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_ਲੱਦੋ(C + i * N + r * KN);
        }
    }

    ਲਈ (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        ਲਈ (ਪੂਰਨ r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_ਲੱਦੋ(B + kk * N + r * KN);
        }
        ਲਈ (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            ਲਈ (ਪੂਰਨ r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    ਲਈ (int64_t i = 0; i < RM; i++) {
        ਲਈ (ਪੂਰਨ r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

ਸਾਂਚਾ <ਪੂਰਨ RM>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ simd_gemm_ukernel_ਪੂਛ(
    ਦਸ਼ਮਲਵ       * GGML_RESTRICT C,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT A,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT B,
    ਪੂਰਨ K, ਪੂਰਨ N, ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_2)
{
#ਜੇ defined(__AVX512F__)
    ਅਚਲ __mmask16 ਨਕਾਬ = (1u << ਖਾਨੇ_ਗਿਣਤੀ_2) - 1;
    __m512 acc[RM];
    ਲਈ (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm512_maskz_loadu_ps(ਨਕਾਬ, C + i * N);
    }
    ਲਈ (int64_t kk = 0; kk < K; kk++) {
        ਅਚਲ __m512 b = _mm512_maskz_loadu_ps(ਨਕਾਬ, B + kk * N);
        ਲਈ (int64_t i = 0; i < RM; i++) {
            acc[i] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(A[i * K + kk]), b, acc[i], ਨਕਾਬ);
        }
    }
    ਲਈ (int64_t i = 0; i < RM; i++) {
        mm512_ਨਕਾਬ_storeu_ps(C + i * N, ਨਕਾਬ, acc[i]);
    }
#ਜਾਂਜੇ defined(__AVX2__)
    ਅਚਲ __m256i ਨਕਾਬ = _mm256_cmpgt_epi32(_mm256_set1_epi32(ਖਾਨੇ_ਗਿਣਤੀ_2), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 acc[RM];
    ਲਈ (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm256_maskload_ps(C + i * N, ਨਕਾਬ);
    }
    ਲਈ (int64_t kk = 0; kk < K; kk++) {
        ਅਚਲ __m256 b = _mm256_maskload_ps(B + kk * N, ਨਕਾਬ);
        ਲਈ (int64_t i = 0; i < RM; i++) {
            acc[i] = GGML_F32_VEC_FMA(acc[i], b, _mm256_set1_ps(A[i * K + kk]));
        }
    }
    ਲਈ (int64_t i = 0; i < RM; i++) {
        _mm256_maskstore_ps(C + i * N, ਨਕਾਬ, acc[i]);
    }
#ਨਹੀਂਤਾਂ
    ਲਈ (int64_t j = 0; j < ਖਾਨੇ_ਗਿਣਤੀ_2; j++) {
        ਲਈ (int64_t i = 0; i < RM; i++) {
            ਦਸ਼ਮਲਵ a = C[i * N + j];
            ਲਈ (int64_t kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
#ਜੇ_ਅੰਤ
}

// C[M x N] += A[M x K] * B[K x N]
ਸਥਿਰ ਖਾਲੀ simd_gemm(
    ਦਸ਼ਮਲਵ       * GGML_RESTRICT C,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT A,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT B,
    ਪੂਰਨ M, ਪੂਰਨ K, ਪੂਰਨ N)
{
    ਸਥਿਰ constexpr ਪੂਰਨ KN = GGML_F32_EPR;

    int64_t ii = 0;
    ਲਈ (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        ਲਈ (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        ਲਈ (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        ਜੇ (jj < N) {
            simd_gemm_ukernel_ਪੂਛ<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    ਲਈ (; ii < M; ii++) {
        int64_t jj = 0;
        ਲਈ (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        ਲਈ (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        ਜੇ (jj < N) {
            simd_gemm_ukernel_ਪੂਛ<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}
#ਜਾਂਜੇ defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
ਸਾਂਚਾ <ਪੂਰਨ RM>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_simd_gemm_ukernel(
    ਦਸ਼ਮਲਵ       * GGML_RESTRICT C,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT A,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT B,
    ਪੂਰਨ K, ਪੂਰਨ N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    ਜੇ constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    ਜੇ constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    ਜੇ constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    ਜੇ constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    ਜੇ constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    ਜੇ constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    ਲਈ (ਪੂਰਨ kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        ਜੇ constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        ਜੇ constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        ਜੇ constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        ਜੇ constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        ਜੇ constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        ਜੇ constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    ਜੇ constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    ਜੇ constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    ਜੇ constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    ਜੇ constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    ਜੇ constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    ਜੇ constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

ਸਾਂਚਾ <ਪੂਰਨ RM>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rvv_simd_gemm_dispatch_ਪੂਛ(
    ਦਸ਼ਮਲਵ       * GGML_RESTRICT C,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT A,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT B,
    ਪੂਰਨ K, ਪੂਰਨ N, ਪੂਰਨ KN, ਪੂਰਨ ਬਚਿਆ_ਪੰਕਤੀਆਂ)
{
    ਜੇ constexpr (RM > 0) {
        ਜੇ (ਬਚਿਆ_ਪੰਕਤੀਆਂ == RM) {
            int64_t jj = 0;
            ਲਈ (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            ਜੇ (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } ਨਹੀਂਤਾਂ {
            rvv_simd_gemm_dispatch_ਪੂਛ<RM - 1>(C, A, B, K, N, KN, ਬਚਿਆ_ਪੰਕਤੀਆਂ);
        }
    }
}

ਸਥਿਰ constexpr ਪੂਰਨ GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
ਸਥਿਰ ਖਾਲੀ simd_gemm(
    ਦਸ਼ਮਲਵ       * GGML_RESTRICT C,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT A,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT B,
    ਪੂਰਨ M, ਪੂਰਨ K, ਪੂਰਨ N)
{
    ਅਚਲ ਪੂਰਨ KN = (ਪੂਰਨ)__riscv_vlenb();
    int64_t ii = 0;
    ਲਈ (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        ਲਈ (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        ਜੇ (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    ਪੂਰਨ ਬਚਿਆ_ਪੰਕਤੀਆਂ = M - ii;
    rvv_simd_gemm_dispatch_ਪੂਛ<GEMM_RM - 1>(C, A, B, K, N, KN, ਬਚਿਆ_ਪੰਕਤੀਆਂ);
}

#ਜੇ defined(__GNUC__) && !defined(__clang__)
#ਹਦਾਇਤ GCC diagnostic ਕੱਢੋ
#ਜੇ_ਅੰਤ

#ਨਹੀਂਤਾਂ // scalar path

ਸਥਿਰ ਖਾਲੀ simd_gemm(
    ਦਸ਼ਮਲਵ       * GGML_RESTRICT C,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT A,
    ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT B,
    ਪੂਰਨ M, ਪੂਰਨ K, ਪੂਰਨ N)
{
    ਲਈ (int64_t i = 0; i < M; i++) {
        ਲਈ (int64_t j = 0; j < N; j++) {
            ਦਸ਼ਮਲਵ ਜੋੜ = C[i * N + j];
            ਲਈ (int64_t kk = 0; kk < K; kk++) {
                ਜੋੜ += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = ਜੋੜ;
        }
    }
}

#ਜੇ_ਅੰਤ // GGML_SIMD
