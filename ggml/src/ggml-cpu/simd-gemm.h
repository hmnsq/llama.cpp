#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#include "simd-mappings.h"

// TODO: add support for sizeless vector types
#ถ้า defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#ถ้า defined(__AVX512F__) || defined (__ARM_NEON__)
    static constexpr จำนวนเต็ม GEMM_RM = 4;
    static constexpr จำนวนเต็ม GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
    static constexpr จำนวนเต็ม GEMM_RM = 6;
    static constexpr จำนวนเต็ม GEMM_RN = 2; // 12+2+1 = 15/16
#มิฉะนั้น
    static constexpr จำนวนเต็ม GEMM_RM = 2;
    static constexpr จำนวนเต็ม GEMM_RN = 2;
#endif

template <จำนวนเต็ม RM, จำนวนเต็ม RN>
static anta_hstha void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    จำนวนเต็ม K, จำนวนเต็ม N)
{
    static constexpr จำนวนเต็ม KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    สำหรับ (int64_t i = 0; i < RM; i++) {
        สำหรับ (จำนวนเต็ม r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    สำหรับ (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        สำหรับ (จำนวนเต็ม r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        สำหรับ (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            สำหรับ (จำนวนเต็ม r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    สำหรับ (int64_t i = 0; i < RM; i++) {
        สำหรับ (จำนวนเต็ม r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

template <จำนวนเต็ม RM>
static anta_hstha void simd_gemm_ukernel_tail(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    จำนวนเต็ม K, จำนวนเต็ม N, จำนวนเต็ม cols)
{
#ถ้า defined(__AVX512F__)
    const __mmask16 mask = (1u << cols) - 1;
    __m512 acc[RM];
    สำหรับ (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm512_maskz_loadu_ps(mask, C + i * N);
    }
    สำหรับ (int64_t kk = 0; kk < K; kk++) {
        const __m512 b = _mm512_maskz_loadu_ps(mask, B + kk * N);
        สำหรับ (int64_t i = 0; i < RM; i++) {
            acc[i] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(A[i * K + kk]), b, acc[i], mask);
        }
    }
    สำหรับ (int64_t i = 0; i < RM; i++) {
        _mm512_mask_storeu_ps(C + i * N, mask, acc[i]);
    }
#elif defined(__AVX2__)
    const __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(cols), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 acc[RM];
    สำหรับ (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm256_maskload_ps(C + i * N, mask);
    }
    สำหรับ (int64_t kk = 0; kk < K; kk++) {
        const __m256 b = _mm256_maskload_ps(B + kk * N, mask);
        สำหรับ (int64_t i = 0; i < RM; i++) {
            acc[i] = GGML_F32_VEC_FMA(acc[i], b, _mm256_set1_ps(A[i * K + kk]));
        }
    }
    สำหรับ (int64_t i = 0; i < RM; i++) {
        _mm256_maskstore_ps(C + i * N, mask, acc[i]);
    }
#มิฉะนั้น
    สำหรับ (int64_t j = 0; j < cols; j++) {
        สำหรับ (int64_t i = 0; i < RM; i++) {
            float a = C[i * N + j];
            สำหรับ (int64_t kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
#endif
}

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    จำนวนเต็ม M, จำนวนเต็ม K, จำนวนเต็ม N)
{
    static constexpr จำนวนเต็ม KN = GGML_F32_EPR;

    int64_t ii = 0;
    สำหรับ (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        สำหรับ (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        สำหรับ (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        ถ้า (jj < N) {
            simd_gemm_ukernel_tail<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    สำหรับ (; ii < M; ii++) {
        int64_t jj = 0;
        สำหรับ (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        สำหรับ (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        ถ้า (jj < N) {
            simd_gemm_ukernel_tail<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}
#elif defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
template <จำนวนเต็ม RM>
static anta_hstha void rvv_simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    จำนวนเต็ม K, จำนวนเต็ม N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    ถ้า constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    ถ้า constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    ถ้า constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    ถ้า constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    ถ้า constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    ถ้า constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    สำหรับ (จำนวนเต็ม kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        ถ้า constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        ถ้า constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        ถ้า constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        ถ้า constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        ถ้า constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        ถ้า constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    ถ้า constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    ถ้า constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    ถ้า constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    ถ้า constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    ถ้า constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    ถ้า constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

template <จำนวนเต็ม RM>
static anta_hstha void rvv_simd_gemm_dispatch_tail(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    จำนวนเต็ม K, จำนวนเต็ม N, จำนวนเต็ม KN, จำนวนเต็ม remaining_rows)
{
    ถ้า constexpr (RM > 0) {
        ถ้า (remaining_rows == RM) {
            int64_t jj = 0;
            สำหรับ (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            ถ้า (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } มิฉะนั้น {
            rvv_simd_gemm_dispatch_tail<RM - 1>(C, A, B, K, N, KN, remaining_rows);
        }
    }
}

static constexpr จำนวนเต็ม GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    จำนวนเต็ม M, จำนวนเต็ม K, จำนวนเต็ม N)
{
    const จำนวนเต็ม KN = (จำนวนเต็ม)__riscv_vlenb();
    int64_t ii = 0;
    สำหรับ (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        สำหรับ (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        ถ้า (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    จำนวนเต็ม remaining_rows = M - ii;
    rvv_simd_gemm_dispatch_tail<GEMM_RM - 1>(C, A, B, K, N, KN, remaining_rows);
}

#ถ้า defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#มิฉะนั้น // scalar path

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    จำนวนเต็ม M, จำนวนเต็ม K, จำนวนเต็ม N)
{
    สำหรับ (int64_t i = 0; i < M; i++) {
        สำหรับ (int64_t j = 0; j < N; j++) {
            float sum = C[i * N + j];
            สำหรับ (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD
