#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#অন্তর্ভাব "simd-mappings.h"

// TODO: add support for sizeless vector types
#যদি defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#যদি defined(__AVX512F__) || defined (__ARM_NEON__)
    স্থির constexpr পূর্ণসংখ্যা GEMM_RM = 4;
    স্থির constexpr পূর্ণসংখ্যা GEMM_RN = 4; // 16+4+1 = 25/32
#নতুবা_যদি defined(__AVX2__) || defined(__AVX__)
    স্থির constexpr পূর্ণসংখ্যা GEMM_RM = 6;
    স্থির constexpr পূর্ণসংখ্যা GEMM_RN = 2; // 12+2+1 = 15/16
#অন্যথা
    স্থির constexpr পূর্ণসংখ্যা GEMM_RM = 2;
    স্থির constexpr পূর্ণসংখ্যা GEMM_RN = 2;
#endif

template <পূর্ণসংখ্যা RM, পূর্ণসংখ্যা RN>
স্থির অন্তঃরেখ শূন্য simd_gemm_ukernel(
    দশমিক       * GGML_RESTRICT C,
    ধ্রুবক দশমিক * GGML_RESTRICT A,
    ধ্রুবক দশমিক * GGML_RESTRICT B,
    পূর্ণসংখ্যা K, পূর্ণসংখ্যা N)
{
    স্থির constexpr পূর্ণসংখ্যা KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    জন্য (int64_t i = 0; i < RM; i++) {
        জন্য (পূর্ণসংখ্যা r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    জন্য (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        জন্য (পূর্ণসংখ্যা r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        জন্য (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            জন্য (পূর্ণসংখ্যা r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    জন্য (int64_t i = 0; i < RM; i++) {
        জন্য (পূর্ণসংখ্যা r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

template <পূর্ণসংখ্যা RM>
স্থির অন্তঃরেখ শূন্য simd_gemm_ukernel_লেজ(
    দশমিক       * GGML_RESTRICT C,
    ধ্রুবক দশমিক * GGML_RESTRICT A,
    ধ্রুবক দশমিক * GGML_RESTRICT B,
    পূর্ণসংখ্যা K, পূর্ণসংখ্যা N, পূর্ণসংখ্যা cols)
{
#যদি defined(__AVX512F__)
    ধ্রুবক __mmask16 mask = (1u << cols) - 1;
    __m512 acc[RM];
    জন্য (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm512_maskz_loadu_ps(mask, C + i * N);
    }
    জন্য (int64_t kk = 0; kk < K; kk++) {
        ধ্রুবক __m512 b = _mm512_maskz_loadu_ps(mask, B + kk * N);
        জন্য (int64_t i = 0; i < RM; i++) {
            acc[i] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(A[i * K + kk]), b, acc[i], mask);
        }
    }
    জন্য (int64_t i = 0; i < RM; i++) {
        _mm512_mask_storeu_ps(C + i * N, mask, acc[i]);
    }
#নতুবা_যদি defined(__AVX2__)
    ধ্রুবক __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(cols), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 acc[RM];
    জন্য (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm256_maskload_ps(C + i * N, mask);
    }
    জন্য (int64_t kk = 0; kk < K; kk++) {
        ধ্রুবক __m256 b = _mm256_maskload_ps(B + kk * N, mask);
        জন্য (int64_t i = 0; i < RM; i++) {
            acc[i] = GGML_F32_VEC_FMA(acc[i], b, _mm256_set1_ps(A[i * K + kk]));
        }
    }
    জন্য (int64_t i = 0; i < RM; i++) {
        _mm256_maskstore_ps(C + i * N, mask, acc[i]);
    }
#অন্যথা
    জন্য (int64_t j = 0; j < cols; j++) {
        জন্য (int64_t i = 0; i < RM; i++) {
            দশমিক a = C[i * N + j];
            জন্য (int64_t kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
#endif
}

// C[M x N] += A[M x K] * B[K x N]
স্থির শূন্য simd_gemm(
    দশমিক       * GGML_RESTRICT C,
    ধ্রুবক দশমিক * GGML_RESTRICT A,
    ধ্রুবক দশমিক * GGML_RESTRICT B,
    পূর্ণসংখ্যা M, পূর্ণসংখ্যা K, পূর্ণসংখ্যা N)
{
    স্থির constexpr পূর্ণসংখ্যা KN = GGML_F32_EPR;

    int64_t ii = 0;
    জন্য (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        জন্য (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        জন্য (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        যদি (jj < N) {
            simd_gemm_ukernel_লেজ<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    জন্য (; ii < M; ii++) {
        int64_t jj = 0;
        জন্য (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        জন্য (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        যদি (jj < N) {
            simd_gemm_ukernel_লেজ<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}
#নতুবা_যদি defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
template <পূর্ণসংখ্যা RM>
স্থির অন্তঃরেখ শূন্য rvv_simd_gemm_ukernel(
    দশমিক       * GGML_RESTRICT C,
    ধ্রুবক দশমিক * GGML_RESTRICT A,
    ধ্রুবক দশমিক * GGML_RESTRICT B,
    পূর্ণসংখ্যা K, পূর্ণসংখ্যা N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    যদি constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    যদি constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    যদি constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    যদি constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    যদি constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    যদি constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    জন্য (পূর্ণসংখ্যা kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        যদি constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        যদি constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        যদি constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        যদি constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        যদি constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        যদি constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    যদি constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    যদি constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    যদি constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    যদি constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    যদি constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    যদি constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

template <পূর্ণসংখ্যা RM>
স্থির অন্তঃরেখ শূন্য rvv_simd_gemm_dispatch_লেজ(
    দশমিক       * GGML_RESTRICT C,
    ধ্রুবক দশমিক * GGML_RESTRICT A,
    ধ্রুবক দশমিক * GGML_RESTRICT B,
    পূর্ণসংখ্যা K, পূর্ণসংখ্যা N, পূর্ণসংখ্যা KN, পূর্ণসংখ্যা remaining_rows)
{
    যদি constexpr (RM > 0) {
        যদি (remaining_rows == RM) {
            int64_t jj = 0;
            জন্য (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            যদি (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } অন্যথা {
            rvv_simd_gemm_dispatch_লেজ<RM - 1>(C, A, B, K, N, KN, remaining_rows);
        }
    }
}

স্থির constexpr পূর্ণসংখ্যা GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
স্থির শূন্য simd_gemm(
    দশমিক       * GGML_RESTRICT C,
    ধ্রুবক দশমিক * GGML_RESTRICT A,
    ধ্রুবক দশমিক * GGML_RESTRICT B,
    পূর্ণসংখ্যা M, পূর্ণসংখ্যা K, পূর্ণসংখ্যা N)
{
    ধ্রুবক পূর্ণসংখ্যা KN = (পূর্ণসংখ্যা)__riscv_vlenb();
    int64_t ii = 0;
    জন্য (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        জন্য (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        যদি (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    পূর্ণসংখ্যা remaining_rows = M - ii;
    rvv_simd_gemm_dispatch_লেজ<GEMM_RM - 1>(C, A, B, K, N, KN, remaining_rows);
}

#যদি defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic বের_করো
#endif

#অন্যথা // scalar path

স্থির শূন্য simd_gemm(
    দশমিক       * GGML_RESTRICT C,
    ধ্রুবক দশমিক * GGML_RESTRICT A,
    ধ্রুবক দশমিক * GGML_RESTRICT B,
    পূর্ণসংখ্যা M, পূর্ণসংখ্যা K, পূর্ণসংখ্যা N)
{
    জন্য (int64_t i = 0; i < M; i++) {
        জন্য (int64_t j = 0; j < N; j++) {
            দশমিক যোগ_2 = C[i * N + j];
            জন্য (int64_t kk = 0; kk < K; kk++) {
                যোগ_2 += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = যোগ_2;
        }
    }
}

#endif // GGML_SIMD
