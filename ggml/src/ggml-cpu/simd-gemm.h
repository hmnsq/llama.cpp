#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#شامل "simd-mappings.h"

// TODO: add support for sizeless vector types
#اگر defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#اگر defined(__AVX512F__) || defined (__ARM_NEON__)
    ایستا constexpr صحیح GEMM_RM = 4;
    ایستا constexpr صحیح GEMM_RN = 4; // 16+4+1 = 25/32
#وگرنهاگر defined(__AVX2__) || defined(__AVX__)
    ایستا constexpr صحیح GEMM_RM = 6;
    ایستا constexpr صحیح GEMM_RN = 2; // 12+2+1 = 15/16
#وگرنه
    ایستا constexpr صحیح GEMM_RM = 2;
    ایستا constexpr صحیح GEMM_RN = 2;
#endif

template <صحیح RM, صحیح RN>
ایستا درونخطی تهی simd_gemm_ukernel(
    اعشاری       * GGML_RESTRICT C,
    ثابت اعشاری * GGML_RESTRICT A,
    ثابت اعشاری * GGML_RESTRICT B,
    صحیح K, صحیح N)
{
    ایستا constexpr صحیح KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    برای (int64_t i = 0; i < RM; i++) {
        برای (صحیح r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    برای (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        برای (صحیح r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        برای (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            برای (صحیح r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    برای (int64_t i = 0; i < RM; i++) {
        برای (صحیح r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

template <صحیح RM>
ایستا درونخطی تهی simd_gemm_ukernel_tail(
    اعشاری       * GGML_RESTRICT C,
    ثابت اعشاری * GGML_RESTRICT A,
    ثابت اعشاری * GGML_RESTRICT B,
    صحیح K, صحیح N, صحیح cols)
{
#اگر defined(__AVX512F__)
    ثابت __mmask16 mask = (1u << cols) - 1;
    __m512 acc[RM];
    برای (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm512_maskz_loadu_ps(mask, C + i * N);
    }
    برای (int64_t kk = 0; kk < K; kk++) {
        ثابت __m512 b = _mm512_maskz_loadu_ps(mask, B + kk * N);
        برای (int64_t i = 0; i < RM; i++) {
            acc[i] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(A[i * K + kk]), b, acc[i], mask);
        }
    }
    برای (int64_t i = 0; i < RM; i++) {
        _mm512_mask_storeu_ps(C + i * N, mask, acc[i]);
    }
#وگرنهاگر defined(__AVX2__)
    ثابت __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(cols), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 acc[RM];
    برای (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm256_maskload_ps(C + i * N, mask);
    }
    برای (int64_t kk = 0; kk < K; kk++) {
        ثابت __m256 b = _mm256_maskload_ps(B + kk * N, mask);
        برای (int64_t i = 0; i < RM; i++) {
            acc[i] = GGML_F32_VEC_FMA(acc[i], b, _mm256_set1_ps(A[i * K + kk]));
        }
    }
    برای (int64_t i = 0; i < RM; i++) {
        _mm256_maskstore_ps(C + i * N, mask, acc[i]);
    }
#وگرنه
    برای (int64_t j = 0; j < cols; j++) {
        برای (int64_t i = 0; i < RM; i++) {
            اعشاری a = C[i * N + j];
            برای (int64_t kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
#endif
}

// C[M x N] += A[M x K] * B[K x N]
ایستا تهی simd_gemm(
    اعشاری       * GGML_RESTRICT C,
    ثابت اعشاری * GGML_RESTRICT A,
    ثابت اعشاری * GGML_RESTRICT B,
    صحیح M, صحیح K, صحیح N)
{
    ایستا constexpr صحیح KN = GGML_F32_EPR;

    int64_t ii = 0;
    برای (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        برای (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        برای (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        اگر (jj < N) {
            simd_gemm_ukernel_tail<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    برای (; ii < M; ii++) {
        int64_t jj = 0;
        برای (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        برای (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        اگر (jj < N) {
            simd_gemm_ukernel_tail<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}
#وگرنهاگر defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
template <صحیح RM>
ایستا درونخطی تهی rvv_simd_gemm_ukernel(
    اعشاری       * GGML_RESTRICT C,
    ثابت اعشاری * GGML_RESTRICT A,
    ثابت اعشاری * GGML_RESTRICT B,
    صحیح K, صحیح N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    اگر constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    اگر constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    اگر constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    اگر constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    اگر constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    اگر constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    برای (صحیح kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        اگر constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        اگر constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        اگر constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        اگر constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        اگر constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        اگر constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    اگر constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    اگر constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    اگر constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    اگر constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    اگر constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    اگر constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

template <صحیح RM>
ایستا درونخطی تهی rvv_simd_gemm_dispatch_tail(
    اعشاری       * GGML_RESTRICT C,
    ثابت اعشاری * GGML_RESTRICT A,
    ثابت اعشاری * GGML_RESTRICT B,
    صحیح K, صحیح N, صحیح KN, صحیح remaining_rows)
{
    اگر constexpr (RM > 0) {
        اگر (remaining_rows == RM) {
            int64_t jj = 0;
            برای (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            اگر (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } وگرنه {
            rvv_simd_gemm_dispatch_tail<RM - 1>(C, A, B, K, N, KN, remaining_rows);
        }
    }
}

ایستا constexpr صحیح GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
ایستا تهی simd_gemm(
    اعشاری       * GGML_RESTRICT C,
    ثابت اعشاری * GGML_RESTRICT A,
    ثابت اعشاری * GGML_RESTRICT B,
    صحیح M, صحیح K, صحیح N)
{
    ثابت صحیح KN = (صحیح)__riscv_vlenb();
    int64_t ii = 0;
    برای (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        برای (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        اگر (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    صحیح remaining_rows = M - ii;
    rvv_simd_gemm_dispatch_tail<GEMM_RM - 1>(C, A, B, K, N, KN, remaining_rows);
}

#اگر defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic برداشت
#endif

#وگرنه // scalar path

ایستا تهی simd_gemm(
    اعشاری       * GGML_RESTRICT C,
    ثابت اعشاری * GGML_RESTRICT A,
    ثابت اعشاری * GGML_RESTRICT B,
    صحیح M, صحیح K, صحیح N)
{
    برای (int64_t i = 0; i < M; i++) {
        برای (int64_t j = 0; j < N; j++) {
            اعشاری sum = C[i * N + j];
            برای (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD
