#సూచన once

// Computes C[M x N] += A[M x K] * B[K x N]

#అన్తర్భావ "simd-mappings.h"

// TODO: add support for sizeless vector types
#ఒకవేళ defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#ఒకవేళ defined(__AVX512F__) || defined (__ARM_NEON__)
    స్థిర constexpr పూర్ణాంకం GEMM_RM = 4;
    స్థిర constexpr పూర్ణాంకం GEMM_RN = 4; // 16+4+1 = 25/32
#లేదావేళ defined(__AVX2__) || defined(__AVX__)
    స్థిర constexpr పూర్ణాంకం GEMM_RM = 6;
    స్థిర constexpr పూర్ణాంకం GEMM_RN = 2; // 12+2+1 = 15/16
#లేకపోతే
    స్థిర constexpr పూర్ణాంకం GEMM_RM = 2;
    స్థిర constexpr పూర్ణాంకం GEMM_RN = 2;
#అయితే_ముగింపు

మూస <పూర్ణాంకం RM, పూర్ణాంకం RN>
స్థిర అంతఃవరుస శూన్య simd_gemm_ukernel(
    దశాంశం       * GGML_RESTRICT C,
    స్థిరరాశి దశాంశం * GGML_RESTRICT A,
    స్థిరరాశి దశాంశం * GGML_RESTRICT B,
    పూర్ణాంకం K, పూర్ణాంకం N)
{
    స్థిర constexpr పూర్ణాంకం KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    కొరకు (int64_t i = 0; i < RM; i++) {
        కొరకు (పూర్ణాంకం r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_ఎక్కించు(C + i * N + r * KN);
        }
    }

    కొరకు (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        కొరకు (పూర్ణాంకం r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_ఎక్కించు(B + kk * N + r * KN);
        }
        కొరకు (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            కొరకు (పూర్ణాంకం r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    కొరకు (int64_t i = 0; i < RM; i++) {
        కొరకు (పూర్ణాంకం r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

మూస <పూర్ణాంకం RM>
స్థిర అంతఃవరుస శూన్య simd_gemm_ukernel_తోక(
    దశాంశం       * GGML_RESTRICT C,
    స్థిరరాశి దశాంశం * GGML_RESTRICT A,
    స్థిరరాశి దశాంశం * GGML_RESTRICT B,
    పూర్ణాంకం K, పూర్ణాంకం N, పూర్ణాంకం నిలువుల_సంఖ్య_2)
{
#ఒకవేళ defined(__AVX512F__)
    స్థిరరాశి __mmask16 ముసుగు = (1u << నిలువుల_సంఖ్య_2) - 1;
    __m512 acc[RM];
    కొరకు (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm512_maskz_loadu_ps(ముసుగు, C + i * N);
    }
    కొరకు (int64_t kk = 0; kk < K; kk++) {
        స్థిరరాశి __m512 b = _mm512_maskz_loadu_ps(ముసుగు, B + kk * N);
        కొరకు (int64_t i = 0; i < RM; i++) {
            acc[i] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(A[i * K + kk]), b, acc[i], ముసుగు);
        }
    }
    కొరకు (int64_t i = 0; i < RM; i++) {
        mm512_ముసుగు_storeu_ps(C + i * N, ముసుగు, acc[i]);
    }
#లేదావేళ defined(__AVX2__)
    స్థిరరాశి __m256i ముసుగు = _mm256_cmpgt_epi32(_mm256_set1_epi32(నిలువుల_సంఖ్య_2), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 acc[RM];
    కొరకు (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm256_maskload_ps(C + i * N, ముసుగు);
    }
    కొరకు (int64_t kk = 0; kk < K; kk++) {
        స్థిరరాశి __m256 b = _mm256_maskload_ps(B + kk * N, ముసుగు);
        కొరకు (int64_t i = 0; i < RM; i++) {
            acc[i] = GGML_F32_VEC_FMA(acc[i], b, _mm256_set1_ps(A[i * K + kk]));
        }
    }
    కొరకు (int64_t i = 0; i < RM; i++) {
        _mm256_maskstore_ps(C + i * N, ముసుగు, acc[i]);
    }
#లేకపోతే
    కొరకు (int64_t j = 0; j < నిలువుల_సంఖ్య_2; j++) {
        కొరకు (int64_t i = 0; i < RM; i++) {
            దశాంశం a = C[i * N + j];
            కొరకు (int64_t kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
#అయితే_ముగింపు
}

// C[M x N] += A[M x K] * B[K x N]
స్థిర శూన్య simd_gemm(
    దశాంశం       * GGML_RESTRICT C,
    స్థిరరాశి దశాంశం * GGML_RESTRICT A,
    స్థిరరాశి దశాంశం * GGML_RESTRICT B,
    పూర్ణాంకం M, పూర్ణాంకం K, పూర్ణాంకం N)
{
    స్థిర constexpr పూర్ణాంకం KN = GGML_F32_EPR;

    int64_t ii = 0;
    కొరకు (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        కొరకు (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        కొరకు (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        ఒకవేళ (jj < N) {
            simd_gemm_ukernel_తోక<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    కొరకు (; ii < M; ii++) {
        int64_t jj = 0;
        కొరకు (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        కొరకు (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        ఒకవేళ (jj < N) {
            simd_gemm_ukernel_తోక<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}
#లేదావేళ defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
మూస <పూర్ణాంకం RM>
స్థిర అంతఃవరుస శూన్య rvv_simd_gemm_ukernel(
    దశాంశం       * GGML_RESTRICT C,
    స్థిరరాశి దశాంశం * GGML_RESTRICT A,
    స్థిరరాశి దశాంశం * GGML_RESTRICT B,
    పూర్ణాంకం K, పూర్ణాంకం N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    ఒకవేళ constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    ఒకవేళ constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    ఒకవేళ constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    ఒకవేళ constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    ఒకవేళ constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    ఒకవేళ constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    కొరకు (పూర్ణాంకం kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        ఒకవేళ constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        ఒకవేళ constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        ఒకవేళ constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        ఒకవేళ constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        ఒకవేళ constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        ఒకవేళ constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    ఒకవేళ constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    ఒకవేళ constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    ఒకవేళ constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    ఒకవేళ constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    ఒకవేళ constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    ఒకవేళ constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

మూస <పూర్ణాంకం RM>
స్థిర అంతఃవరుస శూన్య rvv_simd_gemm_dispatch_తోక(
    దశాంశం       * GGML_RESTRICT C,
    స్థిరరాశి దశాంశం * GGML_RESTRICT A,
    స్థిరరాశి దశాంశం * GGML_RESTRICT B,
    పూర్ణాంకం K, పూర్ణాంకం N, పూర్ణాంకం KN, పూర్ణాంకం మిగిలిన_అడ్డువరుసలు)
{
    ఒకవేళ constexpr (RM > 0) {
        ఒకవేళ (మిగిలిన_అడ్డువరుసలు == RM) {
            int64_t jj = 0;
            కొరకు (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            ఒకవేళ (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } లేకపోతే {
            rvv_simd_gemm_dispatch_తోక<RM - 1>(C, A, B, K, N, KN, మిగిలిన_అడ్డువరుసలు);
        }
    }
}

స్థిర constexpr పూర్ణాంకం GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
స్థిర శూన్య simd_gemm(
    దశాంశం       * GGML_RESTRICT C,
    స్థిరరాశి దశాంశం * GGML_RESTRICT A,
    స్థిరరాశి దశాంశం * GGML_RESTRICT B,
    పూర్ణాంకం M, పూర్ణాంకం K, పూర్ణాంకం N)
{
    స్థిరరాశి పూర్ణాంకం KN = (పూర్ణాంకం)__riscv_vlenb();
    int64_t ii = 0;
    కొరకు (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        కొరకు (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        ఒకవేళ (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    పూర్ణాంకం మిగిలిన_అడ్డువరుసలు = M - ii;
    rvv_simd_gemm_dispatch_తోక<GEMM_RM - 1>(C, A, B, K, N, KN, మిగిలిన_అడ్డువరుసలు);
}

#ఒకవేళ defined(__GNUC__) && !defined(__clang__)
#సూచన GCC diagnostic తీయి
#అయితే_ముగింపు

#లేకపోతే // scalar path

స్థిర శూన్య simd_gemm(
    దశాంశం       * GGML_RESTRICT C,
    స్థిరరాశి దశాంశం * GGML_RESTRICT A,
    స్థిరరాశి దశాంశం * GGML_RESTRICT B,
    పూర్ణాంకం M, పూర్ణాంకం K, పూర్ణాంకం N)
{
    కొరకు (int64_t i = 0; i < M; i++) {
        కొరకు (int64_t j = 0; j < N; j++) {
            దశాంశం కూడిక = C[i * N + j];
            కొరకు (int64_t kk = 0; kk < K; kk++) {
                కూడిక += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = కూడిక;
        }
    }
}

#అయితే_ముగింపు // GGML_SIMD
