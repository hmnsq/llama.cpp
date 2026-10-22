#आशय once

// Computes C[M x N] += A[M x K] * B[K x N]

#समावेश "simd-mappings.h"

// TODO: add support for sizeless vector types
#अगर defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#अगर defined(__AVX512F__) || defined (__ARM_NEON__)
    अटल constexpr पूर्णांक GEMM_RM = 4;
    अटल constexpr पूर्णांक GEMM_RN = 4; // 16+4+1 = 25/32
#या_अगर defined(__AVX2__) || defined(__AVX__)
    अटल constexpr पूर्णांक GEMM_RM = 6;
    अटल constexpr पूर्णांक GEMM_RN = 2; // 12+2+1 = 15/16
#अन्यथा
    अटल constexpr पूर्णांक GEMM_RM = 2;
    अटल constexpr पूर्णांक GEMM_RN = 2;
#पूर्ण_अगर

ढाँचा <पूर्णांक RM, पूर्णांक RN>
अटल अंतरभूत व्योम simd_gemm_ukernel(
    भग्न       * GGML_RESTRICT C,
    स्थिर भग्न * GGML_RESTRICT A,
    स्थिर भग्न * GGML_RESTRICT B,
    पूर्णांक K, पूर्णांक N)
{
    अटल constexpr पूर्णांक KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    क्रम (int64_t i = 0; i < RM; i++) {
        क्रम (पूर्णांक r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    क्रम (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        क्रम (पूर्णांक r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        क्रम (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            क्रम (पूर्णांक r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    क्रम (int64_t i = 0; i < RM; i++) {
        क्रम (पूर्णांक r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

ढाँचा <पूर्णांक RM>
अटल अंतरभूत व्योम simd_gemm_ukernel_पुच्छ(
    भग्न       * GGML_RESTRICT C,
    स्थिर भग्न * GGML_RESTRICT A,
    स्थिर भग्न * GGML_RESTRICT B,
    पूर्णांक K, पूर्णांक N, पूर्णांक cols)
{
#अगर defined(__AVX512F__)
    स्थिर __mmask16 mask = (1u << cols) - 1;
    __m512 acc[RM];
    क्रम (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm512_maskz_loadu_ps(mask, C + i * N);
    }
    क्रम (int64_t kk = 0; kk < K; kk++) {
        स्थिर __m512 b = _mm512_maskz_loadu_ps(mask, B + kk * N);
        क्रम (int64_t i = 0; i < RM; i++) {
            acc[i] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(A[i * K + kk]), b, acc[i], mask);
        }
    }
    क्रम (int64_t i = 0; i < RM; i++) {
        _mm512_mask_storeu_ps(C + i * N, mask, acc[i]);
    }
#या_अगर defined(__AVX2__)
    स्थिर __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(cols), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 acc[RM];
    क्रम (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm256_maskload_ps(C + i * N, mask);
    }
    क्रम (int64_t kk = 0; kk < K; kk++) {
        स्थिर __m256 b = _mm256_maskload_ps(B + kk * N, mask);
        क्रम (int64_t i = 0; i < RM; i++) {
            acc[i] = GGML_F32_VEC_FMA(acc[i], b, _mm256_set1_ps(A[i * K + kk]));
        }
    }
    क्रम (int64_t i = 0; i < RM; i++) {
        _mm256_maskstore_ps(C + i * N, mask, acc[i]);
    }
#अन्यथा
    क्रम (int64_t j = 0; j < cols; j++) {
        क्रम (int64_t i = 0; i < RM; i++) {
            भग्न a = C[i * N + j];
            क्रम (int64_t kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
#पूर्ण_अगर
}

// C[M x N] += A[M x K] * B[K x N]
अटल व्योम simd_gemm(
    भग्न       * GGML_RESTRICT C,
    स्थिर भग्न * GGML_RESTRICT A,
    स्थिर भग्न * GGML_RESTRICT B,
    पूर्णांक M, पूर्णांक K, पूर्णांक N)
{
    अटल constexpr पूर्णांक KN = GGML_F32_EPR;

    int64_t ii = 0;
    क्रम (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        क्रम (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        क्रम (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        अगर (jj < N) {
            simd_gemm_ukernel_पुच्छ<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    क्रम (; ii < M; ii++) {
        int64_t jj = 0;
        क्रम (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        क्रम (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        अगर (jj < N) {
            simd_gemm_ukernel_पुच्छ<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}
#या_अगर defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
ढाँचा <पूर्णांक RM>
अटल अंतरभूत व्योम rvv_simd_gemm_ukernel(
    भग्न       * GGML_RESTRICT C,
    स्थिर भग्न * GGML_RESTRICT A,
    स्थिर भग्न * GGML_RESTRICT B,
    पूर्णांक K, पूर्णांक N, माप_प्रकार vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    अगर constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    अगर constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    अगर constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    अगर constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    अगर constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    अगर constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    क्रम (पूर्णांक kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        अगर constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        अगर constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        अगर constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        अगर constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        अगर constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        अगर constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    अगर constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    अगर constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    अगर constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    अगर constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    अगर constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    अगर constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

ढाँचा <पूर्णांक RM>
अटल अंतरभूत व्योम rvv_simd_gemm_dispatch_पुच्छ(
    भग्न       * GGML_RESTRICT C,
    स्थिर भग्न * GGML_RESTRICT A,
    स्थिर भग्न * GGML_RESTRICT B,
    पूर्णांक K, पूर्णांक N, पूर्णांक KN, पूर्णांक remaining_rows)
{
    अगर constexpr (RM > 0) {
        अगर (remaining_rows == RM) {
            int64_t jj = 0;
            क्रम (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            अगर (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } अन्यथा {
            rvv_simd_gemm_dispatch_पुच्छ<RM - 1>(C, A, B, K, N, KN, remaining_rows);
        }
    }
}

अटल constexpr पूर्णांक GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
अटल व्योम simd_gemm(
    भग्न       * GGML_RESTRICT C,
    स्थिर भग्न * GGML_RESTRICT A,
    स्थिर भग्न * GGML_RESTRICT B,
    पूर्णांक M, पूर्णांक K, पूर्णांक N)
{
    स्थिर पूर्णांक KN = (पूर्णांक)__riscv_vlenb();
    int64_t ii = 0;
    क्रम (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        क्रम (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        अगर (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    पूर्णांक remaining_rows = M - ii;
    rvv_simd_gemm_dispatch_पुच्छ<GEMM_RM - 1>(C, A, B, K, N, KN, remaining_rows);
}

#अगर defined(__GNUC__) && !defined(__clang__)
#आशय GCC diagnostic निकालो
#पूर्ण_अगर

#अन्यथा // scalar path

अटल व्योम simd_gemm(
    भग्न       * GGML_RESTRICT C,
    स्थिर भग्न * GGML_RESTRICT A,
    स्थिर भग्न * GGML_RESTRICT B,
    पूर्णांक M, पूर्णांक K, पूर्णांक N)
{
    क्रम (int64_t i = 0; i < M; i++) {
        क्रम (int64_t j = 0; j < N; j++) {
            भग्न योग_2 = C[i * N + j];
            क्रम (int64_t kk = 0; kk < K; kk++) {
                योग_2 += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = योग_2;
        }
    }
}

#पूर्ण_अगर // GGML_SIMD
