#هدايت once

// Computes C[M x N] += A[M x K] * B[K x N]

#ضمن "simd-mappings.h"

// TODO: add support for sizeless vector types
#إذا defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#إذا defined(__AVX512F__) || defined (__ARM_NEON__)
    ساكن constexpr صحيح GEMM_RM = 4;
    ساكن constexpr صحيح GEMM_RN = 4; // 16+4+1 = 25/32
#وإلاإذا defined(__AVX2__) || defined(__AVX__)
    ساكن constexpr صحيح GEMM_RM = 6;
    ساكن constexpr صحيح GEMM_RN = 2; // 12+2+1 = 15/16
#وإلا
    ساكن constexpr صحيح GEMM_RM = 2;
    ساكن constexpr صحيح GEMM_RN = 2;
#جيڪڏهن_پورو

سانچو <صحيح RM, صحيح RN>
ساكن مضمن فراغ simd_gemm_ukernel(
    عائم       * GGML_RESTRICT C,
    ثابت عائم * GGML_RESTRICT A,
    ثابت عائم * GGML_RESTRICT B,
    صحيح K, صحيح N)
{
    ساكن constexpr صحيح KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    لكل (int64_t i = 0; i < RM; i++) {
        لكل (صحيح r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_ڀريو(C + i * N + r * KN);
        }
    }

    لكل (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        لكل (صحيح r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_ڀريو(B + kk * N + r * KN);
        }
        لكل (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            لكل (صحيح r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    لكل (int64_t i = 0; i < RM; i++) {
        لكل (صحيح r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

سانچو <صحيح RM>
ساكن مضمن فراغ simd_gemm_ukernel_پڇ(
    عائم       * GGML_RESTRICT C,
    ثابت عائم * GGML_RESTRICT A,
    ثابت عائم * GGML_RESTRICT B,
    صحيح K, صحيح N, صحيح کانن_ڳڻپ_2)
{
#إذا defined(__AVX512F__)
    ثابت __mmask16 نقاب = (1u << کانن_ڳڻپ_2) - 1;
    __m512 acc[RM];
    لكل (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm512_maskz_loadu_ps(نقاب, C + i * N);
    }
    لكل (int64_t kk = 0; kk < K; kk++) {
        ثابت __m512 b = _mm512_maskz_loadu_ps(نقاب, B + kk * N);
        لكل (int64_t i = 0; i < RM; i++) {
            acc[i] = _mm512_mask3_fmadd_ps(_mm512_set1_ps(A[i * K + kk]), b, acc[i], نقاب);
        }
    }
    لكل (int64_t i = 0; i < RM; i++) {
        mm512_نقاب_storeu_ps(C + i * N, نقاب, acc[i]);
    }
#وإلاإذا defined(__AVX2__)
    ثابت __m256i نقاب = _mm256_cmpgt_epi32(_mm256_set1_epi32(کانن_ڳڻپ_2), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 acc[RM];
    لكل (int64_t i = 0; i < RM; i++) {
        acc[i] = _mm256_maskload_ps(C + i * N, نقاب);
    }
    لكل (int64_t kk = 0; kk < K; kk++) {
        ثابت __m256 b = _mm256_maskload_ps(B + kk * N, نقاب);
        لكل (int64_t i = 0; i < RM; i++) {
            acc[i] = GGML_F32_VEC_FMA(acc[i], b, _mm256_set1_ps(A[i * K + kk]));
        }
    }
    لكل (int64_t i = 0; i < RM; i++) {
        _mm256_maskstore_ps(C + i * N, نقاب, acc[i]);
    }
#وإلا
    لكل (int64_t j = 0; j < کانن_ڳڻپ_2; j++) {
        لكل (int64_t i = 0; i < RM; i++) {
            عائم a = C[i * N + j];
            لكل (int64_t kk = 0; kk < K; kk++) {
                a += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = a;
        }
    }
#جيڪڏهن_پورو
}

// C[M x N] += A[M x K] * B[K x N]
ساكن فراغ simd_gemm(
    عائم       * GGML_RESTRICT C,
    ثابت عائم * GGML_RESTRICT A,
    ثابت عائم * GGML_RESTRICT B,
    صحيح M, صحيح K, صحيح N)
{
    ساكن constexpr صحيح KN = GGML_F32_EPR;

    int64_t ii = 0;
    لكل (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        لكل (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        لكل (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        إذا (jj < N) {
            simd_gemm_ukernel_پڇ<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    لكل (; ii < M; ii++) {
        int64_t jj = 0;
        لكل (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        لكل (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        إذا (jj < N) {
            simd_gemm_ukernel_پڇ<1>(C + jj, A, B + jj, K, N, N - jj);
        }

        A += K;
        C += N;
    }
}
#وإلاإذا defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
سانچو <صحيح RM>
ساكن مضمن فراغ rvv_simd_gemm_ukernel(
    عائم       * GGML_RESTRICT C,
    ثابت عائم * GGML_RESTRICT A,
    ثابت عائم * GGML_RESTRICT B,
    صحيح K, صحيح N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    إذا constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    إذا constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    إذا constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    إذا constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    إذا constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    إذا constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    لكل (صحيح kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        إذا constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        إذا constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        إذا constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        إذا constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        إذا constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        إذا constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    إذا constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    إذا constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    إذا constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    إذا constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    إذا constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    إذا constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

سانچو <صحيح RM>
ساكن مضمن فراغ rvv_simd_gemm_dispatch_پڇ(
    عائم       * GGML_RESTRICT C,
    ثابت عائم * GGML_RESTRICT A,
    ثابت عائم * GGML_RESTRICT B,
    صحيح K, صحيح N, صحيح KN, صحيح بچيل_صفون)
{
    إذا constexpr (RM > 0) {
        إذا (بچيل_صفون == RM) {
            int64_t jj = 0;
            لكل (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            إذا (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } وإلا {
            rvv_simd_gemm_dispatch_پڇ<RM - 1>(C, A, B, K, N, KN, بچيل_صفون);
        }
    }
}

ساكن constexpr صحيح GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
ساكن فراغ simd_gemm(
    عائم       * GGML_RESTRICT C,
    ثابت عائم * GGML_RESTRICT A,
    ثابت عائم * GGML_RESTRICT B,
    صحيح M, صحيح K, صحيح N)
{
    ثابت صحيح KN = (صحيح)__riscv_vlenb();
    int64_t ii = 0;
    لكل (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        لكل (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        إذا (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    صحيح بچيل_صفون = M - ii;
    rvv_simd_gemm_dispatch_پڇ<GEMM_RM - 1>(C, A, B, K, N, KN, بچيل_صفون);
}

#إذا defined(__GNUC__) && !defined(__clang__)
#هدايت GCC diagnostic ڪڍو
#جيڪڏهن_پورو

#وإلا // scalar path

ساكن فراغ simd_gemm(
    عائم       * GGML_RESTRICT C,
    ثابت عائم * GGML_RESTRICT A,
    ثابت عائم * GGML_RESTRICT B,
    صحيح M, صحيح K, صحيح N)
{
    لكل (int64_t i = 0; i < M; i++) {
        لكل (int64_t j = 0; j < N; j++) {
            عائم جوڙ = C[i * N + j];
            لكل (int64_t kk = 0; kk < K; kk++) {
                جوڙ += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = جوڙ;
        }
    }
}

#جيڪڏهن_پورو // GGML_SIMD
