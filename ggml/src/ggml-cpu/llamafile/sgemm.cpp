// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#ಆದರೆ defined(__GNUC__)
#ಸೂಚನೆ GCC diagnostic ignored "-Wpedantic"
#ಸೂಚನೆ GCC diagnostic ignored "-Wignored-attributes"
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ "sgemm.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"
#ಅನ್ತರ್ಭಾವ "simd-mappings.h"

#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <type_traits>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _MSC_VER
#ವ್ಯಾಖ್ಯಾನಿಸು NOINLINE __declspec(noinline)
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು NOINLINE __attribute__((__noinline__))
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__ARM_NEON) || defined(__AVX512F__) || defined(__VXE__) || defined(__VXE2__)
#ವ್ಯಾಖ್ಯಾನಿಸು ಸದಿಶ_REGISTERS 32
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು ಸದಿಶ_REGISTERS 16
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_v_intrinsic)
#ವ್ಯಾಖ್ಯಾನಿಸು LMUL 4
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು MM256_ಗಣ_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

ನಾಮಾಕಾಶ {

ಸಾಲೊಳಗೆ ದಶಮಾಂಶ unhalf(ggml_fp16_t d) {
    ಹಿಂತಿರುಗಿ GGML_CPU_FP16_ಗಮ್ಯ_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#ಆದರೆ defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ಸಾಲೊಳಗೆ __m128 ಕೂಡಿಸು_2(__m128 x, __m128 y) { ಹಿಂತಿರುಗಿ mm_ಕೂಡಿಸು_ps(x, y); }
ಸಾಲೊಳಗೆ __m128 sub(__m128 x, __m128 y) { ಹಿಂತಿರುಗಿ _mm_sub_ps(x, y); }
ಸಾಲೊಳಗೆ __m128 mul(__m128 x, __m128 y) { ಹಿಂತಿರುಗಿ _mm_mul_ps(x, y); }
#ಆದರೆ_ಅಂತ್ಯ  // __SSE__

#ಆದರೆ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ಸಾಲೊಳಗೆ __m256 ಕೂಡಿಸು_2(__m256 x, __m256 y) { ಹಿಂತಿರುಗಿ mm256_ಕೂಡಿಸು_ps(x, y); }
ಸಾಲೊಳಗೆ __m256 sub(__m256 x, __m256 y) { ಹಿಂತಿರುಗಿ _mm256_sub_ps(x, y); }
ಸಾಲೊಳಗೆ __m256 mul(__m256 x, __m256 y) { ಹಿಂತಿರುಗಿ _mm256_mul_ps(x, y); }
#ಆದರೆ_ಅಂತ್ಯ // __AVX__

#ಆದರೆ defined(__AVX512F__)
ಸಾಲೊಳಗೆ __m512 ಕೂಡಿಸು_2(__m512 x, __m512 y) { ಹಿಂತಿರುಗಿ mm512_ಕೂಡಿಸು_ps(x, y); }
ಸಾಲೊಳಗೆ __m512 sub(__m512 x, __m512 y) { ಹಿಂತಿರುಗಿ _mm512_sub_ps(x, y); }
ಸಾಲೊಳಗೆ __m512 mul(__m512 x, __m512 y) { ಹಿಂತಿರುಗಿ _mm512_mul_ps(x, y); }
#ಆದರೆ_ಅಂತ್ಯ // __AVX512F__

#ಆದರೆ defined(__ARM_NEON)
ಸಾಲೊಳಗೆ float32x4_t ಕೂಡಿಸು_2(float32x4_t x, float32x4_t y) { ಹಿಂತಿರುಗಿ vaddq_f32(x, y); }
ಸಾಲೊಳಗೆ float32x4_t sub(float32x4_t x, float32x4_t y) { ಹಿಂತಿರುಗಿ vsubq_f32(x, y); }
ಸಾಲೊಳಗೆ float32x4_t mul(float32x4_t x, float32x4_t y) { ಹಿಂತಿರುಗಿ vmulq_f32(x, y); }
#ಆದರೆ_ಅಂತ್ಯ // __ARM_NEON

#ಆದರೆ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
ಸಾಲೊಳಗೆ float16x8_t ಕೂಡಿಸು_2(float16x8_t x, float16x8_t y) { ಹಿಂತಿರುಗಿ vaddq_f16(x, y); }
ಸಾಲೊಳಗೆ float16x8_t sub(float16x8_t x, float16x8_t y) { ಹಿಂತಿರುಗಿ vsubq_f16(x, y); }
ಸಾಲೊಳಗೆ float16x8_t mul(float16x8_t x, float16x8_t y) { ಹಿಂತಿರುಗಿ vmulq_f16(x, y); }
#ಆದರೆ_ಅಂತ್ಯ // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#ಆದರೆ defined(__VXE__) || defined(__VXE2__)
ಸಾಲೊಳಗೆ float32x4_t ಕೂಡಿಸು_2(float32x4_t x, float32x4_t y) { ಹಿಂತಿರುಗಿ vec_ಕೂಡಿಸು(x, y); }
ಸಾಲೊಳಗೆ float32x4_t sub(float32x4_t x, float32x4_t y) { ಹಿಂತಿರುಗಿ vec_sub(x, y); }
ಸಾಲೊಳಗೆ float32x4_t mul(float32x4_t x, float32x4_t y) { ಹಿಂತಿರುಗಿ vec_mul(x, y); }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__MMA__)
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ vec_t;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ __vector_quad acc_t;
#ಆದರೆ_ಅಂತ್ಯ
////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು U>
ಸಾಲೊಳಗೆ U madd(T a, T b, U c) {
    ಹಿಂತಿರುಗಿ ಕೂಡಿಸು_2(mul(a, b), c);
}

#ಆದರೆ defined(__FMA__)
#ಆದರೆ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ಮಾದರಿ <>
ಸಾಲೊಳಗೆ __m256 madd(__m256 a, __m256 b, __m256 c) {
    ಹಿಂತಿರುಗಿ _mm256_fmadd_ps(a, b, c);
}
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(__AVX512F__)
ಮಾದರಿ <>
ಸಾಲೊಳಗೆ __m512 madd(__m512 a, __m512 b, __m512 c) {
    ಹಿಂತಿರುಗಿ _mm512_fmadd_ps(a, b, c);
}
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(__AVX512BF16__)
ಮಾದರಿ <>
ಸಾಲೊಳಗೆ __m512 madd(__m512bh a, __m512bh b, __m512 c) {
    ಹಿಂತಿರುಗಿ _mm512_dpbf16_ps(c, a, b);
}
ಮಾದರಿ <>
ಸಾಲೊಳಗೆ __m256 madd(__m256bh a, __m256bh b, __m256 c) {
    ಹಿಂತಿರುಗಿ _mm256_dpbf16_ps(c, a, b);
}
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__ARM_FEATURE_FMA)
ಮಾದರಿ <>
ಸಾಲೊಳಗೆ float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    ಹಿಂತಿರುಗಿ vfmaq_f32(c, b, a);
}
#ಆದರೆ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
ಮಾದರಿ <>
ಸಾಲೊಳಗೆ float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    ಹಿಂತಿರುಗಿ vfmaq_f16(c, b, a);
}
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__VXE__) || defined(__VXE2__)
ಮಾದರಿ <>
ಸಾಲೊಳಗೆ float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    ಹಿಂತಿರುಗಿ vec_madd(a, b, c);
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_v_intrinsic)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m1_t madd(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m2_t madd(vfloat32m2_t a, vfloat32m2_t b, vfloat32m2_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m4_t madd(vfloat32m4_t a, vfloat32m4_t b, vfloat32m4_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m8_t madd(vfloat32m8_t a, vfloat32m8_t b, vfloat32m8_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_zvfh)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m1_t madd(vfloat16mf2_t a, vfloat16mf2_t b, vfloat32m1_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfwmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m2_t madd(vfloat16m1_t a, vfloat16m1_t b, vfloat32m2_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfwmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m4_t madd(vfloat16m2_t a, vfloat16m2_t b, vfloat32m4_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfwmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m8_t madd(vfloat16m4_t a, vfloat16m4_t b, vfloat32m8_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfwmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_zvfbfwma)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m1_t madd(vbfloat16mf2_t a, vbfloat16mf2_t b, vfloat32m1_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfwmaccbf16_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m2_t madd(vbfloat16m1_t a, vbfloat16m1_t b, vfloat32m2_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfwmaccbf16_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m4_t madd(vbfloat16m2_t a, vbfloat16m2_t b, vfloat32m4_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfwmaccbf16_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m8_t madd(vbfloat16m4_t a, vbfloat16m4_t b, vfloat32m8_t c) {
    ಹಿಂತಿರುಗಿ __riscv_vfwmaccbf16_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#ಆದರೆ_ಅಂತ್ಯ

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#ಆದರೆ defined(__ARM_NEON) || defined(_M_ARM64) || defined(_M_ARM64EC)
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(float32x4_t x) {
    ಹಿಂತಿರುಗಿ vaddvq_f32(x);
}
#ಆದರೆ_ಅಂತ್ಯ // __ARM_NEON

#ಆದರೆ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(float16x8_t x) {
    ಹಿಂತಿರುಗಿ vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_ಕೆಳಗಿನ_f16(x)),
                                vcvt_f32_f16(vget_ಮೇಲಿನ_f16(x))));
}
#ಆದರೆ_ಅಂತ್ಯ // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#ಆದರೆ defined(__VXE__) || defined(__VXE2__)
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(float32x4_t x) {
    float32x4_t tmp = x + vec_reve(x);
    ಹಿಂತಿರುಗಿ tmp[0] + tmp[1];
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(__m128 x) {
#ಆದರೆ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = mm_ಕೂಡಿಸು_ps(x, _mm_movehl_ps(x, x));
    x = mm_ಕೂಡಿಸು_ss(x, _mm_movehdup_ps(x));
#ಇಲ್ಲದಿದ್ದರೆ
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = mm_ಕೂಡಿಸು_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = mm_ಕೂಡಿಸು_ss(x, t);
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ _mm_cvtss_f32(x);
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(__m256 x) {
    ಹಿಂತಿರುಗಿ hsum(mm_ಕೂಡಿಸು_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#ಆದರೆ_ಅಂತ್ಯ // __AVX__

#ಆದರೆ defined(__AVX512F__)
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(__m512 x) {
    ಹಿಂತಿರುಗಿ mm512_reduce_ಕೂಡಿಸು_ps(x);
}
#ಆದರೆ_ಅಂತ್ಯ // __AVX512F__

#ಆದರೆ defined(__riscv_v_intrinsic)
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(vfloat32m1_t x) {
    ಹಿಂತಿರುಗಿ __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m1_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m1()));
}
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(vfloat32m2_t x) {
    ಹಿಂತಿರುಗಿ __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m2()));
}
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(vfloat32m4_t x) {
    ಹಿಂತಿರುಗಿ __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m4()));
}
ಸಾಲೊಳಗೆ ದಶಮಾಂಶ hsum(vfloat32m8_t x) {
    ಹಿಂತಿರುಗಿ __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m8_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m8()));
}
#ಆದರೆ_ಅಂತ್ಯ

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು U> T load(ಸ್ಥಿರಾಂಕ U *);

#ಆದರೆ defined(__ARM_NEON)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ float32x4_t load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ vld1q_f32(p);
}
#ಆದರೆ !defined(_MSC_VER)
// FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
ಮಾದರಿ <> ಸಾಲೊಳಗೆ float16x8_t load(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p) {
    ಹಿಂತಿರುಗಿ vld1q_f16((ಸ್ಥಿರಾಂಕ float16_t *)p);
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ float32x4_t load(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p) {
    ಹಿಂತಿರುಗಿ vcvt_f32_f16(vld1_f16((ಸ್ಥಿರಾಂಕ float16_t *)p));
}
#ಆದರೆ_ಅಂತ್ಯ // _MSC_VER
#ಆದರೆ_ಅಂತ್ಯ // __ARM_NEON

#ಆದರೆ defined(__VXE__) || defined(__VXE2__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ float32x4_t load(ಸ್ಥಿರಾಂಕ ggml_fp16_t * p) {
    ದಶಮಾಂಶ tmp[4];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(p[i]);
    }

    ಹಿಂತಿರುಗಿ vec_xl(0, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(tmp));
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ float32x4_t load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * p) {
    ಹಿಂತಿರುಗಿ vec_xl(0, p);
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m128 load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ _mm_loadu_ps(p);
}
#ಆದರೆ_ಅಂತ್ಯ  // __SSE__

#ಆದರೆ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m256 load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ _mm256_loadu_ps(p);
}
#ಆದರೆ_ಅಂತ್ಯ // __AVX__

#ಆದರೆ defined(__AVX2__) || defined(__AVX512F__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m256 load(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p) {
    ಹಿಂತಿರುಗಿ _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)p)), 16));
}
#ಆದರೆ_ಅಂತ್ಯ // __AVX2__

#ಆದರೆ defined(__F16C__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m256 load(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p) {
    ಹಿಂತಿರುಗಿ _mm256_cvtph_ps(_mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)p));
}
#ಆದರೆ_ಅಂತ್ಯ // __F16C__

#ಆದರೆ defined(__AVX512F__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m512 load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ _mm512_loadu_ps(p);
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m512 load(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p) {
    ಹಿಂತಿರುಗಿ _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)p));
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m512 load(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p) {
    ಹಿಂತಿರುಗಿ _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)p)), 16));
}
#ಆದರೆ_ಅಂತ್ಯ // __AVX512F__

#ಆದರೆ defined(__AVX512BF16__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m512bh load(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p) {
    ಹಿಂತಿರುಗಿ (__m512bh)_mm512_loadu_ps((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)p);
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m256bh load(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p) {
    ಹಿಂತಿರುಗಿ (__m256bh)_mm256_loadu_ps((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)p);
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m512bh load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m256bh load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ _mm512_cvtneps_pbh(_mm512_loadu_ps(p));
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು U> T ತುಂಬು_partial(ಸ್ಥಿರಾಂಕ U *, ಪೂರ್ಣಾಂಕ);
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> T ತುಂಬು_partial_u16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *, ಪೂರ್ಣಾಂಕ);

ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m128i ತುಂಬು_partial_u16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *p, ಪೂರ್ಣಾಂಕ n) {
#ಆದರೆ defined(__AVX512BW__) && defined(__AVX512VL__)
    ಹಿಂತಿರುಗಿ _mm_maskz_loadu_epi16((1u << n) - 1, p);
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ __m128i ಸೂಚ್ಯಂಕ = _mm_setr_epi32(0, 1, 2, 3);
    ಸ್ಥಿರಾಂಕ __m128i ಜೋಡಿಗಳು = _mm_set1_epi32(n / 2);
    __m128i v = _mm_castps_si128(_mm_maskload_ps((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)p, _mm_cmpgt_epi32(ಜೋಡಿಗಳು, ಸೂಚ್ಯಂಕ)));
    ಆದರೆ (n & 1) {
        uint16_t ಕೊನೆಯ;
        ಸ್ಮರಣೆ_ನಕಲು(&ಕೊನೆಯ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)p + 2*(n - 1), ಗಾತ್ರ(ಕೊನೆಯ));
        v = _mm_or_si128(v, _mm_and_si128(_mm_cmpeq_epi32(ಜೋಡಿಗಳು, ಸೂಚ್ಯಂಕ), _mm_set1_epi32(ಕೊನೆಯ)));
    }
    ಹಿಂತಿರುಗಿ v;
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m256 ತುಂಬು_partial(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p, ಪೂರ್ಣಾಂಕ n) {
    ಸ್ಥಿರಾಂಕ __m256 ಸೂಚ್ಯಂಕ = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    ಹಿಂತಿರುಗಿ _mm256_maskload_ps(p, _mm256_castps_si256(_mm256_cmp_ps(ಸೂಚ್ಯಂಕ, _mm256_set1_ps(n), _CMP_LT_OQ)));
}

#ಆದರೆ defined(__F16C__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m256 ತುಂಬು_partial(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p, ಪೂರ್ಣಾಂಕ n) {
    ಹಿಂತಿರುಗಿ _mm256_cvtph_ps(ತುಂಬು_partial_u16<__m128i>(p, n));
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__AVX2__) || defined(__AVX512F__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m256 ತುಂಬು_partial(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p, ಪೂರ್ಣಾಂಕ n) {
    ಹಿಂತಿರುಗಿ _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(ತುಂಬು_partial_u16<__m128i>(p, n)), 16));
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__AVX512F__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m256i ತುಂಬು_partial_u16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *p, ಪೂರ್ಣಾಂಕ n) {
#ಆದರೆ defined(__AVX512BW__) && defined(__AVX512VL__)
    ಹಿಂತಿರುಗಿ _mm256_maskz_loadu_epi16((1u << n) - 1, p);
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ __m256i ಸೂಚ್ಯಂಕ = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    ಸ್ಥಿರಾಂಕ __m256i ಜೋಡಿಗಳು = _mm256_set1_epi32(n / 2);
    __m256i v = _mm256_maskload_epi32((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)p, _mm256_cmpgt_epi32(ಜೋಡಿಗಳು, ಸೂಚ್ಯಂಕ));
    ಆದರೆ (n & 1) {
        uint16_t ಕೊನೆಯ;
        ಸ್ಮರಣೆ_ನಕಲು(&ಕೊನೆಯ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)p + 2*(n - 1), ಗಾತ್ರ(ಕೊನೆಯ));
        v = _mm256_or_si256(v, _mm256_and_si256(_mm256_cmpeq_epi32(ಜೋಡಿಗಳು, ಸೂಚ್ಯಂಕ), _mm256_set1_epi32(ಕೊನೆಯ)));
    }
    ಹಿಂತಿರುಗಿ v;
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m512 ತುಂಬು_partial(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p, ಪೂರ್ಣಾಂಕ n) {
    ಹಿಂತಿರುಗಿ _mm512_maskz_loadu_ps((1u << n) - 1, p);
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m512 ತುಂಬು_partial(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p, ಪೂರ್ಣಾಂಕ n) {
    ಹಿಂತಿರುಗಿ _mm512_cvtph_ps(ತುಂಬು_partial_u16<__m256i>(p, n));
}

ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m512 ತುಂಬು_partial(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p, ಪೂರ್ಣಾಂಕ n) {
    ಹಿಂತಿರುಗಿ _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(ತುಂಬು_partial_u16<__m256i>(p, n)), 16));
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__AVX512BF16__)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ __m512bh ತುಂಬು_partial(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p, ಪೂರ್ಣಾಂಕ n) {
    ಹಿಂತಿರುಗಿ (__m512bh) _mm512_maskz_loadu_epi16((uint64_t(1) << n) - 1, p);
}
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_v_intrinsic)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m1_t load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle32_v_f32m1(p, __riscv_vsetvlmax_e32m1());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m2_t load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle32_v_f32m2(p, __riscv_vsetvlmax_e32m2());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m4_t load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle32_v_f32m4(p, __riscv_vsetvlmax_e32m4());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m8_t load(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle32_v_f32m8(p, __riscv_vsetvlmax_e32m8());
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_zvfh)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat16mf2_t load(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle16_v_f16mf2(reinterpret_cast<ಸ್ಥಿರಾಂಕ _Float16 *>(p), __riscv_vsetvlmax_e16mf2());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat16m1_t load(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle16_v_f16m1(reinterpret_cast<ಸ್ಥಿರಾಂಕ _Float16 *>(p), __riscv_vsetvlmax_e16m1());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat16m2_t load(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle16_v_f16m2(reinterpret_cast<ಸ್ಥಿರಾಂಕ _Float16 *>(p), __riscv_vsetvlmax_e16m2());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat16m4_t load(ಸ್ಥಿರಾಂಕ ggml_fp16_t *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle16_v_f16m4(reinterpret_cast<ಸ್ಥಿರಾಂಕ _Float16 *>(p), __riscv_vsetvlmax_e16m4());
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_zvfbfwma)
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vbfloat16mf2_t load(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle16_v_bf16mf2(reinterpret_cast<ಸ್ಥಿರಾಂಕ __bf16*>(p), __riscv_vsetvlmax_e16mf2());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vbfloat16m1_t load(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle16_v_bf16m1(reinterpret_cast<ಸ್ಥಿರಾಂಕ __bf16*>(p), __riscv_vsetvlmax_e16m1());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vbfloat16m2_t load(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle16_v_bf16m2(reinterpret_cast<ಸ್ಥಿರಾಂಕ __bf16*>(p), __riscv_vsetvlmax_e16m2());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vbfloat16m4_t load(ಸ್ಥಿರಾಂಕ ggml_bf16_t *p) {
    ಹಿಂತಿರುಗಿ __riscv_vle16_v_bf16m4(reinterpret_cast<ಸ್ಥಿರಾಂಕ __bf16*>(p), __riscv_vsetvlmax_e16m4());
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_v_intrinsic)
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> T ಗಣ_zero();

ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m1_t ಗಣ_zero() {
    ಹಿಂತಿರುಗಿ __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m2_t ಗಣ_zero() {
    ಹಿಂತಿರುಗಿ __riscv_vfmv_v_f_f32m2(0, __riscv_vsetvlmax_e32m2());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m4_t ಗಣ_zero() {
    ಹಿಂತಿರುಗಿ __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
}
ಮಾದರಿ <> ಸಾಲೊಳಗೆ vfloat32m8_t ಗಣ_zero() {
    ಹಿಂತಿರುಗಿ __riscv_vfmv_v_f_f32m8(0, __riscv_vsetvlmax_e32m8());
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_v_intrinsic)
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T> size_t vlmax() {
    ಆದರೆ constexpr (std::is_same_v<T, vfloat32m1_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e32m1(); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vfloat32m2_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e32m2(); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vfloat32m4_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e32m4(); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vfloat32m8_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e32m8(); }
    #ಆದರೆ defined (__riscv_zvfh)
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vfloat16mf2_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e16mf2(); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vfloat16m1_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e16m1(); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vfloat16m2_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e16m2(); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vfloat16m4_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e16m4(); }
    #ಆದರೆ_ಅಂತ್ಯ
    #ಆದರೆ defined (__riscv_zvfbfwma)
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vbfloat16mf2_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e16mf2(); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vbfloat16m1_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e16m1(); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vbfloat16m2_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e16m2(); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, vbfloat16m4_t>) { ಹಿಂತಿರುಗಿ  __riscv_vsetvlmax_e16m4(); }
    #ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ 0;
}
#ಆದರೆ_ಅಂತ್ಯ

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ M>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t ಖಂಡ_ಗಾತ್ರ_2(size_t m) {
    ಸ್ಥಿರಾಂಕ int64_t NB_BLOC_M = (m + M - 1) / M;
    ಹಿಂತಿರುಗಿ (m % NB_BLOC_M == 0) ? m / NB_BLOC_M : (m / NB_BLOC_M) + 1;
}

ಸ್ಥಿರ constexpr ಸಾಲೊಳಗೆ int64_t BLOC_POS(int64_t ib, int64_t ibN, int64_t bloc_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ ib < ibN ? ib * bloc_ಗಾತ್ರ : ibN * bloc_ಗಾತ್ರ + (ib - ibN) * (bloc_ಗಾತ್ರ - 1);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ KN, ಬಗೆ_ಹೆಸರು D, ಬಗೆ_ಹೆಸರು V, ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC>
ವರ್ಗ tinyBLAS {
  ಸಾರ್ವಜನಿಕ:
    tinyBLAS(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, int64_t k,
             ಸ್ಥಿರಾಂಕ TA *A, int64_t lda,
             ಸ್ಥಿರಾಂಕ TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    ತಾರ್ಕಿಕ matmul(int64_t m, int64_t n) {
#ಆದರೆ !defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
        ಆದರೆ (k % KN != 0)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
        // compute RM for only need tile with size RM&RM-1
#ಆದರೆ ಸದಿಶ_REGISTERS == 32
        ಆದರೆ (m % 16 == 0 && (m/16 >= params->nth)) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<6>(n);
            mnpack<4, 6, 4>(m, n, ಗಾತ್ರ_N, 12);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 8 == 0 ) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<6>(n);
            mnpack<4, 6, 2>(m, n, ಗಾತ್ರ_N, 12);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 4 == 0) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<6>(n);
            mnpack<4, 6, 1>(m, n, ಗಾತ್ರ_N, 12);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
#ಇಲ್ಲದಿದ್ದರೆ  // VECTOR_REGISTERS == 16
        ಆದರೆ (m % 16 == 0 && (m/16 >= params->nth)) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<3>(n);
            mnpack<4, 3, 4>(m, n, ಗಾತ್ರ_N, 24);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 8 == 0 ) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<3>(n);
            mnpack<4, 3, 2>(m, n, ಗಾತ್ರ_N, 24);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 4 == 0) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<3>(n);
            mnpack<4, 3, 1>(m, n, ಗಾತ್ರ_N, 24);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

  ಖಾಸಗಿ:
    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN, ಪೂರ್ಣಾಂಕ BM>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ mnpack(int64_t m, int64_t n, int64_t ಗಾತ್ರ_N, int64_t BN) {
        ಆದರೆ (ಗಾತ್ರ_N == RN) {
            ಹಿಂತಿರುಗಿ gemm<RM, RN, BM>(m, n, BN);
        }
        ಆದರೆ constexpr (RN > 1) {
            ಹಿಂತಿರುಗಿ mnpack<RM, RN-1, BM>(m, n, ಗಾತ್ರ_N, BN);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (ಪೂರ್ಣಾಂಕ)ಗಾತ್ರ_N);
            GGML_ASSERT(ಸುಳ್ಳು); // we have miss something.
        }
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc(int64_t ii, int64_t jj) {
        D Cv[RN][RM] = {};
        ಗಾಗಿ (int64_t l = 0; l + KN <= k; l += KN) {
            // help compiler for op order.
            ಆದರೆ constexpr (RM <= RN) {
                V Av[RM];
                ಗಾಗಿ (int64_t i = 0; i < RM; ++i) {
                    Av[i] = load<V>(A + lda * (ii + i) + l);
                }
                ಗಾಗಿ (int64_t j = 0; j < RN; ++j) {
                    V Bv = load<V>(B + ldb * (jj + j) + l);
                    ಗಾಗಿ (int64_t i = 0; i < RM; ++i) {
                        Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                V Bv[RN];
                ಗಾಗಿ (int64_t j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(B + ldb * (jj + j) + l);
                }
                ಗಾಗಿ (int64_t i = 0; i < RM; ++i) {
                    V Av = load<V>(A + lda * (ii + i) + l);
                    ಗಾಗಿ (int64_t j = 0; j < RN; ++j) {
                        Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
                    }
                }
            }
        }
#ಆದರೆ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        ಸ್ಥಿರಾಂಕ int64_t rem = k % KN;
        ಆದರೆ (rem != 0) {
            V Av[RM];
            ಗಾಗಿ (int64_t i = 0; i < RM; ++i) {
                Av[i] = ತುಂಬು_partial<V>(A + lda * (ii + i) + k - rem, rem);
            }
            ಗಾಗಿ (int64_t j = 0; j < RN; ++j) {
                V Bv = ತುಂಬು_partial<V>(B + ldb * (jj + j) + k - rem, rem);
                ಗಾಗಿ (int64_t i = 0; i < RM; ++i) {
                    Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                }
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಗಾಗಿ (int64_t j = 0; j < RN; ++j)
            ಗಾಗಿ (int64_t i = 0; i < RM; ++i)
                C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN, ಪೂರ್ಣಾಂಕ BM>
    NOINLINE ಶೂನ್ಯ gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        ಸ್ಥಿರಾಂಕ int64_t ytiles = m / (RM * BM);
        ಸ್ಥಿರಾಂಕ int64_t xtiles = (n + RN -1) / RN;
        ಸ್ಥಿರಾಂಕ int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        ಸ್ಥಿರಾಂಕ int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        ಸ್ಥಿರಾಂಕ int64_t jj_BN = (NB_BN - (NB_BN * ಗಾತ್ರ_BN - xtiles));
        ಸ್ಥಿರಾಂಕ int64_t nb_ಕೆಲಸ = ytiles * NB_BN;

        ಆದರೆ (params->ith == 0) {
            GGML_ASSERT( jj_BN * ಗಾತ್ರ_BN + (NB_BN - jj_BN) * (ಗಾತ್ರ_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_ಗಣ(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t ಕೆಲಸ = params->ith;
        ಆಗ (ಕೆಲಸ < nb_ಕೆಲಸ) {
            ಸ್ಥಿರಾಂಕ int64_t ii = (ಕೆಲಸ % ytiles) * RM * BM;
            ಸ್ಥಿರಾಂಕ int64_t jb =  ಕೆಲಸ / ytiles;
            ಸ್ಥಿರಾಂಕ int64_t jr0 = BLOC_POS(jb  , jj_BN, ಗಾತ್ರ_BN);
            ಸ್ಥಿರಾಂಕ int64_t jrN = BLOC_POS(jb+1, jj_BN, ಗಾತ್ರ_BN);

            ಸ್ಥಿರಾಂಕ int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            ಸ್ಥಿರಾಂಕ int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            ಸ್ಥಿರಾಂಕ int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            ಗಾಗಿ (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                ಗಾಗಿ (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                ಆದರೆ constexpr (RN > 1) {
                    ಗಾಗಿ (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            ಕೆಲಸ = ggml_threadpool_chunk_ಕೂಡಿಸು(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ggml_compute_params * params;
    ಸ್ಥಿರಾಂಕ TA *ಸ್ಥಿರಾಂಕ A;
    ಸ್ಥಿರಾಂಕ TB *ಸ್ಥಿರಾಂಕ B;
    TC *ಸ್ಥಿರಾಂಕ C;
    ಸ್ಥಿರಾಂಕ int64_t k;
    ಸ್ಥಿರಾಂಕ int64_t lda;
    ಸ್ಥಿರಾಂಕ int64_t ldb;
    ಸ್ಥಿರಾಂಕ int64_t ldc;
};

#ಆದರೆ defined(__riscv_v_intrinsic)
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು D, ಬಗೆ_ಹೆಸರು V, ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC>
ವರ್ಗ tinyBLAS_RVV {
  ಸಾರ್ವಜನಿಕ:
    tinyBLAS_RVV(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, int64_t k,
             ಸ್ಥಿರಾಂಕ TA *A, int64_t lda,
             ಸ್ಥಿರಾಂಕ TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    ತಾರ್ಕಿಕ matmul(int64_t m, int64_t n) {
        ಆದರೆ (k % vlmax<V>() != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

#ಆದರೆ LMUL == 1
        ಆದರೆ (m % 16 == 0 && (m/16 >= params->nth)) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<6>(n);
            mnpack<4, 6, 4>(m, n, ಗಾತ್ರ_N, 12);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 8 == 0 ) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<6>(n);
            mnpack<4, 6, 2>(m, n, ಗಾತ್ರ_N, 12);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 4 == 0) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<6>(n);
            mnpack<4, 6, 1>(m, n, ಗಾತ್ರ_N, 12);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
#ಅಥವಾಆದರೆ LMUL == 2
        ಆದರೆ (m % 16 == 0 && (m/16 >= params->nth)) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<3>(n);
            mnpack<4, 3, 4>(m, n, ಗಾತ್ರ_N, 24);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 8 == 0 ) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<3>(n);
            mnpack<4, 3, 2>(m, n, ಗಾತ್ರ_N, 24);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 4 == 0) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<3>(n);
            mnpack<4, 3, 1>(m, n, ಗಾತ್ರ_N, 24);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
#ಇಲ್ಲದಿದ್ದರೆ // LMUL = 4
        ಆದರೆ (m % 16 == 0 && (m/16 >= params->nth)) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<2>(n);
            mnpack<2, 2, 8>(m, n, ಗಾತ್ರ_N, 36);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 8 == 0 ) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<2>(n);
            mnpack<2, 2, 4>(m, n, ಗಾತ್ರ_N, 36);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (m % 4 == 0) {
            ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_N = ಖಂಡ_ಗಾತ್ರ_2<2>(n);
            mnpack<2, 2, 2>(m, n, ಗಾತ್ರ_N, 36);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

  ಖಾಸಗಿ:
    ಮಾದರಿ<ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN, ಪೂರ್ಣಾಂಕ BM>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ mnpack(int64_t m, int64_t n, int64_t ಗಾತ್ರ_N, int64_t BN) {
        ಆದರೆ (ಗಾತ್ರ_N == RN) {
            ಹಿಂತಿರುಗಿ gemm<RM, RN, BM>(m, n, BN);
        }
        ಆದರೆ constexpr (RN > 1) {
            ಹಿಂತಿರುಗಿ mnpack<RM, RN-1, BM>(m, n, ಗಾತ್ರ_N, BN);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (ಪೂರ್ಣಾಂಕ)ಗಾತ್ರ_N);
            GGML_ASSERT(ಸುಳ್ಳು); // we have miss something.
        }
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc_4x6(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ಗಣ_zero<D>();
        D Cv01 = ಗಣ_zero<D>();
        D Cv02 = ಗಣ_zero<D>();
        D Cv03 = ಗಣ_zero<D>();
        D Cv10 = ಗಣ_zero<D>();
        D Cv11 = ಗಣ_zero<D>();
        D Cv12 = ಗಣ_zero<D>();
        D Cv13 = ಗಣ_zero<D>();
        D Cv20 = ಗಣ_zero<D>();
        D Cv21 = ಗಣ_zero<D>();
        D Cv22 = ಗಣ_zero<D>();
        D Cv23 = ಗಣ_zero<D>();
        D Cv30 = ಗಣ_zero<D>();
        D Cv31 = ಗಣ_zero<D>();
        D Cv32 = ಗಣ_zero<D>();
        D Cv33 = ಗಣ_zero<D>();
        D Cv40 = ಗಣ_zero<D>();
        D Cv41 = ಗಣ_zero<D>();
        D Cv42 = ಗಣ_zero<D>();
        D Cv43 = ಗಣ_zero<D>();
        D Cv50 = ಗಣ_zero<D>();
        D Cv51 = ಗಣ_zero<D>();
        D Cv52 = ಗಣ_zero<D>();
        D Cv53 = ಗಣ_zero<D>();

        ಗಾಗಿ (int64_t l = 0; l < k; l += vl) {
            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            V Bv1 = load<V>(B + ldb * (jj + 1) + l);
            V Bv2 = load<V>(B + ldb * (jj + 2) + l);
            V Bv3 = load<V>(B + ldb * (jj + 3) + l);
            V Bv4 = load<V>(B + ldb * (jj + 4) + l);
            V Bv5 = load<V>(B + ldb * (jj + 5) + l);

            V Av0 = load<V>(A + lda * (ii + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv10 = madd(Av0, Bv1, Cv10);
            Cv20 = madd(Av0, Bv2, Cv20);
            Cv30 = madd(Av0, Bv3, Cv30);
            Cv40 = madd(Av0, Bv4, Cv40);
            Cv50 = madd(Av0, Bv5, Cv50);

            V Av1 = load<V>(A + lda * (ii + 1) + l);
            Cv01 = madd(Av1, Bv0, Cv01);
            Cv11 = madd(Av1, Bv1, Cv11);
            Cv21 = madd(Av1, Bv2, Cv21);
            Cv31 = madd(Av1, Bv3, Cv31);
            Cv41 = madd(Av1, Bv4, Cv41);
            Cv51 = madd(Av1, Bv5, Cv51);

            V Av2 = load<V>(A + lda * (ii + 2) + l);
            Cv02 = madd(Av2, Bv0, Cv02);
            Cv12 = madd(Av2, Bv1, Cv12);
            Cv22 = madd(Av2, Bv2, Cv22);
            Cv32 = madd(Av2, Bv3, Cv32);
            Cv42 = madd(Av2, Bv4, Cv42);
            Cv52 = madd(Av2, Bv5, Cv52);

            V Av3 = load<V>(A + lda * (ii + 3) + l);
            Cv03 = madd(Av3, Bv0, Cv03);
            Cv13 = madd(Av3, Bv1, Cv13);
            Cv23 = madd(Av3, Bv2, Cv23);
            Cv33 = madd(Av3, Bv3, Cv33);
            Cv43 = madd(Av3, Bv4, Cv43);
            Cv53 = madd(Av3, Bv5, Cv53);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
        C[ldc * (jj + 0) + (ii + 2)] = hsum(Cv02);
        C[ldc * (jj + 0) + (ii + 3)] = hsum(Cv03);
        C[ldc * (jj + 1) + (ii + 0)] = hsum(Cv10);
        C[ldc * (jj + 1) + (ii + 1)] = hsum(Cv11);
        C[ldc * (jj + 1) + (ii + 2)] = hsum(Cv12);
        C[ldc * (jj + 1) + (ii + 3)] = hsum(Cv13);
        C[ldc * (jj + 2) + (ii + 0)] = hsum(Cv20);
        C[ldc * (jj + 2) + (ii + 1)] = hsum(Cv21);
        C[ldc * (jj + 2) + (ii + 2)] = hsum(Cv22);
        C[ldc * (jj + 2) + (ii + 3)] = hsum(Cv23);
        C[ldc * (jj + 3) + (ii + 0)] = hsum(Cv30);
        C[ldc * (jj + 3) + (ii + 1)] = hsum(Cv31);
        C[ldc * (jj + 3) + (ii + 2)] = hsum(Cv32);
        C[ldc * (jj + 3) + (ii + 3)] = hsum(Cv33);
        C[ldc * (jj + 4) + (ii + 0)] = hsum(Cv40);
        C[ldc * (jj + 4) + (ii + 1)] = hsum(Cv41);
        C[ldc * (jj + 4) + (ii + 2)] = hsum(Cv42);
        C[ldc * (jj + 4) + (ii + 3)] = hsum(Cv43);
        C[ldc * (jj + 5) + (ii + 0)] = hsum(Cv50);
        C[ldc * (jj + 5) + (ii + 1)] = hsum(Cv51);
        C[ldc * (jj + 5) + (ii + 2)] = hsum(Cv52);
        C[ldc * (jj + 5) + (ii + 3)] = hsum(Cv53);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc_4x5(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ಗಣ_zero<D>();
        D Cv01 = ಗಣ_zero<D>();
        D Cv02 = ಗಣ_zero<D>();
        D Cv03 = ಗಣ_zero<D>();
        D Cv10 = ಗಣ_zero<D>();
        D Cv11 = ಗಣ_zero<D>();
        D Cv12 = ಗಣ_zero<D>();
        D Cv13 = ಗಣ_zero<D>();
        D Cv20 = ಗಣ_zero<D>();
        D Cv21 = ಗಣ_zero<D>();
        D Cv22 = ಗಣ_zero<D>();
        D Cv23 = ಗಣ_zero<D>();
        D Cv30 = ಗಣ_zero<D>();
        D Cv31 = ಗಣ_zero<D>();
        D Cv32 = ಗಣ_zero<D>();
        D Cv33 = ಗಣ_zero<D>();
        D Cv40 = ಗಣ_zero<D>();
        D Cv41 = ಗಣ_zero<D>();
        D Cv42 = ಗಣ_zero<D>();
        D Cv43 = ಗಣ_zero<D>();

        ಗಾಗಿ (int64_t l = 0; l < k; l += vl) {
            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            V Bv1 = load<V>(B + ldb * (jj + 1) + l);
            V Bv2 = load<V>(B + ldb * (jj + 2) + l);
            V Bv3 = load<V>(B + ldb * (jj + 3) + l);
            V Bv4 = load<V>(B + ldb * (jj + 4) + l);

            V Av0 = load<V>(A + lda * (ii + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv10 = madd(Av0, Bv1, Cv10);
            Cv20 = madd(Av0, Bv2, Cv20);
            Cv30 = madd(Av0, Bv3, Cv30);
            Cv40 = madd(Av0, Bv4, Cv40);

            V Av1 = load<V>(A + lda * (ii + 1) + l);
            Cv01 = madd(Av1, Bv0, Cv01);
            Cv11 = madd(Av1, Bv1, Cv11);
            Cv21 = madd(Av1, Bv2, Cv21);
            Cv31 = madd(Av1, Bv3, Cv31);
            Cv41 = madd(Av1, Bv4, Cv41);

            V Av2 = load<V>(A + lda * (ii + 2) + l);
            Cv02 = madd(Av2, Bv0, Cv02);
            Cv12 = madd(Av2, Bv1, Cv12);
            Cv22 = madd(Av2, Bv2, Cv22);
            Cv32 = madd(Av2, Bv3, Cv32);
            Cv42 = madd(Av2, Bv4, Cv42);

            V Av3 = load<V>(A + lda * (ii + 3) + l);
            Cv03 = madd(Av3, Bv0, Cv03);
            Cv13 = madd(Av3, Bv1, Cv13);
            Cv23 = madd(Av3, Bv2, Cv23);
            Cv33 = madd(Av3, Bv3, Cv33);
            Cv43 = madd(Av3, Bv4, Cv43);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
        C[ldc * (jj + 0) + (ii + 2)] = hsum(Cv02);
        C[ldc * (jj + 0) + (ii + 3)] = hsum(Cv03);
        C[ldc * (jj + 1) + (ii + 0)] = hsum(Cv10);
        C[ldc * (jj + 1) + (ii + 1)] = hsum(Cv11);
        C[ldc * (jj + 1) + (ii + 2)] = hsum(Cv12);
        C[ldc * (jj + 1) + (ii + 3)] = hsum(Cv13);
        C[ldc * (jj + 2) + (ii + 0)] = hsum(Cv20);
        C[ldc * (jj + 2) + (ii + 1)] = hsum(Cv21);
        C[ldc * (jj + 2) + (ii + 2)] = hsum(Cv22);
        C[ldc * (jj + 2) + (ii + 3)] = hsum(Cv23);
        C[ldc * (jj + 3) + (ii + 0)] = hsum(Cv30);
        C[ldc * (jj + 3) + (ii + 1)] = hsum(Cv31);
        C[ldc * (jj + 3) + (ii + 2)] = hsum(Cv32);
        C[ldc * (jj + 3) + (ii + 3)] = hsum(Cv33);
        C[ldc * (jj + 4) + (ii + 0)] = hsum(Cv40);
        C[ldc * (jj + 4) + (ii + 1)] = hsum(Cv41);
        C[ldc * (jj + 4) + (ii + 2)] = hsum(Cv42);
        C[ldc * (jj + 4) + (ii + 3)] = hsum(Cv43);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc_4x4(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ಗಣ_zero<D>();
        D Cv01 = ಗಣ_zero<D>();
        D Cv02 = ಗಣ_zero<D>();
        D Cv03 = ಗಣ_zero<D>();
        D Cv10 = ಗಣ_zero<D>();
        D Cv11 = ಗಣ_zero<D>();
        D Cv12 = ಗಣ_zero<D>();
        D Cv13 = ಗಣ_zero<D>();
        D Cv20 = ಗಣ_zero<D>();
        D Cv21 = ಗಣ_zero<D>();
        D Cv22 = ಗಣ_zero<D>();
        D Cv23 = ಗಣ_zero<D>();
        D Cv30 = ಗಣ_zero<D>();
        D Cv31 = ಗಣ_zero<D>();
        D Cv32 = ಗಣ_zero<D>();
        D Cv33 = ಗಣ_zero<D>();

        ಗಾಗಿ (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);
            V Av2 = load<V>(A + lda * (ii + 2) + l);
            V Av3 = load<V>(A + lda * (ii + 3) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);
            Cv02 = madd(Av2, Bv0, Cv02);
            Cv03 = madd(Av3, Bv0, Cv03);

            V Bv1 = load<V>(B + ldb * (jj + 1) + l);
            Cv10 = madd(Av0, Bv1, Cv10);
            Cv11 = madd(Av1, Bv1, Cv11);
            Cv12 = madd(Av2, Bv1, Cv12);
            Cv13 = madd(Av3, Bv1, Cv13);

            V Bv2 = load<V>(B + ldb * (jj + 2) + l);
            Cv20 = madd(Av0, Bv2, Cv20);
            Cv21 = madd(Av1, Bv2, Cv21);
            Cv22 = madd(Av2, Bv2, Cv22);
            Cv23 = madd(Av3, Bv2, Cv23);

            V Bv3 = load<V>(B + ldb * (jj + 3) + l);
            Cv30 = madd(Av0, Bv3, Cv30);
            Cv31 = madd(Av1, Bv3, Cv31);
            Cv32 = madd(Av2, Bv3, Cv32);
            Cv33 = madd(Av3, Bv3, Cv33);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
        C[ldc * (jj + 0) + (ii + 2)] = hsum(Cv02);
        C[ldc * (jj + 0) + (ii + 3)] = hsum(Cv03);
        C[ldc * (jj + 1) + (ii + 0)] = hsum(Cv10);
        C[ldc * (jj + 1) + (ii + 1)] = hsum(Cv11);
        C[ldc * (jj + 1) + (ii + 2)] = hsum(Cv12);
        C[ldc * (jj + 1) + (ii + 3)] = hsum(Cv13);
        C[ldc * (jj + 2) + (ii + 0)] = hsum(Cv20);
        C[ldc * (jj + 2) + (ii + 1)] = hsum(Cv21);
        C[ldc * (jj + 2) + (ii + 2)] = hsum(Cv22);
        C[ldc * (jj + 2) + (ii + 3)] = hsum(Cv23);
        C[ldc * (jj + 3) + (ii + 0)] = hsum(Cv30);
        C[ldc * (jj + 3) + (ii + 1)] = hsum(Cv31);
        C[ldc * (jj + 3) + (ii + 2)] = hsum(Cv32);
        C[ldc * (jj + 3) + (ii + 3)] = hsum(Cv33);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc_4x3(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ಗಣ_zero<D>();
        D Cv01 = ಗಣ_zero<D>();
        D Cv02 = ಗಣ_zero<D>();
        D Cv03 = ಗಣ_zero<D>();
        D Cv10 = ಗಣ_zero<D>();
        D Cv11 = ಗಣ_zero<D>();
        D Cv12 = ಗಣ_zero<D>();
        D Cv13 = ಗಣ_zero<D>();
        D Cv20 = ಗಣ_zero<D>();
        D Cv21 = ಗಣ_zero<D>();
        D Cv22 = ಗಣ_zero<D>();
        D Cv23 = ಗಣ_zero<D>();

        ಗಾಗಿ (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);
            V Av2 = load<V>(A + lda * (ii + 2) + l);
            V Av3 = load<V>(A + lda * (ii + 3) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);
            Cv02 = madd(Av2, Bv0, Cv02);
            Cv03 = madd(Av3, Bv0, Cv03);

            V Bv1 = load<V>(B + ldb * (jj + 1) + l);
            Cv10 = madd(Av0, Bv1, Cv10);
            Cv11 = madd(Av1, Bv1, Cv11);
            Cv12 = madd(Av2, Bv1, Cv12);
            Cv13 = madd(Av3, Bv1, Cv13);

            V Bv2 = load<V>(B + ldb * (jj + 2) + l);
            Cv20 = madd(Av0, Bv2, Cv20);
            Cv21 = madd(Av1, Bv2, Cv21);
            Cv22 = madd(Av2, Bv2, Cv22);
            Cv23 = madd(Av3, Bv2, Cv23);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
        C[ldc * (jj + 0) + (ii + 2)] = hsum(Cv02);
        C[ldc * (jj + 0) + (ii + 3)] = hsum(Cv03);
        C[ldc * (jj + 1) + (ii + 0)] = hsum(Cv10);
        C[ldc * (jj + 1) + (ii + 1)] = hsum(Cv11);
        C[ldc * (jj + 1) + (ii + 2)] = hsum(Cv12);
        C[ldc * (jj + 1) + (ii + 3)] = hsum(Cv13);
        C[ldc * (jj + 2) + (ii + 0)] = hsum(Cv20);
        C[ldc * (jj + 2) + (ii + 1)] = hsum(Cv21);
        C[ldc * (jj + 2) + (ii + 2)] = hsum(Cv22);
        C[ldc * (jj + 2) + (ii + 3)] = hsum(Cv23);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc_4x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ಗಣ_zero<D>();
        D Cv01 = ಗಣ_zero<D>();
        D Cv02 = ಗಣ_zero<D>();
        D Cv03 = ಗಣ_zero<D>();
        D Cv10 = ಗಣ_zero<D>();
        D Cv11 = ಗಣ_zero<D>();
        D Cv12 = ಗಣ_zero<D>();
        D Cv13 = ಗಣ_zero<D>();

        ಗಾಗಿ (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);
            V Av2 = load<V>(A + lda * (ii + 2) + l);
            V Av3 = load<V>(A + lda * (ii + 3) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);
            Cv02 = madd(Av2, Bv0, Cv02);
            Cv03 = madd(Av3, Bv0, Cv03);

            V Bv1 = load<V>(B + ldb * (jj + 1) + l);
            Cv10 = madd(Av0, Bv1, Cv10);
            Cv11 = madd(Av1, Bv1, Cv11);
            Cv12 = madd(Av2, Bv1, Cv12);
            Cv13 = madd(Av3, Bv1, Cv13);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
        C[ldc * (jj + 0) + (ii + 2)] = hsum(Cv02);
        C[ldc * (jj + 0) + (ii + 3)] = hsum(Cv03);
        C[ldc * (jj + 1) + (ii + 0)] = hsum(Cv10);
        C[ldc * (jj + 1) + (ii + 1)] = hsum(Cv11);
        C[ldc * (jj + 1) + (ii + 2)] = hsum(Cv12);
        C[ldc * (jj + 1) + (ii + 3)] = hsum(Cv13);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc_4x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ಗಣ_zero<D>();
        D Cv01 = ಗಣ_zero<D>();
        D Cv02 = ಗಣ_zero<D>();
        D Cv03 = ಗಣ_zero<D>();

        ಗಾಗಿ (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);
            V Av2 = load<V>(A + lda * (ii + 2) + l);
            V Av3 = load<V>(A + lda * (ii + 3) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);
            Cv02 = madd(Av2, Bv0, Cv02);
            Cv03 = madd(Av3, Bv0, Cv03);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
        C[ldc * (jj + 0) + (ii + 2)] = hsum(Cv02);
        C[ldc * (jj + 0) + (ii + 3)] = hsum(Cv03);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc_2x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ಗಣ_zero<D>();
        D Cv01 = ಗಣ_zero<D>();
        D Cv10 = ಗಣ_zero<D>();
        D Cv11 = ಗಣ_zero<D>();

        ಗಾಗಿ (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);

            V Bv1 = load<V>(B + ldb * (jj + 1) + l);
            Cv10 = madd(Av0, Bv1, Cv10);
            Cv11 = madd(Av1, Bv1, Cv11);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
        C[ldc * (jj + 1) + (ii + 0)] = hsum(Cv10);
        C[ldc * (jj + 1) + (ii + 1)] = hsum(Cv11);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc_2x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ಗಣ_zero<D>();
        D Cv01 = ಗಣ_zero<D>();

        ಗಾಗಿ (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_bloc(int64_t ii, int64_t jj) {
        ಆದರೆ constexpr (RM == 4) {
            ಆದರೆ constexpr (RN == 6) { ಹಿಂತಿರುಗಿ gemm_bloc_4x6(ii, jj); }
            ಆದರೆ constexpr (RN == 5) { ಹಿಂತಿರುಗಿ gemm_bloc_4x5(ii, jj); }
            ಆದರೆ constexpr (RN == 4) { ಹಿಂತಿರುಗಿ gemm_bloc_4x4(ii, jj); }
            ಆದರೆ constexpr (RN == 3) { ಹಿಂತಿರುಗಿ gemm_bloc_4x3(ii, jj); }
            ಆದರೆ constexpr (RN == 2) { ಹಿಂತಿರುಗಿ gemm_bloc_4x2(ii, jj); }
            ಆದರೆ constexpr (RN == 1) { ಹಿಂತಿರುಗಿ gemm_bloc_4x1(ii, jj); }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (RM == 2) {
            ಆದರೆ constexpr (RN == 2) { ಹಿಂತಿರುಗಿ gemm_bloc_2x2(ii, jj); }
            ಆದರೆ constexpr (RN == 1) { ಹಿಂತಿರುಗಿ gemm_bloc_2x1(ii, jj); }
        }
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN, ಪೂರ್ಣಾಂಕ BM>
    NOINLINE ಶೂನ್ಯ gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        ಸ್ಥಿರಾಂಕ int64_t ytiles = m / (RM * BM);
        ಸ್ಥಿರಾಂಕ int64_t xtiles = (n + RN -1) / RN;
        ಸ್ಥಿರಾಂಕ int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        ಸ್ಥಿರಾಂಕ int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        ಸ್ಥಿರಾಂಕ int64_t ಗಾತ್ರ_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        ಸ್ಥಿರಾಂಕ int64_t jj_BN = (NB_BN - (NB_BN * ಗಾತ್ರ_BN - xtiles));
        ಸ್ಥಿರಾಂಕ int64_t nb_ಕೆಲಸ = ytiles * NB_BN;

        ಆದರೆ (params->ith == 0) {
            GGML_ASSERT( jj_BN * ಗಾತ್ರ_BN + (NB_BN - jj_BN) * (ಗಾತ್ರ_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_ಗಣ(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t ಕೆಲಸ = params->ith;
        ಆಗ (ಕೆಲಸ < nb_ಕೆಲಸ) {
            ಸ್ಥಿರಾಂಕ int64_t ii = (ಕೆಲಸ % ytiles) * RM * BM;
            ಸ್ಥಿರಾಂಕ int64_t jb =  ಕೆಲಸ / ytiles;
            ಸ್ಥಿರಾಂಕ int64_t jr0 = BLOC_POS(jb  , jj_BN, ಗಾತ್ರ_BN);
            ಸ್ಥಿರಾಂಕ int64_t jrN = BLOC_POS(jb+1, jj_BN, ಗಾತ್ರ_BN);

            ಸ್ಥಿರಾಂಕ int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            ಸ್ಥಿರಾಂಕ int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            ಸ್ಥಿರಾಂಕ int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            ಗಾಗಿ (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                ಗಾಗಿ (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                ಆದರೆ constexpr (RN > 1) {
                    ಗಾಗಿ (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            ಕೆಲಸ = ggml_threadpool_chunk_ಕೂಡಿಸು(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ggml_compute_params * params;
    ಸ್ಥಿರಾಂಕ TA *ಸ್ಥಿರಾಂಕ A;
    ಸ್ಥಿರಾಂಕ TB *ಸ್ಥಿರಾಂಕ B;
    TC *ಸ್ಥಿರಾಂಕ C;
    ಸ್ಥಿರಾಂಕ int64_t k;
    ಸ್ಥಿರಾಂಕ int64_t lda;
    ಸ್ಥಿರಾಂಕ int64_t ldb;
    ಸ್ಥಿರಾಂಕ int64_t ldc;
};
#ಆದರೆ_ಅಂತ್ಯ

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#ಆದರೆ defined(__ARM_FEATURE_DOTPROD)
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA>
ವರ್ಗ tinyBLAS_Q0_ARM {
  ಸಾರ್ವಜನಿಕ:
    tinyBLAS_Q0_ARM(int64_t k,
                    ಸ್ಥಿರಾಂಕ TA *A, int64_t lda,
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *B, int64_t ldb,
                    ದಶಮಾಂಶ *C, int64_t ldc,
                    ಪೂರ್ಣಾಂಕ ith, ಪೂರ್ಣಾಂಕ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    ಶೂನ್ಯ matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  ಖಾಸಗಿ:
    NOINLINE ಶೂನ್ಯ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        ಆಯ್ಕೆ ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3ll)) {
        ಸ್ಥಿತಿ 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    NOINLINE ಶೂನ್ಯ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            float32x4_t Cv[RN][RM] = {};
            ಗಾಗಿ (int64_t l = 0; l < k; ++l)
                ಗಾಗಿ (int64_t j = 0; j < RN; ++j)
                    ಗಾಗಿ (int64_t i = 0; i < RM; ++i)
                        Cv[j][i] = vmlaq_n_f32(Cv[j][i],
                                               vcvtq_f32_s32(vdotq_s32(
                                                   vdotq_s32(vdupq_n_s32(0),
                                                             ತುಂಬು_lo(A + lda * (ii + i) + l),
                                                             ತುಂಬು_lo(B + ldb * (jj + j) + l)),
                                                   ತುಂಬು_hi(A + lda * (ii + i) + l),
                                                   ತುಂಬು_hi(B + ldb * (jj + j) + l))),
                                               unhalf(A[lda * (ii + i) + l].d) *
                                               unhalf(B[ldb * (jj + j) + l].d));
            ಗಾಗಿ (int64_t j = 0; j < RN; ++j)
                ಗಾಗಿ (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    ಸಾಲೊಳಗೆ int8x16_t ತುಂಬು_lo(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *b) {
        ಹಿಂತಿರುಗಿ vld1q_s8(b->qs);
    }

    ಸಾಲೊಳಗೆ int8x16_t ತುಂಬು_hi(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *b) {
        ಹಿಂತಿರುಗಿ vld1q_s8(b->qs + 16);
    }

    ಸಾಲೊಳಗೆ int8x16_t ತುಂಬು_lo(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *b) {
        ಹಿಂತಿರುಗಿ vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs),
                                                     vdupq_n_u8(0x0f))),
                        vdupq_n_s8(0x8));
    }

    ಸಾಲೊಳಗೆ int8x16_t ತುಂಬು_hi(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *b) {
        ಹಿಂತಿರುಗಿ vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)),
                        vdupq_n_s8(0x8));
    }

    ಸ್ಥಿರಾಂಕ TA *ಸ್ಥಿರಾಂಕ A;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *ಸ್ಥಿರಾಂಕ B;
    ದಶಮಾಂಶ *ಸ್ಥಿರಾಂಕ C;
    ಸ್ಥಿರಾಂಕ int64_t k;
    ಸ್ಥಿರಾಂಕ int64_t lda;
    ಸ್ಥಿರಾಂಕ int64_t ldb;
    ಸ್ಥಿರಾಂಕ int64_t ldc;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth;
};
#ಆದರೆ_ಅಂತ್ಯ // __ARM_FEATURE_DOTPROD

#ಆದರೆ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC>
ವರ್ಗ tinyBLAS_Q0_AVX {
  ಸಾರ್ವಜನಿಕ:
    tinyBLAS_Q0_AVX(int64_t k,
                    ಸ್ಥಿರಾಂಕ TA *A, int64_t lda,
                    ಸ್ಥಿರಾಂಕ TB *B, int64_t ldb,
                    TC *C, int64_t ldc,
                    ಪೂರ್ಣಾಂಕ ith, ಪೂರ್ಣಾಂಕ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
        ಸ್ಥಿರಾಂಕ int8_t kvalues_iq4nl[16] = {
            -127, -104, -83, -65,
            -49,  -35,  -22, -10,
              1,   13,   25,  38,
             53,   69,   89, 113
        };

        iq4nlt = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)kvalues_iq4nl);
    }

    ಶೂನ್ಯ matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  ಖಾಸಗಿ:
    ಶೂನ್ಯ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        ಆಯ್ಕೆ ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
#ಆದರೆ ಸದಿಶ_REGISTERS == 32
        ಸ್ಥಿತಿ 0x44:
            mc = 4;
            nc = 4;
#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<4>(m0, m, n0, n);
#ಇಲ್ಲದಿದ್ದರೆ
            gemm<4, 4>(m0, m, n0, n);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x43:
            mc = 4;
            nc = 3;
#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<3>(m0, m, n0, n);
#ಇಲ್ಲದಿದ್ದರೆ
            gemm<4, 3>(m0, m, n0, n);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x34:
            mc = 3;
            nc = 4;
#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<3>(m0, m, n0, n);
#ಇಲ್ಲದಿದ್ದರೆ
            gemm<3, 4>(m0, m, n0, n);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x42:
            mc = 4;
            nc = 2;
#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#ಇಲ್ಲದಿದ್ದರೆ
            gemm<4, 2>(m0, m, n0, n);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x24:
            mc = 2;
            nc = 4;
#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#ಇಲ್ಲದಿದ್ದರೆ
            gemm<2, 4>(m0, m, n0, n);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
#ಇಲ್ಲದಿದ್ದರೆ
        ಸ್ಥಿತಿ 0x44:
        ಸ್ಥಿತಿ 0x43:
        ಸ್ಥಿತಿ 0x42:
            mc = 4;
            nc = 2;
#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#ಇಲ್ಲದಿದ್ದರೆ
            gemm<4, 2>(m0, m, n0, n);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x34:
        ಸ್ಥಿತಿ 0x24:
            mc = 2;
            nc = 4;
#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#ಇಲ್ಲದಿದ್ದರೆ
            gemm<2, 4>(m0, m, n0, n);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x33:
#ಆದರೆ_ಅಂತ್ಯ
        ಸ್ಥಿತಿ 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x41:
            mc = 4;
            nc = 1;
#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<1>(m0, m, n0, n);
#ಇಲ್ಲದಿದ್ದರೆ
            gemm<4, 1>(m0, m, n0, n);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x14:
            mc = 1;
            nc = 4;
#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<1>(m0, m, n0, n);
#ಇಲ್ಲದಿದ್ದರೆ
            gemm<1, 4>(m0, m, n0, n);
#ಆದರೆ_ಅಂತ್ಯ
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

#ಆದರೆ defined(__AVX2__) && defined(__F16C__)
// Templated functions for gemm of dimensions 4xN
    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RN>
    NOINLINE ಶೂನ್ಯ gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / 4;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * 4;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            __m256 Cv[RN][4] = {};
            ಗಾಗಿ (int64_t l = 0; l < k; ++l) {
                uint64_t a_delta = ((uint64_t)A[lda * (ii + 3) + l].d << 48) | ((uint64_t)A[lda * (ii + 2) + l].d << 32) | ((uint64_t)A[lda * (ii + 1) + l].d << 16) | (A[lda * (ii + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 da = _mm_cvtph_ps(mm_ಗಣ_epi64x(0, a_delta));
                __m256i avec0 = load(A + lda * (ii + 0) + l);
                __m256i avec1 = load(A + lda * (ii + 1) + l);
                __m256i avec2 = load(A + lda * (ii + 2) + l);
                __m256i avec3 = load(A + lda * (ii + 3) + l);
                ಗಾಗಿ (int64_t j = 0; j < RN; ++j) {
                        __m128 db = _mm_set1_ps(unhalf(B[ldb * (jj + j) + l].d));
                        // Computation of product of delta values for four blocks and replicate it across 256 bit lane
                        __m256 dvec =  _mm256_castps128_ps256(_mm_mul_ps(da, db));
                        dvec = _mm256_permute2f128_ps(dvec ,dvec, 0);
                        // Computation of dot product and multiplication with appropriate delta value products
                        Cv[j][0] = madd(_mm256_shuffle_ps(dvec, dvec, 0),
                                    updot(_mm256_sign_epi8(avec0, avec0),
                                          _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec0)),
                                    Cv[j][0]);
                        Cv[j][1] = madd(_mm256_shuffle_ps(dvec, dvec, 85),
                                    updot(_mm256_sign_epi8(avec1, avec1),
                                            _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec1)),
                                    Cv[j][1]);
                        Cv[j][2] = madd(_mm256_shuffle_ps(dvec, dvec, 170),
                                    updot(_mm256_sign_epi8(avec2, avec2),
                                            _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec2)),
                                    Cv[j][2]);
                        Cv[j][3] = madd(_mm256_shuffle_ps(dvec, dvec, 255),
                                    updot(_mm256_sign_epi8(avec3, avec3),
                                            _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec3)),
                                    Cv[j][3]);
                }
            }

            ಗಾಗಿ (int64_t j = 0; j < RN; ++j)
                ಗಾಗಿ (int64_t i = 0; i < 4; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    // Templated functions for gemm of dimensions Mx4
    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM>
    NOINLINE ಶೂನ್ಯ gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / 4;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * 4;
            __m256 Cv[4][RM] = {};
            ಗಾಗಿ (int64_t l = 0; l < k; ++l) {
                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 3) + l].d << 48) | ((uint64_t)B[ldb * (jj + 2) + l].d << 32) | ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 db = _mm_cvtph_ps(mm_ಗಣ_epi64x(0, b_delta));
                __m256i bvec0 = load(B + ldb * (jj + 0) + l);
                __m256i bvec1 = load(B + ldb * (jj + 1) + l);
                __m256i bvec2 = load(B + ldb * (jj + 2) + l);
                __m256i bvec3 = load(B + ldb * (jj + 3) + l);
                ಗಾಗಿ (int64_t i = 0; i < RM; ++i) {
                    __m128 da = _mm_set1_ps(unhalf((A[lda * (ii + i) + l].d)));
                    // Computation of product of delta values for four blocks and replicate it across 256 bit lane
                    __m256 dvec =  _mm256_castps128_ps256(_mm_mul_ps(da, db));
                    dvec = _mm256_permute2f128_ps(dvec ,dvec, 0);
                    // Computation of dot product and multiplication with appropriate delta value products
                    Cv[0][i] = madd(_mm256_shuffle_ps(dvec, dvec, 0),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec0, load(A + lda * (ii + i) + l))),
                                    Cv[0][i]);
                    Cv[1][i] = madd(_mm256_shuffle_ps(dvec, dvec, 85),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec1, load(A + lda * (ii + i) + l))),
                                    Cv[1][i]);
                    Cv[2][i] = madd(_mm256_shuffle_ps(dvec, dvec, 170),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec2, load(A + lda * (ii + i) + l))),
                                    Cv[2][i]);
                    Cv[3][i] = madd(_mm256_shuffle_ps(dvec, dvec, 255),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec3, load(A + lda * (ii + i) + l))),
                                    Cv[3][i]);
                }
            }
            ಗಾಗಿ (int64_t j = 0; j < 4; ++j)
                ಗಾಗಿ (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    NOINLINE ಶೂನ್ಯ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            ಗಾಗಿ (int64_t l = 0; l < k; ++l)
                ಗಾಗಿ (int64_t j = 0; j < RN; ++j)
                    ಗಾಗಿ (int64_t i = 0; i < RM; ++i) {
#ಆದರೆ defined(__AVX2__)
                        __m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                              load(A + lda * (ii + i) + l)),
                                             _mm256_sign_epi8(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l)));
#ಇಲ್ಲದಿದ್ದರೆ
                        __m128i ali0 = load0(A + lda * (ii + i) + l);
                        __m128i ali1 = load1(A + lda * (ii + i) + l);
                        __m128i blj0 = load0(B + ldb * (jj + j) + l);
                        __m128i blj1 = load1(B + ldb * (jj + j) + l);

                        __m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
                        __m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
                        __m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
                        __m128i sepBA1 = _mm_sign_epi8(blj1, ali1);

                        // updot
                        ಸ್ಥಿರಾಂಕ __m128i oneFill = _mm_set1_epi16(1);
                        __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0);
                        __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1);
                        __m256 udTmp = _mm256_cvtepi32_ps(MM256_ಗಣ_M128_I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
#ಆದರೆ_ಅಂತ್ಯ
                        Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            ಗಾಗಿ (int64_t j = 0; j < RN; ++j)
                ಗಾಗಿ (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    ಸಾಲೊಳಗೆ __m256i load(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *b) {
        ಹಿಂತಿರುಗಿ _mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)b->qs);
    }

    ಸಾಲೊಳಗೆ __m128i load0(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *b) {
        ಹಿಂತಿರುಗಿ _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)b->qs);
    }

    ಸಾಲೊಳಗೆ __m128i load1(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *b) {
        ಹಿಂತಿರುಗಿ _mm_loadu_si128(((ಸ್ಥಿರಾಂಕ __m128i *)b->qs) + 1);
    }

    ಸಾಲೊಳಗೆ __m256i load(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *b) {
        ಹಿಂತಿರುಗಿ _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    ಸಾಲೊಳಗೆ __m128i load0(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *b) {
        ಸ್ಥಿರಾಂಕ __m128i x = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b->qs));
        ಹಿಂತಿರುಗಿ _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), x), _mm_set1_epi8(8));
    }

    ಸಾಲೊಳಗೆ __m128i load1(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *b) {
        ಸ್ಥಿರಾಂಕ __m128i x = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b->qs));
        ಹಿಂತಿರುಗಿ _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)), _mm_set1_epi8(8));
    }

    ಸಾಲೊಳಗೆ __m256i load(ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *b) {
        ಹಿಂತಿರುಗಿ _mm256_or_si256(denibble(b->qs), bittobyte(b->qh));
    }

    ಸಾಲೊಳಗೆ __m128i load0(ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0* b) {
        ಸ್ಥಿರಾಂಕ __m128i x = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b->qs));
        uint32_t x32;
        ಸ್ಮರಣೆ_ನಕಲು(&x32, b->qh, ಗಾತ್ರ(uint32_t));
        __m128i qxl = _mm_and_si128(_mm_set1_epi8(15), x);
        __m128i bytesl = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      mm_ಗಣ_epi64x(0x0101010101010101, 0x0000000000000000))));
        bytesl = _mm_andnot_si128(bytesl, _mm_set1_epi8((ಅಕ್ಷರ)0xF0));
        ಹಿಂತಿರುಗಿ _mm_or_si128(qxl, bytesl);
    }

    ಸಾಲೊಳಗೆ __m128i load1(ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0* b) {
        ಸ್ಥಿರಾಂಕ __m128i x = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b->qs));
        uint32_t x32;
        ಸ್ಮರಣೆ_ನಕಲು(&x32, b->qh, ಗಾತ್ರ(uint32_t));
        __m128i qxh = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
        __m128i bytesh = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      mm_ಗಣ_epi64x(0x0303030303030303, 0x0202020202020202))));
        bytesh = _mm_andnot_si128(bytesh, _mm_set1_epi8((ಅಕ್ಷರ)0xF0));
        ಹಿಂತಿರುಗಿ _mm_or_si128(qxh, bytesh);
    }

    ಸಾಲೊಳಗೆ __m256i load(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl *b) {
        ಹಿಂತಿರುಗಿ MM256_ಗಣ_M128_I(load1(b), load0(b));
    }

    ಸಾಲೊಳಗೆ __m128i load0(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl *b) {
        ಸ್ಥಿರಾಂಕ __m128i x = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b->qs));
        ಹಿಂತಿರುಗಿ _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), x));
    }

    ಸಾಲೊಳಗೆ __m128i load1(ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl *b) {
        ಸ್ಥಿರಾಂಕ __m128i x = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(b->qs));
        ಹಿಂತಿರುಗಿ _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)));
    }

    ಸಾಲೊಳಗೆ __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#ಆದರೆ defined(__AVX512VNNI__) && defined(__AVX512VL__)
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#ಅಥವಾಆದರೆ defined(__AVXVNNI__)
        res = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
#ಇಲ್ಲದಿದ್ದರೆ
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ _mm256_cvtepi32_ps(res);
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ __m256i denibble(ಸ್ಥಿರಾಂಕ uint8_t *p) {
        __m128i x = _mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)p);
        ಹಿಂತಿರುಗಿ _mm256_and_si256(_mm256_set1_epi8(15),
                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                        _mm_srli_epi16(x, 4), 1));
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ __m256i bittobyte(ಸ್ಥಿರಾಂಕ uint8_t *p) {
        uint32_t x32;
        ಸ್ಮರಣೆ_ನಕಲು(&x32, p, ಗಾತ್ರ(uint32_t));
        __m256i ಬೈಟುಗಳು = _mm256_cmpeq_epi8(_mm256_set1_epi64x(-1),
                                          _mm256_or_si256(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                          _mm256_shuffle_epi8(_mm256_set1_epi32(x32),
                                                                              mm256_ಗಣ_epi64x(0x0303030303030303, 0x0202020202020202,
                                                                                                0x0101010101010101, 0x0000000000000000))));
        ಹಿಂತಿರುಗಿ _mm256_andnot_si256(ಬೈಟುಗಳು, _mm256_set1_epi8((ಅಕ್ಷರ)0xF0));
    }

    ಸ್ಥಿರಾಂಕ TA *ಸ್ಥಿರಾಂಕ A;
    ಸ್ಥಿರಾಂಕ TB *ಸ್ಥಿರಾಂಕ B;
    TC *ಸ್ಥಿರಾಂಕ C;
    ಸ್ಥಿರಾಂಕ int64_t k;
    ಸ್ಥಿರಾಂಕ int64_t lda;
    ಸ್ಥಿರಾಂಕ int64_t ldb;
    ಸ್ಥಿರಾಂಕ int64_t ldc;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth;
    __m128i iq4nlt;
};
#ಆದರೆ_ಅಂತ್ಯ // __AVX__

//PPC Implementation
#ಆದರೆ defined(__MMA__)

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ರಚನೆ mma_instr;

ಮಾದರಿ<>
ರಚನೆ mma_instr<ggml_bf16_t> {
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ outer_ಗುಣಲಬ್ಧ(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvbf16ger2pp(acc, a, b);
    }
};

ಮಾದರಿ<>
ರಚನೆ mma_instr<ggml_fp16_t> {
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ outer_ಗುಣಲಬ್ಧ(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvf16ger2pp(acc, a, b);
    }
};

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA, ಬಗೆ_ಹೆಸರು TB, ಬಗೆ_ಹೆಸರು TC>
ವರ್ಗ tinyBLAS_HP16_PPC {
  ಸಾರ್ವಜನಿಕ:
    tinyBLAS_HP16_PPC(int64_t k,
                ಸ್ಥಿರಾಂಕ TA *A, int64_t lda,
                ಸ್ಥಿರಾಂಕ TB *B, int64_t ldb,
                TC *C, int64_t ldc,
                ಪೂರ್ಣಾಂಕ ith, ಪೂರ್ಣಾಂಕ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    ಶೂನ್ಯ matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #ಆದರೆ defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (k < kc) {
            kc = k;
        }
        ತಾರ್ಕಿಕ can_use_tiled = (m % mc == 0) && (n % nc == 0) && (k % kc == 0);
        ಆದರೆ (can_use_tiled) {
            matmul_tiled(m, n, mc, nc, kc);
        } ಇಲ್ಲದಿದ್ದರೆ {
            mnpack(0, m, 0, n);
        }
    }

  ಖಾಸಗಿ:
    __attribute__((always_inline))
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < 4; I++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                *((ದಶಮಾಂಶ *)(C+ii+((jj+J)*ldc)+I)) = *((ದಶಮಾಂಶ *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಕೂಡಿಸು_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < 4; I++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                ದಶಮಾಂಶ * c_ptr = (ದಶಮಾಂಶ *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((ದಶಮಾಂಶ *)&vec_C[I]+J);
            }
        }
    }

    ಶೂನ್ಯ ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store(vec_t *c, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_Vec, ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *vec_ಸ್ಥಾನಾಂತರ) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        ಆದರೆ (ಅಂಕೆ_Vec == 2) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[2], c[3], swiz1);
            s[0] = vec_perm(t[0], t[1], swiz3);
            s[1] = vec_perm(t[0], t[1], swiz4);
            vec_xst(s[0], 0, (vec_t*)vec_ಸ್ಥಾನಾಂತರ);
            vec_xst(s[1], 0, (vec_t*)(vec_ಸ್ಥಾನಾಂತರ + 16));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಅಂಕೆ_Vec == 4) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[0], c[1], swiz2);
            t[2] = vec_perm(c[2], c[3], swiz1);
            t[3] = vec_perm(c[2], c[3], swiz2);
            s[0] = vec_perm(t[0], t[2], swiz3);
            s[1] = vec_perm(t[0], t[2], swiz4);
            s[2] = vec_perm(t[1], t[3], swiz3);
            s[3] = vec_perm(t[1], t[3], swiz4);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; ++i)
                vec_xst(s[i], 0, (vec_t*)(vec_ಸ್ಥಾನಾಂತರ + i * 16));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಅಂಕೆ_Vec == 8) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i += 2) {
                t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
                t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 4; i < 8; i += 2) {
                t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
                t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
            }
            s[0] = vec_perm(t[0], t[2], swiz3);
            s[1] = vec_perm(t[0], t[2], swiz4);
            s[2] = vec_perm(t[1], t[3], swiz3);
            s[3] = vec_perm(t[1], t[3], swiz4);
            s[4] = vec_perm(t[4], t[6], swiz3);
            s[5] = vec_perm(t[4], t[6], swiz4);
            s[6] = vec_perm(t[5], t[7], swiz3);
            s[7] = vec_perm(t[5], t[7], swiz4);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; ++i)
                vec_xst(s[i], 0, (vec_t*)(vec_ಸ್ಥಾನಾಂತರ + i * 16));
        }
    }

    ಶೂನ್ಯ pack_ಸಾಮಾನ್ಯ(ಸ್ಥಿರಾಂಕ TA* a, int64_t lda, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ* vec) {
        int64_t i, j;
        TA *aoffset = ಖಾಲಿ;
        ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *vec_ಸ್ಥಾನಾಂತರ = ಖಾಲಿ;
        TA * aoffsets[8];
        vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ c_arr[8];
        aoffset = const_cast<TA*>(a);
        vec_ಸ್ಥಾನಾಂತರ = vec;
        j = (ಅಡ್ಡಸಾಲುಗಳು >> 3);
        ಆದರೆ (j > 0) {
            ಮಾಡು {
                aoffsets[0] = aoffset;
                ಆದರೆ (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 == 4) {
                    aoffsets[0] = aoffset;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 4; ++it)
                        aoffsets[it] = aoffsets[it-1] + lda;
                    aoffset += 4 * lda;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; ++i)
                        c_arr[i] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[i]);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store(c_arr, 4, vec_ಸ್ಥಾನಾಂತರ);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i<4; i++)
                        aoffsets[i] = aoffsets[i]+lda;
                    vec_ಸ್ಥಾನಾಂತರ +=64;
                }
                i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 3);
                ಆದರೆ (i > 0) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 8; ++it) {
                        aoffsets[it] = aoffsets[it-1] + lda;
                    }
                    aoffset += 8 * lda;

                    ಮಾಡು {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 8; ++it)
                            c_arr[it] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[it]);
                        ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store(c_arr, 8, vec_ಸ್ಥಾನಾಂತರ);
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 8; ++it)
                            aoffsets[it] = aoffsets[it] + 8;
                        vec_ಸ್ಥಾನಾಂತರ += 128;
                        i--;
                    } ಆಗ(i > 0);
                }
                j--;
            } ಆಗ(j > 0);
        }
        ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು & 4) {
            aoffsets[0] = aoffset;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            ಆದರೆ (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 == 4) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 4; ++it)
                    c_arr[it] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[it]);
                ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store(c_arr, 2, vec_ಸ್ಥಾನಾಂತರ);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it< 4; it++)
                    aoffsets[it] = aoffsets[it] + lda;
                vec_ಸ್ಥಾನಾಂತರ += 32;
            }
            i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 3);
            ಆದರೆ (i > 0) {
                ಮಾಡು {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 4; ++it)
                        c_arr[it] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[it]);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store(c_arr, 4, vec_ಸ್ಥಾನಾಂತರ);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it< 4; it++)
                        aoffsets[it] = aoffsets[it] + 8*lda;
                    vec_ಸ್ಥಾನಾಂತರ += 64;
                    i--;
                } ಆಗ(i > 0);
            }
        }
        ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು & 3) {
            aoffsets[0] = aoffset;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            ಆದರೆ (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 == 4) {
                ಆಯ್ಕೆ(ಅಡ್ಡಸಾಲುಗಳು) {
                    ಸ್ಥಿತಿ 3: c_arr[2] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[2]);
                    ಸ್ಥಿತಿ 2: c_arr[1] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[1]);
                    ಸ್ಥಿತಿ 1: c_arr[0] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[0]);
                        ನಿಲ್ಲಿಸು;
                }
                ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store(c_arr, 2, vec_ಸ್ಥಾನಾಂತರ);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it< 4; it++)
                     aoffsets[it] = aoffsets[it] + lda;
                vec_ಸ್ಥಾನಾಂತರ += 32;
            }
            i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 3);
            ಆದರೆ (i > 0) {
                ಮಾಡು {
                    ಆಯ್ಕೆ(ಅಡ್ಡಸಾಲುಗಳು) {
                        ಸ್ಥಿತಿ 3: c_arr[2] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[2]);
                        ಸ್ಥಿತಿ 2: c_arr[1] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[1]);
                        ಸ್ಥಿತಿ 1: c_arr[0] = vec_xl(0, (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ*)aoffsets[0]);
                            ನಿಲ್ಲಿಸು;
                    }
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store(c_arr, 4, vec_ಸ್ಥಾನಾಂತರ);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it <4; it++)
                         aoffsets[it] = aoffsets[it] + 8* lda;
                    vec_ಸ್ಥಾನಾಂತರ += 64;
                    i--;
                } ಆಗ(i > 0);
            }
        }
    }

    ಶೂನ್ಯ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        ಪೂರ್ಣಾಂಕ m_rem = MIN(m - m0, 8);
        ಪೂರ್ಣಾಂಕ n_rem = MIN(n - n0, 8);

        ಆದರೆ (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4,8>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_rem >=8 && n_rem >=4){
                mc = 8;
                nc = 4;
                gemm<8,4>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((m_rem < 4) && (n_rem >= 8)) {
            nc = 8;
            ಆಯ್ಕೆ(m_rem) {
                ಸ್ಥಿತಿ 1:
                    mc = 1;
                    gemm_Mx8<1>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 2:
                    mc = 2;
                    gemm_Mx8<2>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 3:
                    mc = 3;
                    gemm_Mx8<3>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ;
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small<4, 4>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((m_rem > 4) && (n_rem < 4)) {
            mc = 4;
            ಆಯ್ಕೆ(n_rem) {
                ಸ್ಥಿತಿ 1:
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 2:
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 3:
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;

                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆಯ್ಕೆ((m_rem << 4) | n_rem) {
                ಸ್ಥಿತಿ 0x43:
                    mc = 4;
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x42:
                    mc = 4;
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x41:
                    mc = 4;
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x34:
                    mc = 3;
                    nc = 4;
                    gemm_small<3, 4>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x33:
                    mc = 3;
                    nc = 3;
                    gemm_small<3, 3>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x32:
                    mc = 3;
                    nc = 2;
                    gemm_small<3, 2>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x31:
                    mc = 3;
                    nc = 1;
                    gemm_small<3, 1>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x24:
                    mc = 2;
                    nc = 4;
                    gemm_small<2,4>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x23:
                    mc = 2;
                    nc = 3;
                    gemm_small<2, 3>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x22:
                    mc = 2;
                    nc = 2;
                    gemm_small<2, 2>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x21:
                    mc = 2;
                    nc = 1;
                    gemm_small<2, 1>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x14:
                    mc = 1;
                    nc = 4;
                    gemm_small<1, 4>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x13:
                    mc = 1;
                    nc = 3;
                    gemm_small<1, 3>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x12:
                    mc = 1;
                    nc = 2;
                    gemm_small<1, 2>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 0x11:
                    mc = 1;
                    nc = 1;
                    gemm_small<1, 1>(m0, m, n0, n);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ;
            }
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    ಶೂನ್ಯ KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l+=8) {
            pack_ಸಾಮಾನ್ಯ((A+(ii*lda)+l), lda, 4, 8, (uint8_t*)vec_A);
            pack_ಸಾಮಾನ್ಯ((B+(jj*ldb)+l), ldb, 8, 8, (uint8_t*)vec_B);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 4; x++) {
                mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_1, vec_A[x], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
    }

    ಶೂನ್ಯ KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l+=8) {
            pack_ಸಾಮಾನ್ಯ((A+(ii*lda)+l), lda, 8, 8, (uint8_t*)vec_A);
            pack_ಸಾಮಾನ್ಯ((B+(jj*ldb)+l), ldb, 8, 4, (uint8_t*)vec_B);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 4; x++) {
                mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_1, vec_A[x+4], vec_B[x]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii+4, jj);
    }


    ಶೂನ್ಯ KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l+=8) {
            pack_ಸಾಮಾನ್ಯ(A+(ii*lda)+l, lda, 8, 8, (uint8_t*)vec_A);
            pack_ಸಾಮಾನ್ಯ(B+(jj*ldb)+l, ldb, 8, 8, (uint8_t*)vec_B);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 4; x++) {
                mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_1, vec_A[x], vec_B[x+4]);
                mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_2, vec_A[x+4], vec_B[x]);
                mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_3, vec_A[x+4], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
        save_acc(&acc_2, ii+4, jj);
        save_acc(&acc_3, ii+4, jj+4);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 4; x ++) {
             mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc[0], vec_A0[x], vec_B[x]);
             mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc[1], vec_A0[x], vec_B[x+4]);
             mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc[2], vec_A0[x+4], vec_B[x]);
             mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc[3], vec_A0[x+4], vec_B[x+4]);
             mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc[4], vec_A1[x], vec_B[x]);
             mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc[5], vec_A1[x], vec_B[x+4]);
             mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc[6], vec_A1[x+4], vec_B[x]);
             mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc[7], vec_A1[x+4], vec_B[x+4]);
        }
    }
    ಶೂನ್ಯ KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        ಗಾಗಿ (int64_t i = 0; i < mc; i += 16) {
            ಪೂರ್ಣಾಂಕ A_ತಳ_addr = (mc / 8) * (i / 8) * 8;
            ಗಾಗಿ (int64_t j = 0; j < nc; j += 8) {
                 ಪೂರ್ಣಾಂಕ B_ತಳ_addr = (nc / 8) * (j / 8) * 8;
                 acc_t acc[8];
                 vec_t A0_ಖಂಡ[8]; vec_t A1_ಖಂಡ[8];
                 ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 ಗಾಗಿ (int64_t l = 0; l < kc; l += 8) {
                     ಪೂರ್ಣಾಂಕ A0_ಖಂಡ_idx = A_ತಳ_addr + (l / 8) * 8;
                     ಪೂರ್ಣಾಂಕ A1_ಖಂಡ_idx = A0_ಖಂಡ_idx + (mc / 8) * 8;
                     ಪೂರ್ಣಾಂಕ B_ಖಂಡ_idx = B_ತಳ_addr + (l / 8) * 8;
                     vec_t* A0_ಖಂಡ = &vec_A[A0_ಖಂಡ_idx];
                     vec_t* A1_ಖಂಡ = &vec_A[A1_ಖಂಡ_idx];
                     vec_t* B_ಖಂಡ = &vec_B[B_ಖಂಡ_idx];
                     MMA_16x8(A0_ಖಂಡ, A1_ಖಂಡ, B_ಖಂಡ, acc);
                 }
                 ಆದರೆ (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } ಇಲ್ಲದಿದ್ದರೆ {
                     ಕೂಡಿಸು_save_acc(&acc[0], ii + i, jj + j);
                     ಕೂಡಿಸು_save_acc(&acc[1], ii + i, jj + j + 4);
                     ಕೂಡಿಸು_save_acc(&acc[2], ii + i + 4, jj + j);
                     ಕೂಡಿಸು_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     ಕೂಡಿಸು_save_acc(&acc[4], ii + i + 8, jj + j);
                     ಕೂಡಿಸು_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     ಕೂಡಿಸು_save_acc(&acc[6], ii + i + 12, jj + j);
                     ಕೂಡಿಸು_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }
    ಮಾದರಿ<ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    ಶೂನ್ಯ gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[2], vec_B[2];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l=0; l<k; l+=4) {
                pack_ಸಾಮಾನ್ಯ(A+(ii*lda)+l, lda, RM, 4, (uint8_t*)vec_A);
                pack_ಸಾಮಾನ್ಯ(B+(jj*ldb)+l, ldb, RN, 4, (uint8_t*)vec_B);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x<2; x++) {
                    mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_0, vec_A[x], vec_B[x]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < RM; I++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < RN; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    ಮಾದರಿ<ಪೂರ್ಣಾಂಕ RM>
    ಶೂನ್ಯ gemm_Mx8(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        ಪೂರ್ಣಾಂಕ RN = 8;
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0, acc_1;
            __builtin_mma_xxsetaccz(&acc_0);
            __builtin_mma_xxsetaccz(&acc_1);
            vec_t vec_A[4], vec_B[8];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l=0; l<k; l+=8) {
                pack_ಸಾಮಾನ್ಯ(A+(ii*lda)+l, lda, RM, 8, (uint8_t*)vec_A);
                pack_ಸಾಮಾನ್ಯ(B+(jj*ldb)+l, ldb, RN, 8, (uint8_t*)vec_B);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x<4; x++) {
                    mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_0, vec_A[x], vec_B[x]);
                    mma_instr<TA>::outer_ಗುಣಲಬ್ಧ(&acc_1, vec_A[x], vec_B[x+4]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < RM; I++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_1);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < RM; I++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+4+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    ಮಾದರಿ<ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ kernel(int64_t ii, int64_t jj) {
       ಆದರೆ constexpr(RM == 4 && RN == 8) {
          KERNEL_4x8(ii,jj);
       } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr(RM == 8 && RN == 8) {
          KERNEL_8x8(ii,jj);
       } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr(RM == 8 && RN == 4) {
          KERNEL_8x4(ii,jj);
       } ಇಲ್ಲದಿದ್ದರೆ {
          ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು && "RN/RM values not supported");
       }
    }

    ಶೂನ್ಯ matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles) {
            end = tiles;
        }
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = (ಕೆಲಸ / xtiles) * mc;
            int64_t jj = (ಕೆಲಸ % xtiles) * nc;
            ಗಾಗಿ (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 8];
                 vec_t B_pack[kc * nc / 8];
                 pack_ಸಾಮಾನ್ಯ(A + (ii * lda) + kk, lda, kc, mc, (uint8_t *)A_pack);
                 pack_ಸಾಮಾನ್ಯ(B + (jj * ldb) + kk, ldb, kc, nc, (uint8_t *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    NOINLINE ಶೂನ್ಯ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    ಸ್ಥಿರಾಂಕ TA *ಸ್ಥಿರಾಂಕ A;
    ಸ್ಥಿರಾಂಕ TB *ಸ್ಥಿರಾಂಕ B;
    TC *C;
    ಸ್ಥಿರಾಂಕ int64_t k;
    ಸ್ಥಿರಾಂಕ int64_t lda;
    ಸ್ಥಿರಾಂಕ int64_t ldb;
    ಸ್ಥಿರಾಂಕ int64_t ldc;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth;
};

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು TA>
ವರ್ಗ tinyBLAS_Q0_PPC {
  ಸಾರ್ವಜನಿಕ:
    tinyBLAS_Q0_PPC(int64_t k,
             ಸ್ಥಿರಾಂಕ TA * A, int64_t lda,
             ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * B, int64_t ldb,
             ದಶಮಾಂಶ * C, int64_t ldc,
             ಪೂರ್ಣಾಂಕ ith, ಪೂರ್ಣಾಂಕ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    ಶೂನ್ಯ matmul(int64_t m, int64_t n) {
        int64_t mc = 64;
        int64_t nc = 64;
        int64_t kc = 64;
        int64_t n_chunk = 64;
    #ಆದರೆ defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 32;
        nc = 32;
        kc = 32;
        n_chunk = 32;
    #ಆದರೆ_ಅಂತ್ಯ
        int64_t n_aligned = 0;
        ಆದರೆ (n % n_chunk == 0) {
            n_aligned = n;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n == 4) {
            n_aligned = 4;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n < n_chunk) {
            n_aligned = (n / 8) * 8;
        } ಇಲ್ಲದಿದ್ದರೆ {
            n_aligned = (n / n_chunk) * n_chunk;
        }
        ಆದರೆ (n_aligned > 0) {
            ಆದರೆ (n_aligned % n_chunk == 0) nc = n_chunk;
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n_aligned == n)      nc = n;
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n_aligned % 32 == 0) nc = 32;
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n_aligned % 24 == 0) nc = 24;
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n_aligned % 16 == 0) nc = 16;
            ಇಲ್ಲದಿದ್ದರೆ                          nc = 8;
        }
        ತಾರ್ಕಿಕ can_use_tiled = n_aligned > 0 && (m % mc == 0);
        ಆದರೆ (can_use_tiled) {
            matmul_tiled(m, n_aligned, mc, nc, kc);
            ಆದರೆ (n > n_aligned) {
                mnpack(0, m, n_aligned, n);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            mnpack(0, m, 0, n);
        }
    }

  ಖಾಸಗಿ:
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ save_res(ಪೂರ್ಣಾಂಕ ii, ಪೂರ್ಣಾಂಕ jj, ಪೂರ್ಣಾಂಕ idx, vector ದಶಮಾಂಶ * fin_res, ಪೂರ್ಣಾಂಕ RM = 4, ಪೂರ್ಣಾಂಕ RN = 4) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < RM; I++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < RN; J++) {
                *((ದಶಮಾಂಶ *)(C + ii + ((jj + J) * ldc) + I)) = *((ದಶಮಾಂಶ *)&fin_res[idx + I] + J);
            }
        }
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < 4; I++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                *((ದಶಮಾಂಶ *)(C + ii + ((jj + J) * ldc) + I)) = *((ದಶಮಾಂಶ *)&vec_C[I] + J);
            }
        }
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಕೂಡಿಸು_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < 4; I++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                ದಶಮಾಂಶ * c_ptr = (ದಶಮಾಂಶ *)(C + ii+ ((jj + J) * ldc) + I);
                *c_ptr += *((ದಶಮಾಂಶ *)&vec_C[I] + J);
            }
        }
    }

    ಮಾದರಿ<ಬಗೆ_ಹೆಸರು ArrayType>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ compute(acc_t * ACC, ಪೂರ್ಣಾಂಕ c_idx, ಪೂರ್ಣಾಂಕ s_idx, ArrayType & comparray, vector ದಶಮಾಂಶ * vs, vector ದಶಮಾಂಶ * fin_res) {
        vector ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vec_C[4];
        vector ದಶಮಾಂಶ CA[4] = {0};
        vector ದಶಮಾಂಶ res[4] = {0};
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            CA[i] = vec_splats((ದಶಮಾಂಶ)(((ದ್ವಿಗುಣ)comparray[c_idx + i]) * -128.0));
            res[i] = vec_ಕೂಡಿಸು(vec_ctf(vec_C[i], 0), CA[i]);
            fin_res[s_idx + i] = vec_madd(res[i], vs[s_idx + i], fin_res[s_idx + i]);
        }
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ process_q4_elements(vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ (&c)[2], ಪೂರ್ಣಾಂಕ * ca) {
        ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
        ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);
        ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ v8 = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0x8);
        vector ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsum = {0};
        vector ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsum2 = {0};
        c[0] = vec_and(c[1], ಕೆಳಗಿನ_ಮುಖವಾಡ);
        c[1] = vec_sr(c[1], v4);
        c[0] = vec_sub(c[0], v8);
        c[1] = vec_sub(c[1], v8);
        vsum = vec_sum4s(c[0], vsum);
        vsum2 = vec_sum4s(c[1], vsum2);
        vsum = vec_ಕೂಡಿಸು(vsum, vsum2);
        *(ca) = vsum[0] + vsum[1] + vsum[2] + vsum[3];
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು V1, ಬಗೆ_ಹೆಸರು V2>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store(V2 & s1, V2 & s2, V2 & s3, V2 & s4, V1 * vec_ಸ್ಥಾನಾಂತರ, ತಾರ್ಕಿಕ flip) {
        vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ swiz1 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ swiz2 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ swiz3 = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
        vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ swiz4 = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
        V2 t1, t2, t3, t4, t5, t6, t7, t8;
        vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ xor_ಸದಿಶ;
        uint8_t flip_vec = 0x80;
        xor_ಸದಿಶ = vec_splats(flip_vec);
        t1 = vec_perm(s1, s2, swiz1);
        t2 = vec_perm(s1, s2, swiz2);
        t3 = vec_perm(s3, s4, swiz1);
        t4 = vec_perm(s3, s4, swiz2);
        t5 = vec_perm(t1, t3, swiz3);
        t6 = vec_perm(t1, t3, swiz4);
        t7 = vec_perm(t2, t4, swiz3);
        t8 = vec_perm(t2, t4, swiz4);
        ಆದರೆ (flip == ನಿಜ) {
            t5 = vec_xor(t5, xor_ಸದಿಶ);
            t6 = vec_xor(t6, xor_ಸದಿಶ);
            t7 = vec_xor(t7, xor_ಸದಿಶ);
            t8 = vec_xor(t8, xor_ಸದಿಶ);
        }
        vec_xst(t5, 0, vec_ಸ್ಥಾನಾಂತರ);
        vec_xst(t6, 0, vec_ಸ್ಥಾನಾಂತರ + 16);
        vec_xst(t7, 0, vec_ಸ್ಥಾನಾಂತರ + 32);
        vec_xst(t8, 0, vec_ಸ್ಥಾನಾಂತರ + 48);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ unpack_q4_ಗಮ್ಯ_q8(vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ packed, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ & lo, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ & hi) {
        ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0x0F);
        ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ v8      = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0x08);
        ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4    = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)4);
        lo = vec_and(packed, ಕೆಳಗಿನ_ಮುಖವಾಡ);
        hi = vec_sr(packed, v4);
        lo = vec_sub(lo, v8);
        hi = vec_sub(hi, v8);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_fp16(vec_t * c, ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * vec_ಸ್ಥಾನಾಂತರ) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i += 2) {
            t[i + 0] = vec_perm(c[i + 0], c[i + 1], swiz1);
            t[i + 1] = vec_perm(c[i + 0], c[i + 1], swiz2);
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 4; i < 8; i += 2) {
            t[i + 0] = vec_perm(c[i + 0], c[i + 1], swiz1);
            t[i + 1] = vec_perm(c[i + 0], c[i + 1], swiz2);
        }
        s[0] = vec_perm(t[0], t[2], swiz3);
        s[1] = vec_perm(t[0], t[2], swiz4);
        s[2] = vec_perm(t[1], t[3], swiz3);
        s[3] = vec_perm(t[1], t[3], swiz4);
        s[4] = vec_perm(t[4], t[6], swiz3);
        s[5] = vec_perm(t[4], t[6], swiz4);
        s[6] = vec_perm(t[5], t[7], swiz3);
        s[7] = vec_perm(t[5], t[7], swiz4);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; ++i) {
            vec_xst(s[i], 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + i * 16));
        }
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ convert_and_scale_q8(vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ raw, vector ದಶಮಾಂಶ v_scale, vector ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ & ಹೊರಹೋಗು_hi, vector ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ & ಹೊರಹೋಗು_lo) {
        vector ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ i16_hi = vec_unpackh(raw);
        vector ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ i16_lo = vec_unpackl(raw);

        vector ದಶಮಾಂಶ f_hi_h = vec_ctf(vec_unpackh(i16_hi), 0);
        vector ದಶಮಾಂಶ f_hi_l = vec_ctf(vec_unpackl(i16_hi), 0);
        vector ದಶಮಾಂಶ f_lo_h = vec_ctf(vec_unpackh(i16_lo), 0);
        vector ದಶಮಾಂಶ f_lo_l = vec_ctf(vec_unpackl(i16_lo), 0);
        ಹೊರಹೋಗು_hi = vec_pack_ಗಮ್ಯ_short_fp32(vec_mul(f_hi_h, v_scale), vec_mul(f_hi_l, v_scale));
        ಹೊರಹೋಗು_lo = vec_pack_ಗಮ್ಯ_short_fp32(vec_mul(f_lo_h, v_scale), vec_mul(f_lo_l, v_scale));
    }

    ಶೂನ್ಯ pack_ಸಾಮಾನ್ಯ_q4_fp16(ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * a, int64_t lda, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು, ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * vec) {
        ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * vec_ಸ್ಥಾನಾಂತರ = vec;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಡ್ಡಸಾಲುಗಳು; i += 8) {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * ಅಡ್ಡಸಾಲುಗಳು_ತಳ[8];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < 8; r++) {
                ಅಡ್ಡಸಾಲುಗಳು_ತಳ[r] = a + (i + r) * lda;
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ blk = 0; blk < ಖಂಡಗಳು; blk++) {
                vector ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ hp_res[8][4];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < 8; r++) {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * current_blk = ಅಡ್ಡಸಾಲುಗಳು_ತಳ[r] + blk;
                    vector ದಶಮಾಂಶ v_scale = vec_ಹೊರತೆಗೆ_fp32_from_shorth(vec_splats(current_blk->d));
                    vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ v_qs = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)current_blk->qs);
                    vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ c1, c2;
                    unpack_q4_ಗಮ್ಯ_q8(v_qs, c1, c2);
                    convert_and_scale_q8(c1, v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c2, v_scale, hp_res[r][2], hp_res[r][3]);
                }
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < 4; c++) {
                    vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ c_arr[8];
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < 8; r++) {
                        c_arr[r] = (vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)hp_res[r][c];
                    }
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_fp16((vec_t *)c_arr, vec_ಸ್ಥಾನಾಂತರ);
                    vec_ಸ್ಥಾನಾಂತರ += 128;
                }
            }
        }
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ chunk_ಗಾತ್ರ>
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ pack_q8_ಖಂಡ(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * a, int64_t lda, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು, ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * vec) {
        ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * vec_ಸ್ಥಾನಾಂತರ = vec;
        ಸ್ಥಿರಾಂಕ vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        ಸ್ಥಿರಾಂಕ vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        ಸ್ಥಿರಾಂಕ vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        ಸ್ಥಿರಾಂಕ vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಡ್ಡಸಾಲುಗಳು; i += chunk_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * ಅಡ್ಡಸಾಲುಗಳು_ತಳ[chunk_ಗಾತ್ರ];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < chunk_ಗಾತ್ರ; r++) {
                ಅಡ್ಡಸಾಲುಗಳು_ತಳ[r] = a + (i + r) * lda;
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ blk = 0; blk < ಖಂಡಗಳು; blk++) {
                vector ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ hp_res[chunk_ಗಾತ್ರ][4];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < chunk_ಗಾತ್ರ; r++) {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * b = ಅಡ್ಡಸಾಲುಗಳು_ತಳ[r] + blk;
                    vector ದಶಮಾಂಶ v_scale = vec_ಹೊರತೆಗೆ_fp32_from_shorth(vec_splats(b->d));
                    vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ c[2];
                    __vector_pair pair = __builtin_vsx_lxvp(0, (__vector_pair *)b->qs);
                    __builtin_vsx_disassemble_pair(c, & pair);
                    convert_and_scale_q8(c[0], v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c[1], v_scale, hp_res[r][2], hp_res[r][3]);
                }
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = 0; ಲಂಬಸಾಲು < 4; ಲಂಬಸಾಲು++) {
                    ಆದರೆ constexpr (chunk_ಗಾತ್ರ == 8) {
                        vec_t t[8];
                        t[0] = vec_perm((vec_t)hp_res[0][ಲಂಬಸಾಲು], (vec_t)hp_res[1][ಲಂಬಸಾಲು], swiz1);
                        t[1] = vec_perm((vec_t)hp_res[0][ಲಂಬಸಾಲು], (vec_t)hp_res[1][ಲಂಬಸಾಲು], swiz2);
                        t[2] = vec_perm((vec_t)hp_res[2][ಲಂಬಸಾಲು], (vec_t)hp_res[3][ಲಂಬಸಾಲು], swiz1);
                        t[3] = vec_perm((vec_t)hp_res[2][ಲಂಬಸಾಲು], (vec_t)hp_res[3][ಲಂಬಸಾಲು], swiz2);
                        t[4] = vec_perm((vec_t)hp_res[4][ಲಂಬಸಾಲು], (vec_t)hp_res[5][ಲಂಬಸಾಲು], swiz1);
                        t[5] = vec_perm((vec_t)hp_res[4][ಲಂಬಸಾಲು], (vec_t)hp_res[5][ಲಂಬಸಾಲು], swiz2);
                        t[6] = vec_perm((vec_t)hp_res[6][ಲಂಬಸಾಲು], (vec_t)hp_res[7][ಲಂಬಸಾಲು], swiz1);
                        t[7] = vec_perm((vec_t)hp_res[6][ಲಂಬಸಾಲು], (vec_t)hp_res[7][ಲಂಬಸಾಲು], swiz2);

                        vec_xst(vec_perm(t[0], t[2], swiz3), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 0));
                        vec_xst(vec_perm(t[0], t[2], swiz4), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 16));
                        vec_xst(vec_perm(t[1], t[3], swiz3), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 32));
                        vec_xst(vec_perm(t[1], t[3], swiz4), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 48));
                        vec_xst(vec_perm(t[4], t[6], swiz3), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 64));
                        vec_xst(vec_perm(t[4], t[6], swiz4), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 80));
                        vec_xst(vec_perm(t[5], t[7], swiz3), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 96));
                        vec_xst(vec_perm(t[5], t[7], swiz4), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 112));
                        vec_ಸ್ಥಾನಾಂತರ += 128;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        vec_t t0 = vec_perm((vec_t)hp_res[0][ಲಂಬಸಾಲು], (vec_t)hp_res[1][ಲಂಬಸಾಲು], swiz1);
                        vec_t t1 = vec_perm((vec_t)hp_res[0][ಲಂಬಸಾಲು], (vec_t)hp_res[1][ಲಂಬಸಾಲು], swiz2);
                        vec_t t2 = vec_perm((vec_t)hp_res[2][ಲಂಬಸಾಲು], (vec_t)hp_res[3][ಲಂಬಸಾಲು], swiz1);
                        vec_t t3 = vec_perm((vec_t)hp_res[2][ಲಂಬಸಾಲು], (vec_t)hp_res[3][ಲಂಬಸಾಲು], swiz2);

                        vec_xst(vec_perm(t0, t2, swiz3), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 0));
                        vec_xst(vec_perm(t0, t2, swiz4), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 16));
                        vec_xst(vec_perm(t1, t3, swiz3), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 32));
                        vec_xst(vec_perm(t1, t3, swiz4), 0, (vec_t *)(vec_ಸ್ಥಾನಾಂತರ + 48));
                        vec_ಸ್ಥಾನಾಂತರ += 64;
                    }
                }
            }
        }
    }

    ಶೂನ್ಯ pack_ಸಾಮಾನ್ಯ_q8_fp16(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * a, int64_t lda, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಖಂಡಗಳು, ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * vec) {
        ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು == 4) {
            pack_q8_ಖಂಡ<4>(a, lda, ಅಡ್ಡಸಾಲುಗಳು, ಖಂಡಗಳು, vec);
        } ಇಲ್ಲದಿದ್ದರೆ {
            pack_q8_ಖಂಡ<8>(a, lda, ಅಡ್ಡಸಾಲುಗಳು, ಖಂಡಗಳು, vec);
        }
    }

    ಮಾದರಿ<ಪೂರ್ಣಾಂಕ size>
    ಶೂನ್ಯ pack_ಸಾಮಾನ್ಯ_Int4(ಸ್ಥಿರಾಂಕ TA * a, int64_t lda, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, int8_t * vec, std::array<ಪೂರ್ಣಾಂಕ, size> & comparray) {
        int64_t i, j;
        TA * aoffset = ಖಾಲಿ;
        int8_t * vec_ಸ್ಥಾನಾಂತರ = ಖಾಲಿ;
        TA * aoffset1 = ಖಾಲಿ, * aoffset2 = ಖಾಲಿ, * aoffset3 = ಖಾಲಿ, * aoffset4 = ಖಾಲಿ;
        TA * aoffset5 = ಖಾಲಿ, * aoffset6 = ಖಾಲಿ, * aoffset7 = ಖಾಲಿ, * aoffset8 = ಖಾಲಿ;
        vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ c1[2] = {0}, c2[2] = {0}, c3[2] = {0}, c4[2] = {0};
        vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ c5[2] = {0}, c6[2] = {0}, c7[2] = {0}, c8[2] = {0};
        aoffset = const_cast<TA *>(a);
        vec_ಸ್ಥಾನಾಂತರ = vec;
        j = (ಅಡ್ಡಸಾಲುಗಳು >> 3);
        ಆದರೆ (j > 0) {
            ಮಾಡು {
                aoffset1 = aoffset;
                aoffset2 = aoffset1 + lda;
                aoffset3 = aoffset2 + lda;
                aoffset4 = aoffset3 + lda;
                aoffset5 = aoffset4 + lda;
                aoffset6 = aoffset5 + lda;
                aoffset7 = aoffset6 + lda;
                aoffset8 = aoffset7 + lda;
                aoffset += 8 * lda;
                i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 2);
                ಆದರೆ (i > 0) {
                    ಮಾಡು {
                        c1[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset1->qs);
                        c2[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset2->qs);
                        c3[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset3->qs);
                        c4[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset4->qs);
                        c5[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset5->qs);
                        c6[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset6->qs);
                        c7[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset7->qs);
                        c8[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset8->qs);

                        process_q4_elements(c1, & comparray[0]);
                        process_q4_elements(c2, & comparray[1]);
                        process_q4_elements(c3, & comparray[2]);
                        process_q4_elements(c4, & comparray[3]);
                        process_q4_elements(c5, & comparray[4]);
                        process_q4_elements(c6, & comparray[5]);
                        process_q4_elements(c7, & comparray[6]);
                        process_q4_elements(c8, & comparray[7]);
                        ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>(c1[0], c2[0], c3[0], c4[0], vec_ಸ್ಥಾನಾಂತರ, ಸುಳ್ಳು);
                        ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>(c1[1], c2[1], c3[1], c4[1], vec_ಸ್ಥಾನಾಂತರ + 64, ಸುಳ್ಳು);
                        ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>(c5[0], c6[0], c7[0], c8[0], vec_ಸ್ಥಾನಾಂತರ + 128, ಸುಳ್ಳು);
                        ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>(c5[1], c6[1], c7[1], c8[1], vec_ಸ್ಥಾನಾಂತರ + 192, ಸುಳ್ಳು);
                        aoffset1 += lda;
                        aoffset2 += lda;
                        aoffset3 += lda;
                        aoffset4 += lda;
                        aoffset5 += lda;
                        aoffset6 += lda;
                        aoffset7 += lda;
                        aoffset8 += lda;
                        vec_ಸ್ಥಾನಾಂತರ += 256;
                        i--;
                    } ಆಗ (i > 0);
                }
                j--;
            } ಆಗ (j > 0);
        }

        ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು & 4) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset += 4 * lda;
            i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 2);
            ಆದರೆ (i > 0) {
                ಮಾಡು {
                    c1[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset1->qs);
                    c2[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset2->qs);
                    c3[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset3->qs);
                    c4[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset4->qs);

                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>(c1[0], c2[0], c3[0], c4[0], vec_ಸ್ಥಾನಾಂತರ, ಸುಳ್ಳು);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>(c1[1], c2[1], c3[1], c4[1], vec_ಸ್ಥಾನಾಂತರ + 64, ಸುಳ್ಳು);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    aoffset4 += lda;
                    vec_ಸ್ಥಾನಾಂತರ += 128;
                    i--;
                } ಆಗ (i > 0);
            }
        }

        ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು & 3) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 2);
            ಆದರೆ (i > 0) {
                ಮಾಡು {
                    ಆಯ್ಕೆ(ಅಡ್ಡಸಾಲುಗಳು) {
                        ಸ್ಥಿತಿ 3: c3[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset3->qs);
                        ಸ್ಥಿತಿ 2: c2[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset2->qs);
                        ಸ್ಥಿತಿ 1: c1[1] = vec_xl(0, (ಸ್ಥಿರಾಂಕ vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)aoffset1->qs);
                            ನಿಲ್ಲಿಸು;
                    }
                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>(c1[0], c2[0], c3[0], c4[0], vec_ಸ್ಥಾನಾಂತರ, ಸುಳ್ಳು);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>(c1[1], c2[1], c3[1], c4[1], vec_ಸ್ಥಾನಾಂತರ + 64, ಸುಳ್ಳು);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    vec_ಸ್ಥಾನಾಂತರ += 128;
                    i--;
                } ಆಗ(i > 0);
            }
        }
    }

    ಮಾದರಿ<ಬಗೆ_ಹೆಸರು VA, ಬಗೆ_ಹೆಸರು VB>
    ಶೂನ್ಯ pack_ಸಾಮಾನ್ಯ(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * a, int64_t lda, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, VA * vec, ತಾರ್ಕಿಕ flip) {
        int64_t i, j;
        ಖಂಡ_q8_0 * aoffset = ಖಾಲಿ;
        VA * vec_ಸ್ಥಾನಾಂತರ = ಖಾಲಿ;
        ಖಂಡ_q8_0 * aoffsets[8];
        __vector_pair arr[8];
        VB c[8][2] = {0};
        VB c1[8] = {0}; VB c2[8] = {0};
        aoffset = const_cast<ಖಂಡ_q8_0 *>(a);
        vec_ಸ್ಥಾನಾಂತರ = vec;
        j = (ಅಡ್ಡಸಾಲುಗಳು >> 3);
        ಆದರೆ (j > 0) {
            ಮಾಡು {
                aoffsets[0] = aoffset;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it - 1] + lda;
                aoffset += 8 * lda;

                i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 3);
                ಆದರೆ (i > 0) {
                ಮಾಡು {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 8; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_ಸ್ಥಾನಾಂತರ, flip);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_ಸ್ಥಾನಾಂತರ + 64, flip);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<VA, VB>(c1[4], c1[5], c1[6], c1[7], vec_ಸ್ಥಾನಾಂತರ + 128, flip);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<VA, VB>(c2[4], c2[5], c2[6], c2[7], vec_ಸ್ಥಾನಾಂತರ + 192, flip);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 8; it++)
                        aoffsets[it] += lda;
                    vec_ಸ್ಥಾನಾಂತರ += 256;
                    i--;
               } ಆಗ(i > 0);
            }
            j--;
        } ಆಗ(j > 0);
    }
    ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು & 4) {
            aoffsets[0]  = aoffset;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 4; it++ )
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
        i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 3);
            ಆದರೆ (i > 0) {
               ಮಾಡು {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_ಸ್ಥಾನಾಂತರ, flip);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_ಸ್ಥಾನಾಂತರ + 64, flip);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 4; it++) {
                        aoffsets[it] += lda;
                    }
                    vec_ಸ್ಥಾನಾಂತರ += 128;
                    i--;
               } ಆಗ(i > 0);
            }
        }

        ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು & 3) {
            aoffsets[0]  = aoffset;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 3; it++ )
                aoffsets[it] = aoffsets[it - 1] + lda;
            i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 3);
            ಆದರೆ (i > 0) {
                ಮಾಡು {
                    ಆಯ್ಕೆ(ಅಡ್ಡಸಾಲುಗಳು) {
                        ಸ್ಥಿತಿ 3: arr[2] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[2]->qs);
                                __builtin_vsx_disassemble_pair(c[2], & arr[2]);
                                c1[2] = c[2][0]; c2[2] = c[2][1];
                        ಸ್ಥಿತಿ 2: arr[1] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[1]->qs);
                                __builtin_vsx_disassemble_pair(c[1], & arr[1]);
                                c1[1] = c[1][0]; c2[1] = c[1][1];
                        ಸ್ಥಿತಿ 1: arr[0] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[0]->qs);
                                __builtin_vsx_disassemble_pair(c[0], & arr[0]);
                                c1[0] = c[0][0]; c2[0] = c[0][1];
                                ನಿಲ್ಲಿಸು;
                    }
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_ಸ್ಥಾನಾಂತರ, flip);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_ಸ್ಥಾನಾಂತರ + 64, flip);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 3; it++)
                         aoffsets[it] += lda;
                    vec_ಸ್ಥಾನಾಂತರ += 128;
                    i--;
               } ಆಗ(i > 0);
            }
        }
    }

    ಶೂನ್ಯ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        ಪೂರ್ಣಾಂಕ m_rem = MIN(m - m0, 16);
        ಪೂರ್ಣಾಂಕ n_rem = MIN(n - n0, 16);

        ಪೂರ್ಣಾಂಕ mc = 0, nc = 0;

        ಆದರೆ (m_rem >= 8 && n_rem >= 8) {
           mc = 8;
           nc = 8;
           gemm<8, 8>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small(m0, m, n0, n, mc, nc);
        } ಇಲ್ಲದಿದ್ದರೆ {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            ಆದರೆ (mc == 0 || nc == 0)
               ಹಿಂತಿರುಗಿ;
            gemm_small(m0, m, n0, n, mc, nc);
        }

        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }


    ಶೂನ್ಯ KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[16] = {0};
        acc_t acc_0, acc_1;
        std::array<ಪೂರ್ಣಾಂಕ, 4> comparray {};
        vector ದಶಮಾಂಶ fin_res[8] = {0};
        vector ದಶಮಾಂಶ vs[8] = {0};
        ತಾರ್ಕಿಕ isAblock_q4 = std::is_same_v<TA, ಖಂಡ_q4_0>;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            ಆದರೆ (std::is_same_v<TA, ಖಂಡ_q4_0>) {
               pack_ಸಾಮಾನ್ಯ_Int4<4>((A + (ii * lda) + l), lda, 4, 4, (int8_t *)vec_A, comparray);
            } ಇಲ್ಲದಿದ್ದರೆ {
               pack_ಸಾಮಾನ್ಯ<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)(A + (ii * lda) + l), lda, 4, 8, (int8_t *)vec_A, ಸುಳ್ಳು);
            }
            pack_ಸಾಮಾನ್ಯ<uint8_t, vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, ನಿಜ);
            ಗಾಗಿ(ಪೂರ್ಣಾಂಕ x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x], vec_B[x+8]);
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I<4; I++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J<4; J++) {
                    *((ದಶಮಾಂಶ *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((ದಶಮಾಂಶ *)& vs[I + 4] + J) = (unhalf((A +((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            ಆದರೆ (!isAblock_q4) {
                ಸ್ವಯಂಚಾಲಿತ aoffset = A + (ii * lda) + l;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
                    comparray[i] = 0;
                    ಪೂರ್ಣಾಂಕ ca = 0;
                    ಸ್ವಯಂಚಾಲಿತ *at = aoffset->qs;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; j++)
                        ca += (ಪೂರ್ಣಾಂಕ)*at++;
                    comparray[i] = ca;
                    aoffset += lda;
                }
            }
            compute(& acc_0, 0, 0, comparray, vs, fin_res);
            compute(& acc_1, 0, 4, comparray, vs, fin_res);
        }
        save_res(ii, jj, 0, fin_res);
        save_res(ii, jj + 4, 4, fin_res);
    }

    ಶೂನ್ಯ KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[8] = {0};
        acc_t acc_0, acc_1;
        std::array<ಪೂರ್ಣಾಂಕ, 8> comparray {};
        vector ದಶಮಾಂಶ fin_res[8] = {0};
        vector ದಶಮಾಂಶ vs[8] = {0};
        ತಾರ್ಕಿಕ isAblock_q4 = std::is_same_v<TA, ಖಂಡ_q4_0>;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            ಆದರೆ (std::is_same_v<TA, ಖಂಡ_q4_0>) {
               pack_ಸಾಮಾನ್ಯ_Int4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } ಇಲ್ಲದಿದ್ದರೆ {
               pack_ಸಾಮಾನ್ಯ<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, ಸುಳ್ಳು);
            }
            pack_ಸಾಮಾನ್ಯ<uint8_t, vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ>((B + (jj * ldb) + l), ldb, 4, 8, (uint8_t *)vec_B, ನಿಜ);
            ಗಾಗಿ(ಪೂರ್ಣಾಂಕ x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < 8; I++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                    *((ದಶಮಾಂಶ *)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                }
            }
            ಆದರೆ (!isAblock_q4) {
                ಸ್ವಯಂಚಾಲಿತ aoffset = A + (ii * lda) + l;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    ಪೂರ್ಣಾಂಕ ca = 0;
                    ಸ್ವಯಂಚಾಲಿತ *at = aoffset->qs;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; j++)
                        ca += (ಪೂರ್ಣಾಂಕ)*at++;
                    comparray[i] = ca;
                    aoffset += lda;
                }
            }
            compute(& acc_0, 0, 0, comparray, vs, fin_res);
            compute(& acc_1, 4, 4, comparray, vs, fin_res);
        }
        save_res(ii, jj, 0, fin_res);
        save_res(ii + 4, jj, 4, fin_res);
    }

    ಶೂನ್ಯ KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16] = {0};
        acc_t acc_0, acc_1, acc_2, acc_3;
        acc_t acc_4, acc_5, acc_6, acc_7;
        std::array<ಪೂರ್ಣಾಂಕ, 8> comparray {};
        vector ದಶಮಾಂಶ fin_res[16] = {0};
        vector ದಶಮಾಂಶ vs[16] = {0};
        ತಾರ್ಕಿಕ isAblock_q4 = std::is_same_v<TA, ಖಂಡ_q4_0>;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            __builtin_mma_xxsetaccz(& acc_2);
            __builtin_mma_xxsetaccz(& acc_3);
            ಆದರೆ (std::is_same_v<TA, ಖಂಡ_q4_0>) {
               pack_ಸಾಮಾನ್ಯ_Int4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } ಇಲ್ಲದಿದ್ದರೆ {
               pack_ಸಾಮಾನ್ಯ<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, ಸುಳ್ಳು);
            }
            pack_ಸಾಮಾನ್ಯ<uint8_t, vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, ನಿಜ);
            ಗಾಗಿ(ಪೂರ್ಣಾಂಕ x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_2, vec_A[x], vec_B[x + 8]);
                __builtin_mma_xvi8ger4pp(& acc_3, vec_A[x + 8], vec_B[x + 8]);
            }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < 8 ; I++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                    *((ದಶಮಾಂಶ *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((ದಶಮಾಂಶ *)& vs[I + 8] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            ಆದರೆ (!isAblock_q4) {
                ಸ್ವಯಂಚಾಲಿತ aoffset = A + (ii * lda) + l;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    ಪೂರ್ಣಾಂಕ ca = 0;
                    ಸ್ವಯಂಚಾಲಿತ *at = aoffset->qs;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; j++)
                        ca += (ಪೂರ್ಣಾಂಕ)*at++;
                    comparray[i] = ca;
                    aoffset += lda;
                }
            }
            compute(& acc_0, 0, 0, comparray, vs, fin_res);
            compute(& acc_1, 4, 4, comparray, vs, fin_res);
            compute(& acc_2, 0, 8, comparray, vs, fin_res);
            compute(& acc_3, 4, 12, comparray, vs, fin_res);
        }
        save_res(ii, jj, 0, fin_res);
        save_res(ii + 4, jj, 4, fin_res);
        save_res(ii, jj + 4, 8, fin_res);
        save_res(ii + 4, jj + 4, 12, fin_res);
    }

    ಶೂನ್ಯ KERNEL_Q0(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, int64_t l, vec_t * vec_A, vec_t * vec_B) {
        acc_t acc[8];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < mc ; i += 16) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < nc; j += 8) {
                ಪೂರ್ಣಾಂಕ A0_ತಳ = (i / 16) * (2 * 32 * kc);
                ಪೂರ್ಣಾಂಕ B0_ತಳ = (j / 8) * (32 * kc);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 8; x++) {
                     __builtin_mma_xxsetaccz(&acc[x]);
                }
                ಗಾಗಿ (int64_t kk = 0; kk < kc; kk++) {
                    ಪೂರ್ಣಾಂಕ A0_ಖಂಡ_idx = A0_ತಳ + kk * 32;
                    ಪೂರ್ಣಾಂಕ B0_ಖಂಡ_idx = B0_ತಳ + kk * 32;
                    ಪೂರ್ಣಾಂಕ A1_ಖಂಡ_idx = A0_ಖಂಡ_idx + 32 * kc;
                    ಪೂರ್ಣಾಂಕ B1_ಖಂಡ_idx = B0_ಖಂಡ_idx + 32 * kc;
                    vec_t * A0_ಖಂಡ = & vec_A[A0_ಖಂಡ_idx];
                    vec_t * B0_ಖಂಡ = & vec_B[B0_ಖಂಡ_idx];
                    vec_t * A1_ಖಂಡ = & vec_A[A1_ಖಂಡ_idx];
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 4; it++) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 4; x++) {
                            __builtin_mma_xvf16ger2pp(& acc[0], A0_ಖಂಡ[8 * it + x], B0_ಖಂಡ[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[1], A0_ಖಂಡ[8 * it + x], B0_ಖಂಡ[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[2], A0_ಖಂಡ[8 * it + x + 4], B0_ಖಂಡ[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[3], A0_ಖಂಡ[8 * it + x + 4], B0_ಖಂಡ[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[4], A1_ಖಂಡ[8 * it + x], B0_ಖಂಡ[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[5], A1_ಖಂಡ[8 * it + x], B0_ಖಂಡ[8 * it+ x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[6], A1_ಖಂಡ[8 * it + x + 4], B0_ಖಂಡ[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[7], A1_ಖಂಡ[8 * it + x + 4], B0_ಖಂಡ[8 * it + x + 4]);
                        }
                    }
                }
                ಆದರೆ (l == 0) {
                    save_acc(& acc[0], ii + i, jj + j);
                    save_acc(& acc[1], ii + i, jj + j + 4);
                    save_acc(& acc[2], ii + i + 4, jj + j);
                    save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    save_acc(& acc[4], ii + i + 8, jj + j);
                    save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    save_acc(& acc[6], ii + i + 12, jj + j);
                    save_acc(& acc[7], ii + i + 12, jj + j + 4);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಕೂಡಿಸು_save_acc(& acc[0], ii + i, jj + j);
                    ಕೂಡಿಸು_save_acc(& acc[1], ii + i, jj + j + 4);
                    ಕೂಡಿಸು_save_acc(& acc[2], ii + i + 4, jj + j);
                    ಕೂಡಿಸು_save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    ಕೂಡಿಸು_save_acc(& acc[4], ii + i + 8, jj + j);
                    ಕೂಡಿಸು_save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    ಕೂಡಿಸು_save_acc(& acc[6], ii + i + 12, jj + j);
                    ಕೂಡಿಸು_save_acc(& acc[7], ii + i + 12, jj + j + 4);
                }
            }
        }
    }

    ಶೂನ್ಯ matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        vec_t A_pack[mc * kc * 4];
        vec_t B_pack[nc * kc * 4];
        constexpr ತಾರ್ಕಿಕ is_Ablock_q4 = std::is_same_v<TA, ಖಂಡ_q4_0>;
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles  = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles) {
            end = tiles;
        }
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = (ಕೆಲಸ / xtiles) * mc;
            int64_t jj = (ಕೆಲಸ % xtiles) * nc;
            ಗಾಗಿ (int64_t kk = 0; kk < k; kk += kc) {
                int64_t k_cur = MIN(kc, k - kk);
                ಆದರೆ constexpr(is_Ablock_q4) {
                    pack_ಸಾಮಾನ್ಯ_q4_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    pack_ಸಾಮಾನ್ಯ_q8_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                }
                pack_ಸಾಮಾನ್ಯ_q8_fp16(B + jj * ldb + kk, ldb, nc, k_cur, (uint8_t *)B_pack);
                KERNEL_Q0(ii, jj, mc, nc, k_cur, kk, A_pack, B_pack);
            }
        }
    }

    ಶೂನ್ಯ gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        vec_t vec_A[8] = {0}, vec_B[8] = {0};
        vector ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vec_C[4];
        acc_t acc_0;
        ತಾರ್ಕಿಕ isAblock_q4 = std::is_same_v<TA, ಖಂಡ_q4_0>;

        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            std::array<ಪೂರ್ಣಾಂಕ, 4> comparray{};
            vector ದಶಮಾಂಶ res[4] = {0};
            vector ದಶಮಾಂಶ fin_res[4] = {0};
            vector ದಶಮಾಂಶ vs[4] = {0};
            vector ದಶಮಾಂಶ CA[4] = {0};
            __builtin_prefetch((A + (ii * lda) + 0)->qs, 0, 1); // prefetch first value
            __builtin_prefetch((B + (jj * ldb) + 0)->qs, 0, 1); // prefetch first value
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l++) {
                __builtin_prefetch((A + (ii * lda) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_prefetch((B + (jj * ldb) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_mma_xxsetaccz(& acc_0);
                ಆದರೆ (isAblock_q4) {
                    pack_ಸಾಮಾನ್ಯ_Int4<4>((A + (ii * lda) + l), lda, RM, 4, (int8_t *)vec_A, comparray);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    pack_ಸಾಮಾನ್ಯ<int8_t, vector ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ>((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)(A + (ii * lda) + l), lda, RM, 8, (int8_t *)vec_A, ಸುಳ್ಳು);
                }
                pack_ಸಾಮಾನ್ಯ<uint8_t, vector ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ>((B + (jj * ldb) + l), ldb, RN, 8, (uint8_t *)vec_B, ನಿಜ);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 8; x += 4) {
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 1], vec_B[x + 1]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 2], vec_B[x + 2]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 3], vec_B[x + 3]);
                }
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < RM; I++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < RN; J++) {
                        *((ದಶಮಾಂಶ*)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    }
                }
                __builtin_mma_disassemble_acc(vec_C, & acc_0);
                ಆದರೆ (!isAblock_q4) {
                    ಸ್ವಯಂಚಾಲಿತ aoffset = A + (ii * lda) + l;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < RM; i++) {
                        comparray[i] = 0;
                        ಪೂರ್ಣಾಂಕ ca = 0;
                        ಸ್ವಯಂಚಾಲಿತ *at = aoffset->qs;
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; j++)
                            ca += (ಪೂರ್ಣಾಂಕ)*at++;
                        comparray[i] = ca;
                        aoffset += lda;
                    }
                }
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < RM; i++) {
                    CA[i] = vec_splats((ದಶಮಾಂಶ)(((ದ್ವಿಗುಣ)comparray[i]) * -128.0));
                    res[i] = vec_ಕೂಡಿಸು(vec_ctf(vec_C[i], 0), CA[i]);
                    fin_res[i] = vec_madd(res[i], vs[i], fin_res[i]);
                }
            }
            save_res(ii, jj, 0, fin_res, RM, RN);
        }
    }

    ಮಾದರಿ<ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ kernel(int64_t ii, int64_t jj) {
        ಆದರೆ constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii,jj);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii,jj);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii,jj);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು && "RN/RM values not supported");
        }
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    NOINLINE ಶೂನ್ಯ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }
    ಸ್ಥಿರಾಂಕ TA * ಸ್ಥಿರಾಂಕ A;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * ಸ್ಥಿರಾಂಕ B;
    ದಶಮಾಂಶ * C;
    ಸ್ಥಿರಾಂಕ int64_t k;
    int64_t kc;
    ಸ್ಥಿರಾಂಕ int64_t lda;
    ಸ್ಥಿರಾಂಕ int64_t ldb;
    ಸ್ಥಿರಾಂಕ int64_t ldc;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth;
};

ವರ್ಗ tinyBLAS_PPC {
  ಸಾರ್ವಜನಿಕ:
    tinyBLAS_PPC(int64_t k,
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A, int64_t lda,
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B, int64_t ldb,
                ದಶಮಾಂಶ * C, int64_t ldc,
                ಪೂರ್ಣಾಂಕ ith, ಪೂರ್ಣಾಂಕ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    ಶೂನ್ಯ matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #ಆದರೆ defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (m % mc == 0 && n % nc == 0 && k % kc == 0) {
            matmul_tiled(m, n, mc, nc, kc);
        } ಇಲ್ಲದಿದ್ದರೆ {
            mnpack(0, m, 0, n);
        }
    }

  ಖಾಸಗಿ:

    __attribute__((always_inline))
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < 4; I++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                *((ದಶಮಾಂಶ *)(C+ii+((jj+J)*ldc)+I)) = *((ದಶಮಾಂಶ *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಕೂಡಿಸು_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < 4; I++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < 4; J++) {
                ದಶಮಾಂಶ * c_ptr = (ದಶಮಾಂಶ *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((ದಶಮಾಂಶ *)&vec_C[I]+J);
            }
        }
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_4(vector ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * vec_ಸ್ಥಾನಾಂತರ) {
        vector ದಶಮಾಂಶ t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergel(src[0], src[1]);
        t4 = vec_mergel(src[2], src[3]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t1, t2, 3);
        t7 = vec_xxpermdi(t3, t4, 0);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_ಸ್ಥಾನಾಂತರ);
        vec_xst(t6, 0, vec_ಸ್ಥಾನಾಂತರ + 4);
        vec_xst(t7, 0, vec_ಸ್ಥಾನಾಂತರ + 8);
        vec_xst(t8, 0, vec_ಸ್ಥಾನಾಂತರ + 12);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_8(vector ದಶಮಾಂಶ * src, ದಶಮಾಂಶ * vec_ಸ್ಥಾನಾಂತರ) {
        vector ದಶಮಾಂಶ t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergeh(src[4], src[5]);
        t4 = vec_mergeh(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_ಸ್ಥಾನಾಂತರ);
        vec_xst(t6, 0, vec_ಸ್ಥಾನಾಂತರ + 4);
        vec_xst(t7, 0, vec_ಸ್ಥಾನಾಂತರ + 8);
        vec_xst(t8, 0, vec_ಸ್ಥಾನಾಂತರ + 12);

        t1 = vec_mergel(src[0], src[1]);
        t2 = vec_mergel(src[2], src[3]);
        t3 = vec_mergel(src[4], src[5]);
        t4 = vec_mergel(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_ಸ್ಥಾನಾಂತರ + 16);
        vec_xst(t6, 0, vec_ಸ್ಥಾನಾಂತರ + 20);
        vec_xst(t7, 0, vec_ಸ್ಥಾನಾಂತರ + 24);
        vec_xst(t8, 0, vec_ಸ್ಥಾನಾಂತರ + 28);
    }

    ಶೂನ್ಯ packTranspose(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a, int64_t lda, ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, ದಶಮಾಂಶ * vec) {
        int64_t i, j;
        ದಶಮಾಂಶ * aoffsets[8];
        ದಶಮಾಂಶ * aoffset = ಖಾಲಿ, * boffset = ಖಾಲಿ;
        __vector_pair arr[8];
        vector ದಶಮಾಂಶ c[8][2] = {0};
        vector ದಶಮಾಂಶ c1[8] = {0};
        vector ದಶಮಾಂಶ c2[8] = {0};
        aoffset = const_cast<ದಶಮಾಂಶ *>(a);
        boffset = vec;
        j = (ಅಡ್ಡಸಾಲುಗಳು >> 3);
        ಆದರೆ (j > 0) {
            ಮಾಡು {
                aoffsets[0] = aoffset;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it-1] + lda;
                aoffset += 8 * lda;
                i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 3);
                ಆದರೆ (i > 0) {
                    ಮಾಡು {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 8; it++) {
                            arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                            __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                            c1[it] = c[it][0];
                            c2[it] = c[it][1];
                        }

                        ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_8(c1, boffset);
                        ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_8(c2, boffset + 32);
                        boffset += 64;
                        i--;
                        ಆದರೆ (i > 0) {
                           ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 8; it++) {
                               aoffsets[it] = aoffsets[it] + 8;
                           }
                        }
                    } ಆಗ(i > 0);
                }
                ಆದರೆ (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 & 4) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 8 ; it++)
                        c1[it] = vec_xl(0, aoffsets[it]);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_8(c1, boffset);
                }
            j--;
            } ಆಗ(j > 0);
        }

        ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು & 4) {
            aoffsets[0] = aoffset;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 4; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            i = (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 >> 3);
            ಆದರೆ (i > 0) {
                ಮಾಡು {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                        __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_4(c1, boffset);
                    ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_4(c2, boffset + 16);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 4; it++)
                        aoffsets[it] += 8 * lda;
                    boffset += 32;
                    i--;
                } ಆಗ(i > 0);
            }

            ಆದರೆ (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 & 4) {
               ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 4; it++)
                   c1[it] = vec_xl(0, aoffsets[it]);
                ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_4(c1, boffset);
            }
        }
        ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು & 3) {
            aoffsets[0] = aoffset;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 1; it < 3; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            ಆದರೆ (ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 & 4) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ it = 0; it < 3; it++)
                    c1[it] = vec_xl(0, aoffsets[it]);
                ಸದಿಶ_ಕ್ರಮಬದಲಾವಣೆ_store_4(c1, boffset);
            }
        }
    }

    ಶೂನ್ಯ KERNEL_4x4(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[4], vec_C[4];
        acc_t acc_0;
        __builtin_mma_xxsetaccz(&acc_0);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (ದಶಮಾಂಶ *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (ದಶಮಾಂಶ *)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
        }
        save_acc(&acc_0, ii, jj);
    }

    ಶೂನ್ಯ KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        ಗಾಗಿ (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (ದಶಮಾಂಶ *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 4, (ದಶಮಾಂಶ *)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], (vec_t)vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_1, vec_A[0], (vec_t)vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], (vec_t)vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_1, vec_A[1], (vec_t)vec_B[3]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], (vec_t)vec_B[4]);
            __builtin_mma_xvf32gerpp(&acc_1, vec_A[2], (vec_t)vec_B[5]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], (vec_t)vec_B[6]);
            __builtin_mma_xvf32gerpp(&acc_1, vec_A[3], (vec_t)vec_B[7]);
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj + 4);
    }

    ಶೂನ್ಯ KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        ಗಾಗಿ (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 8, 4, (ದಶಮಾಂಶ *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (ದಶಮಾಂಶ *)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[0], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[1], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[2], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[3], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[4], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[5], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[6], vec_B[3]);
            __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[7], vec_B[3]);
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii + 4, jj);
    }

    ಶೂನ್ಯ KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l+=8) {
            packTranspose(A + (ii * lda) + l, lda, 8, 8, (ದಶಮಾಂಶ *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 8, (ದಶಮಾಂಶ *)vec_B);
            ಗಾಗಿ(ಪೂರ್ಣಾಂಕ x = 0; x < 16; x+=2) {
                __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[x], vec_B[x]);
                __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[x], vec_B[x + 1]);
                __builtin_mma_xvf32gerpp(&acc_2, (vec_t)vec_A[x + 1], vec_B[x]);
                __builtin_mma_xvf32gerpp(&acc_3, (vec_t)vec_A[x + 1], vec_B[x + 1]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj + 4);
        save_acc(&acc_2, ii + 4, jj);
        save_acc(&acc_3, ii + 4, jj + 4);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 16; x += 2) {
            __builtin_mma_xvf32gerpp(&acc[0], vec_A0[x + 0], vec_B[x]);
            __builtin_mma_xvf32gerpp(&acc[1], vec_A0[x + 0], vec_B[x + 1]);
            __builtin_mma_xvf32gerpp(&acc[2], vec_A0[x + 1], vec_B[x]);
            __builtin_mma_xvf32gerpp(&acc[3], vec_A0[x + 1], vec_B[x + 1]);
            __builtin_mma_xvf32gerpp(&acc[4], vec_A1[x + 0], vec_B[x]);
            __builtin_mma_xvf32gerpp(&acc[5], vec_A1[x + 0], vec_B[x + 1]);
            __builtin_mma_xvf32gerpp(&acc[6], vec_A1[x + 1], vec_B[x]);
            __builtin_mma_xvf32gerpp(&acc[7], vec_A1[x + 1], vec_B[x + 1]);
        }
    }

    ಶೂನ್ಯ KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        ಗಾಗಿ (int64_t i = 0; i < mc; i += 16) {
            ಪೂರ್ಣಾಂಕ A_ತಳ_addr = (mc / 8) * (i / 8) * 16;
            ಗಾಗಿ (int64_t j = 0; j < nc; j += 8) {
                 ಪೂರ್ಣಾಂಕ B_ತಳ_addr = (nc / 8) * (j / 8) * 16;
                 acc_t acc[8];
                 vec_t A0_ಖಂಡ[16]; vec_t A1_ಖಂಡ[16];
                 ಗಾಗಿ (ಪೂರ್ಣಾಂಕ x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 ಗಾಗಿ (int64_t l = 0; l < kc; l += 8) {
                     ಪೂರ್ಣಾಂಕ A0_ಖಂಡ_idx = A_ತಳ_addr + (l / 8) * 16;
                     ಪೂರ್ಣಾಂಕ A1_ಖಂಡ_idx = A0_ಖಂಡ_idx + (mc / 8) * 16;
                     ಪೂರ್ಣಾಂಕ B_ಖಂಡ_idx = B_ತಳ_addr + (l / 8) * 16;
                     vec_t* A0_ಖಂಡ = &vec_A[A0_ಖಂಡ_idx];
                     vec_t* A1_ಖಂಡ = &vec_A[A1_ಖಂಡ_idx];
                     vec_t* B_ಖಂಡ = &vec_B[B_ಖಂಡ_idx];
                     MMA_16x8(A0_ಖಂಡ, A1_ಖಂಡ, B_ಖಂಡ, acc);
                 }
                 ಆದರೆ (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } ಇಲ್ಲದಿದ್ದರೆ {
                     ಕೂಡಿಸು_save_acc(&acc[0], ii + i, jj + j);
                     ಕೂಡಿಸು_save_acc(&acc[1], ii + i, jj + j + 4);
                     ಕೂಡಿಸು_save_acc(&acc[2], ii + i + 4, jj + j);
                     ಕೂಡಿಸು_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     ಕೂಡಿಸು_save_acc(&acc[4], ii + i + 8, jj + j);
                     ಕೂಡಿಸು_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     ಕೂಡಿಸು_save_acc(&acc[6], ii + i + 12, jj + j);
                     ಕೂಡಿಸು_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }

    ಶೂನ್ಯ matmul_tiled(int64_t m , int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles) {
            end = tiles;
        }
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = (ಕೆಲಸ / xtiles) * mc;
            int64_t jj = (ಕೆಲಸ % xtiles) * nc;
            ಗಾಗಿ (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 4];
                 vec_t B_pack[kc * nc / 4];
                 packTranspose(A + (ii * lda) + kk, lda, kc, mc, (ದಶಮಾಂಶ *)A_pack);
                 packTranspose(B + (jj * ldb) + kk, ldb, kc, nc, (ದಶಮಾಂಶ *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    ಶೂನ್ಯ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        ಪೂರ್ಣಾಂಕ m_rem = MIN(m - m0, 8);
        ಪೂರ್ಣಾಂಕ n_rem = MIN(n - n0, 8);
        ಪೂರ್ಣಾಂಕ mc = 0, nc = 0;
        ಆದರೆ (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8, 8>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n);
        } ಇಲ್ಲದಿದ್ದರೆ {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            ಆದರೆ (mc == 0 || nc == 0)
                ಹಿಂತಿರುಗಿ;
            gemm_small(m0, m, n0, n, mc, nc);
        }
        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    ಶೂನ್ಯ gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[4] = {0}, vec_B[4] = {0};
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < k; l += 4) {
                /* 'GEMV Forwarding' concept is used in first two conditional loops.
                 * when one of the matrix has a single row/column, the elements are
                 * broadcasted, instead of using packing routine to prepack the
                 * matrix elements.
                 */
                ಆದರೆ (RM == 1) {
                    ದಶಮಾಂಶ * a = const_cast<ದಶಮಾಂಶ *>(A + (ii) * lda + l);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (ದಶಮಾಂಶ *)vec_B);
                    vec_A[0] = (vec_t)vec_xl(0,a);
                    vec_A[1] = (vec_t)vec_splats(*((ದಶಮಾಂಶ *)&vec_A+1));
                    vec_A[2] = (vec_t)vec_splats(*((ದಶಮಾಂಶ *)&vec_A+2));
                    vec_A[3] = (vec_t)vec_splats(*((ದಶಮಾಂಶ *)&vec_A+3));
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (RN == 1) {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (ದಶಮಾಂಶ *)vec_A);
                    ದಶಮಾಂಶ * b = const_cast<ದಶಮಾಂಶ *>(B + (jj) * ldb + l);
                    vec_B[0] = (vec_t)vec_xl(0,b);
                    vec_B[1] = (vec_t)vec_splats(*((ದಶಮಾಂಶ *)&vec_B+1));
                    vec_B[2] = (vec_t)vec_splats(*((ದಶಮಾಂಶ *)&vec_B+2));
                    vec_B[3] = (vec_t)vec_splats(*((ದಶಮಾಂಶ *)&vec_B+3));
                } ಇಲ್ಲದಿದ್ದರೆ {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (ದಶಮಾಂಶ *)vec_A);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (ದಶಮಾಂಶ *)vec_B);
                }
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ I = 0; I < RM; I++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ J = 0; J < RN; J++) {
                    *((ದಶಮಾಂಶ *)(C+ii+((jj+J)*ldc)+I)) = *((ದಶಮಾಂಶ *)&vec_C[I]+J);
                }
            }
       }
    }

    ಮಾದರಿ<ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ kernel(int64_t ii, int64_t jj) {
        ಆದರೆ constexpr(RM == 4 && RN == 4) {
            KERNEL_4x4(ii, jj);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii, jj);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii, jj);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii, jj);
        } ಇಲ್ಲದಿದ್ದರೆ {
            static_assert(ಸುಳ್ಳು, "RN/RM values not supported");
        }
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ RM, ಪೂರ್ಣಾಂಕ RN>
    NOINLINE ಶೂನ್ಯ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ಆದರೆ (end > tiles)
            end = tiles;
        ಗಾಗಿ (int64_t ಕೆಲಸ = start; ಕೆಲಸ < end; ++ಕೆಲಸ) {
            int64_t ii = m0 + ಕೆಲಸ / xtiles * RM;
            int64_t jj = n0 + ಕೆಲಸ % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ A;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸ್ಥಿರಾಂಕ B;
    ದಶಮಾಂಶ * C;
    ಸ್ಥಿರಾಂಕ int64_t k;
    ಸ್ಥಿರಾಂಕ int64_t lda;
    ಸ್ಥಿರಾಂಕ int64_t ldb;
    ಸ್ಥಿರಾಂಕ int64_t ldc;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ith;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nth;
};
#ಆದರೆ_ಅಂತ್ಯ
} // namespace

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc,
 *                     0, 1,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
ತಾರ್ಕಿಕ llamafile_sgemm(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *A, int64_t lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *B, int64_t ldb, ಶೂನ್ಯ *C,
                     int64_t ldc, ಪೂರ್ಣಾಂಕ Atype, ಪೂರ್ಣಾಂಕ Btype, ಪೂರ್ಣಾಂಕ Ctype) {

    ಪ್ರತಿಜ್ಞಾ(m >= 0);
    ಪ್ರತಿಜ್ಞಾ(n >= 0);
    ಪ್ರತಿಜ್ಞಾ(k >= 0);
    ಪ್ರತಿಜ್ಞಾ(lda >= k);
    ಪ್ರತಿಜ್ಞಾ(ldb >= k);
    ಪ್ರತಿಜ್ಞಾ(ldc >= m);
    ಪ್ರತಿಜ್ಞಾ(params->nth > 0);
    ಪ್ರತಿಜ್ಞಾ(params->ith < params->nth);

    // only enable sgemm for prompt processing
#ಆದರೆ !defined(__MMA__)
    ಆದರೆ (n < 2)
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (Ctype != GGML_TYPE_F32)
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಆಯ್ಕೆ (Atype) {

    ಸ್ಥಿತಿ GGML_TYPE_F32: {
        ಆದರೆ (Btype != GGML_TYPE_F32)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ defined(__AVX512F__)
        tinyBLAS<16, __m512, __m512, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ> tb{ params,
            k, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)A, lda,
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc};
        ಹಿಂತಿರುಗಿ tb.matmul(m, n);
#ಅಥವಾಆದರೆ defined(__AVX__) || defined(__AVX2__)
        tinyBLAS<8, __m256, __m256, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ> tb{ params,
            k, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)A, lda,
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc};
        ಹಿಂತಿರುಗಿ tb.matmul(m, n);
#ಅಥವಾಆದರೆ defined(__ARM_NEON)
        ಆದರೆ (n < 4)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        tinyBLAS<4, float32x4_t, float32x4_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ> tb{ params,
            k, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)A, lda,
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc};
        ಹಿಂತಿರುಗಿ tb.matmul(m, n);
#ಅಥವಾಆದರೆ defined(__VXE__) || defined(__VXE2__)
        ಆದರೆ (n < 4)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        tinyBLAS<4, float32x4_t, float32x4_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ> tb{ params,
            k, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)A, lda,
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc};
        ಹಿಂತಿರುಗಿ tb.matmul(m, n);
#ಅಥವಾಆದರೆ defined(__MMA__)
        ಆದರೆ (k % 8)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        tinyBLAS_PPC tb{
            k, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)A, lda,
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ಹಿಂತಿರುಗಿ ನಿಜ;
#ಅಥವಾಆದರೆ defined(__riscv_v_intrinsic)
    #ಆದರೆ LMUL == 1
        tinyBLAS_RVV<vfloat32m1_t, vfloat32m1_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ> tb{ params,
            k, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)A, lda,
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc};
    #ಅಥವಾಆದರೆ LMUL == 2
        tinyBLAS_RVV<vfloat32m2_t, vfloat32m2_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ> tb{ params,
            k, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)A, lda,
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc};
    #ಇಲ್ಲದಿದ್ದರೆ // LMUL = 4
        tinyBLAS_RVV<vfloat32m4_t, vfloat32m4_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ> tb{ params,
            k, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)A, lda,
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc};
    #ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ tb.matmul(m, n);
#ಇಲ್ಲದಿದ್ದರೆ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
    }

    ಸ್ಥಿತಿ GGML_TYPE_BF16: {
#ಆದರೆ defined(__AVX512BF16__)
        ಆದರೆ (Btype == GGML_TYPE_BF16) {
            tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, ದಶಮಾಂಶ> tb{ params, k,
                (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
            ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಅಥವಾಆದರೆ defined(__AVX512F__)
        ಆದರೆ (Btype == GGML_TYPE_BF16) {
            tinyBLAS<16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, ದಶಮಾಂಶ> tb{ params, k,
                (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
            ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಅಥವಾಆದರೆ defined(__AVX2__)
        ಆದರೆ (Btype == GGML_TYPE_BF16) {
            tinyBLAS<8, __m256, __m256, ggml_bf16_t, ggml_bf16_t, ದಶಮಾಂಶ> tb{ params, k,
                (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
            ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಅಥವಾಆದರೆ defined(__MMA__)
        ಆದರೆ (k % 8) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (Btype == GGML_TYPE_BF16) {
            tinyBLAS_HP16_PPC<ggml_bf16_t, ggml_bf16_t, ದಶಮಾಂಶ> tb{ k,
                (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
#ಅಥವಾಆದರೆ defined(__riscv_zvfbfwma)
        ಆದರೆ (Btype == GGML_TYPE_BF16) {
            #ಆದರೆ LMUL == 1
                tinyBLAS_RVV<vfloat32m1_t, vbfloat16mf2_t, ggml_bf16_t, ggml_bf16_t, ದಶಮಾಂಶ> tb{ params,
                    k, (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)A, lda,
                    (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)B, ldb,
                    (ದಶಮಾಂಶ *)C, ldc};
            #ಅಥವಾಆದರೆ LMUL == 2
                tinyBLAS_RVV<vfloat32m2_t, vbfloat16m1_t, ggml_bf16_t, ggml_bf16_t, ದಶಮಾಂಶ> tb{ params,
                    k, (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)A, lda,
                    (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)B, ldb,
                    (ದಶಮಾಂಶ *)C, ldc};
            #ಇಲ್ಲದಿದ್ದರೆ // LMUL = 4
                tinyBLAS_RVV<vfloat32m4_t, vbfloat16m2_t, ggml_bf16_t, ggml_bf16_t, ದಶಮಾಂಶ> tb{ params,
                    k, (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)A, lda,
                    (ಸ್ಥಿರಾಂಕ ggml_bf16_t *)B, ldb,
                    (ದಶಮಾಂಶ *)C, ldc};
            #ಆದರೆ_ಅಂತ್ಯ
                ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿತಿ GGML_TYPE_F16: {
#ಆದರೆ defined(__AVX512F__)
        ಆದರೆ (Btype == GGML_TYPE_F16) {
            tinyBLAS<16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, ದಶಮಾಂಶ> tb{ params, k,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
            ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಅಥವಾಆದರೆ (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        ಆದರೆ (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, ದಶಮಾಂಶ> tb{ params, k,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
            ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಅಥವಾಆದರೆ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
        ಆದರೆ (n < 8)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, ದಶಮಾಂಶ> tb{ params,
                k, (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
            ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಅಥವಾಆದರೆ defined(__ARM_NEON) && !defined(_MSC_VER)
        ಆದರೆ (Btype == GGML_TYPE_F32) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ> tb{ params,
                k, (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
            ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಅಥವಾಆದರೆ defined(__VXE__) || defined(__VXE2__)
        ಆದರೆ (n < 4)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (Btype == GGML_TYPE_F16) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, ದಶಮಾಂಶ> tb{ params,
                k, (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
            ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಅಥವಾಆದರೆ defined(__riscv_zvfh)
        ಆದರೆ (Btype == GGML_TYPE_F16) {
        #ಆದರೆ LMUL == 1
            tinyBLAS_RVV<vfloat32m1_t, vfloat16mf2_t, ggml_fp16_t, ggml_fp16_t, ದಶಮಾಂಶ> tb{ params,
                k, (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
        #ಅಥವಾಆದರೆ LMUL == 2
            tinyBLAS_RVV<vfloat32m2_t, vfloat16m1_t, ggml_fp16_t, ggml_fp16_t, ದಶಮಾಂಶ> tb{ params,
                k, (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
        #ಇಲ್ಲದಿದ್ದರೆ // LMUL = 4
            tinyBLAS_RVV<vfloat32m4_t, vfloat16m2_t, ggml_fp16_t, ggml_fp16_t, ದಶಮಾಂಶ> tb{ params,
                k, (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc};
        #ಆದರೆ_ಅಂತ್ಯ
            ಹಿಂತಿರುಗಿ tb.matmul(m, n);
        }
#ಅಥವಾಆದರೆ defined(__MMA__)
        ಆದರೆ (k % 8) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (Btype == GGML_TYPE_F16) {
            tinyBLAS_HP16_PPC<ggml_fp16_t, ggml_fp16_t, ದಶಮಾಂಶ> tb{ k,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)A, lda,
                (ಸ್ಥಿರಾಂಕ ggml_fp16_t *)B, ldb,
                (ದಶಮಾಂಶ *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿತಿ GGML_TYPE_Q8_0: {
        ಆದರೆ (Btype != GGML_TYPE_Q8_0)
           ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ಖಂಡ_q8_0, ಖಂಡ_q8_0, ದಶಮಾಂಶ> tb{
            k, (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)A, lda,
            (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ಹಿಂತಿರುಗಿ ನಿಜ;
#ಅಥವಾಆದರೆ defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<ಖಂಡ_q8_0> tb{
            k, (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)A, lda,
            (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ಹಿಂತಿರುಗಿ ನಿಜ;
#ಅಥವಾಆದರೆ defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        ಆದರೆ (n < 8 && n != 4)
           ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (m < 8 && m != 4)
           ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        tinyBLAS_Q0_PPC<ಖಂಡ_q8_0> tb{
            k, (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)A, lda,
            (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ಹಿಂತಿರುಗಿ ನಿಜ;
#ಇಲ್ಲದಿದ್ದರೆ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
    }

    ಸ್ಥಿತಿ GGML_TYPE_Q4_0: {
        ಆದರೆ (Btype != GGML_TYPE_Q8_0)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ಖಂಡ_q4_0, ಖಂಡ_q8_0, ದಶಮಾಂಶ> tb{
            k, (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *)A, lda,
            (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ಹಿಂತಿರುಗಿ ನಿಜ;
#ಅಥವಾಆದರೆ defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<ಖಂಡ_q4_0> tb{
            k, (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *)A, lda,
            (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ಹಿಂತಿರುಗಿ ನಿಜ;
#ಅಥವಾಆದರೆ defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        ಆದರೆ (n < 8 && n != 4)
           ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (m < 8 && m != 4)
           ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        tinyBLAS_Q0_PPC<ಖಂಡ_q4_0> tb{
            k, (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *)A, lda,
            (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ಹಿಂತಿರುಗಿ ನಿಜ;
#ಇಲ್ಲದಿದ್ದರೆ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
    }

    ಸ್ಥಿತಿ GGML_TYPE_Q5_0: {
        ಆದರೆ (Btype != GGML_TYPE_Q8_0)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ಖಂಡ_q5_0, ಖಂಡ_q8_0, ದಶಮಾಂಶ> tb{
            k, (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *)A, lda,
            (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ಹಿಂತಿರುಗಿ ನಿಜ;
#ಇಲ್ಲದಿದ್ದರೆ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
    }

    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL: {
        ಆದರೆ (Btype != GGML_TYPE_Q8_0)
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ಖಂಡ_iq4_nl, ಖಂಡ_q8_0, ದಶಮಾಂಶ> tb{
            k, (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl *)A, lda,
            (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *)B, ldb,
            (ದಶಮಾಂಶ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ಹಿಂತಿರುಗಿ ನಿಜ;
#ಇಲ್ಲದಿದ್ದರೆ
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
    }

    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    (ಶೂನ್ಯ)params;
    (ಶೂನ್ಯ)m;
    (ಶೂನ್ಯ)n;
    (ಶೂನ್ಯ)k;
    (ಶೂನ್ಯ)A;
    (ಶೂನ್ಯ)lda;
    (ಶೂನ್ಯ)B;
    (ಶೂನ್ಯ)ldb;
    (ಶೂನ್ಯ)C;
    (ಶೂನ್ಯ)ldc;
    (ಶೂನ್ಯ)Atype;
    (ಶೂನ್ಯ)Btype;
    (ಶೂನ್ಯ)Ctype;
}
