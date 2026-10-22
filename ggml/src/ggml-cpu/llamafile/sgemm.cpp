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

#ਜੇ defined(__GNUC__)
#ਹਦਾਇਤ GCC diagnostic ignored "-Wpedantic"
#ਹਦਾਇਤ GCC diagnostic ignored "-Wignored-attributes"
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ "sgemm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "simd-mappings.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <array>
#ਅਨ੍ਤਰ੍ਭਾਵ <type_traits>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _MSC_VER
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NOINLINE __declspec(noinline)
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NOINLINE __attribute__((__noinline__))
#ਜੇ_ਅੰਤ

#ਜੇ defined(__ARM_NEON) || defined(__AVX512F__) || defined(__VXE__) || defined(__VXE2__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਦਿਸ਼_REGISTERS 32
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਦਿਸ਼_REGISTERS 16
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_v_intrinsic)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LMUL 4
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MM256_ਸੈੱਟ_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

ਨਾਮਾਕਾਸ਼ {

ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ unhalf(ggml_fp16_t d) {
    ਵਾਪਸ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#ਜੇ defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ਸਤਰਅੰਦਰ __m128 ਜੋੜੋ_2(__m128 x, __m128 y) { ਵਾਪਸ mm_ਜੋੜੋ_ps(x, y); }
ਸਤਰਅੰਦਰ __m128 sub(__m128 x, __m128 y) { ਵਾਪਸ _mm_sub_ps(x, y); }
ਸਤਰਅੰਦਰ __m128 mul(__m128 x, __m128 y) { ਵਾਪਸ _mm_mul_ps(x, y); }
#ਜੇ_ਅੰਤ  // __SSE__

#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ਸਤਰਅੰਦਰ __m256 ਜੋੜੋ_2(__m256 x, __m256 y) { ਵਾਪਸ mm256_ਜੋੜੋ_ps(x, y); }
ਸਤਰਅੰਦਰ __m256 sub(__m256 x, __m256 y) { ਵਾਪਸ _mm256_sub_ps(x, y); }
ਸਤਰਅੰਦਰ __m256 mul(__m256 x, __m256 y) { ਵਾਪਸ _mm256_mul_ps(x, y); }
#ਜੇ_ਅੰਤ // __AVX__

#ਜੇ defined(__AVX512F__)
ਸਤਰਅੰਦਰ __m512 ਜੋੜੋ_2(__m512 x, __m512 y) { ਵਾਪਸ mm512_ਜੋੜੋ_ps(x, y); }
ਸਤਰਅੰਦਰ __m512 sub(__m512 x, __m512 y) { ਵਾਪਸ _mm512_sub_ps(x, y); }
ਸਤਰਅੰਦਰ __m512 mul(__m512 x, __m512 y) { ਵਾਪਸ _mm512_mul_ps(x, y); }
#ਜੇ_ਅੰਤ // __AVX512F__

#ਜੇ defined(__ARM_NEON)
ਸਤਰਅੰਦਰ float32x4_t ਜੋੜੋ_2(float32x4_t x, float32x4_t y) { ਵਾਪਸ vaddq_f32(x, y); }
ਸਤਰਅੰਦਰ float32x4_t sub(float32x4_t x, float32x4_t y) { ਵਾਪਸ vsubq_f32(x, y); }
ਸਤਰਅੰਦਰ float32x4_t mul(float32x4_t x, float32x4_t y) { ਵਾਪਸ vmulq_f32(x, y); }
#ਜੇ_ਅੰਤ // __ARM_NEON

#ਜੇ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
ਸਤਰਅੰਦਰ float16x8_t ਜੋੜੋ_2(float16x8_t x, float16x8_t y) { ਵਾਪਸ vaddq_f16(x, y); }
ਸਤਰਅੰਦਰ float16x8_t sub(float16x8_t x, float16x8_t y) { ਵਾਪਸ vsubq_f16(x, y); }
ਸਤਰਅੰਦਰ float16x8_t mul(float16x8_t x, float16x8_t y) { ਵਾਪਸ vmulq_f16(x, y); }
#ਜੇ_ਅੰਤ // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#ਜੇ defined(__VXE__) || defined(__VXE2__)
ਸਤਰਅੰਦਰ float32x4_t ਜੋੜੋ_2(float32x4_t x, float32x4_t y) { ਵਾਪਸ vec_ਜੋੜੋ(x, y); }
ਸਤਰਅੰਦਰ float32x4_t sub(float32x4_t x, float32x4_t y) { ਵਾਪਸ vec_sub(x, y); }
ਸਤਰਅੰਦਰ float32x4_t mul(float32x4_t x, float32x4_t y) { ਵਾਪਸ vec_mul(x, y); }
#ਜੇ_ਅੰਤ

#ਜੇ defined(__MMA__)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ vec_t;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ __vector_quad acc_t;
#ਜੇ_ਅੰਤ
////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ U>
ਸਤਰਅੰਦਰ U madd(T a, T b, U c) {
    ਵਾਪਸ ਜੋੜੋ_2(mul(a, b), c);
}

#ਜੇ defined(__FMA__)
#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ __m256 madd(__m256 a, __m256 b, __m256 c) {
    ਵਾਪਸ _mm256_fmadd_ps(a, b, c);
}
#ਜੇ_ਅੰਤ
#ਜੇ defined(__AVX512F__)
ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ __m512 madd(__m512 a, __m512 b, __m512 c) {
    ਵਾਪਸ _mm512_fmadd_ps(a, b, c);
}
#ਜੇ_ਅੰਤ
#ਜੇ defined(__AVX512BF16__)
ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ __m512 madd(__m512bh a, __m512bh b, __m512 c) {
    ਵਾਪਸ _mm512_dpbf16_ps(c, a, b);
}
ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ __m256 madd(__m256bh a, __m256bh b, __m256 c) {
    ਵਾਪਸ _mm256_dpbf16_ps(c, a, b);
}
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ

#ਜੇ defined(__ARM_FEATURE_FMA)
ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    ਵਾਪਸ vfmaq_f32(c, b, a);
}
#ਜੇ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    ਵਾਪਸ vfmaq_f16(c, b, a);
}
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ

#ਜੇ defined(__VXE__) || defined(__VXE2__)
ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    ਵਾਪਸ vec_madd(a, b, c);
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_v_intrinsic)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m1_t madd(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t c) {
    ਵਾਪਸ __riscv_vfmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m2_t madd(vfloat32m2_t a, vfloat32m2_t b, vfloat32m2_t c) {
    ਵਾਪਸ __riscv_vfmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m4_t madd(vfloat32m4_t a, vfloat32m4_t b, vfloat32m4_t c) {
    ਵਾਪਸ __riscv_vfmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m8_t madd(vfloat32m8_t a, vfloat32m8_t b, vfloat32m8_t c) {
    ਵਾਪਸ __riscv_vfmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_zvfh)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m1_t madd(vfloat16mf2_t a, vfloat16mf2_t b, vfloat32m1_t c) {
    ਵਾਪਸ __riscv_vfwmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m2_t madd(vfloat16m1_t a, vfloat16m1_t b, vfloat32m2_t c) {
    ਵਾਪਸ __riscv_vfwmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m4_t madd(vfloat16m2_t a, vfloat16m2_t b, vfloat32m4_t c) {
    ਵਾਪਸ __riscv_vfwmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m8_t madd(vfloat16m4_t a, vfloat16m4_t b, vfloat32m8_t c) {
    ਵਾਪਸ __riscv_vfwmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_zvfbfwma)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m1_t madd(vbfloat16mf2_t a, vbfloat16mf2_t b, vfloat32m1_t c) {
    ਵਾਪਸ __riscv_vfwmaccbf16_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m2_t madd(vbfloat16m1_t a, vbfloat16m1_t b, vfloat32m2_t c) {
    ਵਾਪਸ __riscv_vfwmaccbf16_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m4_t madd(vbfloat16m2_t a, vbfloat16m2_t b, vfloat32m4_t c) {
    ਵਾਪਸ __riscv_vfwmaccbf16_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m8_t madd(vbfloat16m4_t a, vbfloat16m4_t b, vfloat32m8_t c) {
    ਵਾਪਸ __riscv_vfwmaccbf16_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#ਜੇ_ਅੰਤ

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#ਜੇ defined(__ARM_NEON) || defined(_M_ARM64) || defined(_M_ARM64EC)
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(float32x4_t x) {
    ਵਾਪਸ vaddvq_f32(x);
}
#ਜੇ_ਅੰਤ // __ARM_NEON

#ਜੇ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(float16x8_t x) {
    ਵਾਪਸ vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_ਨੀਵਾਂ_f16(x)),
                                vcvt_f32_f16(vget_ਉੱਚਾ_f16(x))));
}
#ਜੇ_ਅੰਤ // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#ਜੇ defined(__VXE__) || defined(__VXE2__)
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(float32x4_t x) {
    float32x4_t tmp = x + vec_reve(x);
    ਵਾਪਸ tmp[0] + tmp[1];
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(__m128 x) {
#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = mm_ਜੋੜੋ_ps(x, _mm_movehl_ps(x, x));
    x = mm_ਜੋੜੋ_ss(x, _mm_movehdup_ps(x));
#ਨਹੀਂਤਾਂ
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = mm_ਜੋੜੋ_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = mm_ਜੋੜੋ_ss(x, t);
#ਜੇ_ਅੰਤ
    ਵਾਪਸ _mm_cvtss_f32(x);
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(__m256 x) {
    ਵਾਪਸ hsum(mm_ਜੋੜੋ_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#ਜੇ_ਅੰਤ // __AVX__

#ਜੇ defined(__AVX512F__)
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(__m512 x) {
    ਵਾਪਸ mm512_reduce_ਜੋੜੋ_ps(x);
}
#ਜੇ_ਅੰਤ // __AVX512F__

#ਜੇ defined(__riscv_v_intrinsic)
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(vfloat32m1_t x) {
    ਵਾਪਸ __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m1_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m1()));
}
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(vfloat32m2_t x) {
    ਵਾਪਸ __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m2()));
}
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(vfloat32m4_t x) {
    ਵਾਪਸ __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m4()));
}
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ hsum(vfloat32m8_t x) {
    ਵਾਪਸ __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m8_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m8()));
}
#ਜੇ_ਅੰਤ

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ U> T load(ਅਚਲ U *);

#ਜੇ defined(__ARM_NEON)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ float32x4_t load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ vld1q_f32(p);
}
#ਜੇ !defined(_MSC_VER)
// FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ float16x8_t load(ਅਚਲ ggml_fp16_t *p) {
    ਵਾਪਸ vld1q_f16((ਅਚਲ float16_t *)p);
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ float32x4_t load(ਅਚਲ ggml_fp16_t *p) {
    ਵਾਪਸ vcvt_f32_f16(vld1_f16((ਅਚਲ float16_t *)p));
}
#ਜੇ_ਅੰਤ // _MSC_VER
#ਜੇ_ਅੰਤ // __ARM_NEON

#ਜੇ defined(__VXE__) || defined(__VXE2__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ float32x4_t load(ਅਚਲ ggml_fp16_t * p) {
    ਦਸ਼ਮਲਵ tmp[4];

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(p[i]);
    }

    ਵਾਪਸ vec_xl(0, (ਅਚਲ ਦਸ਼ਮਲਵ *)(tmp));
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ float32x4_t load(ਅਚਲ ਦਸ਼ਮਲਵ * p) {
    ਵਾਪਸ vec_xl(0, p);
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m128 load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ _mm_loadu_ps(p);
}
#ਜੇ_ਅੰਤ  // __SSE__

#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m256 load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ _mm256_loadu_ps(p);
}
#ਜੇ_ਅੰਤ // __AVX__

#ਜੇ defined(__AVX2__) || defined(__AVX512F__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m256 load(ਅਚਲ ggml_bf16_t *p) {
    ਵਾਪਸ _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((ਅਚਲ __m128i *)p)), 16));
}
#ਜੇ_ਅੰਤ // __AVX2__

#ਜੇ defined(__F16C__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m256 load(ਅਚਲ ggml_fp16_t *p) {
    ਵਾਪਸ _mm256_cvtph_ps(_mm_loadu_si128((ਅਚਲ __m128i *)p));
}
#ਜੇ_ਅੰਤ // __F16C__

#ਜੇ defined(__AVX512F__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m512 load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ _mm512_loadu_ps(p);
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m512 load(ਅਚਲ ggml_fp16_t *p) {
    ਵਾਪਸ _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)p));
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m512 load(ਅਚਲ ggml_bf16_t *p) {
    ਵਾਪਸ _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((ਅਚਲ __m256i *)p)), 16));
}
#ਜੇ_ਅੰਤ // __AVX512F__

#ਜੇ defined(__AVX512BF16__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m512bh load(ਅਚਲ ggml_bf16_t *p) {
    ਵਾਪਸ (__m512bh)_mm512_loadu_ps((ਅਚਲ ਦਸ਼ਮਲਵ *)p);
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m256bh load(ਅਚਲ ggml_bf16_t *p) {
    ਵਾਪਸ (__m256bh)_mm256_loadu_ps((ਅਚਲ ਦਸ਼ਮਲਵ *)p);
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m512bh load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m256bh load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ _mm512_cvtneps_pbh(_mm512_loadu_ps(p));
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ U> T ਲੱਦੋ_partial(ਅਚਲ U *, ਪੂਰਨ);
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> T ਲੱਦੋ_partial_u16(ਅਚਲ ਖਾਲੀ *, ਪੂਰਨ);

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m128i ਲੱਦੋ_partial_u16(ਅਚਲ ਖਾਲੀ *p, ਪੂਰਨ n) {
#ਜੇ defined(__AVX512BW__) && defined(__AVX512VL__)
    ਵਾਪਸ _mm_maskz_loadu_epi16((1u << n) - 1, p);
#ਨਹੀਂਤਾਂ
    ਅਚਲ __m128i ਸੂਚਕਾਂਕ = _mm_setr_epi32(0, 1, 2, 3);
    ਅਚਲ __m128i ਜੋੜੇ = _mm_set1_epi32(n / 2);
    __m128i v = _mm_castps_si128(_mm_maskload_ps((ਅਚਲ ਦਸ਼ਮਲਵ *)p, _mm_cmpgt_epi32(ਜੋੜੇ, ਸੂਚਕਾਂਕ)));
    ਜੇ (n & 1) {
        uint16_t ਆਖਰੀ;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਆਖਰੀ, (ਅਚਲ ਅੱਖਰ *)p + 2*(n - 1), ਆਕਾਰ(ਆਖਰੀ));
        v = _mm_or_si128(v, _mm_and_si128(_mm_cmpeq_epi32(ਜੋੜੇ, ਸੂਚਕਾਂਕ), _mm_set1_epi32(ਆਖਰੀ)));
    }
    ਵਾਪਸ v;
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m256 ਲੱਦੋ_partial(ਅਚਲ ਦਸ਼ਮਲਵ *p, ਪੂਰਨ n) {
    ਅਚਲ __m256 ਸੂਚਕਾਂਕ = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    ਵਾਪਸ _mm256_maskload_ps(p, _mm256_castps_si256(_mm256_cmp_ps(ਸੂਚਕਾਂਕ, _mm256_set1_ps(n), _CMP_LT_OQ)));
}

#ਜੇ defined(__F16C__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m256 ਲੱਦੋ_partial(ਅਚਲ ggml_fp16_t *p, ਪੂਰਨ n) {
    ਵਾਪਸ _mm256_cvtph_ps(ਲੱਦੋ_partial_u16<__m128i>(p, n));
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__AVX2__) || defined(__AVX512F__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m256 ਲੱਦੋ_partial(ਅਚਲ ggml_bf16_t *p, ਪੂਰਨ n) {
    ਵਾਪਸ _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(ਲੱਦੋ_partial_u16<__m128i>(p, n)), 16));
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__AVX512F__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m256i ਲੱਦੋ_partial_u16(ਅਚਲ ਖਾਲੀ *p, ਪੂਰਨ n) {
#ਜੇ defined(__AVX512BW__) && defined(__AVX512VL__)
    ਵਾਪਸ _mm256_maskz_loadu_epi16((1u << n) - 1, p);
#ਨਹੀਂਤਾਂ
    ਅਚਲ __m256i ਸੂਚਕਾਂਕ = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    ਅਚਲ __m256i ਜੋੜੇ = _mm256_set1_epi32(n / 2);
    __m256i v = _mm256_maskload_epi32((ਅਚਲ ਪੂਰਨ *)p, _mm256_cmpgt_epi32(ਜੋੜੇ, ਸੂਚਕਾਂਕ));
    ਜੇ (n & 1) {
        uint16_t ਆਖਰੀ;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਆਖਰੀ, (ਅਚਲ ਅੱਖਰ *)p + 2*(n - 1), ਆਕਾਰ(ਆਖਰੀ));
        v = _mm256_or_si256(v, _mm256_and_si256(_mm256_cmpeq_epi32(ਜੋੜੇ, ਸੂਚਕਾਂਕ), _mm256_set1_epi32(ਆਖਰੀ)));
    }
    ਵਾਪਸ v;
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m512 ਲੱਦੋ_partial(ਅਚਲ ਦਸ਼ਮਲਵ *p, ਪੂਰਨ n) {
    ਵਾਪਸ _mm512_maskz_loadu_ps((1u << n) - 1, p);
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m512 ਲੱਦੋ_partial(ਅਚਲ ggml_fp16_t *p, ਪੂਰਨ n) {
    ਵਾਪਸ _mm512_cvtph_ps(ਲੱਦੋ_partial_u16<__m256i>(p, n));
}

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m512 ਲੱਦੋ_partial(ਅਚਲ ggml_bf16_t *p, ਪੂਰਨ n) {
    ਵਾਪਸ _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(ਲੱਦੋ_partial_u16<__m256i>(p, n)), 16));
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__AVX512BF16__)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ __m512bh ਲੱਦੋ_partial(ਅਚਲ ggml_bf16_t *p, ਪੂਰਨ n) {
    ਵਾਪਸ (__m512bh) _mm512_maskz_loadu_epi16((uint64_t(1) << n) - 1, p);
}
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_v_intrinsic)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m1_t load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ __riscv_vle32_v_f32m1(p, __riscv_vsetvlmax_e32m1());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m2_t load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ __riscv_vle32_v_f32m2(p, __riscv_vsetvlmax_e32m2());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m4_t load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ __riscv_vle32_v_f32m4(p, __riscv_vsetvlmax_e32m4());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m8_t load(ਅਚਲ ਦਸ਼ਮਲਵ *p) {
    ਵਾਪਸ __riscv_vle32_v_f32m8(p, __riscv_vsetvlmax_e32m8());
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_zvfh)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat16mf2_t load(ਅਚਲ ggml_fp16_t *p) {
    ਵਾਪਸ __riscv_vle16_v_f16mf2(reinterpret_cast<ਅਚਲ _Float16 *>(p), __riscv_vsetvlmax_e16mf2());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat16m1_t load(ਅਚਲ ggml_fp16_t *p) {
    ਵਾਪਸ __riscv_vle16_v_f16m1(reinterpret_cast<ਅਚਲ _Float16 *>(p), __riscv_vsetvlmax_e16m1());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat16m2_t load(ਅਚਲ ggml_fp16_t *p) {
    ਵਾਪਸ __riscv_vle16_v_f16m2(reinterpret_cast<ਅਚਲ _Float16 *>(p), __riscv_vsetvlmax_e16m2());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat16m4_t load(ਅਚਲ ggml_fp16_t *p) {
    ਵਾਪਸ __riscv_vle16_v_f16m4(reinterpret_cast<ਅਚਲ _Float16 *>(p), __riscv_vsetvlmax_e16m4());
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_zvfbfwma)
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vbfloat16mf2_t load(ਅਚਲ ggml_bf16_t *p) {
    ਵਾਪਸ __riscv_vle16_v_bf16mf2(reinterpret_cast<ਅਚਲ __bf16*>(p), __riscv_vsetvlmax_e16mf2());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vbfloat16m1_t load(ਅਚਲ ggml_bf16_t *p) {
    ਵਾਪਸ __riscv_vle16_v_bf16m1(reinterpret_cast<ਅਚਲ __bf16*>(p), __riscv_vsetvlmax_e16m1());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vbfloat16m2_t load(ਅਚਲ ggml_bf16_t *p) {
    ਵਾਪਸ __riscv_vle16_v_bf16m2(reinterpret_cast<ਅਚਲ __bf16*>(p), __riscv_vsetvlmax_e16m2());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vbfloat16m4_t load(ਅਚਲ ggml_bf16_t *p) {
    ਵਾਪਸ __riscv_vle16_v_bf16m4(reinterpret_cast<ਅਚਲ __bf16*>(p), __riscv_vsetvlmax_e16m4());
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_v_intrinsic)
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> T ਸੈੱਟ_zero();

ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m1_t ਸੈੱਟ_zero() {
    ਵਾਪਸ __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m2_t ਸੈੱਟ_zero() {
    ਵਾਪਸ __riscv_vfmv_v_f_f32m2(0, __riscv_vsetvlmax_e32m2());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m4_t ਸੈੱਟ_zero() {
    ਵਾਪਸ __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
}
ਸਾਂਚਾ <> ਸਤਰਅੰਦਰ vfloat32m8_t ਸੈੱਟ_zero() {
    ਵਾਪਸ __riscv_vfmv_v_f_f32m8(0, __riscv_vsetvlmax_e32m8());
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_v_intrinsic)
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T> size_t vlmax() {
    ਜੇ constexpr (std::is_same_v<T, vfloat32m1_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e32m1(); }
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vfloat32m2_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e32m2(); }
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vfloat32m4_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e32m4(); }
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vfloat32m8_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e32m8(); }
    #ਜੇ defined (__riscv_zvfh)
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vfloat16mf2_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e16mf2(); }
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vfloat16m1_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e16m1(); }
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vfloat16m2_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e16m2(); }
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vfloat16m4_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e16m4(); }
    #ਜੇ_ਅੰਤ
    #ਜੇ defined (__riscv_zvfbfwma)
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vbfloat16mf2_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e16mf2(); }
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vbfloat16m1_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e16m1(); }
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vbfloat16m2_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e16m2(); }
    ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, vbfloat16m4_t>) { ਵਾਪਸ  __riscv_vsetvlmax_e16m4(); }
    #ਜੇ_ਅੰਤ
    ਵਾਪਸ 0;
}
#ਜੇ_ਅੰਤ

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

ਸਾਂਚਾ <ਪੂਰਨ M>
ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t ਖੰਡ_ਮਾਪ_2(size_t m) {
    ਅਚਲ int64_t NB_BLOC_M = (m + M - 1) / M;
    ਵਾਪਸ (m % NB_BLOC_M == 0) ? m / NB_BLOC_M : (m / NB_BLOC_M) + 1;
}

ਸਥਿਰ constexpr ਸਤਰਅੰਦਰ int64_t BLOC_POS(int64_t ib, int64_t ibN, int64_t bloc_ਮਾਪ) {
    ਵਾਪਸ ib < ibN ? ib * bloc_ਮਾਪ : ibN * bloc_ਮਾਪ + (ib - ibN) * (bloc_ਮਾਪ - 1);
}

ਸਾਂਚਾ <ਪੂਰਨ KN, ਕਿਸਮ_ਨਾਂ D, ਕਿਸਮ_ਨਾਂ V, ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC>
ਵਰਗ tinyBLAS {
  ਜਨਤਕ:
    tinyBLAS(ਅਚਲ ggml_compute_params * params, int64_t k,
             ਅਚਲ TA *A, int64_t lda,
             ਅਚਲ TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    ਤਾਰਕਿਕ matmul(int64_t m, int64_t n) {
#ਜੇ !defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
        ਜੇ (k % KN != 0)
            ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
        // compute RM for only need tile with size RM&RM-1
#ਜੇ ਸਦਿਸ਼_REGISTERS == 32
        ਜੇ (m % 16 == 0 && (m/16 >= params->nth)) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<6>(n);
            mnpack<4, 6, 4>(m, n, ਮਾਪ_N, 12);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 8 == 0 ) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<6>(n);
            mnpack<4, 6, 2>(m, n, ਮਾਪ_N, 12);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 4 == 0) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<6>(n);
            mnpack<4, 6, 1>(m, n, ਮਾਪ_N, 12);
            ਵਾਪਸ ਸੱਚ;
        }
#ਨਹੀਂਤਾਂ  // VECTOR_REGISTERS == 16
        ਜੇ (m % 16 == 0 && (m/16 >= params->nth)) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<3>(n);
            mnpack<4, 3, 4>(m, n, ਮਾਪ_N, 24);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 8 == 0 ) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<3>(n);
            mnpack<4, 3, 2>(m, n, ਮਾਪ_N, 24);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 4 == 0) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<3>(n);
            mnpack<4, 3, 1>(m, n, ਮਾਪ_N, 24);
            ਵਾਪਸ ਸੱਚ;
        }
#ਜੇ_ਅੰਤ
        ਵਾਪਸ ਝੂਠ;
    }

  ਨਿਜੀ:
    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN, ਪੂਰਨ BM>
    ਸਤਰਅੰਦਰ ਖਾਲੀ mnpack(int64_t m, int64_t n, int64_t ਮਾਪ_N, int64_t BN) {
        ਜੇ (ਮਾਪ_N == RN) {
            ਵਾਪਸ gemm<RM, RN, BM>(m, n, BN);
        }
        ਜੇ constexpr (RN > 1) {
            ਵਾਪਸ mnpack<RM, RN-1, BM>(m, n, ਮਾਪ_N, BN);
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (ਪੂਰਨ)ਮਾਪ_N);
            GGML_ASSERT(ਝੂਠ); // we have miss something.
        }
    }

    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN>
    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc(int64_t ii, int64_t jj) {
        D Cv[RN][RM] = {};
        ਲਈ (int64_t l = 0; l + KN <= k; l += KN) {
            // help compiler for op order.
            ਜੇ constexpr (RM <= RN) {
                V Av[RM];
                ਲਈ (int64_t i = 0; i < RM; ++i) {
                    Av[i] = load<V>(A + lda * (ii + i) + l);
                }
                ਲਈ (int64_t j = 0; j < RN; ++j) {
                    V Bv = load<V>(B + ldb * (jj + j) + l);
                    ਲਈ (int64_t i = 0; i < RM; ++i) {
                        Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                    }
                }
            } ਨਹੀਂਤਾਂ {
                V Bv[RN];
                ਲਈ (int64_t j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(B + ldb * (jj + j) + l);
                }
                ਲਈ (int64_t i = 0; i < RM; ++i) {
                    V Av = load<V>(A + lda * (ii + i) + l);
                    ਲਈ (int64_t j = 0; j < RN; ++j) {
                        Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
                    }
                }
            }
        }
#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        ਅਚਲ int64_t rem = k % KN;
        ਜੇ (rem != 0) {
            V Av[RM];
            ਲਈ (int64_t i = 0; i < RM; ++i) {
                Av[i] = ਲੱਦੋ_partial<V>(A + lda * (ii + i) + k - rem, rem);
            }
            ਲਈ (int64_t j = 0; j < RN; ++j) {
                V Bv = ਲੱਦੋ_partial<V>(B + ldb * (jj + j) + k - rem, rem);
                ਲਈ (int64_t i = 0; i < RM; ++i) {
                    Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                }
            }
        }
#ਜੇ_ਅੰਤ
        ਲਈ (int64_t j = 0; j < RN; ++j)
            ਲਈ (int64_t i = 0; i < RM; ++i)
                C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
    }

    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN, ਪੂਰਨ BM>
    NOINLINE ਖਾਲੀ gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        ਅਚਲ int64_t ytiles = m / (RM * BM);
        ਅਚਲ int64_t xtiles = (n + RN -1) / RN;
        ਅਚਲ int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        ਅਚਲ int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        ਅਚਲ int64_t ਮਾਪ_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        ਅਚਲ int64_t jj_BN = (NB_BN - (NB_BN * ਮਾਪ_BN - xtiles));
        ਅਚਲ int64_t nb_ਕੰਮ = ytiles * NB_BN;

        ਜੇ (params->ith == 0) {
            GGML_ASSERT( jj_BN * ਮਾਪ_BN + (NB_BN - jj_BN) * (ਮਾਪ_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_ਸੈੱਟ(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t ਕੰਮ = params->ith;
        ਜਦੋਂ (ਕੰਮ < nb_ਕੰਮ) {
            ਅਚਲ int64_t ii = (ਕੰਮ % ytiles) * RM * BM;
            ਅਚਲ int64_t jb =  ਕੰਮ / ytiles;
            ਅਚਲ int64_t jr0 = BLOC_POS(jb  , jj_BN, ਮਾਪ_BN);
            ਅਚਲ int64_t jrN = BLOC_POS(jb+1, jj_BN, ਮਾਪ_BN);

            ਅਚਲ int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            ਅਚਲ int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            ਅਚਲ int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            ਲਈ (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                ਲਈ (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                ਜੇ constexpr (RN > 1) {
                    ਲਈ (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            ਕੰਮ = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        ਵਾਪਸ;
    }

    ਅਚਲ ggml_compute_params * params;
    ਅਚਲ TA *ਅਚਲ A;
    ਅਚਲ TB *ਅਚਲ B;
    TC *ਅਚਲ C;
    ਅਚਲ int64_t k;
    ਅਚਲ int64_t lda;
    ਅਚਲ int64_t ldb;
    ਅਚਲ int64_t ldc;
};

#ਜੇ defined(__riscv_v_intrinsic)
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ D, ਕਿਸਮ_ਨਾਂ V, ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC>
ਵਰਗ tinyBLAS_RVV {
  ਜਨਤਕ:
    tinyBLAS_RVV(ਅਚਲ ggml_compute_params * params, int64_t k,
             ਅਚਲ TA *A, int64_t lda,
             ਅਚਲ TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    ਤਾਰਕਿਕ matmul(int64_t m, int64_t n) {
        ਜੇ (k % vlmax<V>() != 0) {
            ਵਾਪਸ ਝੂਠ;
        }

#ਜੇ LMUL == 1
        ਜੇ (m % 16 == 0 && (m/16 >= params->nth)) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<6>(n);
            mnpack<4, 6, 4>(m, n, ਮਾਪ_N, 12);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 8 == 0 ) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<6>(n);
            mnpack<4, 6, 2>(m, n, ਮਾਪ_N, 12);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 4 == 0) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<6>(n);
            mnpack<4, 6, 1>(m, n, ਮਾਪ_N, 12);
            ਵਾਪਸ ਸੱਚ;
        }
#ਜਾਂਜੇ LMUL == 2
        ਜੇ (m % 16 == 0 && (m/16 >= params->nth)) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<3>(n);
            mnpack<4, 3, 4>(m, n, ਮਾਪ_N, 24);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 8 == 0 ) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<3>(n);
            mnpack<4, 3, 2>(m, n, ਮਾਪ_N, 24);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 4 == 0) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<3>(n);
            mnpack<4, 3, 1>(m, n, ਮਾਪ_N, 24);
            ਵਾਪਸ ਸੱਚ;
        }
#ਨਹੀਂਤਾਂ // LMUL = 4
        ਜੇ (m % 16 == 0 && (m/16 >= params->nth)) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<2>(n);
            mnpack<2, 2, 8>(m, n, ਮਾਪ_N, 36);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 8 == 0 ) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<2>(n);
            mnpack<2, 2, 4>(m, n, ਮਾਪ_N, 36);
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (m % 4 == 0) {
            ਅਚਲ int64_t ਮਾਪ_N = ਖੰਡ_ਮਾਪ_2<2>(n);
            mnpack<2, 2, 2>(m, n, ਮਾਪ_N, 36);
            ਵਾਪਸ ਸੱਚ;
        }
#ਜੇ_ਅੰਤ
        ਵਾਪਸ ਝੂਠ;
    }

  ਨਿਜੀ:
    ਸਾਂਚਾ<ਪੂਰਨ RM, ਪੂਰਨ RN, ਪੂਰਨ BM>
    ਸਤਰਅੰਦਰ ਖਾਲੀ mnpack(int64_t m, int64_t n, int64_t ਮਾਪ_N, int64_t BN) {
        ਜੇ (ਮਾਪ_N == RN) {
            ਵਾਪਸ gemm<RM, RN, BM>(m, n, BN);
        }
        ਜੇ constexpr (RN > 1) {
            ਵਾਪਸ mnpack<RM, RN-1, BM>(m, n, ਮਾਪ_N, BN);
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (ਪੂਰਨ)ਮਾਪ_N);
            GGML_ASSERT(ਝੂਠ); // we have miss something.
        }
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc_4x6(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ਸੈੱਟ_zero<D>();
        D Cv01 = ਸੈੱਟ_zero<D>();
        D Cv02 = ਸੈੱਟ_zero<D>();
        D Cv03 = ਸੈੱਟ_zero<D>();
        D Cv10 = ਸੈੱਟ_zero<D>();
        D Cv11 = ਸੈੱਟ_zero<D>();
        D Cv12 = ਸੈੱਟ_zero<D>();
        D Cv13 = ਸੈੱਟ_zero<D>();
        D Cv20 = ਸੈੱਟ_zero<D>();
        D Cv21 = ਸੈੱਟ_zero<D>();
        D Cv22 = ਸੈੱਟ_zero<D>();
        D Cv23 = ਸੈੱਟ_zero<D>();
        D Cv30 = ਸੈੱਟ_zero<D>();
        D Cv31 = ਸੈੱਟ_zero<D>();
        D Cv32 = ਸੈੱਟ_zero<D>();
        D Cv33 = ਸੈੱਟ_zero<D>();
        D Cv40 = ਸੈੱਟ_zero<D>();
        D Cv41 = ਸੈੱਟ_zero<D>();
        D Cv42 = ਸੈੱਟ_zero<D>();
        D Cv43 = ਸੈੱਟ_zero<D>();
        D Cv50 = ਸੈੱਟ_zero<D>();
        D Cv51 = ਸੈੱਟ_zero<D>();
        D Cv52 = ਸੈੱਟ_zero<D>();
        D Cv53 = ਸੈੱਟ_zero<D>();

        ਲਈ (int64_t l = 0; l < k; l += vl) {
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

    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc_4x5(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ਸੈੱਟ_zero<D>();
        D Cv01 = ਸੈੱਟ_zero<D>();
        D Cv02 = ਸੈੱਟ_zero<D>();
        D Cv03 = ਸੈੱਟ_zero<D>();
        D Cv10 = ਸੈੱਟ_zero<D>();
        D Cv11 = ਸੈੱਟ_zero<D>();
        D Cv12 = ਸੈੱਟ_zero<D>();
        D Cv13 = ਸੈੱਟ_zero<D>();
        D Cv20 = ਸੈੱਟ_zero<D>();
        D Cv21 = ਸੈੱਟ_zero<D>();
        D Cv22 = ਸੈੱਟ_zero<D>();
        D Cv23 = ਸੈੱਟ_zero<D>();
        D Cv30 = ਸੈੱਟ_zero<D>();
        D Cv31 = ਸੈੱਟ_zero<D>();
        D Cv32 = ਸੈੱਟ_zero<D>();
        D Cv33 = ਸੈੱਟ_zero<D>();
        D Cv40 = ਸੈੱਟ_zero<D>();
        D Cv41 = ਸੈੱਟ_zero<D>();
        D Cv42 = ਸੈੱਟ_zero<D>();
        D Cv43 = ਸੈੱਟ_zero<D>();

        ਲਈ (int64_t l = 0; l < k; l += vl) {
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

    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc_4x4(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ਸੈੱਟ_zero<D>();
        D Cv01 = ਸੈੱਟ_zero<D>();
        D Cv02 = ਸੈੱਟ_zero<D>();
        D Cv03 = ਸੈੱਟ_zero<D>();
        D Cv10 = ਸੈੱਟ_zero<D>();
        D Cv11 = ਸੈੱਟ_zero<D>();
        D Cv12 = ਸੈੱਟ_zero<D>();
        D Cv13 = ਸੈੱਟ_zero<D>();
        D Cv20 = ਸੈੱਟ_zero<D>();
        D Cv21 = ਸੈੱਟ_zero<D>();
        D Cv22 = ਸੈੱਟ_zero<D>();
        D Cv23 = ਸੈੱਟ_zero<D>();
        D Cv30 = ਸੈੱਟ_zero<D>();
        D Cv31 = ਸੈੱਟ_zero<D>();
        D Cv32 = ਸੈੱਟ_zero<D>();
        D Cv33 = ਸੈੱਟ_zero<D>();

        ਲਈ (int64_t l = 0; l < k; l += vl) {
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

    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc_4x3(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ਸੈੱਟ_zero<D>();
        D Cv01 = ਸੈੱਟ_zero<D>();
        D Cv02 = ਸੈੱਟ_zero<D>();
        D Cv03 = ਸੈੱਟ_zero<D>();
        D Cv10 = ਸੈੱਟ_zero<D>();
        D Cv11 = ਸੈੱਟ_zero<D>();
        D Cv12 = ਸੈੱਟ_zero<D>();
        D Cv13 = ਸੈੱਟ_zero<D>();
        D Cv20 = ਸੈੱਟ_zero<D>();
        D Cv21 = ਸੈੱਟ_zero<D>();
        D Cv22 = ਸੈੱਟ_zero<D>();
        D Cv23 = ਸੈੱਟ_zero<D>();

        ਲਈ (int64_t l = 0; l < k; l += vl) {
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

    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc_4x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ਸੈੱਟ_zero<D>();
        D Cv01 = ਸੈੱਟ_zero<D>();
        D Cv02 = ਸੈੱਟ_zero<D>();
        D Cv03 = ਸੈੱਟ_zero<D>();
        D Cv10 = ਸੈੱਟ_zero<D>();
        D Cv11 = ਸੈੱਟ_zero<D>();
        D Cv12 = ਸੈੱਟ_zero<D>();
        D Cv13 = ਸੈੱਟ_zero<D>();

        ਲਈ (int64_t l = 0; l < k; l += vl) {
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

    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc_4x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ਸੈੱਟ_zero<D>();
        D Cv01 = ਸੈੱਟ_zero<D>();
        D Cv02 = ਸੈੱਟ_zero<D>();
        D Cv03 = ਸੈੱਟ_zero<D>();

        ਲਈ (int64_t l = 0; l < k; l += vl) {
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

    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc_2x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ਸੈੱਟ_zero<D>();
        D Cv01 = ਸੈੱਟ_zero<D>();
        D Cv10 = ਸੈੱਟ_zero<D>();
        D Cv11 = ਸੈੱਟ_zero<D>();

        ਲਈ (int64_t l = 0; l < k; l += vl) {
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

    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc_2x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = ਸੈੱਟ_zero<D>();
        D Cv01 = ਸੈੱਟ_zero<D>();

        ਲਈ (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
    }

    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN>
    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_bloc(int64_t ii, int64_t jj) {
        ਜੇ constexpr (RM == 4) {
            ਜੇ constexpr (RN == 6) { ਵਾਪਸ gemm_bloc_4x6(ii, jj); }
            ਜੇ constexpr (RN == 5) { ਵਾਪਸ gemm_bloc_4x5(ii, jj); }
            ਜੇ constexpr (RN == 4) { ਵਾਪਸ gemm_bloc_4x4(ii, jj); }
            ਜੇ constexpr (RN == 3) { ਵਾਪਸ gemm_bloc_4x3(ii, jj); }
            ਜੇ constexpr (RN == 2) { ਵਾਪਸ gemm_bloc_4x2(ii, jj); }
            ਜੇ constexpr (RN == 1) { ਵਾਪਸ gemm_bloc_4x1(ii, jj); }
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (RM == 2) {
            ਜੇ constexpr (RN == 2) { ਵਾਪਸ gemm_bloc_2x2(ii, jj); }
            ਜੇ constexpr (RN == 1) { ਵਾਪਸ gemm_bloc_2x1(ii, jj); }
        }
    }

    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN, ਪੂਰਨ BM>
    NOINLINE ਖਾਲੀ gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        ਅਚਲ int64_t ytiles = m / (RM * BM);
        ਅਚਲ int64_t xtiles = (n + RN -1) / RN;
        ਅਚਲ int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        ਅਚਲ int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        ਅਚਲ int64_t ਮਾਪ_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        ਅਚਲ int64_t jj_BN = (NB_BN - (NB_BN * ਮਾਪ_BN - xtiles));
        ਅਚਲ int64_t nb_ਕੰਮ = ytiles * NB_BN;

        ਜੇ (params->ith == 0) {
            GGML_ASSERT( jj_BN * ਮਾਪ_BN + (NB_BN - jj_BN) * (ਮਾਪ_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_ਸੈੱਟ(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t ਕੰਮ = params->ith;
        ਜਦੋਂ (ਕੰਮ < nb_ਕੰਮ) {
            ਅਚਲ int64_t ii = (ਕੰਮ % ytiles) * RM * BM;
            ਅਚਲ int64_t jb =  ਕੰਮ / ytiles;
            ਅਚਲ int64_t jr0 = BLOC_POS(jb  , jj_BN, ਮਾਪ_BN);
            ਅਚਲ int64_t jrN = BLOC_POS(jb+1, jj_BN, ਮਾਪ_BN);

            ਅਚਲ int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            ਅਚਲ int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            ਅਚਲ int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            ਲਈ (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                ਲਈ (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                ਜੇ constexpr (RN > 1) {
                    ਲਈ (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            ਕੰਮ = ggml_threadpool_chunk_ਜੋੜੋ(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        ਵਾਪਸ;
    }

    ਅਚਲ ggml_compute_params * params;
    ਅਚਲ TA *ਅਚਲ A;
    ਅਚਲ TB *ਅਚਲ B;
    TC *ਅਚਲ C;
    ਅਚਲ int64_t k;
    ਅਚਲ int64_t lda;
    ਅਚਲ int64_t ldb;
    ਅਚਲ int64_t ldc;
};
#ਜੇ_ਅੰਤ

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#ਜੇ defined(__ARM_FEATURE_DOTPROD)
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA>
ਵਰਗ tinyBLAS_Q0_ARM {
  ਜਨਤਕ:
    tinyBLAS_Q0_ARM(int64_t k,
                    ਅਚਲ TA *A, int64_t lda,
                    ਅਚਲ ਖੰਡ_q8_0 *B, int64_t ldb,
                    ਦਸ਼ਮਲਵ *C, int64_t ldc,
                    ਪੂਰਨ ith, ਪੂਰਨ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    ਖਾਲੀ matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  ਨਿਜੀ:
    NOINLINE ਖਾਲੀ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        ਚੋਣ ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3ll)) {
        ਹਾਲਤ 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN>
    NOINLINE ਖਾਲੀ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            float32x4_t Cv[RN][RM] = {};
            ਲਈ (int64_t l = 0; l < k; ++l)
                ਲਈ (int64_t j = 0; j < RN; ++j)
                    ਲਈ (int64_t i = 0; i < RM; ++i)
                        Cv[j][i] = vmlaq_n_f32(Cv[j][i],
                                               vcvtq_f32_s32(vdotq_s32(
                                                   vdotq_s32(vdupq_n_s32(0),
                                                             ਲੱਦੋ_lo(A + lda * (ii + i) + l),
                                                             ਲੱਦੋ_lo(B + ldb * (jj + j) + l)),
                                                   ਲੱਦੋ_hi(A + lda * (ii + i) + l),
                                                   ਲੱਦੋ_hi(B + ldb * (jj + j) + l))),
                                               unhalf(A[lda * (ii + i) + l].d) *
                                               unhalf(B[ldb * (jj + j) + l].d));
            ਲਈ (int64_t j = 0; j < RN; ++j)
                ਲਈ (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    ਸਤਰਅੰਦਰ int8x16_t ਲੱਦੋ_lo(ਅਚਲ ਖੰਡ_q8_0 *b) {
        ਵਾਪਸ vld1q_s8(b->qs);
    }

    ਸਤਰਅੰਦਰ int8x16_t ਲੱਦੋ_hi(ਅਚਲ ਖੰਡ_q8_0 *b) {
        ਵਾਪਸ vld1q_s8(b->qs + 16);
    }

    ਸਤਰਅੰਦਰ int8x16_t ਲੱਦੋ_lo(ਅਚਲ ਖੰਡ_q4_0 *b) {
        ਵਾਪਸ vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs),
                                                     vdupq_n_u8(0x0f))),
                        vdupq_n_s8(0x8));
    }

    ਸਤਰਅੰਦਰ int8x16_t ਲੱਦੋ_hi(ਅਚਲ ਖੰਡ_q4_0 *b) {
        ਵਾਪਸ vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)),
                        vdupq_n_s8(0x8));
    }

    ਅਚਲ TA *ਅਚਲ A;
    ਅਚਲ ਖੰਡ_q8_0 *ਅਚਲ B;
    ਦਸ਼ਮਲਵ *ਅਚਲ C;
    ਅਚਲ int64_t k;
    ਅਚਲ int64_t lda;
    ਅਚਲ int64_t ldb;
    ਅਚਲ int64_t ldc;
    ਅਚਲ ਪੂਰਨ ith;
    ਅਚਲ ਪੂਰਨ nth;
};
#ਜੇ_ਅੰਤ // __ARM_FEATURE_DOTPROD

#ਜੇ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC>
ਵਰਗ tinyBLAS_Q0_AVX {
  ਜਨਤਕ:
    tinyBLAS_Q0_AVX(int64_t k,
                    ਅਚਲ TA *A, int64_t lda,
                    ਅਚਲ TB *B, int64_t ldb,
                    TC *C, int64_t ldc,
                    ਪੂਰਨ ith, ਪੂਰਨ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
        ਅਚਲ int8_t kvalues_iq4nl[16] = {
            -127, -104, -83, -65,
            -49,  -35,  -22, -10,
              1,   13,   25,  38,
             53,   69,   89, 113
        };

        iq4nlt = _mm_loadu_si128((ਅਚਲ __m128i *)kvalues_iq4nl);
    }

    ਖਾਲੀ matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  ਨਿਜੀ:
    ਖਾਲੀ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        ਚੋਣ ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
#ਜੇ ਸਦਿਸ਼_REGISTERS == 32
        ਹਾਲਤ 0x44:
            mc = 4;
            nc = 4;
#ਜੇ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<4>(m0, m, n0, n);
#ਨਹੀਂਤਾਂ
            gemm<4, 4>(m0, m, n0, n);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
        ਹਾਲਤ 0x43:
            mc = 4;
            nc = 3;
#ਜੇ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<3>(m0, m, n0, n);
#ਨਹੀਂਤਾਂ
            gemm<4, 3>(m0, m, n0, n);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
        ਹਾਲਤ 0x34:
            mc = 3;
            nc = 4;
#ਜੇ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<3>(m0, m, n0, n);
#ਨਹੀਂਤਾਂ
            gemm<3, 4>(m0, m, n0, n);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
        ਹਾਲਤ 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x42:
            mc = 4;
            nc = 2;
#ਜੇ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#ਨਹੀਂਤਾਂ
            gemm<4, 2>(m0, m, n0, n);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
        ਹਾਲਤ 0x24:
            mc = 2;
            nc = 4;
#ਜੇ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#ਨਹੀਂਤਾਂ
            gemm<2, 4>(m0, m, n0, n);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
#ਨਹੀਂਤਾਂ
        ਹਾਲਤ 0x44:
        ਹਾਲਤ 0x43:
        ਹਾਲਤ 0x42:
            mc = 4;
            nc = 2;
#ਜੇ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#ਨਹੀਂਤਾਂ
            gemm<4, 2>(m0, m, n0, n);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
        ਹਾਲਤ 0x34:
        ਹਾਲਤ 0x24:
            mc = 2;
            nc = 4;
#ਜੇ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#ਨਹੀਂਤਾਂ
            gemm<2, 4>(m0, m, n0, n);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
        ਹਾਲਤ 0x33:
#ਜੇ_ਅੰਤ
        ਹਾਲਤ 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x41:
            mc = 4;
            nc = 1;
#ਜੇ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<1>(m0, m, n0, n);
#ਨਹੀਂਤਾਂ
            gemm<4, 1>(m0, m, n0, n);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
        ਹਾਲਤ 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x14:
            mc = 1;
            nc = 4;
#ਜੇ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<1>(m0, m, n0, n);
#ਨਹੀਂਤਾਂ
            gemm<1, 4>(m0, m, n0, n);
#ਜੇ_ਅੰਤ
            ਰੋਕੋ;
        ਹਾਲਤ 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            ਰੋਕੋ;
        ਹਾਲਤ 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

#ਜੇ defined(__AVX2__) && defined(__F16C__)
// Templated functions for gemm of dimensions 4xN
    ਸਾਂਚਾ <ਪੂਰਨ RN>
    NOINLINE ਖਾਲੀ gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / 4;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * 4;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            __m256 Cv[RN][4] = {};
            ਲਈ (int64_t l = 0; l < k; ++l) {
                uint64_t a_delta = ((uint64_t)A[lda * (ii + 3) + l].d << 48) | ((uint64_t)A[lda * (ii + 2) + l].d << 32) | ((uint64_t)A[lda * (ii + 1) + l].d << 16) | (A[lda * (ii + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 da = _mm_cvtph_ps(mm_ਸੈੱਟ_epi64x(0, a_delta));
                __m256i avec0 = load(A + lda * (ii + 0) + l);
                __m256i avec1 = load(A + lda * (ii + 1) + l);
                __m256i avec2 = load(A + lda * (ii + 2) + l);
                __m256i avec3 = load(A + lda * (ii + 3) + l);
                ਲਈ (int64_t j = 0; j < RN; ++j) {
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

            ਲਈ (int64_t j = 0; j < RN; ++j)
                ਲਈ (int64_t i = 0; i < 4; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    // Templated functions for gemm of dimensions Mx4
    ਸਾਂਚਾ <ਪੂਰਨ RM>
    NOINLINE ਖਾਲੀ gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / 4;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * 4;
            __m256 Cv[4][RM] = {};
            ਲਈ (int64_t l = 0; l < k; ++l) {
                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 3) + l].d << 48) | ((uint64_t)B[ldb * (jj + 2) + l].d << 32) | ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 db = _mm_cvtph_ps(mm_ਸੈੱਟ_epi64x(0, b_delta));
                __m256i bvec0 = load(B + ldb * (jj + 0) + l);
                __m256i bvec1 = load(B + ldb * (jj + 1) + l);
                __m256i bvec2 = load(B + ldb * (jj + 2) + l);
                __m256i bvec3 = load(B + ldb * (jj + 3) + l);
                ਲਈ (int64_t i = 0; i < RM; ++i) {
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
            ਲਈ (int64_t j = 0; j < 4; ++j)
                ਲਈ (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }
#ਜੇ_ਅੰਤ

    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN>
    NOINLINE ਖਾਲੀ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            ਲਈ (int64_t l = 0; l < k; ++l)
                ਲਈ (int64_t j = 0; j < RN; ++j)
                    ਲਈ (int64_t i = 0; i < RM; ++i) {
#ਜੇ defined(__AVX2__)
                        __m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                              load(A + lda * (ii + i) + l)),
                                             _mm256_sign_epi8(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l)));
#ਨਹੀਂਤਾਂ
                        __m128i ali0 = load0(A + lda * (ii + i) + l);
                        __m128i ali1 = load1(A + lda * (ii + i) + l);
                        __m128i blj0 = load0(B + ldb * (jj + j) + l);
                        __m128i blj1 = load1(B + ldb * (jj + j) + l);

                        __m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
                        __m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
                        __m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
                        __m128i sepBA1 = _mm_sign_epi8(blj1, ali1);

                        // updot
                        ਅਚਲ __m128i oneFill = _mm_set1_epi16(1);
                        __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0);
                        __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1);
                        __m256 udTmp = _mm256_cvtepi32_ps(MM256_ਸੈੱਟ_M128_I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
#ਜੇ_ਅੰਤ
                        Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            ਲਈ (int64_t j = 0; j < RN; ++j)
                ਲਈ (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    ਸਤਰਅੰਦਰ __m256i load(ਅਚਲ ਖੰਡ_q8_0 *b) {
        ਵਾਪਸ _mm256_loadu_si256((ਅਚਲ __m256i *)b->qs);
    }

    ਸਤਰਅੰਦਰ __m128i load0(ਅਚਲ ਖੰਡ_q8_0 *b) {
        ਵਾਪਸ _mm_loadu_si128((ਅਚਲ __m128i *)b->qs);
    }

    ਸਤਰਅੰਦਰ __m128i load1(ਅਚਲ ਖੰਡ_q8_0 *b) {
        ਵਾਪਸ _mm_loadu_si128(((ਅਚਲ __m128i *)b->qs) + 1);
    }

    ਸਤਰਅੰਦਰ __m256i load(ਅਚਲ ਖੰਡ_q4_0 *b) {
        ਵਾਪਸ _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    ਸਤਰਅੰਦਰ __m128i load0(ਅਚਲ ਖੰਡ_q4_0 *b) {
        ਅਚਲ __m128i x = _mm_loadu_si128((ਅਚਲ __m128i *)(b->qs));
        ਵਾਪਸ _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), x), _mm_set1_epi8(8));
    }

    ਸਤਰਅੰਦਰ __m128i load1(ਅਚਲ ਖੰਡ_q4_0 *b) {
        ਅਚਲ __m128i x = _mm_loadu_si128((ਅਚਲ __m128i *)(b->qs));
        ਵਾਪਸ _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)), _mm_set1_epi8(8));
    }

    ਸਤਰਅੰਦਰ __m256i load(ਅਚਲ ਖੰਡ_q5_0 *b) {
        ਵਾਪਸ _mm256_or_si256(denibble(b->qs), bittobyte(b->qh));
    }

    ਸਤਰਅੰਦਰ __m128i load0(ਅਚਲ ਖੰਡ_q5_0* b) {
        ਅਚਲ __m128i x = _mm_loadu_si128((ਅਚਲ __m128i *)(b->qs));
        uint32_t x32;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&x32, b->qh, ਆਕਾਰ(uint32_t));
        __m128i qxl = _mm_and_si128(_mm_set1_epi8(15), x);
        __m128i bytesl = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      mm_ਸੈੱਟ_epi64x(0x0101010101010101, 0x0000000000000000))));
        bytesl = _mm_andnot_si128(bytesl, _mm_set1_epi8((ਅੱਖਰ)0xF0));
        ਵਾਪਸ _mm_or_si128(qxl, bytesl);
    }

    ਸਤਰਅੰਦਰ __m128i load1(ਅਚਲ ਖੰਡ_q5_0* b) {
        ਅਚਲ __m128i x = _mm_loadu_si128((ਅਚਲ __m128i *)(b->qs));
        uint32_t x32;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&x32, b->qh, ਆਕਾਰ(uint32_t));
        __m128i qxh = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
        __m128i bytesh = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      mm_ਸੈੱਟ_epi64x(0x0303030303030303, 0x0202020202020202))));
        bytesh = _mm_andnot_si128(bytesh, _mm_set1_epi8((ਅੱਖਰ)0xF0));
        ਵਾਪਸ _mm_or_si128(qxh, bytesh);
    }

    ਸਤਰਅੰਦਰ __m256i load(ਅਚਲ ਖੰਡ_iq4_nl *b) {
        ਵਾਪਸ MM256_ਸੈੱਟ_M128_I(load1(b), load0(b));
    }

    ਸਤਰਅੰਦਰ __m128i load0(ਅਚਲ ਖੰਡ_iq4_nl *b) {
        ਅਚਲ __m128i x = _mm_loadu_si128((ਅਚਲ __m128i *)(b->qs));
        ਵਾਪਸ _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), x));
    }

    ਸਤਰਅੰਦਰ __m128i load1(ਅਚਲ ਖੰਡ_iq4_nl *b) {
        ਅਚਲ __m128i x = _mm_loadu_si128((ਅਚਲ __m128i *)(b->qs));
        ਵਾਪਸ _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)));
    }

    ਸਤਰਅੰਦਰ __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#ਜੇ defined(__AVX512VNNI__) && defined(__AVX512VL__)
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#ਜਾਂਜੇ defined(__AVXVNNI__)
        res = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
#ਨਹੀਂਤਾਂ
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#ਜੇ_ਅੰਤ
        ਵਾਪਸ _mm256_cvtepi32_ps(res);
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ __m256i denibble(ਅਚਲ uint8_t *p) {
        __m128i x = _mm_loadu_si128((ਅਚਲ __m128i *)p);
        ਵਾਪਸ _mm256_and_si256(_mm256_set1_epi8(15),
                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                        _mm_srli_epi16(x, 4), 1));
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ __m256i bittobyte(ਅਚਲ uint8_t *p) {
        uint32_t x32;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&x32, p, ਆਕਾਰ(uint32_t));
        __m256i ਬਾਈਟਾਂ = _mm256_cmpeq_epi8(_mm256_set1_epi64x(-1),
                                          _mm256_or_si256(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                          _mm256_shuffle_epi8(_mm256_set1_epi32(x32),
                                                                              mm256_ਸੈੱਟ_epi64x(0x0303030303030303, 0x0202020202020202,
                                                                                                0x0101010101010101, 0x0000000000000000))));
        ਵਾਪਸ _mm256_andnot_si256(ਬਾਈਟਾਂ, _mm256_set1_epi8((ਅੱਖਰ)0xF0));
    }

    ਅਚਲ TA *ਅਚਲ A;
    ਅਚਲ TB *ਅਚਲ B;
    TC *ਅਚਲ C;
    ਅਚਲ int64_t k;
    ਅਚਲ int64_t lda;
    ਅਚਲ int64_t ldb;
    ਅਚਲ int64_t ldc;
    ਅਚਲ ਪੂਰਨ ith;
    ਅਚਲ ਪੂਰਨ nth;
    __m128i iq4nlt;
};
#ਜੇ_ਅੰਤ // __AVX__

//PPC Implementation
#ਜੇ defined(__MMA__)

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਬਣਤਰ mma_instr;

ਸਾਂਚਾ<>
ਬਣਤਰ mma_instr<ggml_bf16_t> {
    ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ outer_ਗੁਣਨਫਲ(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvbf16ger2pp(acc, a, b);
    }
};

ਸਾਂਚਾ<>
ਬਣਤਰ mma_instr<ggml_fp16_t> {
    ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ outer_ਗੁਣਨਫਲ(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvf16ger2pp(acc, a, b);
    }
};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA, ਕਿਸਮ_ਨਾਂ TB, ਕਿਸਮ_ਨਾਂ TC>
ਵਰਗ tinyBLAS_HP16_PPC {
  ਜਨਤਕ:
    tinyBLAS_HP16_PPC(int64_t k,
                ਅਚਲ TA *A, int64_t lda,
                ਅਚਲ TB *B, int64_t ldb,
                TC *C, int64_t ldc,
                ਪੂਰਨ ith, ਪੂਰਨ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    ਖਾਲੀ matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #ਜੇ defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #ਜੇ_ਅੰਤ
        ਜੇ (k < kc) {
            kc = k;
        }
        ਤਾਰਕਿਕ can_use_tiled = (m % mc == 0) && (n % nc == 0) && (k % kc == 0);
        ਜੇ (can_use_tiled) {
            matmul_tiled(m, n, mc, nc, kc);
        } ਨਹੀਂਤਾਂ {
            mnpack(0, m, 0, n);
        }
    }

  ਨਿਜੀ:
    __attribute__((always_inline))
    ਸਤਰਅੰਦਰ ਖਾਲੀ save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ਲਈ (ਪੂਰਨ I = 0; I < 4; I++) {
            ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                *((ਦਸ਼ਮਲਵ *)(C+ii+((jj+J)*ldc)+I)) = *((ਦਸ਼ਮਲਵ *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    ਸਤਰਅੰਦਰ ਖਾਲੀ ਜੋੜੋ_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ਲਈ (ਪੂਰਨ I = 0; I < 4; I++) {
            ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                ਦਸ਼ਮਲਵ * c_ptr = (ਦਸ਼ਮਲਵ *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((ਦਸ਼ਮਲਵ *)&vec_C[I]+J);
            }
        }
    }

    ਖਾਲੀ ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store(vec_t *c, ਪੂਰਨ ਅੰਕ_Vec, ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *vec_ਵਿੱਥ) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        ਜੇ (ਅੰਕ_Vec == 2) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[2], c[3], swiz1);
            s[0] = vec_perm(t[0], t[1], swiz3);
            s[1] = vec_perm(t[0], t[1], swiz4);
            vec_xst(s[0], 0, (vec_t*)vec_ਵਿੱਥ);
            vec_xst(s[1], 0, (vec_t*)(vec_ਵਿੱਥ + 16));
        } ਨਹੀਂਤਾਂ ਜੇ (ਅੰਕ_Vec == 4) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[0], c[1], swiz2);
            t[2] = vec_perm(c[2], c[3], swiz1);
            t[3] = vec_perm(c[2], c[3], swiz2);
            s[0] = vec_perm(t[0], t[2], swiz3);
            s[1] = vec_perm(t[0], t[2], swiz4);
            s[2] = vec_perm(t[1], t[3], swiz3);
            s[3] = vec_perm(t[1], t[3], swiz4);
            ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i)
                vec_xst(s[i], 0, (vec_t*)(vec_ਵਿੱਥ + i * 16));
        } ਨਹੀਂਤਾਂ ਜੇ (ਅੰਕ_Vec == 8) {
            ਲਈ (ਪੂਰਨ i = 0; i < 4; i += 2) {
                t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
                t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
            }
            ਲਈ (ਪੂਰਨ i = 4; i < 8; i += 2) {
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
            ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i)
                vec_xst(s[i], 0, (vec_t*)(vec_ਵਿੱਥ + i * 16));
        }
    }

    ਖਾਲੀ pack_ਸਾਧਾਰਨ(ਅਚਲ TA* a, int64_t lda, ਪੂਰਨ ਪੰਕਤੀਆਂ, ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_2, ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ* vec) {
        int64_t i, j;
        TA *aoffset = ਨਿਰਾ;
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *vec_ਵਿੱਥ = ਨਿਰਾ;
        TA * aoffsets[8];
        vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ c_arr[8];
        aoffset = const_cast<TA*>(a);
        vec_ਵਿੱਥ = vec;
        j = (ਪੰਕਤੀਆਂ >> 3);
        ਜੇ (j > 0) {
            ਕਰੋ {
                aoffsets[0] = aoffset;
                ਜੇ (ਖਾਨੇ_ਗਿਣਤੀ_2 == 4) {
                    aoffsets[0] = aoffset;
                    ਲਈ (ਪੂਰਨ it = 1; it < 4; ++it)
                        aoffsets[it] = aoffsets[it-1] + lda;
                    aoffset += 4 * lda;
                    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i)
                        c_arr[i] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[i]);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store(c_arr, 4, vec_ਵਿੱਥ);
                    ਲਈ (ਪੂਰਨ i = 0; i<4; i++)
                        aoffsets[i] = aoffsets[i]+lda;
                    vec_ਵਿੱਥ +=64;
                }
                i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 3);
                ਜੇ (i > 0) {
                    ਲਈ (ਪੂਰਨ it = 1; it < 8; ++it) {
                        aoffsets[it] = aoffsets[it-1] + lda;
                    }
                    aoffset += 8 * lda;

                    ਕਰੋ {
                        ਲਈ (ਪੂਰਨ it = 0; it < 8; ++it)
                            c_arr[it] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[it]);
                        ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store(c_arr, 8, vec_ਵਿੱਥ);
                        ਲਈ (ਪੂਰਨ it = 0; it < 8; ++it)
                            aoffsets[it] = aoffsets[it] + 8;
                        vec_ਵਿੱਥ += 128;
                        i--;
                    } ਜਦੋਂ(i > 0);
                }
                j--;
            } ਜਦੋਂ(j > 0);
        }
        ਜੇ (ਪੰਕਤੀਆਂ & 4) {
            aoffsets[0] = aoffset;
            ਲਈ (ਪੂਰਨ it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            ਜੇ (ਖਾਨੇ_ਗਿਣਤੀ_2 == 4) {
                ਲਈ (ਪੂਰਨ it = 0; it < 4; ++it)
                    c_arr[it] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[it]);
                ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store(c_arr, 2, vec_ਵਿੱਥ);
                ਲਈ (ਪੂਰਨ it = 0; it< 4; it++)
                    aoffsets[it] = aoffsets[it] + lda;
                vec_ਵਿੱਥ += 32;
            }
            i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 3);
            ਜੇ (i > 0) {
                ਕਰੋ {
                    ਲਈ (ਪੂਰਨ it = 0; it < 4; ++it)
                        c_arr[it] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[it]);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store(c_arr, 4, vec_ਵਿੱਥ);
                    ਲਈ (ਪੂਰਨ it = 0; it< 4; it++)
                        aoffsets[it] = aoffsets[it] + 8*lda;
                    vec_ਵਿੱਥ += 64;
                    i--;
                } ਜਦੋਂ(i > 0);
            }
        }
        ਜੇ (ਪੰਕਤੀਆਂ & 3) {
            aoffsets[0] = aoffset;
            ਲਈ (ਪੂਰਨ it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            ਜੇ (ਖਾਨੇ_ਗਿਣਤੀ_2 == 4) {
                ਚੋਣ(ਪੰਕਤੀਆਂ) {
                    ਹਾਲਤ 3: c_arr[2] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[2]);
                    ਹਾਲਤ 2: c_arr[1] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[1]);
                    ਹਾਲਤ 1: c_arr[0] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[0]);
                        ਰੋਕੋ;
                }
                ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store(c_arr, 2, vec_ਵਿੱਥ);
                ਲਈ (ਪੂਰਨ it = 0; it< 4; it++)
                     aoffsets[it] = aoffsets[it] + lda;
                vec_ਵਿੱਥ += 32;
            }
            i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 3);
            ਜੇ (i > 0) {
                ਕਰੋ {
                    ਚੋਣ(ਪੰਕਤੀਆਂ) {
                        ਹਾਲਤ 3: c_arr[2] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[2]);
                        ਹਾਲਤ 2: c_arr[1] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[1]);
                        ਹਾਲਤ 1: c_arr[0] = vec_xl(0, (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ*)aoffsets[0]);
                            ਰੋਕੋ;
                    }
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store(c_arr, 4, vec_ਵਿੱਥ);
                    ਲਈ (ਪੂਰਨ it = 0; it <4; it++)
                         aoffsets[it] = aoffsets[it] + 8* lda;
                    vec_ਵਿੱਥ += 64;
                    i--;
                } ਜਦੋਂ(i > 0);
            }
        }
    }

    ਖਾਲੀ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        ਪੂਰਨ m_rem = MIN(m - m0, 8);
        ਪੂਰਨ n_rem = MIN(n - n0, 8);

        ਜੇ (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4,8>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ (m_rem >=8 && n_rem >=4){
                mc = 8;
                nc = 4;
                gemm<8,4>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ ((m_rem < 4) && (n_rem >= 8)) {
            nc = 8;
            ਚੋਣ(m_rem) {
                ਹਾਲਤ 1:
                    mc = 1;
                    gemm_Mx8<1>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 2:
                    mc = 2;
                    gemm_Mx8<2>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 3:
                    mc = 3;
                    gemm_Mx8<3>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ;
            }
        } ਨਹੀਂਤਾਂ ਜੇ (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small<4, 4>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ ((m_rem > 4) && (n_rem < 4)) {
            mc = 4;
            ਚੋਣ(n_rem) {
                ਹਾਲਤ 1:
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 2:
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 3:
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    ਰੋਕੋ;

                ਮੂਲ:
                    ਵਾਪਸ;
            }
        } ਨਹੀਂਤਾਂ {
            ਚੋਣ((m_rem << 4) | n_rem) {
                ਹਾਲਤ 0x43:
                    mc = 4;
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x42:
                    mc = 4;
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x41:
                    mc = 4;
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x34:
                    mc = 3;
                    nc = 4;
                    gemm_small<3, 4>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x33:
                    mc = 3;
                    nc = 3;
                    gemm_small<3, 3>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x32:
                    mc = 3;
                    nc = 2;
                    gemm_small<3, 2>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x31:
                    mc = 3;
                    nc = 1;
                    gemm_small<3, 1>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x24:
                    mc = 2;
                    nc = 4;
                    gemm_small<2,4>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x23:
                    mc = 2;
                    nc = 3;
                    gemm_small<2, 3>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x22:
                    mc = 2;
                    nc = 2;
                    gemm_small<2, 2>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x21:
                    mc = 2;
                    nc = 1;
                    gemm_small<2, 1>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x14:
                    mc = 1;
                    nc = 4;
                    gemm_small<1, 4>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x13:
                    mc = 1;
                    nc = 3;
                    gemm_small<1, 3>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x12:
                    mc = 1;
                    nc = 2;
                    gemm_small<1, 2>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਹਾਲਤ 0x11:
                    mc = 1;
                    nc = 1;
                    gemm_small<1, 1>(m0, m, n0, n);
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ;
            }
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    ਖਾਲੀ KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        ਲਈ (ਪੂਰਨ l = 0; l < k; l+=8) {
            pack_ਸਾਧਾਰਨ((A+(ii*lda)+l), lda, 4, 8, (uint8_t*)vec_A);
            pack_ਸਾਧਾਰਨ((B+(jj*ldb)+l), ldb, 8, 8, (uint8_t*)vec_B);
            ਲਈ (ਪੂਰਨ x = 0; x < 4; x++) {
                mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_1, vec_A[x], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
    }

    ਖਾਲੀ KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        ਲਈ (ਪੂਰਨ l = 0; l < k; l+=8) {
            pack_ਸਾਧਾਰਨ((A+(ii*lda)+l), lda, 8, 8, (uint8_t*)vec_A);
            pack_ਸਾਧਾਰਨ((B+(jj*ldb)+l), ldb, 8, 4, (uint8_t*)vec_B);
            ਲਈ (ਪੂਰਨ x = 0; x < 4; x++) {
                mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_1, vec_A[x+4], vec_B[x]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii+4, jj);
    }


    ਖਾਲੀ KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        ਲਈ (ਪੂਰਨ l = 0; l < k; l+=8) {
            pack_ਸਾਧਾਰਨ(A+(ii*lda)+l, lda, 8, 8, (uint8_t*)vec_A);
            pack_ਸਾਧਾਰਨ(B+(jj*ldb)+l, ldb, 8, 8, (uint8_t*)vec_B);
            ਲਈ (ਪੂਰਨ x = 0; x < 4; x++) {
                mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_1, vec_A[x], vec_B[x+4]);
                mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_2, vec_A[x+4], vec_B[x]);
                mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_3, vec_A[x+4], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
        save_acc(&acc_2, ii+4, jj);
        save_acc(&acc_3, ii+4, jj+4);
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        ਲਈ (ਪੂਰਨ x = 0; x < 4; x ++) {
             mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc[0], vec_A0[x], vec_B[x]);
             mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc[1], vec_A0[x], vec_B[x+4]);
             mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc[2], vec_A0[x+4], vec_B[x]);
             mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc[3], vec_A0[x+4], vec_B[x+4]);
             mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc[4], vec_A1[x], vec_B[x]);
             mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc[5], vec_A1[x], vec_B[x+4]);
             mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc[6], vec_A1[x+4], vec_B[x]);
             mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc[7], vec_A1[x+4], vec_B[x+4]);
        }
    }
    ਖਾਲੀ KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        ਲਈ (int64_t i = 0; i < mc; i += 16) {
            ਪੂਰਨ A_ਅਧਾਰ_addr = (mc / 8) * (i / 8) * 8;
            ਲਈ (int64_t j = 0; j < nc; j += 8) {
                 ਪੂਰਨ B_ਅਧਾਰ_addr = (nc / 8) * (j / 8) * 8;
                 acc_t acc[8];
                 vec_t A0_ਖੰਡ[8]; vec_t A1_ਖੰਡ[8];
                 ਲਈ (ਪੂਰਨ x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 ਲਈ (int64_t l = 0; l < kc; l += 8) {
                     ਪੂਰਨ A0_ਖੰਡ_idx = A_ਅਧਾਰ_addr + (l / 8) * 8;
                     ਪੂਰਨ A1_ਖੰਡ_idx = A0_ਖੰਡ_idx + (mc / 8) * 8;
                     ਪੂਰਨ B_ਖੰਡ_idx = B_ਅਧਾਰ_addr + (l / 8) * 8;
                     vec_t* A0_ਖੰਡ = &vec_A[A0_ਖੰਡ_idx];
                     vec_t* A1_ਖੰਡ = &vec_A[A1_ਖੰਡ_idx];
                     vec_t* B_ਖੰਡ = &vec_B[B_ਖੰਡ_idx];
                     MMA_16x8(A0_ਖੰਡ, A1_ਖੰਡ, B_ਖੰਡ, acc);
                 }
                 ਜੇ (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } ਨਹੀਂਤਾਂ {
                     ਜੋੜੋ_save_acc(&acc[0], ii + i, jj + j);
                     ਜੋੜੋ_save_acc(&acc[1], ii + i, jj + j + 4);
                     ਜੋੜੋ_save_acc(&acc[2], ii + i + 4, jj + j);
                     ਜੋੜੋ_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     ਜੋੜੋ_save_acc(&acc[4], ii + i + 8, jj + j);
                     ਜੋੜੋ_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     ਜੋੜੋ_save_acc(&acc[6], ii + i + 12, jj + j);
                     ਜੋੜੋ_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }
    ਸਾਂਚਾ<ਪੂਰਨ RM, ਪੂਰਨ RN>
    ਖਾਲੀ gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[2], vec_B[2];
            ਲਈ (ਪੂਰਨ l=0; l<k; l+=4) {
                pack_ਸਾਧਾਰਨ(A+(ii*lda)+l, lda, RM, 4, (uint8_t*)vec_A);
                pack_ਸਾਧਾਰਨ(B+(jj*ldb)+l, ldb, RN, 4, (uint8_t*)vec_B);
                ਲਈ (ਪੂਰਨ x = 0; x<2; x++) {
                    mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_0, vec_A[x], vec_B[x]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            ਲਈ (ਪੂਰਨ I = 0; I < RM; I++) {
                ਲਈ (ਪੂਰਨ J = 0; J < RN; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    ਸਾਂਚਾ<ਪੂਰਨ RM>
    ਖਾਲੀ gemm_Mx8(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        ਪੂਰਨ RN = 8;
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0, acc_1;
            __builtin_mma_xxsetaccz(&acc_0);
            __builtin_mma_xxsetaccz(&acc_1);
            vec_t vec_A[4], vec_B[8];
            ਲਈ (ਪੂਰਨ l=0; l<k; l+=8) {
                pack_ਸਾਧਾਰਨ(A+(ii*lda)+l, lda, RM, 8, (uint8_t*)vec_A);
                pack_ਸਾਧਾਰਨ(B+(jj*ldb)+l, ldb, RN, 8, (uint8_t*)vec_B);
                ਲਈ (ਪੂਰਨ x = 0; x<4; x++) {
                    mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_0, vec_A[x], vec_B[x]);
                    mma_instr<TA>::outer_ਗੁਣਨਫਲ(&acc_1, vec_A[x], vec_B[x+4]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            ਲਈ (ਪੂਰਨ I = 0; I < RM; I++) {
                ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_1);
            ਲਈ (ਪੂਰਨ I = 0; I < RM; I++) {
                ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+4+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    ਸਾਂਚਾ<ਪੂਰਨ RM, ਪੂਰਨ RN>
    ਸਤਰਅੰਦਰ ਖਾਲੀ kernel(int64_t ii, int64_t jj) {
       ਜੇ constexpr(RM == 4 && RN == 8) {
          KERNEL_4x8(ii,jj);
       } ਨਹੀਂਤਾਂ ਜੇ constexpr(RM == 8 && RN == 8) {
          KERNEL_8x8(ii,jj);
       } ਨਹੀਂਤਾਂ ਜੇ constexpr(RM == 8 && RN == 4) {
          KERNEL_8x4(ii,jj);
       } ਨਹੀਂਤਾਂ {
          ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ && "RN/RM values not supported");
       }
    }

    ਖਾਲੀ matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles) {
            end = tiles;
        }
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = (ਕੰਮ / xtiles) * mc;
            int64_t jj = (ਕੰਮ % xtiles) * nc;
            ਲਈ (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 8];
                 vec_t B_pack[kc * nc / 8];
                 pack_ਸਾਧਾਰਨ(A + (ii * lda) + kk, lda, kc, mc, (uint8_t *)A_pack);
                 pack_ਸਾਧਾਰਨ(B + (jj * ldb) + kk, ldb, kc, nc, (uint8_t *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN>
    NOINLINE ਖਾਲੀ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    ਅਚਲ TA *ਅਚਲ A;
    ਅਚਲ TB *ਅਚਲ B;
    TC *C;
    ਅਚਲ int64_t k;
    ਅਚਲ int64_t lda;
    ਅਚਲ int64_t ldb;
    ਅਚਲ int64_t ldc;
    ਅਚਲ ਪੂਰਨ ith;
    ਅਚਲ ਪੂਰਨ nth;
};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ TA>
ਵਰਗ tinyBLAS_Q0_PPC {
  ਜਨਤਕ:
    tinyBLAS_Q0_PPC(int64_t k,
             ਅਚਲ TA * A, int64_t lda,
             ਅਚਲ ਖੰਡ_q8_0 * B, int64_t ldb,
             ਦਸ਼ਮਲਵ * C, int64_t ldc,
             ਪੂਰਨ ith, ਪੂਰਨ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    ਖਾਲੀ matmul(int64_t m, int64_t n) {
        int64_t mc = 64;
        int64_t nc = 64;
        int64_t kc = 64;
        int64_t n_chunk = 64;
    #ਜੇ defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 32;
        nc = 32;
        kc = 32;
        n_chunk = 32;
    #ਜੇ_ਅੰਤ
        int64_t n_aligned = 0;
        ਜੇ (n % n_chunk == 0) {
            n_aligned = n;
        } ਨਹੀਂਤਾਂ ਜੇ (n == 4) {
            n_aligned = 4;
        } ਨਹੀਂਤਾਂ ਜੇ (n < n_chunk) {
            n_aligned = (n / 8) * 8;
        } ਨਹੀਂਤਾਂ {
            n_aligned = (n / n_chunk) * n_chunk;
        }
        ਜੇ (n_aligned > 0) {
            ਜੇ (n_aligned % n_chunk == 0) nc = n_chunk;
            ਨਹੀਂਤਾਂ ਜੇ (n_aligned == n)      nc = n;
            ਨਹੀਂਤਾਂ ਜੇ (n_aligned % 32 == 0) nc = 32;
            ਨਹੀਂਤਾਂ ਜੇ (n_aligned % 24 == 0) nc = 24;
            ਨਹੀਂਤਾਂ ਜੇ (n_aligned % 16 == 0) nc = 16;
            ਨਹੀਂਤਾਂ                          nc = 8;
        }
        ਤਾਰਕਿਕ can_use_tiled = n_aligned > 0 && (m % mc == 0);
        ਜੇ (can_use_tiled) {
            matmul_tiled(m, n_aligned, mc, nc, kc);
            ਜੇ (n > n_aligned) {
                mnpack(0, m, n_aligned, n);
            }
        } ਨਹੀਂਤਾਂ {
            mnpack(0, m, 0, n);
        }
    }

  ਨਿਜੀ:
    ਸਤਰਅੰਦਰ ਖਾਲੀ save_res(ਪੂਰਨ ii, ਪੂਰਨ jj, ਪੂਰਨ idx, vector ਦਸ਼ਮਲਵ * fin_res, ਪੂਰਨ RM = 4, ਪੂਰਨ RN = 4) {
        ਲਈ (ਪੂਰਨ I = 0; I < RM; I++) {
            ਲਈ (ਪੂਰਨ J = 0; J < RN; J++) {
                *((ਦਸ਼ਮਲਵ *)(C + ii + ((jj + J) * ldc) + I)) = *((ਦਸ਼ਮਲਵ *)&fin_res[idx + I] + J);
            }
        }
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ਲਈ (ਪੂਰਨ I = 0; I < 4; I++) {
            ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                *((ਦਸ਼ਮਲਵ *)(C + ii + ((jj + J) * ldc) + I)) = *((ਦਸ਼ਮਲਵ *)&vec_C[I] + J);
            }
        }
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ ਜੋੜੋ_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ਲਈ (ਪੂਰਨ I = 0; I < 4; I++) {
            ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                ਦਸ਼ਮਲਵ * c_ptr = (ਦਸ਼ਮਲਵ *)(C + ii+ ((jj + J) * ldc) + I);
                *c_ptr += *((ਦਸ਼ਮਲਵ *)&vec_C[I] + J);
            }
        }
    }

    ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ ArrayType>
    ਸਤਰਅੰਦਰ ਖਾਲੀ compute(acc_t * ACC, ਪੂਰਨ c_idx, ਪੂਰਨ s_idx, ArrayType & comparray, vector ਦਸ਼ਮਲਵ * vs, vector ਦਸ਼ਮਲਵ * fin_res) {
        vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vec_C[4];
        vector ਦਸ਼ਮਲਵ CA[4] = {0};
        vector ਦਸ਼ਮਲਵ res[4] = {0};
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            CA[i] = vec_splats((ਦਸ਼ਮਲਵ)(((ਦੁੱਗਣਾ)comparray[c_idx + i]) * -128.0));
            res[i] = vec_ਜੋੜੋ(vec_ctf(vec_C[i], 0), CA[i]);
            fin_res[s_idx + i] = vec_madd(res[i], vs[s_idx + i], fin_res[s_idx + i]);
        }
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ process_q4_elements(vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ (&c)[2], ਪੂਰਨ * ca) {
        ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
        ਅਚਲ vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);
        ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ v8 = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0x8);
        vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsum = {0};
        vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsum2 = {0};
        c[0] = vec_and(c[1], ਨੀਵਾਂ_ਨਕਾਬ);
        c[1] = vec_sr(c[1], v4);
        c[0] = vec_sub(c[0], v8);
        c[1] = vec_sub(c[1], v8);
        vsum = vec_sum4s(c[0], vsum);
        vsum2 = vec_sum4s(c[1], vsum2);
        vsum = vec_ਜੋੜੋ(vsum, vsum2);
        *(ca) = vsum[0] + vsum[1] + vsum[2] + vsum[3];
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ V1, ਕਿਸਮ_ਨਾਂ V2>
    ਸਤਰਅੰਦਰ ਖਾਲੀ ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store(V2 & s1, V2 & s2, V2 & s3, V2 & s4, V1 * vec_ਵਿੱਥ, ਤਾਰਕਿਕ flip) {
        vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ swiz1 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ swiz2 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ swiz3 = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
        vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ swiz4 = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
        V2 t1, t2, t3, t4, t5, t6, t7, t8;
        vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ xor_ਸਦਿਸ਼;
        uint8_t flip_vec = 0x80;
        xor_ਸਦਿਸ਼ = vec_splats(flip_vec);
        t1 = vec_perm(s1, s2, swiz1);
        t2 = vec_perm(s1, s2, swiz2);
        t3 = vec_perm(s3, s4, swiz1);
        t4 = vec_perm(s3, s4, swiz2);
        t5 = vec_perm(t1, t3, swiz3);
        t6 = vec_perm(t1, t3, swiz4);
        t7 = vec_perm(t2, t4, swiz3);
        t8 = vec_perm(t2, t4, swiz4);
        ਜੇ (flip == ਸੱਚ) {
            t5 = vec_xor(t5, xor_ਸਦਿਸ਼);
            t6 = vec_xor(t6, xor_ਸਦਿਸ਼);
            t7 = vec_xor(t7, xor_ਸਦਿਸ਼);
            t8 = vec_xor(t8, xor_ਸਦਿਸ਼);
        }
        vec_xst(t5, 0, vec_ਵਿੱਥ);
        vec_xst(t6, 0, vec_ਵਿੱਥ + 16);
        vec_xst(t7, 0, vec_ਵਿੱਥ + 32);
        vec_xst(t8, 0, vec_ਵਿੱਥ + 48);
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ unpack_q4_ਮੰਜ਼ਿਲ_q8(vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ packed, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ & lo, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ & hi) {
        ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0x0F);
        ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ v8      = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0x08);
        ਅਚਲ vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4    = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)4);
        lo = vec_and(packed, ਨੀਵਾਂ_ਨਕਾਬ);
        hi = vec_sr(packed, v4);
        lo = vec_sub(lo, v8);
        hi = vec_sub(hi, v8);
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_fp16(vec_t * c, ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * vec_ਵਿੱਥ) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i += 2) {
            t[i + 0] = vec_perm(c[i + 0], c[i + 1], swiz1);
            t[i + 1] = vec_perm(c[i + 0], c[i + 1], swiz2);
        }
        ਲਈ (ਪੂਰਨ i = 4; i < 8; i += 2) {
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
        ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
            vec_xst(s[i], 0, (vec_t *)(vec_ਵਿੱਥ + i * 16));
        }
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ convert_and_scale_q8(vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ raw, vector ਦਸ਼ਮਲਵ v_scale, vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ & ਬਾਹਰਲਾ_hi, vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ & ਬਾਹਰਲਾ_lo) {
        vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ i16_hi = vec_unpackh(raw);
        vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ i16_lo = vec_unpackl(raw);

        vector ਦਸ਼ਮਲਵ f_hi_h = vec_ctf(vec_unpackh(i16_hi), 0);
        vector ਦਸ਼ਮਲਵ f_hi_l = vec_ctf(vec_unpackl(i16_hi), 0);
        vector ਦਸ਼ਮਲਵ f_lo_h = vec_ctf(vec_unpackh(i16_lo), 0);
        vector ਦਸ਼ਮਲਵ f_lo_l = vec_ctf(vec_unpackl(i16_lo), 0);
        ਬਾਹਰਲਾ_hi = vec_pack_ਮੰਜ਼ਿਲ_short_fp32(vec_mul(f_hi_h, v_scale), vec_mul(f_hi_l, v_scale));
        ਬਾਹਰਲਾ_lo = vec_pack_ਮੰਜ਼ਿਲ_short_fp32(vec_mul(f_lo_h, v_scale), vec_mul(f_lo_l, v_scale));
    }

    ਖਾਲੀ pack_ਸਾਧਾਰਨ_q4_fp16(ਅਚਲ ਖੰਡ_q4_0 * a, int64_t lda, ਪੂਰਨ ਪੰਕਤੀਆਂ, ਪੂਰਨ ਖੰਡਾਂ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * vec) {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * vec_ਵਿੱਥ = vec;
        ਲਈ (ਪੂਰਨ i = 0; i < ਪੰਕਤੀਆਂ; i += 8) {
            ਅਚਲ ਖੰਡ_q4_0 * ਪੰਕਤੀਆਂ_ਅਧਾਰ[8];
            ਲਈ (ਪੂਰਨ r = 0; r < 8; r++) {
                ਪੰਕਤੀਆਂ_ਅਧਾਰ[r] = a + (i + r) * lda;
            }
            ਲਈ (ਪੂਰਨ blk = 0; blk < ਖੰਡਾਂ; blk++) {
                vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ hp_res[8][4];
                ਲਈ (ਪੂਰਨ r = 0; r < 8; r++) {
                    ਅਚਲ ਖੰਡ_q4_0 * current_blk = ਪੰਕਤੀਆਂ_ਅਧਾਰ[r] + blk;
                    vector ਦਸ਼ਮਲਵ v_scale = vec_ਕੱਢੋ_fp32_from_shorth(vec_splats(current_blk->d));
                    vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ v_qs = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)current_blk->qs);
                    vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ c1, c2;
                    unpack_q4_ਮੰਜ਼ਿਲ_q8(v_qs, c1, c2);
                    convert_and_scale_q8(c1, v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c2, v_scale, hp_res[r][2], hp_res[r][3]);
                }
                ਲਈ (ਪੂਰਨ c = 0; c < 4; c++) {
                    vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ c_arr[8];
                    ਲਈ (ਪੂਰਨ r = 0; r < 8; r++) {
                        c_arr[r] = (vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)hp_res[r][c];
                    }
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_fp16((vec_t *)c_arr, vec_ਵਿੱਥ);
                    vec_ਵਿੱਥ += 128;
                }
            }
        }
    }

    ਸਾਂਚਾ <ਪੂਰਨ chunk_ਮਾਪ>
    ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ pack_q8_ਖੰਡ(ਅਚਲ ਖੰਡ_q8_0 * a, int64_t lda, ਪੂਰਨ ਪੰਕਤੀਆਂ, ਪੂਰਨ ਖੰਡਾਂ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * vec) {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * vec_ਵਿੱਥ = vec;
        ਅਚਲ vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        ਅਚਲ vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        ਅਚਲ vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        ਅਚਲ vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        ਲਈ (ਪੂਰਨ i = 0; i < ਪੰਕਤੀਆਂ; i += chunk_ਮਾਪ) {
            ਅਚਲ ਖੰਡ_q8_0 * ਪੰਕਤੀਆਂ_ਅਧਾਰ[chunk_ਮਾਪ];
            ਲਈ (ਪੂਰਨ r = 0; r < chunk_ਮਾਪ; r++) {
                ਪੰਕਤੀਆਂ_ਅਧਾਰ[r] = a + (i + r) * lda;
            }
            ਲਈ (ਪੂਰਨ blk = 0; blk < ਖੰਡਾਂ; blk++) {
                vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ hp_res[chunk_ਮਾਪ][4];
                ਲਈ (ਪੂਰਨ r = 0; r < chunk_ਮਾਪ; r++) {
                    ਅਚਲ ਖੰਡ_q8_0 * b = ਪੰਕਤੀਆਂ_ਅਧਾਰ[r] + blk;
                    vector ਦਸ਼ਮਲਵ v_scale = vec_ਕੱਢੋ_fp32_from_shorth(vec_splats(b->d));
                    vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ c[2];
                    __vector_pair pair = __builtin_vsx_lxvp(0, (__vector_pair *)b->qs);
                    __builtin_vsx_disassemble_pair(c, & pair);
                    convert_and_scale_q8(c[0], v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c[1], v_scale, hp_res[r][2], hp_res[r][3]);
                }
                ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < 4; ਖਾਨਾ++) {
                    ਜੇ constexpr (chunk_ਮਾਪ == 8) {
                        vec_t t[8];
                        t[0] = vec_perm((vec_t)hp_res[0][ਖਾਨਾ], (vec_t)hp_res[1][ਖਾਨਾ], swiz1);
                        t[1] = vec_perm((vec_t)hp_res[0][ਖਾਨਾ], (vec_t)hp_res[1][ਖਾਨਾ], swiz2);
                        t[2] = vec_perm((vec_t)hp_res[2][ਖਾਨਾ], (vec_t)hp_res[3][ਖਾਨਾ], swiz1);
                        t[3] = vec_perm((vec_t)hp_res[2][ਖਾਨਾ], (vec_t)hp_res[3][ਖਾਨਾ], swiz2);
                        t[4] = vec_perm((vec_t)hp_res[4][ਖਾਨਾ], (vec_t)hp_res[5][ਖਾਨਾ], swiz1);
                        t[5] = vec_perm((vec_t)hp_res[4][ਖਾਨਾ], (vec_t)hp_res[5][ਖਾਨਾ], swiz2);
                        t[6] = vec_perm((vec_t)hp_res[6][ਖਾਨਾ], (vec_t)hp_res[7][ਖਾਨਾ], swiz1);
                        t[7] = vec_perm((vec_t)hp_res[6][ਖਾਨਾ], (vec_t)hp_res[7][ਖਾਨਾ], swiz2);

                        vec_xst(vec_perm(t[0], t[2], swiz3), 0, (vec_t *)(vec_ਵਿੱਥ + 0));
                        vec_xst(vec_perm(t[0], t[2], swiz4), 0, (vec_t *)(vec_ਵਿੱਥ + 16));
                        vec_xst(vec_perm(t[1], t[3], swiz3), 0, (vec_t *)(vec_ਵਿੱਥ + 32));
                        vec_xst(vec_perm(t[1], t[3], swiz4), 0, (vec_t *)(vec_ਵਿੱਥ + 48));
                        vec_xst(vec_perm(t[4], t[6], swiz3), 0, (vec_t *)(vec_ਵਿੱਥ + 64));
                        vec_xst(vec_perm(t[4], t[6], swiz4), 0, (vec_t *)(vec_ਵਿੱਥ + 80));
                        vec_xst(vec_perm(t[5], t[7], swiz3), 0, (vec_t *)(vec_ਵਿੱਥ + 96));
                        vec_xst(vec_perm(t[5], t[7], swiz4), 0, (vec_t *)(vec_ਵਿੱਥ + 112));
                        vec_ਵਿੱਥ += 128;
                    } ਨਹੀਂਤਾਂ {
                        vec_t t0 = vec_perm((vec_t)hp_res[0][ਖਾਨਾ], (vec_t)hp_res[1][ਖਾਨਾ], swiz1);
                        vec_t t1 = vec_perm((vec_t)hp_res[0][ਖਾਨਾ], (vec_t)hp_res[1][ਖਾਨਾ], swiz2);
                        vec_t t2 = vec_perm((vec_t)hp_res[2][ਖਾਨਾ], (vec_t)hp_res[3][ਖਾਨਾ], swiz1);
                        vec_t t3 = vec_perm((vec_t)hp_res[2][ਖਾਨਾ], (vec_t)hp_res[3][ਖਾਨਾ], swiz2);

                        vec_xst(vec_perm(t0, t2, swiz3), 0, (vec_t *)(vec_ਵਿੱਥ + 0));
                        vec_xst(vec_perm(t0, t2, swiz4), 0, (vec_t *)(vec_ਵਿੱਥ + 16));
                        vec_xst(vec_perm(t1, t3, swiz3), 0, (vec_t *)(vec_ਵਿੱਥ + 32));
                        vec_xst(vec_perm(t1, t3, swiz4), 0, (vec_t *)(vec_ਵਿੱਥ + 48));
                        vec_ਵਿੱਥ += 64;
                    }
                }
            }
        }
    }

    ਖਾਲੀ pack_ਸਾਧਾਰਨ_q8_fp16(ਅਚਲ ਖੰਡ_q8_0 * a, int64_t lda, ਪੂਰਨ ਪੰਕਤੀਆਂ, ਪੂਰਨ ਖੰਡਾਂ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * vec) {
        ਜੇ (ਪੰਕਤੀਆਂ == 4) {
            pack_q8_ਖੰਡ<4>(a, lda, ਪੰਕਤੀਆਂ, ਖੰਡਾਂ, vec);
        } ਨਹੀਂਤਾਂ {
            pack_q8_ਖੰਡ<8>(a, lda, ਪੰਕਤੀਆਂ, ਖੰਡਾਂ, vec);
        }
    }

    ਸਾਂਚਾ<ਪੂਰਨ size>
    ਖਾਲੀ pack_ਸਾਧਾਰਨ_Int4(ਅਚਲ TA * a, int64_t lda, ਪੂਰਨ ਪੰਕਤੀਆਂ, ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_2, int8_t * vec, std::array<ਪੂਰਨ, size> & comparray) {
        int64_t i, j;
        TA * aoffset = ਨਿਰਾ;
        int8_t * vec_ਵਿੱਥ = ਨਿਰਾ;
        TA * aoffset1 = ਨਿਰਾ, * aoffset2 = ਨਿਰਾ, * aoffset3 = ਨਿਰਾ, * aoffset4 = ਨਿਰਾ;
        TA * aoffset5 = ਨਿਰਾ, * aoffset6 = ਨਿਰਾ, * aoffset7 = ਨਿਰਾ, * aoffset8 = ਨਿਰਾ;
        vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ c1[2] = {0}, c2[2] = {0}, c3[2] = {0}, c4[2] = {0};
        vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ c5[2] = {0}, c6[2] = {0}, c7[2] = {0}, c8[2] = {0};
        aoffset = const_cast<TA *>(a);
        vec_ਵਿੱਥ = vec;
        j = (ਪੰਕਤੀਆਂ >> 3);
        ਜੇ (j > 0) {
            ਕਰੋ {
                aoffset1 = aoffset;
                aoffset2 = aoffset1 + lda;
                aoffset3 = aoffset2 + lda;
                aoffset4 = aoffset3 + lda;
                aoffset5 = aoffset4 + lda;
                aoffset6 = aoffset5 + lda;
                aoffset7 = aoffset6 + lda;
                aoffset8 = aoffset7 + lda;
                aoffset += 8 * lda;
                i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 2);
                ਜੇ (i > 0) {
                    ਕਰੋ {
                        c1[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset1->qs);
                        c2[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset2->qs);
                        c3[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset3->qs);
                        c4[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset4->qs);
                        c5[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset5->qs);
                        c6[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset6->qs);
                        c7[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset7->qs);
                        c8[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset8->qs);

                        process_q4_elements(c1, & comparray[0]);
                        process_q4_elements(c2, & comparray[1]);
                        process_q4_elements(c3, & comparray[2]);
                        process_q4_elements(c4, & comparray[3]);
                        process_q4_elements(c5, & comparray[4]);
                        process_q4_elements(c6, & comparray[5]);
                        process_q4_elements(c7, & comparray[6]);
                        process_q4_elements(c8, & comparray[7]);
                        ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>(c1[0], c2[0], c3[0], c4[0], vec_ਵਿੱਥ, ਝੂਠ);
                        ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>(c1[1], c2[1], c3[1], c4[1], vec_ਵਿੱਥ + 64, ਝੂਠ);
                        ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>(c5[0], c6[0], c7[0], c8[0], vec_ਵਿੱਥ + 128, ਝੂਠ);
                        ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>(c5[1], c6[1], c7[1], c8[1], vec_ਵਿੱਥ + 192, ਝੂਠ);
                        aoffset1 += lda;
                        aoffset2 += lda;
                        aoffset3 += lda;
                        aoffset4 += lda;
                        aoffset5 += lda;
                        aoffset6 += lda;
                        aoffset7 += lda;
                        aoffset8 += lda;
                        vec_ਵਿੱਥ += 256;
                        i--;
                    } ਜਦੋਂ (i > 0);
                }
                j--;
            } ਜਦੋਂ (j > 0);
        }

        ਜੇ (ਪੰਕਤੀਆਂ & 4) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset += 4 * lda;
            i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 2);
            ਜੇ (i > 0) {
                ਕਰੋ {
                    c1[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset1->qs);
                    c2[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset2->qs);
                    c3[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset3->qs);
                    c4[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset4->qs);

                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>(c1[0], c2[0], c3[0], c4[0], vec_ਵਿੱਥ, ਝੂਠ);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>(c1[1], c2[1], c3[1], c4[1], vec_ਵਿੱਥ + 64, ਝੂਠ);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    aoffset4 += lda;
                    vec_ਵਿੱਥ += 128;
                    i--;
                } ਜਦੋਂ (i > 0);
            }
        }

        ਜੇ (ਪੰਕਤੀਆਂ & 3) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 2);
            ਜੇ (i > 0) {
                ਕਰੋ {
                    ਚੋਣ(ਪੰਕਤੀਆਂ) {
                        ਹਾਲਤ 3: c3[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset3->qs);
                        ਹਾਲਤ 2: c2[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset2->qs);
                        ਹਾਲਤ 1: c1[1] = vec_xl(0, (ਅਚਲ vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)aoffset1->qs);
                            ਰੋਕੋ;
                    }
                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>(c1[0], c2[0], c3[0], c4[0], vec_ਵਿੱਥ, ਝੂਠ);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>(c1[1], c2[1], c3[1], c4[1], vec_ਵਿੱਥ + 64, ਝੂਠ);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    vec_ਵਿੱਥ += 128;
                    i--;
                } ਜਦੋਂ(i > 0);
            }
        }
    }

    ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ VA, ਕਿਸਮ_ਨਾਂ VB>
    ਖਾਲੀ pack_ਸਾਧਾਰਨ(ਅਚਲ ਖੰਡ_q8_0 * a, int64_t lda, ਪੂਰਨ ਪੰਕਤੀਆਂ, ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_2, VA * vec, ਤਾਰਕਿਕ flip) {
        int64_t i, j;
        ਖੰਡ_q8_0 * aoffset = ਨਿਰਾ;
        VA * vec_ਵਿੱਥ = ਨਿਰਾ;
        ਖੰਡ_q8_0 * aoffsets[8];
        __vector_pair arr[8];
        VB c[8][2] = {0};
        VB c1[8] = {0}; VB c2[8] = {0};
        aoffset = const_cast<ਖੰਡ_q8_0 *>(a);
        vec_ਵਿੱਥ = vec;
        j = (ਪੰਕਤੀਆਂ >> 3);
        ਜੇ (j > 0) {
            ਕਰੋ {
                aoffsets[0] = aoffset;
                ਲਈ (ਪੂਰਨ it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it - 1] + lda;
                aoffset += 8 * lda;

                i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 3);
                ਜੇ (i > 0) {
                ਕਰੋ {
                    ਲਈ (ਪੂਰਨ it = 0; it < 8; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_ਵਿੱਥ, flip);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_ਵਿੱਥ + 64, flip);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<VA, VB>(c1[4], c1[5], c1[6], c1[7], vec_ਵਿੱਥ + 128, flip);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<VA, VB>(c2[4], c2[5], c2[6], c2[7], vec_ਵਿੱਥ + 192, flip);
                    ਲਈ (ਪੂਰਨ it = 0; it < 8; it++)
                        aoffsets[it] += lda;
                    vec_ਵਿੱਥ += 256;
                    i--;
               } ਜਦੋਂ(i > 0);
            }
            j--;
        } ਜਦੋਂ(j > 0);
    }
    ਜੇ (ਪੰਕਤੀਆਂ & 4) {
            aoffsets[0]  = aoffset;
            ਲਈ (ਪੂਰਨ it = 1; it < 4; it++ )
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
        i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 3);
            ਜੇ (i > 0) {
               ਕਰੋ {
                    ਲਈ (ਪੂਰਨ it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_ਵਿੱਥ, flip);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_ਵਿੱਥ + 64, flip);
                    ਲਈ (ਪੂਰਨ it = 0; it < 4; it++) {
                        aoffsets[it] += lda;
                    }
                    vec_ਵਿੱਥ += 128;
                    i--;
               } ਜਦੋਂ(i > 0);
            }
        }

        ਜੇ (ਪੰਕਤੀਆਂ & 3) {
            aoffsets[0]  = aoffset;
            ਲਈ (ਪੂਰਨ it = 1; it < 3; it++ )
                aoffsets[it] = aoffsets[it - 1] + lda;
            i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 3);
            ਜੇ (i > 0) {
                ਕਰੋ {
                    ਚੋਣ(ਪੰਕਤੀਆਂ) {
                        ਹਾਲਤ 3: arr[2] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[2]->qs);
                                __builtin_vsx_disassemble_pair(c[2], & arr[2]);
                                c1[2] = c[2][0]; c2[2] = c[2][1];
                        ਹਾਲਤ 2: arr[1] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[1]->qs);
                                __builtin_vsx_disassemble_pair(c[1], & arr[1]);
                                c1[1] = c[1][0]; c2[1] = c[1][1];
                        ਹਾਲਤ 1: arr[0] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[0]->qs);
                                __builtin_vsx_disassemble_pair(c[0], & arr[0]);
                                c1[0] = c[0][0]; c2[0] = c[0][1];
                                ਰੋਕੋ;
                    }
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_ਵਿੱਥ, flip);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_ਵਿੱਥ + 64, flip);
                    ਲਈ (ਪੂਰਨ it = 0; it < 3; it++)
                         aoffsets[it] += lda;
                    vec_ਵਿੱਥ += 128;
                    i--;
               } ਜਦੋਂ(i > 0);
            }
        }
    }

    ਖਾਲੀ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        ਪੂਰਨ m_rem = MIN(m - m0, 16);
        ਪੂਰਨ n_rem = MIN(n - n0, 16);

        ਪੂਰਨ mc = 0, nc = 0;

        ਜੇ (m_rem >= 8 && n_rem >= 8) {
           mc = 8;
           nc = 8;
           gemm<8, 8>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small(m0, m, n0, n, mc, nc);
        } ਨਹੀਂਤਾਂ {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            ਜੇ (mc == 0 || nc == 0)
               ਵਾਪਸ;
            gemm_small(m0, m, n0, n, mc, nc);
        }

        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }


    ਖਾਲੀ KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[16] = {0};
        acc_t acc_0, acc_1;
        std::array<ਪੂਰਨ, 4> comparray {};
        vector ਦਸ਼ਮਲਵ fin_res[8] = {0};
        vector ਦਸ਼ਮਲਵ vs[8] = {0};
        ਤਾਰਕਿਕ isAblock_q4 = std::is_same_v<TA, ਖੰਡ_q4_0>;
        ਲਈ (ਪੂਰਨ l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            ਜੇ (std::is_same_v<TA, ਖੰਡ_q4_0>) {
               pack_ਸਾਧਾਰਨ_Int4<4>((A + (ii * lda) + l), lda, 4, 4, (int8_t *)vec_A, comparray);
            } ਨਹੀਂਤਾਂ {
               pack_ਸਾਧਾਰਨ<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>((ਅਚਲ ਖੰਡ_q8_0 *)(A + (ii * lda) + l), lda, 4, 8, (int8_t *)vec_A, ਝੂਠ);
            }
            pack_ਸਾਧਾਰਨ<uint8_t, vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, ਸੱਚ);
            ਲਈ(ਪੂਰਨ x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x], vec_B[x+8]);
            }
            ਲਈ (ਪੂਰਨ I = 0; I<4; I++) {
                ਲਈ (ਪੂਰਨ J = 0; J<4; J++) {
                    *((ਦਸ਼ਮਲਵ *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((ਦਸ਼ਮਲਵ *)& vs[I + 4] + J) = (unhalf((A +((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            ਜੇ (!isAblock_q4) {
                ਸਵੈਚਾਲਿਤ aoffset = A + (ii * lda) + l;
                ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
                    comparray[i] = 0;
                    ਪੂਰਨ ca = 0;
                    ਸਵੈਚਾਲਿਤ *at = aoffset->qs;
                    ਲਈ (ਪੂਰਨ j = 0; j < 32; j++)
                        ca += (ਪੂਰਨ)*at++;
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

    ਖਾਲੀ KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[8] = {0};
        acc_t acc_0, acc_1;
        std::array<ਪੂਰਨ, 8> comparray {};
        vector ਦਸ਼ਮਲਵ fin_res[8] = {0};
        vector ਦਸ਼ਮਲਵ vs[8] = {0};
        ਤਾਰਕਿਕ isAblock_q4 = std::is_same_v<TA, ਖੰਡ_q4_0>;
        ਲਈ (ਪੂਰਨ l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            ਜੇ (std::is_same_v<TA, ਖੰਡ_q4_0>) {
               pack_ਸਾਧਾਰਨ_Int4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } ਨਹੀਂਤਾਂ {
               pack_ਸਾਧਾਰਨ<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>((ਅਚਲ ਖੰਡ_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, ਝੂਠ);
            }
            pack_ਸਾਧਾਰਨ<uint8_t, vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ>((B + (jj * ldb) + l), ldb, 4, 8, (uint8_t *)vec_B, ਸੱਚ);
            ਲਈ(ਪੂਰਨ x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
            }
            ਲਈ (ਪੂਰਨ I = 0; I < 8; I++) {
                ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                    *((ਦਸ਼ਮਲਵ *)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                }
            }
            ਜੇ (!isAblock_q4) {
                ਸਵੈਚਾਲਿਤ aoffset = A + (ii * lda) + l;
                ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    ਪੂਰਨ ca = 0;
                    ਸਵੈਚਾਲਿਤ *at = aoffset->qs;
                    ਲਈ (ਪੂਰਨ j = 0; j < 32; j++)
                        ca += (ਪੂਰਨ)*at++;
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

    ਖਾਲੀ KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16] = {0};
        acc_t acc_0, acc_1, acc_2, acc_3;
        acc_t acc_4, acc_5, acc_6, acc_7;
        std::array<ਪੂਰਨ, 8> comparray {};
        vector ਦਸ਼ਮਲਵ fin_res[16] = {0};
        vector ਦਸ਼ਮਲਵ vs[16] = {0};
        ਤਾਰਕਿਕ isAblock_q4 = std::is_same_v<TA, ਖੰਡ_q4_0>;
        ਲਈ (ਪੂਰਨ l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            __builtin_mma_xxsetaccz(& acc_2);
            __builtin_mma_xxsetaccz(& acc_3);
            ਜੇ (std::is_same_v<TA, ਖੰਡ_q4_0>) {
               pack_ਸਾਧਾਰਨ_Int4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } ਨਹੀਂਤਾਂ {
               pack_ਸਾਧਾਰਨ<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>((ਅਚਲ ਖੰਡ_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, ਝੂਠ);
            }
            pack_ਸਾਧਾਰਨ<uint8_t, vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, ਸੱਚ);
            ਲਈ(ਪੂਰਨ x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_2, vec_A[x], vec_B[x + 8]);
                __builtin_mma_xvi8ger4pp(& acc_3, vec_A[x + 8], vec_B[x + 8]);
            }
            ਲਈ (ਪੂਰਨ I = 0; I < 8 ; I++) {
                ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                    *((ਦਸ਼ਮਲਵ *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((ਦਸ਼ਮਲਵ *)& vs[I + 8] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            ਜੇ (!isAblock_q4) {
                ਸਵੈਚਾਲਿਤ aoffset = A + (ii * lda) + l;
                ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    ਪੂਰਨ ca = 0;
                    ਸਵੈਚਾਲਿਤ *at = aoffset->qs;
                    ਲਈ (ਪੂਰਨ j = 0; j < 32; j++)
                        ca += (ਪੂਰਨ)*at++;
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

    ਖਾਲੀ KERNEL_Q0(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, int64_t l, vec_t * vec_A, vec_t * vec_B) {
        acc_t acc[8];
        ਲਈ (ਪੂਰਨ i = 0; i < mc ; i += 16) {
            ਲਈ (ਪੂਰਨ j = 0; j < nc; j += 8) {
                ਪੂਰਨ A0_ਅਧਾਰ = (i / 16) * (2 * 32 * kc);
                ਪੂਰਨ B0_ਅਧਾਰ = (j / 8) * (32 * kc);
                ਲਈ (ਪੂਰਨ x = 0; x < 8; x++) {
                     __builtin_mma_xxsetaccz(&acc[x]);
                }
                ਲਈ (int64_t kk = 0; kk < kc; kk++) {
                    ਪੂਰਨ A0_ਖੰਡ_idx = A0_ਅਧਾਰ + kk * 32;
                    ਪੂਰਨ B0_ਖੰਡ_idx = B0_ਅਧਾਰ + kk * 32;
                    ਪੂਰਨ A1_ਖੰਡ_idx = A0_ਖੰਡ_idx + 32 * kc;
                    ਪੂਰਨ B1_ਖੰਡ_idx = B0_ਖੰਡ_idx + 32 * kc;
                    vec_t * A0_ਖੰਡ = & vec_A[A0_ਖੰਡ_idx];
                    vec_t * B0_ਖੰਡ = & vec_B[B0_ਖੰਡ_idx];
                    vec_t * A1_ਖੰਡ = & vec_A[A1_ਖੰਡ_idx];
                    ਲਈ (ਪੂਰਨ it = 0; it < 4; it++) {
                        ਲਈ (ਪੂਰਨ x = 0; x < 4; x++) {
                            __builtin_mma_xvf16ger2pp(& acc[0], A0_ਖੰਡ[8 * it + x], B0_ਖੰਡ[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[1], A0_ਖੰਡ[8 * it + x], B0_ਖੰਡ[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[2], A0_ਖੰਡ[8 * it + x + 4], B0_ਖੰਡ[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[3], A0_ਖੰਡ[8 * it + x + 4], B0_ਖੰਡ[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[4], A1_ਖੰਡ[8 * it + x], B0_ਖੰਡ[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[5], A1_ਖੰਡ[8 * it + x], B0_ਖੰਡ[8 * it+ x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[6], A1_ਖੰਡ[8 * it + x + 4], B0_ਖੰਡ[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[7], A1_ਖੰਡ[8 * it + x + 4], B0_ਖੰਡ[8 * it + x + 4]);
                        }
                    }
                }
                ਜੇ (l == 0) {
                    save_acc(& acc[0], ii + i, jj + j);
                    save_acc(& acc[1], ii + i, jj + j + 4);
                    save_acc(& acc[2], ii + i + 4, jj + j);
                    save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    save_acc(& acc[4], ii + i + 8, jj + j);
                    save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    save_acc(& acc[6], ii + i + 12, jj + j);
                    save_acc(& acc[7], ii + i + 12, jj + j + 4);
                } ਨਹੀਂਤਾਂ {
                    ਜੋੜੋ_save_acc(& acc[0], ii + i, jj + j);
                    ਜੋੜੋ_save_acc(& acc[1], ii + i, jj + j + 4);
                    ਜੋੜੋ_save_acc(& acc[2], ii + i + 4, jj + j);
                    ਜੋੜੋ_save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    ਜੋੜੋ_save_acc(& acc[4], ii + i + 8, jj + j);
                    ਜੋੜੋ_save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    ਜੋੜੋ_save_acc(& acc[6], ii + i + 12, jj + j);
                    ਜੋੜੋ_save_acc(& acc[7], ii + i + 12, jj + j + 4);
                }
            }
        }
    }

    ਖਾਲੀ matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        vec_t A_pack[mc * kc * 4];
        vec_t B_pack[nc * kc * 4];
        constexpr ਤਾਰਕਿਕ is_Ablock_q4 = std::is_same_v<TA, ਖੰਡ_q4_0>;
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles  = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles) {
            end = tiles;
        }
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = (ਕੰਮ / xtiles) * mc;
            int64_t jj = (ਕੰਮ % xtiles) * nc;
            ਲਈ (int64_t kk = 0; kk < k; kk += kc) {
                int64_t k_cur = MIN(kc, k - kk);
                ਜੇ constexpr(is_Ablock_q4) {
                    pack_ਸਾਧਾਰਨ_q4_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                } ਨਹੀਂਤਾਂ {
                    pack_ਸਾਧਾਰਨ_q8_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                }
                pack_ਸਾਧਾਰਨ_q8_fp16(B + jj * ldb + kk, ldb, nc, k_cur, (uint8_t *)B_pack);
                KERNEL_Q0(ii, jj, mc, nc, k_cur, kk, A_pack, B_pack);
            }
        }
    }

    ਖਾਲੀ gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, ਪੂਰਨ RM, ਪੂਰਨ RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        vec_t vec_A[8] = {0}, vec_B[8] = {0};
        vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vec_C[4];
        acc_t acc_0;
        ਤਾਰਕਿਕ isAblock_q4 = std::is_same_v<TA, ਖੰਡ_q4_0>;

        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            std::array<ਪੂਰਨ, 4> comparray{};
            vector ਦਸ਼ਮਲਵ res[4] = {0};
            vector ਦਸ਼ਮਲਵ fin_res[4] = {0};
            vector ਦਸ਼ਮਲਵ vs[4] = {0};
            vector ਦਸ਼ਮਲਵ CA[4] = {0};
            __builtin_prefetch((A + (ii * lda) + 0)->qs, 0, 1); // prefetch first value
            __builtin_prefetch((B + (jj * ldb) + 0)->qs, 0, 1); // prefetch first value
            ਲਈ (ਪੂਰਨ l = 0; l < k; l++) {
                __builtin_prefetch((A + (ii * lda) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_prefetch((B + (jj * ldb) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_mma_xxsetaccz(& acc_0);
                ਜੇ (isAblock_q4) {
                    pack_ਸਾਧਾਰਨ_Int4<4>((A + (ii * lda) + l), lda, RM, 4, (int8_t *)vec_A, comparray);
                } ਨਹੀਂਤਾਂ {
                    pack_ਸਾਧਾਰਨ<int8_t, vector ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ>((ਅਚਲ ਖੰਡ_q8_0 *)(A + (ii * lda) + l), lda, RM, 8, (int8_t *)vec_A, ਝੂਠ);
                }
                pack_ਸਾਧਾਰਨ<uint8_t, vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ>((B + (jj * ldb) + l), ldb, RN, 8, (uint8_t *)vec_B, ਸੱਚ);
                ਲਈ (ਪੂਰਨ x = 0; x < 8; x += 4) {
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 1], vec_B[x + 1]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 2], vec_B[x + 2]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 3], vec_B[x + 3]);
                }
                ਲਈ (ਪੂਰਨ I = 0; I < RM; I++) {
                    ਲਈ (ਪੂਰਨ J = 0; J < RN; J++) {
                        *((ਦਸ਼ਮਲਵ*)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    }
                }
                __builtin_mma_disassemble_acc(vec_C, & acc_0);
                ਜੇ (!isAblock_q4) {
                    ਸਵੈਚਾਲਿਤ aoffset = A + (ii * lda) + l;
                    ਲਈ (ਪੂਰਨ i = 0; i < RM; i++) {
                        comparray[i] = 0;
                        ਪੂਰਨ ca = 0;
                        ਸਵੈਚਾਲਿਤ *at = aoffset->qs;
                        ਲਈ (ਪੂਰਨ j = 0; j < 32; j++)
                            ca += (ਪੂਰਨ)*at++;
                        comparray[i] = ca;
                        aoffset += lda;
                    }
                }
                ਲਈ (ਪੂਰਨ i = 0; i < RM; i++) {
                    CA[i] = vec_splats((ਦਸ਼ਮਲਵ)(((ਦੁੱਗਣਾ)comparray[i]) * -128.0));
                    res[i] = vec_ਜੋੜੋ(vec_ctf(vec_C[i], 0), CA[i]);
                    fin_res[i] = vec_madd(res[i], vs[i], fin_res[i]);
                }
            }
            save_res(ii, jj, 0, fin_res, RM, RN);
        }
    }

    ਸਾਂਚਾ<ਪੂਰਨ RM, ਪੂਰਨ RN>
    ਸਤਰਅੰਦਰ ਖਾਲੀ kernel(int64_t ii, int64_t jj) {
        ਜੇ constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii,jj);
        } ਨਹੀਂਤਾਂ ਜੇ constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii,jj);
        } ਨਹੀਂਤਾਂ ਜੇ constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii,jj);
        } ਨਹੀਂਤਾਂ {
            ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ && "RN/RM values not supported");
        }
    }

    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN>
    NOINLINE ਖਾਲੀ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }
    ਅਚਲ TA * ਅਚਲ A;
    ਅਚਲ ਖੰਡ_q8_0 * ਅਚਲ B;
    ਦਸ਼ਮਲਵ * C;
    ਅਚਲ int64_t k;
    int64_t kc;
    ਅਚਲ int64_t lda;
    ਅਚਲ int64_t ldb;
    ਅਚਲ int64_t ldc;
    ਅਚਲ ਪੂਰਨ ith;
    ਅਚਲ ਪੂਰਨ nth;
};

ਵਰਗ tinyBLAS_PPC {
  ਜਨਤਕ:
    tinyBLAS_PPC(int64_t k,
                ਅਚਲ ਦਸ਼ਮਲਵ * A, int64_t lda,
                ਅਚਲ ਦਸ਼ਮਲਵ * B, int64_t ldb,
                ਦਸ਼ਮਲਵ * C, int64_t ldc,
                ਪੂਰਨ ith, ਪੂਰਨ nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    ਖਾਲੀ matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #ਜੇ defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #ਜੇ_ਅੰਤ
        ਜੇ (m % mc == 0 && n % nc == 0 && k % kc == 0) {
            matmul_tiled(m, n, mc, nc, kc);
        } ਨਹੀਂਤਾਂ {
            mnpack(0, m, 0, n);
        }
    }

  ਨਿਜੀ:

    __attribute__((always_inline))
    ਸਤਰਅੰਦਰ ਖਾਲੀ save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ਲਈ (ਪੂਰਨ I = 0; I < 4; I++) {
            ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                *((ਦਸ਼ਮਲਵ *)(C+ii+((jj+J)*ldc)+I)) = *((ਦਸ਼ਮਲਵ *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    ਸਤਰਅੰਦਰ ਖਾਲੀ ਜੋੜੋ_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        ਲਈ (ਪੂਰਨ I = 0; I < 4; I++) {
            ਲਈ (ਪੂਰਨ J = 0; J < 4; J++) {
                ਦਸ਼ਮਲਵ * c_ptr = (ਦਸ਼ਮਲਵ *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((ਦਸ਼ਮਲਵ *)&vec_C[I]+J);
            }
        }
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_4(vector ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * vec_ਵਿੱਥ) {
        vector ਦਸ਼ਮਲਵ t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergel(src[0], src[1]);
        t4 = vec_mergel(src[2], src[3]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t1, t2, 3);
        t7 = vec_xxpermdi(t3, t4, 0);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_ਵਿੱਥ);
        vec_xst(t6, 0, vec_ਵਿੱਥ + 4);
        vec_xst(t7, 0, vec_ਵਿੱਥ + 8);
        vec_xst(t8, 0, vec_ਵਿੱਥ + 12);
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_8(vector ਦਸ਼ਮਲਵ * src, ਦਸ਼ਮਲਵ * vec_ਵਿੱਥ) {
        vector ਦਸ਼ਮਲਵ t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergeh(src[4], src[5]);
        t4 = vec_mergeh(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_ਵਿੱਥ);
        vec_xst(t6, 0, vec_ਵਿੱਥ + 4);
        vec_xst(t7, 0, vec_ਵਿੱਥ + 8);
        vec_xst(t8, 0, vec_ਵਿੱਥ + 12);

        t1 = vec_mergel(src[0], src[1]);
        t2 = vec_mergel(src[2], src[3]);
        t3 = vec_mergel(src[4], src[5]);
        t4 = vec_mergel(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_ਵਿੱਥ + 16);
        vec_xst(t6, 0, vec_ਵਿੱਥ + 20);
        vec_xst(t7, 0, vec_ਵਿੱਥ + 24);
        vec_xst(t8, 0, vec_ਵਿੱਥ + 28);
    }

    ਖਾਲੀ packTranspose(ਅਚਲ ਦਸ਼ਮਲਵ * a, int64_t lda, ਪੂਰਨ ਪੰਕਤੀਆਂ, ਪੂਰਨ ਖਾਨੇ_ਗਿਣਤੀ_2, ਦਸ਼ਮਲਵ * vec) {
        int64_t i, j;
        ਦਸ਼ਮਲਵ * aoffsets[8];
        ਦਸ਼ਮਲਵ * aoffset = ਨਿਰਾ, * boffset = ਨਿਰਾ;
        __vector_pair arr[8];
        vector ਦਸ਼ਮਲਵ c[8][2] = {0};
        vector ਦਸ਼ਮਲਵ c1[8] = {0};
        vector ਦਸ਼ਮਲਵ c2[8] = {0};
        aoffset = const_cast<ਦਸ਼ਮਲਵ *>(a);
        boffset = vec;
        j = (ਪੰਕਤੀਆਂ >> 3);
        ਜੇ (j > 0) {
            ਕਰੋ {
                aoffsets[0] = aoffset;
                ਲਈ (ਪੂਰਨ it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it-1] + lda;
                aoffset += 8 * lda;
                i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 3);
                ਜੇ (i > 0) {
                    ਕਰੋ {
                        ਲਈ (ਪੂਰਨ it = 0; it < 8; it++) {
                            arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                            __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                            c1[it] = c[it][0];
                            c2[it] = c[it][1];
                        }

                        ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_8(c1, boffset);
                        ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_8(c2, boffset + 32);
                        boffset += 64;
                        i--;
                        ਜੇ (i > 0) {
                           ਲਈ (ਪੂਰਨ it = 0; it < 8; it++) {
                               aoffsets[it] = aoffsets[it] + 8;
                           }
                        }
                    } ਜਦੋਂ(i > 0);
                }
                ਜੇ (ਖਾਨੇ_ਗਿਣਤੀ_2 & 4) {
                    ਲਈ (ਪੂਰਨ it = 0; it < 8 ; it++)
                        c1[it] = vec_xl(0, aoffsets[it]);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_8(c1, boffset);
                }
            j--;
            } ਜਦੋਂ(j > 0);
        }

        ਜੇ (ਪੰਕਤੀਆਂ & 4) {
            aoffsets[0] = aoffset;
            ਲਈ (ਪੂਰਨ it = 1; it < 4; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            i = (ਖਾਨੇ_ਗਿਣਤੀ_2 >> 3);
            ਜੇ (i > 0) {
                ਕਰੋ {
                    ਲਈ (ਪੂਰਨ it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                        __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_4(c1, boffset);
                    ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_4(c2, boffset + 16);
                    ਲਈ (ਪੂਰਨ it = 0; it < 4; it++)
                        aoffsets[it] += 8 * lda;
                    boffset += 32;
                    i--;
                } ਜਦੋਂ(i > 0);
            }

            ਜੇ (ਖਾਨੇ_ਗਿਣਤੀ_2 & 4) {
               ਲਈ (ਪੂਰਨ it = 0; it < 4; it++)
                   c1[it] = vec_xl(0, aoffsets[it]);
                ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_4(c1, boffset);
            }
        }
        ਜੇ (ਪੰਕਤੀਆਂ & 3) {
            aoffsets[0] = aoffset;
            ਲਈ (ਪੂਰਨ it = 1; it < 3; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            ਜੇ (ਖਾਨੇ_ਗਿਣਤੀ_2 & 4) {
                ਲਈ (ਪੂਰਨ it = 0; it < 3; it++)
                    c1[it] = vec_xl(0, aoffsets[it]);
                ਸਦਿਸ਼_ਕ੍ਰਮ_ਬਦਲੋ_store_4(c1, boffset);
            }
        }
    }

    ਖਾਲੀ KERNEL_4x4(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[4], vec_C[4];
        acc_t acc_0;
        __builtin_mma_xxsetaccz(&acc_0);
        ਲਈ (ਪੂਰਨ l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (ਦਸ਼ਮਲਵ *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (ਦਸ਼ਮਲਵ *)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
        }
        save_acc(&acc_0, ii, jj);
    }

    ਖਾਲੀ KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        ਲਈ (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (ਦਸ਼ਮਲਵ *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 4, (ਦਸ਼ਮਲਵ *)vec_B);
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

    ਖਾਲੀ KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        ਲਈ (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 8, 4, (ਦਸ਼ਮਲਵ *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (ਦਸ਼ਮਲਵ *)vec_B);
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

    ਖਾਲੀ KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        ਲਈ (ਪੂਰਨ l = 0; l < k; l+=8) {
            packTranspose(A + (ii * lda) + l, lda, 8, 8, (ਦਸ਼ਮਲਵ *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 8, (ਦਸ਼ਮਲਵ *)vec_B);
            ਲਈ(ਪੂਰਨ x = 0; x < 16; x+=2) {
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

    ਸਤਰਅੰਦਰ ਖਾਲੀ MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        ਲਈ (ਪੂਰਨ x = 0; x < 16; x += 2) {
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

    ਖਾਲੀ KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        ਲਈ (int64_t i = 0; i < mc; i += 16) {
            ਪੂਰਨ A_ਅਧਾਰ_addr = (mc / 8) * (i / 8) * 16;
            ਲਈ (int64_t j = 0; j < nc; j += 8) {
                 ਪੂਰਨ B_ਅਧਾਰ_addr = (nc / 8) * (j / 8) * 16;
                 acc_t acc[8];
                 vec_t A0_ਖੰਡ[16]; vec_t A1_ਖੰਡ[16];
                 ਲਈ (ਪੂਰਨ x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 ਲਈ (int64_t l = 0; l < kc; l += 8) {
                     ਪੂਰਨ A0_ਖੰਡ_idx = A_ਅਧਾਰ_addr + (l / 8) * 16;
                     ਪੂਰਨ A1_ਖੰਡ_idx = A0_ਖੰਡ_idx + (mc / 8) * 16;
                     ਪੂਰਨ B_ਖੰਡ_idx = B_ਅਧਾਰ_addr + (l / 8) * 16;
                     vec_t* A0_ਖੰਡ = &vec_A[A0_ਖੰਡ_idx];
                     vec_t* A1_ਖੰਡ = &vec_A[A1_ਖੰਡ_idx];
                     vec_t* B_ਖੰਡ = &vec_B[B_ਖੰਡ_idx];
                     MMA_16x8(A0_ਖੰਡ, A1_ਖੰਡ, B_ਖੰਡ, acc);
                 }
                 ਜੇ (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } ਨਹੀਂਤਾਂ {
                     ਜੋੜੋ_save_acc(&acc[0], ii + i, jj + j);
                     ਜੋੜੋ_save_acc(&acc[1], ii + i, jj + j + 4);
                     ਜੋੜੋ_save_acc(&acc[2], ii + i + 4, jj + j);
                     ਜੋੜੋ_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     ਜੋੜੋ_save_acc(&acc[4], ii + i + 8, jj + j);
                     ਜੋੜੋ_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     ਜੋੜੋ_save_acc(&acc[6], ii + i + 12, jj + j);
                     ਜੋੜੋ_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }

    ਖਾਲੀ matmul_tiled(int64_t m , int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles) {
            end = tiles;
        }
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = (ਕੰਮ / xtiles) * mc;
            int64_t jj = (ਕੰਮ % xtiles) * nc;
            ਲਈ (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 4];
                 vec_t B_pack[kc * nc / 4];
                 packTranspose(A + (ii * lda) + kk, lda, kc, mc, (ਦਸ਼ਮਲਵ *)A_pack);
                 packTranspose(B + (jj * ldb) + kk, ldb, kc, nc, (ਦਸ਼ਮਲਵ *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    ਖਾਲੀ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        ਪੂਰਨ m_rem = MIN(m - m0, 8);
        ਪੂਰਨ n_rem = MIN(n - n0, 8);
        ਪੂਰਨ mc = 0, nc = 0;
        ਜੇ (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8, 8>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ ਜੇ (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n);
        } ਨਹੀਂਤਾਂ {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            ਜੇ (mc == 0 || nc == 0)
                ਵਾਪਸ;
            gemm_small(m0, m, n0, n, mc, nc);
        }
        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    ਖਾਲੀ gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, ਪੂਰਨ RM, ਪੂਰਨ RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[4] = {0}, vec_B[4] = {0};
            ਲਈ (ਪੂਰਨ l = 0; l < k; l += 4) {
                /* 'GEMV Forwarding' concept is used in first two conditional loops.
                 * when one of the matrix has a single row/column, the elements are
                 * broadcasted, instead of using packing routine to prepack the
                 * matrix elements.
                 */
                ਜੇ (RM == 1) {
                    ਦਸ਼ਮਲਵ * a = const_cast<ਦਸ਼ਮਲਵ *>(A + (ii) * lda + l);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (ਦਸ਼ਮਲਵ *)vec_B);
                    vec_A[0] = (vec_t)vec_xl(0,a);
                    vec_A[1] = (vec_t)vec_splats(*((ਦਸ਼ਮਲਵ *)&vec_A+1));
                    vec_A[2] = (vec_t)vec_splats(*((ਦਸ਼ਮਲਵ *)&vec_A+2));
                    vec_A[3] = (vec_t)vec_splats(*((ਦਸ਼ਮਲਵ *)&vec_A+3));
                } ਨਹੀਂਤਾਂ ਜੇ (RN == 1) {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (ਦਸ਼ਮਲਵ *)vec_A);
                    ਦਸ਼ਮਲਵ * b = const_cast<ਦਸ਼ਮਲਵ *>(B + (jj) * ldb + l);
                    vec_B[0] = (vec_t)vec_xl(0,b);
                    vec_B[1] = (vec_t)vec_splats(*((ਦਸ਼ਮਲਵ *)&vec_B+1));
                    vec_B[2] = (vec_t)vec_splats(*((ਦਸ਼ਮਲਵ *)&vec_B+2));
                    vec_B[3] = (vec_t)vec_splats(*((ਦਸ਼ਮਲਵ *)&vec_B+3));
                } ਨਹੀਂਤਾਂ {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (ਦਸ਼ਮਲਵ *)vec_A);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (ਦਸ਼ਮਲਵ *)vec_B);
                }
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            ਲਈ (ਪੂਰਨ I = 0; I < RM; I++) {
                ਲਈ (ਪੂਰਨ J = 0; J < RN; J++) {
                    *((ਦਸ਼ਮਲਵ *)(C+ii+((jj+J)*ldc)+I)) = *((ਦਸ਼ਮਲਵ *)&vec_C[I]+J);
                }
            }
       }
    }

    ਸਾਂਚਾ<ਪੂਰਨ RM, ਪੂਰਨ RN>
    ਸਤਰਅੰਦਰ ਖਾਲੀ kernel(int64_t ii, int64_t jj) {
        ਜੇ constexpr(RM == 4 && RN == 4) {
            KERNEL_4x4(ii, jj);
        } ਨਹੀਂਤਾਂ ਜੇ constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii, jj);
        } ਨਹੀਂਤਾਂ ਜੇ constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii, jj);
        } ਨਹੀਂਤਾਂ ਜੇ constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii, jj);
        } ਨਹੀਂਤਾਂ {
            static_assert(ਝੂਠ, "RN/RM values not supported");
        }
    }

    ਸਾਂਚਾ <ਪੂਰਨ RM, ਪੂਰਨ RN>
    NOINLINE ਖਾਲੀ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ਜੇ (end > tiles)
            end = tiles;
        ਲਈ (int64_t ਕੰਮ = start; ਕੰਮ < end; ++ਕੰਮ) {
            int64_t ii = m0 + ਕੰਮ / xtiles * RM;
            int64_t jj = n0 + ਕੰਮ % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ A;
    ਅਚਲ ਦਸ਼ਮਲਵ * ਅਚਲ B;
    ਦਸ਼ਮਲਵ * C;
    ਅਚਲ int64_t k;
    ਅਚਲ int64_t lda;
    ਅਚਲ int64_t ldb;
    ਅਚਲ int64_t ldc;
    ਅਚਲ ਪੂਰਨ ith;
    ਅਚਲ ਪੂਰਨ nth;
};
#ਜੇ_ਅੰਤ
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
ਤਾਰਕਿਕ llamafile_sgemm(ਅਚਲ ਬਣਤਰ ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                     ਅਚਲ ਖਾਲੀ *A, int64_t lda, ਅਚਲ ਖਾਲੀ *B, int64_t ldb, ਖਾਲੀ *C,
                     int64_t ldc, ਪੂਰਨ Atype, ਪੂਰਨ Btype, ਪੂਰਨ Ctype) {

    ਪ੍ਰਤਿਜ੍ਞਾ(m >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(n >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(k >= 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(lda >= k);
    ਪ੍ਰਤਿਜ੍ਞਾ(ldb >= k);
    ਪ੍ਰਤਿਜ੍ਞਾ(ldc >= m);
    ਪ੍ਰਤਿਜ੍ਞਾ(params->nth > 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(params->ith < params->nth);

    // only enable sgemm for prompt processing
#ਜੇ !defined(__MMA__)
    ਜੇ (n < 2)
        ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ

    ਜੇ (Ctype != GGML_TYPE_F32)
        ਵਾਪਸ ਝੂਠ;

    ਚੋਣ (Atype) {

    ਹਾਲਤ GGML_TYPE_F32: {
        ਜੇ (Btype != GGML_TYPE_F32)
            ਵਾਪਸ ਝੂਠ;
#ਜੇ defined(__AVX512F__)
        tinyBLAS<16, __m512, __m512, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ> tb{ params,
            k, (ਅਚਲ ਦਸ਼ਮਲਵ *)A, lda,
            (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc};
        ਵਾਪਸ tb.matmul(m, n);
#ਜਾਂਜੇ defined(__AVX__) || defined(__AVX2__)
        tinyBLAS<8, __m256, __m256, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ> tb{ params,
            k, (ਅਚਲ ਦਸ਼ਮਲਵ *)A, lda,
            (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc};
        ਵਾਪਸ tb.matmul(m, n);
#ਜਾਂਜੇ defined(__ARM_NEON)
        ਜੇ (n < 4)
            ਵਾਪਸ ਝੂਠ;
        tinyBLAS<4, float32x4_t, float32x4_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ> tb{ params,
            k, (ਅਚਲ ਦਸ਼ਮਲਵ *)A, lda,
            (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc};
        ਵਾਪਸ tb.matmul(m, n);
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
        ਜੇ (n < 4)
            ਵਾਪਸ ਝੂਠ;
        tinyBLAS<4, float32x4_t, float32x4_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ> tb{ params,
            k, (ਅਚਲ ਦਸ਼ਮਲਵ *)A, lda,
            (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc};
        ਵਾਪਸ tb.matmul(m, n);
#ਜਾਂਜੇ defined(__MMA__)
        ਜੇ (k % 8)
            ਵਾਪਸ ਝੂਠ;
        tinyBLAS_PPC tb{
            k, (ਅਚਲ ਦਸ਼ਮਲਵ *)A, lda,
            (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ਵਾਪਸ ਸੱਚ;
#ਜਾਂਜੇ defined(__riscv_v_intrinsic)
    #ਜੇ LMUL == 1
        tinyBLAS_RVV<vfloat32m1_t, vfloat32m1_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ> tb{ params,
            k, (ਅਚਲ ਦਸ਼ਮਲਵ *)A, lda,
            (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc};
    #ਜਾਂਜੇ LMUL == 2
        tinyBLAS_RVV<vfloat32m2_t, vfloat32m2_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ> tb{ params,
            k, (ਅਚਲ ਦਸ਼ਮਲਵ *)A, lda,
            (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc};
    #ਨਹੀਂਤਾਂ // LMUL = 4
        tinyBLAS_RVV<vfloat32m4_t, vfloat32m4_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ> tb{ params,
            k, (ਅਚਲ ਦਸ਼ਮਲਵ *)A, lda,
            (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc};
    #ਜੇ_ਅੰਤ
        ਵਾਪਸ tb.matmul(m, n);
#ਨਹੀਂਤਾਂ
        ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
    }

    ਹਾਲਤ GGML_TYPE_BF16: {
#ਜੇ defined(__AVX512BF16__)
        ਜੇ (Btype == GGML_TYPE_BF16) {
            tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, ਦਸ਼ਮਲਵ> tb{ params, k,
                (ਅਚਲ ggml_bf16_t *)A, lda,
                (ਅਚਲ ggml_bf16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
            ਵਾਪਸ tb.matmul(m, n);
        }
#ਜਾਂਜੇ defined(__AVX512F__)
        ਜੇ (Btype == GGML_TYPE_BF16) {
            tinyBLAS<16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, ਦਸ਼ਮਲਵ> tb{ params, k,
                (ਅਚਲ ggml_bf16_t *)A, lda,
                (ਅਚਲ ggml_bf16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
            ਵਾਪਸ tb.matmul(m, n);
        }
#ਜਾਂਜੇ defined(__AVX2__)
        ਜੇ (Btype == GGML_TYPE_BF16) {
            tinyBLAS<8, __m256, __m256, ggml_bf16_t, ggml_bf16_t, ਦਸ਼ਮਲਵ> tb{ params, k,
                (ਅਚਲ ggml_bf16_t *)A, lda,
                (ਅਚਲ ggml_bf16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
            ਵਾਪਸ tb.matmul(m, n);
        }
#ਜਾਂਜੇ defined(__MMA__)
        ਜੇ (k % 8) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (Btype == GGML_TYPE_BF16) {
            tinyBLAS_HP16_PPC<ggml_bf16_t, ggml_bf16_t, ਦਸ਼ਮਲਵ> tb{ k,
                (ਅਚਲ ggml_bf16_t *)A, lda,
                (ਅਚਲ ggml_bf16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            ਵਾਪਸ ਸੱਚ;
        }
#ਜਾਂਜੇ defined(__riscv_zvfbfwma)
        ਜੇ (Btype == GGML_TYPE_BF16) {
            #ਜੇ LMUL == 1
                tinyBLAS_RVV<vfloat32m1_t, vbfloat16mf2_t, ggml_bf16_t, ggml_bf16_t, ਦਸ਼ਮਲਵ> tb{ params,
                    k, (ਅਚਲ ggml_bf16_t *)A, lda,
                    (ਅਚਲ ggml_bf16_t *)B, ldb,
                    (ਦਸ਼ਮਲਵ *)C, ldc};
            #ਜਾਂਜੇ LMUL == 2
                tinyBLAS_RVV<vfloat32m2_t, vbfloat16m1_t, ggml_bf16_t, ggml_bf16_t, ਦਸ਼ਮਲਵ> tb{ params,
                    k, (ਅਚਲ ggml_bf16_t *)A, lda,
                    (ਅਚਲ ggml_bf16_t *)B, ldb,
                    (ਦਸ਼ਮਲਵ *)C, ldc};
            #ਨਹੀਂਤਾਂ // LMUL = 4
                tinyBLAS_RVV<vfloat32m4_t, vbfloat16m2_t, ggml_bf16_t, ggml_bf16_t, ਦਸ਼ਮਲਵ> tb{ params,
                    k, (ਅਚਲ ggml_bf16_t *)A, lda,
                    (ਅਚਲ ggml_bf16_t *)B, ldb,
                    (ਦਸ਼ਮਲਵ *)C, ldc};
            #ਜੇ_ਅੰਤ
                ਵਾਪਸ tb.matmul(m, n);
        }
#ਜੇ_ਅੰਤ
        ਵਾਪਸ ਝੂਠ;
    }

    ਹਾਲਤ GGML_TYPE_F16: {
#ਜੇ defined(__AVX512F__)
        ਜੇ (Btype == GGML_TYPE_F16) {
            tinyBLAS<16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, ਦਸ਼ਮਲਵ> tb{ params, k,
                (ਅਚਲ ggml_fp16_t *)A, lda,
                (ਅਚਲ ggml_fp16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
            ਵਾਪਸ tb.matmul(m, n);
        }
#ਜਾਂਜੇ (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        ਜੇ (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, ਦਸ਼ਮਲਵ> tb{ params, k,
                (ਅਚਲ ggml_fp16_t *)A, lda,
                (ਅਚਲ ggml_fp16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
            ਵਾਪਸ tb.matmul(m, n);
        }
#ਜਾਂਜੇ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
        ਜੇ (n < 8)
            ਵਾਪਸ ਝੂਠ;
        ਜੇ (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, ਦਸ਼ਮਲਵ> tb{ params,
                k, (ਅਚਲ ggml_fp16_t *)A, lda,
                (ਅਚਲ ggml_fp16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
            ਵਾਪਸ tb.matmul(m, n);
        }
#ਜਾਂਜੇ defined(__ARM_NEON) && !defined(_MSC_VER)
        ਜੇ (Btype == GGML_TYPE_F32) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ> tb{ params,
                k, (ਅਚਲ ggml_fp16_t *)A, lda,
                (ਅਚਲ ਦਸ਼ਮਲਵ *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
            ਵਾਪਸ tb.matmul(m, n);
        }
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
        ਜੇ (n < 4)
            ਵਾਪਸ ਝੂਠ;
        ਜੇ (Btype == GGML_TYPE_F16) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, ਦਸ਼ਮਲਵ> tb{ params,
                k, (ਅਚਲ ggml_fp16_t *)A, lda,
                (ਅਚਲ ggml_fp16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
            ਵਾਪਸ tb.matmul(m, n);
        }
#ਜਾਂਜੇ defined(__riscv_zvfh)
        ਜੇ (Btype == GGML_TYPE_F16) {
        #ਜੇ LMUL == 1
            tinyBLAS_RVV<vfloat32m1_t, vfloat16mf2_t, ggml_fp16_t, ggml_fp16_t, ਦਸ਼ਮਲਵ> tb{ params,
                k, (ਅਚਲ ggml_fp16_t *)A, lda,
                (ਅਚਲ ggml_fp16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
        #ਜਾਂਜੇ LMUL == 2
            tinyBLAS_RVV<vfloat32m2_t, vfloat16m1_t, ggml_fp16_t, ggml_fp16_t, ਦਸ਼ਮਲਵ> tb{ params,
                k, (ਅਚਲ ggml_fp16_t *)A, lda,
                (ਅਚਲ ggml_fp16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
        #ਨਹੀਂਤਾਂ // LMUL = 4
            tinyBLAS_RVV<vfloat32m4_t, vfloat16m2_t, ggml_fp16_t, ggml_fp16_t, ਦਸ਼ਮਲਵ> tb{ params,
                k, (ਅਚਲ ggml_fp16_t *)A, lda,
                (ਅਚਲ ggml_fp16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc};
        #ਜੇ_ਅੰਤ
            ਵਾਪਸ tb.matmul(m, n);
        }
#ਜਾਂਜੇ defined(__MMA__)
        ਜੇ (k % 8) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (Btype == GGML_TYPE_F16) {
            tinyBLAS_HP16_PPC<ggml_fp16_t, ggml_fp16_t, ਦਸ਼ਮਲਵ> tb{ k,
                (ਅਚਲ ggml_fp16_t *)A, lda,
                (ਅਚਲ ggml_fp16_t *)B, ldb,
                (ਦਸ਼ਮਲਵ *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            ਵਾਪਸ ਸੱਚ;
        }
#ਜੇ_ਅੰਤ
        ਵਾਪਸ ਝੂਠ;
    }

    ਹਾਲਤ GGML_TYPE_Q8_0: {
        ਜੇ (Btype != GGML_TYPE_Q8_0)
           ਵਾਪਸ ਝੂਠ;
#ਜੇ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ਖੰਡ_q8_0, ਖੰਡ_q8_0, ਦਸ਼ਮਲਵ> tb{
            k, (ਅਚਲ ਖੰਡ_q8_0 *)A, lda,
            (ਅਚਲ ਖੰਡ_q8_0 *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ਵਾਪਸ ਸੱਚ;
#ਜਾਂਜੇ defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<ਖੰਡ_q8_0> tb{
            k, (ਅਚਲ ਖੰਡ_q8_0 *)A, lda,
            (ਅਚਲ ਖੰਡ_q8_0 *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ਵਾਪਸ ਸੱਚ;
#ਜਾਂਜੇ defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        ਜੇ (n < 8 && n != 4)
           ਵਾਪਸ ਝੂਠ;
        ਜੇ (m < 8 && m != 4)
           ਵਾਪਸ ਝੂਠ;
        tinyBLAS_Q0_PPC<ਖੰਡ_q8_0> tb{
            k, (ਅਚਲ ਖੰਡ_q8_0 *)A, lda,
            (ਅਚਲ ਖੰਡ_q8_0 *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ਵਾਪਸ ਸੱਚ;
#ਨਹੀਂਤਾਂ
        ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
    }

    ਹਾਲਤ GGML_TYPE_Q4_0: {
        ਜੇ (Btype != GGML_TYPE_Q8_0)
            ਵਾਪਸ ਝੂਠ;
#ਜੇ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ਖੰਡ_q4_0, ਖੰਡ_q8_0, ਦਸ਼ਮਲਵ> tb{
            k, (ਅਚਲ ਖੰਡ_q4_0 *)A, lda,
            (ਅਚਲ ਖੰਡ_q8_0 *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ਵਾਪਸ ਸੱਚ;
#ਜਾਂਜੇ defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<ਖੰਡ_q4_0> tb{
            k, (ਅਚਲ ਖੰਡ_q4_0 *)A, lda,
            (ਅਚਲ ਖੰਡ_q8_0 *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ਵਾਪਸ ਸੱਚ;
#ਜਾਂਜੇ defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        ਜੇ (n < 8 && n != 4)
           ਵਾਪਸ ਝੂਠ;
        ਜੇ (m < 8 && m != 4)
           ਵਾਪਸ ਝੂਠ;
        tinyBLAS_Q0_PPC<ਖੰਡ_q4_0> tb{
            k, (ਅਚਲ ਖੰਡ_q4_0 *)A, lda,
            (ਅਚਲ ਖੰਡ_q8_0 *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ਵਾਪਸ ਸੱਚ;
#ਨਹੀਂਤਾਂ
        ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
    }

    ਹਾਲਤ GGML_TYPE_Q5_0: {
        ਜੇ (Btype != GGML_TYPE_Q8_0)
            ਵਾਪਸ ਝੂਠ;
#ਜੇ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ਖੰਡ_q5_0, ਖੰਡ_q8_0, ਦਸ਼ਮਲਵ> tb{
            k, (ਅਚਲ ਖੰਡ_q5_0 *)A, lda,
            (ਅਚਲ ਖੰਡ_q8_0 *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ਵਾਪਸ ਸੱਚ;
#ਨਹੀਂਤਾਂ
        ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
    }

    ਹਾਲਤ GGML_TYPE_IQ4_NL: {
        ਜੇ (Btype != GGML_TYPE_Q8_0)
            ਵਾਪਸ ਝੂਠ;
#ਜੇ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ਖੰਡ_iq4_nl, ਖੰਡ_q8_0, ਦਸ਼ਮਲਵ> tb{
            k, (ਅਚਲ ਖੰਡ_iq4_nl *)A, lda,
            (ਅਚਲ ਖੰਡ_q8_0 *)B, ldb,
            (ਦਸ਼ਮਲਵ *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        ਵਾਪਸ ਸੱਚ;
#ਨਹੀਂਤਾਂ
        ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
    }

    ਮੂਲ:
        ਵਾਪਸ ਝੂਠ;
    }

    (ਖਾਲੀ)params;
    (ਖਾਲੀ)m;
    (ਖਾਲੀ)n;
    (ਖਾਲੀ)k;
    (ਖਾਲੀ)A;
    (ਖਾਲੀ)lda;
    (ਖਾਲੀ)B;
    (ਖਾਲੀ)ldb;
    (ਖਾਲੀ)C;
    (ਖਾਲੀ)ldc;
    (ਖਾਲੀ)Atype;
    (ਖਾਲੀ)Btype;
    (ਖਾਲੀ)Ctype;
}
