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

#ఒకవేళ defined(__GNUC__)
#సూచన GCC diagnostic ignored "-Wpedantic"
#సూచన GCC diagnostic ignored "-Wignored-attributes"
#అయితే_ముగింపు

#అన్తర్భావ "sgemm.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-cpu-impl.h"
#అన్తర్భావ "ggml-quants.h"
#అన్తర్భావ "simd-mappings.h"

#అన్తర్భావ <array>
#అన్తర్భావ <type_traits>

#నిర్వచితమైతే _MSC_VER
#నిర్వచించు NOINLINE __declspec(noinline)
#లేకపోతే
#నిర్వచించు NOINLINE __attribute__((__noinline__))
#అయితే_ముగింపు

#ఒకవేళ defined(__ARM_NEON) || defined(__AVX512F__) || defined(__VXE__) || defined(__VXE2__)
#నిర్వచించు సదిశ_REGISTERS 32
#లేకపోతే
#నిర్వచించు సదిశ_REGISTERS 16
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_v_intrinsic)
#నిర్వచించు LMUL 4
#అయితే_ముగింపు

#నిర్వచించు MM256_సమితి_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

నామాకాశ {

అంతఃవరుస దశాంశం unhalf(ggml_fp16_t d) {
    తిరిగి GGML_CPU_FP16_గమ్యం_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#ఒకవేళ defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
అంతఃవరుస __m128 కూడు_2(__m128 x, __m128 y) { తిరిగి mm_కూడు_ps(x, y); }
అంతఃవరుస __m128 sub(__m128 x, __m128 y) { తిరిగి _mm_sub_ps(x, y); }
అంతఃవరుస __m128 mul(__m128 x, __m128 y) { తిరిగి _mm_mul_ps(x, y); }
#అయితే_ముగింపు  // __SSE__

#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
అంతఃవరుస __m256 కూడు_2(__m256 x, __m256 y) { తిరిగి mm256_కూడు_ps(x, y); }
అంతఃవరుస __m256 sub(__m256 x, __m256 y) { తిరిగి _mm256_sub_ps(x, y); }
అంతఃవరుస __m256 mul(__m256 x, __m256 y) { తిరిగి _mm256_mul_ps(x, y); }
#అయితే_ముగింపు // __AVX__

#ఒకవేళ defined(__AVX512F__)
అంతఃవరుస __m512 కూడు_2(__m512 x, __m512 y) { తిరిగి mm512_కూడు_ps(x, y); }
అంతఃవరుస __m512 sub(__m512 x, __m512 y) { తిరిగి _mm512_sub_ps(x, y); }
అంతఃవరుస __m512 mul(__m512 x, __m512 y) { తిరిగి _mm512_mul_ps(x, y); }
#అయితే_ముగింపు // __AVX512F__

#ఒకవేళ defined(__ARM_NEON)
అంతఃవరుస float32x4_t కూడు_2(float32x4_t x, float32x4_t y) { తిరిగి vaddq_f32(x, y); }
అంతఃవరుస float32x4_t sub(float32x4_t x, float32x4_t y) { తిరిగి vsubq_f32(x, y); }
అంతఃవరుస float32x4_t mul(float32x4_t x, float32x4_t y) { తిరిగి vmulq_f32(x, y); }
#అయితే_ముగింపు // __ARM_NEON

#ఒకవేళ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
అంతఃవరుస float16x8_t కూడు_2(float16x8_t x, float16x8_t y) { తిరిగి vaddq_f16(x, y); }
అంతఃవరుస float16x8_t sub(float16x8_t x, float16x8_t y) { తిరిగి vsubq_f16(x, y); }
అంతఃవరుస float16x8_t mul(float16x8_t x, float16x8_t y) { తిరిగి vmulq_f16(x, y); }
#అయితే_ముగింపు // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
అంతఃవరుస float32x4_t కూడు_2(float32x4_t x, float32x4_t y) { తిరిగి vec_కూడు(x, y); }
అంతఃవరుస float32x4_t sub(float32x4_t x, float32x4_t y) { తిరిగి vec_sub(x, y); }
అంతఃవరుస float32x4_t mul(float32x4_t x, float32x4_t y) { తిరిగి vec_mul(x, y); }
#అయితే_ముగింపు

#ఒకవేళ defined(__MMA__)
రకనిర్వచనం vector గుర్తులేని అక్షరం vec_t;
రకనిర్వచనం __vector_quad acc_t;
#అయితే_ముగింపు
////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
మూస <రకం_పేరు T, రకం_పేరు U>
అంతఃవరుస U madd(T a, T b, U c) {
    తిరిగి కూడు_2(mul(a, b), c);
}

#ఒకవేళ defined(__FMA__)
#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
మూస <>
అంతఃవరుస __m256 madd(__m256 a, __m256 b, __m256 c) {
    తిరిగి _mm256_fmadd_ps(a, b, c);
}
#అయితే_ముగింపు
#ఒకవేళ defined(__AVX512F__)
మూస <>
అంతఃవరుస __m512 madd(__m512 a, __m512 b, __m512 c) {
    తిరిగి _mm512_fmadd_ps(a, b, c);
}
#అయితే_ముగింపు
#ఒకవేళ defined(__AVX512BF16__)
మూస <>
అంతఃవరుస __m512 madd(__m512bh a, __m512bh b, __m512 c) {
    తిరిగి _mm512_dpbf16_ps(c, a, b);
}
మూస <>
అంతఃవరుస __m256 madd(__m256bh a, __m256bh b, __m256 c) {
    తిరిగి _mm256_dpbf16_ps(c, a, b);
}
#అయితే_ముగింపు
#అయితే_ముగింపు

#ఒకవేళ defined(__ARM_FEATURE_FMA)
మూస <>
అంతఃవరుస float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    తిరిగి vfmaq_f32(c, b, a);
}
#ఒకవేళ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
మూస <>
అంతఃవరుస float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    తిరిగి vfmaq_f16(c, b, a);
}
#అయితే_ముగింపు
#అయితే_ముగింపు

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
మూస <>
అంతఃవరుస float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    తిరిగి vec_madd(a, b, c);
}
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_v_intrinsic)
మూస <> అంతఃవరుస vfloat32m1_t madd(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t c) {
    తిరిగి __riscv_vfmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
మూస <> అంతఃవరుస vfloat32m2_t madd(vfloat32m2_t a, vfloat32m2_t b, vfloat32m2_t c) {
    తిరిగి __riscv_vfmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
మూస <> అంతఃవరుస vfloat32m4_t madd(vfloat32m4_t a, vfloat32m4_t b, vfloat32m4_t c) {
    తిరిగి __riscv_vfmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
మూస <> అంతఃవరుస vfloat32m8_t madd(vfloat32m8_t a, vfloat32m8_t b, vfloat32m8_t c) {
    తిరిగి __riscv_vfmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_zvfh)
మూస <> అంతఃవరుస vfloat32m1_t madd(vfloat16mf2_t a, vfloat16mf2_t b, vfloat32m1_t c) {
    తిరిగి __riscv_vfwmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
మూస <> అంతఃవరుస vfloat32m2_t madd(vfloat16m1_t a, vfloat16m1_t b, vfloat32m2_t c) {
    తిరిగి __riscv_vfwmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
మూస <> అంతఃవరుస vfloat32m4_t madd(vfloat16m2_t a, vfloat16m2_t b, vfloat32m4_t c) {
    తిరిగి __riscv_vfwmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
మూస <> అంతఃవరుస vfloat32m8_t madd(vfloat16m4_t a, vfloat16m4_t b, vfloat32m8_t c) {
    తిరిగి __riscv_vfwmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_zvfbfwma)
మూస <> అంతఃవరుస vfloat32m1_t madd(vbfloat16mf2_t a, vbfloat16mf2_t b, vfloat32m1_t c) {
    తిరిగి __riscv_vfwmaccbf16_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
మూస <> అంతఃవరుస vfloat32m2_t madd(vbfloat16m1_t a, vbfloat16m1_t b, vfloat32m2_t c) {
    తిరిగి __riscv_vfwmaccbf16_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
మూస <> అంతఃవరుస vfloat32m4_t madd(vbfloat16m2_t a, vbfloat16m2_t b, vfloat32m4_t c) {
    తిరిగి __riscv_vfwmaccbf16_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
మూస <> అంతఃవరుస vfloat32m8_t madd(vbfloat16m4_t a, vbfloat16m4_t b, vfloat32m8_t c) {
    తిరిగి __riscv_vfwmaccbf16_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#అయితే_ముగింపు

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#ఒకవేళ defined(__ARM_NEON) || defined(_M_ARM64) || defined(_M_ARM64EC)
అంతఃవరుస దశాంశం hsum(float32x4_t x) {
    తిరిగి vaddvq_f32(x);
}
#అయితే_ముగింపు // __ARM_NEON

#ఒకవేళ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
అంతఃవరుస దశాంశం hsum(float16x8_t x) {
    తిరిగి vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_దిగువ_f16(x)),
                                vcvt_f32_f16(vget_ఎగువ_f16(x))));
}
#అయితే_ముగింపు // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
అంతఃవరుస దశాంశం hsum(float32x4_t x) {
    float32x4_t tmp = x + vec_reve(x);
    తిరిగి tmp[0] + tmp[1];
}
#అయితే_ముగింపు

#ఒకవేళ defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
అంతఃవరుస దశాంశం hsum(__m128 x) {
#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = mm_కూడు_ps(x, _mm_movehl_ps(x, x));
    x = mm_కూడు_ss(x, _mm_movehdup_ps(x));
#లేకపోతే
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = mm_కూడు_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = mm_కూడు_ss(x, t);
#అయితే_ముగింపు
    తిరిగి _mm_cvtss_f32(x);
}
#అయితే_ముగింపు

#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
అంతఃవరుస దశాంశం hsum(__m256 x) {
    తిరిగి hsum(mm_కూడు_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#అయితే_ముగింపు // __AVX__

#ఒకవేళ defined(__AVX512F__)
అంతఃవరుస దశాంశం hsum(__m512 x) {
    తిరిగి mm512_reduce_కూడు_ps(x);
}
#అయితే_ముగింపు // __AVX512F__

#ఒకవేళ defined(__riscv_v_intrinsic)
అంతఃవరుస దశాంశం hsum(vfloat32m1_t x) {
    తిరిగి __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m1_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m1()));
}
అంతఃవరుస దశాంశం hsum(vfloat32m2_t x) {
    తిరిగి __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m2()));
}
అంతఃవరుస దశాంశం hsum(vfloat32m4_t x) {
    తిరిగి __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m4()));
}
అంతఃవరుస దశాంశం hsum(vfloat32m8_t x) {
    తిరిగి __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m8_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m8()));
}
#అయితే_ముగింపు

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

మూస <రకం_పేరు T, రకం_పేరు U> T load(స్థిరరాశి U *);

#ఒకవేళ defined(__ARM_NEON)
మూస <> అంతఃవరుస float32x4_t load(స్థిరరాశి దశాంశం *p) {
    తిరిగి vld1q_f32(p);
}
#ఒకవేళ !defined(_MSC_VER)
// FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
మూస <> అంతఃవరుస float16x8_t load(స్థిరరాశి ggml_fp16_t *p) {
    తిరిగి vld1q_f16((స్థిరరాశి float16_t *)p);
}
మూస <> అంతఃవరుస float32x4_t load(స్థిరరాశి ggml_fp16_t *p) {
    తిరిగి vcvt_f32_f16(vld1_f16((స్థిరరాశి float16_t *)p));
}
#అయితే_ముగింపు // _MSC_VER
#అయితే_ముగింపు // __ARM_NEON

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
మూస <> అంతఃవరుస float32x4_t load(స్థిరరాశి ggml_fp16_t * p) {
    దశాంశం tmp[4];

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_గమ్యం_FP32(p[i]);
    }

    తిరిగి vec_xl(0, (స్థిరరాశి దశాంశం *)(tmp));
}
మూస <> అంతఃవరుస float32x4_t load(స్థిరరాశి దశాంశం * p) {
    తిరిగి vec_xl(0, p);
}
#అయితే_ముగింపు

#ఒకవేళ defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
మూస <> అంతఃవరుస __m128 load(స్థిరరాశి దశాంశం *p) {
    తిరిగి _mm_loadu_ps(p);
}
#అయితే_ముగింపు  // __SSE__

#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
మూస <> అంతఃవరుస __m256 load(స్థిరరాశి దశాంశం *p) {
    తిరిగి _mm256_loadu_ps(p);
}
#అయితే_ముగింపు // __AVX__

#ఒకవేళ defined(__AVX2__) || defined(__AVX512F__)
మూస <> అంతఃవరుస __m256 load(స్థిరరాశి ggml_bf16_t *p) {
    తిరిగి _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((స్థిరరాశి __m128i *)p)), 16));
}
#అయితే_ముగింపు // __AVX2__

#ఒకవేళ defined(__F16C__)
మూస <> అంతఃవరుస __m256 load(స్థిరరాశి ggml_fp16_t *p) {
    తిరిగి _mm256_cvtph_ps(_mm_loadu_si128((స్థిరరాశి __m128i *)p));
}
#అయితే_ముగింపు // __F16C__

#ఒకవేళ defined(__AVX512F__)
మూస <> అంతఃవరుస __m512 load(స్థిరరాశి దశాంశం *p) {
    తిరిగి _mm512_loadu_ps(p);
}
మూస <> అంతఃవరుస __m512 load(స్థిరరాశి ggml_fp16_t *p) {
    తిరిగి _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)p));
}
మూస <> అంతఃవరుస __m512 load(స్థిరరాశి ggml_bf16_t *p) {
    తిరిగి _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((స్థిరరాశి __m256i *)p)), 16));
}
#అయితే_ముగింపు // __AVX512F__

#ఒకవేళ defined(__AVX512BF16__)
మూస <> అంతఃవరుస __m512bh load(స్థిరరాశి ggml_bf16_t *p) {
    తిరిగి (__m512bh)_mm512_loadu_ps((స్థిరరాశి దశాంశం *)p);
}
మూస <> అంతఃవరుస __m256bh load(స్థిరరాశి ggml_bf16_t *p) {
    తిరిగి (__m256bh)_mm256_loadu_ps((స్థిరరాశి దశాంశం *)p);
}
మూస <> అంతఃవరుస __m512bh load(స్థిరరాశి దశాంశం *p) {
    తిరిగి _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
మూస <> అంతఃవరుస __m256bh load(స్థిరరాశి దశాంశం *p) {
    తిరిగి _mm512_cvtneps_pbh(_mm512_loadu_ps(p));
}
#అయితే_ముగింపు

#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
మూస <రకం_పేరు T, రకం_పేరు U> T ఎక్కించు_partial(స్థిరరాశి U *, పూర్ణాంకం);
మూస <రకం_పేరు T> T ఎక్కించు_partial_u16(స్థిరరాశి శూన్య *, పూర్ణాంకం);

మూస <> అంతఃవరుస __m128i ఎక్కించు_partial_u16(స్థిరరాశి శూన్య *p, పూర్ణాంకం n) {
#ఒకవేళ defined(__AVX512BW__) && defined(__AVX512VL__)
    తిరిగి _mm_maskz_loadu_epi16((1u << n) - 1, p);
#లేకపోతే
    స్థిరరాశి __m128i సూచీ = _mm_setr_epi32(0, 1, 2, 3);
    స్థిరరాశి __m128i జతలు = _mm_set1_epi32(n / 2);
    __m128i v = _mm_castps_si128(_mm_maskload_ps((స్థిరరాశి దశాంశం *)p, _mm_cmpgt_epi32(జతలు, సూచీ)));
    ఒకవేళ (n & 1) {
        uint16_t చివరి;
        జ్ఞాపకం_నకలు(&చివరి, (స్థిరరాశి అక్షరం *)p + 2*(n - 1), పరిమాణం(చివరి));
        v = _mm_or_si128(v, _mm_and_si128(_mm_cmpeq_epi32(జతలు, సూచీ), _mm_set1_epi32(చివరి)));
    }
    తిరిగి v;
#అయితే_ముగింపు
}

మూస <> అంతఃవరుస __m256 ఎక్కించు_partial(స్థిరరాశి దశాంశం *p, పూర్ణాంకం n) {
    స్థిరరాశి __m256 సూచీ = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    తిరిగి _mm256_maskload_ps(p, _mm256_castps_si256(_mm256_cmp_ps(సూచీ, _mm256_set1_ps(n), _CMP_LT_OQ)));
}

#ఒకవేళ defined(__F16C__)
మూస <> అంతఃవరుస __m256 ఎక్కించు_partial(స్థిరరాశి ggml_fp16_t *p, పూర్ణాంకం n) {
    తిరిగి _mm256_cvtph_ps(ఎక్కించు_partial_u16<__m128i>(p, n));
}
#అయితే_ముగింపు

#ఒకవేళ defined(__AVX2__) || defined(__AVX512F__)
మూస <> అంతఃవరుస __m256 ఎక్కించు_partial(స్థిరరాశి ggml_bf16_t *p, పూర్ణాంకం n) {
    తిరిగి _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(ఎక్కించు_partial_u16<__m128i>(p, n)), 16));
}
#అయితే_ముగింపు

#ఒకవేళ defined(__AVX512F__)
మూస <> అంతఃవరుస __m256i ఎక్కించు_partial_u16(స్థిరరాశి శూన్య *p, పూర్ణాంకం n) {
#ఒకవేళ defined(__AVX512BW__) && defined(__AVX512VL__)
    తిరిగి _mm256_maskz_loadu_epi16((1u << n) - 1, p);
#లేకపోతే
    స్థిరరాశి __m256i సూచీ = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    స్థిరరాశి __m256i జతలు = _mm256_set1_epi32(n / 2);
    __m256i v = _mm256_maskload_epi32((స్థిరరాశి పూర్ణాంకం *)p, _mm256_cmpgt_epi32(జతలు, సూచీ));
    ఒకవేళ (n & 1) {
        uint16_t చివరి;
        జ్ఞాపకం_నకలు(&చివరి, (స్థిరరాశి అక్షరం *)p + 2*(n - 1), పరిమాణం(చివరి));
        v = _mm256_or_si256(v, _mm256_and_si256(_mm256_cmpeq_epi32(జతలు, సూచీ), _mm256_set1_epi32(చివరి)));
    }
    తిరిగి v;
#అయితే_ముగింపు
}

మూస <> అంతఃవరుస __m512 ఎక్కించు_partial(స్థిరరాశి దశాంశం *p, పూర్ణాంకం n) {
    తిరిగి _mm512_maskz_loadu_ps((1u << n) - 1, p);
}

మూస <> అంతఃవరుస __m512 ఎక్కించు_partial(స్థిరరాశి ggml_fp16_t *p, పూర్ణాంకం n) {
    తిరిగి _mm512_cvtph_ps(ఎక్కించు_partial_u16<__m256i>(p, n));
}

మూస <> అంతఃవరుస __m512 ఎక్కించు_partial(స్థిరరాశి ggml_bf16_t *p, పూర్ణాంకం n) {
    తిరిగి _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(ఎక్కించు_partial_u16<__m256i>(p, n)), 16));
}
#అయితే_ముగింపు

#ఒకవేళ defined(__AVX512BF16__)
మూస <> అంతఃవరుస __m512bh ఎక్కించు_partial(స్థిరరాశి ggml_bf16_t *p, పూర్ణాంకం n) {
    తిరిగి (__m512bh) _mm512_maskz_loadu_epi16((uint64_t(1) << n) - 1, p);
}
#అయితే_ముగింపు
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_v_intrinsic)
మూస <> అంతఃవరుస vfloat32m1_t load(స్థిరరాశి దశాంశం *p) {
    తిరిగి __riscv_vle32_v_f32m1(p, __riscv_vsetvlmax_e32m1());
}
మూస <> అంతఃవరుస vfloat32m2_t load(స్థిరరాశి దశాంశం *p) {
    తిరిగి __riscv_vle32_v_f32m2(p, __riscv_vsetvlmax_e32m2());
}
మూస <> అంతఃవరుస vfloat32m4_t load(స్థిరరాశి దశాంశం *p) {
    తిరిగి __riscv_vle32_v_f32m4(p, __riscv_vsetvlmax_e32m4());
}
మూస <> అంతఃవరుస vfloat32m8_t load(స్థిరరాశి దశాంశం *p) {
    తిరిగి __riscv_vle32_v_f32m8(p, __riscv_vsetvlmax_e32m8());
}
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_zvfh)
మూస <> అంతఃవరుస vfloat16mf2_t load(స్థిరరాశి ggml_fp16_t *p) {
    తిరిగి __riscv_vle16_v_f16mf2(reinterpret_cast<స్థిరరాశి _Float16 *>(p), __riscv_vsetvlmax_e16mf2());
}
మూస <> అంతఃవరుస vfloat16m1_t load(స్థిరరాశి ggml_fp16_t *p) {
    తిరిగి __riscv_vle16_v_f16m1(reinterpret_cast<స్థిరరాశి _Float16 *>(p), __riscv_vsetvlmax_e16m1());
}
మూస <> అంతఃవరుస vfloat16m2_t load(స్థిరరాశి ggml_fp16_t *p) {
    తిరిగి __riscv_vle16_v_f16m2(reinterpret_cast<స్థిరరాశి _Float16 *>(p), __riscv_vsetvlmax_e16m2());
}
మూస <> అంతఃవరుస vfloat16m4_t load(స్థిరరాశి ggml_fp16_t *p) {
    తిరిగి __riscv_vle16_v_f16m4(reinterpret_cast<స్థిరరాశి _Float16 *>(p), __riscv_vsetvlmax_e16m4());
}
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_zvfbfwma)
మూస <> అంతఃవరుస vbfloat16mf2_t load(స్థిరరాశి ggml_bf16_t *p) {
    తిరిగి __riscv_vle16_v_bf16mf2(reinterpret_cast<స్థిరరాశి __bf16*>(p), __riscv_vsetvlmax_e16mf2());
}
మూస <> అంతఃవరుస vbfloat16m1_t load(స్థిరరాశి ggml_bf16_t *p) {
    తిరిగి __riscv_vle16_v_bf16m1(reinterpret_cast<స్థిరరాశి __bf16*>(p), __riscv_vsetvlmax_e16m1());
}
మూస <> అంతఃవరుస vbfloat16m2_t load(స్థిరరాశి ggml_bf16_t *p) {
    తిరిగి __riscv_vle16_v_bf16m2(reinterpret_cast<స్థిరరాశి __bf16*>(p), __riscv_vsetvlmax_e16m2());
}
మూస <> అంతఃవరుస vbfloat16m4_t load(స్థిరరాశి ggml_bf16_t *p) {
    తిరిగి __riscv_vle16_v_bf16m4(reinterpret_cast<స్థిరరాశి __bf16*>(p), __riscv_vsetvlmax_e16m4());
}
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_v_intrinsic)
మూస <రకం_పేరు T> T సమితి_zero();

మూస <> అంతఃవరుస vfloat32m1_t సమితి_zero() {
    తిరిగి __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
}
మూస <> అంతఃవరుస vfloat32m2_t సమితి_zero() {
    తిరిగి __riscv_vfmv_v_f_f32m2(0, __riscv_vsetvlmax_e32m2());
}
మూస <> అంతఃవరుస vfloat32m4_t సమితి_zero() {
    తిరిగి __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
}
మూస <> అంతఃవరుస vfloat32m8_t సమితి_zero() {
    తిరిగి __riscv_vfmv_v_f_f32m8(0, __riscv_vsetvlmax_e32m8());
}
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_v_intrinsic)
మూస <రకం_పేరు T> size_t vlmax() {
    ఒకవేళ constexpr (std::is_same_v<T, vfloat32m1_t>) { తిరిగి  __riscv_vsetvlmax_e32m1(); }
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vfloat32m2_t>) { తిరిగి  __riscv_vsetvlmax_e32m2(); }
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vfloat32m4_t>) { తిరిగి  __riscv_vsetvlmax_e32m4(); }
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vfloat32m8_t>) { తిరిగి  __riscv_vsetvlmax_e32m8(); }
    #ఒకవేళ defined (__riscv_zvfh)
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vfloat16mf2_t>) { తిరిగి  __riscv_vsetvlmax_e16mf2(); }
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vfloat16m1_t>) { తిరిగి  __riscv_vsetvlmax_e16m1(); }
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vfloat16m2_t>) { తిరిగి  __riscv_vsetvlmax_e16m2(); }
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vfloat16m4_t>) { తిరిగి  __riscv_vsetvlmax_e16m4(); }
    #అయితే_ముగింపు
    #ఒకవేళ defined (__riscv_zvfbfwma)
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vbfloat16mf2_t>) { తిరిగి  __riscv_vsetvlmax_e16mf2(); }
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vbfloat16m1_t>) { తిరిగి  __riscv_vsetvlmax_e16m1(); }
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vbfloat16m2_t>) { తిరిగి  __riscv_vsetvlmax_e16m2(); }
    లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, vbfloat16m4_t>) { తిరిగి  __riscv_vsetvlmax_e16m4(); }
    #అయితే_ముగింపు
    తిరిగి 0;
}
#అయితే_ముగింపు

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

మూస <పూర్ణాంకం M>
స్థిర అంతఃవరుస int64_t ఖండం_కొలత_2(size_t m) {
    స్థిరరాశి int64_t NB_BLOC_M = (m + M - 1) / M;
    తిరిగి (m % NB_BLOC_M == 0) ? m / NB_BLOC_M : (m / NB_BLOC_M) + 1;
}

స్థిర constexpr అంతఃవరుస int64_t BLOC_POS(int64_t ib, int64_t ibN, int64_t bloc_కొలత) {
    తిరిగి ib < ibN ? ib * bloc_కొలత : ibN * bloc_కొలత + (ib - ibN) * (bloc_కొలత - 1);
}

మూస <పూర్ణాంకం KN, రకం_పేరు D, రకం_పేరు V, రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC>
తరగతి tinyBLAS {
  బహిరంగ:
    tinyBLAS(స్థిరరాశి ggml_compute_params * params, int64_t k,
             స్థిరరాశి TA *A, int64_t lda,
             స్థిరరాశి TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    తార్కిక matmul(int64_t m, int64_t n) {
#ఒకవేళ !defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
        ఒకవేళ (k % KN != 0)
            తిరిగి అబద్ధం;
#అయితే_ముగింపు
        // compute RM for only need tile with size RM&RM-1
#ఒకవేళ సదిశ_REGISTERS == 32
        ఒకవేళ (m % 16 == 0 && (m/16 >= params->nth)) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<6>(n);
            mnpack<4, 6, 4>(m, n, కొలత_N, 12);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 8 == 0 ) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<6>(n);
            mnpack<4, 6, 2>(m, n, కొలత_N, 12);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 4 == 0) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<6>(n);
            mnpack<4, 6, 1>(m, n, కొలత_N, 12);
            తిరిగి నిజం;
        }
#లేకపోతే  // VECTOR_REGISTERS == 16
        ఒకవేళ (m % 16 == 0 && (m/16 >= params->nth)) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<3>(n);
            mnpack<4, 3, 4>(m, n, కొలత_N, 24);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 8 == 0 ) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<3>(n);
            mnpack<4, 3, 2>(m, n, కొలత_N, 24);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 4 == 0) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<3>(n);
            mnpack<4, 3, 1>(m, n, కొలత_N, 24);
            తిరిగి నిజం;
        }
#అయితే_ముగింపు
        తిరిగి అబద్ధం;
    }

  అంతరంగ:
    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN, పూర్ణాంకం BM>
    అంతఃవరుస శూన్య mnpack(int64_t m, int64_t n, int64_t కొలత_N, int64_t BN) {
        ఒకవేళ (కొలత_N == RN) {
            తిరిగి gemm<RM, RN, BM>(m, n, BN);
        }
        ఒకవేళ constexpr (RN > 1) {
            తిరిగి mnpack<RM, RN-1, BM>(m, n, కొలత_N, BN);
        } లేకపోతే {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (పూర్ణాంకం)కొలత_N);
            GGML_ASSERT(అబద్ధం); // we have miss something.
        }
    }

    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN>
    అంతఃవరుస శూన్య gemm_bloc(int64_t ii, int64_t jj) {
        D Cv[RN][RM] = {};
        కొరకు (int64_t l = 0; l + KN <= k; l += KN) {
            // help compiler for op order.
            ఒకవేళ constexpr (RM <= RN) {
                V Av[RM];
                కొరకు (int64_t i = 0; i < RM; ++i) {
                    Av[i] = load<V>(A + lda * (ii + i) + l);
                }
                కొరకు (int64_t j = 0; j < RN; ++j) {
                    V Bv = load<V>(B + ldb * (jj + j) + l);
                    కొరకు (int64_t i = 0; i < RM; ++i) {
                        Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                    }
                }
            } లేకపోతే {
                V Bv[RN];
                కొరకు (int64_t j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(B + ldb * (jj + j) + l);
                }
                కొరకు (int64_t i = 0; i < RM; ++i) {
                    V Av = load<V>(A + lda * (ii + i) + l);
                    కొరకు (int64_t j = 0; j < RN; ++j) {
                        Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
                    }
                }
            }
        }
#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        స్థిరరాశి int64_t rem = k % KN;
        ఒకవేళ (rem != 0) {
            V Av[RM];
            కొరకు (int64_t i = 0; i < RM; ++i) {
                Av[i] = ఎక్కించు_partial<V>(A + lda * (ii + i) + k - rem, rem);
            }
            కొరకు (int64_t j = 0; j < RN; ++j) {
                V Bv = ఎక్కించు_partial<V>(B + ldb * (jj + j) + k - rem, rem);
                కొరకు (int64_t i = 0; i < RM; ++i) {
                    Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                }
            }
        }
#అయితే_ముగింపు
        కొరకు (int64_t j = 0; j < RN; ++j)
            కొరకు (int64_t i = 0; i < RM; ++i)
                C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
    }

    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN, పూర్ణాంకం BM>
    NOINLINE శూన్య gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        స్థిరరాశి int64_t ytiles = m / (RM * BM);
        స్థిరరాశి int64_t xtiles = (n + RN -1) / RN;
        స్థిరరాశి int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        స్థిరరాశి int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        స్థిరరాశి int64_t కొలత_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        స్థిరరాశి int64_t jj_BN = (NB_BN - (NB_BN * కొలత_BN - xtiles));
        స్థిరరాశి int64_t nb_పని = ytiles * NB_BN;

        ఒకవేళ (params->ith == 0) {
            GGML_ASSERT( jj_BN * కొలత_BN + (NB_BN - jj_BN) * (కొలత_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_సమితి(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t పని = params->ith;
        అయితే (పని < nb_పని) {
            స్థిరరాశి int64_t ii = (పని % ytiles) * RM * BM;
            స్థిరరాశి int64_t jb =  పని / ytiles;
            స్థిరరాశి int64_t jr0 = BLOC_POS(jb  , jj_BN, కొలత_BN);
            స్థిరరాశి int64_t jrN = BLOC_POS(jb+1, jj_BN, కొలత_BN);

            స్థిరరాశి int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            స్థిరరాశి int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            స్థిరరాశి int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            కొరకు (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                కొరకు (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                ఒకవేళ constexpr (RN > 1) {
                    కొరకు (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            పని = ggml_threadpool_chunk_కూడు(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        తిరిగి;
    }

    స్థిరరాశి ggml_compute_params * params;
    స్థిరరాశి TA *స్థిరరాశి A;
    స్థిరరాశి TB *స్థిరరాశి B;
    TC *స్థిరరాశి C;
    స్థిరరాశి int64_t k;
    స్థిరరాశి int64_t lda;
    స్థిరరాశి int64_t ldb;
    స్థిరరాశి int64_t ldc;
};

#ఒకవేళ defined(__riscv_v_intrinsic)
మూస <రకం_పేరు D, రకం_పేరు V, రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC>
తరగతి tinyBLAS_RVV {
  బహిరంగ:
    tinyBLAS_RVV(స్థిరరాశి ggml_compute_params * params, int64_t k,
             స్థిరరాశి TA *A, int64_t lda,
             స్థిరరాశి TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    తార్కిక matmul(int64_t m, int64_t n) {
        ఒకవేళ (k % vlmax<V>() != 0) {
            తిరిగి అబద్ధం;
        }

#ఒకవేళ LMUL == 1
        ఒకవేళ (m % 16 == 0 && (m/16 >= params->nth)) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<6>(n);
            mnpack<4, 6, 4>(m, n, కొలత_N, 12);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 8 == 0 ) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<6>(n);
            mnpack<4, 6, 2>(m, n, కొలత_N, 12);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 4 == 0) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<6>(n);
            mnpack<4, 6, 1>(m, n, కొలత_N, 12);
            తిరిగి నిజం;
        }
#లేదావేళ LMUL == 2
        ఒకవేళ (m % 16 == 0 && (m/16 >= params->nth)) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<3>(n);
            mnpack<4, 3, 4>(m, n, కొలత_N, 24);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 8 == 0 ) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<3>(n);
            mnpack<4, 3, 2>(m, n, కొలత_N, 24);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 4 == 0) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<3>(n);
            mnpack<4, 3, 1>(m, n, కొలత_N, 24);
            తిరిగి నిజం;
        }
#లేకపోతే // LMUL = 4
        ఒకవేళ (m % 16 == 0 && (m/16 >= params->nth)) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<2>(n);
            mnpack<2, 2, 8>(m, n, కొలత_N, 36);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 8 == 0 ) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<2>(n);
            mnpack<2, 2, 4>(m, n, కొలత_N, 36);
            తిరిగి నిజం;
        }
        ఒకవేళ (m % 4 == 0) {
            స్థిరరాశి int64_t కొలత_N = ఖండం_కొలత_2<2>(n);
            mnpack<2, 2, 2>(m, n, కొలత_N, 36);
            తిరిగి నిజం;
        }
#అయితే_ముగింపు
        తిరిగి అబద్ధం;
    }

  అంతరంగ:
    మూస<పూర్ణాంకం RM, పూర్ణాంకం RN, పూర్ణాంకం BM>
    అంతఃవరుస శూన్య mnpack(int64_t m, int64_t n, int64_t కొలత_N, int64_t BN) {
        ఒకవేళ (కొలత_N == RN) {
            తిరిగి gemm<RM, RN, BM>(m, n, BN);
        }
        ఒకవేళ constexpr (RN > 1) {
            తిరిగి mnpack<RM, RN-1, BM>(m, n, కొలత_N, BN);
        } లేకపోతే {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (పూర్ణాంకం)కొలత_N);
            GGML_ASSERT(అబద్ధం); // we have miss something.
        }
    }

    అంతఃవరుస శూన్య gemm_bloc_4x6(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = సమితి_zero<D>();
        D Cv01 = సమితి_zero<D>();
        D Cv02 = సమితి_zero<D>();
        D Cv03 = సమితి_zero<D>();
        D Cv10 = సమితి_zero<D>();
        D Cv11 = సమితి_zero<D>();
        D Cv12 = సమితి_zero<D>();
        D Cv13 = సమితి_zero<D>();
        D Cv20 = సమితి_zero<D>();
        D Cv21 = సమితి_zero<D>();
        D Cv22 = సమితి_zero<D>();
        D Cv23 = సమితి_zero<D>();
        D Cv30 = సమితి_zero<D>();
        D Cv31 = సమితి_zero<D>();
        D Cv32 = సమితి_zero<D>();
        D Cv33 = సమితి_zero<D>();
        D Cv40 = సమితి_zero<D>();
        D Cv41 = సమితి_zero<D>();
        D Cv42 = సమితి_zero<D>();
        D Cv43 = సమితి_zero<D>();
        D Cv50 = సమితి_zero<D>();
        D Cv51 = సమితి_zero<D>();
        D Cv52 = సమితి_zero<D>();
        D Cv53 = సమితి_zero<D>();

        కొరకు (int64_t l = 0; l < k; l += vl) {
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

    అంతఃవరుస శూన్య gemm_bloc_4x5(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = సమితి_zero<D>();
        D Cv01 = సమితి_zero<D>();
        D Cv02 = సమితి_zero<D>();
        D Cv03 = సమితి_zero<D>();
        D Cv10 = సమితి_zero<D>();
        D Cv11 = సమితి_zero<D>();
        D Cv12 = సమితి_zero<D>();
        D Cv13 = సమితి_zero<D>();
        D Cv20 = సమితి_zero<D>();
        D Cv21 = సమితి_zero<D>();
        D Cv22 = సమితి_zero<D>();
        D Cv23 = సమితి_zero<D>();
        D Cv30 = సమితి_zero<D>();
        D Cv31 = సమితి_zero<D>();
        D Cv32 = సమితి_zero<D>();
        D Cv33 = సమితి_zero<D>();
        D Cv40 = సమితి_zero<D>();
        D Cv41 = సమితి_zero<D>();
        D Cv42 = సమితి_zero<D>();
        D Cv43 = సమితి_zero<D>();

        కొరకు (int64_t l = 0; l < k; l += vl) {
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

    అంతఃవరుస శూన్య gemm_bloc_4x4(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = సమితి_zero<D>();
        D Cv01 = సమితి_zero<D>();
        D Cv02 = సమితి_zero<D>();
        D Cv03 = సమితి_zero<D>();
        D Cv10 = సమితి_zero<D>();
        D Cv11 = సమితి_zero<D>();
        D Cv12 = సమితి_zero<D>();
        D Cv13 = సమితి_zero<D>();
        D Cv20 = సమితి_zero<D>();
        D Cv21 = సమితి_zero<D>();
        D Cv22 = సమితి_zero<D>();
        D Cv23 = సమితి_zero<D>();
        D Cv30 = సమితి_zero<D>();
        D Cv31 = సమితి_zero<D>();
        D Cv32 = సమితి_zero<D>();
        D Cv33 = సమితి_zero<D>();

        కొరకు (int64_t l = 0; l < k; l += vl) {
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

    అంతఃవరుస శూన్య gemm_bloc_4x3(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = సమితి_zero<D>();
        D Cv01 = సమితి_zero<D>();
        D Cv02 = సమితి_zero<D>();
        D Cv03 = సమితి_zero<D>();
        D Cv10 = సమితి_zero<D>();
        D Cv11 = సమితి_zero<D>();
        D Cv12 = సమితి_zero<D>();
        D Cv13 = సమితి_zero<D>();
        D Cv20 = సమితి_zero<D>();
        D Cv21 = సమితి_zero<D>();
        D Cv22 = సమితి_zero<D>();
        D Cv23 = సమితి_zero<D>();

        కొరకు (int64_t l = 0; l < k; l += vl) {
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

    అంతఃవరుస శూన్య gemm_bloc_4x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = సమితి_zero<D>();
        D Cv01 = సమితి_zero<D>();
        D Cv02 = సమితి_zero<D>();
        D Cv03 = సమితి_zero<D>();
        D Cv10 = సమితి_zero<D>();
        D Cv11 = సమితి_zero<D>();
        D Cv12 = సమితి_zero<D>();
        D Cv13 = సమితి_zero<D>();

        కొరకు (int64_t l = 0; l < k; l += vl) {
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

    అంతఃవరుస శూన్య gemm_bloc_4x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = సమితి_zero<D>();
        D Cv01 = సమితి_zero<D>();
        D Cv02 = సమితి_zero<D>();
        D Cv03 = సమితి_zero<D>();

        కొరకు (int64_t l = 0; l < k; l += vl) {
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

    అంతఃవరుస శూన్య gemm_bloc_2x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = సమితి_zero<D>();
        D Cv01 = సమితి_zero<D>();
        D Cv10 = సమితి_zero<D>();
        D Cv11 = సమితి_zero<D>();

        కొరకు (int64_t l = 0; l < k; l += vl) {
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

    అంతఃవరుస శూన్య gemm_bloc_2x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = సమితి_zero<D>();
        D Cv01 = సమితి_zero<D>();

        కొరకు (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
    }

    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN>
    అంతఃవరుస శూన్య gemm_bloc(int64_t ii, int64_t jj) {
        ఒకవేళ constexpr (RM == 4) {
            ఒకవేళ constexpr (RN == 6) { తిరిగి gemm_bloc_4x6(ii, jj); }
            ఒకవేళ constexpr (RN == 5) { తిరిగి gemm_bloc_4x5(ii, jj); }
            ఒకవేళ constexpr (RN == 4) { తిరిగి gemm_bloc_4x4(ii, jj); }
            ఒకవేళ constexpr (RN == 3) { తిరిగి gemm_bloc_4x3(ii, jj); }
            ఒకవేళ constexpr (RN == 2) { తిరిగి gemm_bloc_4x2(ii, jj); }
            ఒకవేళ constexpr (RN == 1) { తిరిగి gemm_bloc_4x1(ii, jj); }
        } లేకపోతే ఒకవేళ constexpr (RM == 2) {
            ఒకవేళ constexpr (RN == 2) { తిరిగి gemm_bloc_2x2(ii, jj); }
            ఒకవేళ constexpr (RN == 1) { తిరిగి gemm_bloc_2x1(ii, jj); }
        }
    }

    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN, పూర్ణాంకం BM>
    NOINLINE శూన్య gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        స్థిరరాశి int64_t ytiles = m / (RM * BM);
        స్థిరరాశి int64_t xtiles = (n + RN -1) / RN;
        స్థిరరాశి int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        స్థిరరాశి int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        స్థిరరాశి int64_t కొలత_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        స్థిరరాశి int64_t jj_BN = (NB_BN - (NB_BN * కొలత_BN - xtiles));
        స్థిరరాశి int64_t nb_పని = ytiles * NB_BN;

        ఒకవేళ (params->ith == 0) {
            GGML_ASSERT( jj_BN * కొలత_BN + (NB_BN - jj_BN) * (కొలత_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_సమితి(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t పని = params->ith;
        అయితే (పని < nb_పని) {
            స్థిరరాశి int64_t ii = (పని % ytiles) * RM * BM;
            స్థిరరాశి int64_t jb =  పని / ytiles;
            స్థిరరాశి int64_t jr0 = BLOC_POS(jb  , jj_BN, కొలత_BN);
            స్థిరరాశి int64_t jrN = BLOC_POS(jb+1, jj_BN, కొలత_BN);

            స్థిరరాశి int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            స్థిరరాశి int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            స్థిరరాశి int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            కొరకు (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                కొరకు (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                ఒకవేళ constexpr (RN > 1) {
                    కొరకు (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            పని = ggml_threadpool_chunk_కూడు(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        తిరిగి;
    }

    స్థిరరాశి ggml_compute_params * params;
    స్థిరరాశి TA *స్థిరరాశి A;
    స్థిరరాశి TB *స్థిరరాశి B;
    TC *స్థిరరాశి C;
    స్థిరరాశి int64_t k;
    స్థిరరాశి int64_t lda;
    స్థిరరాశి int64_t ldb;
    స్థిరరాశి int64_t ldc;
};
#అయితే_ముగింపు

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#ఒకవేళ defined(__ARM_FEATURE_DOTPROD)
మూస <రకం_పేరు TA>
తరగతి tinyBLAS_Q0_ARM {
  బహిరంగ:
    tinyBLAS_Q0_ARM(int64_t k,
                    స్థిరరాశి TA *A, int64_t lda,
                    స్థిరరాశి ఖండం_q8_0 *B, int64_t ldb,
                    దశాంశం *C, int64_t ldc,
                    పూర్ణాంకం ith, పూర్ణాంకం nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    శూన్య matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  అంతరంగ:
    NOINLINE శూన్య mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        ఎంపిక ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3ll)) {
        స్థితి 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            ఆపు;
        అప్రమేయ:
            తిరిగి;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN>
    NOINLINE శూన్య gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * RN;
            float32x4_t Cv[RN][RM] = {};
            కొరకు (int64_t l = 0; l < k; ++l)
                కొరకు (int64_t j = 0; j < RN; ++j)
                    కొరకు (int64_t i = 0; i < RM; ++i)
                        Cv[j][i] = vmlaq_n_f32(Cv[j][i],
                                               vcvtq_f32_s32(vdotq_s32(
                                                   vdotq_s32(vdupq_n_s32(0),
                                                             ఎక్కించు_lo(A + lda * (ii + i) + l),
                                                             ఎక్కించు_lo(B + ldb * (jj + j) + l)),
                                                   ఎక్కించు_hi(A + lda * (ii + i) + l),
                                                   ఎక్కించు_hi(B + ldb * (jj + j) + l))),
                                               unhalf(A[lda * (ii + i) + l].d) *
                                               unhalf(B[ldb * (jj + j) + l].d));
            కొరకు (int64_t j = 0; j < RN; ++j)
                కొరకు (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    అంతఃవరుస int8x16_t ఎక్కించు_lo(స్థిరరాశి ఖండం_q8_0 *b) {
        తిరిగి vld1q_s8(b->qs);
    }

    అంతఃవరుస int8x16_t ఎక్కించు_hi(స్థిరరాశి ఖండం_q8_0 *b) {
        తిరిగి vld1q_s8(b->qs + 16);
    }

    అంతఃవరుస int8x16_t ఎక్కించు_lo(స్థిరరాశి ఖండం_q4_0 *b) {
        తిరిగి vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs),
                                                     vdupq_n_u8(0x0f))),
                        vdupq_n_s8(0x8));
    }

    అంతఃవరుస int8x16_t ఎక్కించు_hi(స్థిరరాశి ఖండం_q4_0 *b) {
        తిరిగి vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)),
                        vdupq_n_s8(0x8));
    }

    స్థిరరాశి TA *స్థిరరాశి A;
    స్థిరరాశి ఖండం_q8_0 *స్థిరరాశి B;
    దశాంశం *స్థిరరాశి C;
    స్థిరరాశి int64_t k;
    స్థిరరాశి int64_t lda;
    స్థిరరాశి int64_t ldb;
    స్థిరరాశి int64_t ldc;
    స్థిరరాశి పూర్ణాంకం ith;
    స్థిరరాశి పూర్ణాంకం nth;
};
#అయితే_ముగింపు // __ARM_FEATURE_DOTPROD

#ఒకవేళ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
మూస <రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC>
తరగతి tinyBLAS_Q0_AVX {
  బహిరంగ:
    tinyBLAS_Q0_AVX(int64_t k,
                    స్థిరరాశి TA *A, int64_t lda,
                    స్థిరరాశి TB *B, int64_t ldb,
                    TC *C, int64_t ldc,
                    పూర్ణాంకం ith, పూర్ణాంకం nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
        స్థిరరాశి int8_t kvalues_iq4nl[16] = {
            -127, -104, -83, -65,
            -49,  -35,  -22, -10,
              1,   13,   25,  38,
             53,   69,   89, 113
        };

        iq4nlt = _mm_loadu_si128((స్థిరరాశి __m128i *)kvalues_iq4nl);
    }

    శూన్య matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  అంతరంగ:
    శూన్య mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        ఎంపిక ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
#ఒకవేళ సదిశ_REGISTERS == 32
        స్థితి 0x44:
            mc = 4;
            nc = 4;
#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<4>(m0, m, n0, n);
#లేకపోతే
            gemm<4, 4>(m0, m, n0, n);
#అయితే_ముగింపు
            ఆపు;
        స్థితి 0x43:
            mc = 4;
            nc = 3;
#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<3>(m0, m, n0, n);
#లేకపోతే
            gemm<4, 3>(m0, m, n0, n);
#అయితే_ముగింపు
            ఆపు;
        స్థితి 0x34:
            mc = 3;
            nc = 4;
#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<3>(m0, m, n0, n);
#లేకపోతే
            gemm<3, 4>(m0, m, n0, n);
#అయితే_ముగింపు
            ఆపు;
        స్థితి 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x42:
            mc = 4;
            nc = 2;
#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#లేకపోతే
            gemm<4, 2>(m0, m, n0, n);
#అయితే_ముగింపు
            ఆపు;
        స్థితి 0x24:
            mc = 2;
            nc = 4;
#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#లేకపోతే
            gemm<2, 4>(m0, m, n0, n);
#అయితే_ముగింపు
            ఆపు;
#లేకపోతే
        స్థితి 0x44:
        స్థితి 0x43:
        స్థితి 0x42:
            mc = 4;
            nc = 2;
#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#లేకపోతే
            gemm<4, 2>(m0, m, n0, n);
#అయితే_ముగింపు
            ఆపు;
        స్థితి 0x34:
        స్థితి 0x24:
            mc = 2;
            nc = 4;
#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#లేకపోతే
            gemm<2, 4>(m0, m, n0, n);
#అయితే_ముగింపు
            ఆపు;
        స్థితి 0x33:
#అయితే_ముగింపు
        స్థితి 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x41:
            mc = 4;
            nc = 1;
#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
            gemm4xN<1>(m0, m, n0, n);
#లేకపోతే
            gemm<4, 1>(m0, m, n0, n);
#అయితే_ముగింపు
            ఆపు;
        స్థితి 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x14:
            mc = 1;
            nc = 4;
#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
            gemmMx4<1>(m0, m, n0, n);
#లేకపోతే
            gemm<1, 4>(m0, m, n0, n);
#అయితే_ముగింపు
            ఆపు;
        స్థితి 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            ఆపు;
        స్థితి 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            ఆపు;
        అప్రమేయ:
            తిరిగి;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

#ఒకవేళ defined(__AVX2__) && defined(__F16C__)
// Templated functions for gemm of dimensions 4xN
    మూస <పూర్ణాంకం RN>
    NOINLINE శూన్య gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / 4;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * 4;
            int64_t jj = n0 + పని % xtiles * RN;
            __m256 Cv[RN][4] = {};
            కొరకు (int64_t l = 0; l < k; ++l) {
                uint64_t a_delta = ((uint64_t)A[lda * (ii + 3) + l].d << 48) | ((uint64_t)A[lda * (ii + 2) + l].d << 32) | ((uint64_t)A[lda * (ii + 1) + l].d << 16) | (A[lda * (ii + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 da = _mm_cvtph_ps(mm_సమితి_epi64x(0, a_delta));
                __m256i avec0 = load(A + lda * (ii + 0) + l);
                __m256i avec1 = load(A + lda * (ii + 1) + l);
                __m256i avec2 = load(A + lda * (ii + 2) + l);
                __m256i avec3 = load(A + lda * (ii + 3) + l);
                కొరకు (int64_t j = 0; j < RN; ++j) {
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

            కొరకు (int64_t j = 0; j < RN; ++j)
                కొరకు (int64_t i = 0; i < 4; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    // Templated functions for gemm of dimensions Mx4
    మూస <పూర్ణాంకం RM>
    NOINLINE శూన్య gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / 4;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * 4;
            __m256 Cv[4][RM] = {};
            కొరకు (int64_t l = 0; l < k; ++l) {
                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 3) + l].d << 48) | ((uint64_t)B[ldb * (jj + 2) + l].d << 32) | ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 db = _mm_cvtph_ps(mm_సమితి_epi64x(0, b_delta));
                __m256i bvec0 = load(B + ldb * (jj + 0) + l);
                __m256i bvec1 = load(B + ldb * (jj + 1) + l);
                __m256i bvec2 = load(B + ldb * (jj + 2) + l);
                __m256i bvec3 = load(B + ldb * (jj + 3) + l);
                కొరకు (int64_t i = 0; i < RM; ++i) {
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
            కొరకు (int64_t j = 0; j < 4; ++j)
                కొరకు (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }
#అయితే_ముగింపు

    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN>
    NOINLINE శూన్య gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            కొరకు (int64_t l = 0; l < k; ++l)
                కొరకు (int64_t j = 0; j < RN; ++j)
                    కొరకు (int64_t i = 0; i < RM; ++i) {
#ఒకవేళ defined(__AVX2__)
                        __m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                              load(A + lda * (ii + i) + l)),
                                             _mm256_sign_epi8(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l)));
#లేకపోతే
                        __m128i ali0 = load0(A + lda * (ii + i) + l);
                        __m128i ali1 = load1(A + lda * (ii + i) + l);
                        __m128i blj0 = load0(B + ldb * (jj + j) + l);
                        __m128i blj1 = load1(B + ldb * (jj + j) + l);

                        __m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
                        __m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
                        __m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
                        __m128i sepBA1 = _mm_sign_epi8(blj1, ali1);

                        // updot
                        స్థిరరాశి __m128i oneFill = _mm_set1_epi16(1);
                        __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0);
                        __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1);
                        __m256 udTmp = _mm256_cvtepi32_ps(MM256_సమితి_M128_I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
#అయితే_ముగింపు
                        Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            కొరకు (int64_t j = 0; j < RN; ++j)
                కొరకు (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    అంతఃవరుస __m256i load(స్థిరరాశి ఖండం_q8_0 *b) {
        తిరిగి _mm256_loadu_si256((స్థిరరాశి __m256i *)b->qs);
    }

    అంతఃవరుస __m128i load0(స్థిరరాశి ఖండం_q8_0 *b) {
        తిరిగి _mm_loadu_si128((స్థిరరాశి __m128i *)b->qs);
    }

    అంతఃవరుస __m128i load1(స్థిరరాశి ఖండం_q8_0 *b) {
        తిరిగి _mm_loadu_si128(((స్థిరరాశి __m128i *)b->qs) + 1);
    }

    అంతఃవరుస __m256i load(స్థిరరాశి ఖండం_q4_0 *b) {
        తిరిగి _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    అంతఃవరుస __m128i load0(స్థిరరాశి ఖండం_q4_0 *b) {
        స్థిరరాశి __m128i x = _mm_loadu_si128((స్థిరరాశి __m128i *)(b->qs));
        తిరిగి _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), x), _mm_set1_epi8(8));
    }

    అంతఃవరుస __m128i load1(స్థిరరాశి ఖండం_q4_0 *b) {
        స్థిరరాశి __m128i x = _mm_loadu_si128((స్థిరరాశి __m128i *)(b->qs));
        తిరిగి _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)), _mm_set1_epi8(8));
    }

    అంతఃవరుస __m256i load(స్థిరరాశి ఖండం_q5_0 *b) {
        తిరిగి _mm256_or_si256(denibble(b->qs), bittobyte(b->qh));
    }

    అంతఃవరుస __m128i load0(స్థిరరాశి ఖండం_q5_0* b) {
        స్థిరరాశి __m128i x = _mm_loadu_si128((స్థిరరాశి __m128i *)(b->qs));
        uint32_t x32;
        జ్ఞాపకం_నకలు(&x32, b->qh, పరిమాణం(uint32_t));
        __m128i qxl = _mm_and_si128(_mm_set1_epi8(15), x);
        __m128i bytesl = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      mm_సమితి_epi64x(0x0101010101010101, 0x0000000000000000))));
        bytesl = _mm_andnot_si128(bytesl, _mm_set1_epi8((అక్షరం)0xF0));
        తిరిగి _mm_or_si128(qxl, bytesl);
    }

    అంతఃవరుస __m128i load1(స్థిరరాశి ఖండం_q5_0* b) {
        స్థిరరాశి __m128i x = _mm_loadu_si128((స్థిరరాశి __m128i *)(b->qs));
        uint32_t x32;
        జ్ఞాపకం_నకలు(&x32, b->qh, పరిమాణం(uint32_t));
        __m128i qxh = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
        __m128i bytesh = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      mm_సమితి_epi64x(0x0303030303030303, 0x0202020202020202))));
        bytesh = _mm_andnot_si128(bytesh, _mm_set1_epi8((అక్షరం)0xF0));
        తిరిగి _mm_or_si128(qxh, bytesh);
    }

    అంతఃవరుస __m256i load(స్థిరరాశి ఖండం_iq4_nl *b) {
        తిరిగి MM256_సమితి_M128_I(load1(b), load0(b));
    }

    అంతఃవరుస __m128i load0(స్థిరరాశి ఖండం_iq4_nl *b) {
        స్థిరరాశి __m128i x = _mm_loadu_si128((స్థిరరాశి __m128i *)(b->qs));
        తిరిగి _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), x));
    }

    అంతఃవరుస __m128i load1(స్థిరరాశి ఖండం_iq4_nl *b) {
        స్థిరరాశి __m128i x = _mm_loadu_si128((స్థిరరాశి __m128i *)(b->qs));
        తిరిగి _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)));
    }

    అంతఃవరుస __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#ఒకవేళ defined(__AVX512VNNI__) && defined(__AVX512VL__)
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#లేదావేళ defined(__AVXVNNI__)
        res = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
#లేకపోతే
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#అయితే_ముగింపు
        తిరిగి _mm256_cvtepi32_ps(res);
    }

    స్థిర అంతఃవరుస __m256i denibble(స్థిరరాశి uint8_t *p) {
        __m128i x = _mm_loadu_si128((స్థిరరాశి __m128i *)p);
        తిరిగి _mm256_and_si256(_mm256_set1_epi8(15),
                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                        _mm_srli_epi16(x, 4), 1));
    }

    స్థిర అంతఃవరుస __m256i bittobyte(స్థిరరాశి uint8_t *p) {
        uint32_t x32;
        జ్ఞాపకం_నకలు(&x32, p, పరిమాణం(uint32_t));
        __m256i బైట్లు = _mm256_cmpeq_epi8(_mm256_set1_epi64x(-1),
                                          _mm256_or_si256(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                          _mm256_shuffle_epi8(_mm256_set1_epi32(x32),
                                                                              mm256_సమితి_epi64x(0x0303030303030303, 0x0202020202020202,
                                                                                                0x0101010101010101, 0x0000000000000000))));
        తిరిగి _mm256_andnot_si256(బైట్లు, _mm256_set1_epi8((అక్షరం)0xF0));
    }

    స్థిరరాశి TA *స్థిరరాశి A;
    స్థిరరాశి TB *స్థిరరాశి B;
    TC *స్థిరరాశి C;
    స్థిరరాశి int64_t k;
    స్థిరరాశి int64_t lda;
    స్థిరరాశి int64_t ldb;
    స్థిరరాశి int64_t ldc;
    స్థిరరాశి పూర్ణాంకం ith;
    స్థిరరాశి పూర్ణాంకం nth;
    __m128i iq4nlt;
};
#అయితే_ముగింపు // __AVX__

//PPC Implementation
#ఒకవేళ defined(__MMA__)

మూస<రకం_పేరు T>
నిర్మాణం mma_instr;

మూస<>
నిర్మాణం mma_instr<ggml_bf16_t> {
    స్థిర అంతఃవరుస శూన్య outer_లబ్ధం(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvbf16ger2pp(acc, a, b);
    }
};

మూస<>
నిర్మాణం mma_instr<ggml_fp16_t> {
    స్థిర అంతఃవరుస శూన్య outer_లబ్ధం(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvf16ger2pp(acc, a, b);
    }
};

మూస <రకం_పేరు TA, రకం_పేరు TB, రకం_పేరు TC>
తరగతి tinyBLAS_HP16_PPC {
  బహిరంగ:
    tinyBLAS_HP16_PPC(int64_t k,
                స్థిరరాశి TA *A, int64_t lda,
                స్థిరరాశి TB *B, int64_t ldb,
                TC *C, int64_t ldc,
                పూర్ణాంకం ith, పూర్ణాంకం nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    శూన్య matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #ఒకవేళ defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #అయితే_ముగింపు
        ఒకవేళ (k < kc) {
            kc = k;
        }
        తార్కిక can_use_tiled = (m % mc == 0) && (n % nc == 0) && (k % kc == 0);
        ఒకవేళ (can_use_tiled) {
            matmul_tiled(m, n, mc, nc, kc);
        } లేకపోతే {
            mnpack(0, m, 0, n);
        }
    }

  అంతరంగ:
    __attribute__((always_inline))
    అంతఃవరుస శూన్య save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        కొరకు (పూర్ణాంకం I = 0; I < 4; I++) {
            కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                *((దశాంశం *)(C+ii+((jj+J)*ldc)+I)) = *((దశాంశం *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    అంతఃవరుస శూన్య కూడు_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        కొరకు (పూర్ణాంకం I = 0; I < 4; I++) {
            కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                దశాంశం * c_ptr = (దశాంశం *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((దశాంశం *)&vec_C[I]+J);
            }
        }
    }

    శూన్య సదిశ_క్రమమార్పు_store(vec_t *c, పూర్ణాంకం అంకె_Vec, గుర్తులేని అక్షరం *vec_స్థానభ్రంశం) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        ఒకవేళ (అంకె_Vec == 2) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[2], c[3], swiz1);
            s[0] = vec_perm(t[0], t[1], swiz3);
            s[1] = vec_perm(t[0], t[1], swiz4);
            vec_xst(s[0], 0, (vec_t*)vec_స్థానభ్రంశం);
            vec_xst(s[1], 0, (vec_t*)(vec_స్థానభ్రంశం + 16));
        } లేకపోతే ఒకవేళ (అంకె_Vec == 4) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[0], c[1], swiz2);
            t[2] = vec_perm(c[2], c[3], swiz1);
            t[3] = vec_perm(c[2], c[3], swiz2);
            s[0] = vec_perm(t[0], t[2], swiz3);
            s[1] = vec_perm(t[0], t[2], swiz4);
            s[2] = vec_perm(t[1], t[3], swiz3);
            s[3] = vec_perm(t[1], t[3], swiz4);
            కొరకు (పూర్ణాంకం i = 0; i < 4; ++i)
                vec_xst(s[i], 0, (vec_t*)(vec_స్థానభ్రంశం + i * 16));
        } లేకపోతే ఒకవేళ (అంకె_Vec == 8) {
            కొరకు (పూర్ణాంకం i = 0; i < 4; i += 2) {
                t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
                t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
            }
            కొరకు (పూర్ణాంకం i = 4; i < 8; i += 2) {
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
            కొరకు (పూర్ణాంకం i = 0; i < 8; ++i)
                vec_xst(s[i], 0, (vec_t*)(vec_స్థానభ్రంశం + i * 16));
        }
    }

    శూన్య pack_సాధారణ(స్థిరరాశి TA* a, int64_t lda, పూర్ణాంకం అడ్డువరుసలు, పూర్ణాంకం నిలువుల_సంఖ్య_2, గుర్తులేని అక్షరం* vec) {
        int64_t i, j;
        TA *aoffset = ఖాళీ;
        గుర్తులేని అక్షరం *vec_స్థానభ్రంశం = ఖాళీ;
        TA * aoffsets[8];
        vector గుర్తులేని అక్షరం c_arr[8];
        aoffset = const_cast<TA*>(a);
        vec_స్థానభ్రంశం = vec;
        j = (అడ్డువరుసలు >> 3);
        ఒకవేళ (j > 0) {
            చేయి {
                aoffsets[0] = aoffset;
                ఒకవేళ (నిలువుల_సంఖ్య_2 == 4) {
                    aoffsets[0] = aoffset;
                    కొరకు (పూర్ణాంకం it = 1; it < 4; ++it)
                        aoffsets[it] = aoffsets[it-1] + lda;
                    aoffset += 4 * lda;
                    కొరకు (పూర్ణాంకం i = 0; i < 4; ++i)
                        c_arr[i] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[i]);
                    సదిశ_క్రమమార్పు_store(c_arr, 4, vec_స్థానభ్రంశం);
                    కొరకు (పూర్ణాంకం i = 0; i<4; i++)
                        aoffsets[i] = aoffsets[i]+lda;
                    vec_స్థానభ్రంశం +=64;
                }
                i = (నిలువుల_సంఖ్య_2 >> 3);
                ఒకవేళ (i > 0) {
                    కొరకు (పూర్ణాంకం it = 1; it < 8; ++it) {
                        aoffsets[it] = aoffsets[it-1] + lda;
                    }
                    aoffset += 8 * lda;

                    చేయి {
                        కొరకు (పూర్ణాంకం it = 0; it < 8; ++it)
                            c_arr[it] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[it]);
                        సదిశ_క్రమమార్పు_store(c_arr, 8, vec_స్థానభ్రంశం);
                        కొరకు (పూర్ణాంకం it = 0; it < 8; ++it)
                            aoffsets[it] = aoffsets[it] + 8;
                        vec_స్థానభ్రంశం += 128;
                        i--;
                    } అయితే(i > 0);
                }
                j--;
            } అయితే(j > 0);
        }
        ఒకవేళ (అడ్డువరుసలు & 4) {
            aoffsets[0] = aoffset;
            కొరకు (పూర్ణాంకం it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            ఒకవేళ (నిలువుల_సంఖ్య_2 == 4) {
                కొరకు (పూర్ణాంకం it = 0; it < 4; ++it)
                    c_arr[it] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[it]);
                సదిశ_క్రమమార్పు_store(c_arr, 2, vec_స్థానభ్రంశం);
                కొరకు (పూర్ణాంకం it = 0; it< 4; it++)
                    aoffsets[it] = aoffsets[it] + lda;
                vec_స్థానభ్రంశం += 32;
            }
            i = (నిలువుల_సంఖ్య_2 >> 3);
            ఒకవేళ (i > 0) {
                చేయి {
                    కొరకు (పూర్ణాంకం it = 0; it < 4; ++it)
                        c_arr[it] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[it]);
                    సదిశ_క్రమమార్పు_store(c_arr, 4, vec_స్థానభ్రంశం);
                    కొరకు (పూర్ణాంకం it = 0; it< 4; it++)
                        aoffsets[it] = aoffsets[it] + 8*lda;
                    vec_స్థానభ్రంశం += 64;
                    i--;
                } అయితే(i > 0);
            }
        }
        ఒకవేళ (అడ్డువరుసలు & 3) {
            aoffsets[0] = aoffset;
            కొరకు (పూర్ణాంకం it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            ఒకవేళ (నిలువుల_సంఖ్య_2 == 4) {
                ఎంపిక(అడ్డువరుసలు) {
                    స్థితి 3: c_arr[2] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[2]);
                    స్థితి 2: c_arr[1] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[1]);
                    స్థితి 1: c_arr[0] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[0]);
                        ఆపు;
                }
                సదిశ_క్రమమార్పు_store(c_arr, 2, vec_స్థానభ్రంశం);
                కొరకు (పూర్ణాంకం it = 0; it< 4; it++)
                     aoffsets[it] = aoffsets[it] + lda;
                vec_స్థానభ్రంశం += 32;
            }
            i = (నిలువుల_సంఖ్య_2 >> 3);
            ఒకవేళ (i > 0) {
                చేయి {
                    ఎంపిక(అడ్డువరుసలు) {
                        స్థితి 3: c_arr[2] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[2]);
                        స్థితి 2: c_arr[1] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[1]);
                        స్థితి 1: c_arr[0] = vec_xl(0, (vector గుర్తులేని అక్షరం*)aoffsets[0]);
                            ఆపు;
                    }
                    సదిశ_క్రమమార్పు_store(c_arr, 4, vec_స్థానభ్రంశం);
                    కొరకు (పూర్ణాంకం it = 0; it <4; it++)
                         aoffsets[it] = aoffsets[it] + 8* lda;
                    vec_స్థానభ్రంశం += 64;
                    i--;
                } అయితే(i > 0);
            }
        }
    }

    శూన్య mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        పూర్ణాంకం m_rem = MIN(m - m0, 8);
        పూర్ణాంకం n_rem = MIN(n - n0, 8);

        ఒకవేళ (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4,8>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ (m_rem >=8 && n_rem >=4){
                mc = 8;
                nc = 4;
                gemm<8,4>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ ((m_rem < 4) && (n_rem >= 8)) {
            nc = 8;
            ఎంపిక(m_rem) {
                స్థితి 1:
                    mc = 1;
                    gemm_Mx8<1>(m0, m, n0, n);
                    ఆపు;
                స్థితి 2:
                    mc = 2;
                    gemm_Mx8<2>(m0, m, n0, n);
                    ఆపు;
                స్థితి 3:
                    mc = 3;
                    gemm_Mx8<3>(m0, m, n0, n);
                    ఆపు;
                అప్రమేయ:
                    తిరిగి;
            }
        } లేకపోతే ఒకవేళ (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small<4, 4>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ ((m_rem > 4) && (n_rem < 4)) {
            mc = 4;
            ఎంపిక(n_rem) {
                స్థితి 1:
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    ఆపు;
                స్థితి 2:
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    ఆపు;
                స్థితి 3:
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    ఆపు;

                అప్రమేయ:
                    తిరిగి;
            }
        } లేకపోతే {
            ఎంపిక((m_rem << 4) | n_rem) {
                స్థితి 0x43:
                    mc = 4;
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x42:
                    mc = 4;
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x41:
                    mc = 4;
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x34:
                    mc = 3;
                    nc = 4;
                    gemm_small<3, 4>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x33:
                    mc = 3;
                    nc = 3;
                    gemm_small<3, 3>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x32:
                    mc = 3;
                    nc = 2;
                    gemm_small<3, 2>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x31:
                    mc = 3;
                    nc = 1;
                    gemm_small<3, 1>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x24:
                    mc = 2;
                    nc = 4;
                    gemm_small<2,4>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x23:
                    mc = 2;
                    nc = 3;
                    gemm_small<2, 3>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x22:
                    mc = 2;
                    nc = 2;
                    gemm_small<2, 2>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x21:
                    mc = 2;
                    nc = 1;
                    gemm_small<2, 1>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x14:
                    mc = 1;
                    nc = 4;
                    gemm_small<1, 4>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x13:
                    mc = 1;
                    nc = 3;
                    gemm_small<1, 3>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x12:
                    mc = 1;
                    nc = 2;
                    gemm_small<1, 2>(m0, m, n0, n);
                    ఆపు;
                స్థితి 0x11:
                    mc = 1;
                    nc = 1;
                    gemm_small<1, 1>(m0, m, n0, n);
                    ఆపు;
                అప్రమేయ:
                    తిరిగి;
            }
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    శూన్య KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        కొరకు (పూర్ణాంకం l = 0; l < k; l+=8) {
            pack_సాధారణ((A+(ii*lda)+l), lda, 4, 8, (uint8_t*)vec_A);
            pack_సాధారణ((B+(jj*ldb)+l), ldb, 8, 8, (uint8_t*)vec_B);
            కొరకు (పూర్ణాంకం x = 0; x < 4; x++) {
                mma_instr<TA>::outer_లబ్ధం(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_లబ్ధం(&acc_1, vec_A[x], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
    }

    శూన్య KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        కొరకు (పూర్ణాంకం l = 0; l < k; l+=8) {
            pack_సాధారణ((A+(ii*lda)+l), lda, 8, 8, (uint8_t*)vec_A);
            pack_సాధారణ((B+(jj*ldb)+l), ldb, 8, 4, (uint8_t*)vec_B);
            కొరకు (పూర్ణాంకం x = 0; x < 4; x++) {
                mma_instr<TA>::outer_లబ్ధం(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_లబ్ధం(&acc_1, vec_A[x+4], vec_B[x]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii+4, jj);
    }


    శూన్య KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        కొరకు (పూర్ణాంకం l = 0; l < k; l+=8) {
            pack_సాధారణ(A+(ii*lda)+l, lda, 8, 8, (uint8_t*)vec_A);
            pack_సాధారణ(B+(jj*ldb)+l, ldb, 8, 8, (uint8_t*)vec_B);
            కొరకు (పూర్ణాంకం x = 0; x < 4; x++) {
                mma_instr<TA>::outer_లబ్ధం(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_లబ్ధం(&acc_1, vec_A[x], vec_B[x+4]);
                mma_instr<TA>::outer_లబ్ధం(&acc_2, vec_A[x+4], vec_B[x]);
                mma_instr<TA>::outer_లబ్ధం(&acc_3, vec_A[x+4], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
        save_acc(&acc_2, ii+4, jj);
        save_acc(&acc_3, ii+4, jj+4);
    }

    అంతఃవరుస శూన్య MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        కొరకు (పూర్ణాంకం x = 0; x < 4; x ++) {
             mma_instr<TA>::outer_లబ్ధం(&acc[0], vec_A0[x], vec_B[x]);
             mma_instr<TA>::outer_లబ్ధం(&acc[1], vec_A0[x], vec_B[x+4]);
             mma_instr<TA>::outer_లబ్ధం(&acc[2], vec_A0[x+4], vec_B[x]);
             mma_instr<TA>::outer_లబ్ధం(&acc[3], vec_A0[x+4], vec_B[x+4]);
             mma_instr<TA>::outer_లబ్ధం(&acc[4], vec_A1[x], vec_B[x]);
             mma_instr<TA>::outer_లబ్ధం(&acc[5], vec_A1[x], vec_B[x+4]);
             mma_instr<TA>::outer_లబ్ధం(&acc[6], vec_A1[x+4], vec_B[x]);
             mma_instr<TA>::outer_లబ్ధం(&acc[7], vec_A1[x+4], vec_B[x+4]);
        }
    }
    శూన్య KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        కొరకు (int64_t i = 0; i < mc; i += 16) {
            పూర్ణాంకం A_పునాది_addr = (mc / 8) * (i / 8) * 8;
            కొరకు (int64_t j = 0; j < nc; j += 8) {
                 పూర్ణాంకం B_పునాది_addr = (nc / 8) * (j / 8) * 8;
                 acc_t acc[8];
                 vec_t A0_ఖండం[8]; vec_t A1_ఖండం[8];
                 కొరకు (పూర్ణాంకం x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 కొరకు (int64_t l = 0; l < kc; l += 8) {
                     పూర్ణాంకం A0_ఖండం_idx = A_పునాది_addr + (l / 8) * 8;
                     పూర్ణాంకం A1_ఖండం_idx = A0_ఖండం_idx + (mc / 8) * 8;
                     పూర్ణాంకం B_ఖండం_idx = B_పునాది_addr + (l / 8) * 8;
                     vec_t* A0_ఖండం = &vec_A[A0_ఖండం_idx];
                     vec_t* A1_ఖండం = &vec_A[A1_ఖండం_idx];
                     vec_t* B_ఖండం = &vec_B[B_ఖండం_idx];
                     MMA_16x8(A0_ఖండం, A1_ఖండం, B_ఖండం, acc);
                 }
                 ఒకవేళ (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } లేకపోతే {
                     కూడు_save_acc(&acc[0], ii + i, jj + j);
                     కూడు_save_acc(&acc[1], ii + i, jj + j + 4);
                     కూడు_save_acc(&acc[2], ii + i + 4, jj + j);
                     కూడు_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     కూడు_save_acc(&acc[4], ii + i + 8, jj + j);
                     కూడు_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     కూడు_save_acc(&acc[6], ii + i + 12, jj + j);
                     కూడు_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }
    మూస<పూర్ణాంకం RM, పూర్ణాంకం RN>
    శూన్య gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[2], vec_B[2];
            కొరకు (పూర్ణాంకం l=0; l<k; l+=4) {
                pack_సాధారణ(A+(ii*lda)+l, lda, RM, 4, (uint8_t*)vec_A);
                pack_సాధారణ(B+(jj*ldb)+l, ldb, RN, 4, (uint8_t*)vec_B);
                కొరకు (పూర్ణాంకం x = 0; x<2; x++) {
                    mma_instr<TA>::outer_లబ్ధం(&acc_0, vec_A[x], vec_B[x]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            కొరకు (పూర్ణాంకం I = 0; I < RM; I++) {
                కొరకు (పూర్ణాంకం J = 0; J < RN; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    మూస<పూర్ణాంకం RM>
    శూన్య gemm_Mx8(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        పూర్ణాంకం RN = 8;
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0, acc_1;
            __builtin_mma_xxsetaccz(&acc_0);
            __builtin_mma_xxsetaccz(&acc_1);
            vec_t vec_A[4], vec_B[8];
            కొరకు (పూర్ణాంకం l=0; l<k; l+=8) {
                pack_సాధారణ(A+(ii*lda)+l, lda, RM, 8, (uint8_t*)vec_A);
                pack_సాధారణ(B+(jj*ldb)+l, ldb, RN, 8, (uint8_t*)vec_B);
                కొరకు (పూర్ణాంకం x = 0; x<4; x++) {
                    mma_instr<TA>::outer_లబ్ధం(&acc_0, vec_A[x], vec_B[x]);
                    mma_instr<TA>::outer_లబ్ధం(&acc_1, vec_A[x], vec_B[x+4]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            కొరకు (పూర్ణాంకం I = 0; I < RM; I++) {
                కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_1);
            కొరకు (పూర్ణాంకం I = 0; I < RM; I++) {
                కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+4+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    మూస<పూర్ణాంకం RM, పూర్ణాంకం RN>
    అంతఃవరుస శూన్య kernel(int64_t ii, int64_t jj) {
       ఒకవేళ constexpr(RM == 4 && RN == 8) {
          KERNEL_4x8(ii,jj);
       } లేకపోతే ఒకవేళ constexpr(RM == 8 && RN == 8) {
          KERNEL_8x8(ii,jj);
       } లేకపోతే ఒకవేళ constexpr(RM == 8 && RN == 4) {
          KERNEL_8x4(ii,jj);
       } లేకపోతే {
          ప్రతిజ్ఞా(అబద్ధం && "RN/RM values not supported");
       }
    }

    శూన్య matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles) {
            end = tiles;
        }
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = (పని / xtiles) * mc;
            int64_t jj = (పని % xtiles) * nc;
            కొరకు (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 8];
                 vec_t B_pack[kc * nc / 8];
                 pack_సాధారణ(A + (ii * lda) + kk, lda, kc, mc, (uint8_t *)A_pack);
                 pack_సాధారణ(B + (jj * ldb) + kk, ldb, kc, nc, (uint8_t *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN>
    NOINLINE శూన్య gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    స్థిరరాశి TA *స్థిరరాశి A;
    స్థిరరాశి TB *స్థిరరాశి B;
    TC *C;
    స్థిరరాశి int64_t k;
    స్థిరరాశి int64_t lda;
    స్థిరరాశి int64_t ldb;
    స్థిరరాశి int64_t ldc;
    స్థిరరాశి పూర్ణాంకం ith;
    స్థిరరాశి పూర్ణాంకం nth;
};

మూస <రకం_పేరు TA>
తరగతి tinyBLAS_Q0_PPC {
  బహిరంగ:
    tinyBLAS_Q0_PPC(int64_t k,
             స్థిరరాశి TA * A, int64_t lda,
             స్థిరరాశి ఖండం_q8_0 * B, int64_t ldb,
             దశాంశం * C, int64_t ldc,
             పూర్ణాంకం ith, పూర్ణాంకం nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    శూన్య matmul(int64_t m, int64_t n) {
        int64_t mc = 64;
        int64_t nc = 64;
        int64_t kc = 64;
        int64_t n_chunk = 64;
    #ఒకవేళ defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 32;
        nc = 32;
        kc = 32;
        n_chunk = 32;
    #అయితే_ముగింపు
        int64_t n_aligned = 0;
        ఒకవేళ (n % n_chunk == 0) {
            n_aligned = n;
        } లేకపోతే ఒకవేళ (n == 4) {
            n_aligned = 4;
        } లేకపోతే ఒకవేళ (n < n_chunk) {
            n_aligned = (n / 8) * 8;
        } లేకపోతే {
            n_aligned = (n / n_chunk) * n_chunk;
        }
        ఒకవేళ (n_aligned > 0) {
            ఒకవేళ (n_aligned % n_chunk == 0) nc = n_chunk;
            లేకపోతే ఒకవేళ (n_aligned == n)      nc = n;
            లేకపోతే ఒకవేళ (n_aligned % 32 == 0) nc = 32;
            లేకపోతే ఒకవేళ (n_aligned % 24 == 0) nc = 24;
            లేకపోతే ఒకవేళ (n_aligned % 16 == 0) nc = 16;
            లేకపోతే                          nc = 8;
        }
        తార్కిక can_use_tiled = n_aligned > 0 && (m % mc == 0);
        ఒకవేళ (can_use_tiled) {
            matmul_tiled(m, n_aligned, mc, nc, kc);
            ఒకవేళ (n > n_aligned) {
                mnpack(0, m, n_aligned, n);
            }
        } లేకపోతే {
            mnpack(0, m, 0, n);
        }
    }

  అంతరంగ:
    అంతఃవరుస శూన్య save_res(పూర్ణాంకం ii, పూర్ణాంకం jj, పూర్ణాంకం idx, vector దశాంశం * fin_res, పూర్ణాంకం RM = 4, పూర్ణాంకం RN = 4) {
        కొరకు (పూర్ణాంకం I = 0; I < RM; I++) {
            కొరకు (పూర్ణాంకం J = 0; J < RN; J++) {
                *((దశాంశం *)(C + ii + ((jj + J) * ldc) + I)) = *((దశాంశం *)&fin_res[idx + I] + J);
            }
        }
    }

    అంతఃవరుస శూన్య save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        కొరకు (పూర్ణాంకం I = 0; I < 4; I++) {
            కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                *((దశాంశం *)(C + ii + ((jj + J) * ldc) + I)) = *((దశాంశం *)&vec_C[I] + J);
            }
        }
    }

    అంతఃవరుస శూన్య కూడు_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        కొరకు (పూర్ణాంకం I = 0; I < 4; I++) {
            కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                దశాంశం * c_ptr = (దశాంశం *)(C + ii+ ((jj + J) * ldc) + I);
                *c_ptr += *((దశాంశం *)&vec_C[I] + J);
            }
        }
    }

    మూస<రకం_పేరు ArrayType>
    అంతఃవరుస శూన్య compute(acc_t * ACC, పూర్ణాంకం c_idx, పూర్ణాంకం s_idx, ArrayType & comparray, vector దశాంశం * vs, vector దశాంశం * fin_res) {
        vector గుర్తుగల పూర్ణాంకం vec_C[4];
        vector దశాంశం CA[4] = {0};
        vector దశాంశం res[4] = {0};
        __builtin_mma_disassemble_acc(vec_C, ACC);
        కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            CA[i] = vec_splats((దశాంశం)(((ద్విగుణ)comparray[c_idx + i]) * -128.0));
            res[i] = vec_కూడు(vec_ctf(vec_C[i], 0), CA[i]);
            fin_res[s_idx + i] = vec_madd(res[i], vs[s_idx + i], fin_res[s_idx + i]);
        }
    }

    అంతఃవరుస శూన్య process_q4_elements(vector గుర్తుగల అక్షరం (&c)[2], పూర్ణాంకం * ca) {
        స్థిరరాశి vector గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
        స్థిరరాశి vector గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);
        స్థిరరాశి vector గుర్తుగల అక్షరం v8 = vec_splats((గుర్తుగల అక్షరం)0x8);
        vector గుర్తుగల పూర్ణాంకం vsum = {0};
        vector గుర్తుగల పూర్ణాంకం vsum2 = {0};
        c[0] = vec_and(c[1], దిగువ_ముసుగు);
        c[1] = vec_sr(c[1], v4);
        c[0] = vec_sub(c[0], v8);
        c[1] = vec_sub(c[1], v8);
        vsum = vec_sum4s(c[0], vsum);
        vsum2 = vec_sum4s(c[1], vsum2);
        vsum = vec_కూడు(vsum, vsum2);
        *(ca) = vsum[0] + vsum[1] + vsum[2] + vsum[3];
    }

    మూస <రకం_పేరు V1, రకం_పేరు V2>
    అంతఃవరుస శూన్య సదిశ_క్రమమార్పు_store(V2 & s1, V2 & s2, V2 & s3, V2 & s4, V1 * vec_స్థానభ్రంశం, తార్కిక flip) {
        vector గుర్తులేని అక్షరం swiz1 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vector గుర్తులేని అక్షరం swiz2 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        vector గుర్తులేని అక్షరం swiz3 = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
        vector గుర్తులేని అక్షరం swiz4 = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
        V2 t1, t2, t3, t4, t5, t6, t7, t8;
        vector గుర్తులేని అక్షరం xor_సదిశ;
        uint8_t flip_vec = 0x80;
        xor_సదిశ = vec_splats(flip_vec);
        t1 = vec_perm(s1, s2, swiz1);
        t2 = vec_perm(s1, s2, swiz2);
        t3 = vec_perm(s3, s4, swiz1);
        t4 = vec_perm(s3, s4, swiz2);
        t5 = vec_perm(t1, t3, swiz3);
        t6 = vec_perm(t1, t3, swiz4);
        t7 = vec_perm(t2, t4, swiz3);
        t8 = vec_perm(t2, t4, swiz4);
        ఒకవేళ (flip == నిజం) {
            t5 = vec_xor(t5, xor_సదిశ);
            t6 = vec_xor(t6, xor_సదిశ);
            t7 = vec_xor(t7, xor_సదిశ);
            t8 = vec_xor(t8, xor_సదిశ);
        }
        vec_xst(t5, 0, vec_స్థానభ్రంశం);
        vec_xst(t6, 0, vec_స్థానభ్రంశం + 16);
        vec_xst(t7, 0, vec_స్థానభ్రంశం + 32);
        vec_xst(t8, 0, vec_స్థానభ్రంశం + 48);
    }

    అంతఃవరుస శూన్య unpack_q4_గమ్యం_q8(vector గుర్తుగల అక్షరం packed, vector గుర్తుగల అక్షరం & lo, vector గుర్తుగల అక్షరం & hi) {
        స్థిరరాశి vector గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0x0F);
        స్థిరరాశి vector గుర్తుగల అక్షరం v8      = vec_splats((గుర్తుగల అక్షరం)0x08);
        స్థిరరాశి vector గుర్తులేని అక్షరం v4    = vec_splats((గుర్తులేని అక్షరం)4);
        lo = vec_and(packed, దిగువ_ముసుగు);
        hi = vec_sr(packed, v4);
        lo = vec_sub(lo, v8);
        hi = vec_sub(hi, v8);
    }

    అంతఃవరుస శూన్య సదిశ_క్రమమార్పు_store_fp16(vec_t * c, గుర్తులేని అక్షరం * vec_స్థానభ్రంశం) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        కొరకు (పూర్ణాంకం i = 0; i < 4; i += 2) {
            t[i + 0] = vec_perm(c[i + 0], c[i + 1], swiz1);
            t[i + 1] = vec_perm(c[i + 0], c[i + 1], swiz2);
        }
        కొరకు (పూర్ణాంకం i = 4; i < 8; i += 2) {
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
        కొరకు (పూర్ణాంకం i = 0; i < 8; ++i) {
            vec_xst(s[i], 0, (vec_t *)(vec_స్థానభ్రంశం + i * 16));
        }
    }

    స్థిర అంతఃవరుస శూన్య convert_and_scale_q8(vector గుర్తుగల అక్షరం raw, vector దశాంశం v_scale, vector గుర్తులేని పొట్టి & బహిర్గతం_hi, vector గుర్తులేని పొట్టి & బహిర్గతం_lo) {
        vector గుర్తుగల పొట్టి i16_hi = vec_unpackh(raw);
        vector గుర్తుగల పొట్టి i16_lo = vec_unpackl(raw);

        vector దశాంశం f_hi_h = vec_ctf(vec_unpackh(i16_hi), 0);
        vector దశాంశం f_hi_l = vec_ctf(vec_unpackl(i16_hi), 0);
        vector దశాంశం f_lo_h = vec_ctf(vec_unpackh(i16_lo), 0);
        vector దశాంశం f_lo_l = vec_ctf(vec_unpackl(i16_lo), 0);
        బహిర్గతం_hi = vec_pack_గమ్యం_short_fp32(vec_mul(f_hi_h, v_scale), vec_mul(f_hi_l, v_scale));
        బహిర్గతం_lo = vec_pack_గమ్యం_short_fp32(vec_mul(f_lo_h, v_scale), vec_mul(f_lo_l, v_scale));
    }

    శూన్య pack_సాధారణ_q4_fp16(స్థిరరాశి ఖండం_q4_0 * a, int64_t lda, పూర్ణాంకం అడ్డువరుసలు, పూర్ణాంకం ఖండాలు, గుర్తులేని అక్షరం * vec) {
        గుర్తులేని అక్షరం * vec_స్థానభ్రంశం = vec;
        కొరకు (పూర్ణాంకం i = 0; i < అడ్డువరుసలు; i += 8) {
            స్థిరరాశి ఖండం_q4_0 * అడ్డువరుసలు_పునాది[8];
            కొరకు (పూర్ణాంకం r = 0; r < 8; r++) {
                అడ్డువరుసలు_పునాది[r] = a + (i + r) * lda;
            }
            కొరకు (పూర్ణాంకం blk = 0; blk < ఖండాలు; blk++) {
                vector గుర్తులేని పొట్టి hp_res[8][4];
                కొరకు (పూర్ణాంకం r = 0; r < 8; r++) {
                    స్థిరరాశి ఖండం_q4_0 * current_blk = అడ్డువరుసలు_పునాది[r] + blk;
                    vector దశాంశం v_scale = vec_వెలికితీయి_fp32_from_shorth(vec_splats(current_blk->d));
                    vector గుర్తుగల అక్షరం v_qs = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)current_blk->qs);
                    vector గుర్తుగల అక్షరం c1, c2;
                    unpack_q4_గమ్యం_q8(v_qs, c1, c2);
                    convert_and_scale_q8(c1, v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c2, v_scale, hp_res[r][2], hp_res[r][3]);
                }
                కొరకు (పూర్ణాంకం c = 0; c < 4; c++) {
                    vector గుర్తులేని అక్షరం c_arr[8];
                    కొరకు (పూర్ణాంకం r = 0; r < 8; r++) {
                        c_arr[r] = (vector గుర్తులేని అక్షరం)hp_res[r][c];
                    }
                    సదిశ_క్రమమార్పు_store_fp16((vec_t *)c_arr, vec_స్థానభ్రంశం);
                    vec_స్థానభ్రంశం += 128;
                }
            }
        }
    }

    మూస <పూర్ణాంకం chunk_కొలత>
    స్థిర అంతఃవరుస శూన్య pack_q8_ఖండం(స్థిరరాశి ఖండం_q8_0 * a, int64_t lda, పూర్ణాంకం అడ్డువరుసలు, పూర్ణాంకం ఖండాలు, గుర్తులేని అక్షరం * vec) {
        గుర్తులేని అక్షరం * vec_స్థానభ్రంశం = vec;
        స్థిరరాశి vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        స్థిరరాశి vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        స్థిరరాశి vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        స్థిరరాశి vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        కొరకు (పూర్ణాంకం i = 0; i < అడ్డువరుసలు; i += chunk_కొలత) {
            స్థిరరాశి ఖండం_q8_0 * అడ్డువరుసలు_పునాది[chunk_కొలత];
            కొరకు (పూర్ణాంకం r = 0; r < chunk_కొలత; r++) {
                అడ్డువరుసలు_పునాది[r] = a + (i + r) * lda;
            }
            కొరకు (పూర్ణాంకం blk = 0; blk < ఖండాలు; blk++) {
                vector గుర్తులేని పొట్టి hp_res[chunk_కొలత][4];
                కొరకు (పూర్ణాంకం r = 0; r < chunk_కొలత; r++) {
                    స్థిరరాశి ఖండం_q8_0 * b = అడ్డువరుసలు_పునాది[r] + blk;
                    vector దశాంశం v_scale = vec_వెలికితీయి_fp32_from_shorth(vec_splats(b->d));
                    vector గుర్తుగల అక్షరం c[2];
                    __vector_pair pair = __builtin_vsx_lxvp(0, (__vector_pair *)b->qs);
                    __builtin_vsx_disassemble_pair(c, & pair);
                    convert_and_scale_q8(c[0], v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c[1], v_scale, hp_res[r][2], hp_res[r][3]);
                }
                కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < 4; నిలువు++) {
                    ఒకవేళ constexpr (chunk_కొలత == 8) {
                        vec_t t[8];
                        t[0] = vec_perm((vec_t)hp_res[0][నిలువు], (vec_t)hp_res[1][నిలువు], swiz1);
                        t[1] = vec_perm((vec_t)hp_res[0][నిలువు], (vec_t)hp_res[1][నిలువు], swiz2);
                        t[2] = vec_perm((vec_t)hp_res[2][నిలువు], (vec_t)hp_res[3][నిలువు], swiz1);
                        t[3] = vec_perm((vec_t)hp_res[2][నిలువు], (vec_t)hp_res[3][నిలువు], swiz2);
                        t[4] = vec_perm((vec_t)hp_res[4][నిలువు], (vec_t)hp_res[5][నిలువు], swiz1);
                        t[5] = vec_perm((vec_t)hp_res[4][నిలువు], (vec_t)hp_res[5][నిలువు], swiz2);
                        t[6] = vec_perm((vec_t)hp_res[6][నిలువు], (vec_t)hp_res[7][నిలువు], swiz1);
                        t[7] = vec_perm((vec_t)hp_res[6][నిలువు], (vec_t)hp_res[7][నిలువు], swiz2);

                        vec_xst(vec_perm(t[0], t[2], swiz3), 0, (vec_t *)(vec_స్థానభ్రంశం + 0));
                        vec_xst(vec_perm(t[0], t[2], swiz4), 0, (vec_t *)(vec_స్థానభ్రంశం + 16));
                        vec_xst(vec_perm(t[1], t[3], swiz3), 0, (vec_t *)(vec_స్థానభ్రంశం + 32));
                        vec_xst(vec_perm(t[1], t[3], swiz4), 0, (vec_t *)(vec_స్థానభ్రంశం + 48));
                        vec_xst(vec_perm(t[4], t[6], swiz3), 0, (vec_t *)(vec_స్థానభ్రంశం + 64));
                        vec_xst(vec_perm(t[4], t[6], swiz4), 0, (vec_t *)(vec_స్థానభ్రంశం + 80));
                        vec_xst(vec_perm(t[5], t[7], swiz3), 0, (vec_t *)(vec_స్థానభ్రంశం + 96));
                        vec_xst(vec_perm(t[5], t[7], swiz4), 0, (vec_t *)(vec_స్థానభ్రంశం + 112));
                        vec_స్థానభ్రంశం += 128;
                    } లేకపోతే {
                        vec_t t0 = vec_perm((vec_t)hp_res[0][నిలువు], (vec_t)hp_res[1][నిలువు], swiz1);
                        vec_t t1 = vec_perm((vec_t)hp_res[0][నిలువు], (vec_t)hp_res[1][నిలువు], swiz2);
                        vec_t t2 = vec_perm((vec_t)hp_res[2][నిలువు], (vec_t)hp_res[3][నిలువు], swiz1);
                        vec_t t3 = vec_perm((vec_t)hp_res[2][నిలువు], (vec_t)hp_res[3][నిలువు], swiz2);

                        vec_xst(vec_perm(t0, t2, swiz3), 0, (vec_t *)(vec_స్థానభ్రంశం + 0));
                        vec_xst(vec_perm(t0, t2, swiz4), 0, (vec_t *)(vec_స్థానభ్రంశం + 16));
                        vec_xst(vec_perm(t1, t3, swiz3), 0, (vec_t *)(vec_స్థానభ్రంశం + 32));
                        vec_xst(vec_perm(t1, t3, swiz4), 0, (vec_t *)(vec_స్థానభ్రంశం + 48));
                        vec_స్థానభ్రంశం += 64;
                    }
                }
            }
        }
    }

    శూన్య pack_సాధారణ_q8_fp16(స్థిరరాశి ఖండం_q8_0 * a, int64_t lda, పూర్ణాంకం అడ్డువరుసలు, పూర్ణాంకం ఖండాలు, గుర్తులేని అక్షరం * vec) {
        ఒకవేళ (అడ్డువరుసలు == 4) {
            pack_q8_ఖండం<4>(a, lda, అడ్డువరుసలు, ఖండాలు, vec);
        } లేకపోతే {
            pack_q8_ఖండం<8>(a, lda, అడ్డువరుసలు, ఖండాలు, vec);
        }
    }

    మూస<పూర్ణాంకం size>
    శూన్య pack_సాధారణ_Int4(స్థిరరాశి TA * a, int64_t lda, పూర్ణాంకం అడ్డువరుసలు, పూర్ణాంకం నిలువుల_సంఖ్య_2, int8_t * vec, std::array<పూర్ణాంకం, size> & comparray) {
        int64_t i, j;
        TA * aoffset = ఖాళీ;
        int8_t * vec_స్థానభ్రంశం = ఖాళీ;
        TA * aoffset1 = ఖాళీ, * aoffset2 = ఖాళీ, * aoffset3 = ఖాళీ, * aoffset4 = ఖాళీ;
        TA * aoffset5 = ఖాళీ, * aoffset6 = ఖాళీ, * aoffset7 = ఖాళీ, * aoffset8 = ఖాళీ;
        vector గుర్తుగల అక్షరం c1[2] = {0}, c2[2] = {0}, c3[2] = {0}, c4[2] = {0};
        vector గుర్తుగల అక్షరం c5[2] = {0}, c6[2] = {0}, c7[2] = {0}, c8[2] = {0};
        aoffset = const_cast<TA *>(a);
        vec_స్థానభ్రంశం = vec;
        j = (అడ్డువరుసలు >> 3);
        ఒకవేళ (j > 0) {
            చేయి {
                aoffset1 = aoffset;
                aoffset2 = aoffset1 + lda;
                aoffset3 = aoffset2 + lda;
                aoffset4 = aoffset3 + lda;
                aoffset5 = aoffset4 + lda;
                aoffset6 = aoffset5 + lda;
                aoffset7 = aoffset6 + lda;
                aoffset8 = aoffset7 + lda;
                aoffset += 8 * lda;
                i = (నిలువుల_సంఖ్య_2 >> 2);
                ఒకవేళ (i > 0) {
                    చేయి {
                        c1[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset1->qs);
                        c2[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset2->qs);
                        c3[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset3->qs);
                        c4[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset4->qs);
                        c5[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset5->qs);
                        c6[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset6->qs);
                        c7[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset7->qs);
                        c8[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset8->qs);

                        process_q4_elements(c1, & comparray[0]);
                        process_q4_elements(c2, & comparray[1]);
                        process_q4_elements(c3, & comparray[2]);
                        process_q4_elements(c4, & comparray[3]);
                        process_q4_elements(c5, & comparray[4]);
                        process_q4_elements(c6, & comparray[5]);
                        process_q4_elements(c7, & comparray[6]);
                        process_q4_elements(c8, & comparray[7]);
                        సదిశ_క్రమమార్పు_store<int8_t, vector గుర్తుగల అక్షరం>(c1[0], c2[0], c3[0], c4[0], vec_స్థానభ్రంశం, అబద్ధం);
                        సదిశ_క్రమమార్పు_store<int8_t, vector గుర్తుగల అక్షరం>(c1[1], c2[1], c3[1], c4[1], vec_స్థానభ్రంశం + 64, అబద్ధం);
                        సదిశ_క్రమమార్పు_store<int8_t, vector గుర్తుగల అక్షరం>(c5[0], c6[0], c7[0], c8[0], vec_స్థానభ్రంశం + 128, అబద్ధం);
                        సదిశ_క్రమమార్పు_store<int8_t, vector గుర్తుగల అక్షరం>(c5[1], c6[1], c7[1], c8[1], vec_స్థానభ్రంశం + 192, అబద్ధం);
                        aoffset1 += lda;
                        aoffset2 += lda;
                        aoffset3 += lda;
                        aoffset4 += lda;
                        aoffset5 += lda;
                        aoffset6 += lda;
                        aoffset7 += lda;
                        aoffset8 += lda;
                        vec_స్థానభ్రంశం += 256;
                        i--;
                    } అయితే (i > 0);
                }
                j--;
            } అయితే (j > 0);
        }

        ఒకవేళ (అడ్డువరుసలు & 4) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset += 4 * lda;
            i = (నిలువుల_సంఖ్య_2 >> 2);
            ఒకవేళ (i > 0) {
                చేయి {
                    c1[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset1->qs);
                    c2[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset2->qs);
                    c3[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset3->qs);
                    c4[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset4->qs);

                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    సదిశ_క్రమమార్పు_store<int8_t, vector గుర్తుగల అక్షరం>(c1[0], c2[0], c3[0], c4[0], vec_స్థానభ్రంశం, అబద్ధం);
                    సదిశ_క్రమమార్పు_store<int8_t, vector గుర్తుగల అక్షరం>(c1[1], c2[1], c3[1], c4[1], vec_స్థానభ్రంశం + 64, అబద్ధం);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    aoffset4 += lda;
                    vec_స్థానభ్రంశం += 128;
                    i--;
                } అయితే (i > 0);
            }
        }

        ఒకవేళ (అడ్డువరుసలు & 3) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            i = (నిలువుల_సంఖ్య_2 >> 2);
            ఒకవేళ (i > 0) {
                చేయి {
                    ఎంపిక(అడ్డువరుసలు) {
                        స్థితి 3: c3[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset3->qs);
                        స్థితి 2: c2[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset2->qs);
                        స్థితి 1: c1[1] = vec_xl(0, (స్థిరరాశి vector గుర్తుగల అక్షరం *)aoffset1->qs);
                            ఆపు;
                    }
                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    సదిశ_క్రమమార్పు_store<int8_t, vector గుర్తుగల అక్షరం>(c1[0], c2[0], c3[0], c4[0], vec_స్థానభ్రంశం, అబద్ధం);
                    సదిశ_క్రమమార్పు_store<int8_t, vector గుర్తుగల అక్షరం>(c1[1], c2[1], c3[1], c4[1], vec_స్థానభ్రంశం + 64, అబద్ధం);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    vec_స్థానభ్రంశం += 128;
                    i--;
                } అయితే(i > 0);
            }
        }
    }

    మూస<రకం_పేరు VA, రకం_పేరు VB>
    శూన్య pack_సాధారణ(స్థిరరాశి ఖండం_q8_0 * a, int64_t lda, పూర్ణాంకం అడ్డువరుసలు, పూర్ణాంకం నిలువుల_సంఖ్య_2, VA * vec, తార్కిక flip) {
        int64_t i, j;
        ఖండం_q8_0 * aoffset = ఖాళీ;
        VA * vec_స్థానభ్రంశం = ఖాళీ;
        ఖండం_q8_0 * aoffsets[8];
        __vector_pair arr[8];
        VB c[8][2] = {0};
        VB c1[8] = {0}; VB c2[8] = {0};
        aoffset = const_cast<ఖండం_q8_0 *>(a);
        vec_స్థానభ్రంశం = vec;
        j = (అడ్డువరుసలు >> 3);
        ఒకవేళ (j > 0) {
            చేయి {
                aoffsets[0] = aoffset;
                కొరకు (పూర్ణాంకం it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it - 1] + lda;
                aoffset += 8 * lda;

                i = (నిలువుల_సంఖ్య_2 >> 3);
                ఒకవేళ (i > 0) {
                చేయి {
                    కొరకు (పూర్ణాంకం it = 0; it < 8; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    సదిశ_క్రమమార్పు_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_స్థానభ్రంశం, flip);
                    సదిశ_క్రమమార్పు_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_స్థానభ్రంశం + 64, flip);
                    సదిశ_క్రమమార్పు_store<VA, VB>(c1[4], c1[5], c1[6], c1[7], vec_స్థానభ్రంశం + 128, flip);
                    సదిశ_క్రమమార్పు_store<VA, VB>(c2[4], c2[5], c2[6], c2[7], vec_స్థానభ్రంశం + 192, flip);
                    కొరకు (పూర్ణాంకం it = 0; it < 8; it++)
                        aoffsets[it] += lda;
                    vec_స్థానభ్రంశం += 256;
                    i--;
               } అయితే(i > 0);
            }
            j--;
        } అయితే(j > 0);
    }
    ఒకవేళ (అడ్డువరుసలు & 4) {
            aoffsets[0]  = aoffset;
            కొరకు (పూర్ణాంకం it = 1; it < 4; it++ )
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
        i = (నిలువుల_సంఖ్య_2 >> 3);
            ఒకవేళ (i > 0) {
               చేయి {
                    కొరకు (పూర్ణాంకం it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    సదిశ_క్రమమార్పు_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_స్థానభ్రంశం, flip);
                    సదిశ_క్రమమార్పు_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_స్థానభ్రంశం + 64, flip);
                    కొరకు (పూర్ణాంకం it = 0; it < 4; it++) {
                        aoffsets[it] += lda;
                    }
                    vec_స్థానభ్రంశం += 128;
                    i--;
               } అయితే(i > 0);
            }
        }

        ఒకవేళ (అడ్డువరుసలు & 3) {
            aoffsets[0]  = aoffset;
            కొరకు (పూర్ణాంకం it = 1; it < 3; it++ )
                aoffsets[it] = aoffsets[it - 1] + lda;
            i = (నిలువుల_సంఖ్య_2 >> 3);
            ఒకవేళ (i > 0) {
                చేయి {
                    ఎంపిక(అడ్డువరుసలు) {
                        స్థితి 3: arr[2] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[2]->qs);
                                __builtin_vsx_disassemble_pair(c[2], & arr[2]);
                                c1[2] = c[2][0]; c2[2] = c[2][1];
                        స్థితి 2: arr[1] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[1]->qs);
                                __builtin_vsx_disassemble_pair(c[1], & arr[1]);
                                c1[1] = c[1][0]; c2[1] = c[1][1];
                        స్థితి 1: arr[0] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[0]->qs);
                                __builtin_vsx_disassemble_pair(c[0], & arr[0]);
                                c1[0] = c[0][0]; c2[0] = c[0][1];
                                ఆపు;
                    }
                    సదిశ_క్రమమార్పు_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_స్థానభ్రంశం, flip);
                    సదిశ_క్రమమార్పు_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_స్థానభ్రంశం + 64, flip);
                    కొరకు (పూర్ణాంకం it = 0; it < 3; it++)
                         aoffsets[it] += lda;
                    vec_స్థానభ్రంశం += 128;
                    i--;
               } అయితే(i > 0);
            }
        }
    }

    శూన్య mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        పూర్ణాంకం m_rem = MIN(m - m0, 16);
        పూర్ణాంకం n_rem = MIN(n - n0, 16);

        పూర్ణాంకం mc = 0, nc = 0;

        ఒకవేళ (m_rem >= 8 && n_rem >= 8) {
           mc = 8;
           nc = 8;
           gemm<8, 8>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small(m0, m, n0, n, mc, nc);
        } లేకపోతే {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            ఒకవేళ (mc == 0 || nc == 0)
               తిరిగి;
            gemm_small(m0, m, n0, n, mc, nc);
        }

        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }


    శూన్య KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[16] = {0};
        acc_t acc_0, acc_1;
        std::array<పూర్ణాంకం, 4> comparray {};
        vector దశాంశం fin_res[8] = {0};
        vector దశాంశం vs[8] = {0};
        తార్కిక isAblock_q4 = std::is_same_v<TA, ఖండం_q4_0>;
        కొరకు (పూర్ణాంకం l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            ఒకవేళ (std::is_same_v<TA, ఖండం_q4_0>) {
               pack_సాధారణ_Int4<4>((A + (ii * lda) + l), lda, 4, 4, (int8_t *)vec_A, comparray);
            } లేకపోతే {
               pack_సాధారణ<int8_t, vector గుర్తుగల అక్షరం>((స్థిరరాశి ఖండం_q8_0 *)(A + (ii * lda) + l), lda, 4, 8, (int8_t *)vec_A, అబద్ధం);
            }
            pack_సాధారణ<uint8_t, vector గుర్తులేని అక్షరం>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, నిజం);
            కొరకు(పూర్ణాంకం x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x], vec_B[x+8]);
            }
            కొరకు (పూర్ణాంకం I = 0; I<4; I++) {
                కొరకు (పూర్ణాంకం J = 0; J<4; J++) {
                    *((దశాంశం *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((దశాంశం *)& vs[I + 4] + J) = (unhalf((A +((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            ఒకవేళ (!isAblock_q4) {
                స్వయంచాలక aoffset = A + (ii * lda) + l;
                కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
                    comparray[i] = 0;
                    పూర్ణాంకం ca = 0;
                    స్వయంచాలక *at = aoffset->qs;
                    కొరకు (పూర్ణాంకం j = 0; j < 32; j++)
                        ca += (పూర్ణాంకం)*at++;
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

    శూన్య KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[8] = {0};
        acc_t acc_0, acc_1;
        std::array<పూర్ణాంకం, 8> comparray {};
        vector దశాంశం fin_res[8] = {0};
        vector దశాంశం vs[8] = {0};
        తార్కిక isAblock_q4 = std::is_same_v<TA, ఖండం_q4_0>;
        కొరకు (పూర్ణాంకం l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            ఒకవేళ (std::is_same_v<TA, ఖండం_q4_0>) {
               pack_సాధారణ_Int4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } లేకపోతే {
               pack_సాధారణ<int8_t, vector గుర్తుగల అక్షరం>((స్థిరరాశి ఖండం_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, అబద్ధం);
            }
            pack_సాధారణ<uint8_t, vector గుర్తులేని అక్షరం>((B + (jj * ldb) + l), ldb, 4, 8, (uint8_t *)vec_B, నిజం);
            కొరకు(పూర్ణాంకం x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
            }
            కొరకు (పూర్ణాంకం I = 0; I < 8; I++) {
                కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                    *((దశాంశం *)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                }
            }
            ఒకవేళ (!isAblock_q4) {
                స్వయంచాలక aoffset = A + (ii * lda) + l;
                కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    పూర్ణాంకం ca = 0;
                    స్వయంచాలక *at = aoffset->qs;
                    కొరకు (పూర్ణాంకం j = 0; j < 32; j++)
                        ca += (పూర్ణాంకం)*at++;
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

    శూన్య KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16] = {0};
        acc_t acc_0, acc_1, acc_2, acc_3;
        acc_t acc_4, acc_5, acc_6, acc_7;
        std::array<పూర్ణాంకం, 8> comparray {};
        vector దశాంశం fin_res[16] = {0};
        vector దశాంశం vs[16] = {0};
        తార్కిక isAblock_q4 = std::is_same_v<TA, ఖండం_q4_0>;
        కొరకు (పూర్ణాంకం l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            __builtin_mma_xxsetaccz(& acc_2);
            __builtin_mma_xxsetaccz(& acc_3);
            ఒకవేళ (std::is_same_v<TA, ఖండం_q4_0>) {
               pack_సాధారణ_Int4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } లేకపోతే {
               pack_సాధారణ<int8_t, vector గుర్తుగల అక్షరం>((స్థిరరాశి ఖండం_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, అబద్ధం);
            }
            pack_సాధారణ<uint8_t, vector గుర్తులేని అక్షరం>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, నిజం);
            కొరకు(పూర్ణాంకం x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_2, vec_A[x], vec_B[x + 8]);
                __builtin_mma_xvi8ger4pp(& acc_3, vec_A[x + 8], vec_B[x + 8]);
            }
            కొరకు (పూర్ణాంకం I = 0; I < 8 ; I++) {
                కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                    *((దశాంశం *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((దశాంశం *)& vs[I + 8] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            ఒకవేళ (!isAblock_q4) {
                స్వయంచాలక aoffset = A + (ii * lda) + l;
                కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    పూర్ణాంకం ca = 0;
                    స్వయంచాలక *at = aoffset->qs;
                    కొరకు (పూర్ణాంకం j = 0; j < 32; j++)
                        ca += (పూర్ణాంకం)*at++;
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

    శూన్య KERNEL_Q0(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, int64_t l, vec_t * vec_A, vec_t * vec_B) {
        acc_t acc[8];
        కొరకు (పూర్ణాంకం i = 0; i < mc ; i += 16) {
            కొరకు (పూర్ణాంకం j = 0; j < nc; j += 8) {
                పూర్ణాంకం A0_పునాది = (i / 16) * (2 * 32 * kc);
                పూర్ణాంకం B0_పునాది = (j / 8) * (32 * kc);
                కొరకు (పూర్ణాంకం x = 0; x < 8; x++) {
                     __builtin_mma_xxsetaccz(&acc[x]);
                }
                కొరకు (int64_t kk = 0; kk < kc; kk++) {
                    పూర్ణాంకం A0_ఖండం_idx = A0_పునాది + kk * 32;
                    పూర్ణాంకం B0_ఖండం_idx = B0_పునాది + kk * 32;
                    పూర్ణాంకం A1_ఖండం_idx = A0_ఖండం_idx + 32 * kc;
                    పూర్ణాంకం B1_ఖండం_idx = B0_ఖండం_idx + 32 * kc;
                    vec_t * A0_ఖండం = & vec_A[A0_ఖండం_idx];
                    vec_t * B0_ఖండం = & vec_B[B0_ఖండం_idx];
                    vec_t * A1_ఖండం = & vec_A[A1_ఖండం_idx];
                    కొరకు (పూర్ణాంకం it = 0; it < 4; it++) {
                        కొరకు (పూర్ణాంకం x = 0; x < 4; x++) {
                            __builtin_mma_xvf16ger2pp(& acc[0], A0_ఖండం[8 * it + x], B0_ఖండం[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[1], A0_ఖండం[8 * it + x], B0_ఖండం[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[2], A0_ఖండం[8 * it + x + 4], B0_ఖండం[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[3], A0_ఖండం[8 * it + x + 4], B0_ఖండం[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[4], A1_ఖండం[8 * it + x], B0_ఖండం[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[5], A1_ఖండం[8 * it + x], B0_ఖండం[8 * it+ x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[6], A1_ఖండం[8 * it + x + 4], B0_ఖండం[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[7], A1_ఖండం[8 * it + x + 4], B0_ఖండం[8 * it + x + 4]);
                        }
                    }
                }
                ఒకవేళ (l == 0) {
                    save_acc(& acc[0], ii + i, jj + j);
                    save_acc(& acc[1], ii + i, jj + j + 4);
                    save_acc(& acc[2], ii + i + 4, jj + j);
                    save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    save_acc(& acc[4], ii + i + 8, jj + j);
                    save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    save_acc(& acc[6], ii + i + 12, jj + j);
                    save_acc(& acc[7], ii + i + 12, jj + j + 4);
                } లేకపోతే {
                    కూడు_save_acc(& acc[0], ii + i, jj + j);
                    కూడు_save_acc(& acc[1], ii + i, jj + j + 4);
                    కూడు_save_acc(& acc[2], ii + i + 4, jj + j);
                    కూడు_save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    కూడు_save_acc(& acc[4], ii + i + 8, jj + j);
                    కూడు_save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    కూడు_save_acc(& acc[6], ii + i + 12, jj + j);
                    కూడు_save_acc(& acc[7], ii + i + 12, jj + j + 4);
                }
            }
        }
    }

    శూన్య matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        vec_t A_pack[mc * kc * 4];
        vec_t B_pack[nc * kc * 4];
        constexpr తార్కిక is_Ablock_q4 = std::is_same_v<TA, ఖండం_q4_0>;
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles  = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles) {
            end = tiles;
        }
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = (పని / xtiles) * mc;
            int64_t jj = (పని % xtiles) * nc;
            కొరకు (int64_t kk = 0; kk < k; kk += kc) {
                int64_t k_cur = MIN(kc, k - kk);
                ఒకవేళ constexpr(is_Ablock_q4) {
                    pack_సాధారణ_q4_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                } లేకపోతే {
                    pack_సాధారణ_q8_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                }
                pack_సాధారణ_q8_fp16(B + jj * ldb + kk, ldb, nc, k_cur, (uint8_t *)B_pack);
                KERNEL_Q0(ii, jj, mc, nc, k_cur, kk, A_pack, B_pack);
            }
        }
    }

    శూన్య gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, పూర్ణాంకం RM, పూర్ణాంకం RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        vec_t vec_A[8] = {0}, vec_B[8] = {0};
        vector గుర్తుగల పూర్ణాంకం vec_C[4];
        acc_t acc_0;
        తార్కిక isAblock_q4 = std::is_same_v<TA, ఖండం_q4_0>;

        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * RN;
            std::array<పూర్ణాంకం, 4> comparray{};
            vector దశాంశం res[4] = {0};
            vector దశాంశం fin_res[4] = {0};
            vector దశాంశం vs[4] = {0};
            vector దశాంశం CA[4] = {0};
            __builtin_prefetch((A + (ii * lda) + 0)->qs, 0, 1); // prefetch first value
            __builtin_prefetch((B + (jj * ldb) + 0)->qs, 0, 1); // prefetch first value
            కొరకు (పూర్ణాంకం l = 0; l < k; l++) {
                __builtin_prefetch((A + (ii * lda) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_prefetch((B + (jj * ldb) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_mma_xxsetaccz(& acc_0);
                ఒకవేళ (isAblock_q4) {
                    pack_సాధారణ_Int4<4>((A + (ii * lda) + l), lda, RM, 4, (int8_t *)vec_A, comparray);
                } లేకపోతే {
                    pack_సాధారణ<int8_t, vector గుర్తుగల అక్షరం>((స్థిరరాశి ఖండం_q8_0 *)(A + (ii * lda) + l), lda, RM, 8, (int8_t *)vec_A, అబద్ధం);
                }
                pack_సాధారణ<uint8_t, vector గుర్తులేని అక్షరం>((B + (jj * ldb) + l), ldb, RN, 8, (uint8_t *)vec_B, నిజం);
                కొరకు (పూర్ణాంకం x = 0; x < 8; x += 4) {
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 1], vec_B[x + 1]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 2], vec_B[x + 2]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 3], vec_B[x + 3]);
                }
                కొరకు (పూర్ణాంకం I = 0; I < RM; I++) {
                    కొరకు (పూర్ణాంకం J = 0; J < RN; J++) {
                        *((దశాంశం*)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    }
                }
                __builtin_mma_disassemble_acc(vec_C, & acc_0);
                ఒకవేళ (!isAblock_q4) {
                    స్వయంచాలక aoffset = A + (ii * lda) + l;
                    కొరకు (పూర్ణాంకం i = 0; i < RM; i++) {
                        comparray[i] = 0;
                        పూర్ణాంకం ca = 0;
                        స్వయంచాలక *at = aoffset->qs;
                        కొరకు (పూర్ణాంకం j = 0; j < 32; j++)
                            ca += (పూర్ణాంకం)*at++;
                        comparray[i] = ca;
                        aoffset += lda;
                    }
                }
                కొరకు (పూర్ణాంకం i = 0; i < RM; i++) {
                    CA[i] = vec_splats((దశాంశం)(((ద్విగుణ)comparray[i]) * -128.0));
                    res[i] = vec_కూడు(vec_ctf(vec_C[i], 0), CA[i]);
                    fin_res[i] = vec_madd(res[i], vs[i], fin_res[i]);
                }
            }
            save_res(ii, jj, 0, fin_res, RM, RN);
        }
    }

    మూస<పూర్ణాంకం RM, పూర్ణాంకం RN>
    అంతఃవరుస శూన్య kernel(int64_t ii, int64_t jj) {
        ఒకవేళ constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii,jj);
        } లేకపోతే ఒకవేళ constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii,jj);
        } లేకపోతే ఒకవేళ constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii,jj);
        } లేకపోతే {
            ప్రతిజ్ఞా(అబద్ధం && "RN/RM values not supported");
        }
    }

    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN>
    NOINLINE శూన్య gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }
    స్థిరరాశి TA * స్థిరరాశి A;
    స్థిరరాశి ఖండం_q8_0 * స్థిరరాశి B;
    దశాంశం * C;
    స్థిరరాశి int64_t k;
    int64_t kc;
    స్థిరరాశి int64_t lda;
    స్థిరరాశి int64_t ldb;
    స్థిరరాశి int64_t ldc;
    స్థిరరాశి పూర్ణాంకం ith;
    స్థిరరాశి పూర్ణాంకం nth;
};

తరగతి tinyBLAS_PPC {
  బహిరంగ:
    tinyBLAS_PPC(int64_t k,
                స్థిరరాశి దశాంశం * A, int64_t lda,
                స్థిరరాశి దశాంశం * B, int64_t ldb,
                దశాంశం * C, int64_t ldc,
                పూర్ణాంకం ith, పూర్ణాంకం nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    శూన్య matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #ఒకవేళ defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #అయితే_ముగింపు
        ఒకవేళ (m % mc == 0 && n % nc == 0 && k % kc == 0) {
            matmul_tiled(m, n, mc, nc, kc);
        } లేకపోతే {
            mnpack(0, m, 0, n);
        }
    }

  అంతరంగ:

    __attribute__((always_inline))
    అంతఃవరుస శూన్య save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        కొరకు (పూర్ణాంకం I = 0; I < 4; I++) {
            కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                *((దశాంశం *)(C+ii+((jj+J)*ldc)+I)) = *((దశాంశం *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    అంతఃవరుస శూన్య కూడు_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        కొరకు (పూర్ణాంకం I = 0; I < 4; I++) {
            కొరకు (పూర్ణాంకం J = 0; J < 4; J++) {
                దశాంశం * c_ptr = (దశాంశం *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((దశాంశం *)&vec_C[I]+J);
            }
        }
    }

    అంతఃవరుస శూన్య సదిశ_క్రమమార్పు_store_4(vector దశాంశం * src, దశాంశం * vec_స్థానభ్రంశం) {
        vector దశాంశం t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergel(src[0], src[1]);
        t4 = vec_mergel(src[2], src[3]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t1, t2, 3);
        t7 = vec_xxpermdi(t3, t4, 0);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_స్థానభ్రంశం);
        vec_xst(t6, 0, vec_స్థానభ్రంశం + 4);
        vec_xst(t7, 0, vec_స్థానభ్రంశం + 8);
        vec_xst(t8, 0, vec_స్థానభ్రంశం + 12);
    }

    అంతఃవరుస శూన్య సదిశ_క్రమమార్పు_store_8(vector దశాంశం * src, దశాంశం * vec_స్థానభ్రంశం) {
        vector దశాంశం t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergeh(src[4], src[5]);
        t4 = vec_mergeh(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_స్థానభ్రంశం);
        vec_xst(t6, 0, vec_స్థానభ్రంశం + 4);
        vec_xst(t7, 0, vec_స్థానభ్రంశం + 8);
        vec_xst(t8, 0, vec_స్థానభ్రంశం + 12);

        t1 = vec_mergel(src[0], src[1]);
        t2 = vec_mergel(src[2], src[3]);
        t3 = vec_mergel(src[4], src[5]);
        t4 = vec_mergel(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_స్థానభ్రంశం + 16);
        vec_xst(t6, 0, vec_స్థానభ్రంశం + 20);
        vec_xst(t7, 0, vec_స్థానభ్రంశం + 24);
        vec_xst(t8, 0, vec_స్థానభ్రంశం + 28);
    }

    శూన్య packTranspose(స్థిరరాశి దశాంశం * a, int64_t lda, పూర్ణాంకం అడ్డువరుసలు, పూర్ణాంకం నిలువుల_సంఖ్య_2, దశాంశం * vec) {
        int64_t i, j;
        దశాంశం * aoffsets[8];
        దశాంశం * aoffset = ఖాళీ, * boffset = ఖాళీ;
        __vector_pair arr[8];
        vector దశాంశం c[8][2] = {0};
        vector దశాంశం c1[8] = {0};
        vector దశాంశం c2[8] = {0};
        aoffset = const_cast<దశాంశం *>(a);
        boffset = vec;
        j = (అడ్డువరుసలు >> 3);
        ఒకవేళ (j > 0) {
            చేయి {
                aoffsets[0] = aoffset;
                కొరకు (పూర్ణాంకం it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it-1] + lda;
                aoffset += 8 * lda;
                i = (నిలువుల_సంఖ్య_2 >> 3);
                ఒకవేళ (i > 0) {
                    చేయి {
                        కొరకు (పూర్ణాంకం it = 0; it < 8; it++) {
                            arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                            __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                            c1[it] = c[it][0];
                            c2[it] = c[it][1];
                        }

                        సదిశ_క్రమమార్పు_store_8(c1, boffset);
                        సదిశ_క్రమమార్పు_store_8(c2, boffset + 32);
                        boffset += 64;
                        i--;
                        ఒకవేళ (i > 0) {
                           కొరకు (పూర్ణాంకం it = 0; it < 8; it++) {
                               aoffsets[it] = aoffsets[it] + 8;
                           }
                        }
                    } అయితే(i > 0);
                }
                ఒకవేళ (నిలువుల_సంఖ్య_2 & 4) {
                    కొరకు (పూర్ణాంకం it = 0; it < 8 ; it++)
                        c1[it] = vec_xl(0, aoffsets[it]);
                    సదిశ_క్రమమార్పు_store_8(c1, boffset);
                }
            j--;
            } అయితే(j > 0);
        }

        ఒకవేళ (అడ్డువరుసలు & 4) {
            aoffsets[0] = aoffset;
            కొరకు (పూర్ణాంకం it = 1; it < 4; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            i = (నిలువుల_సంఖ్య_2 >> 3);
            ఒకవేళ (i > 0) {
                చేయి {
                    కొరకు (పూర్ణాంకం it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                        __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    సదిశ_క్రమమార్పు_store_4(c1, boffset);
                    సదిశ_క్రమమార్పు_store_4(c2, boffset + 16);
                    కొరకు (పూర్ణాంకం it = 0; it < 4; it++)
                        aoffsets[it] += 8 * lda;
                    boffset += 32;
                    i--;
                } అయితే(i > 0);
            }

            ఒకవేళ (నిలువుల_సంఖ్య_2 & 4) {
               కొరకు (పూర్ణాంకం it = 0; it < 4; it++)
                   c1[it] = vec_xl(0, aoffsets[it]);
                సదిశ_క్రమమార్పు_store_4(c1, boffset);
            }
        }
        ఒకవేళ (అడ్డువరుసలు & 3) {
            aoffsets[0] = aoffset;
            కొరకు (పూర్ణాంకం it = 1; it < 3; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            ఒకవేళ (నిలువుల_సంఖ్య_2 & 4) {
                కొరకు (పూర్ణాంకం it = 0; it < 3; it++)
                    c1[it] = vec_xl(0, aoffsets[it]);
                సదిశ_క్రమమార్పు_store_4(c1, boffset);
            }
        }
    }

    శూన్య KERNEL_4x4(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[4], vec_C[4];
        acc_t acc_0;
        __builtin_mma_xxsetaccz(&acc_0);
        కొరకు (పూర్ణాంకం l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (దశాంశం *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (దశాంశం *)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
        }
        save_acc(&acc_0, ii, jj);
    }

    శూన్య KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        కొరకు (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (దశాంశం *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 4, (దశాంశం *)vec_B);
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

    శూన్య KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        కొరకు (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 8, 4, (దశాంశం *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (దశాంశం *)vec_B);
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

    శూన్య KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        కొరకు (పూర్ణాంకం l = 0; l < k; l+=8) {
            packTranspose(A + (ii * lda) + l, lda, 8, 8, (దశాంశం *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 8, (దశాంశం *)vec_B);
            కొరకు(పూర్ణాంకం x = 0; x < 16; x+=2) {
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

    అంతఃవరుస శూన్య MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        కొరకు (పూర్ణాంకం x = 0; x < 16; x += 2) {
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

    శూన్య KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        కొరకు (int64_t i = 0; i < mc; i += 16) {
            పూర్ణాంకం A_పునాది_addr = (mc / 8) * (i / 8) * 16;
            కొరకు (int64_t j = 0; j < nc; j += 8) {
                 పూర్ణాంకం B_పునాది_addr = (nc / 8) * (j / 8) * 16;
                 acc_t acc[8];
                 vec_t A0_ఖండం[16]; vec_t A1_ఖండం[16];
                 కొరకు (పూర్ణాంకం x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 కొరకు (int64_t l = 0; l < kc; l += 8) {
                     పూర్ణాంకం A0_ఖండం_idx = A_పునాది_addr + (l / 8) * 16;
                     పూర్ణాంకం A1_ఖండం_idx = A0_ఖండం_idx + (mc / 8) * 16;
                     పూర్ణాంకం B_ఖండం_idx = B_పునాది_addr + (l / 8) * 16;
                     vec_t* A0_ఖండం = &vec_A[A0_ఖండం_idx];
                     vec_t* A1_ఖండం = &vec_A[A1_ఖండం_idx];
                     vec_t* B_ఖండం = &vec_B[B_ఖండం_idx];
                     MMA_16x8(A0_ఖండం, A1_ఖండం, B_ఖండం, acc);
                 }
                 ఒకవేళ (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } లేకపోతే {
                     కూడు_save_acc(&acc[0], ii + i, jj + j);
                     కూడు_save_acc(&acc[1], ii + i, jj + j + 4);
                     కూడు_save_acc(&acc[2], ii + i + 4, jj + j);
                     కూడు_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     కూడు_save_acc(&acc[4], ii + i + 8, jj + j);
                     కూడు_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     కూడు_save_acc(&acc[6], ii + i + 12, jj + j);
                     కూడు_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }

    శూన్య matmul_tiled(int64_t m , int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles) {
            end = tiles;
        }
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = (పని / xtiles) * mc;
            int64_t jj = (పని % xtiles) * nc;
            కొరకు (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 4];
                 vec_t B_pack[kc * nc / 4];
                 packTranspose(A + (ii * lda) + kk, lda, kc, mc, (దశాంశం *)A_pack);
                 packTranspose(B + (jj * ldb) + kk, ldb, kc, nc, (దశాంశం *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    శూన్య mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        పూర్ణాంకం m_rem = MIN(m - m0, 8);
        పూర్ణాంకం n_rem = MIN(n - n0, 8);
        పూర్ణాంకం mc = 0, nc = 0;
        ఒకవేళ (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8, 8>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } లేకపోతే ఒకవేళ (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n);
        } లేకపోతే {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            ఒకవేళ (mc == 0 || nc == 0)
                తిరిగి;
            gemm_small(m0, m, n0, n, mc, nc);
        }
        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    శూన్య gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, పూర్ణాంకం RM, పూర్ణాంకం RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[4] = {0}, vec_B[4] = {0};
            కొరకు (పూర్ణాంకం l = 0; l < k; l += 4) {
                /* 'GEMV Forwarding' concept is used in first two conditional loops.
                 * when one of the matrix has a single row/column, the elements are
                 * broadcasted, instead of using packing routine to prepack the
                 * matrix elements.
                 */
                ఒకవేళ (RM == 1) {
                    దశాంశం * a = const_cast<దశాంశం *>(A + (ii) * lda + l);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (దశాంశం *)vec_B);
                    vec_A[0] = (vec_t)vec_xl(0,a);
                    vec_A[1] = (vec_t)vec_splats(*((దశాంశం *)&vec_A+1));
                    vec_A[2] = (vec_t)vec_splats(*((దశాంశం *)&vec_A+2));
                    vec_A[3] = (vec_t)vec_splats(*((దశాంశం *)&vec_A+3));
                } లేకపోతే ఒకవేళ (RN == 1) {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (దశాంశం *)vec_A);
                    దశాంశం * b = const_cast<దశాంశం *>(B + (jj) * ldb + l);
                    vec_B[0] = (vec_t)vec_xl(0,b);
                    vec_B[1] = (vec_t)vec_splats(*((దశాంశం *)&vec_B+1));
                    vec_B[2] = (vec_t)vec_splats(*((దశాంశం *)&vec_B+2));
                    vec_B[3] = (vec_t)vec_splats(*((దశాంశం *)&vec_B+3));
                } లేకపోతే {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (దశాంశం *)vec_A);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (దశాంశం *)vec_B);
                }
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            కొరకు (పూర్ణాంకం I = 0; I < RM; I++) {
                కొరకు (పూర్ణాంకం J = 0; J < RN; J++) {
                    *((దశాంశం *)(C+ii+((jj+J)*ldc)+I)) = *((దశాంశం *)&vec_C[I]+J);
                }
            }
       }
    }

    మూస<పూర్ణాంకం RM, పూర్ణాంకం RN>
    అంతఃవరుస శూన్య kernel(int64_t ii, int64_t jj) {
        ఒకవేళ constexpr(RM == 4 && RN == 4) {
            KERNEL_4x4(ii, jj);
        } లేకపోతే ఒకవేళ constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii, jj);
        } లేకపోతే ఒకవేళ constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii, jj);
        } లేకపోతే ఒకవేళ constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii, jj);
        } లేకపోతే {
            static_assert(అబద్ధం, "RN/RM values not supported");
        }
    }

    మూస <పూర్ణాంకం RM, పూర్ణాంకం RN>
    NOINLINE శూన్య gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        ఒకవేళ (end > tiles)
            end = tiles;
        కొరకు (int64_t పని = start; పని < end; ++పని) {
            int64_t ii = m0 + పని / xtiles * RM;
            int64_t jj = n0 + పని % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    స్థిరరాశి దశాంశం * స్థిరరాశి A;
    స్థిరరాశి దశాంశం * స్థిరరాశి B;
    దశాంశం * C;
    స్థిరరాశి int64_t k;
    స్థిరరాశి int64_t lda;
    స్థిరరాశి int64_t ldb;
    స్థిరరాశి int64_t ldc;
    స్థిరరాశి పూర్ణాంకం ith;
    స్థిరరాశి పూర్ణాంకం nth;
};
#అయితే_ముగింపు
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
తార్కిక llamafile_sgemm(స్థిరరాశి నిర్మాణం ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                     స్థిరరాశి శూన్య *A, int64_t lda, స్థిరరాశి శూన్య *B, int64_t ldb, శూన్య *C,
                     int64_t ldc, పూర్ణాంకం Atype, పూర్ణాంకం Btype, పూర్ణాంకం Ctype) {

    ప్రతిజ్ఞా(m >= 0);
    ప్రతిజ్ఞా(n >= 0);
    ప్రతిజ్ఞా(k >= 0);
    ప్రతిజ్ఞా(lda >= k);
    ప్రతిజ్ఞా(ldb >= k);
    ప్రతిజ్ఞా(ldc >= m);
    ప్రతిజ్ఞా(params->nth > 0);
    ప్రతిజ్ఞా(params->ith < params->nth);

    // only enable sgemm for prompt processing
#ఒకవేళ !defined(__MMA__)
    ఒకవేళ (n < 2)
        తిరిగి అబద్ధం;
#అయితే_ముగింపు

    ఒకవేళ (Ctype != GGML_TYPE_F32)
        తిరిగి అబద్ధం;

    ఎంపిక (Atype) {

    స్థితి GGML_TYPE_F32: {
        ఒకవేళ (Btype != GGML_TYPE_F32)
            తిరిగి అబద్ధం;
#ఒకవేళ defined(__AVX512F__)
        tinyBLAS<16, __m512, __m512, దశాంశం, దశాంశం, దశాంశం> tb{ params,
            k, (స్థిరరాశి దశాంశం *)A, lda,
            (స్థిరరాశి దశాంశం *)B, ldb,
            (దశాంశం *)C, ldc};
        తిరిగి tb.matmul(m, n);
#లేదావేళ defined(__AVX__) || defined(__AVX2__)
        tinyBLAS<8, __m256, __m256, దశాంశం, దశాంశం, దశాంశం> tb{ params,
            k, (స్థిరరాశి దశాంశం *)A, lda,
            (స్థిరరాశి దశాంశం *)B, ldb,
            (దశాంశం *)C, ldc};
        తిరిగి tb.matmul(m, n);
#లేదావేళ defined(__ARM_NEON)
        ఒకవేళ (n < 4)
            తిరిగి అబద్ధం;
        tinyBLAS<4, float32x4_t, float32x4_t, దశాంశం, దశాంశం, దశాంశం> tb{ params,
            k, (స్థిరరాశి దశాంశం *)A, lda,
            (స్థిరరాశి దశాంశం *)B, ldb,
            (దశాంశం *)C, ldc};
        తిరిగి tb.matmul(m, n);
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
        ఒకవేళ (n < 4)
            తిరిగి అబద్ధం;
        tinyBLAS<4, float32x4_t, float32x4_t, దశాంశం, దశాంశం, దశాంశం> tb{ params,
            k, (స్థిరరాశి దశాంశం *)A, lda,
            (స్థిరరాశి దశాంశం *)B, ldb,
            (దశాంశం *)C, ldc};
        తిరిగి tb.matmul(m, n);
#లేదావేళ defined(__MMA__)
        ఒకవేళ (k % 8)
            తిరిగి అబద్ధం;
        tinyBLAS_PPC tb{
            k, (స్థిరరాశి దశాంశం *)A, lda,
            (స్థిరరాశి దశాంశం *)B, ldb,
            (దశాంశం *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        తిరిగి నిజం;
#లేదావేళ defined(__riscv_v_intrinsic)
    #ఒకవేళ LMUL == 1
        tinyBLAS_RVV<vfloat32m1_t, vfloat32m1_t, దశాంశం, దశాంశం, దశాంశం> tb{ params,
            k, (స్థిరరాశి దశాంశం *)A, lda,
            (స్థిరరాశి దశాంశం *)B, ldb,
            (దశాంశం *)C, ldc};
    #లేదావేళ LMUL == 2
        tinyBLAS_RVV<vfloat32m2_t, vfloat32m2_t, దశాంశం, దశాంశం, దశాంశం> tb{ params,
            k, (స్థిరరాశి దశాంశం *)A, lda,
            (స్థిరరాశి దశాంశం *)B, ldb,
            (దశాంశం *)C, ldc};
    #లేకపోతే // LMUL = 4
        tinyBLAS_RVV<vfloat32m4_t, vfloat32m4_t, దశాంశం, దశాంశం, దశాంశం> tb{ params,
            k, (స్థిరరాశి దశాంశం *)A, lda,
            (స్థిరరాశి దశాంశం *)B, ldb,
            (దశాంశం *)C, ldc};
    #అయితే_ముగింపు
        తిరిగి tb.matmul(m, n);
#లేకపోతే
        తిరిగి అబద్ధం;
#అయితే_ముగింపు
    }

    స్థితి GGML_TYPE_BF16: {
#ఒకవేళ defined(__AVX512BF16__)
        ఒకవేళ (Btype == GGML_TYPE_BF16) {
            tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, దశాంశం> tb{ params, k,
                (స్థిరరాశి ggml_bf16_t *)A, lda,
                (స్థిరరాశి ggml_bf16_t *)B, ldb,
                (దశాంశం *)C, ldc};
            తిరిగి tb.matmul(m, n);
        }
#లేదావేళ defined(__AVX512F__)
        ఒకవేళ (Btype == GGML_TYPE_BF16) {
            tinyBLAS<16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, దశాంశం> tb{ params, k,
                (స్థిరరాశి ggml_bf16_t *)A, lda,
                (స్థిరరాశి ggml_bf16_t *)B, ldb,
                (దశాంశం *)C, ldc};
            తిరిగి tb.matmul(m, n);
        }
#లేదావేళ defined(__AVX2__)
        ఒకవేళ (Btype == GGML_TYPE_BF16) {
            tinyBLAS<8, __m256, __m256, ggml_bf16_t, ggml_bf16_t, దశాంశం> tb{ params, k,
                (స్థిరరాశి ggml_bf16_t *)A, lda,
                (స్థిరరాశి ggml_bf16_t *)B, ldb,
                (దశాంశం *)C, ldc};
            తిరిగి tb.matmul(m, n);
        }
#లేదావేళ defined(__MMA__)
        ఒకవేళ (k % 8) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (Btype == GGML_TYPE_BF16) {
            tinyBLAS_HP16_PPC<ggml_bf16_t, ggml_bf16_t, దశాంశం> tb{ k,
                (స్థిరరాశి ggml_bf16_t *)A, lda,
                (స్థిరరాశి ggml_bf16_t *)B, ldb,
                (దశాంశం *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            తిరిగి నిజం;
        }
#లేదావేళ defined(__riscv_zvfbfwma)
        ఒకవేళ (Btype == GGML_TYPE_BF16) {
            #ఒకవేళ LMUL == 1
                tinyBLAS_RVV<vfloat32m1_t, vbfloat16mf2_t, ggml_bf16_t, ggml_bf16_t, దశాంశం> tb{ params,
                    k, (స్థిరరాశి ggml_bf16_t *)A, lda,
                    (స్థిరరాశి ggml_bf16_t *)B, ldb,
                    (దశాంశం *)C, ldc};
            #లేదావేళ LMUL == 2
                tinyBLAS_RVV<vfloat32m2_t, vbfloat16m1_t, ggml_bf16_t, ggml_bf16_t, దశాంశం> tb{ params,
                    k, (స్థిరరాశి ggml_bf16_t *)A, lda,
                    (స్థిరరాశి ggml_bf16_t *)B, ldb,
                    (దశాంశం *)C, ldc};
            #లేకపోతే // LMUL = 4
                tinyBLAS_RVV<vfloat32m4_t, vbfloat16m2_t, ggml_bf16_t, ggml_bf16_t, దశాంశం> tb{ params,
                    k, (స్థిరరాశి ggml_bf16_t *)A, lda,
                    (స్థిరరాశి ggml_bf16_t *)B, ldb,
                    (దశాంశం *)C, ldc};
            #అయితే_ముగింపు
                తిరిగి tb.matmul(m, n);
        }
#అయితే_ముగింపు
        తిరిగి అబద్ధం;
    }

    స్థితి GGML_TYPE_F16: {
#ఒకవేళ defined(__AVX512F__)
        ఒకవేళ (Btype == GGML_TYPE_F16) {
            tinyBLAS<16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, దశాంశం> tb{ params, k,
                (స్థిరరాశి ggml_fp16_t *)A, lda,
                (స్థిరరాశి ggml_fp16_t *)B, ldb,
                (దశాంశం *)C, ldc};
            తిరిగి tb.matmul(m, n);
        }
#లేదావేళ (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        ఒకవేళ (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, దశాంశం> tb{ params, k,
                (స్థిరరాశి ggml_fp16_t *)A, lda,
                (స్థిరరాశి ggml_fp16_t *)B, ldb,
                (దశాంశం *)C, ldc};
            తిరిగి tb.matmul(m, n);
        }
#లేదావేళ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
        ఒకవేళ (n < 8)
            తిరిగి అబద్ధం;
        ఒకవేళ (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, దశాంశం> tb{ params,
                k, (స్థిరరాశి ggml_fp16_t *)A, lda,
                (స్థిరరాశి ggml_fp16_t *)B, ldb,
                (దశాంశం *)C, ldc};
            తిరిగి tb.matmul(m, n);
        }
#లేదావేళ defined(__ARM_NEON) && !defined(_MSC_VER)
        ఒకవేళ (Btype == GGML_TYPE_F32) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, దశాంశం, దశాంశం> tb{ params,
                k, (స్థిరరాశి ggml_fp16_t *)A, lda,
                (స్థిరరాశి దశాంశం *)B, ldb,
                (దశాంశం *)C, ldc};
            తిరిగి tb.matmul(m, n);
        }
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
        ఒకవేళ (n < 4)
            తిరిగి అబద్ధం;
        ఒకవేళ (Btype == GGML_TYPE_F16) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, దశాంశం> tb{ params,
                k, (స్థిరరాశి ggml_fp16_t *)A, lda,
                (స్థిరరాశి ggml_fp16_t *)B, ldb,
                (దశాంశం *)C, ldc};
            తిరిగి tb.matmul(m, n);
        }
#లేదావేళ defined(__riscv_zvfh)
        ఒకవేళ (Btype == GGML_TYPE_F16) {
        #ఒకవేళ LMUL == 1
            tinyBLAS_RVV<vfloat32m1_t, vfloat16mf2_t, ggml_fp16_t, ggml_fp16_t, దశాంశం> tb{ params,
                k, (స్థిరరాశి ggml_fp16_t *)A, lda,
                (స్థిరరాశి ggml_fp16_t *)B, ldb,
                (దశాంశం *)C, ldc};
        #లేదావేళ LMUL == 2
            tinyBLAS_RVV<vfloat32m2_t, vfloat16m1_t, ggml_fp16_t, ggml_fp16_t, దశాంశం> tb{ params,
                k, (స్థిరరాశి ggml_fp16_t *)A, lda,
                (స్థిరరాశి ggml_fp16_t *)B, ldb,
                (దశాంశం *)C, ldc};
        #లేకపోతే // LMUL = 4
            tinyBLAS_RVV<vfloat32m4_t, vfloat16m2_t, ggml_fp16_t, ggml_fp16_t, దశాంశం> tb{ params,
                k, (స్థిరరాశి ggml_fp16_t *)A, lda,
                (స్థిరరాశి ggml_fp16_t *)B, ldb,
                (దశాంశం *)C, ldc};
        #అయితే_ముగింపు
            తిరిగి tb.matmul(m, n);
        }
#లేదావేళ defined(__MMA__)
        ఒకవేళ (k % 8) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (Btype == GGML_TYPE_F16) {
            tinyBLAS_HP16_PPC<ggml_fp16_t, ggml_fp16_t, దశాంశం> tb{ k,
                (స్థిరరాశి ggml_fp16_t *)A, lda,
                (స్థిరరాశి ggml_fp16_t *)B, ldb,
                (దశాంశం *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            తిరిగి నిజం;
        }
#అయితే_ముగింపు
        తిరిగి అబద్ధం;
    }

    స్థితి GGML_TYPE_Q8_0: {
        ఒకవేళ (Btype != GGML_TYPE_Q8_0)
           తిరిగి అబద్ధం;
#ఒకవేళ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ఖండం_q8_0, ఖండం_q8_0, దశాంశం> tb{
            k, (స్థిరరాశి ఖండం_q8_0 *)A, lda,
            (స్థిరరాశి ఖండం_q8_0 *)B, ldb,
            (దశాంశం *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        తిరిగి నిజం;
#లేదావేళ defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<ఖండం_q8_0> tb{
            k, (స్థిరరాశి ఖండం_q8_0 *)A, lda,
            (స్థిరరాశి ఖండం_q8_0 *)B, ldb,
            (దశాంశం *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        తిరిగి నిజం;
#లేదావేళ defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        ఒకవేళ (n < 8 && n != 4)
           తిరిగి అబద్ధం;
        ఒకవేళ (m < 8 && m != 4)
           తిరిగి అబద్ధం;
        tinyBLAS_Q0_PPC<ఖండం_q8_0> tb{
            k, (స్థిరరాశి ఖండం_q8_0 *)A, lda,
            (స్థిరరాశి ఖండం_q8_0 *)B, ldb,
            (దశాంశం *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        తిరిగి నిజం;
#లేకపోతే
        తిరిగి అబద్ధం;
#అయితే_ముగింపు
    }

    స్థితి GGML_TYPE_Q4_0: {
        ఒకవేళ (Btype != GGML_TYPE_Q8_0)
            తిరిగి అబద్ధం;
#ఒకవేళ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ఖండం_q4_0, ఖండం_q8_0, దశాంశం> tb{
            k, (స్థిరరాశి ఖండం_q4_0 *)A, lda,
            (స్థిరరాశి ఖండం_q8_0 *)B, ldb,
            (దశాంశం *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        తిరిగి నిజం;
#లేదావేళ defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<ఖండం_q4_0> tb{
            k, (స్థిరరాశి ఖండం_q4_0 *)A, lda,
            (స్థిరరాశి ఖండం_q8_0 *)B, ldb,
            (దశాంశం *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        తిరిగి నిజం;
#లేదావేళ defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        ఒకవేళ (n < 8 && n != 4)
           తిరిగి అబద్ధం;
        ఒకవేళ (m < 8 && m != 4)
           తిరిగి అబద్ధం;
        tinyBLAS_Q0_PPC<ఖండం_q4_0> tb{
            k, (స్థిరరాశి ఖండం_q4_0 *)A, lda,
            (స్థిరరాశి ఖండం_q8_0 *)B, ldb,
            (దశాంశం *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        తిరిగి నిజం;
#లేకపోతే
        తిరిగి అబద్ధం;
#అయితే_ముగింపు
    }

    స్థితి GGML_TYPE_Q5_0: {
        ఒకవేళ (Btype != GGML_TYPE_Q8_0)
            తిరిగి అబద్ధం;
#ఒకవేళ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ఖండం_q5_0, ఖండం_q8_0, దశాంశం> tb{
            k, (స్థిరరాశి ఖండం_q5_0 *)A, lda,
            (స్థిరరాశి ఖండం_q8_0 *)B, ldb,
            (దశాంశం *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        తిరిగి నిజం;
#లేకపోతే
        తిరిగి అబద్ధం;
#అయితే_ముగింపు
    }

    స్థితి GGML_TYPE_IQ4_NL: {
        ఒకవేళ (Btype != GGML_TYPE_Q8_0)
            తిరిగి అబద్ధం;
#ఒకవేళ defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<ఖండం_iq4_nl, ఖండం_q8_0, దశాంశం> tb{
            k, (స్థిరరాశి ఖండం_iq4_nl *)A, lda,
            (స్థిరరాశి ఖండం_q8_0 *)B, ldb,
            (దశాంశం *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        తిరిగి నిజం;
#లేకపోతే
        తిరిగి అబద్ధం;
#అయితే_ముగింపు
    }

    అప్రమేయ:
        తిరిగి అబద్ధం;
    }

    (శూన్య)params;
    (శూన్య)m;
    (శూన్య)n;
    (శూన్య)k;
    (శూన్య)A;
    (శూన్య)lda;
    (శూన్య)B;
    (శూన్య)ldb;
    (శూన్య)C;
    (శూన్య)ldc;
    (శూన్య)Atype;
    (శూన్య)Btype;
    (శూన్య)Ctype;
}
