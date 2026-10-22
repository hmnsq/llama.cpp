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

#اگر defined(__GNUC__)
#ہدایت GCC diagnostic ignored "-Wpedantic"
#ہدایت GCC diagnostic ignored "-Wignored-attributes"
#اگر_ختم

#شامل "sgemm.h"
#شامل "ggml-impl.h"
#شامل "ggml-cpu-impl.h"
#شامل "ggml-quants.h"
#شامل "simd-mappings.h"

#شامل <array>
#شامل <type_traits>

#اگر_تعریف_شدہ _MSC_VER
#تعریف_کرو NOINLINE __declspec(noinline)
#ورنہ
#تعریف_کرو NOINLINE __attribute__((__noinline__))
#اگر_ختم

#اگر defined(__ARM_NEON) || defined(__AVX512F__) || defined(__VXE__) || defined(__VXE2__)
#تعریف_کرو سمتیہ_REGISTERS 32
#ورنہ
#تعریف_کرو سمتیہ_REGISTERS 16
#اگر_ختم

#اگر defined(__riscv_v_intrinsic)
#تعریف_کرو LMUL 4
#اگر_ختم

#تعریف_کرو MM256_سیٹ_M128_I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

نامخلا {

اندرونی اعشاری unhalf(ggml_fp16_t d) {
    واپس GGML_CPU_FP16_مقصد_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#اگر defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
اندرونی __m128 جوڑو_2(__m128 x, __m128 y) { واپس mm_جوڑو_ps(x, y); }
اندرونی __m128 sub(__m128 x, __m128 y) { واپس _mm_sub_ps(x, y); }
اندرونی __m128 mul(__m128 x, __m128 y) { واپس _mm_mul_ps(x, y); }
#اگر_ختم  // __SSE__

#اگر defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
اندرونی __m256 جوڑو_2(__m256 x, __m256 y) { واپس mm256_جوڑو_ps(x, y); }
اندرونی __m256 sub(__m256 x, __m256 y) { واپس _mm256_sub_ps(x, y); }
اندرونی __m256 mul(__m256 x, __m256 y) { واپس _mm256_mul_ps(x, y); }
#اگر_ختم // __AVX__

#اگر defined(__AVX512F__)
اندرونی __m512 جوڑو_2(__m512 x, __m512 y) { واپس mm512_جوڑو_ps(x, y); }
اندرونی __m512 sub(__m512 x, __m512 y) { واپس _mm512_sub_ps(x, y); }
اندرونی __m512 mul(__m512 x, __m512 y) { واپس _mm512_mul_ps(x, y); }
#اگر_ختم // __AVX512F__

#اگر defined(__ARM_NEON)
اندرونی float32x4_t جوڑو_2(float32x4_t x, float32x4_t y) { واپس vaddq_f32(x, y); }
اندرونی float32x4_t sub(float32x4_t x, float32x4_t y) { واپس vsubq_f32(x, y); }
اندرونی float32x4_t mul(float32x4_t x, float32x4_t y) { واپس vmulq_f32(x, y); }
#اگر_ختم // __ARM_NEON

#اگر defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
اندرونی float16x8_t جوڑو_2(float16x8_t x, float16x8_t y) { واپس vaddq_f16(x, y); }
اندرونی float16x8_t sub(float16x8_t x, float16x8_t y) { واپس vsubq_f16(x, y); }
اندرونی float16x8_t mul(float16x8_t x, float16x8_t y) { واپس vmulq_f16(x, y); }
#اگر_ختم // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#اگر defined(__VXE__) || defined(__VXE2__)
اندرونی float32x4_t جوڑو_2(float32x4_t x, float32x4_t y) { واپس vec_جوڑو(x, y); }
اندرونی float32x4_t sub(float32x4_t x, float32x4_t y) { واپس vec_sub(x, y); }
اندرونی float32x4_t mul(float32x4_t x, float32x4_t y) { واپس vec_mul(x, y); }
#اگر_ختم

#اگر defined(__MMA__)
قسمتعریف vector غیرعلامتی حرف vec_t;
قسمتعریف __vector_quad acc_t;
#اگر_ختم
////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
سانچہ <قسم_نام T, قسم_نام U>
اندرونی U madd(T a, T b, U c) {
    واپس جوڑو_2(mul(a, b), c);
}

#اگر defined(__FMA__)
#اگر defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
سانچہ <>
اندرونی __m256 madd(__m256 a, __m256 b, __m256 c) {
    واپس _mm256_fmadd_ps(a, b, c);
}
#اگر_ختم
#اگر defined(__AVX512F__)
سانچہ <>
اندرونی __m512 madd(__m512 a, __m512 b, __m512 c) {
    واپس _mm512_fmadd_ps(a, b, c);
}
#اگر_ختم
#اگر defined(__AVX512BF16__)
سانچہ <>
اندرونی __m512 madd(__m512bh a, __m512bh b, __m512 c) {
    واپس _mm512_dpbf16_ps(c, a, b);
}
سانچہ <>
اندرونی __m256 madd(__m256bh a, __m256bh b, __m256 c) {
    واپس _mm256_dpbf16_ps(c, a, b);
}
#اگر_ختم
#اگر_ختم

#اگر defined(__ARM_FEATURE_FMA)
سانچہ <>
اندرونی float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    واپس vfmaq_f32(c, b, a);
}
#اگر defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
سانچہ <>
اندرونی float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    واپس vfmaq_f16(c, b, a);
}
#اگر_ختم
#اگر_ختم

#اگر defined(__VXE__) || defined(__VXE2__)
سانچہ <>
اندرونی float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    واپس vec_madd(a, b, c);
}
#اگر_ختم

#اگر defined(__riscv_v_intrinsic)
سانچہ <> اندرونی vfloat32m1_t madd(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t c) {
    واپس __riscv_vfmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
سانچہ <> اندرونی vfloat32m2_t madd(vfloat32m2_t a, vfloat32m2_t b, vfloat32m2_t c) {
    واپس __riscv_vfmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
سانچہ <> اندرونی vfloat32m4_t madd(vfloat32m4_t a, vfloat32m4_t b, vfloat32m4_t c) {
    واپس __riscv_vfmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
سانچہ <> اندرونی vfloat32m8_t madd(vfloat32m8_t a, vfloat32m8_t b, vfloat32m8_t c) {
    واپس __riscv_vfmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#اگر_ختم

#اگر defined(__riscv_zvfh)
سانچہ <> اندرونی vfloat32m1_t madd(vfloat16mf2_t a, vfloat16mf2_t b, vfloat32m1_t c) {
    واپس __riscv_vfwmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
سانچہ <> اندرونی vfloat32m2_t madd(vfloat16m1_t a, vfloat16m1_t b, vfloat32m2_t c) {
    واپس __riscv_vfwmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
سانچہ <> اندرونی vfloat32m4_t madd(vfloat16m2_t a, vfloat16m2_t b, vfloat32m4_t c) {
    واپس __riscv_vfwmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
سانچہ <> اندرونی vfloat32m8_t madd(vfloat16m4_t a, vfloat16m4_t b, vfloat32m8_t c) {
    واپس __riscv_vfwmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#اگر_ختم

#اگر defined(__riscv_zvfbfwma)
سانچہ <> اندرونی vfloat32m1_t madd(vbfloat16mf2_t a, vbfloat16mf2_t b, vfloat32m1_t c) {
    واپس __riscv_vfwmaccbf16_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
سانچہ <> اندرونی vfloat32m2_t madd(vbfloat16m1_t a, vbfloat16m1_t b, vfloat32m2_t c) {
    واپس __riscv_vfwmaccbf16_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
سانچہ <> اندرونی vfloat32m4_t madd(vbfloat16m2_t a, vbfloat16m2_t b, vfloat32m4_t c) {
    واپس __riscv_vfwmaccbf16_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
سانچہ <> اندرونی vfloat32m8_t madd(vbfloat16m4_t a, vbfloat16m4_t b, vfloat32m8_t c) {
    واپس __riscv_vfwmaccbf16_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#اگر_ختم

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#اگر defined(__ARM_NEON) || defined(_M_ARM64) || defined(_M_ARM64EC)
اندرونی اعشاری hsum(float32x4_t x) {
    واپس vaddvq_f32(x);
}
#اگر_ختم // __ARM_NEON

#اگر defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
اندرونی اعشاری hsum(float16x8_t x) {
    واپس vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_کم_f16(x)),
                                vcvt_f32_f16(vget_بلند_f16(x))));
}
#اگر_ختم // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#اگر defined(__VXE__) || defined(__VXE2__)
اندرونی اعشاری hsum(float32x4_t x) {
    float32x4_t tmp = x + vec_reve(x);
    واپس tmp[0] + tmp[1];
}
#اگر_ختم

#اگر defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
اندرونی اعشاری hsum(__m128 x) {
#اگر defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = mm_جوڑو_ps(x, _mm_movehl_ps(x, x));
    x = mm_جوڑو_ss(x, _mm_movehdup_ps(x));
#ورنہ
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = mm_جوڑو_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = mm_جوڑو_ss(x, t);
#اگر_ختم
    واپس _mm_cvtss_f32(x);
}
#اگر_ختم

#اگر defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
اندرونی اعشاری hsum(__m256 x) {
    واپس hsum(mm_جوڑو_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#اگر_ختم // __AVX__

#اگر defined(__AVX512F__)
اندرونی اعشاری hsum(__m512 x) {
    واپس mm512_reduce_جوڑو_ps(x);
}
#اگر_ختم // __AVX512F__

#اگر defined(__riscv_v_intrinsic)
اندرونی اعشاری hsum(vfloat32m1_t x) {
    واپس __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m1_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m1()));
}
اندرونی اعشاری hsum(vfloat32m2_t x) {
    واپس __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m2()));
}
اندرونی اعشاری hsum(vfloat32m4_t x) {
    واپس __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m4()));
}
اندرونی اعشاری hsum(vfloat32m8_t x) {
    واپس __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m8_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m8()));
}
#اگر_ختم

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

سانچہ <قسم_نام T, قسم_نام U> T load(ثابت U *);

#اگر defined(__ARM_NEON)
سانچہ <> اندرونی float32x4_t load(ثابت اعشاری *p) {
    واپس vld1q_f32(p);
}
#اگر !defined(_MSC_VER)
// FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
سانچہ <> اندرونی float16x8_t load(ثابت ggml_fp16_t *p) {
    واپس vld1q_f16((ثابت float16_t *)p);
}
سانچہ <> اندرونی float32x4_t load(ثابت ggml_fp16_t *p) {
    واپس vcvt_f32_f16(vld1_f16((ثابت float16_t *)p));
}
#اگر_ختم // _MSC_VER
#اگر_ختم // __ARM_NEON

#اگر defined(__VXE__) || defined(__VXE2__)
سانچہ <> اندرونی float32x4_t load(ثابت ggml_fp16_t * p) {
    اعشاری tmp[4];

    برائے (صحیح i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_مقصد_FP32(p[i]);
    }

    واپس vec_xl(0, (ثابت اعشاری *)(tmp));
}
سانچہ <> اندرونی float32x4_t load(ثابت اعشاری * p) {
    واپس vec_xl(0, p);
}
#اگر_ختم

#اگر defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
سانچہ <> اندرونی __m128 load(ثابت اعشاری *p) {
    واپس _mm_loadu_ps(p);
}
#اگر_ختم  // __SSE__

#اگر defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
سانچہ <> اندرونی __m256 load(ثابت اعشاری *p) {
    واپس _mm256_loadu_ps(p);
}
#اگر_ختم // __AVX__

#اگر defined(__AVX2__) || defined(__AVX512F__)
سانچہ <> اندرونی __m256 load(ثابت ggml_bf16_t *p) {
    واپس _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((ثابت __m128i *)p)), 16));
}
#اگر_ختم // __AVX2__

#اگر defined(__F16C__)
سانچہ <> اندرونی __m256 load(ثابت ggml_fp16_t *p) {
    واپس _mm256_cvtph_ps(_mm_loadu_si128((ثابت __m128i *)p));
}
#اگر_ختم // __F16C__

#اگر defined(__AVX512F__)
سانچہ <> اندرونی __m512 load(ثابت اعشاری *p) {
    واپس _mm512_loadu_ps(p);
}
سانچہ <> اندرونی __m512 load(ثابت ggml_fp16_t *p) {
    واپس _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)p));
}
سانچہ <> اندرونی __m512 load(ثابت ggml_bf16_t *p) {
    واپس _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((ثابت __m256i *)p)), 16));
}
#اگر_ختم // __AVX512F__

#اگر defined(__AVX512BF16__)
سانچہ <> اندرونی __m512bh load(ثابت ggml_bf16_t *p) {
    واپس (__m512bh)_mm512_loadu_ps((ثابت اعشاری *)p);
}
سانچہ <> اندرونی __m256bh load(ثابت ggml_bf16_t *p) {
    واپس (__m256bh)_mm256_loadu_ps((ثابت اعشاری *)p);
}
سانچہ <> اندرونی __m512bh load(ثابت اعشاری *p) {
    واپس _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
سانچہ <> اندرونی __m256bh load(ثابت اعشاری *p) {
    واپس _mm512_cvtneps_pbh(_mm512_loadu_ps(p));
}
#اگر_ختم

#اگر defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
سانچہ <قسم_نام T, قسم_نام U> T بھرو_partial(ثابت U *, صحیح);
سانچہ <قسم_نام T> T بھرو_partial_u16(ثابت خالی *, صحیح);

سانچہ <> اندرونی __m128i بھرو_partial_u16(ثابت خالی *p, صحیح n) {
#اگر defined(__AVX512BW__) && defined(__AVX512VL__)
    واپس _mm_maskz_loadu_epi16((1u << n) - 1, p);
#ورنہ
    ثابت __m128i اشاریہ = _mm_setr_epi32(0, 1, 2, 3);
    ثابت __m128i جوڑے = _mm_set1_epi32(n / 2);
    __m128i v = _mm_castps_si128(_mm_maskload_ps((ثابت اعشاری *)p, _mm_cmpgt_epi32(جوڑے, اشاریہ)));
    اگر (n & 1) {
        uint16_t آخری;
        حافظہ_نقل(&آخری, (ثابت حرف *)p + 2*(n - 1), حجم(آخری));
        v = _mm_or_si128(v, _mm_and_si128(_mm_cmpeq_epi32(جوڑے, اشاریہ), _mm_set1_epi32(آخری)));
    }
    واپس v;
#اگر_ختم
}

سانچہ <> اندرونی __m256 بھرو_partial(ثابت اعشاری *p, صحیح n) {
    ثابت __m256 اشاریہ = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    واپس _mm256_maskload_ps(p, _mm256_castps_si256(_mm256_cmp_ps(اشاریہ, _mm256_set1_ps(n), _CMP_LT_OQ)));
}

#اگر defined(__F16C__)
سانچہ <> اندرونی __m256 بھرو_partial(ثابت ggml_fp16_t *p, صحیح n) {
    واپس _mm256_cvtph_ps(بھرو_partial_u16<__m128i>(p, n));
}
#اگر_ختم

#اگر defined(__AVX2__) || defined(__AVX512F__)
سانچہ <> اندرونی __m256 بھرو_partial(ثابت ggml_bf16_t *p, صحیح n) {
    واپس _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(بھرو_partial_u16<__m128i>(p, n)), 16));
}
#اگر_ختم

#اگر defined(__AVX512F__)
سانچہ <> اندرونی __m256i بھرو_partial_u16(ثابت خالی *p, صحیح n) {
#اگر defined(__AVX512BW__) && defined(__AVX512VL__)
    واپس _mm256_maskz_loadu_epi16((1u << n) - 1, p);
#ورنہ
    ثابت __m256i اشاریہ = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    ثابت __m256i جوڑے = _mm256_set1_epi32(n / 2);
    __m256i v = _mm256_maskload_epi32((ثابت صحیح *)p, _mm256_cmpgt_epi32(جوڑے, اشاریہ));
    اگر (n & 1) {
        uint16_t آخری;
        حافظہ_نقل(&آخری, (ثابت حرف *)p + 2*(n - 1), حجم(آخری));
        v = _mm256_or_si256(v, _mm256_and_si256(_mm256_cmpeq_epi32(جوڑے, اشاریہ), _mm256_set1_epi32(آخری)));
    }
    واپس v;
#اگر_ختم
}

سانچہ <> اندرونی __m512 بھرو_partial(ثابت اعشاری *p, صحیح n) {
    واپس _mm512_maskz_loadu_ps((1u << n) - 1, p);
}

سانچہ <> اندرونی __m512 بھرو_partial(ثابت ggml_fp16_t *p, صحیح n) {
    واپس _mm512_cvtph_ps(بھرو_partial_u16<__m256i>(p, n));
}

سانچہ <> اندرونی __m512 بھرو_partial(ثابت ggml_bf16_t *p, صحیح n) {
    واپس _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(بھرو_partial_u16<__m256i>(p, n)), 16));
}
#اگر_ختم

#اگر defined(__AVX512BF16__)
سانچہ <> اندرونی __m512bh بھرو_partial(ثابت ggml_bf16_t *p, صحیح n) {
    واپس (__m512bh) _mm512_maskz_loadu_epi16((uint64_t(1) << n) - 1, p);
}
#اگر_ختم
#اگر_ختم

#اگر defined(__riscv_v_intrinsic)
سانچہ <> اندرونی vfloat32m1_t load(ثابت اعشاری *p) {
    واپس __riscv_vle32_v_f32m1(p, __riscv_vsetvlmax_e32m1());
}
سانچہ <> اندرونی vfloat32m2_t load(ثابت اعشاری *p) {
    واپس __riscv_vle32_v_f32m2(p, __riscv_vsetvlmax_e32m2());
}
سانچہ <> اندرونی vfloat32m4_t load(ثابت اعشاری *p) {
    واپس __riscv_vle32_v_f32m4(p, __riscv_vsetvlmax_e32m4());
}
سانچہ <> اندرونی vfloat32m8_t load(ثابت اعشاری *p) {
    واپس __riscv_vle32_v_f32m8(p, __riscv_vsetvlmax_e32m8());
}
#اگر_ختم

#اگر defined(__riscv_zvfh)
سانچہ <> اندرونی vfloat16mf2_t load(ثابت ggml_fp16_t *p) {
    واپس __riscv_vle16_v_f16mf2(reinterpret_cast<ثابت _Float16 *>(p), __riscv_vsetvlmax_e16mf2());
}
سانچہ <> اندرونی vfloat16m1_t load(ثابت ggml_fp16_t *p) {
    واپس __riscv_vle16_v_f16m1(reinterpret_cast<ثابت _Float16 *>(p), __riscv_vsetvlmax_e16m1());
}
سانچہ <> اندرونی vfloat16m2_t load(ثابت ggml_fp16_t *p) {
    واپس __riscv_vle16_v_f16m2(reinterpret_cast<ثابت _Float16 *>(p), __riscv_vsetvlmax_e16m2());
}
سانچہ <> اندرونی vfloat16m4_t load(ثابت ggml_fp16_t *p) {
    واپس __riscv_vle16_v_f16m4(reinterpret_cast<ثابت _Float16 *>(p), __riscv_vsetvlmax_e16m4());
}
#اگر_ختم

#اگر defined(__riscv_zvfbfwma)
سانچہ <> اندرونی vbfloat16mf2_t load(ثابت ggml_bf16_t *p) {
    واپس __riscv_vle16_v_bf16mf2(reinterpret_cast<ثابت __bf16*>(p), __riscv_vsetvlmax_e16mf2());
}
سانچہ <> اندرونی vbfloat16m1_t load(ثابت ggml_bf16_t *p) {
    واپس __riscv_vle16_v_bf16m1(reinterpret_cast<ثابت __bf16*>(p), __riscv_vsetvlmax_e16m1());
}
سانچہ <> اندرونی vbfloat16m2_t load(ثابت ggml_bf16_t *p) {
    واپس __riscv_vle16_v_bf16m2(reinterpret_cast<ثابت __bf16*>(p), __riscv_vsetvlmax_e16m2());
}
سانچہ <> اندرونی vbfloat16m4_t load(ثابت ggml_bf16_t *p) {
    واپس __riscv_vle16_v_bf16m4(reinterpret_cast<ثابت __bf16*>(p), __riscv_vsetvlmax_e16m4());
}
#اگر_ختم

#اگر defined(__riscv_v_intrinsic)
سانچہ <قسم_نام T> T سیٹ_zero();

سانچہ <> اندرونی vfloat32m1_t سیٹ_zero() {
    واپس __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
}
سانچہ <> اندرونی vfloat32m2_t سیٹ_zero() {
    واپس __riscv_vfmv_v_f_f32m2(0, __riscv_vsetvlmax_e32m2());
}
سانچہ <> اندرونی vfloat32m4_t سیٹ_zero() {
    واپس __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
}
سانچہ <> اندرونی vfloat32m8_t سیٹ_zero() {
    واپس __riscv_vfmv_v_f_f32m8(0, __riscv_vsetvlmax_e32m8());
}
#اگر_ختم

#اگر defined(__riscv_v_intrinsic)
سانچہ <قسم_نام T> size_t vlmax() {
    اگر constexpr (std::is_same_v<T, vfloat32m1_t>) { واپس  __riscv_vsetvlmax_e32m1(); }
    ورنہ اگر constexpr (std::is_same_v<T, vfloat32m2_t>) { واپس  __riscv_vsetvlmax_e32m2(); }
    ورنہ اگر constexpr (std::is_same_v<T, vfloat32m4_t>) { واپس  __riscv_vsetvlmax_e32m4(); }
    ورنہ اگر constexpr (std::is_same_v<T, vfloat32m8_t>) { واپس  __riscv_vsetvlmax_e32m8(); }
    #اگر defined (__riscv_zvfh)
    ورنہ اگر constexpr (std::is_same_v<T, vfloat16mf2_t>) { واپس  __riscv_vsetvlmax_e16mf2(); }
    ورنہ اگر constexpr (std::is_same_v<T, vfloat16m1_t>) { واپس  __riscv_vsetvlmax_e16m1(); }
    ورنہ اگر constexpr (std::is_same_v<T, vfloat16m2_t>) { واپس  __riscv_vsetvlmax_e16m2(); }
    ورنہ اگر constexpr (std::is_same_v<T, vfloat16m4_t>) { واپس  __riscv_vsetvlmax_e16m4(); }
    #اگر_ختم
    #اگر defined (__riscv_zvfbfwma)
    ورنہ اگر constexpr (std::is_same_v<T, vbfloat16mf2_t>) { واپس  __riscv_vsetvlmax_e16mf2(); }
    ورنہ اگر constexpr (std::is_same_v<T, vbfloat16m1_t>) { واپس  __riscv_vsetvlmax_e16m1(); }
    ورنہ اگر constexpr (std::is_same_v<T, vbfloat16m2_t>) { واپس  __riscv_vsetvlmax_e16m2(); }
    ورنہ اگر constexpr (std::is_same_v<T, vbfloat16m4_t>) { واپس  __riscv_vsetvlmax_e16m4(); }
    #اگر_ختم
    واپس 0;
}
#اگر_ختم

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

سانچہ <صحیح M>
ساکن اندرونی int64_t بلاک_سائز_2(size_t m) {
    ثابت int64_t NB_BLOC_M = (m + M - 1) / M;
    واپس (m % NB_BLOC_M == 0) ? m / NB_BLOC_M : (m / NB_BLOC_M) + 1;
}

ساکن constexpr اندرونی int64_t BLOC_POS(int64_t ib, int64_t ibN, int64_t bloc_سائز) {
    واپس ib < ibN ? ib * bloc_سائز : ibN * bloc_سائز + (ib - ibN) * (bloc_سائز - 1);
}

سانچہ <صحیح KN, قسم_نام D, قسم_نام V, قسم_نام TA, قسم_نام TB, قسم_نام TC>
جماعت tinyBLAS {
  عوامی:
    tinyBLAS(ثابت ggml_compute_params * params, int64_t k,
             ثابت TA *A, int64_t lda,
             ثابت TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    منطقی matmul(int64_t m, int64_t n) {
#اگر !defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
        اگر (k % KN != 0)
            واپس جھوٹ;
#اگر_ختم
        // compute RM for only need tile with size RM&RM-1
#اگر سمتیہ_REGISTERS == 32
        اگر (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t سائز_N = بلاک_سائز_2<6>(n);
            mnpack<4, 6, 4>(m, n, سائز_N, 12);
            واپس سچ;
        }
        اگر (m % 8 == 0 ) {
            ثابت int64_t سائز_N = بلاک_سائز_2<6>(n);
            mnpack<4, 6, 2>(m, n, سائز_N, 12);
            واپس سچ;
        }
        اگر (m % 4 == 0) {
            ثابت int64_t سائز_N = بلاک_سائز_2<6>(n);
            mnpack<4, 6, 1>(m, n, سائز_N, 12);
            واپس سچ;
        }
#ورنہ  // VECTOR_REGISTERS == 16
        اگر (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t سائز_N = بلاک_سائز_2<3>(n);
            mnpack<4, 3, 4>(m, n, سائز_N, 24);
            واپس سچ;
        }
        اگر (m % 8 == 0 ) {
            ثابت int64_t سائز_N = بلاک_سائز_2<3>(n);
            mnpack<4, 3, 2>(m, n, سائز_N, 24);
            واپس سچ;
        }
        اگر (m % 4 == 0) {
            ثابت int64_t سائز_N = بلاک_سائز_2<3>(n);
            mnpack<4, 3, 1>(m, n, سائز_N, 24);
            واپس سچ;
        }
#اگر_ختم
        واپس جھوٹ;
    }

  نجی:
    سانچہ <صحیح RM, صحیح RN, صحیح BM>
    اندرونی خالی mnpack(int64_t m, int64_t n, int64_t سائز_N, int64_t BN) {
        اگر (سائز_N == RN) {
            واپس gemm<RM, RN, BM>(m, n, BN);
        }
        اگر constexpr (RN > 1) {
            واپس mnpack<RM, RN-1, BM>(m, n, سائز_N, BN);
        } ورنہ {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (صحیح)سائز_N);
            GGML_ASSERT(جھوٹ); // we have miss something.
        }
    }

    سانچہ <صحیح RM, صحیح RN>
    اندرونی خالی gemm_bloc(int64_t ii, int64_t jj) {
        D Cv[RN][RM] = {};
        برائے (int64_t l = 0; l + KN <= k; l += KN) {
            // help compiler for op order.
            اگر constexpr (RM <= RN) {
                V Av[RM];
                برائے (int64_t i = 0; i < RM; ++i) {
                    Av[i] = load<V>(A + lda * (ii + i) + l);
                }
                برائے (int64_t j = 0; j < RN; ++j) {
                    V Bv = load<V>(B + ldb * (jj + j) + l);
                    برائے (int64_t i = 0; i < RM; ++i) {
                        Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                    }
                }
            } ورنہ {
                V Bv[RN];
                برائے (int64_t j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(B + ldb * (jj + j) + l);
                }
                برائے (int64_t i = 0; i < RM; ++i) {
                    V Av = load<V>(A + lda * (ii + i) + l);
                    برائے (int64_t j = 0; j < RN; ++j) {
                        Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
                    }
                }
            }
        }
#اگر defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        ثابت int64_t rem = k % KN;
        اگر (rem != 0) {
            V Av[RM];
            برائے (int64_t i = 0; i < RM; ++i) {
                Av[i] = بھرو_partial<V>(A + lda * (ii + i) + k - rem, rem);
            }
            برائے (int64_t j = 0; j < RN; ++j) {
                V Bv = بھرو_partial<V>(B + ldb * (jj + j) + k - rem, rem);
                برائے (int64_t i = 0; i < RM; ++i) {
                    Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                }
            }
        }
#اگر_ختم
        برائے (int64_t j = 0; j < RN; ++j)
            برائے (int64_t i = 0; i < RM; ++i)
                C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
    }

    سانچہ <صحیح RM, صحیح RN, صحیح BM>
    NOINLINE خالی gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        ثابت int64_t ytiles = m / (RM * BM);
        ثابت int64_t xtiles = (n + RN -1) / RN;
        ثابت int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        ثابت int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        ثابت int64_t سائز_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        ثابت int64_t jj_BN = (NB_BN - (NB_BN * سائز_BN - xtiles));
        ثابت int64_t nb_کام = ytiles * NB_BN;

        اگر (params->ith == 0) {
            GGML_ASSERT( jj_BN * سائز_BN + (NB_BN - jj_BN) * (سائز_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_سیٹ(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t کام = params->ith;
        جب (کام < nb_کام) {
            ثابت int64_t ii = (کام % ytiles) * RM * BM;
            ثابت int64_t jb =  کام / ytiles;
            ثابت int64_t jr0 = BLOC_POS(jb  , jj_BN, سائز_BN);
            ثابت int64_t jrN = BLOC_POS(jb+1, jj_BN, سائز_BN);

            ثابت int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            ثابت int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            ثابت int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            برائے (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                برائے (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                اگر constexpr (RN > 1) {
                    برائے (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            کام = ggml_threadpool_chunk_جوڑو(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        واپس;
    }

    ثابت ggml_compute_params * params;
    ثابت TA *ثابت A;
    ثابت TB *ثابت B;
    TC *ثابت C;
    ثابت int64_t k;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
};

#اگر defined(__riscv_v_intrinsic)
سانچہ <قسم_نام D, قسم_نام V, قسم_نام TA, قسم_نام TB, قسم_نام TC>
جماعت tinyBLAS_RVV {
  عوامی:
    tinyBLAS_RVV(ثابت ggml_compute_params * params, int64_t k,
             ثابت TA *A, int64_t lda,
             ثابت TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    منطقی matmul(int64_t m, int64_t n) {
        اگر (k % vlmax<V>() != 0) {
            واپس جھوٹ;
        }

#اگر LMUL == 1
        اگر (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t سائز_N = بلاک_سائز_2<6>(n);
            mnpack<4, 6, 4>(m, n, سائز_N, 12);
            واپس سچ;
        }
        اگر (m % 8 == 0 ) {
            ثابت int64_t سائز_N = بلاک_سائز_2<6>(n);
            mnpack<4, 6, 2>(m, n, سائز_N, 12);
            واپس سچ;
        }
        اگر (m % 4 == 0) {
            ثابت int64_t سائز_N = بلاک_سائز_2<6>(n);
            mnpack<4, 6, 1>(m, n, سائز_N, 12);
            واپس سچ;
        }
#وگرنہ_اگر LMUL == 2
        اگر (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t سائز_N = بلاک_سائز_2<3>(n);
            mnpack<4, 3, 4>(m, n, سائز_N, 24);
            واپس سچ;
        }
        اگر (m % 8 == 0 ) {
            ثابت int64_t سائز_N = بلاک_سائز_2<3>(n);
            mnpack<4, 3, 2>(m, n, سائز_N, 24);
            واپس سچ;
        }
        اگر (m % 4 == 0) {
            ثابت int64_t سائز_N = بلاک_سائز_2<3>(n);
            mnpack<4, 3, 1>(m, n, سائز_N, 24);
            واپس سچ;
        }
#ورنہ // LMUL = 4
        اگر (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t سائز_N = بلاک_سائز_2<2>(n);
            mnpack<2, 2, 8>(m, n, سائز_N, 36);
            واپس سچ;
        }
        اگر (m % 8 == 0 ) {
            ثابت int64_t سائز_N = بلاک_سائز_2<2>(n);
            mnpack<2, 2, 4>(m, n, سائز_N, 36);
            واپس سچ;
        }
        اگر (m % 4 == 0) {
            ثابت int64_t سائز_N = بلاک_سائز_2<2>(n);
            mnpack<2, 2, 2>(m, n, سائز_N, 36);
            واپس سچ;
        }
#اگر_ختم
        واپس جھوٹ;
    }

  نجی:
    سانچہ<صحیح RM, صحیح RN, صحیح BM>
    اندرونی خالی mnpack(int64_t m, int64_t n, int64_t سائز_N, int64_t BN) {
        اگر (سائز_N == RN) {
            واپس gemm<RM, RN, BM>(m, n, BN);
        }
        اگر constexpr (RN > 1) {
            واپس mnpack<RM, RN-1, BM>(m, n, سائز_N, BN);
        } ورنہ {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (صحیح)سائز_N);
            GGML_ASSERT(جھوٹ); // we have miss something.
        }
    }

    اندرونی خالی gemm_bloc_4x6(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = سیٹ_zero<D>();
        D Cv01 = سیٹ_zero<D>();
        D Cv02 = سیٹ_zero<D>();
        D Cv03 = سیٹ_zero<D>();
        D Cv10 = سیٹ_zero<D>();
        D Cv11 = سیٹ_zero<D>();
        D Cv12 = سیٹ_zero<D>();
        D Cv13 = سیٹ_zero<D>();
        D Cv20 = سیٹ_zero<D>();
        D Cv21 = سیٹ_zero<D>();
        D Cv22 = سیٹ_zero<D>();
        D Cv23 = سیٹ_zero<D>();
        D Cv30 = سیٹ_zero<D>();
        D Cv31 = سیٹ_zero<D>();
        D Cv32 = سیٹ_zero<D>();
        D Cv33 = سیٹ_zero<D>();
        D Cv40 = سیٹ_zero<D>();
        D Cv41 = سیٹ_zero<D>();
        D Cv42 = سیٹ_zero<D>();
        D Cv43 = سیٹ_zero<D>();
        D Cv50 = سیٹ_zero<D>();
        D Cv51 = سیٹ_zero<D>();
        D Cv52 = سیٹ_zero<D>();
        D Cv53 = سیٹ_zero<D>();

        برائے (int64_t l = 0; l < k; l += vl) {
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

    اندرونی خالی gemm_bloc_4x5(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = سیٹ_zero<D>();
        D Cv01 = سیٹ_zero<D>();
        D Cv02 = سیٹ_zero<D>();
        D Cv03 = سیٹ_zero<D>();
        D Cv10 = سیٹ_zero<D>();
        D Cv11 = سیٹ_zero<D>();
        D Cv12 = سیٹ_zero<D>();
        D Cv13 = سیٹ_zero<D>();
        D Cv20 = سیٹ_zero<D>();
        D Cv21 = سیٹ_zero<D>();
        D Cv22 = سیٹ_zero<D>();
        D Cv23 = سیٹ_zero<D>();
        D Cv30 = سیٹ_zero<D>();
        D Cv31 = سیٹ_zero<D>();
        D Cv32 = سیٹ_zero<D>();
        D Cv33 = سیٹ_zero<D>();
        D Cv40 = سیٹ_zero<D>();
        D Cv41 = سیٹ_zero<D>();
        D Cv42 = سیٹ_zero<D>();
        D Cv43 = سیٹ_zero<D>();

        برائے (int64_t l = 0; l < k; l += vl) {
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

    اندرونی خالی gemm_bloc_4x4(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = سیٹ_zero<D>();
        D Cv01 = سیٹ_zero<D>();
        D Cv02 = سیٹ_zero<D>();
        D Cv03 = سیٹ_zero<D>();
        D Cv10 = سیٹ_zero<D>();
        D Cv11 = سیٹ_zero<D>();
        D Cv12 = سیٹ_zero<D>();
        D Cv13 = سیٹ_zero<D>();
        D Cv20 = سیٹ_zero<D>();
        D Cv21 = سیٹ_zero<D>();
        D Cv22 = سیٹ_zero<D>();
        D Cv23 = سیٹ_zero<D>();
        D Cv30 = سیٹ_zero<D>();
        D Cv31 = سیٹ_zero<D>();
        D Cv32 = سیٹ_zero<D>();
        D Cv33 = سیٹ_zero<D>();

        برائے (int64_t l = 0; l < k; l += vl) {
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

    اندرونی خالی gemm_bloc_4x3(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = سیٹ_zero<D>();
        D Cv01 = سیٹ_zero<D>();
        D Cv02 = سیٹ_zero<D>();
        D Cv03 = سیٹ_zero<D>();
        D Cv10 = سیٹ_zero<D>();
        D Cv11 = سیٹ_zero<D>();
        D Cv12 = سیٹ_zero<D>();
        D Cv13 = سیٹ_zero<D>();
        D Cv20 = سیٹ_zero<D>();
        D Cv21 = سیٹ_zero<D>();
        D Cv22 = سیٹ_zero<D>();
        D Cv23 = سیٹ_zero<D>();

        برائے (int64_t l = 0; l < k; l += vl) {
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

    اندرونی خالی gemm_bloc_4x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = سیٹ_zero<D>();
        D Cv01 = سیٹ_zero<D>();
        D Cv02 = سیٹ_zero<D>();
        D Cv03 = سیٹ_zero<D>();
        D Cv10 = سیٹ_zero<D>();
        D Cv11 = سیٹ_zero<D>();
        D Cv12 = سیٹ_zero<D>();
        D Cv13 = سیٹ_zero<D>();

        برائے (int64_t l = 0; l < k; l += vl) {
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

    اندرونی خالی gemm_bloc_4x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = سیٹ_zero<D>();
        D Cv01 = سیٹ_zero<D>();
        D Cv02 = سیٹ_zero<D>();
        D Cv03 = سیٹ_zero<D>();

        برائے (int64_t l = 0; l < k; l += vl) {
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

    اندرونی خالی gemm_bloc_2x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = سیٹ_zero<D>();
        D Cv01 = سیٹ_zero<D>();
        D Cv10 = سیٹ_zero<D>();
        D Cv11 = سیٹ_zero<D>();

        برائے (int64_t l = 0; l < k; l += vl) {
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

    اندرونی خالی gemm_bloc_2x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = سیٹ_zero<D>();
        D Cv01 = سیٹ_zero<D>();

        برائے (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
    }

    سانچہ <صحیح RM, صحیح RN>
    اندرونی خالی gemm_bloc(int64_t ii, int64_t jj) {
        اگر constexpr (RM == 4) {
            اگر constexpr (RN == 6) { واپس gemm_bloc_4x6(ii, jj); }
            اگر constexpr (RN == 5) { واپس gemm_bloc_4x5(ii, jj); }
            اگر constexpr (RN == 4) { واپس gemm_bloc_4x4(ii, jj); }
            اگر constexpr (RN == 3) { واپس gemm_bloc_4x3(ii, jj); }
            اگر constexpr (RN == 2) { واپس gemm_bloc_4x2(ii, jj); }
            اگر constexpr (RN == 1) { واپس gemm_bloc_4x1(ii, jj); }
        } ورنہ اگر constexpr (RM == 2) {
            اگر constexpr (RN == 2) { واپس gemm_bloc_2x2(ii, jj); }
            اگر constexpr (RN == 1) { واپس gemm_bloc_2x1(ii, jj); }
        }
    }

    سانچہ <صحیح RM, صحیح RN, صحیح BM>
    NOINLINE خالی gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        ثابت int64_t ytiles = m / (RM * BM);
        ثابت int64_t xtiles = (n + RN -1) / RN;
        ثابت int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        ثابت int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        ثابت int64_t سائز_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        ثابت int64_t jj_BN = (NB_BN - (NB_BN * سائز_BN - xtiles));
        ثابت int64_t nb_کام = ytiles * NB_BN;

        اگر (params->ith == 0) {
            GGML_ASSERT( jj_BN * سائز_BN + (NB_BN - jj_BN) * (سائز_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_سیٹ(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t کام = params->ith;
        جب (کام < nb_کام) {
            ثابت int64_t ii = (کام % ytiles) * RM * BM;
            ثابت int64_t jb =  کام / ytiles;
            ثابت int64_t jr0 = BLOC_POS(jb  , jj_BN, سائز_BN);
            ثابت int64_t jrN = BLOC_POS(jb+1, jj_BN, سائز_BN);

            ثابت int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            ثابت int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            ثابت int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            برائے (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                برائے (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                اگر constexpr (RN > 1) {
                    برائے (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            کام = ggml_threadpool_chunk_جوڑو(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        واپس;
    }

    ثابت ggml_compute_params * params;
    ثابت TA *ثابت A;
    ثابت TB *ثابت B;
    TC *ثابت C;
    ثابت int64_t k;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
};
#اگر_ختم

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#اگر defined(__ARM_FEATURE_DOTPROD)
سانچہ <قسم_نام TA>
جماعت tinyBLAS_Q0_ARM {
  عوامی:
    tinyBLAS_Q0_ARM(int64_t k,
                    ثابت TA *A, int64_t lda,
                    ثابت بلاک_q8_0 *B, int64_t ldb,
                    اعشاری *C, int64_t ldc,
                    صحیح ith, صحیح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    خالی matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  نجی:
    NOINLINE خالی mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        انتخاب ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3ll)) {
        حالت 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            توڑو;
        حالت 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            توڑو;
        حالت 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            توڑو;
        حالت 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            توڑو;
        حالت 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            توڑو;
        حالت 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            توڑو;
        حالت 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            توڑو;
        حالت 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            توڑو;
        حالت 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            توڑو;
        طےشدہ:
            واپس;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    سانچہ <صحیح RM, صحیح RN>
    NOINLINE خالی gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * RN;
            float32x4_t Cv[RN][RM] = {};
            برائے (int64_t l = 0; l < k; ++l)
                برائے (int64_t j = 0; j < RN; ++j)
                    برائے (int64_t i = 0; i < RM; ++i)
                        Cv[j][i] = vmlaq_n_f32(Cv[j][i],
                                               vcvtq_f32_s32(vdotq_s32(
                                                   vdotq_s32(vdupq_n_s32(0),
                                                             بھرو_lo(A + lda * (ii + i) + l),
                                                             بھرو_lo(B + ldb * (jj + j) + l)),
                                                   بھرو_hi(A + lda * (ii + i) + l),
                                                   بھرو_hi(B + ldb * (jj + j) + l))),
                                               unhalf(A[lda * (ii + i) + l].d) *
                                               unhalf(B[ldb * (jj + j) + l].d));
            برائے (int64_t j = 0; j < RN; ++j)
                برائے (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    اندرونی int8x16_t بھرو_lo(ثابت بلاک_q8_0 *b) {
        واپس vld1q_s8(b->qs);
    }

    اندرونی int8x16_t بھرو_hi(ثابت بلاک_q8_0 *b) {
        واپس vld1q_s8(b->qs + 16);
    }

    اندرونی int8x16_t بھرو_lo(ثابت بلاک_q4_0 *b) {
        واپس vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs),
                                                     vdupq_n_u8(0x0f))),
                        vdupq_n_s8(0x8));
    }

    اندرونی int8x16_t بھرو_hi(ثابت بلاک_q4_0 *b) {
        واپس vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)),
                        vdupq_n_s8(0x8));
    }

    ثابت TA *ثابت A;
    ثابت بلاک_q8_0 *ثابت B;
    اعشاری *ثابت C;
    ثابت int64_t k;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
    ثابت صحیح ith;
    ثابت صحیح nth;
};
#اگر_ختم // __ARM_FEATURE_DOTPROD

#اگر defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
سانچہ <قسم_نام TA, قسم_نام TB, قسم_نام TC>
جماعت tinyBLAS_Q0_AVX {
  عوامی:
    tinyBLAS_Q0_AVX(int64_t k,
                    ثابت TA *A, int64_t lda,
                    ثابت TB *B, int64_t ldb,
                    TC *C, int64_t ldc,
                    صحیح ith, صحیح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
        ثابت int8_t kvalues_iq4nl[16] = {
            -127, -104, -83, -65,
            -49,  -35,  -22, -10,
              1,   13,   25,  38,
             53,   69,   89, 113
        };

        iq4nlt = _mm_loadu_si128((ثابت __m128i *)kvalues_iq4nl);
    }

    خالی matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  نجی:
    خالی mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        انتخاب ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
#اگر سمتیہ_REGISTERS == 32
        حالت 0x44:
            mc = 4;
            nc = 4;
#اگر defined(__AVX2__) && defined(__F16C__)
            gemm4xN<4>(m0, m, n0, n);
#ورنہ
            gemm<4, 4>(m0, m, n0, n);
#اگر_ختم
            توڑو;
        حالت 0x43:
            mc = 4;
            nc = 3;
#اگر defined(__AVX2__) && defined(__F16C__)
            gemm4xN<3>(m0, m, n0, n);
#ورنہ
            gemm<4, 3>(m0, m, n0, n);
#اگر_ختم
            توڑو;
        حالت 0x34:
            mc = 3;
            nc = 4;
#اگر defined(__AVX2__) && defined(__F16C__)
            gemmMx4<3>(m0, m, n0, n);
#ورنہ
            gemm<3, 4>(m0, m, n0, n);
#اگر_ختم
            توڑو;
        حالت 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            توڑو;
        حالت 0x42:
            mc = 4;
            nc = 2;
#اگر defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#ورنہ
            gemm<4, 2>(m0, m, n0, n);
#اگر_ختم
            توڑو;
        حالت 0x24:
            mc = 2;
            nc = 4;
#اگر defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#ورنہ
            gemm<2, 4>(m0, m, n0, n);
#اگر_ختم
            توڑو;
#ورنہ
        حالت 0x44:
        حالت 0x43:
        حالت 0x42:
            mc = 4;
            nc = 2;
#اگر defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#ورنہ
            gemm<4, 2>(m0, m, n0, n);
#اگر_ختم
            توڑو;
        حالت 0x34:
        حالت 0x24:
            mc = 2;
            nc = 4;
#اگر defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#ورنہ
            gemm<2, 4>(m0, m, n0, n);
#اگر_ختم
            توڑو;
        حالت 0x33:
#اگر_ختم
        حالت 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            توڑو;
        حالت 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            توڑو;
        حالت 0x41:
            mc = 4;
            nc = 1;
#اگر defined(__AVX2__) && defined(__F16C__)
            gemm4xN<1>(m0, m, n0, n);
#ورنہ
            gemm<4, 1>(m0, m, n0, n);
#اگر_ختم
            توڑو;
        حالت 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            توڑو;
        حالت 0x14:
            mc = 1;
            nc = 4;
#اگر defined(__AVX2__) && defined(__F16C__)
            gemmMx4<1>(m0, m, n0, n);
#ورنہ
            gemm<1, 4>(m0, m, n0, n);
#اگر_ختم
            توڑو;
        حالت 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            توڑو;
        حالت 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            توڑو;
        حالت 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            توڑو;
        حالت 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            توڑو;
        حالت 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            توڑو;
        طےشدہ:
            واپس;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

#اگر defined(__AVX2__) && defined(__F16C__)
// Templated functions for gemm of dimensions 4xN
    سانچہ <صحیح RN>
    NOINLINE خالی gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / 4;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * 4;
            int64_t jj = n0 + کام % xtiles * RN;
            __m256 Cv[RN][4] = {};
            برائے (int64_t l = 0; l < k; ++l) {
                uint64_t a_delta = ((uint64_t)A[lda * (ii + 3) + l].d << 48) | ((uint64_t)A[lda * (ii + 2) + l].d << 32) | ((uint64_t)A[lda * (ii + 1) + l].d << 16) | (A[lda * (ii + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 da = _mm_cvtph_ps(mm_سیٹ_epi64x(0, a_delta));
                __m256i avec0 = load(A + lda * (ii + 0) + l);
                __m256i avec1 = load(A + lda * (ii + 1) + l);
                __m256i avec2 = load(A + lda * (ii + 2) + l);
                __m256i avec3 = load(A + lda * (ii + 3) + l);
                برائے (int64_t j = 0; j < RN; ++j) {
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

            برائے (int64_t j = 0; j < RN; ++j)
                برائے (int64_t i = 0; i < 4; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    // Templated functions for gemm of dimensions Mx4
    سانچہ <صحیح RM>
    NOINLINE خالی gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / 4;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * 4;
            __m256 Cv[4][RM] = {};
            برائے (int64_t l = 0; l < k; ++l) {
                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 3) + l].d << 48) | ((uint64_t)B[ldb * (jj + 2) + l].d << 32) | ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 db = _mm_cvtph_ps(mm_سیٹ_epi64x(0, b_delta));
                __m256i bvec0 = load(B + ldb * (jj + 0) + l);
                __m256i bvec1 = load(B + ldb * (jj + 1) + l);
                __m256i bvec2 = load(B + ldb * (jj + 2) + l);
                __m256i bvec3 = load(B + ldb * (jj + 3) + l);
                برائے (int64_t i = 0; i < RM; ++i) {
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
            برائے (int64_t j = 0; j < 4; ++j)
                برائے (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }
#اگر_ختم

    سانچہ <صحیح RM, صحیح RN>
    NOINLINE خالی gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            برائے (int64_t l = 0; l < k; ++l)
                برائے (int64_t j = 0; j < RN; ++j)
                    برائے (int64_t i = 0; i < RM; ++i) {
#اگر defined(__AVX2__)
                        __m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                              load(A + lda * (ii + i) + l)),
                                             _mm256_sign_epi8(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l)));
#ورنہ
                        __m128i ali0 = load0(A + lda * (ii + i) + l);
                        __m128i ali1 = load1(A + lda * (ii + i) + l);
                        __m128i blj0 = load0(B + ldb * (jj + j) + l);
                        __m128i blj1 = load1(B + ldb * (jj + j) + l);

                        __m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
                        __m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
                        __m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
                        __m128i sepBA1 = _mm_sign_epi8(blj1, ali1);

                        // updot
                        ثابت __m128i oneFill = _mm_set1_epi16(1);
                        __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0);
                        __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1);
                        __m256 udTmp = _mm256_cvtepi32_ps(MM256_سیٹ_M128_I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
#اگر_ختم
                        Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            برائے (int64_t j = 0; j < RN; ++j)
                برائے (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    اندرونی __m256i load(ثابت بلاک_q8_0 *b) {
        واپس _mm256_loadu_si256((ثابت __m256i *)b->qs);
    }

    اندرونی __m128i load0(ثابت بلاک_q8_0 *b) {
        واپس _mm_loadu_si128((ثابت __m128i *)b->qs);
    }

    اندرونی __m128i load1(ثابت بلاک_q8_0 *b) {
        واپس _mm_loadu_si128(((ثابت __m128i *)b->qs) + 1);
    }

    اندرونی __m256i load(ثابت بلاک_q4_0 *b) {
        واپس _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    اندرونی __m128i load0(ثابت بلاک_q4_0 *b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        واپس _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), x), _mm_set1_epi8(8));
    }

    اندرونی __m128i load1(ثابت بلاک_q4_0 *b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        واپس _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)), _mm_set1_epi8(8));
    }

    اندرونی __m256i load(ثابت بلاک_q5_0 *b) {
        واپس _mm256_or_si256(denibble(b->qs), bittobyte(b->qh));
    }

    اندرونی __m128i load0(ثابت بلاک_q5_0* b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        uint32_t x32;
        حافظہ_نقل(&x32, b->qh, حجم(uint32_t));
        __m128i qxl = _mm_and_si128(_mm_set1_epi8(15), x);
        __m128i bytesl = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      mm_سیٹ_epi64x(0x0101010101010101, 0x0000000000000000))));
        bytesl = _mm_andnot_si128(bytesl, _mm_set1_epi8((حرف)0xF0));
        واپس _mm_or_si128(qxl, bytesl);
    }

    اندرونی __m128i load1(ثابت بلاک_q5_0* b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        uint32_t x32;
        حافظہ_نقل(&x32, b->qh, حجم(uint32_t));
        __m128i qxh = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
        __m128i bytesh = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      mm_سیٹ_epi64x(0x0303030303030303, 0x0202020202020202))));
        bytesh = _mm_andnot_si128(bytesh, _mm_set1_epi8((حرف)0xF0));
        واپس _mm_or_si128(qxh, bytesh);
    }

    اندرونی __m256i load(ثابت بلاک_iq4_nl *b) {
        واپس MM256_سیٹ_M128_I(load1(b), load0(b));
    }

    اندرونی __m128i load0(ثابت بلاک_iq4_nl *b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        واپس _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), x));
    }

    اندرونی __m128i load1(ثابت بلاک_iq4_nl *b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        واپس _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)));
    }

    اندرونی __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#اگر defined(__AVX512VNNI__) && defined(__AVX512VL__)
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#وگرنہ_اگر defined(__AVXVNNI__)
        res = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
#ورنہ
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#اگر_ختم
        واپس _mm256_cvtepi32_ps(res);
    }

    ساکن اندرونی __m256i denibble(ثابت uint8_t *p) {
        __m128i x = _mm_loadu_si128((ثابت __m128i *)p);
        واپس _mm256_and_si256(_mm256_set1_epi8(15),
                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                        _mm_srli_epi16(x, 4), 1));
    }

    ساکن اندرونی __m256i bittobyte(ثابت uint8_t *p) {
        uint32_t x32;
        حافظہ_نقل(&x32, p, حجم(uint32_t));
        __m256i بائٹیں = _mm256_cmpeq_epi8(_mm256_set1_epi64x(-1),
                                          _mm256_or_si256(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                          _mm256_shuffle_epi8(_mm256_set1_epi32(x32),
                                                                              mm256_سیٹ_epi64x(0x0303030303030303, 0x0202020202020202,
                                                                                                0x0101010101010101, 0x0000000000000000))));
        واپس _mm256_andnot_si256(بائٹیں, _mm256_set1_epi8((حرف)0xF0));
    }

    ثابت TA *ثابت A;
    ثابت TB *ثابت B;
    TC *ثابت C;
    ثابت int64_t k;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
    ثابت صحیح ith;
    ثابت صحیح nth;
    __m128i iq4nlt;
};
#اگر_ختم // __AVX__

//PPC Implementation
#اگر defined(__MMA__)

سانچہ<قسم_نام T>
ساخت mma_instr;

سانچہ<>
ساخت mma_instr<ggml_bf16_t> {
    ساکن اندرونی خالی outer_حاصل_ضرب(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvbf16ger2pp(acc, a, b);
    }
};

سانچہ<>
ساخت mma_instr<ggml_fp16_t> {
    ساکن اندرونی خالی outer_حاصل_ضرب(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvf16ger2pp(acc, a, b);
    }
};

سانچہ <قسم_نام TA, قسم_نام TB, قسم_نام TC>
جماعت tinyBLAS_HP16_PPC {
  عوامی:
    tinyBLAS_HP16_PPC(int64_t k,
                ثابت TA *A, int64_t lda,
                ثابت TB *B, int64_t ldb,
                TC *C, int64_t ldc,
                صحیح ith, صحیح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    خالی matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #اگر defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #اگر_ختم
        اگر (k < kc) {
            kc = k;
        }
        منطقی can_use_tiled = (m % mc == 0) && (n % nc == 0) && (k % kc == 0);
        اگر (can_use_tiled) {
            matmul_tiled(m, n, mc, nc, kc);
        } ورنہ {
            mnpack(0, m, 0, n);
        }
    }

  نجی:
    __attribute__((always_inline))
    اندرونی خالی save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        برائے (صحیح I = 0; I < 4; I++) {
            برائے (صحیح J = 0; J < 4; J++) {
                *((اعشاری *)(C+ii+((jj+J)*ldc)+I)) = *((اعشاری *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    اندرونی خالی جوڑو_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        برائے (صحیح I = 0; I < 4; I++) {
            برائے (صحیح J = 0; J < 4; J++) {
                اعشاری * c_ptr = (اعشاری *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((اعشاری *)&vec_C[I]+J);
            }
        }
    }

    خالی سمتیہ_ترتیب_بدلو_store(vec_t *c, صحیح عدد_Vec, غیرعلامتی حرف *vec_انحراف) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        اگر (عدد_Vec == 2) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[2], c[3], swiz1);
            s[0] = vec_perm(t[0], t[1], swiz3);
            s[1] = vec_perm(t[0], t[1], swiz4);
            vec_xst(s[0], 0, (vec_t*)vec_انحراف);
            vec_xst(s[1], 0, (vec_t*)(vec_انحراف + 16));
        } ورنہ اگر (عدد_Vec == 4) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[0], c[1], swiz2);
            t[2] = vec_perm(c[2], c[3], swiz1);
            t[3] = vec_perm(c[2], c[3], swiz2);
            s[0] = vec_perm(t[0], t[2], swiz3);
            s[1] = vec_perm(t[0], t[2], swiz4);
            s[2] = vec_perm(t[1], t[3], swiz3);
            s[3] = vec_perm(t[1], t[3], swiz4);
            برائے (صحیح i = 0; i < 4; ++i)
                vec_xst(s[i], 0, (vec_t*)(vec_انحراف + i * 16));
        } ورنہ اگر (عدد_Vec == 8) {
            برائے (صحیح i = 0; i < 4; i += 2) {
                t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
                t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
            }
            برائے (صحیح i = 4; i < 8; i += 2) {
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
            برائے (صحیح i = 0; i < 8; ++i)
                vec_xst(s[i], 0, (vec_t*)(vec_انحراف + i * 16));
        }
    }

    خالی pack_معمول(ثابت TA* a, int64_t lda, صحیح صفیں, صحیح ستون_تعداد_2, غیرعلامتی حرف* vec) {
        int64_t i, j;
        TA *aoffset = تہی;
        غیرعلامتی حرف *vec_انحراف = تہی;
        TA * aoffsets[8];
        vector غیرعلامتی حرف c_arr[8];
        aoffset = const_cast<TA*>(a);
        vec_انحراف = vec;
        j = (صفیں >> 3);
        اگر (j > 0) {
            کرو {
                aoffsets[0] = aoffset;
                اگر (ستون_تعداد_2 == 4) {
                    aoffsets[0] = aoffset;
                    برائے (صحیح it = 1; it < 4; ++it)
                        aoffsets[it] = aoffsets[it-1] + lda;
                    aoffset += 4 * lda;
                    برائے (صحیح i = 0; i < 4; ++i)
                        c_arr[i] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[i]);
                    سمتیہ_ترتیب_بدلو_store(c_arr, 4, vec_انحراف);
                    برائے (صحیح i = 0; i<4; i++)
                        aoffsets[i] = aoffsets[i]+lda;
                    vec_انحراف +=64;
                }
                i = (ستون_تعداد_2 >> 3);
                اگر (i > 0) {
                    برائے (صحیح it = 1; it < 8; ++it) {
                        aoffsets[it] = aoffsets[it-1] + lda;
                    }
                    aoffset += 8 * lda;

                    کرو {
                        برائے (صحیح it = 0; it < 8; ++it)
                            c_arr[it] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[it]);
                        سمتیہ_ترتیب_بدلو_store(c_arr, 8, vec_انحراف);
                        برائے (صحیح it = 0; it < 8; ++it)
                            aoffsets[it] = aoffsets[it] + 8;
                        vec_انحراف += 128;
                        i--;
                    } جب(i > 0);
                }
                j--;
            } جب(j > 0);
        }
        اگر (صفیں & 4) {
            aoffsets[0] = aoffset;
            برائے (صحیح it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            اگر (ستون_تعداد_2 == 4) {
                برائے (صحیح it = 0; it < 4; ++it)
                    c_arr[it] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[it]);
                سمتیہ_ترتیب_بدلو_store(c_arr, 2, vec_انحراف);
                برائے (صحیح it = 0; it< 4; it++)
                    aoffsets[it] = aoffsets[it] + lda;
                vec_انحراف += 32;
            }
            i = (ستون_تعداد_2 >> 3);
            اگر (i > 0) {
                کرو {
                    برائے (صحیح it = 0; it < 4; ++it)
                        c_arr[it] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[it]);
                    سمتیہ_ترتیب_بدلو_store(c_arr, 4, vec_انحراف);
                    برائے (صحیح it = 0; it< 4; it++)
                        aoffsets[it] = aoffsets[it] + 8*lda;
                    vec_انحراف += 64;
                    i--;
                } جب(i > 0);
            }
        }
        اگر (صفیں & 3) {
            aoffsets[0] = aoffset;
            برائے (صحیح it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            اگر (ستون_تعداد_2 == 4) {
                انتخاب(صفیں) {
                    حالت 3: c_arr[2] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[2]);
                    حالت 2: c_arr[1] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[1]);
                    حالت 1: c_arr[0] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[0]);
                        توڑو;
                }
                سمتیہ_ترتیب_بدلو_store(c_arr, 2, vec_انحراف);
                برائے (صحیح it = 0; it< 4; it++)
                     aoffsets[it] = aoffsets[it] + lda;
                vec_انحراف += 32;
            }
            i = (ستون_تعداد_2 >> 3);
            اگر (i > 0) {
                کرو {
                    انتخاب(صفیں) {
                        حالت 3: c_arr[2] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[2]);
                        حالت 2: c_arr[1] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[1]);
                        حالت 1: c_arr[0] = vec_xl(0, (vector غیرعلامتی حرف*)aoffsets[0]);
                            توڑو;
                    }
                    سمتیہ_ترتیب_بدلو_store(c_arr, 4, vec_انحراف);
                    برائے (صحیح it = 0; it <4; it++)
                         aoffsets[it] = aoffsets[it] + 8* lda;
                    vec_انحراف += 64;
                    i--;
                } جب(i > 0);
            }
        }
    }

    خالی mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        صحیح m_rem = MIN(m - m0, 8);
        صحیح n_rem = MIN(n - n0, 8);

        اگر (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } ورنہ اگر (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4,8>(m0, m, n0, n);
        } ورنہ اگر (m_rem >=8 && n_rem >=4){
                mc = 8;
                nc = 4;
                gemm<8,4>(m0, m, n0, n);
        } ورنہ اگر ((m_rem < 4) && (n_rem >= 8)) {
            nc = 8;
            انتخاب(m_rem) {
                حالت 1:
                    mc = 1;
                    gemm_Mx8<1>(m0, m, n0, n);
                    توڑو;
                حالت 2:
                    mc = 2;
                    gemm_Mx8<2>(m0, m, n0, n);
                    توڑو;
                حالت 3:
                    mc = 3;
                    gemm_Mx8<3>(m0, m, n0, n);
                    توڑو;
                طےشدہ:
                    واپس;
            }
        } ورنہ اگر (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small<4, 4>(m0, m, n0, n);
        } ورنہ اگر ((m_rem > 4) && (n_rem < 4)) {
            mc = 4;
            انتخاب(n_rem) {
                حالت 1:
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    توڑو;
                حالت 2:
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    توڑو;
                حالت 3:
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    توڑو;

                طےشدہ:
                    واپس;
            }
        } ورنہ {
            انتخاب((m_rem << 4) | n_rem) {
                حالت 0x43:
                    mc = 4;
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    توڑو;
                حالت 0x42:
                    mc = 4;
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    توڑو;
                حالت 0x41:
                    mc = 4;
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    توڑو;
                حالت 0x34:
                    mc = 3;
                    nc = 4;
                    gemm_small<3, 4>(m0, m, n0, n);
                    توڑو;
                حالت 0x33:
                    mc = 3;
                    nc = 3;
                    gemm_small<3, 3>(m0, m, n0, n);
                    توڑو;
                حالت 0x32:
                    mc = 3;
                    nc = 2;
                    gemm_small<3, 2>(m0, m, n0, n);
                    توڑو;
                حالت 0x31:
                    mc = 3;
                    nc = 1;
                    gemm_small<3, 1>(m0, m, n0, n);
                    توڑو;
                حالت 0x24:
                    mc = 2;
                    nc = 4;
                    gemm_small<2,4>(m0, m, n0, n);
                    توڑو;
                حالت 0x23:
                    mc = 2;
                    nc = 3;
                    gemm_small<2, 3>(m0, m, n0, n);
                    توڑو;
                حالت 0x22:
                    mc = 2;
                    nc = 2;
                    gemm_small<2, 2>(m0, m, n0, n);
                    توڑو;
                حالت 0x21:
                    mc = 2;
                    nc = 1;
                    gemm_small<2, 1>(m0, m, n0, n);
                    توڑو;
                حالت 0x14:
                    mc = 1;
                    nc = 4;
                    gemm_small<1, 4>(m0, m, n0, n);
                    توڑو;
                حالت 0x13:
                    mc = 1;
                    nc = 3;
                    gemm_small<1, 3>(m0, m, n0, n);
                    توڑو;
                حالت 0x12:
                    mc = 1;
                    nc = 2;
                    gemm_small<1, 2>(m0, m, n0, n);
                    توڑو;
                حالت 0x11:
                    mc = 1;
                    nc = 1;
                    gemm_small<1, 1>(m0, m, n0, n);
                    توڑو;
                طےشدہ:
                    واپس;
            }
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    خالی KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        برائے (صحیح l = 0; l < k; l+=8) {
            pack_معمول((A+(ii*lda)+l), lda, 4, 8, (uint8_t*)vec_A);
            pack_معمول((B+(jj*ldb)+l), ldb, 8, 8, (uint8_t*)vec_B);
            برائے (صحیح x = 0; x < 4; x++) {
                mma_instr<TA>::outer_حاصل_ضرب(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_حاصل_ضرب(&acc_1, vec_A[x], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
    }

    خالی KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        برائے (صحیح l = 0; l < k; l+=8) {
            pack_معمول((A+(ii*lda)+l), lda, 8, 8, (uint8_t*)vec_A);
            pack_معمول((B+(jj*ldb)+l), ldb, 8, 4, (uint8_t*)vec_B);
            برائے (صحیح x = 0; x < 4; x++) {
                mma_instr<TA>::outer_حاصل_ضرب(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_حاصل_ضرب(&acc_1, vec_A[x+4], vec_B[x]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii+4, jj);
    }


    خالی KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        برائے (صحیح l = 0; l < k; l+=8) {
            pack_معمول(A+(ii*lda)+l, lda, 8, 8, (uint8_t*)vec_A);
            pack_معمول(B+(jj*ldb)+l, ldb, 8, 8, (uint8_t*)vec_B);
            برائے (صحیح x = 0; x < 4; x++) {
                mma_instr<TA>::outer_حاصل_ضرب(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_حاصل_ضرب(&acc_1, vec_A[x], vec_B[x+4]);
                mma_instr<TA>::outer_حاصل_ضرب(&acc_2, vec_A[x+4], vec_B[x]);
                mma_instr<TA>::outer_حاصل_ضرب(&acc_3, vec_A[x+4], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
        save_acc(&acc_2, ii+4, jj);
        save_acc(&acc_3, ii+4, jj+4);
    }

    اندرونی خالی MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        برائے (صحیح x = 0; x < 4; x ++) {
             mma_instr<TA>::outer_حاصل_ضرب(&acc[0], vec_A0[x], vec_B[x]);
             mma_instr<TA>::outer_حاصل_ضرب(&acc[1], vec_A0[x], vec_B[x+4]);
             mma_instr<TA>::outer_حاصل_ضرب(&acc[2], vec_A0[x+4], vec_B[x]);
             mma_instr<TA>::outer_حاصل_ضرب(&acc[3], vec_A0[x+4], vec_B[x+4]);
             mma_instr<TA>::outer_حاصل_ضرب(&acc[4], vec_A1[x], vec_B[x]);
             mma_instr<TA>::outer_حاصل_ضرب(&acc[5], vec_A1[x], vec_B[x+4]);
             mma_instr<TA>::outer_حاصل_ضرب(&acc[6], vec_A1[x+4], vec_B[x]);
             mma_instr<TA>::outer_حاصل_ضرب(&acc[7], vec_A1[x+4], vec_B[x+4]);
        }
    }
    خالی KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        برائے (int64_t i = 0; i < mc; i += 16) {
            صحیح A_بنیاد_addr = (mc / 8) * (i / 8) * 8;
            برائے (int64_t j = 0; j < nc; j += 8) {
                 صحیح B_بنیاد_addr = (nc / 8) * (j / 8) * 8;
                 acc_t acc[8];
                 vec_t A0_بلاک[8]; vec_t A1_بلاک[8];
                 برائے (صحیح x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 برائے (int64_t l = 0; l < kc; l += 8) {
                     صحیح A0_بلاک_idx = A_بنیاد_addr + (l / 8) * 8;
                     صحیح A1_بلاک_idx = A0_بلاک_idx + (mc / 8) * 8;
                     صحیح B_بلاک_idx = B_بنیاد_addr + (l / 8) * 8;
                     vec_t* A0_بلاک = &vec_A[A0_بلاک_idx];
                     vec_t* A1_بلاک = &vec_A[A1_بلاک_idx];
                     vec_t* B_بلاک = &vec_B[B_بلاک_idx];
                     MMA_16x8(A0_بلاک, A1_بلاک, B_بلاک, acc);
                 }
                 اگر (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } ورنہ {
                     جوڑو_save_acc(&acc[0], ii + i, jj + j);
                     جوڑو_save_acc(&acc[1], ii + i, jj + j + 4);
                     جوڑو_save_acc(&acc[2], ii + i + 4, jj + j);
                     جوڑو_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     جوڑو_save_acc(&acc[4], ii + i + 8, jj + j);
                     جوڑو_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     جوڑو_save_acc(&acc[6], ii + i + 12, jj + j);
                     جوڑو_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }
    سانچہ<صحیح RM, صحیح RN>
    خالی gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[2], vec_B[2];
            برائے (صحیح l=0; l<k; l+=4) {
                pack_معمول(A+(ii*lda)+l, lda, RM, 4, (uint8_t*)vec_A);
                pack_معمول(B+(jj*ldb)+l, ldb, RN, 4, (uint8_t*)vec_B);
                برائے (صحیح x = 0; x<2; x++) {
                    mma_instr<TA>::outer_حاصل_ضرب(&acc_0, vec_A[x], vec_B[x]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            برائے (صحیح I = 0; I < RM; I++) {
                برائے (صحیح J = 0; J < RN; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    سانچہ<صحیح RM>
    خالی gemm_Mx8(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        صحیح RN = 8;
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0, acc_1;
            __builtin_mma_xxsetaccz(&acc_0);
            __builtin_mma_xxsetaccz(&acc_1);
            vec_t vec_A[4], vec_B[8];
            برائے (صحیح l=0; l<k; l+=8) {
                pack_معمول(A+(ii*lda)+l, lda, RM, 8, (uint8_t*)vec_A);
                pack_معمول(B+(jj*ldb)+l, ldb, RN, 8, (uint8_t*)vec_B);
                برائے (صحیح x = 0; x<4; x++) {
                    mma_instr<TA>::outer_حاصل_ضرب(&acc_0, vec_A[x], vec_B[x]);
                    mma_instr<TA>::outer_حاصل_ضرب(&acc_1, vec_A[x], vec_B[x+4]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            برائے (صحیح I = 0; I < RM; I++) {
                برائے (صحیح J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_1);
            برائے (صحیح I = 0; I < RM; I++) {
                برائے (صحیح J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+4+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    سانچہ<صحیح RM, صحیح RN>
    اندرونی خالی kernel(int64_t ii, int64_t jj) {
       اگر constexpr(RM == 4 && RN == 8) {
          KERNEL_4x8(ii,jj);
       } ورنہ اگر constexpr(RM == 8 && RN == 8) {
          KERNEL_8x8(ii,jj);
       } ورنہ اگر constexpr(RM == 8 && RN == 4) {
          KERNEL_8x4(ii,jj);
       } ورنہ {
          أكد(جھوٹ && "RN/RM values not supported");
       }
    }

    خالی matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles) {
            end = tiles;
        }
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = (کام / xtiles) * mc;
            int64_t jj = (کام % xtiles) * nc;
            برائے (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 8];
                 vec_t B_pack[kc * nc / 8];
                 pack_معمول(A + (ii * lda) + kk, lda, kc, mc, (uint8_t *)A_pack);
                 pack_معمول(B + (jj * ldb) + kk, ldb, kc, nc, (uint8_t *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    سانچہ <صحیح RM, صحیح RN>
    NOINLINE خالی gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    ثابت TA *ثابت A;
    ثابت TB *ثابت B;
    TC *C;
    ثابت int64_t k;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
    ثابت صحیح ith;
    ثابت صحیح nth;
};

سانچہ <قسم_نام TA>
جماعت tinyBLAS_Q0_PPC {
  عوامی:
    tinyBLAS_Q0_PPC(int64_t k,
             ثابت TA * A, int64_t lda,
             ثابت بلاک_q8_0 * B, int64_t ldb,
             اعشاری * C, int64_t ldc,
             صحیح ith, صحیح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    خالی matmul(int64_t m, int64_t n) {
        int64_t mc = 64;
        int64_t nc = 64;
        int64_t kc = 64;
        int64_t n_chunk = 64;
    #اگر defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 32;
        nc = 32;
        kc = 32;
        n_chunk = 32;
    #اگر_ختم
        int64_t n_aligned = 0;
        اگر (n % n_chunk == 0) {
            n_aligned = n;
        } ورنہ اگر (n == 4) {
            n_aligned = 4;
        } ورنہ اگر (n < n_chunk) {
            n_aligned = (n / 8) * 8;
        } ورنہ {
            n_aligned = (n / n_chunk) * n_chunk;
        }
        اگر (n_aligned > 0) {
            اگر (n_aligned % n_chunk == 0) nc = n_chunk;
            ورنہ اگر (n_aligned == n)      nc = n;
            ورنہ اگر (n_aligned % 32 == 0) nc = 32;
            ورنہ اگر (n_aligned % 24 == 0) nc = 24;
            ورنہ اگر (n_aligned % 16 == 0) nc = 16;
            ورنہ                          nc = 8;
        }
        منطقی can_use_tiled = n_aligned > 0 && (m % mc == 0);
        اگر (can_use_tiled) {
            matmul_tiled(m, n_aligned, mc, nc, kc);
            اگر (n > n_aligned) {
                mnpack(0, m, n_aligned, n);
            }
        } ورنہ {
            mnpack(0, m, 0, n);
        }
    }

  نجی:
    اندرونی خالی save_res(صحیح ii, صحیح jj, صحیح idx, vector اعشاری * fin_res, صحیح RM = 4, صحیح RN = 4) {
        برائے (صحیح I = 0; I < RM; I++) {
            برائے (صحیح J = 0; J < RN; J++) {
                *((اعشاری *)(C + ii + ((jj + J) * ldc) + I)) = *((اعشاری *)&fin_res[idx + I] + J);
            }
        }
    }

    اندرونی خالی save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        برائے (صحیح I = 0; I < 4; I++) {
            برائے (صحیح J = 0; J < 4; J++) {
                *((اعشاری *)(C + ii + ((jj + J) * ldc) + I)) = *((اعشاری *)&vec_C[I] + J);
            }
        }
    }

    اندرونی خالی جوڑو_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        برائے (صحیح I = 0; I < 4; I++) {
            برائے (صحیح J = 0; J < 4; J++) {
                اعشاری * c_ptr = (اعشاری *)(C + ii+ ((jj + J) * ldc) + I);
                *c_ptr += *((اعشاری *)&vec_C[I] + J);
            }
        }
    }

    سانچہ<قسم_نام ArrayType>
    اندرونی خالی compute(acc_t * ACC, صحیح c_idx, صحیح s_idx, ArrayType & comparray, vector اعشاری * vs, vector اعشاری * fin_res) {
        vector علامتی صحیح vec_C[4];
        vector اعشاری CA[4] = {0};
        vector اعشاری res[4] = {0};
        __builtin_mma_disassemble_acc(vec_C, ACC);
        برائے (صحیح i = 0; i < 4; i++) {
            CA[i] = vec_splats((اعشاری)(((دوہرا)comparray[c_idx + i]) * -128.0));
            res[i] = vec_جوڑو(vec_ctf(vec_C[i], 0), CA[i]);
            fin_res[s_idx + i] = vec_madd(res[i], vs[s_idx + i], fin_res[s_idx + i]);
        }
    }

    اندرونی خالی process_q4_elements(vector علامتی حرف (&c)[2], صحیح * ca) {
        ثابت vector علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
        ثابت vector غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);
        ثابت vector علامتی حرف v8 = vec_splats((علامتی حرف)0x8);
        vector علامتی صحیح vsum = {0};
        vector علامتی صحیح vsum2 = {0};
        c[0] = vec_and(c[1], کم_نقاب);
        c[1] = vec_sr(c[1], v4);
        c[0] = vec_sub(c[0], v8);
        c[1] = vec_sub(c[1], v8);
        vsum = vec_sum4s(c[0], vsum);
        vsum2 = vec_sum4s(c[1], vsum2);
        vsum = vec_جوڑو(vsum, vsum2);
        *(ca) = vsum[0] + vsum[1] + vsum[2] + vsum[3];
    }

    سانچہ <قسم_نام V1, قسم_نام V2>
    اندرونی خالی سمتیہ_ترتیب_بدلو_store(V2 & s1, V2 & s2, V2 & s3, V2 & s4, V1 * vec_انحراف, منطقی flip) {
        vector غیرعلامتی حرف swiz1 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vector غیرعلامتی حرف swiz2 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        vector غیرعلامتی حرف swiz3 = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
        vector غیرعلامتی حرف swiz4 = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
        V2 t1, t2, t3, t4, t5, t6, t7, t8;
        vector غیرعلامتی حرف xor_سمتیہ;
        uint8_t flip_vec = 0x80;
        xor_سمتیہ = vec_splats(flip_vec);
        t1 = vec_perm(s1, s2, swiz1);
        t2 = vec_perm(s1, s2, swiz2);
        t3 = vec_perm(s3, s4, swiz1);
        t4 = vec_perm(s3, s4, swiz2);
        t5 = vec_perm(t1, t3, swiz3);
        t6 = vec_perm(t1, t3, swiz4);
        t7 = vec_perm(t2, t4, swiz3);
        t8 = vec_perm(t2, t4, swiz4);
        اگر (flip == سچ) {
            t5 = vec_xor(t5, xor_سمتیہ);
            t6 = vec_xor(t6, xor_سمتیہ);
            t7 = vec_xor(t7, xor_سمتیہ);
            t8 = vec_xor(t8, xor_سمتیہ);
        }
        vec_xst(t5, 0, vec_انحراف);
        vec_xst(t6, 0, vec_انحراف + 16);
        vec_xst(t7, 0, vec_انحراف + 32);
        vec_xst(t8, 0, vec_انحراف + 48);
    }

    اندرونی خالی unpack_q4_مقصد_q8(vector علامتی حرف packed, vector علامتی حرف & lo, vector علامتی حرف & hi) {
        ثابت vector علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0x0F);
        ثابت vector علامتی حرف v8      = vec_splats((علامتی حرف)0x08);
        ثابت vector غیرعلامتی حرف v4    = vec_splats((غیرعلامتی حرف)4);
        lo = vec_and(packed, کم_نقاب);
        hi = vec_sr(packed, v4);
        lo = vec_sub(lo, v8);
        hi = vec_sub(hi, v8);
    }

    اندرونی خالی سمتیہ_ترتیب_بدلو_store_fp16(vec_t * c, غیرعلامتی حرف * vec_انحراف) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        برائے (صحیح i = 0; i < 4; i += 2) {
            t[i + 0] = vec_perm(c[i + 0], c[i + 1], swiz1);
            t[i + 1] = vec_perm(c[i + 0], c[i + 1], swiz2);
        }
        برائے (صحیح i = 4; i < 8; i += 2) {
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
        برائے (صحیح i = 0; i < 8; ++i) {
            vec_xst(s[i], 0, (vec_t *)(vec_انحراف + i * 16));
        }
    }

    ساکن اندرونی خالی convert_and_scale_q8(vector علامتی حرف raw, vector اعشاری v_scale, vector غیرعلامتی مختصر & برآمد_hi, vector غیرعلامتی مختصر & برآمد_lo) {
        vector علامتی مختصر i16_hi = vec_unpackh(raw);
        vector علامتی مختصر i16_lo = vec_unpackl(raw);

        vector اعشاری f_hi_h = vec_ctf(vec_unpackh(i16_hi), 0);
        vector اعشاری f_hi_l = vec_ctf(vec_unpackl(i16_hi), 0);
        vector اعشاری f_lo_h = vec_ctf(vec_unpackh(i16_lo), 0);
        vector اعشاری f_lo_l = vec_ctf(vec_unpackl(i16_lo), 0);
        برآمد_hi = vec_pack_مقصد_short_fp32(vec_mul(f_hi_h, v_scale), vec_mul(f_hi_l, v_scale));
        برآمد_lo = vec_pack_مقصد_short_fp32(vec_mul(f_lo_h, v_scale), vec_mul(f_lo_l, v_scale));
    }

    خالی pack_معمول_q4_fp16(ثابت بلاک_q4_0 * a, int64_t lda, صحیح صفیں, صحیح بلاکیں, غیرعلامتی حرف * vec) {
        غیرعلامتی حرف * vec_انحراف = vec;
        برائے (صحیح i = 0; i < صفیں; i += 8) {
            ثابت بلاک_q4_0 * صفیں_بنیاد[8];
            برائے (صحیح r = 0; r < 8; r++) {
                صفیں_بنیاد[r] = a + (i + r) * lda;
            }
            برائے (صحیح blk = 0; blk < بلاکیں; blk++) {
                vector غیرعلامتی مختصر hp_res[8][4];
                برائے (صحیح r = 0; r < 8; r++) {
                    ثابت بلاک_q4_0 * current_blk = صفیں_بنیاد[r] + blk;
                    vector اعشاری v_scale = vec_نکالو_fp32_from_shorth(vec_splats(current_blk->d));
                    vector علامتی حرف v_qs = vec_xl(0, (ثابت vector علامتی حرف *)current_blk->qs);
                    vector علامتی حرف c1, c2;
                    unpack_q4_مقصد_q8(v_qs, c1, c2);
                    convert_and_scale_q8(c1, v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c2, v_scale, hp_res[r][2], hp_res[r][3]);
                }
                برائے (صحیح c = 0; c < 4; c++) {
                    vector غیرعلامتی حرف c_arr[8];
                    برائے (صحیح r = 0; r < 8; r++) {
                        c_arr[r] = (vector غیرعلامتی حرف)hp_res[r][c];
                    }
                    سمتیہ_ترتیب_بدلو_store_fp16((vec_t *)c_arr, vec_انحراف);
                    vec_انحراف += 128;
                }
            }
        }
    }

    سانچہ <صحیح chunk_سائز>
    ساکن اندرونی خالی pack_q8_بلاک(ثابت بلاک_q8_0 * a, int64_t lda, صحیح صفیں, صحیح بلاکیں, غیرعلامتی حرف * vec) {
        غیرعلامتی حرف * vec_انحراف = vec;
        ثابت vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        ثابت vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        ثابت vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        ثابت vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        برائے (صحیح i = 0; i < صفیں; i += chunk_سائز) {
            ثابت بلاک_q8_0 * صفیں_بنیاد[chunk_سائز];
            برائے (صحیح r = 0; r < chunk_سائز; r++) {
                صفیں_بنیاد[r] = a + (i + r) * lda;
            }
            برائے (صحیح blk = 0; blk < بلاکیں; blk++) {
                vector غیرعلامتی مختصر hp_res[chunk_سائز][4];
                برائے (صحیح r = 0; r < chunk_سائز; r++) {
                    ثابت بلاک_q8_0 * b = صفیں_بنیاد[r] + blk;
                    vector اعشاری v_scale = vec_نکالو_fp32_from_shorth(vec_splats(b->d));
                    vector علامتی حرف c[2];
                    __vector_pair pair = __builtin_vsx_lxvp(0, (__vector_pair *)b->qs);
                    __builtin_vsx_disassemble_pair(c, & pair);
                    convert_and_scale_q8(c[0], v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c[1], v_scale, hp_res[r][2], hp_res[r][3]);
                }
                برائے (صحیح ستون = 0; ستون < 4; ستون++) {
                    اگر constexpr (chunk_سائز == 8) {
                        vec_t t[8];
                        t[0] = vec_perm((vec_t)hp_res[0][ستون], (vec_t)hp_res[1][ستون], swiz1);
                        t[1] = vec_perm((vec_t)hp_res[0][ستون], (vec_t)hp_res[1][ستون], swiz2);
                        t[2] = vec_perm((vec_t)hp_res[2][ستون], (vec_t)hp_res[3][ستون], swiz1);
                        t[3] = vec_perm((vec_t)hp_res[2][ستون], (vec_t)hp_res[3][ستون], swiz2);
                        t[4] = vec_perm((vec_t)hp_res[4][ستون], (vec_t)hp_res[5][ستون], swiz1);
                        t[5] = vec_perm((vec_t)hp_res[4][ستون], (vec_t)hp_res[5][ستون], swiz2);
                        t[6] = vec_perm((vec_t)hp_res[6][ستون], (vec_t)hp_res[7][ستون], swiz1);
                        t[7] = vec_perm((vec_t)hp_res[6][ستون], (vec_t)hp_res[7][ستون], swiz2);

                        vec_xst(vec_perm(t[0], t[2], swiz3), 0, (vec_t *)(vec_انحراف + 0));
                        vec_xst(vec_perm(t[0], t[2], swiz4), 0, (vec_t *)(vec_انحراف + 16));
                        vec_xst(vec_perm(t[1], t[3], swiz3), 0, (vec_t *)(vec_انحراف + 32));
                        vec_xst(vec_perm(t[1], t[3], swiz4), 0, (vec_t *)(vec_انحراف + 48));
                        vec_xst(vec_perm(t[4], t[6], swiz3), 0, (vec_t *)(vec_انحراف + 64));
                        vec_xst(vec_perm(t[4], t[6], swiz4), 0, (vec_t *)(vec_انحراف + 80));
                        vec_xst(vec_perm(t[5], t[7], swiz3), 0, (vec_t *)(vec_انحراف + 96));
                        vec_xst(vec_perm(t[5], t[7], swiz4), 0, (vec_t *)(vec_انحراف + 112));
                        vec_انحراف += 128;
                    } ورنہ {
                        vec_t t0 = vec_perm((vec_t)hp_res[0][ستون], (vec_t)hp_res[1][ستون], swiz1);
                        vec_t t1 = vec_perm((vec_t)hp_res[0][ستون], (vec_t)hp_res[1][ستون], swiz2);
                        vec_t t2 = vec_perm((vec_t)hp_res[2][ستون], (vec_t)hp_res[3][ستون], swiz1);
                        vec_t t3 = vec_perm((vec_t)hp_res[2][ستون], (vec_t)hp_res[3][ستون], swiz2);

                        vec_xst(vec_perm(t0, t2, swiz3), 0, (vec_t *)(vec_انحراف + 0));
                        vec_xst(vec_perm(t0, t2, swiz4), 0, (vec_t *)(vec_انحراف + 16));
                        vec_xst(vec_perm(t1, t3, swiz3), 0, (vec_t *)(vec_انحراف + 32));
                        vec_xst(vec_perm(t1, t3, swiz4), 0, (vec_t *)(vec_انحراف + 48));
                        vec_انحراف += 64;
                    }
                }
            }
        }
    }

    خالی pack_معمول_q8_fp16(ثابت بلاک_q8_0 * a, int64_t lda, صحیح صفیں, صحیح بلاکیں, غیرعلامتی حرف * vec) {
        اگر (صفیں == 4) {
            pack_q8_بلاک<4>(a, lda, صفیں, بلاکیں, vec);
        } ورنہ {
            pack_q8_بلاک<8>(a, lda, صفیں, بلاکیں, vec);
        }
    }

    سانچہ<صحیح size>
    خالی pack_معمول_Int4(ثابت TA * a, int64_t lda, صحیح صفیں, صحیح ستون_تعداد_2, int8_t * vec, std::array<صحیح, size> & comparray) {
        int64_t i, j;
        TA * aoffset = تہی;
        int8_t * vec_انحراف = تہی;
        TA * aoffset1 = تہی, * aoffset2 = تہی, * aoffset3 = تہی, * aoffset4 = تہی;
        TA * aoffset5 = تہی, * aoffset6 = تہی, * aoffset7 = تہی, * aoffset8 = تہی;
        vector علامتی حرف c1[2] = {0}, c2[2] = {0}, c3[2] = {0}, c4[2] = {0};
        vector علامتی حرف c5[2] = {0}, c6[2] = {0}, c7[2] = {0}, c8[2] = {0};
        aoffset = const_cast<TA *>(a);
        vec_انحراف = vec;
        j = (صفیں >> 3);
        اگر (j > 0) {
            کرو {
                aoffset1 = aoffset;
                aoffset2 = aoffset1 + lda;
                aoffset3 = aoffset2 + lda;
                aoffset4 = aoffset3 + lda;
                aoffset5 = aoffset4 + lda;
                aoffset6 = aoffset5 + lda;
                aoffset7 = aoffset6 + lda;
                aoffset8 = aoffset7 + lda;
                aoffset += 8 * lda;
                i = (ستون_تعداد_2 >> 2);
                اگر (i > 0) {
                    کرو {
                        c1[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset1->qs);
                        c2[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset2->qs);
                        c3[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset3->qs);
                        c4[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset4->qs);
                        c5[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset5->qs);
                        c6[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset6->qs);
                        c7[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset7->qs);
                        c8[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset8->qs);

                        process_q4_elements(c1, & comparray[0]);
                        process_q4_elements(c2, & comparray[1]);
                        process_q4_elements(c3, & comparray[2]);
                        process_q4_elements(c4, & comparray[3]);
                        process_q4_elements(c5, & comparray[4]);
                        process_q4_elements(c6, & comparray[5]);
                        process_q4_elements(c7, & comparray[6]);
                        process_q4_elements(c8, & comparray[7]);
                        سمتیہ_ترتیب_بدلو_store<int8_t, vector علامتی حرف>(c1[0], c2[0], c3[0], c4[0], vec_انحراف, جھوٹ);
                        سمتیہ_ترتیب_بدلو_store<int8_t, vector علامتی حرف>(c1[1], c2[1], c3[1], c4[1], vec_انحراف + 64, جھوٹ);
                        سمتیہ_ترتیب_بدلو_store<int8_t, vector علامتی حرف>(c5[0], c6[0], c7[0], c8[0], vec_انحراف + 128, جھوٹ);
                        سمتیہ_ترتیب_بدلو_store<int8_t, vector علامتی حرف>(c5[1], c6[1], c7[1], c8[1], vec_انحراف + 192, جھوٹ);
                        aoffset1 += lda;
                        aoffset2 += lda;
                        aoffset3 += lda;
                        aoffset4 += lda;
                        aoffset5 += lda;
                        aoffset6 += lda;
                        aoffset7 += lda;
                        aoffset8 += lda;
                        vec_انحراف += 256;
                        i--;
                    } جب (i > 0);
                }
                j--;
            } جب (j > 0);
        }

        اگر (صفیں & 4) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset += 4 * lda;
            i = (ستون_تعداد_2 >> 2);
            اگر (i > 0) {
                کرو {
                    c1[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset1->qs);
                    c2[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset2->qs);
                    c3[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset3->qs);
                    c4[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset4->qs);

                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    سمتیہ_ترتیب_بدلو_store<int8_t, vector علامتی حرف>(c1[0], c2[0], c3[0], c4[0], vec_انحراف, جھوٹ);
                    سمتیہ_ترتیب_بدلو_store<int8_t, vector علامتی حرف>(c1[1], c2[1], c3[1], c4[1], vec_انحراف + 64, جھوٹ);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    aoffset4 += lda;
                    vec_انحراف += 128;
                    i--;
                } جب (i > 0);
            }
        }

        اگر (صفیں & 3) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            i = (ستون_تعداد_2 >> 2);
            اگر (i > 0) {
                کرو {
                    انتخاب(صفیں) {
                        حالت 3: c3[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset3->qs);
                        حالت 2: c2[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset2->qs);
                        حالت 1: c1[1] = vec_xl(0, (ثابت vector علامتی حرف *)aoffset1->qs);
                            توڑو;
                    }
                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    سمتیہ_ترتیب_بدلو_store<int8_t, vector علامتی حرف>(c1[0], c2[0], c3[0], c4[0], vec_انحراف, جھوٹ);
                    سمتیہ_ترتیب_بدلو_store<int8_t, vector علامتی حرف>(c1[1], c2[1], c3[1], c4[1], vec_انحراف + 64, جھوٹ);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    vec_انحراف += 128;
                    i--;
                } جب(i > 0);
            }
        }
    }

    سانچہ<قسم_نام VA, قسم_نام VB>
    خالی pack_معمول(ثابت بلاک_q8_0 * a, int64_t lda, صحیح صفیں, صحیح ستون_تعداد_2, VA * vec, منطقی flip) {
        int64_t i, j;
        بلاک_q8_0 * aoffset = تہی;
        VA * vec_انحراف = تہی;
        بلاک_q8_0 * aoffsets[8];
        __vector_pair arr[8];
        VB c[8][2] = {0};
        VB c1[8] = {0}; VB c2[8] = {0};
        aoffset = const_cast<بلاک_q8_0 *>(a);
        vec_انحراف = vec;
        j = (صفیں >> 3);
        اگر (j > 0) {
            کرو {
                aoffsets[0] = aoffset;
                برائے (صحیح it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it - 1] + lda;
                aoffset += 8 * lda;

                i = (ستون_تعداد_2 >> 3);
                اگر (i > 0) {
                کرو {
                    برائے (صحیح it = 0; it < 8; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    سمتیہ_ترتیب_بدلو_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_انحراف, flip);
                    سمتیہ_ترتیب_بدلو_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_انحراف + 64, flip);
                    سمتیہ_ترتیب_بدلو_store<VA, VB>(c1[4], c1[5], c1[6], c1[7], vec_انحراف + 128, flip);
                    سمتیہ_ترتیب_بدلو_store<VA, VB>(c2[4], c2[5], c2[6], c2[7], vec_انحراف + 192, flip);
                    برائے (صحیح it = 0; it < 8; it++)
                        aoffsets[it] += lda;
                    vec_انحراف += 256;
                    i--;
               } جب(i > 0);
            }
            j--;
        } جب(j > 0);
    }
    اگر (صفیں & 4) {
            aoffsets[0]  = aoffset;
            برائے (صحیح it = 1; it < 4; it++ )
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
        i = (ستون_تعداد_2 >> 3);
            اگر (i > 0) {
               کرو {
                    برائے (صحیح it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    سمتیہ_ترتیب_بدلو_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_انحراف, flip);
                    سمتیہ_ترتیب_بدلو_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_انحراف + 64, flip);
                    برائے (صحیح it = 0; it < 4; it++) {
                        aoffsets[it] += lda;
                    }
                    vec_انحراف += 128;
                    i--;
               } جب(i > 0);
            }
        }

        اگر (صفیں & 3) {
            aoffsets[0]  = aoffset;
            برائے (صحیح it = 1; it < 3; it++ )
                aoffsets[it] = aoffsets[it - 1] + lda;
            i = (ستون_تعداد_2 >> 3);
            اگر (i > 0) {
                کرو {
                    انتخاب(صفیں) {
                        حالت 3: arr[2] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[2]->qs);
                                __builtin_vsx_disassemble_pair(c[2], & arr[2]);
                                c1[2] = c[2][0]; c2[2] = c[2][1];
                        حالت 2: arr[1] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[1]->qs);
                                __builtin_vsx_disassemble_pair(c[1], & arr[1]);
                                c1[1] = c[1][0]; c2[1] = c[1][1];
                        حالت 1: arr[0] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[0]->qs);
                                __builtin_vsx_disassemble_pair(c[0], & arr[0]);
                                c1[0] = c[0][0]; c2[0] = c[0][1];
                                توڑو;
                    }
                    سمتیہ_ترتیب_بدلو_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vec_انحراف, flip);
                    سمتیہ_ترتیب_بدلو_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vec_انحراف + 64, flip);
                    برائے (صحیح it = 0; it < 3; it++)
                         aoffsets[it] += lda;
                    vec_انحراف += 128;
                    i--;
               } جب(i > 0);
            }
        }
    }

    خالی mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        صحیح m_rem = MIN(m - m0, 16);
        صحیح n_rem = MIN(n - n0, 16);

        صحیح mc = 0, nc = 0;

        اگر (m_rem >= 8 && n_rem >= 8) {
           mc = 8;
           nc = 8;
           gemm<8, 8>(m0, m, n0, n);
        } ورنہ اگر (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } ورنہ اگر (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } ورنہ اگر (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small(m0, m, n0, n, mc, nc);
        } ورنہ {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            اگر (mc == 0 || nc == 0)
               واپس;
            gemm_small(m0, m, n0, n, mc, nc);
        }

        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }


    خالی KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[16] = {0};
        acc_t acc_0, acc_1;
        std::array<صحیح, 4> comparray {};
        vector اعشاری fin_res[8] = {0};
        vector اعشاری vs[8] = {0};
        منطقی isAblock_q4 = std::is_same_v<TA, بلاک_q4_0>;
        برائے (صحیح l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            اگر (std::is_same_v<TA, بلاک_q4_0>) {
               pack_معمول_Int4<4>((A + (ii * lda) + l), lda, 4, 4, (int8_t *)vec_A, comparray);
            } ورنہ {
               pack_معمول<int8_t, vector علامتی حرف>((ثابت بلاک_q8_0 *)(A + (ii * lda) + l), lda, 4, 8, (int8_t *)vec_A, جھوٹ);
            }
            pack_معمول<uint8_t, vector غیرعلامتی حرف>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, سچ);
            برائے(صحیح x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x], vec_B[x+8]);
            }
            برائے (صحیح I = 0; I<4; I++) {
                برائے (صحیح J = 0; J<4; J++) {
                    *((اعشاری *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((اعشاری *)& vs[I + 4] + J) = (unhalf((A +((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            اگر (!isAblock_q4) {
                خودکار aoffset = A + (ii * lda) + l;
                برائے (صحیح i = 0; i < 4; i++) {
                    comparray[i] = 0;
                    صحیح ca = 0;
                    خودکار *at = aoffset->qs;
                    برائے (صحیح j = 0; j < 32; j++)
                        ca += (صحیح)*at++;
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

    خالی KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[8] = {0};
        acc_t acc_0, acc_1;
        std::array<صحیح, 8> comparray {};
        vector اعشاری fin_res[8] = {0};
        vector اعشاری vs[8] = {0};
        منطقی isAblock_q4 = std::is_same_v<TA, بلاک_q4_0>;
        برائے (صحیح l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            اگر (std::is_same_v<TA, بلاک_q4_0>) {
               pack_معمول_Int4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } ورنہ {
               pack_معمول<int8_t, vector علامتی حرف>((ثابت بلاک_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, جھوٹ);
            }
            pack_معمول<uint8_t, vector غیرعلامتی حرف>((B + (jj * ldb) + l), ldb, 4, 8, (uint8_t *)vec_B, سچ);
            برائے(صحیح x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
            }
            برائے (صحیح I = 0; I < 8; I++) {
                برائے (صحیح J = 0; J < 4; J++) {
                    *((اعشاری *)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                }
            }
            اگر (!isAblock_q4) {
                خودکار aoffset = A + (ii * lda) + l;
                برائے (صحیح i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    صحیح ca = 0;
                    خودکار *at = aoffset->qs;
                    برائے (صحیح j = 0; j < 32; j++)
                        ca += (صحیح)*at++;
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

    خالی KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16] = {0};
        acc_t acc_0, acc_1, acc_2, acc_3;
        acc_t acc_4, acc_5, acc_6, acc_7;
        std::array<صحیح, 8> comparray {};
        vector اعشاری fin_res[16] = {0};
        vector اعشاری vs[16] = {0};
        منطقی isAblock_q4 = std::is_same_v<TA, بلاک_q4_0>;
        برائے (صحیح l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            __builtin_mma_xxsetaccz(& acc_2);
            __builtin_mma_xxsetaccz(& acc_3);
            اگر (std::is_same_v<TA, بلاک_q4_0>) {
               pack_معمول_Int4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } ورنہ {
               pack_معمول<int8_t, vector علامتی حرف>((ثابت بلاک_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, جھوٹ);
            }
            pack_معمول<uint8_t, vector غیرعلامتی حرف>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, سچ);
            برائے(صحیح x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_2, vec_A[x], vec_B[x + 8]);
                __builtin_mma_xvi8ger4pp(& acc_3, vec_A[x + 8], vec_B[x + 8]);
            }
            برائے (صحیح I = 0; I < 8 ; I++) {
                برائے (صحیح J = 0; J < 4; J++) {
                    *((اعشاری *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((اعشاری *)& vs[I + 8] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            اگر (!isAblock_q4) {
                خودکار aoffset = A + (ii * lda) + l;
                برائے (صحیح i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    صحیح ca = 0;
                    خودکار *at = aoffset->qs;
                    برائے (صحیح j = 0; j < 32; j++)
                        ca += (صحیح)*at++;
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

    خالی KERNEL_Q0(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, int64_t l, vec_t * vec_A, vec_t * vec_B) {
        acc_t acc[8];
        برائے (صحیح i = 0; i < mc ; i += 16) {
            برائے (صحیح j = 0; j < nc; j += 8) {
                صحیح A0_بنیاد = (i / 16) * (2 * 32 * kc);
                صحیح B0_بنیاد = (j / 8) * (32 * kc);
                برائے (صحیح x = 0; x < 8; x++) {
                     __builtin_mma_xxsetaccz(&acc[x]);
                }
                برائے (int64_t kk = 0; kk < kc; kk++) {
                    صحیح A0_بلاک_idx = A0_بنیاد + kk * 32;
                    صحیح B0_بلاک_idx = B0_بنیاد + kk * 32;
                    صحیح A1_بلاک_idx = A0_بلاک_idx + 32 * kc;
                    صحیح B1_بلاک_idx = B0_بلاک_idx + 32 * kc;
                    vec_t * A0_بلاک = & vec_A[A0_بلاک_idx];
                    vec_t * B0_بلاک = & vec_B[B0_بلاک_idx];
                    vec_t * A1_بلاک = & vec_A[A1_بلاک_idx];
                    برائے (صحیح it = 0; it < 4; it++) {
                        برائے (صحیح x = 0; x < 4; x++) {
                            __builtin_mma_xvf16ger2pp(& acc[0], A0_بلاک[8 * it + x], B0_بلاک[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[1], A0_بلاک[8 * it + x], B0_بلاک[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[2], A0_بلاک[8 * it + x + 4], B0_بلاک[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[3], A0_بلاک[8 * it + x + 4], B0_بلاک[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[4], A1_بلاک[8 * it + x], B0_بلاک[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[5], A1_بلاک[8 * it + x], B0_بلاک[8 * it+ x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[6], A1_بلاک[8 * it + x + 4], B0_بلاک[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[7], A1_بلاک[8 * it + x + 4], B0_بلاک[8 * it + x + 4]);
                        }
                    }
                }
                اگر (l == 0) {
                    save_acc(& acc[0], ii + i, jj + j);
                    save_acc(& acc[1], ii + i, jj + j + 4);
                    save_acc(& acc[2], ii + i + 4, jj + j);
                    save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    save_acc(& acc[4], ii + i + 8, jj + j);
                    save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    save_acc(& acc[6], ii + i + 12, jj + j);
                    save_acc(& acc[7], ii + i + 12, jj + j + 4);
                } ورنہ {
                    جوڑو_save_acc(& acc[0], ii + i, jj + j);
                    جوڑو_save_acc(& acc[1], ii + i, jj + j + 4);
                    جوڑو_save_acc(& acc[2], ii + i + 4, jj + j);
                    جوڑو_save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    جوڑو_save_acc(& acc[4], ii + i + 8, jj + j);
                    جوڑو_save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    جوڑو_save_acc(& acc[6], ii + i + 12, jj + j);
                    جوڑو_save_acc(& acc[7], ii + i + 12, jj + j + 4);
                }
            }
        }
    }

    خالی matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        vec_t A_pack[mc * kc * 4];
        vec_t B_pack[nc * kc * 4];
        constexpr منطقی is_Ablock_q4 = std::is_same_v<TA, بلاک_q4_0>;
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles  = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles) {
            end = tiles;
        }
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = (کام / xtiles) * mc;
            int64_t jj = (کام % xtiles) * nc;
            برائے (int64_t kk = 0; kk < k; kk += kc) {
                int64_t k_cur = MIN(kc, k - kk);
                اگر constexpr(is_Ablock_q4) {
                    pack_معمول_q4_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                } ورنہ {
                    pack_معمول_q8_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                }
                pack_معمول_q8_fp16(B + jj * ldb + kk, ldb, nc, k_cur, (uint8_t *)B_pack);
                KERNEL_Q0(ii, jj, mc, nc, k_cur, kk, A_pack, B_pack);
            }
        }
    }

    خالی gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, صحیح RM, صحیح RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        vec_t vec_A[8] = {0}, vec_B[8] = {0};
        vector علامتی صحیح vec_C[4];
        acc_t acc_0;
        منطقی isAblock_q4 = std::is_same_v<TA, بلاک_q4_0>;

        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * RN;
            std::array<صحیح, 4> comparray{};
            vector اعشاری res[4] = {0};
            vector اعشاری fin_res[4] = {0};
            vector اعشاری vs[4] = {0};
            vector اعشاری CA[4] = {0};
            __builtin_prefetch((A + (ii * lda) + 0)->qs, 0, 1); // prefetch first value
            __builtin_prefetch((B + (jj * ldb) + 0)->qs, 0, 1); // prefetch first value
            برائے (صحیح l = 0; l < k; l++) {
                __builtin_prefetch((A + (ii * lda) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_prefetch((B + (jj * ldb) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_mma_xxsetaccz(& acc_0);
                اگر (isAblock_q4) {
                    pack_معمول_Int4<4>((A + (ii * lda) + l), lda, RM, 4, (int8_t *)vec_A, comparray);
                } ورنہ {
                    pack_معمول<int8_t, vector علامتی حرف>((ثابت بلاک_q8_0 *)(A + (ii * lda) + l), lda, RM, 8, (int8_t *)vec_A, جھوٹ);
                }
                pack_معمول<uint8_t, vector غیرعلامتی حرف>((B + (jj * ldb) + l), ldb, RN, 8, (uint8_t *)vec_B, سچ);
                برائے (صحیح x = 0; x < 8; x += 4) {
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 1], vec_B[x + 1]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 2], vec_B[x + 2]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 3], vec_B[x + 3]);
                }
                برائے (صحیح I = 0; I < RM; I++) {
                    برائے (صحیح J = 0; J < RN; J++) {
                        *((اعشاری*)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    }
                }
                __builtin_mma_disassemble_acc(vec_C, & acc_0);
                اگر (!isAblock_q4) {
                    خودکار aoffset = A + (ii * lda) + l;
                    برائے (صحیح i = 0; i < RM; i++) {
                        comparray[i] = 0;
                        صحیح ca = 0;
                        خودکار *at = aoffset->qs;
                        برائے (صحیح j = 0; j < 32; j++)
                            ca += (صحیح)*at++;
                        comparray[i] = ca;
                        aoffset += lda;
                    }
                }
                برائے (صحیح i = 0; i < RM; i++) {
                    CA[i] = vec_splats((اعشاری)(((دوہرا)comparray[i]) * -128.0));
                    res[i] = vec_جوڑو(vec_ctf(vec_C[i], 0), CA[i]);
                    fin_res[i] = vec_madd(res[i], vs[i], fin_res[i]);
                }
            }
            save_res(ii, jj, 0, fin_res, RM, RN);
        }
    }

    سانچہ<صحیح RM, صحیح RN>
    اندرونی خالی kernel(int64_t ii, int64_t jj) {
        اگر constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii,jj);
        } ورنہ اگر constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii,jj);
        } ورنہ اگر constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii,jj);
        } ورنہ {
            أكد(جھوٹ && "RN/RM values not supported");
        }
    }

    سانچہ <صحیح RM, صحیح RN>
    NOINLINE خالی gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }
    ثابت TA * ثابت A;
    ثابت بلاک_q8_0 * ثابت B;
    اعشاری * C;
    ثابت int64_t k;
    int64_t kc;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
    ثابت صحیح ith;
    ثابت صحیح nth;
};

جماعت tinyBLAS_PPC {
  عوامی:
    tinyBLAS_PPC(int64_t k,
                ثابت اعشاری * A, int64_t lda,
                ثابت اعشاری * B, int64_t ldb,
                اعشاری * C, int64_t ldc,
                صحیح ith, صحیح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    خالی matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #اگر defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #اگر_ختم
        اگر (m % mc == 0 && n % nc == 0 && k % kc == 0) {
            matmul_tiled(m, n, mc, nc, kc);
        } ورنہ {
            mnpack(0, m, 0, n);
        }
    }

  نجی:

    __attribute__((always_inline))
    اندرونی خالی save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        برائے (صحیح I = 0; I < 4; I++) {
            برائے (صحیح J = 0; J < 4; J++) {
                *((اعشاری *)(C+ii+((jj+J)*ldc)+I)) = *((اعشاری *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    اندرونی خالی جوڑو_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        برائے (صحیح I = 0; I < 4; I++) {
            برائے (صحیح J = 0; J < 4; J++) {
                اعشاری * c_ptr = (اعشاری *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((اعشاری *)&vec_C[I]+J);
            }
        }
    }

    اندرونی خالی سمتیہ_ترتیب_بدلو_store_4(vector اعشاری * src, اعشاری * vec_انحراف) {
        vector اعشاری t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergel(src[0], src[1]);
        t4 = vec_mergel(src[2], src[3]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t1, t2, 3);
        t7 = vec_xxpermdi(t3, t4, 0);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_انحراف);
        vec_xst(t6, 0, vec_انحراف + 4);
        vec_xst(t7, 0, vec_انحراف + 8);
        vec_xst(t8, 0, vec_انحراف + 12);
    }

    اندرونی خالی سمتیہ_ترتیب_بدلو_store_8(vector اعشاری * src, اعشاری * vec_انحراف) {
        vector اعشاری t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergeh(src[4], src[5]);
        t4 = vec_mergeh(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_انحراف);
        vec_xst(t6, 0, vec_انحراف + 4);
        vec_xst(t7, 0, vec_انحراف + 8);
        vec_xst(t8, 0, vec_انحراف + 12);

        t1 = vec_mergel(src[0], src[1]);
        t2 = vec_mergel(src[2], src[3]);
        t3 = vec_mergel(src[4], src[5]);
        t4 = vec_mergel(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vec_انحراف + 16);
        vec_xst(t6, 0, vec_انحراف + 20);
        vec_xst(t7, 0, vec_انحراف + 24);
        vec_xst(t8, 0, vec_انحراف + 28);
    }

    خالی packTranspose(ثابت اعشاری * a, int64_t lda, صحیح صفیں, صحیح ستون_تعداد_2, اعشاری * vec) {
        int64_t i, j;
        اعشاری * aoffsets[8];
        اعشاری * aoffset = تہی, * boffset = تہی;
        __vector_pair arr[8];
        vector اعشاری c[8][2] = {0};
        vector اعشاری c1[8] = {0};
        vector اعشاری c2[8] = {0};
        aoffset = const_cast<اعشاری *>(a);
        boffset = vec;
        j = (صفیں >> 3);
        اگر (j > 0) {
            کرو {
                aoffsets[0] = aoffset;
                برائے (صحیح it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it-1] + lda;
                aoffset += 8 * lda;
                i = (ستون_تعداد_2 >> 3);
                اگر (i > 0) {
                    کرو {
                        برائے (صحیح it = 0; it < 8; it++) {
                            arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                            __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                            c1[it] = c[it][0];
                            c2[it] = c[it][1];
                        }

                        سمتیہ_ترتیب_بدلو_store_8(c1, boffset);
                        سمتیہ_ترتیب_بدلو_store_8(c2, boffset + 32);
                        boffset += 64;
                        i--;
                        اگر (i > 0) {
                           برائے (صحیح it = 0; it < 8; it++) {
                               aoffsets[it] = aoffsets[it] + 8;
                           }
                        }
                    } جب(i > 0);
                }
                اگر (ستون_تعداد_2 & 4) {
                    برائے (صحیح it = 0; it < 8 ; it++)
                        c1[it] = vec_xl(0, aoffsets[it]);
                    سمتیہ_ترتیب_بدلو_store_8(c1, boffset);
                }
            j--;
            } جب(j > 0);
        }

        اگر (صفیں & 4) {
            aoffsets[0] = aoffset;
            برائے (صحیح it = 1; it < 4; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            i = (ستون_تعداد_2 >> 3);
            اگر (i > 0) {
                کرو {
                    برائے (صحیح it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                        __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    سمتیہ_ترتیب_بدلو_store_4(c1, boffset);
                    سمتیہ_ترتیب_بدلو_store_4(c2, boffset + 16);
                    برائے (صحیح it = 0; it < 4; it++)
                        aoffsets[it] += 8 * lda;
                    boffset += 32;
                    i--;
                } جب(i > 0);
            }

            اگر (ستون_تعداد_2 & 4) {
               برائے (صحیح it = 0; it < 4; it++)
                   c1[it] = vec_xl(0, aoffsets[it]);
                سمتیہ_ترتیب_بدلو_store_4(c1, boffset);
            }
        }
        اگر (صفیں & 3) {
            aoffsets[0] = aoffset;
            برائے (صحیح it = 1; it < 3; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            اگر (ستون_تعداد_2 & 4) {
                برائے (صحیح it = 0; it < 3; it++)
                    c1[it] = vec_xl(0, aoffsets[it]);
                سمتیہ_ترتیب_بدلو_store_4(c1, boffset);
            }
        }
    }

    خالی KERNEL_4x4(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[4], vec_C[4];
        acc_t acc_0;
        __builtin_mma_xxsetaccz(&acc_0);
        برائے (صحیح l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (اعشاری *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (اعشاری *)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
        }
        save_acc(&acc_0, ii, jj);
    }

    خالی KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        برائے (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (اعشاری *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 4, (اعشاری *)vec_B);
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

    خالی KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        برائے (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 8, 4, (اعشاری *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (اعشاری *)vec_B);
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

    خالی KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        برائے (صحیح l = 0; l < k; l+=8) {
            packTranspose(A + (ii * lda) + l, lda, 8, 8, (اعشاری *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 8, (اعشاری *)vec_B);
            برائے(صحیح x = 0; x < 16; x+=2) {
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

    اندرونی خالی MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        برائے (صحیح x = 0; x < 16; x += 2) {
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

    خالی KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        برائے (int64_t i = 0; i < mc; i += 16) {
            صحیح A_بنیاد_addr = (mc / 8) * (i / 8) * 16;
            برائے (int64_t j = 0; j < nc; j += 8) {
                 صحیح B_بنیاد_addr = (nc / 8) * (j / 8) * 16;
                 acc_t acc[8];
                 vec_t A0_بلاک[16]; vec_t A1_بلاک[16];
                 برائے (صحیح x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 برائے (int64_t l = 0; l < kc; l += 8) {
                     صحیح A0_بلاک_idx = A_بنیاد_addr + (l / 8) * 16;
                     صحیح A1_بلاک_idx = A0_بلاک_idx + (mc / 8) * 16;
                     صحیح B_بلاک_idx = B_بنیاد_addr + (l / 8) * 16;
                     vec_t* A0_بلاک = &vec_A[A0_بلاک_idx];
                     vec_t* A1_بلاک = &vec_A[A1_بلاک_idx];
                     vec_t* B_بلاک = &vec_B[B_بلاک_idx];
                     MMA_16x8(A0_بلاک, A1_بلاک, B_بلاک, acc);
                 }
                 اگر (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } ورنہ {
                     جوڑو_save_acc(&acc[0], ii + i, jj + j);
                     جوڑو_save_acc(&acc[1], ii + i, jj + j + 4);
                     جوڑو_save_acc(&acc[2], ii + i + 4, jj + j);
                     جوڑو_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     جوڑو_save_acc(&acc[4], ii + i + 8, jj + j);
                     جوڑو_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     جوڑو_save_acc(&acc[6], ii + i + 12, jj + j);
                     جوڑو_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }

    خالی matmul_tiled(int64_t m , int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles) {
            end = tiles;
        }
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = (کام / xtiles) * mc;
            int64_t jj = (کام % xtiles) * nc;
            برائے (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 4];
                 vec_t B_pack[kc * nc / 4];
                 packTranspose(A + (ii * lda) + kk, lda, kc, mc, (اعشاری *)A_pack);
                 packTranspose(B + (jj * ldb) + kk, ldb, kc, nc, (اعشاری *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    خالی mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        صحیح m_rem = MIN(m - m0, 8);
        صحیح n_rem = MIN(n - n0, 8);
        صحیح mc = 0, nc = 0;
        اگر (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8, 8>(m0, m, n0, n);
        } ورنہ اگر (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } ورنہ اگر (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } ورنہ اگر (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n);
        } ورنہ {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            اگر (mc == 0 || nc == 0)
                واپس;
            gemm_small(m0, m, n0, n, mc, nc);
        }
        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    خالی gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, صحیح RM, صحیح RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[4] = {0}, vec_B[4] = {0};
            برائے (صحیح l = 0; l < k; l += 4) {
                /* 'GEMV Forwarding' concept is used in first two conditional loops.
                 * when one of the matrix has a single row/column, the elements are
                 * broadcasted, instead of using packing routine to prepack the
                 * matrix elements.
                 */
                اگر (RM == 1) {
                    اعشاری * a = const_cast<اعشاری *>(A + (ii) * lda + l);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (اعشاری *)vec_B);
                    vec_A[0] = (vec_t)vec_xl(0,a);
                    vec_A[1] = (vec_t)vec_splats(*((اعشاری *)&vec_A+1));
                    vec_A[2] = (vec_t)vec_splats(*((اعشاری *)&vec_A+2));
                    vec_A[3] = (vec_t)vec_splats(*((اعشاری *)&vec_A+3));
                } ورنہ اگر (RN == 1) {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (اعشاری *)vec_A);
                    اعشاری * b = const_cast<اعشاری *>(B + (jj) * ldb + l);
                    vec_B[0] = (vec_t)vec_xl(0,b);
                    vec_B[1] = (vec_t)vec_splats(*((اعشاری *)&vec_B+1));
                    vec_B[2] = (vec_t)vec_splats(*((اعشاری *)&vec_B+2));
                    vec_B[3] = (vec_t)vec_splats(*((اعشاری *)&vec_B+3));
                } ورنہ {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (اعشاری *)vec_A);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (اعشاری *)vec_B);
                }
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            برائے (صحیح I = 0; I < RM; I++) {
                برائے (صحیح J = 0; J < RN; J++) {
                    *((اعشاری *)(C+ii+((jj+J)*ldc)+I)) = *((اعشاری *)&vec_C[I]+J);
                }
            }
       }
    }

    سانچہ<صحیح RM, صحیح RN>
    اندرونی خالی kernel(int64_t ii, int64_t jj) {
        اگر constexpr(RM == 4 && RN == 4) {
            KERNEL_4x4(ii, jj);
        } ورنہ اگر constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii, jj);
        } ورنہ اگر constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii, jj);
        } ورنہ اگر constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii, jj);
        } ورنہ {
            static_assert(جھوٹ, "RN/RM values not supported");
        }
    }

    سانچہ <صحیح RM, صحیح RN>
    NOINLINE خالی gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        اگر (end > tiles)
            end = tiles;
        برائے (int64_t کام = start; کام < end; ++کام) {
            int64_t ii = m0 + کام / xtiles * RM;
            int64_t jj = n0 + کام % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    ثابت اعشاری * ثابت A;
    ثابت اعشاری * ثابت B;
    اعشاری * C;
    ثابت int64_t k;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
    ثابت صحیح ith;
    ثابت صحیح nth;
};
#اگر_ختم
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
منطقی llamafile_sgemm(ثابت ساخت ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                     ثابت خالی *A, int64_t lda, ثابت خالی *B, int64_t ldb, خالی *C,
                     int64_t ldc, صحیح Atype, صحیح Btype, صحیح Ctype) {

    أكد(m >= 0);
    أكد(n >= 0);
    أكد(k >= 0);
    أكد(lda >= k);
    أكد(ldb >= k);
    أكد(ldc >= m);
    أكد(params->nth > 0);
    أكد(params->ith < params->nth);

    // only enable sgemm for prompt processing
#اگر !defined(__MMA__)
    اگر (n < 2)
        واپس جھوٹ;
#اگر_ختم

    اگر (Ctype != GGML_TYPE_F32)
        واپس جھوٹ;

    انتخاب (Atype) {

    حالت GGML_TYPE_F32: {
        اگر (Btype != GGML_TYPE_F32)
            واپس جھوٹ;
#اگر defined(__AVX512F__)
        tinyBLAS<16, __m512, __m512, اعشاری, اعشاری, اعشاری> tb{ params,
            k, (ثابت اعشاری *)A, lda,
            (ثابت اعشاری *)B, ldb,
            (اعشاری *)C, ldc};
        واپس tb.matmul(m, n);
#وگرنہ_اگر defined(__AVX__) || defined(__AVX2__)
        tinyBLAS<8, __m256, __m256, اعشاری, اعشاری, اعشاری> tb{ params,
            k, (ثابت اعشاری *)A, lda,
            (ثابت اعشاری *)B, ldb,
            (اعشاری *)C, ldc};
        واپس tb.matmul(m, n);
#وگرنہ_اگر defined(__ARM_NEON)
        اگر (n < 4)
            واپس جھوٹ;
        tinyBLAS<4, float32x4_t, float32x4_t, اعشاری, اعشاری, اعشاری> tb{ params,
            k, (ثابت اعشاری *)A, lda,
            (ثابت اعشاری *)B, ldb,
            (اعشاری *)C, ldc};
        واپس tb.matmul(m, n);
#وگرنہ_اگر defined(__VXE__) || defined(__VXE2__)
        اگر (n < 4)
            واپس جھوٹ;
        tinyBLAS<4, float32x4_t, float32x4_t, اعشاری, اعشاری, اعشاری> tb{ params,
            k, (ثابت اعشاری *)A, lda,
            (ثابت اعشاری *)B, ldb,
            (اعشاری *)C, ldc};
        واپس tb.matmul(m, n);
#وگرنہ_اگر defined(__MMA__)
        اگر (k % 8)
            واپس جھوٹ;
        tinyBLAS_PPC tb{
            k, (ثابت اعشاری *)A, lda,
            (ثابت اعشاری *)B, ldb,
            (اعشاری *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        واپس سچ;
#وگرنہ_اگر defined(__riscv_v_intrinsic)
    #اگر LMUL == 1
        tinyBLAS_RVV<vfloat32m1_t, vfloat32m1_t, اعشاری, اعشاری, اعشاری> tb{ params,
            k, (ثابت اعشاری *)A, lda,
            (ثابت اعشاری *)B, ldb,
            (اعشاری *)C, ldc};
    #وگرنہ_اگر LMUL == 2
        tinyBLAS_RVV<vfloat32m2_t, vfloat32m2_t, اعشاری, اعشاری, اعشاری> tb{ params,
            k, (ثابت اعشاری *)A, lda,
            (ثابت اعشاری *)B, ldb,
            (اعشاری *)C, ldc};
    #ورنہ // LMUL = 4
        tinyBLAS_RVV<vfloat32m4_t, vfloat32m4_t, اعشاری, اعشاری, اعشاری> tb{ params,
            k, (ثابت اعشاری *)A, lda,
            (ثابت اعشاری *)B, ldb,
            (اعشاری *)C, ldc};
    #اگر_ختم
        واپس tb.matmul(m, n);
#ورنہ
        واپس جھوٹ;
#اگر_ختم
    }

    حالت GGML_TYPE_BF16: {
#اگر defined(__AVX512BF16__)
        اگر (Btype == GGML_TYPE_BF16) {
            tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, اعشاری> tb{ params, k,
                (ثابت ggml_bf16_t *)A, lda,
                (ثابت ggml_bf16_t *)B, ldb,
                (اعشاری *)C, ldc};
            واپس tb.matmul(m, n);
        }
#وگرنہ_اگر defined(__AVX512F__)
        اگر (Btype == GGML_TYPE_BF16) {
            tinyBLAS<16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, اعشاری> tb{ params, k,
                (ثابت ggml_bf16_t *)A, lda,
                (ثابت ggml_bf16_t *)B, ldb,
                (اعشاری *)C, ldc};
            واپس tb.matmul(m, n);
        }
#وگرنہ_اگر defined(__AVX2__)
        اگر (Btype == GGML_TYPE_BF16) {
            tinyBLAS<8, __m256, __m256, ggml_bf16_t, ggml_bf16_t, اعشاری> tb{ params, k,
                (ثابت ggml_bf16_t *)A, lda,
                (ثابت ggml_bf16_t *)B, ldb,
                (اعشاری *)C, ldc};
            واپس tb.matmul(m, n);
        }
#وگرنہ_اگر defined(__MMA__)
        اگر (k % 8) {
            واپس جھوٹ;
        }

        اگر (Btype == GGML_TYPE_BF16) {
            tinyBLAS_HP16_PPC<ggml_bf16_t, ggml_bf16_t, اعشاری> tb{ k,
                (ثابت ggml_bf16_t *)A, lda,
                (ثابت ggml_bf16_t *)B, ldb,
                (اعشاری *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            واپس سچ;
        }
#وگرنہ_اگر defined(__riscv_zvfbfwma)
        اگر (Btype == GGML_TYPE_BF16) {
            #اگر LMUL == 1
                tinyBLAS_RVV<vfloat32m1_t, vbfloat16mf2_t, ggml_bf16_t, ggml_bf16_t, اعشاری> tb{ params,
                    k, (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (اعشاری *)C, ldc};
            #وگرنہ_اگر LMUL == 2
                tinyBLAS_RVV<vfloat32m2_t, vbfloat16m1_t, ggml_bf16_t, ggml_bf16_t, اعشاری> tb{ params,
                    k, (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (اعشاری *)C, ldc};
            #ورنہ // LMUL = 4
                tinyBLAS_RVV<vfloat32m4_t, vbfloat16m2_t, ggml_bf16_t, ggml_bf16_t, اعشاری> tb{ params,
                    k, (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (اعشاری *)C, ldc};
            #اگر_ختم
                واپس tb.matmul(m, n);
        }
#اگر_ختم
        واپس جھوٹ;
    }

    حالت GGML_TYPE_F16: {
#اگر defined(__AVX512F__)
        اگر (Btype == GGML_TYPE_F16) {
            tinyBLAS<16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, اعشاری> tb{ params, k,
                (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (اعشاری *)C, ldc};
            واپس tb.matmul(m, n);
        }
#وگرنہ_اگر (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        اگر (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, اعشاری> tb{ params, k,
                (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (اعشاری *)C, ldc};
            واپس tb.matmul(m, n);
        }
#وگرنہ_اگر defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
        اگر (n < 8)
            واپس جھوٹ;
        اگر (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, اعشاری> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (اعشاری *)C, ldc};
            واپس tb.matmul(m, n);
        }
#وگرنہ_اگر defined(__ARM_NEON) && !defined(_MSC_VER)
        اگر (Btype == GGML_TYPE_F32) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, اعشاری, اعشاری> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت اعشاری *)B, ldb,
                (اعشاری *)C, ldc};
            واپس tb.matmul(m, n);
        }
#وگرنہ_اگر defined(__VXE__) || defined(__VXE2__)
        اگر (n < 4)
            واپس جھوٹ;
        اگر (Btype == GGML_TYPE_F16) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, اعشاری> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (اعشاری *)C, ldc};
            واپس tb.matmul(m, n);
        }
#وگرنہ_اگر defined(__riscv_zvfh)
        اگر (Btype == GGML_TYPE_F16) {
        #اگر LMUL == 1
            tinyBLAS_RVV<vfloat32m1_t, vfloat16mf2_t, ggml_fp16_t, ggml_fp16_t, اعشاری> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (اعشاری *)C, ldc};
        #وگرنہ_اگر LMUL == 2
            tinyBLAS_RVV<vfloat32m2_t, vfloat16m1_t, ggml_fp16_t, ggml_fp16_t, اعشاری> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (اعشاری *)C, ldc};
        #ورنہ // LMUL = 4
            tinyBLAS_RVV<vfloat32m4_t, vfloat16m2_t, ggml_fp16_t, ggml_fp16_t, اعشاری> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (اعشاری *)C, ldc};
        #اگر_ختم
            واپس tb.matmul(m, n);
        }
#وگرنہ_اگر defined(__MMA__)
        اگر (k % 8) {
            واپس جھوٹ;
        }

        اگر (Btype == GGML_TYPE_F16) {
            tinyBLAS_HP16_PPC<ggml_fp16_t, ggml_fp16_t, اعشاری> tb{ k,
                (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (اعشاری *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            واپس سچ;
        }
#اگر_ختم
        واپس جھوٹ;
    }

    حالت GGML_TYPE_Q8_0: {
        اگر (Btype != GGML_TYPE_Q8_0)
           واپس جھوٹ;
#اگر defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<بلاک_q8_0, بلاک_q8_0, اعشاری> tb{
            k, (ثابت بلاک_q8_0 *)A, lda,
            (ثابت بلاک_q8_0 *)B, ldb,
            (اعشاری *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        واپس سچ;
#وگرنہ_اگر defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<بلاک_q8_0> tb{
            k, (ثابت بلاک_q8_0 *)A, lda,
            (ثابت بلاک_q8_0 *)B, ldb,
            (اعشاری *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        واپس سچ;
#وگرنہ_اگر defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        اگر (n < 8 && n != 4)
           واپس جھوٹ;
        اگر (m < 8 && m != 4)
           واپس جھوٹ;
        tinyBLAS_Q0_PPC<بلاک_q8_0> tb{
            k, (ثابت بلاک_q8_0 *)A, lda,
            (ثابت بلاک_q8_0 *)B, ldb,
            (اعشاری *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        واپس سچ;
#ورنہ
        واپس جھوٹ;
#اگر_ختم
    }

    حالت GGML_TYPE_Q4_0: {
        اگر (Btype != GGML_TYPE_Q8_0)
            واپس جھوٹ;
#اگر defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<بلاک_q4_0, بلاک_q8_0, اعشاری> tb{
            k, (ثابت بلاک_q4_0 *)A, lda,
            (ثابت بلاک_q8_0 *)B, ldb,
            (اعشاری *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        واپس سچ;
#وگرنہ_اگر defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<بلاک_q4_0> tb{
            k, (ثابت بلاک_q4_0 *)A, lda,
            (ثابت بلاک_q8_0 *)B, ldb,
            (اعشاری *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        واپس سچ;
#وگرنہ_اگر defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        اگر (n < 8 && n != 4)
           واپس جھوٹ;
        اگر (m < 8 && m != 4)
           واپس جھوٹ;
        tinyBLAS_Q0_PPC<بلاک_q4_0> tb{
            k, (ثابت بلاک_q4_0 *)A, lda,
            (ثابت بلاک_q8_0 *)B, ldb,
            (اعشاری *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        واپس سچ;
#ورنہ
        واپس جھوٹ;
#اگر_ختم
    }

    حالت GGML_TYPE_Q5_0: {
        اگر (Btype != GGML_TYPE_Q8_0)
            واپس جھوٹ;
#اگر defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<بلاک_q5_0, بلاک_q8_0, اعشاری> tb{
            k, (ثابت بلاک_q5_0 *)A, lda,
            (ثابت بلاک_q8_0 *)B, ldb,
            (اعشاری *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        واپس سچ;
#ورنہ
        واپس جھوٹ;
#اگر_ختم
    }

    حالت GGML_TYPE_IQ4_NL: {
        اگر (Btype != GGML_TYPE_Q8_0)
            واپس جھوٹ;
#اگر defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<بلاک_iq4_nl, بلاک_q8_0, اعشاری> tb{
            k, (ثابت بلاک_iq4_nl *)A, lda,
            (ثابت بلاک_q8_0 *)B, ldb,
            (اعشاری *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        واپس سچ;
#ورنہ
        واپس جھوٹ;
#اگر_ختم
    }

    طےشدہ:
        واپس جھوٹ;
    }

    (خالی)params;
    (خالی)m;
    (خالی)n;
    (خالی)k;
    (خالی)A;
    (خالی)lda;
    (خالی)B;
    (خالی)ldb;
    (خالی)C;
    (خالی)ldc;
    (خالی)Atype;
    (خالی)Btype;
    (خالی)Ctype;
}
