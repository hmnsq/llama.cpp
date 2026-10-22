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

#إذا defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#ضمن "sgemm.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "ggml-quants.h"
#ضمن "simd-mappings.h"

#ضمن <array>
#ضمن <type_traits>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#وإلا
#define NOINLINE __attribute__((__noinline__))
#endif

#إذا defined(__ARM_NEON) || defined(__AVX512F__) || defined(__VXE__) || defined(__VXE2__)
#define VECTOR_REGISTERS 32
#وإلا
#define VECTOR_REGISTERS 16
#endif

#إذا defined(__riscv_v_intrinsic)
#define LMUL 4
#endif

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

نطاق {

مضمن عائم unhalf(ggml_fp16_t d) {
    أرجع GGML_CPU_FP16_TO_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#إذا defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
مضمن __m128 add(__m128 x, __m128 y) { أرجع _mm_add_ps(x, y); }
مضمن __m128 sub(__m128 x, __m128 y) { أرجع _mm_sub_ps(x, y); }
مضمن __m128 mul(__m128 x, __m128 y) { أرجع _mm_mul_ps(x, y); }
#endif  // __SSE__

#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
مضمن __m256 add(__m256 x, __m256 y) { أرجع _mm256_add_ps(x, y); }
مضمن __m256 sub(__m256 x, __m256 y) { أرجع _mm256_sub_ps(x, y); }
مضمن __m256 mul(__m256 x, __m256 y) { أرجع _mm256_mul_ps(x, y); }
#endif // __AVX__

#إذا defined(__AVX512F__)
مضمن __m512 add(__m512 x, __m512 y) { أرجع _mm512_add_ps(x, y); }
مضمن __m512 sub(__m512 x, __m512 y) { أرجع _mm512_sub_ps(x, y); }
مضمن __m512 mul(__m512 x, __m512 y) { أرجع _mm512_mul_ps(x, y); }
#endif // __AVX512F__

#إذا defined(__ARM_NEON)
مضمن float32x4_t add(float32x4_t x, float32x4_t y) { أرجع vaddq_f32(x, y); }
مضمن float32x4_t sub(float32x4_t x, float32x4_t y) { أرجع vsubq_f32(x, y); }
مضمن float32x4_t mul(float32x4_t x, float32x4_t y) { أرجع vmulq_f32(x, y); }
#endif // __ARM_NEON

#إذا defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
مضمن float16x8_t add(float16x8_t x, float16x8_t y) { أرجع vaddq_f16(x, y); }
مضمن float16x8_t sub(float16x8_t x, float16x8_t y) { أرجع vsubq_f16(x, y); }
مضمن float16x8_t mul(float16x8_t x, float16x8_t y) { أرجع vmulq_f16(x, y); }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#إذا defined(__VXE__) || defined(__VXE2__)
مضمن float32x4_t add(float32x4_t x, float32x4_t y) { أرجع vec_add(x, y); }
مضمن float32x4_t sub(float32x4_t x, float32x4_t y) { أرجع vec_sub(x, y); }
مضمن float32x4_t mul(float32x4_t x, float32x4_t y) { أرجع vec_mul(x, y); }
#endif

#إذا defined(__MMA__)
تعريفنوع vector unsigned حرف vec_t;
تعريفنوع __vector_quad acc_t;
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
template <typename T, typename U>
مضمن U madd(T a, T b, U c) {
    أرجع add(mul(a, b), c);
}

#إذا defined(__FMA__)
#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
مضمن __m256 madd(__m256 a, __m256 b, __m256 c) {
    أرجع _mm256_fmadd_ps(a, b, c);
}
#endif
#إذا defined(__AVX512F__)
template <>
مضمن __m512 madd(__m512 a, __m512 b, __m512 c) {
    أرجع _mm512_fmadd_ps(a, b, c);
}
#endif
#إذا defined(__AVX512BF16__)
template <>
مضمن __m512 madd(__m512bh a, __m512bh b, __m512 c) {
    أرجع _mm512_dpbf16_ps(c, a, b);
}
template <>
مضمن __m256 madd(__m256bh a, __m256bh b, __m256 c) {
    أرجع _mm256_dpbf16_ps(c, a, b);
}
#endif
#endif

#إذا defined(__ARM_FEATURE_FMA)
template <>
مضمن float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    أرجع vfmaq_f32(c, b, a);
}
#إذا defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
template <>
مضمن float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    أرجع vfmaq_f16(c, b, a);
}
#endif
#endif

#إذا defined(__VXE__) || defined(__VXE2__)
template <>
مضمن float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    أرجع vec_madd(a, b, c);
}
#endif

#إذا defined(__riscv_v_intrinsic)
template <> مضمن vfloat32m1_t madd(vfloat32m1_t a, vfloat32m1_t b, vfloat32m1_t c) {
    أرجع __riscv_vfmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
template <> مضمن vfloat32m2_t madd(vfloat32m2_t a, vfloat32m2_t b, vfloat32m2_t c) {
    أرجع __riscv_vfmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
template <> مضمن vfloat32m4_t madd(vfloat32m4_t a, vfloat32m4_t b, vfloat32m4_t c) {
    أرجع __riscv_vfmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
template <> مضمن vfloat32m8_t madd(vfloat32m8_t a, vfloat32m8_t b, vfloat32m8_t c) {
    أرجع __riscv_vfmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#endif

#إذا defined(__riscv_zvfh)
template <> مضمن vfloat32m1_t madd(vfloat16mf2_t a, vfloat16mf2_t b, vfloat32m1_t c) {
    أرجع __riscv_vfwmacc_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
template <> مضمن vfloat32m2_t madd(vfloat16m1_t a, vfloat16m1_t b, vfloat32m2_t c) {
    أرجع __riscv_vfwmacc_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
template <> مضمن vfloat32m4_t madd(vfloat16m2_t a, vfloat16m2_t b, vfloat32m4_t c) {
    أرجع __riscv_vfwmacc_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
template <> مضمن vfloat32m8_t madd(vfloat16m4_t a, vfloat16m4_t b, vfloat32m8_t c) {
    أرجع __riscv_vfwmacc_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#endif

#إذا defined(__riscv_zvfbfwma)
template <> مضمن vfloat32m1_t madd(vbfloat16mf2_t a, vbfloat16mf2_t b, vfloat32m1_t c) {
    أرجع __riscv_vfwmaccbf16_vv_f32m1(c, a, b, __riscv_vsetvlmax_e32m1());
}
template <> مضمن vfloat32m2_t madd(vbfloat16m1_t a, vbfloat16m1_t b, vfloat32m2_t c) {
    أرجع __riscv_vfwmaccbf16_vv_f32m2(c, a, b, __riscv_vsetvlmax_e32m2());
}
template <> مضمن vfloat32m4_t madd(vbfloat16m2_t a, vbfloat16m2_t b, vfloat32m4_t c) {
    أرجع __riscv_vfwmaccbf16_vv_f32m4(c, a, b, __riscv_vsetvlmax_e32m4());
}
template <> مضمن vfloat32m8_t madd(vbfloat16m4_t a, vbfloat16m4_t b, vfloat32m8_t c) {
    أرجع __riscv_vfwmaccbf16_vv_f32m8(c, a, b, __riscv_vsetvlmax_e32m8());
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#إذا defined(__ARM_NEON) || defined(_M_ARM64) || defined(_M_ARM64EC)
مضمن عائم hsum(float32x4_t x) {
    أرجع vaddvq_f32(x);
}
#endif // __ARM_NEON

#إذا defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
مضمن عائم hsum(float16x8_t x) {
    أرجع vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_رنده_f16(x)),
                                vcvt_f32_f16(vget_تيڠݢي_f16(x))));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#إذا defined(__VXE__) || defined(__VXE2__)
مضمن عائم hsum(float32x4_t x) {
    float32x4_t tmp = x + vec_reve(x);
    أرجع tmp[0] + tmp[1];
}
#endif

#إذا defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
مضمن عائم hsum(__m128 x) {
#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
#وإلا
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
#endif
    أرجع _mm_cvtss_f32(x);
}
#endif

#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
مضمن عائم hsum(__m256 x) {
    أرجع hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#endif // __AVX__

#إذا defined(__AVX512F__)
مضمن عائم hsum(__m512 x) {
    أرجع _mm512_reduce_add_ps(x);
}
#endif // __AVX512F__

#إذا defined(__riscv_v_intrinsic)
مضمن عائم hsum(vfloat32m1_t x) {
    أرجع __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m1_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m1()));
}
مضمن عائم hsum(vfloat32m2_t x) {
    أرجع __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m2()));
}
مضمن عائم hsum(vfloat32m4_t x) {
    أرجع __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m4_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m4()));
}
مضمن عائم hsum(vfloat32m8_t x) {
    أرجع __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m8_f32m1(x, __riscv_vfmv_v_f_f32m1(0, 1), __riscv_vsetvlmax_e32m8()));
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

template <typename T, typename U> T load(ثابت U *);

#إذا defined(__ARM_NEON)
template <> مضمن float32x4_t load(ثابت عائم *p) {
    أرجع vld1q_f32(p);
}
#إذا !defined(_MSC_VER)
// FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <> مضمن float16x8_t load(ثابت ggml_fp16_t *p) {
    أرجع vld1q_f16((ثابت float16_t *)p);
}
template <> مضمن float32x4_t load(ثابت ggml_fp16_t *p) {
    أرجع vcvt_f32_f16(vld1_f16((ثابت float16_t *)p));
}
#endif // _MSC_VER
#endif // __ARM_NEON

#إذا defined(__VXE__) || defined(__VXE2__)
template <> مضمن float32x4_t load(ثابت ggml_fp16_t * p) {
    عائم tmp[4];

    لكل (صحيح i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(p[i]);
    }

    أرجع vec_xl(0, (ثابت عائم *)(tmp));
}
template <> مضمن float32x4_t load(ثابت عائم * p) {
    أرجع vec_xl(0, p);
}
#endif

#إذا defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> مضمن __m128 load(ثابت عائم *p) {
    أرجع _mm_loadu_ps(p);
}
#endif  // __SSE__

#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> مضمن __m256 load(ثابت عائم *p) {
    أرجع _mm256_loadu_ps(p);
}
#endif // __AVX__

#إذا defined(__AVX2__) || defined(__AVX512F__)
template <> مضمن __m256 load(ثابت ggml_bf16_t *p) {
    أرجع _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((ثابت __m128i *)p)), 16));
}
#endif // __AVX2__

#إذا defined(__F16C__)
template <> مضمن __m256 load(ثابت ggml_fp16_t *p) {
    أرجع _mm256_cvtph_ps(_mm_loadu_si128((ثابت __m128i *)p));
}
#endif // __F16C__

#إذا defined(__AVX512F__)
template <> مضمن __m512 load(ثابت عائم *p) {
    أرجع _mm512_loadu_ps(p);
}
template <> مضمن __m512 load(ثابت ggml_fp16_t *p) {
    أرجع _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)p));
}
template <> مضمن __m512 load(ثابت ggml_bf16_t *p) {
    أرجع _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((ثابت __m256i *)p)), 16));
}
#endif // __AVX512F__

#إذا defined(__AVX512BF16__)
template <> مضمن __m512bh load(ثابت ggml_bf16_t *p) {
    أرجع (__m512bh)_mm512_loadu_ps((ثابت عائم *)p);
}
template <> مضمن __m256bh load(ثابت ggml_bf16_t *p) {
    أرجع (__m256bh)_mm256_loadu_ps((ثابت عائم *)p);
}
template <> مضمن __m512bh load(ثابت عائم *p) {
    أرجع _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
template <> مضمن __m256bh load(ثابت عائم *p) {
    أرجع _mm512_cvtneps_pbh(_mm512_loadu_ps(p));
}
#endif

#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <typename T, typename U> T load_partial(ثابت U *, صحيح);
template <typename T> T load_partial_u16(ثابت فراغ *, صحيح);

template <> مضمن __m128i load_partial_u16(ثابت فراغ *p, صحيح n) {
#إذا defined(__AVX512BW__) && defined(__AVX512VL__)
    أرجع _mm_maskz_loadu_epi16((1u << n) - 1, p);
#وإلا
    ثابت __m128i index = _mm_setr_epi32(0, 1, 2, 3);
    ثابت __m128i pairs = _mm_set1_epi32(n / 2);
    __m128i v = _mm_castps_si128(_mm_maskload_ps((ثابت عائم *)p, _mm_cmpgt_epi32(pairs, index)));
    إذا (n & 1) {
        uint16_t last;
        memcpy(&last, (ثابت حرف *)p + 2*(n - 1), حجم(last));
        v = _mm_or_si128(v, _mm_and_si128(_mm_cmpeq_epi32(pairs, index), _mm_set1_epi32(last)));
    }
    أرجع v;
#endif
}

template <> مضمن __m256 load_partial(ثابت عائم *p, صحيح n) {
    ثابت __m256 index = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    أرجع _mm256_maskload_ps(p, _mm256_castps_si256(_mm256_cmp_ps(index, _mm256_set1_ps(n), _CMP_LT_OQ)));
}

#إذا defined(__F16C__)
template <> مضمن __m256 load_partial(ثابت ggml_fp16_t *p, صحيح n) {
    أرجع _mm256_cvtph_ps(load_partial_u16<__m128i>(p, n));
}
#endif

#إذا defined(__AVX2__) || defined(__AVX512F__)
template <> مضمن __m256 load_partial(ثابت ggml_bf16_t *p, صحيح n) {
    أرجع _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(load_partial_u16<__m128i>(p, n)), 16));
}
#endif

#إذا defined(__AVX512F__)
template <> مضمن __m256i load_partial_u16(ثابت فراغ *p, صحيح n) {
#إذا defined(__AVX512BW__) && defined(__AVX512VL__)
    أرجع _mm256_maskz_loadu_epi16((1u << n) - 1, p);
#وإلا
    ثابت __m256i index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    ثابت __m256i pairs = _mm256_set1_epi32(n / 2);
    __m256i v = _mm256_maskload_epi32((ثابت صحيح *)p, _mm256_cmpgt_epi32(pairs, index));
    إذا (n & 1) {
        uint16_t last;
        memcpy(&last, (ثابت حرف *)p + 2*(n - 1), حجم(last));
        v = _mm256_or_si256(v, _mm256_and_si256(_mm256_cmpeq_epi32(pairs, index), _mm256_set1_epi32(last)));
    }
    أرجع v;
#endif
}

template <> مضمن __m512 load_partial(ثابت عائم *p, صحيح n) {
    أرجع _mm512_maskz_loadu_ps((1u << n) - 1, p);
}

template <> مضمن __m512 load_partial(ثابت ggml_fp16_t *p, صحيح n) {
    أرجع _mm512_cvtph_ps(load_partial_u16<__m256i>(p, n));
}

template <> مضمن __m512 load_partial(ثابت ggml_bf16_t *p, صحيح n) {
    أرجع _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(load_partial_u16<__m256i>(p, n)), 16));
}
#endif

#إذا defined(__AVX512BF16__)
template <> مضمن __m512bh load_partial(ثابت ggml_bf16_t *p, صحيح n) {
    أرجع (__m512bh) _mm512_maskz_loadu_epi16((uint64_t(1) << n) - 1, p);
}
#endif
#endif

#إذا defined(__riscv_v_intrinsic)
template <> مضمن vfloat32m1_t load(ثابت عائم *p) {
    أرجع __riscv_vle32_v_f32m1(p, __riscv_vsetvlmax_e32m1());
}
template <> مضمن vfloat32m2_t load(ثابت عائم *p) {
    أرجع __riscv_vle32_v_f32m2(p, __riscv_vsetvlmax_e32m2());
}
template <> مضمن vfloat32m4_t load(ثابت عائم *p) {
    أرجع __riscv_vle32_v_f32m4(p, __riscv_vsetvlmax_e32m4());
}
template <> مضمن vfloat32m8_t load(ثابت عائم *p) {
    أرجع __riscv_vle32_v_f32m8(p, __riscv_vsetvlmax_e32m8());
}
#endif

#إذا defined(__riscv_zvfh)
template <> مضمن vfloat16mf2_t load(ثابت ggml_fp16_t *p) {
    أرجع __riscv_vle16_v_f16mf2(reinterpret_cast<ثابت _Float16 *>(p), __riscv_vsetvlmax_e16mf2());
}
template <> مضمن vfloat16m1_t load(ثابت ggml_fp16_t *p) {
    أرجع __riscv_vle16_v_f16m1(reinterpret_cast<ثابت _Float16 *>(p), __riscv_vsetvlmax_e16m1());
}
template <> مضمن vfloat16m2_t load(ثابت ggml_fp16_t *p) {
    أرجع __riscv_vle16_v_f16m2(reinterpret_cast<ثابت _Float16 *>(p), __riscv_vsetvlmax_e16m2());
}
template <> مضمن vfloat16m4_t load(ثابت ggml_fp16_t *p) {
    أرجع __riscv_vle16_v_f16m4(reinterpret_cast<ثابت _Float16 *>(p), __riscv_vsetvlmax_e16m4());
}
#endif

#إذا defined(__riscv_zvfbfwma)
template <> مضمن vbfloat16mf2_t load(ثابت ggml_bf16_t *p) {
    أرجع __riscv_vle16_v_bf16mf2(reinterpret_cast<ثابت __bf16*>(p), __riscv_vsetvlmax_e16mf2());
}
template <> مضمن vbfloat16m1_t load(ثابت ggml_bf16_t *p) {
    أرجع __riscv_vle16_v_bf16m1(reinterpret_cast<ثابت __bf16*>(p), __riscv_vsetvlmax_e16m1());
}
template <> مضمن vbfloat16m2_t load(ثابت ggml_bf16_t *p) {
    أرجع __riscv_vle16_v_bf16m2(reinterpret_cast<ثابت __bf16*>(p), __riscv_vsetvlmax_e16m2());
}
template <> مضمن vbfloat16m4_t load(ثابت ggml_bf16_t *p) {
    أرجع __riscv_vle16_v_bf16m4(reinterpret_cast<ثابت __bf16*>(p), __riscv_vsetvlmax_e16m4());
}
#endif

#إذا defined(__riscv_v_intrinsic)
template <typename T> T set_zero();

template <> مضمن vfloat32m1_t set_zero() {
    أرجع __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
}
template <> مضمن vfloat32m2_t set_zero() {
    أرجع __riscv_vfmv_v_f_f32m2(0, __riscv_vsetvlmax_e32m2());
}
template <> مضمن vfloat32m4_t set_zero() {
    أرجع __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
}
template <> مضمن vfloat32m8_t set_zero() {
    أرجع __riscv_vfmv_v_f_f32m8(0, __riscv_vsetvlmax_e32m8());
}
#endif

#إذا defined(__riscv_v_intrinsic)
template <typename T> size_t vlmax() {
    إذا constexpr (std::is_same_v<T, vfloat32m1_t>) { أرجع  __riscv_vsetvlmax_e32m1(); }
    وإلا إذا constexpr (std::is_same_v<T, vfloat32m2_t>) { أرجع  __riscv_vsetvlmax_e32m2(); }
    وإلا إذا constexpr (std::is_same_v<T, vfloat32m4_t>) { أرجع  __riscv_vsetvlmax_e32m4(); }
    وإلا إذا constexpr (std::is_same_v<T, vfloat32m8_t>) { أرجع  __riscv_vsetvlmax_e32m8(); }
    #إذا defined (__riscv_zvfh)
    وإلا إذا constexpr (std::is_same_v<T, vfloat16mf2_t>) { أرجع  __riscv_vsetvlmax_e16mf2(); }
    وإلا إذا constexpr (std::is_same_v<T, vfloat16m1_t>) { أرجع  __riscv_vsetvlmax_e16m1(); }
    وإلا إذا constexpr (std::is_same_v<T, vfloat16m2_t>) { أرجع  __riscv_vsetvlmax_e16m2(); }
    وإلا إذا constexpr (std::is_same_v<T, vfloat16m4_t>) { أرجع  __riscv_vsetvlmax_e16m4(); }
    #endif
    #إذا defined (__riscv_zvfbfwma)
    وإلا إذا constexpr (std::is_same_v<T, vbfloat16mf2_t>) { أرجع  __riscv_vsetvlmax_e16mf2(); }
    وإلا إذا constexpr (std::is_same_v<T, vbfloat16m1_t>) { أرجع  __riscv_vsetvlmax_e16m1(); }
    وإلا إذا constexpr (std::is_same_v<T, vbfloat16m2_t>) { أرجع  __riscv_vsetvlmax_e16m2(); }
    وإلا إذا constexpr (std::is_same_v<T, vbfloat16m4_t>) { أرجع  __riscv_vsetvlmax_e16m4(); }
    #endif
    أرجع 0;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

template <صحيح M>
ساكن مضمن int64_t BLOCK_SIZE(size_t m) {
    ثابت int64_t NB_BLOC_M = (m + M - 1) / M;
    أرجع (m % NB_BLOC_M == 0) ? m / NB_BLOC_M : (m / NB_BLOC_M) + 1;
}

ساكن constexpr مضمن int64_t BLOC_POS(int64_t ib, int64_t ibN, int64_t bloc_size) {
    أرجع ib < ibN ? ib * bloc_size : ibN * bloc_size + (ib - ibN) * (bloc_size - 1);
}

template <صحيح KN, typename D, typename V, typename TA, typename TB, typename TC>
صنف tinyBLAS {
  عام:
    tinyBLAS(ثابت ggml_compute_params * params, int64_t k,
             ثابت TA *A, int64_t lda,
             ثابت TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    منطقي matmul(int64_t m, int64_t n) {
#إذا !defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
        إذا (k % KN != 0)
            أرجع خطأ;
#endif
        // compute RM for only need tile with size RM&RM-1
#إذا VECTOR_REGISTERS == 32
        إذا (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 4>(m, n, SIZE_N, 12);
            أرجع true;
        }
        إذا (m % 8 == 0 ) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 2>(m, n, SIZE_N, 12);
            أرجع true;
        }
        إذا (m % 4 == 0) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 1>(m, n, SIZE_N, 12);
            أرجع true;
        }
#وإلا  // VECTOR_REGISTERS == 16
        إذا (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 4>(m, n, SIZE_N, 24);
            أرجع true;
        }
        إذا (m % 8 == 0 ) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 2>(m, n, SIZE_N, 24);
            أرجع true;
        }
        إذا (m % 4 == 0) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 1>(m, n, SIZE_N, 24);
            أرجع true;
        }
#endif
        أرجع خطأ;
    }

  خاص:
    template <صحيح RM, صحيح RN, صحيح BM>
    مضمن فراغ mnpack(int64_t m, int64_t n, int64_t SIZE_N, int64_t BN) {
        إذا (SIZE_N == RN) {
            أرجع gemm<RM, RN, BM>(m, n, BN);
        }
        إذا constexpr (RN > 1) {
            أرجع mnpack<RM, RN-1, BM>(m, n, SIZE_N, BN);
        } وإلا {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (صحيح)SIZE_N);
            GGML_ASSERT(خطأ); // we have miss something.
        }
    }

    template <صحيح RM, صحيح RN>
    مضمن فراغ gemm_bloc(int64_t ii, int64_t jj) {
        D Cv[RN][RM] = {};
        لكل (int64_t l = 0; l + KN <= k; l += KN) {
            // help compiler for op order.
            إذا constexpr (RM <= RN) {
                V Av[RM];
                لكل (int64_t i = 0; i < RM; ++i) {
                    Av[i] = load<V>(A + lda * (ii + i) + l);
                }
                لكل (int64_t j = 0; j < RN; ++j) {
                    V Bv = load<V>(B + ldb * (jj + j) + l);
                    لكل (int64_t i = 0; i < RM; ++i) {
                        Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                    }
                }
            } وإلا {
                V Bv[RN];
                لكل (int64_t j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(B + ldb * (jj + j) + l);
                }
                لكل (int64_t i = 0; i < RM; ++i) {
                    V Av = load<V>(A + lda * (ii + i) + l);
                    لكل (int64_t j = 0; j < RN; ++j) {
                        Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
                    }
                }
            }
        }
#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        ثابت int64_t rem = k % KN;
        إذا (rem != 0) {
            V Av[RM];
            لكل (int64_t i = 0; i < RM; ++i) {
                Av[i] = load_partial<V>(A + lda * (ii + i) + k - rem, rem);
            }
            لكل (int64_t j = 0; j < RN; ++j) {
                V Bv = load_partial<V>(B + ldb * (jj + j) + k - rem, rem);
                لكل (int64_t i = 0; i < RM; ++i) {
                    Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                }
            }
        }
#endif
        لكل (int64_t j = 0; j < RN; ++j)
            لكل (int64_t i = 0; i < RM; ++i)
                C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
    }

    template <صحيح RM, صحيح RN, صحيح BM>
    NOINLINE فراغ gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        ثابت int64_t ytiles = m / (RM * BM);
        ثابت int64_t xtiles = (n + RN -1) / RN;
        ثابت int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        ثابت int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        ثابت int64_t SIZE_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        ثابت int64_t jj_BN = (NB_BN - (NB_BN * SIZE_BN - xtiles));
        ثابت int64_t nb_job = ytiles * NB_BN;

        إذا (params->ith == 0) {
            GGML_ASSERT( jj_BN * SIZE_BN + (NB_BN - jj_BN) * (SIZE_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t job = params->ith;
        طالما (job < nb_job) {
            ثابت int64_t ii = (job % ytiles) * RM * BM;
            ثابت int64_t jb =  job / ytiles;
            ثابت int64_t jr0 = BLOC_POS(jb  , jj_BN, SIZE_BN);
            ثابت int64_t jrN = BLOC_POS(jb+1, jj_BN, SIZE_BN);

            ثابت int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            ثابت int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            ثابت int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            لكل (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                لكل (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                إذا constexpr (RN > 1) {
                    لكل (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            job = ggml_threadpool_chunk_add(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        أرجع;
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

#إذا defined(__riscv_v_intrinsic)
template <typename D, typename V, typename TA, typename TB, typename TC>
صنف tinyBLAS_RVV {
  عام:
    tinyBLAS_RVV(ثابت ggml_compute_params * params, int64_t k,
             ثابت TA *A, int64_t lda,
             ثابت TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    منطقي matmul(int64_t m, int64_t n) {
        إذا (k % vlmax<V>() != 0) {
            أرجع خطأ;
        }

#إذا LMUL == 1
        إذا (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 4>(m, n, SIZE_N, 12);
            أرجع true;
        }
        إذا (m % 8 == 0 ) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 2>(m, n, SIZE_N, 12);
            أرجع true;
        }
        إذا (m % 4 == 0) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 1>(m, n, SIZE_N, 12);
            أرجع true;
        }
#وإلاإذا LMUL == 2
        إذا (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 4>(m, n, SIZE_N, 24);
            أرجع true;
        }
        إذا (m % 8 == 0 ) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 2>(m, n, SIZE_N, 24);
            أرجع true;
        }
        إذا (m % 4 == 0) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 1>(m, n, SIZE_N, 24);
            أرجع true;
        }
#وإلا // LMUL = 4
        إذا (m % 16 == 0 && (m/16 >= params->nth)) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<2>(n);
            mnpack<2, 2, 8>(m, n, SIZE_N, 36);
            أرجع true;
        }
        إذا (m % 8 == 0 ) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<2>(n);
            mnpack<2, 2, 4>(m, n, SIZE_N, 36);
            أرجع true;
        }
        إذا (m % 4 == 0) {
            ثابت int64_t SIZE_N = BLOCK_SIZE<2>(n);
            mnpack<2, 2, 2>(m, n, SIZE_N, 36);
            أرجع true;
        }
#endif
        أرجع خطأ;
    }

  خاص:
    template<صحيح RM, صحيح RN, صحيح BM>
    مضمن فراغ mnpack(int64_t m, int64_t n, int64_t SIZE_N, int64_t BN) {
        إذا (SIZE_N == RN) {
            أرجع gemm<RM, RN, BM>(m, n, BN);
        }
        إذا constexpr (RN > 1) {
            أرجع mnpack<RM, RN-1, BM>(m, n, SIZE_N, BN);
        } وإلا {
            GGML_LOG_ERROR("mnpack<%d, %d> block size not supported\n", RM, (صحيح)SIZE_N);
            GGML_ASSERT(خطأ); // we have miss something.
        }
    }

    مضمن فراغ gemm_bloc_4x6(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = set_zero<D>();
        D Cv01 = set_zero<D>();
        D Cv02 = set_zero<D>();
        D Cv03 = set_zero<D>();
        D Cv10 = set_zero<D>();
        D Cv11 = set_zero<D>();
        D Cv12 = set_zero<D>();
        D Cv13 = set_zero<D>();
        D Cv20 = set_zero<D>();
        D Cv21 = set_zero<D>();
        D Cv22 = set_zero<D>();
        D Cv23 = set_zero<D>();
        D Cv30 = set_zero<D>();
        D Cv31 = set_zero<D>();
        D Cv32 = set_zero<D>();
        D Cv33 = set_zero<D>();
        D Cv40 = set_zero<D>();
        D Cv41 = set_zero<D>();
        D Cv42 = set_zero<D>();
        D Cv43 = set_zero<D>();
        D Cv50 = set_zero<D>();
        D Cv51 = set_zero<D>();
        D Cv52 = set_zero<D>();
        D Cv53 = set_zero<D>();

        لكل (int64_t l = 0; l < k; l += vl) {
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

    مضمن فراغ gemm_bloc_4x5(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = set_zero<D>();
        D Cv01 = set_zero<D>();
        D Cv02 = set_zero<D>();
        D Cv03 = set_zero<D>();
        D Cv10 = set_zero<D>();
        D Cv11 = set_zero<D>();
        D Cv12 = set_zero<D>();
        D Cv13 = set_zero<D>();
        D Cv20 = set_zero<D>();
        D Cv21 = set_zero<D>();
        D Cv22 = set_zero<D>();
        D Cv23 = set_zero<D>();
        D Cv30 = set_zero<D>();
        D Cv31 = set_zero<D>();
        D Cv32 = set_zero<D>();
        D Cv33 = set_zero<D>();
        D Cv40 = set_zero<D>();
        D Cv41 = set_zero<D>();
        D Cv42 = set_zero<D>();
        D Cv43 = set_zero<D>();

        لكل (int64_t l = 0; l < k; l += vl) {
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

    مضمن فراغ gemm_bloc_4x4(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = set_zero<D>();
        D Cv01 = set_zero<D>();
        D Cv02 = set_zero<D>();
        D Cv03 = set_zero<D>();
        D Cv10 = set_zero<D>();
        D Cv11 = set_zero<D>();
        D Cv12 = set_zero<D>();
        D Cv13 = set_zero<D>();
        D Cv20 = set_zero<D>();
        D Cv21 = set_zero<D>();
        D Cv22 = set_zero<D>();
        D Cv23 = set_zero<D>();
        D Cv30 = set_zero<D>();
        D Cv31 = set_zero<D>();
        D Cv32 = set_zero<D>();
        D Cv33 = set_zero<D>();

        لكل (int64_t l = 0; l < k; l += vl) {
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

    مضمن فراغ gemm_bloc_4x3(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = set_zero<D>();
        D Cv01 = set_zero<D>();
        D Cv02 = set_zero<D>();
        D Cv03 = set_zero<D>();
        D Cv10 = set_zero<D>();
        D Cv11 = set_zero<D>();
        D Cv12 = set_zero<D>();
        D Cv13 = set_zero<D>();
        D Cv20 = set_zero<D>();
        D Cv21 = set_zero<D>();
        D Cv22 = set_zero<D>();
        D Cv23 = set_zero<D>();

        لكل (int64_t l = 0; l < k; l += vl) {
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

    مضمن فراغ gemm_bloc_4x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = set_zero<D>();
        D Cv01 = set_zero<D>();
        D Cv02 = set_zero<D>();
        D Cv03 = set_zero<D>();
        D Cv10 = set_zero<D>();
        D Cv11 = set_zero<D>();
        D Cv12 = set_zero<D>();
        D Cv13 = set_zero<D>();

        لكل (int64_t l = 0; l < k; l += vl) {
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

    مضمن فراغ gemm_bloc_4x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = set_zero<D>();
        D Cv01 = set_zero<D>();
        D Cv02 = set_zero<D>();
        D Cv03 = set_zero<D>();

        لكل (int64_t l = 0; l < k; l += vl) {
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

    مضمن فراغ gemm_bloc_2x2(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = set_zero<D>();
        D Cv01 = set_zero<D>();
        D Cv10 = set_zero<D>();
        D Cv11 = set_zero<D>();

        لكل (int64_t l = 0; l < k; l += vl) {
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

    مضمن فراغ gemm_bloc_2x1(int64_t ii, int64_t jj) {
        size_t vl = vlmax<V>();
        D Cv00 = set_zero<D>();
        D Cv01 = set_zero<D>();

        لكل (int64_t l = 0; l < k; l += vl) {
            V Av0 = load<V>(A + lda * (ii + 0) + l);
            V Av1 = load<V>(A + lda * (ii + 1) + l);

            V Bv0 = load<V>(B + ldb * (jj + 0) + l);
            Cv00 = madd(Av0, Bv0, Cv00);
            Cv01 = madd(Av1, Bv0, Cv01);
        }

        C[ldc * (jj + 0) + (ii + 0)] = hsum(Cv00);
        C[ldc * (jj + 0) + (ii + 1)] = hsum(Cv01);
    }

    template <صحيح RM, صحيح RN>
    مضمن فراغ gemm_bloc(int64_t ii, int64_t jj) {
        إذا constexpr (RM == 4) {
            إذا constexpr (RN == 6) { أرجع gemm_bloc_4x6(ii, jj); }
            إذا constexpr (RN == 5) { أرجع gemm_bloc_4x5(ii, jj); }
            إذا constexpr (RN == 4) { أرجع gemm_bloc_4x4(ii, jj); }
            إذا constexpr (RN == 3) { أرجع gemm_bloc_4x3(ii, jj); }
            إذا constexpr (RN == 2) { أرجع gemm_bloc_4x2(ii, jj); }
            إذا constexpr (RN == 1) { أرجع gemm_bloc_4x1(ii, jj); }
        } وإلا إذا constexpr (RM == 2) {
            إذا constexpr (RN == 2) { أرجع gemm_bloc_2x2(ii, jj); }
            إذا constexpr (RN == 1) { أرجع gemm_bloc_2x1(ii, jj); }
        }
    }

    template <صحيح RM, صحيح RN, صحيح BM>
    NOINLINE فراغ gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        ثابت int64_t ytiles = m / (RM * BM);
        ثابت int64_t xtiles = (n + RN -1) / RN;
        ثابت int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        ثابت int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        ثابت int64_t SIZE_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        ثابت int64_t jj_BN = (NB_BN - (NB_BN * SIZE_BN - xtiles));
        ثابت int64_t nb_job = ytiles * NB_BN;

        إذا (params->ith == 0) {
            GGML_ASSERT( jj_BN * SIZE_BN + (NB_BN - jj_BN) * (SIZE_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t job = params->ith;
        طالما (job < nb_job) {
            ثابت int64_t ii = (job % ytiles) * RM * BM;
            ثابت int64_t jb =  job / ytiles;
            ثابت int64_t jr0 = BLOC_POS(jb  , jj_BN, SIZE_BN);
            ثابت int64_t jrN = BLOC_POS(jb+1, jj_BN, SIZE_BN);

            ثابت int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            ثابت int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            ثابت int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            لكل (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                لكل (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                إذا constexpr (RN > 1) {
                    لكل (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            job = ggml_threadpool_chunk_add(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        أرجع;
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
#endif

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#إذا defined(__ARM_FEATURE_DOTPROD)
template <typename TA>
صنف tinyBLAS_Q0_ARM {
  عام:
    tinyBLAS_Q0_ARM(int64_t k,
                    ثابت TA *A, int64_t lda,
                    ثابت block_q8_0 *B, int64_t ldb,
                    عائم *C, int64_t ldc,
                    صحيح ith, صحيح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    فراغ matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  خاص:
    NOINLINE فراغ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        بدل ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3ll)) {
        حالة 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            اكسر;
        حالة 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            اكسر;
        حالة 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            اكسر;
        حالة 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            اكسر;
        حالة 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            اكسر;
        حالة 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            اكسر;
        حالة 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            اكسر;
        حالة 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            اكسر;
        حالة 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            اكسر;
        افتراضي:
            أرجع;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <صحيح RM, صحيح RN>
    NOINLINE فراغ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            float32x4_t Cv[RN][RM] = {};
            لكل (int64_t l = 0; l < k; ++l)
                لكل (int64_t j = 0; j < RN; ++j)
                    لكل (int64_t i = 0; i < RM; ++i)
                        Cv[j][i] = vmlaq_n_f32(Cv[j][i],
                                               vcvtq_f32_s32(vdotq_s32(
                                                   vdotq_s32(vdupq_n_s32(0),
                                                             load_lo(A + lda * (ii + i) + l),
                                                             load_lo(B + ldb * (jj + j) + l)),
                                                   load_hi(A + lda * (ii + i) + l),
                                                   load_hi(B + ldb * (jj + j) + l))),
                                               unhalf(A[lda * (ii + i) + l].d) *
                                               unhalf(B[ldb * (jj + j) + l].d));
            لكل (int64_t j = 0; j < RN; ++j)
                لكل (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    مضمن int8x16_t load_lo(ثابت block_q8_0 *b) {
        أرجع vld1q_s8(b->qs);
    }

    مضمن int8x16_t load_hi(ثابت block_q8_0 *b) {
        أرجع vld1q_s8(b->qs + 16);
    }

    مضمن int8x16_t load_lo(ثابت block_q4_0 *b) {
        أرجع vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs),
                                                     vdupq_n_u8(0x0f))),
                        vdupq_n_s8(0x8));
    }

    مضمن int8x16_t load_hi(ثابت block_q4_0 *b) {
        أرجع vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)),
                        vdupq_n_s8(0x8));
    }

    ثابت TA *ثابت A;
    ثابت block_q8_0 *ثابت B;
    عائم *ثابت C;
    ثابت int64_t k;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
    ثابت صحيح ith;
    ثابت صحيح nth;
};
#endif // __ARM_FEATURE_DOTPROD

#إذا defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
template <typename TA, typename TB, typename TC>
صنف tinyBLAS_Q0_AVX {
  عام:
    tinyBLAS_Q0_AVX(int64_t k,
                    ثابت TA *A, int64_t lda,
                    ثابت TB *B, int64_t ldb,
                    TC *C, int64_t ldc,
                    صحيح ith, صحيح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
        ثابت int8_t kvalues_iq4nl[16] = {
            -127, -104, -83, -65,
            -49,  -35,  -22, -10,
              1,   13,   25,  38,
             53,   69,   89, 113
        };

        iq4nlt = _mm_loadu_si128((ثابت __m128i *)kvalues_iq4nl);
    }

    فراغ matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  خاص:
    فراغ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        بدل ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
#إذا VECTOR_REGISTERS == 32
        حالة 0x44:
            mc = 4;
            nc = 4;
#إذا defined(__AVX2__) && defined(__F16C__)
            gemm4xN<4>(m0, m, n0, n);
#وإلا
            gemm<4, 4>(m0, m, n0, n);
#endif
            اكسر;
        حالة 0x43:
            mc = 4;
            nc = 3;
#إذا defined(__AVX2__) && defined(__F16C__)
            gemm4xN<3>(m0, m, n0, n);
#وإلا
            gemm<4, 3>(m0, m, n0, n);
#endif
            اكسر;
        حالة 0x34:
            mc = 3;
            nc = 4;
#إذا defined(__AVX2__) && defined(__F16C__)
            gemmMx4<3>(m0, m, n0, n);
#وإلا
            gemm<3, 4>(m0, m, n0, n);
#endif
            اكسر;
        حالة 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            اكسر;
        حالة 0x42:
            mc = 4;
            nc = 2;
#إذا defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#وإلا
            gemm<4, 2>(m0, m, n0, n);
#endif
            اكسر;
        حالة 0x24:
            mc = 2;
            nc = 4;
#إذا defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#وإلا
            gemm<2, 4>(m0, m, n0, n);
#endif
            اكسر;
#وإلا
        حالة 0x44:
        حالة 0x43:
        حالة 0x42:
            mc = 4;
            nc = 2;
#إذا defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#وإلا
            gemm<4, 2>(m0, m, n0, n);
#endif
            اكسر;
        حالة 0x34:
        حالة 0x24:
            mc = 2;
            nc = 4;
#إذا defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#وإلا
            gemm<2, 4>(m0, m, n0, n);
#endif
            اكسر;
        حالة 0x33:
#endif
        حالة 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            اكسر;
        حالة 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            اكسر;
        حالة 0x41:
            mc = 4;
            nc = 1;
#إذا defined(__AVX2__) && defined(__F16C__)
            gemm4xN<1>(m0, m, n0, n);
#وإلا
            gemm<4, 1>(m0, m, n0, n);
#endif
            اكسر;
        حالة 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            اكسر;
        حالة 0x14:
            mc = 1;
            nc = 4;
#إذا defined(__AVX2__) && defined(__F16C__)
            gemmMx4<1>(m0, m, n0, n);
#وإلا
            gemm<1, 4>(m0, m, n0, n);
#endif
            اكسر;
        حالة 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            اكسر;
        حالة 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            اكسر;
        حالة 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            اكسر;
        حالة 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            اكسر;
        حالة 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            اكسر;
        افتراضي:
            أرجع;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

#إذا defined(__AVX2__) && defined(__F16C__)
// Templated functions for gemm of dimensions 4xN
    template <صحيح RN>
    NOINLINE فراغ gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / 4;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * 4;
            int64_t jj = n0 + job % xtiles * RN;
            __m256 Cv[RN][4] = {};
            لكل (int64_t l = 0; l < k; ++l) {
                uint64_t a_delta = ((uint64_t)A[lda * (ii + 3) + l].d << 48) | ((uint64_t)A[lda * (ii + 2) + l].d << 32) | ((uint64_t)A[lda * (ii + 1) + l].d << 16) | (A[lda * (ii + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 da = _mm_cvtph_ps(_mm_set_epi64x(0, a_delta));
                __m256i avec0 = load(A + lda * (ii + 0) + l);
                __m256i avec1 = load(A + lda * (ii + 1) + l);
                __m256i avec2 = load(A + lda * (ii + 2) + l);
                __m256i avec3 = load(A + lda * (ii + 3) + l);
                لكل (int64_t j = 0; j < RN; ++j) {
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

            لكل (int64_t j = 0; j < RN; ++j)
                لكل (int64_t i = 0; i < 4; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    // Templated functions for gemm of dimensions Mx4
    template <صحيح RM>
    NOINLINE فراغ gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / 4;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * 4;
            __m256 Cv[4][RM] = {};
            لكل (int64_t l = 0; l < k; ++l) {
                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 3) + l].d << 48) | ((uint64_t)B[ldb * (jj + 2) + l].d << 32) | ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 db = _mm_cvtph_ps(_mm_set_epi64x(0, b_delta));
                __m256i bvec0 = load(B + ldb * (jj + 0) + l);
                __m256i bvec1 = load(B + ldb * (jj + 1) + l);
                __m256i bvec2 = load(B + ldb * (jj + 2) + l);
                __m256i bvec3 = load(B + ldb * (jj + 3) + l);
                لكل (int64_t i = 0; i < RM; ++i) {
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
            لكل (int64_t j = 0; j < 4; ++j)
                لكل (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }
#endif

    template <صحيح RM, صحيح RN>
    NOINLINE فراغ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            لكل (int64_t l = 0; l < k; ++l)
                لكل (int64_t j = 0; j < RN; ++j)
                    لكل (int64_t i = 0; i < RM; ++i) {
#إذا defined(__AVX2__)
                        __m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                              load(A + lda * (ii + i) + l)),
                                             _mm256_sign_epi8(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l)));
#وإلا
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
                        __m256 udTmp = _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
#endif
                        Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            لكل (int64_t j = 0; j < RN; ++j)
                لكل (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    مضمن __m256i load(ثابت block_q8_0 *b) {
        أرجع _mm256_loadu_si256((ثابت __m256i *)b->qs);
    }

    مضمن __m128i load0(ثابت block_q8_0 *b) {
        أرجع _mm_loadu_si128((ثابت __m128i *)b->qs);
    }

    مضمن __m128i load1(ثابت block_q8_0 *b) {
        أرجع _mm_loadu_si128(((ثابت __m128i *)b->qs) + 1);
    }

    مضمن __m256i load(ثابت block_q4_0 *b) {
        أرجع _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    مضمن __m128i load0(ثابت block_q4_0 *b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        أرجع _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), x), _mm_set1_epi8(8));
    }

    مضمن __m128i load1(ثابت block_q4_0 *b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        أرجع _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)), _mm_set1_epi8(8));
    }

    مضمن __m256i load(ثابت block_q5_0 *b) {
        أرجع _mm256_or_si256(denibble(b->qs), bittobyte(b->qh));
    }

    مضمن __m128i load0(ثابت block_q5_0* b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        uint32_t x32;
        memcpy(&x32, b->qh, حجم(uint32_t));
        __m128i qxl = _mm_and_si128(_mm_set1_epi8(15), x);
        __m128i bytesl = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      _mm_set_epi64x(0x0101010101010101, 0x0000000000000000))));
        bytesl = _mm_andnot_si128(bytesl, _mm_set1_epi8((حرف)0xF0));
        أرجع _mm_or_si128(qxl, bytesl);
    }

    مضمن __m128i load1(ثابت block_q5_0* b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        uint32_t x32;
        memcpy(&x32, b->qh, حجم(uint32_t));
        __m128i qxh = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
        __m128i bytesh = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      _mm_set_epi64x(0x0303030303030303, 0x0202020202020202))));
        bytesh = _mm_andnot_si128(bytesh, _mm_set1_epi8((حرف)0xF0));
        أرجع _mm_or_si128(qxh, bytesh);
    }

    مضمن __m256i load(ثابت block_iq4_nl *b) {
        أرجع MM256_SET_M128I(load1(b), load0(b));
    }

    مضمن __m128i load0(ثابت block_iq4_nl *b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        أرجع _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), x));
    }

    مضمن __m128i load1(ثابت block_iq4_nl *b) {
        ثابت __m128i x = _mm_loadu_si128((ثابت __m128i *)(b->qs));
        أرجع _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)));
    }

    مضمن __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#إذا defined(__AVX512VNNI__) && defined(__AVX512VL__)
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#وإلاإذا defined(__AVXVNNI__)
        res = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
#وإلا
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
        أرجع _mm256_cvtepi32_ps(res);
    }

    ساكن مضمن __m256i denibble(ثابت uint8_t *p) {
        __m128i x = _mm_loadu_si128((ثابت __m128i *)p);
        أرجع _mm256_and_si256(_mm256_set1_epi8(15),
                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                        _mm_srli_epi16(x, 4), 1));
    }

    ساكن مضمن __m256i bittobyte(ثابت uint8_t *p) {
        uint32_t x32;
        memcpy(&x32, p, حجم(uint32_t));
        __m256i bytes = _mm256_cmpeq_epi8(_mm256_set1_epi64x(-1),
                                          _mm256_or_si256(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                          _mm256_shuffle_epi8(_mm256_set1_epi32(x32),
                                                                              _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,
                                                                                                0x0101010101010101, 0x0000000000000000))));
        أرجع _mm256_andnot_si256(bytes, _mm256_set1_epi8((حرف)0xF0));
    }

    ثابت TA *ثابت A;
    ثابت TB *ثابت B;
    TC *ثابت C;
    ثابت int64_t k;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
    ثابت صحيح ith;
    ثابت صحيح nth;
    __m128i iq4nlt;
};
#endif // __AVX__

//PPC Implementation
#إذا defined(__MMA__)

template<typename T>
بنية mma_instr;

template<>
بنية mma_instr<ggml_bf16_t> {
    ساكن مضمن فراغ outer_product(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvbf16ger2pp(acc, a, b);
    }
};

template<>
بنية mma_instr<ggml_fp16_t> {
    ساكن مضمن فراغ outer_product(acc_t *acc, vec_t a, vec_t b) {
        __builtin_mma_xvf16ger2pp(acc, a, b);
    }
};

template <typename TA, typename TB, typename TC>
صنف tinyBLAS_HP16_PPC {
  عام:
    tinyBLAS_HP16_PPC(int64_t k,
                ثابت TA *A, int64_t lda,
                ثابت TB *B, int64_t ldb,
                TC *C, int64_t ldc,
                صحيح ith, صحيح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    فراغ matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #إذا defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #endif
        إذا (k < kc) {
            kc = k;
        }
        منطقي can_use_tiled = (m % mc == 0) && (n % nc == 0) && (k % kc == 0);
        إذا (can_use_tiled) {
            matmul_tiled(m, n, mc, nc, kc);
        } وإلا {
            mnpack(0, m, 0, n);
        }
    }

  خاص:
    __attribute__((always_inline))
    مضمن فراغ save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        لكل (صحيح I = 0; I < 4; I++) {
            لكل (صحيح J = 0; J < 4; J++) {
                *((عائم *)(C+ii+((jj+J)*ldc)+I)) = *((عائم *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    مضمن فراغ add_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        لكل (صحيح I = 0; I < 4; I++) {
            لكل (صحيح J = 0; J < 4; J++) {
                عائم * c_ptr = (عائم *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((عائم *)&vec_C[I]+J);
            }
        }
    }

    فراغ vector_permute_store(vec_t *c, صحيح اڠک_Vec, unsigned حرف *vecOffset) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        إذا (اڠک_Vec == 2) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[2], c[3], swiz1);
            s[0] = vec_perm(t[0], t[1], swiz3);
            s[1] = vec_perm(t[0], t[1], swiz4);
            vec_xst(s[0], 0, (vec_t*)vecOffset);
            vec_xst(s[1], 0, (vec_t*)(vecOffset + 16));
        } وإلا إذا (اڠک_Vec == 4) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[0], c[1], swiz2);
            t[2] = vec_perm(c[2], c[3], swiz1);
            t[3] = vec_perm(c[2], c[3], swiz2);
            s[0] = vec_perm(t[0], t[2], swiz3);
            s[1] = vec_perm(t[0], t[2], swiz4);
            s[2] = vec_perm(t[1], t[3], swiz3);
            s[3] = vec_perm(t[1], t[3], swiz4);
            لكل (صحيح i = 0; i < 4; ++i)
                vec_xst(s[i], 0, (vec_t*)(vecOffset + i * 16));
        } وإلا إذا (اڠک_Vec == 8) {
            لكل (صحيح i = 0; i < 4; i += 2) {
                t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
                t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
            }
            لكل (صحيح i = 4; i < 8; i += 2) {
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
            لكل (صحيح i = 0; i < 8; ++i)
                vec_xst(s[i], 0, (vec_t*)(vecOffset + i * 16));
        }
    }

    فراغ packNormal(ثابت TA* a, int64_t lda, صحيح rows, صحيح cols, unsigned حرف* vec) {
        int64_t i, j;
        TA *aoffset = عدم;
        unsigned حرف *vecOffset = عدم;
        TA * aoffsets[8];
        vector unsigned حرف c_arr[8];
        aoffset = const_cast<TA*>(a);
        vecOffset = vec;
        j = (rows >> 3);
        إذا (j > 0) {
            افعل {
                aoffsets[0] = aoffset;
                إذا (cols == 4) {
                    aoffsets[0] = aoffset;
                    لكل (صحيح it = 1; it < 4; ++it)
                        aoffsets[it] = aoffsets[it-1] + lda;
                    aoffset += 4 * lda;
                    لكل (صحيح i = 0; i < 4; ++i)
                        c_arr[i] = vec_xl(0, (vector unsigned حرف*)aoffsets[i]);
                    vector_permute_store(c_arr, 4, vecOffset);
                    لكل (صحيح i = 0; i<4; i++)
                        aoffsets[i] = aoffsets[i]+lda;
                    vecOffset +=64;
                }
                i = (cols >> 3);
                إذا (i > 0) {
                    لكل (صحيح it = 1; it < 8; ++it) {
                        aoffsets[it] = aoffsets[it-1] + lda;
                    }
                    aoffset += 8 * lda;

                    افعل {
                        لكل (صحيح it = 0; it < 8; ++it)
                            c_arr[it] = vec_xl(0, (vector unsigned حرف*)aoffsets[it]);
                        vector_permute_store(c_arr, 8, vecOffset);
                        لكل (صحيح it = 0; it < 8; ++it)
                            aoffsets[it] = aoffsets[it] + 8;
                        vecOffset += 128;
                        i--;
                    } طالما(i > 0);
                }
                j--;
            } طالما(j > 0);
        }
        إذا (rows & 4) {
            aoffsets[0] = aoffset;
            لكل (صحيح it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            إذا (cols == 4) {
                لكل (صحيح it = 0; it < 4; ++it)
                    c_arr[it] = vec_xl(0, (vector unsigned حرف*)aoffsets[it]);
                vector_permute_store(c_arr, 2, vecOffset);
                لكل (صحيح it = 0; it< 4; it++)
                    aoffsets[it] = aoffsets[it] + lda;
                vecOffset += 32;
            }
            i = (cols >> 3);
            إذا (i > 0) {
                افعل {
                    لكل (صحيح it = 0; it < 4; ++it)
                        c_arr[it] = vec_xl(0, (vector unsigned حرف*)aoffsets[it]);
                    vector_permute_store(c_arr, 4, vecOffset);
                    لكل (صحيح it = 0; it< 4; it++)
                        aoffsets[it] = aoffsets[it] + 8*lda;
                    vecOffset += 64;
                    i--;
                } طالما(i > 0);
            }
        }
        إذا (rows & 3) {
            aoffsets[0] = aoffset;
            لكل (صحيح it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            إذا (cols == 4) {
                بدل(rows) {
                    حالة 3: c_arr[2] = vec_xl(0, (vector unsigned حرف*)aoffsets[2]);
                    حالة 2: c_arr[1] = vec_xl(0, (vector unsigned حرف*)aoffsets[1]);
                    حالة 1: c_arr[0] = vec_xl(0, (vector unsigned حرف*)aoffsets[0]);
                        اكسر;
                }
                vector_permute_store(c_arr, 2, vecOffset);
                لكل (صحيح it = 0; it< 4; it++)
                     aoffsets[it] = aoffsets[it] + lda;
                vecOffset += 32;
            }
            i = (cols >> 3);
            إذا (i > 0) {
                افعل {
                    بدل(rows) {
                        حالة 3: c_arr[2] = vec_xl(0, (vector unsigned حرف*)aoffsets[2]);
                        حالة 2: c_arr[1] = vec_xl(0, (vector unsigned حرف*)aoffsets[1]);
                        حالة 1: c_arr[0] = vec_xl(0, (vector unsigned حرف*)aoffsets[0]);
                            اكسر;
                    }
                    vector_permute_store(c_arr, 4, vecOffset);
                    لكل (صحيح it = 0; it <4; it++)
                         aoffsets[it] = aoffsets[it] + 8* lda;
                    vecOffset += 64;
                    i--;
                } طالما(i > 0);
            }
        }
    }

    فراغ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        صحيح m_rem = MIN(m - m0, 8);
        صحيح n_rem = MIN(n - n0, 8);

        إذا (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } وإلا إذا (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4,8>(m0, m, n0, n);
        } وإلا إذا (m_rem >=8 && n_rem >=4){
                mc = 8;
                nc = 4;
                gemm<8,4>(m0, m, n0, n);
        } وإلا إذا ((m_rem < 4) && (n_rem >= 8)) {
            nc = 8;
            بدل(m_rem) {
                حالة 1:
                    mc = 1;
                    gemm_Mx8<1>(m0, m, n0, n);
                    اكسر;
                حالة 2:
                    mc = 2;
                    gemm_Mx8<2>(m0, m, n0, n);
                    اكسر;
                حالة 3:
                    mc = 3;
                    gemm_Mx8<3>(m0, m, n0, n);
                    اكسر;
                افتراضي:
                    أرجع;
            }
        } وإلا إذا (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small<4, 4>(m0, m, n0, n);
        } وإلا إذا ((m_rem > 4) && (n_rem < 4)) {
            mc = 4;
            بدل(n_rem) {
                حالة 1:
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    اكسر;
                حالة 2:
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    اكسر;
                حالة 3:
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    اكسر;

                افتراضي:
                    أرجع;
            }
        } وإلا {
            بدل((m_rem << 4) | n_rem) {
                حالة 0x43:
                    mc = 4;
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    اكسر;
                حالة 0x42:
                    mc = 4;
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    اكسر;
                حالة 0x41:
                    mc = 4;
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    اكسر;
                حالة 0x34:
                    mc = 3;
                    nc = 4;
                    gemm_small<3, 4>(m0, m, n0, n);
                    اكسر;
                حالة 0x33:
                    mc = 3;
                    nc = 3;
                    gemm_small<3, 3>(m0, m, n0, n);
                    اكسر;
                حالة 0x32:
                    mc = 3;
                    nc = 2;
                    gemm_small<3, 2>(m0, m, n0, n);
                    اكسر;
                حالة 0x31:
                    mc = 3;
                    nc = 1;
                    gemm_small<3, 1>(m0, m, n0, n);
                    اكسر;
                حالة 0x24:
                    mc = 2;
                    nc = 4;
                    gemm_small<2,4>(m0, m, n0, n);
                    اكسر;
                حالة 0x23:
                    mc = 2;
                    nc = 3;
                    gemm_small<2, 3>(m0, m, n0, n);
                    اكسر;
                حالة 0x22:
                    mc = 2;
                    nc = 2;
                    gemm_small<2, 2>(m0, m, n0, n);
                    اكسر;
                حالة 0x21:
                    mc = 2;
                    nc = 1;
                    gemm_small<2, 1>(m0, m, n0, n);
                    اكسر;
                حالة 0x14:
                    mc = 1;
                    nc = 4;
                    gemm_small<1, 4>(m0, m, n0, n);
                    اكسر;
                حالة 0x13:
                    mc = 1;
                    nc = 3;
                    gemm_small<1, 3>(m0, m, n0, n);
                    اكسر;
                حالة 0x12:
                    mc = 1;
                    nc = 2;
                    gemm_small<1, 2>(m0, m, n0, n);
                    اكسر;
                حالة 0x11:
                    mc = 1;
                    nc = 1;
                    gemm_small<1, 1>(m0, m, n0, n);
                    اكسر;
                افتراضي:
                    أرجع;
            }
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    فراغ KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        لكل (صحيح l = 0; l < k; l+=8) {
            packNormal((A+(ii*lda)+l), lda, 4, 8, (uint8_t*)vec_A);
            packNormal((B+(jj*ldb)+l), ldb, 8, 8, (uint8_t*)vec_B);
            لكل (صحيح x = 0; x < 4; x++) {
                mma_instr<TA>::outer_product(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_product(&acc_1, vec_A[x], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
    }

    فراغ KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        لكل (صحيح l = 0; l < k; l+=8) {
            packNormal((A+(ii*lda)+l), lda, 8, 8, (uint8_t*)vec_A);
            packNormal((B+(jj*ldb)+l), ldb, 8, 4, (uint8_t*)vec_B);
            لكل (صحيح x = 0; x < 4; x++) {
                mma_instr<TA>::outer_product(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_product(&acc_1, vec_A[x+4], vec_B[x]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii+4, jj);
    }


    فراغ KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        لكل (صحيح l = 0; l < k; l+=8) {
            packNormal(A+(ii*lda)+l, lda, 8, 8, (uint8_t*)vec_A);
            packNormal(B+(jj*ldb)+l, ldb, 8, 8, (uint8_t*)vec_B);
            لكل (صحيح x = 0; x < 4; x++) {
                mma_instr<TA>::outer_product(&acc_0, vec_A[x], vec_B[x]);
                mma_instr<TA>::outer_product(&acc_1, vec_A[x], vec_B[x+4]);
                mma_instr<TA>::outer_product(&acc_2, vec_A[x+4], vec_B[x]);
                mma_instr<TA>::outer_product(&acc_3, vec_A[x+4], vec_B[x+4]);
            }
        }
        save_acc(&acc_0, ii, jj);
        save_acc(&acc_1, ii, jj+4);
        save_acc(&acc_2, ii+4, jj);
        save_acc(&acc_3, ii+4, jj+4);
    }

    مضمن فراغ MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        لكل (صحيح x = 0; x < 4; x ++) {
             mma_instr<TA>::outer_product(&acc[0], vec_A0[x], vec_B[x]);
             mma_instr<TA>::outer_product(&acc[1], vec_A0[x], vec_B[x+4]);
             mma_instr<TA>::outer_product(&acc[2], vec_A0[x+4], vec_B[x]);
             mma_instr<TA>::outer_product(&acc[3], vec_A0[x+4], vec_B[x+4]);
             mma_instr<TA>::outer_product(&acc[4], vec_A1[x], vec_B[x]);
             mma_instr<TA>::outer_product(&acc[5], vec_A1[x], vec_B[x+4]);
             mma_instr<TA>::outer_product(&acc[6], vec_A1[x+4], vec_B[x]);
             mma_instr<TA>::outer_product(&acc[7], vec_A1[x+4], vec_B[x+4]);
        }
    }
    فراغ KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        لكل (int64_t i = 0; i < mc; i += 16) {
            صحيح A_base_addr = (mc / 8) * (i / 8) * 8;
            لكل (int64_t j = 0; j < nc; j += 8) {
                 صحيح B_base_addr = (nc / 8) * (j / 8) * 8;
                 acc_t acc[8];
                 vec_t A0_block[8]; vec_t A1_block[8];
                 لكل (صحيح x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 لكل (int64_t l = 0; l < kc; l += 8) {
                     صحيح A0_block_idx = A_base_addr + (l / 8) * 8;
                     صحيح A1_block_idx = A0_block_idx + (mc / 8) * 8;
                     صحيح B_block_idx = B_base_addr + (l / 8) * 8;
                     vec_t* A0_block = &vec_A[A0_block_idx];
                     vec_t* A1_block = &vec_A[A1_block_idx];
                     vec_t* B_block = &vec_B[B_block_idx];
                     MMA_16x8(A0_block, A1_block, B_block, acc);
                 }
                 إذا (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } وإلا {
                     add_save_acc(&acc[0], ii + i, jj + j);
                     add_save_acc(&acc[1], ii + i, jj + j + 4);
                     add_save_acc(&acc[2], ii + i + 4, jj + j);
                     add_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     add_save_acc(&acc[4], ii + i + 8, jj + j);
                     add_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     add_save_acc(&acc[6], ii + i + 12, jj + j);
                     add_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }
    template<صحيح RM, صحيح RN>
    فراغ gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[2], vec_B[2];
            لكل (صحيح l=0; l<k; l+=4) {
                packNormal(A+(ii*lda)+l, lda, RM, 4, (uint8_t*)vec_A);
                packNormal(B+(jj*ldb)+l, ldb, RN, 4, (uint8_t*)vec_B);
                لكل (صحيح x = 0; x<2; x++) {
                    mma_instr<TA>::outer_product(&acc_0, vec_A[x], vec_B[x]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            لكل (صحيح I = 0; I < RM; I++) {
                لكل (صحيح J = 0; J < RN; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    template<صحيح RM>
    فراغ gemm_Mx8(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        صحيح RN = 8;
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0, acc_1;
            __builtin_mma_xxsetaccz(&acc_0);
            __builtin_mma_xxsetaccz(&acc_1);
            vec_t vec_A[4], vec_B[8];
            لكل (صحيح l=0; l<k; l+=8) {
                packNormal(A+(ii*lda)+l, lda, RM, 8, (uint8_t*)vec_A);
                packNormal(B+(jj*ldb)+l, ldb, RN, 8, (uint8_t*)vec_B);
                لكل (صحيح x = 0; x<4; x++) {
                    mma_instr<TA>::outer_product(&acc_0, vec_A[x], vec_B[x]);
                    mma_instr<TA>::outer_product(&acc_1, vec_A[x], vec_B[x+4]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            لكل (صحيح I = 0; I < RM; I++) {
                لكل (صحيح J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_1);
            لكل (صحيح I = 0; I < RM; I++) {
                لكل (صحيح J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+4+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    template<صحيح RM, صحيح RN>
    مضمن فراغ kernel(int64_t ii, int64_t jj) {
       إذا constexpr(RM == 4 && RN == 8) {
          KERNEL_4x8(ii,jj);
       } وإلا إذا constexpr(RM == 8 && RN == 8) {
          KERNEL_8x8(ii,jj);
       } وإلا إذا constexpr(RM == 8 && RN == 4) {
          KERNEL_8x4(ii,jj);
       } وإلا {
          أكد(خطأ && "RN/RM values not supported");
       }
    }

    فراغ matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles) {
            end = tiles;
        }
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = (job / xtiles) * mc;
            int64_t jj = (job % xtiles) * nc;
            لكل (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 8];
                 vec_t B_pack[kc * nc / 8];
                 packNormal(A + (ii * lda) + kk, lda, kc, mc, (uint8_t *)A_pack);
                 packNormal(B + (jj * ldb) + kk, ldb, kc, nc, (uint8_t *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    template <صحيح RM, صحيح RN>
    NOINLINE فراغ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
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
    ثابت صحيح ith;
    ثابت صحيح nth;
};

template <typename TA>
صنف tinyBLAS_Q0_PPC {
  عام:
    tinyBLAS_Q0_PPC(int64_t k,
             ثابت TA * A, int64_t lda,
             ثابت block_q8_0 * B, int64_t ldb,
             عائم * C, int64_t ldc,
             صحيح ith, صحيح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    فراغ matmul(int64_t m, int64_t n) {
        int64_t mc = 64;
        int64_t nc = 64;
        int64_t kc = 64;
        int64_t n_chunk = 64;
    #إذا defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 32;
        nc = 32;
        kc = 32;
        n_chunk = 32;
    #endif
        int64_t n_aligned = 0;
        إذا (n % n_chunk == 0) {
            n_aligned = n;
        } وإلا إذا (n == 4) {
            n_aligned = 4;
        } وإلا إذا (n < n_chunk) {
            n_aligned = (n / 8) * 8;
        } وإلا {
            n_aligned = (n / n_chunk) * n_chunk;
        }
        إذا (n_aligned > 0) {
            إذا (n_aligned % n_chunk == 0) nc = n_chunk;
            وإلا إذا (n_aligned == n)      nc = n;
            وإلا إذا (n_aligned % 32 == 0) nc = 32;
            وإلا إذا (n_aligned % 24 == 0) nc = 24;
            وإلا إذا (n_aligned % 16 == 0) nc = 16;
            وإلا                          nc = 8;
        }
        منطقي can_use_tiled = n_aligned > 0 && (m % mc == 0);
        إذا (can_use_tiled) {
            matmul_tiled(m, n_aligned, mc, nc, kc);
            إذا (n > n_aligned) {
                mnpack(0, m, n_aligned, n);
            }
        } وإلا {
            mnpack(0, m, 0, n);
        }
    }

  خاص:
    مضمن فراغ save_res(صحيح ii, صحيح jj, صحيح idx, vector عائم * fin_res, صحيح RM = 4, صحيح RN = 4) {
        لكل (صحيح I = 0; I < RM; I++) {
            لكل (صحيح J = 0; J < RN; J++) {
                *((عائم *)(C + ii + ((jj + J) * ldc) + I)) = *((عائم *)&fin_res[idx + I] + J);
            }
        }
    }

    مضمن فراغ save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        لكل (صحيح I = 0; I < 4; I++) {
            لكل (صحيح J = 0; J < 4; J++) {
                *((عائم *)(C + ii + ((jj + J) * ldc) + I)) = *((عائم *)&vec_C[I] + J);
            }
        }
    }

    مضمن فراغ add_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        لكل (صحيح I = 0; I < 4; I++) {
            لكل (صحيح J = 0; J < 4; J++) {
                عائم * c_ptr = (عائم *)(C + ii+ ((jj + J) * ldc) + I);
                *c_ptr += *((عائم *)&vec_C[I] + J);
            }
        }
    }

    template<typename ArrayType>
    مضمن فراغ compute(acc_t * ACC, صحيح c_idx, صحيح s_idx, ArrayType & comparray, vector عائم * vs, vector عائم * fin_res) {
        vector signed صحيح vec_C[4];
        vector عائم CA[4] = {0};
        vector عائم res[4] = {0};
        __builtin_mma_disassemble_acc(vec_C, ACC);
        لكل (صحيح i = 0; i < 4; i++) {
            CA[i] = vec_splats((عائم)(((مزدوج)comparray[c_idx + i]) * -128.0));
            res[i] = vec_add(vec_ctf(vec_C[i], 0), CA[i]);
            fin_res[s_idx + i] = vec_madd(res[i], vs[s_idx + i], fin_res[s_idx + i]);
        }
    }

    مضمن فراغ process_q4_elements(vector signed حرف (&c)[2], صحيح * ca) {
        ثابت vector signed حرف رنده_Mask = vec_splats((signed حرف)0xF);
        ثابت vector unsigned حرف v4 = vec_splats((unsigned حرف)0x4);
        ثابت vector signed حرف v8 = vec_splats((signed حرف)0x8);
        vector signed صحيح vsum = {0};
        vector signed صحيح vsum2 = {0};
        c[0] = vec_and(c[1], رنده_Mask);
        c[1] = vec_sr(c[1], v4);
        c[0] = vec_sub(c[0], v8);
        c[1] = vec_sub(c[1], v8);
        vsum = vec_sum4s(c[0], vsum);
        vsum2 = vec_sum4s(c[1], vsum2);
        vsum = vec_add(vsum, vsum2);
        *(ca) = vsum[0] + vsum[1] + vsum[2] + vsum[3];
    }

    template <typename V1, typename V2>
    مضمن فراغ vector_permute_store(V2 & s1, V2 & s2, V2 & s3, V2 & s4, V1 * vecOffset, منطقي flip) {
        vector unsigned حرف swiz1 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vector unsigned حرف swiz2 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        vector unsigned حرف swiz3 = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
        vector unsigned حرف swiz4 = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
        V2 t1, t2, t3, t4, t5, t6, t7, t8;
        vector unsigned حرف xor_vector;
        uint8_t flip_vec = 0x80;
        xor_vector = vec_splats(flip_vec);
        t1 = vec_perm(s1, s2, swiz1);
        t2 = vec_perm(s1, s2, swiz2);
        t3 = vec_perm(s3, s4, swiz1);
        t4 = vec_perm(s3, s4, swiz2);
        t5 = vec_perm(t1, t3, swiz3);
        t6 = vec_perm(t1, t3, swiz4);
        t7 = vec_perm(t2, t4, swiz3);
        t8 = vec_perm(t2, t4, swiz4);
        إذا (flip == true) {
            t5 = vec_xor(t5, xor_vector);
            t6 = vec_xor(t6, xor_vector);
            t7 = vec_xor(t7, xor_vector);
            t8 = vec_xor(t8, xor_vector);
        }
        vec_xst(t5, 0, vecOffset);
        vec_xst(t6, 0, vecOffset + 16);
        vec_xst(t7, 0, vecOffset + 32);
        vec_xst(t8, 0, vecOffset + 48);
    }

    مضمن فراغ unpack_q4_to_q8(vector signed حرف packed, vector signed حرف & lo, vector signed حرف & hi) {
        ثابت vector signed حرف رنده_Mask = vec_splats((signed حرف)0x0F);
        ثابت vector signed حرف v8      = vec_splats((signed حرف)0x08);
        ثابت vector unsigned حرف v4    = vec_splats((unsigned حرف)4);
        lo = vec_and(packed, رنده_Mask);
        hi = vec_sr(packed, v4);
        lo = vec_sub(lo, v8);
        hi = vec_sub(hi, v8);
    }

    مضمن فراغ vector_permute_store_fp16(vec_t * c, unsigned حرف * vecOffset) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        لكل (صحيح i = 0; i < 4; i += 2) {
            t[i + 0] = vec_perm(c[i + 0], c[i + 1], swiz1);
            t[i + 1] = vec_perm(c[i + 0], c[i + 1], swiz2);
        }
        لكل (صحيح i = 4; i < 8; i += 2) {
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
        لكل (صحيح i = 0; i < 8; ++i) {
            vec_xst(s[i], 0, (vec_t *)(vecOffset + i * 16));
        }
    }

    ساكن مضمن فراغ convert_and_scale_q8(vector signed حرف raw, vector عائم v_scale, vector unsigned short & out_hi, vector unsigned short & out_lo) {
        vector signed short i16_hi = vec_unpackh(raw);
        vector signed short i16_lo = vec_unpackl(raw);

        vector عائم f_hi_h = vec_ctf(vec_unpackh(i16_hi), 0);
        vector عائم f_hi_l = vec_ctf(vec_unpackl(i16_hi), 0);
        vector عائم f_lo_h = vec_ctf(vec_unpackh(i16_lo), 0);
        vector عائم f_lo_l = vec_ctf(vec_unpackl(i16_lo), 0);
        out_hi = vec_pack_to_short_fp32(vec_mul(f_hi_h, v_scale), vec_mul(f_hi_l, v_scale));
        out_lo = vec_pack_to_short_fp32(vec_mul(f_lo_h, v_scale), vec_mul(f_lo_l, v_scale));
    }

    فراغ packNormal_q4_fp16(ثابت block_q4_0 * a, int64_t lda, صحيح rows, صحيح blocks, unsigned حرف * vec) {
        unsigned حرف * vecOffset = vec;
        لكل (صحيح i = 0; i < rows; i += 8) {
            ثابت block_q4_0 * rows_base[8];
            لكل (صحيح r = 0; r < 8; r++) {
                rows_base[r] = a + (i + r) * lda;
            }
            لكل (صحيح blk = 0; blk < blocks; blk++) {
                vector unsigned short hp_res[8][4];
                لكل (صحيح r = 0; r < 8; r++) {
                    ثابت block_q4_0 * current_blk = rows_base[r] + blk;
                    vector عائم v_scale = vec_extract_fp32_from_shorth(vec_splats(current_blk->d));
                    vector signed حرف v_qs = vec_xl(0, (ثابت vector signed حرف *)current_blk->qs);
                    vector signed حرف c1, c2;
                    unpack_q4_to_q8(v_qs, c1, c2);
                    convert_and_scale_q8(c1, v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c2, v_scale, hp_res[r][2], hp_res[r][3]);
                }
                لكل (صحيح c = 0; c < 4; c++) {
                    vector unsigned حرف c_arr[8];
                    لكل (صحيح r = 0; r < 8; r++) {
                        c_arr[r] = (vector unsigned حرف)hp_res[r][c];
                    }
                    vector_permute_store_fp16((vec_t *)c_arr, vecOffset);
                    vecOffset += 128;
                }
            }
        }
    }

    template <صحيح chunk_size>
    ساكن مضمن فراغ pack_q8_block(ثابت block_q8_0 * a, int64_t lda, صحيح rows, صحيح blocks, unsigned حرف * vec) {
        unsigned حرف * vecOffset = vec;
        ثابت vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        ثابت vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        ثابت vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        ثابت vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        لكل (صحيح i = 0; i < rows; i += chunk_size) {
            ثابت block_q8_0 * rows_base[chunk_size];
            لكل (صحيح r = 0; r < chunk_size; r++) {
                rows_base[r] = a + (i + r) * lda;
            }
            لكل (صحيح blk = 0; blk < blocks; blk++) {
                vector unsigned short hp_res[chunk_size][4];
                لكل (صحيح r = 0; r < chunk_size; r++) {
                    ثابت block_q8_0 * b = rows_base[r] + blk;
                    vector عائم v_scale = vec_extract_fp32_from_shorth(vec_splats(b->d));
                    vector signed حرف c[2];
                    __vector_pair pair = __builtin_vsx_lxvp(0, (__vector_pair *)b->qs);
                    __builtin_vsx_disassemble_pair(c, & pair);
                    convert_and_scale_q8(c[0], v_scale, hp_res[r][0], hp_res[r][1]);
                    convert_and_scale_q8(c[1], v_scale, hp_res[r][2], hp_res[r][3]);
                }
                لكل (صحيح col = 0; col < 4; col++) {
                    إذا constexpr (chunk_size == 8) {
                        vec_t t[8];
                        t[0] = vec_perm((vec_t)hp_res[0][col], (vec_t)hp_res[1][col], swiz1);
                        t[1] = vec_perm((vec_t)hp_res[0][col], (vec_t)hp_res[1][col], swiz2);
                        t[2] = vec_perm((vec_t)hp_res[2][col], (vec_t)hp_res[3][col], swiz1);
                        t[3] = vec_perm((vec_t)hp_res[2][col], (vec_t)hp_res[3][col], swiz2);
                        t[4] = vec_perm((vec_t)hp_res[4][col], (vec_t)hp_res[5][col], swiz1);
                        t[5] = vec_perm((vec_t)hp_res[4][col], (vec_t)hp_res[5][col], swiz2);
                        t[6] = vec_perm((vec_t)hp_res[6][col], (vec_t)hp_res[7][col], swiz1);
                        t[7] = vec_perm((vec_t)hp_res[6][col], (vec_t)hp_res[7][col], swiz2);

                        vec_xst(vec_perm(t[0], t[2], swiz3), 0, (vec_t *)(vecOffset + 0));
                        vec_xst(vec_perm(t[0], t[2], swiz4), 0, (vec_t *)(vecOffset + 16));
                        vec_xst(vec_perm(t[1], t[3], swiz3), 0, (vec_t *)(vecOffset + 32));
                        vec_xst(vec_perm(t[1], t[3], swiz4), 0, (vec_t *)(vecOffset + 48));
                        vec_xst(vec_perm(t[4], t[6], swiz3), 0, (vec_t *)(vecOffset + 64));
                        vec_xst(vec_perm(t[4], t[6], swiz4), 0, (vec_t *)(vecOffset + 80));
                        vec_xst(vec_perm(t[5], t[7], swiz3), 0, (vec_t *)(vecOffset + 96));
                        vec_xst(vec_perm(t[5], t[7], swiz4), 0, (vec_t *)(vecOffset + 112));
                        vecOffset += 128;
                    } وإلا {
                        vec_t t0 = vec_perm((vec_t)hp_res[0][col], (vec_t)hp_res[1][col], swiz1);
                        vec_t t1 = vec_perm((vec_t)hp_res[0][col], (vec_t)hp_res[1][col], swiz2);
                        vec_t t2 = vec_perm((vec_t)hp_res[2][col], (vec_t)hp_res[3][col], swiz1);
                        vec_t t3 = vec_perm((vec_t)hp_res[2][col], (vec_t)hp_res[3][col], swiz2);

                        vec_xst(vec_perm(t0, t2, swiz3), 0, (vec_t *)(vecOffset + 0));
                        vec_xst(vec_perm(t0, t2, swiz4), 0, (vec_t *)(vecOffset + 16));
                        vec_xst(vec_perm(t1, t3, swiz3), 0, (vec_t *)(vecOffset + 32));
                        vec_xst(vec_perm(t1, t3, swiz4), 0, (vec_t *)(vecOffset + 48));
                        vecOffset += 64;
                    }
                }
            }
        }
    }

    فراغ packNormal_q8_fp16(ثابت block_q8_0 * a, int64_t lda, صحيح rows, صحيح blocks, unsigned حرف * vec) {
        إذا (rows == 4) {
            pack_q8_block<4>(a, lda, rows, blocks, vec);
        } وإلا {
            pack_q8_block<8>(a, lda, rows, blocks, vec);
        }
    }

    template<صحيح size>
    فراغ packNormalInt4(ثابت TA * a, int64_t lda, صحيح rows, صحيح cols, int8_t * vec, std::array<صحيح, size> & comparray) {
        int64_t i, j;
        TA * aoffset = عدم;
        int8_t * vecOffset = عدم;
        TA * aoffset1 = عدم, * aoffset2 = عدم, * aoffset3 = عدم, * aoffset4 = عدم;
        TA * aoffset5 = عدم, * aoffset6 = عدم, * aoffset7 = عدم, * aoffset8 = عدم;
        vector signed حرف c1[2] = {0}, c2[2] = {0}, c3[2] = {0}, c4[2] = {0};
        vector signed حرف c5[2] = {0}, c6[2] = {0}, c7[2] = {0}, c8[2] = {0};
        aoffset = const_cast<TA *>(a);
        vecOffset = vec;
        j = (rows >> 3);
        إذا (j > 0) {
            افعل {
                aoffset1 = aoffset;
                aoffset2 = aoffset1 + lda;
                aoffset3 = aoffset2 + lda;
                aoffset4 = aoffset3 + lda;
                aoffset5 = aoffset4 + lda;
                aoffset6 = aoffset5 + lda;
                aoffset7 = aoffset6 + lda;
                aoffset8 = aoffset7 + lda;
                aoffset += 8 * lda;
                i = (cols >> 2);
                إذا (i > 0) {
                    افعل {
                        c1[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset1->qs);
                        c2[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset2->qs);
                        c3[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset3->qs);
                        c4[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset4->qs);
                        c5[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset5->qs);
                        c6[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset6->qs);
                        c7[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset7->qs);
                        c8[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset8->qs);

                        process_q4_elements(c1, & comparray[0]);
                        process_q4_elements(c2, & comparray[1]);
                        process_q4_elements(c3, & comparray[2]);
                        process_q4_elements(c4, & comparray[3]);
                        process_q4_elements(c5, & comparray[4]);
                        process_q4_elements(c6, & comparray[5]);
                        process_q4_elements(c7, & comparray[6]);
                        process_q4_elements(c8, & comparray[7]);
                        vector_permute_store<int8_t, vector signed حرف>(c1[0], c2[0], c3[0], c4[0], vecOffset, خطأ);
                        vector_permute_store<int8_t, vector signed حرف>(c1[1], c2[1], c3[1], c4[1], vecOffset + 64, خطأ);
                        vector_permute_store<int8_t, vector signed حرف>(c5[0], c6[0], c7[0], c8[0], vecOffset + 128, خطأ);
                        vector_permute_store<int8_t, vector signed حرف>(c5[1], c6[1], c7[1], c8[1], vecOffset + 192, خطأ);
                        aoffset1 += lda;
                        aoffset2 += lda;
                        aoffset3 += lda;
                        aoffset4 += lda;
                        aoffset5 += lda;
                        aoffset6 += lda;
                        aoffset7 += lda;
                        aoffset8 += lda;
                        vecOffset += 256;
                        i--;
                    } طالما (i > 0);
                }
                j--;
            } طالما (j > 0);
        }

        إذا (rows & 4) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset += 4 * lda;
            i = (cols >> 2);
            إذا (i > 0) {
                افعل {
                    c1[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset1->qs);
                    c2[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset2->qs);
                    c3[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset3->qs);
                    c4[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset4->qs);

                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    vector_permute_store<int8_t, vector signed حرف>(c1[0], c2[0], c3[0], c4[0], vecOffset, خطأ);
                    vector_permute_store<int8_t, vector signed حرف>(c1[1], c2[1], c3[1], c4[1], vecOffset + 64, خطأ);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    aoffset4 += lda;
                    vecOffset += 128;
                    i--;
                } طالما (i > 0);
            }
        }

        إذا (rows & 3) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            i = (cols >> 2);
            إذا (i > 0) {
                افعل {
                    بدل(rows) {
                        حالة 3: c3[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset3->qs);
                        حالة 2: c2[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset2->qs);
                        حالة 1: c1[1] = vec_xl(0, (ثابت vector signed حرف *)aoffset1->qs);
                            اكسر;
                    }
                    process_q4_elements(c1, & comparray[0]);
                    process_q4_elements(c2, & comparray[1]);
                    process_q4_elements(c3, & comparray[2]);
                    process_q4_elements(c4, & comparray[3]);
                    vector_permute_store<int8_t, vector signed حرف>(c1[0], c2[0], c3[0], c4[0], vecOffset, خطأ);
                    vector_permute_store<int8_t, vector signed حرف>(c1[1], c2[1], c3[1], c4[1], vecOffset + 64, خطأ);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    vecOffset += 128;
                    i--;
                } طالما(i > 0);
            }
        }
    }

    template<typename VA, typename VB>
    فراغ packNormal(ثابت block_q8_0 * a, int64_t lda, صحيح rows, صحيح cols, VA * vec, منطقي flip) {
        int64_t i, j;
        block_q8_0 * aoffset = عدم;
        VA * vecOffset = عدم;
        block_q8_0 * aoffsets[8];
        __vector_pair arr[8];
        VB c[8][2] = {0};
        VB c1[8] = {0}; VB c2[8] = {0};
        aoffset = const_cast<block_q8_0 *>(a);
        vecOffset = vec;
        j = (rows >> 3);
        إذا (j > 0) {
            افعل {
                aoffsets[0] = aoffset;
                لكل (صحيح it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it - 1] + lda;
                aoffset += 8 * lda;

                i = (cols >> 3);
                إذا (i > 0) {
                افعل {
                    لكل (صحيح it = 0; it < 8; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    vector_permute_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vecOffset, flip);
                    vector_permute_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vecOffset + 64, flip);
                    vector_permute_store<VA, VB>(c1[4], c1[5], c1[6], c1[7], vecOffset + 128, flip);
                    vector_permute_store<VA, VB>(c2[4], c2[5], c2[6], c2[7], vecOffset + 192, flip);
                    لكل (صحيح it = 0; it < 8; it++)
                        aoffsets[it] += lda;
                    vecOffset += 256;
                    i--;
               } طالما(i > 0);
            }
            j--;
        } طالما(j > 0);
    }
    إذا (rows & 4) {
            aoffsets[0]  = aoffset;
            لكل (صحيح it = 1; it < 4; it++ )
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
        i = (cols >> 3);
            إذا (i > 0) {
               افعل {
                    لكل (صحيح it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], & arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    vector_permute_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vecOffset, flip);
                    vector_permute_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vecOffset + 64, flip);
                    لكل (صحيح it = 0; it < 4; it++) {
                        aoffsets[it] += lda;
                    }
                    vecOffset += 128;
                    i--;
               } طالما(i > 0);
            }
        }

        إذا (rows & 3) {
            aoffsets[0]  = aoffset;
            لكل (صحيح it = 1; it < 3; it++ )
                aoffsets[it] = aoffsets[it - 1] + lda;
            i = (cols >> 3);
            إذا (i > 0) {
                افعل {
                    بدل(rows) {
                        حالة 3: arr[2] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[2]->qs);
                                __builtin_vsx_disassemble_pair(c[2], & arr[2]);
                                c1[2] = c[2][0]; c2[2] = c[2][1];
                        حالة 2: arr[1] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[1]->qs);
                                __builtin_vsx_disassemble_pair(c[1], & arr[1]);
                                c1[1] = c[1][0]; c2[1] = c[1][1];
                        حالة 1: arr[0] = __builtin_vsx_lxvp(0, (__vector_pair *)aoffsets[0]->qs);
                                __builtin_vsx_disassemble_pair(c[0], & arr[0]);
                                c1[0] = c[0][0]; c2[0] = c[0][1];
                                اكسر;
                    }
                    vector_permute_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vecOffset, flip);
                    vector_permute_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vecOffset + 64, flip);
                    لكل (صحيح it = 0; it < 3; it++)
                         aoffsets[it] += lda;
                    vecOffset += 128;
                    i--;
               } طالما(i > 0);
            }
        }
    }

    فراغ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        صحيح m_rem = MIN(m - m0, 16);
        صحيح n_rem = MIN(n - n0, 16);

        صحيح mc = 0, nc = 0;

        إذا (m_rem >= 8 && n_rem >= 8) {
           mc = 8;
           nc = 8;
           gemm<8, 8>(m0, m, n0, n);
        } وإلا إذا (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } وإلا إذا (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } وإلا إذا (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small(m0, m, n0, n, mc, nc);
        } وإلا {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            إذا (mc == 0 || nc == 0)
               أرجع;
            gemm_small(m0, m, n0, n, mc, nc);
        }

        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }


    فراغ KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[16] = {0};
        acc_t acc_0, acc_1;
        std::array<صحيح, 4> comparray {};
        vector عائم fin_res[8] = {0};
        vector عائم vs[8] = {0};
        منطقي isAblock_q4 = std::is_same_v<TA, block_q4_0>;
        لكل (صحيح l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            إذا (std::is_same_v<TA, block_q4_0>) {
               packNormalInt4<4>((A + (ii * lda) + l), lda, 4, 4, (int8_t *)vec_A, comparray);
            } وإلا {
               packNormal<int8_t, vector signed حرف>((ثابت block_q8_0 *)(A + (ii * lda) + l), lda, 4, 8, (int8_t *)vec_A, خطأ);
            }
            packNormal<uint8_t, vector unsigned حرف>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, true);
            لكل(صحيح x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x], vec_B[x+8]);
            }
            لكل (صحيح I = 0; I<4; I++) {
                لكل (صحيح J = 0; J<4; J++) {
                    *((عائم *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((عائم *)& vs[I + 4] + J) = (unhalf((A +((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            إذا (!isAblock_q4) {
                auto aoffset = A + (ii * lda) + l;
                لكل (صحيح i = 0; i < 4; i++) {
                    comparray[i] = 0;
                    صحيح ca = 0;
                    auto *at = aoffset->qs;
                    لكل (صحيح j = 0; j < 32; j++)
                        ca += (صحيح)*at++;
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

    فراغ KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[8] = {0};
        acc_t acc_0, acc_1;
        std::array<صحيح, 8> comparray {};
        vector عائم fin_res[8] = {0};
        vector عائم vs[8] = {0};
        منطقي isAblock_q4 = std::is_same_v<TA, block_q4_0>;
        لكل (صحيح l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            إذا (std::is_same_v<TA, block_q4_0>) {
               packNormalInt4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } وإلا {
               packNormal<int8_t, vector signed حرف>((ثابت block_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, خطأ);
            }
            packNormal<uint8_t, vector unsigned حرف>((B + (jj * ldb) + l), ldb, 4, 8, (uint8_t *)vec_B, true);
            لكل(صحيح x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
            }
            لكل (صحيح I = 0; I < 8; I++) {
                لكل (صحيح J = 0; J < 4; J++) {
                    *((عائم *)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                }
            }
            إذا (!isAblock_q4) {
                auto aoffset = A + (ii * lda) + l;
                لكل (صحيح i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    صحيح ca = 0;
                    auto *at = aoffset->qs;
                    لكل (صحيح j = 0; j < 32; j++)
                        ca += (صحيح)*at++;
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

    فراغ KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16] = {0};
        acc_t acc_0, acc_1, acc_2, acc_3;
        acc_t acc_4, acc_5, acc_6, acc_7;
        std::array<صحيح, 8> comparray {};
        vector عائم fin_res[16] = {0};
        vector عائم vs[16] = {0};
        منطقي isAblock_q4 = std::is_same_v<TA, block_q4_0>;
        لكل (صحيح l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(& acc_0);
            __builtin_mma_xxsetaccz(& acc_1);
            __builtin_mma_xxsetaccz(& acc_2);
            __builtin_mma_xxsetaccz(& acc_3);
            إذا (std::is_same_v<TA, block_q4_0>) {
               packNormalInt4<8>((A + (ii * lda) + l), lda, 8, 4, (int8_t *)vec_A, comparray);
            } وإلا {
               packNormal<int8_t, vector signed حرف>((ثابت block_q8_0 *)(A + (ii * lda) + l), lda, 8, 8, (int8_t *)vec_A, خطأ);
            }
            packNormal<uint8_t, vector unsigned حرف>((B + (jj * ldb) + l), ldb, 8, 8, (uint8_t *)vec_B, true);
            لكل(صحيح x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_1, vec_A[x + 8], vec_B[x]);
                __builtin_mma_xvi8ger4pp(& acc_2, vec_A[x], vec_B[x + 8]);
                __builtin_mma_xvi8ger4pp(& acc_3, vec_A[x + 8], vec_B[x + 8]);
            }
            لكل (صحيح I = 0; I < 8 ; I++) {
                لكل (صحيح J = 0; J < 4; J++) {
                    *((عائم *)& vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    *((عائم *)& vs[I + 8] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J + 4) * ldb) + l)->d));
                }
            }
            إذا (!isAblock_q4) {
                auto aoffset = A + (ii * lda) + l;
                لكل (صحيح i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    صحيح ca = 0;
                    auto *at = aoffset->qs;
                    لكل (صحيح j = 0; j < 32; j++)
                        ca += (صحيح)*at++;
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

    فراغ KERNEL_Q0(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, int64_t l, vec_t * vec_A, vec_t * vec_B) {
        acc_t acc[8];
        لكل (صحيح i = 0; i < mc ; i += 16) {
            لكل (صحيح j = 0; j < nc; j += 8) {
                صحيح A0_base = (i / 16) * (2 * 32 * kc);
                صحيح B0_base = (j / 8) * (32 * kc);
                لكل (صحيح x = 0; x < 8; x++) {
                     __builtin_mma_xxsetaccz(&acc[x]);
                }
                لكل (int64_t kk = 0; kk < kc; kk++) {
                    صحيح A0_block_idx = A0_base + kk * 32;
                    صحيح B0_block_idx = B0_base + kk * 32;
                    صحيح A1_block_idx = A0_block_idx + 32 * kc;
                    صحيح B1_block_idx = B0_block_idx + 32 * kc;
                    vec_t * A0_block = & vec_A[A0_block_idx];
                    vec_t * B0_block = & vec_B[B0_block_idx];
                    vec_t * A1_block = & vec_A[A1_block_idx];
                    لكل (صحيح it = 0; it < 4; it++) {
                        لكل (صحيح x = 0; x < 4; x++) {
                            __builtin_mma_xvf16ger2pp(& acc[0], A0_block[8 * it + x], B0_block[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[1], A0_block[8 * it + x], B0_block[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[2], A0_block[8 * it + x + 4], B0_block[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[3], A0_block[8 * it + x + 4], B0_block[8 * it + x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[4], A1_block[8 * it + x], B0_block[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[5], A1_block[8 * it + x], B0_block[8 * it+ x + 4]);
                            __builtin_mma_xvf16ger2pp(& acc[6], A1_block[8 * it + x + 4], B0_block[8 * it + x]);
                            __builtin_mma_xvf16ger2pp(& acc[7], A1_block[8 * it + x + 4], B0_block[8 * it + x + 4]);
                        }
                    }
                }
                إذا (l == 0) {
                    save_acc(& acc[0], ii + i, jj + j);
                    save_acc(& acc[1], ii + i, jj + j + 4);
                    save_acc(& acc[2], ii + i + 4, jj + j);
                    save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    save_acc(& acc[4], ii + i + 8, jj + j);
                    save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    save_acc(& acc[6], ii + i + 12, jj + j);
                    save_acc(& acc[7], ii + i + 12, jj + j + 4);
                } وإلا {
                    add_save_acc(& acc[0], ii + i, jj + j);
                    add_save_acc(& acc[1], ii + i, jj + j + 4);
                    add_save_acc(& acc[2], ii + i + 4, jj + j);
                    add_save_acc(& acc[3], ii + i + 4, jj + j + 4);
                    add_save_acc(& acc[4], ii + i + 8, jj + j);
                    add_save_acc(& acc[5], ii + i + 8, jj + j + 4);
                    add_save_acc(& acc[6], ii + i + 12, jj + j);
                    add_save_acc(& acc[7], ii + i + 12, jj + j + 4);
                }
            }
        }
    }

    فراغ matmul_tiled(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        vec_t A_pack[mc * kc * 4];
        vec_t B_pack[nc * kc * 4];
        constexpr منطقي is_Ablock_q4 = std::is_same_v<TA, block_q4_0>;
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles  = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles) {
            end = tiles;
        }
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = (job / xtiles) * mc;
            int64_t jj = (job % xtiles) * nc;
            لكل (int64_t kk = 0; kk < k; kk += kc) {
                int64_t k_cur = MIN(kc, k - kk);
                إذا constexpr(is_Ablock_q4) {
                    packNormal_q4_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                } وإلا {
                    packNormal_q8_fp16(A + ii * lda + kk, lda, mc, k_cur, (uint8_t *)A_pack);
                }
                packNormal_q8_fp16(B + jj * ldb + kk, ldb, nc, k_cur, (uint8_t *)B_pack);
                KERNEL_Q0(ii, jj, mc, nc, k_cur, kk, A_pack, B_pack);
            }
        }
    }

    فراغ gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, صحيح RM, صحيح RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        vec_t vec_A[8] = {0}, vec_B[8] = {0};
        vector signed صحيح vec_C[4];
        acc_t acc_0;
        منطقي isAblock_q4 = std::is_same_v<TA, block_q4_0>;

        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            std::array<صحيح, 4> comparray{};
            vector عائم res[4] = {0};
            vector عائم fin_res[4] = {0};
            vector عائم vs[4] = {0};
            vector عائم CA[4] = {0};
            __builtin_prefetch((A + (ii * lda) + 0)->qs, 0, 1); // prefetch first value
            __builtin_prefetch((B + (jj * ldb) + 0)->qs, 0, 1); // prefetch first value
            لكل (صحيح l = 0; l < k; l++) {
                __builtin_prefetch((A + (ii * lda) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_prefetch((B + (jj * ldb) + (l + 1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_mma_xxsetaccz(& acc_0);
                إذا (isAblock_q4) {
                    packNormalInt4<4>((A + (ii * lda) + l), lda, RM, 4, (int8_t *)vec_A, comparray);
                } وإلا {
                    packNormal<int8_t, vector signed حرف>((ثابت block_q8_0 *)(A + (ii * lda) + l), lda, RM, 8, (int8_t *)vec_A, خطأ);
                }
                packNormal<uint8_t, vector unsigned حرف>((B + (jj * ldb) + l), ldb, RN, 8, (uint8_t *)vec_B, true);
                لكل (صحيح x = 0; x < 8; x += 4) {
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x], vec_B[x]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 1], vec_B[x + 1]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 2], vec_B[x + 2]);
                    __builtin_mma_xvi8ger4pp(& acc_0, vec_A[x + 3], vec_B[x + 3]);
                }
                لكل (صحيح I = 0; I < RM; I++) {
                    لكل (صحيح J = 0; J < RN; J++) {
                        *((عائم*)&vs[I] + J) = (unhalf((A + ((ii + I) * lda) + l)->d) * unhalf((B + ((jj + J) * ldb) + l)->d));
                    }
                }
                __builtin_mma_disassemble_acc(vec_C, & acc_0);
                إذا (!isAblock_q4) {
                    auto aoffset = A + (ii * lda) + l;
                    لكل (صحيح i = 0; i < RM; i++) {
                        comparray[i] = 0;
                        صحيح ca = 0;
                        auto *at = aoffset->qs;
                        لكل (صحيح j = 0; j < 32; j++)
                            ca += (صحيح)*at++;
                        comparray[i] = ca;
                        aoffset += lda;
                    }
                }
                لكل (صحيح i = 0; i < RM; i++) {
                    CA[i] = vec_splats((عائم)(((مزدوج)comparray[i]) * -128.0));
                    res[i] = vec_add(vec_ctf(vec_C[i], 0), CA[i]);
                    fin_res[i] = vec_madd(res[i], vs[i], fin_res[i]);
                }
            }
            save_res(ii, jj, 0, fin_res, RM, RN);
        }
    }

    template<صحيح RM, صحيح RN>
    مضمن فراغ kernel(int64_t ii, int64_t jj) {
        إذا constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii,jj);
        } وإلا إذا constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii,jj);
        } وإلا إذا constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii,jj);
        } وإلا {
            أكد(خطأ && "RN/RM values not supported");
        }
    }

    template <صحيح RM, صحيح RN>
    NOINLINE فراغ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }
    ثابت TA * ثابت A;
    ثابت block_q8_0 * ثابت B;
    عائم * C;
    ثابت int64_t k;
    int64_t kc;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
    ثابت صحيح ith;
    ثابت صحيح nth;
};

صنف tinyBLAS_PPC {
  عام:
    tinyBLAS_PPC(int64_t k,
                ثابت عائم * A, int64_t lda,
                ثابت عائم * B, int64_t ldb,
                عائم * C, int64_t ldc,
                صحيح ith, صحيح nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    فراغ matmul(int64_t m, int64_t n) {
        int64_t mc = 256;
        int64_t nc = 256;
        int64_t kc = 256;
    #إذا defined(_AIX) || defined(__BIG_ENDIAN__)
        mc = 128;
        nc = 128;
        kc = 128;
    #endif
        إذا (m % mc == 0 && n % nc == 0 && k % kc == 0) {
            matmul_tiled(m, n, mc, nc, kc);
        } وإلا {
            mnpack(0, m, 0, n);
        }
    }

  خاص:

    __attribute__((always_inline))
    مضمن فراغ save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        لكل (صحيح I = 0; I < 4; I++) {
            لكل (صحيح J = 0; J < 4; J++) {
                *((عائم *)(C+ii+((jj+J)*ldc)+I)) = *((عائم *)&vec_C[I]+J);
            }
        }
    }

    __attribute__((always_inline))
    مضمن فراغ add_save_acc(acc_t * ACC, int64_t ii, int64_t jj) {
        vec_t vec_C[4];
        __builtin_mma_disassemble_acc(vec_C, ACC);
        لكل (صحيح I = 0; I < 4; I++) {
            لكل (صحيح J = 0; J < 4; J++) {
                عائم * c_ptr = (عائم *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((عائم *)&vec_C[I]+J);
            }
        }
    }

    مضمن فراغ vector_permute_store_4(vector عائم * src, عائم * vecOffset) {
        vector عائم t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergel(src[0], src[1]);
        t4 = vec_mergel(src[2], src[3]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t1, t2, 3);
        t7 = vec_xxpermdi(t3, t4, 0);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vecOffset);
        vec_xst(t6, 0, vecOffset + 4);
        vec_xst(t7, 0, vecOffset + 8);
        vec_xst(t8, 0, vecOffset + 12);
    }

    مضمن فراغ vector_permute_store_8(vector عائم * src, عائم * vecOffset) {
        vector عائم t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = vec_mergeh(src[0], src[1]);
        t2 = vec_mergeh(src[2], src[3]);
        t3 = vec_mergeh(src[4], src[5]);
        t4 = vec_mergeh(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vecOffset);
        vec_xst(t6, 0, vecOffset + 4);
        vec_xst(t7, 0, vecOffset + 8);
        vec_xst(t8, 0, vecOffset + 12);

        t1 = vec_mergel(src[0], src[1]);
        t2 = vec_mergel(src[2], src[3]);
        t3 = vec_mergel(src[4], src[5]);
        t4 = vec_mergel(src[6], src[7]);

        t5 = vec_xxpermdi(t1, t2, 0);
        t6 = vec_xxpermdi(t3, t4, 0);
        t7 = vec_xxpermdi(t1, t2, 3);
        t8 = vec_xxpermdi(t3, t4, 3);

        vec_xst(t5, 0, vecOffset + 16);
        vec_xst(t6, 0, vecOffset + 20);
        vec_xst(t7, 0, vecOffset + 24);
        vec_xst(t8, 0, vecOffset + 28);
    }

    فراغ packTranspose(ثابت عائم * a, int64_t lda, صحيح rows, صحيح cols, عائم * vec) {
        int64_t i, j;
        عائم * aoffsets[8];
        عائم * aoffset = عدم, * boffset = عدم;
        __vector_pair arr[8];
        vector عائم c[8][2] = {0};
        vector عائم c1[8] = {0};
        vector عائم c2[8] = {0};
        aoffset = const_cast<عائم *>(a);
        boffset = vec;
        j = (rows >> 3);
        إذا (j > 0) {
            افعل {
                aoffsets[0] = aoffset;
                لكل (صحيح it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it-1] + lda;
                aoffset += 8 * lda;
                i = (cols >> 3);
                إذا (i > 0) {
                    افعل {
                        لكل (صحيح it = 0; it < 8; it++) {
                            arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                            __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                            c1[it] = c[it][0];
                            c2[it] = c[it][1];
                        }

                        vector_permute_store_8(c1, boffset);
                        vector_permute_store_8(c2, boffset + 32);
                        boffset += 64;
                        i--;
                        إذا (i > 0) {
                           لكل (صحيح it = 0; it < 8; it++) {
                               aoffsets[it] = aoffsets[it] + 8;
                           }
                        }
                    } طالما(i > 0);
                }
                إذا (cols & 4) {
                    لكل (صحيح it = 0; it < 8 ; it++)
                        c1[it] = vec_xl(0, aoffsets[it]);
                    vector_permute_store_8(c1, boffset);
                }
            j--;
            } طالما(j > 0);
        }

        إذا (rows & 4) {
            aoffsets[0] = aoffset;
            لكل (صحيح it = 1; it < 4; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            i = (cols >> 3);
            إذا (i > 0) {
                افعل {
                    لكل (صحيح it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                        __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    vector_permute_store_4(c1, boffset);
                    vector_permute_store_4(c2, boffset + 16);
                    لكل (صحيح it = 0; it < 4; it++)
                        aoffsets[it] += 8 * lda;
                    boffset += 32;
                    i--;
                } طالما(i > 0);
            }

            إذا (cols & 4) {
               لكل (صحيح it = 0; it < 4; it++)
                   c1[it] = vec_xl(0, aoffsets[it]);
                vector_permute_store_4(c1, boffset);
            }
        }
        إذا (rows & 3) {
            aoffsets[0] = aoffset;
            لكل (صحيح it = 1; it < 3; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            إذا (cols & 4) {
                لكل (صحيح it = 0; it < 3; it++)
                    c1[it] = vec_xl(0, aoffsets[it]);
                vector_permute_store_4(c1, boffset);
            }
        }
    }

    فراغ KERNEL_4x4(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[4], vec_C[4];
        acc_t acc_0;
        __builtin_mma_xxsetaccz(&acc_0);
        لكل (صحيح l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (عائم *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (عائم *)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
        }
        save_acc(&acc_0, ii, jj);
    }

    فراغ KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        لكل (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 4, 4, (عائم *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 4, (عائم *)vec_B);
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

    فراغ KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        لكل (int64_t l = 0; l < k; l += 4) {
            packTranspose(A + (ii * lda) + l, lda, 8, 4, (عائم *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 4, 4, (عائم *)vec_B);
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

    فراغ KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        لكل (صحيح l = 0; l < k; l+=8) {
            packTranspose(A + (ii * lda) + l, lda, 8, 8, (عائم *)vec_A);
            packTranspose(B + (jj * ldb) + l, ldb, 8, 8, (عائم *)vec_B);
            لكل(صحيح x = 0; x < 16; x+=2) {
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

    مضمن فراغ MMA_16x8(vec_t * vec_A0, vec_t * vec_A1, vec_t * vec_B, acc_t * acc) {
        لكل (صحيح x = 0; x < 16; x += 2) {
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

    فراغ KERNEL(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, vec_t * vec_A, vec_t * vec_B, int64_t kk) {
        لكل (int64_t i = 0; i < mc; i += 16) {
            صحيح A_base_addr = (mc / 8) * (i / 8) * 16;
            لكل (int64_t j = 0; j < nc; j += 8) {
                 صحيح B_base_addr = (nc / 8) * (j / 8) * 16;
                 acc_t acc[8];
                 vec_t A0_block[16]; vec_t A1_block[16];
                 لكل (صحيح x = 0; x < 8; x++)
                     __builtin_mma_xxsetaccz(&acc[x]);
                 لكل (int64_t l = 0; l < kc; l += 8) {
                     صحيح A0_block_idx = A_base_addr + (l / 8) * 16;
                     صحيح A1_block_idx = A0_block_idx + (mc / 8) * 16;
                     صحيح B_block_idx = B_base_addr + (l / 8) * 16;
                     vec_t* A0_block = &vec_A[A0_block_idx];
                     vec_t* A1_block = &vec_A[A1_block_idx];
                     vec_t* B_block = &vec_B[B_block_idx];
                     MMA_16x8(A0_block, A1_block, B_block, acc);
                 }
                 إذا (kk == 0) {
                     save_acc(&acc[0], ii + i, jj + j);
                     save_acc(&acc[1], ii + i, jj + j + 4);
                     save_acc(&acc[2], ii + i + 4, jj + j);
                     save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     save_acc(&acc[4], ii + i + 8, jj + j);
                     save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     save_acc(&acc[6], ii + i + 12, jj + j);
                     save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 } وإلا {
                     add_save_acc(&acc[0], ii + i, jj + j);
                     add_save_acc(&acc[1], ii + i, jj + j + 4);
                     add_save_acc(&acc[2], ii + i + 4, jj + j);
                     add_save_acc(&acc[3], ii + i + 4, jj + j + 4);
                     add_save_acc(&acc[4], ii + i + 8, jj + j);
                     add_save_acc(&acc[5], ii + i + 8, jj + j + 4);
                     add_save_acc(&acc[6], ii + i + 12, jj + j);
                     add_save_acc(&acc[7], ii + i + 12, jj + j + 4);
                 }
            }
        }
    }

    فراغ matmul_tiled(int64_t m , int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles) {
            end = tiles;
        }
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = (job / xtiles) * mc;
            int64_t jj = (job % xtiles) * nc;
            لكل (int64_t kk = 0; kk < k; kk += kc) {
                 vec_t A_pack[kc * mc / 4];
                 vec_t B_pack[kc * nc / 4];
                 packTranspose(A + (ii * lda) + kk, lda, kc, mc, (عائم *)A_pack);
                 packTranspose(B + (jj * ldb) + kk, ldb, kc, nc, (عائم *)B_pack);
                 KERNEL(ii, jj, mc, nc, kc, A_pack, B_pack, kk);
            }
        }
    }

    فراغ mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        صحيح m_rem = MIN(m - m0, 8);
        صحيح n_rem = MIN(n - n0, 8);
        صحيح mc = 0, nc = 0;
        إذا (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8, 8>(m0, m, n0, n);
        } وإلا إذا (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } وإلا إذا (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } وإلا إذا (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n);
        } وإلا {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            إذا (mc == 0 || nc == 0)
                أرجع;
            gemm_small(m0, m, n0, n, mc, nc);
        }
        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    فراغ gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, صحيح RM, صحيح RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[4] = {0}, vec_B[4] = {0};
            لكل (صحيح l = 0; l < k; l += 4) {
                /* 'GEMV Forwarding' concept is used in first two conditional loops.
                 * when one of the matrix has a single row/column, the elements are
                 * broadcasted, instead of using packing routine to prepack the
                 * matrix elements.
                 */
                إذا (RM == 1) {
                    عائم * a = const_cast<عائم *>(A + (ii) * lda + l);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (عائم *)vec_B);
                    vec_A[0] = (vec_t)vec_xl(0,a);
                    vec_A[1] = (vec_t)vec_splats(*((عائم *)&vec_A+1));
                    vec_A[2] = (vec_t)vec_splats(*((عائم *)&vec_A+2));
                    vec_A[3] = (vec_t)vec_splats(*((عائم *)&vec_A+3));
                } وإلا إذا (RN == 1) {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (عائم *)vec_A);
                    عائم * b = const_cast<عائم *>(B + (jj) * ldb + l);
                    vec_B[0] = (vec_t)vec_xl(0,b);
                    vec_B[1] = (vec_t)vec_splats(*((عائم *)&vec_B+1));
                    vec_B[2] = (vec_t)vec_splats(*((عائم *)&vec_B+2));
                    vec_B[3] = (vec_t)vec_splats(*((عائم *)&vec_B+3));
                } وإلا {
                    packTranspose(A + (ii * lda) + l, lda, RM, 4, (عائم *)vec_A);
                    packTranspose(B + (jj * ldb) + l, ldb, RN, 4, (عائم *)vec_B);
                }
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            لكل (صحيح I = 0; I < RM; I++) {
                لكل (صحيح J = 0; J < RN; J++) {
                    *((عائم *)(C+ii+((jj+J)*ldc)+I)) = *((عائم *)&vec_C[I]+J);
                }
            }
       }
    }

    template<صحيح RM, صحيح RN>
    مضمن فراغ kernel(int64_t ii, int64_t jj) {
        إذا constexpr(RM == 4 && RN == 4) {
            KERNEL_4x4(ii, jj);
        } وإلا إذا constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii, jj);
        } وإلا إذا constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii, jj);
        } وإلا إذا constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii, jj);
        } وإلا {
            static_assert(خطأ, "RN/RM values not supported");
        }
    }

    template <صحيح RM, صحيح RN>
    NOINLINE فراغ gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        إذا (end > tiles)
            end = tiles;
        لكل (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    ثابت عائم * ثابت A;
    ثابت عائم * ثابت B;
    عائم * C;
    ثابت int64_t k;
    ثابت int64_t lda;
    ثابت int64_t ldb;
    ثابت int64_t ldc;
    ثابت صحيح ith;
    ثابت صحيح nth;
};
#endif
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
منطقي llamafile_sgemm(ثابت بنية ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                     ثابت فراغ *A, int64_t lda, ثابت فراغ *B, int64_t ldb, فراغ *C,
                     int64_t ldc, صحيح Atype, صحيح Btype, صحيح Ctype) {

    أكد(m >= 0);
    أكد(n >= 0);
    أكد(k >= 0);
    أكد(lda >= k);
    أكد(ldb >= k);
    أكد(ldc >= m);
    أكد(params->nth > 0);
    أكد(params->ith < params->nth);

    // only enable sgemm for prompt processing
#إذا !defined(__MMA__)
    إذا (n < 2)
        أرجع خطأ;
#endif

    إذا (Ctype != GGML_TYPE_F32)
        أرجع خطأ;

    بدل (Atype) {

    حالة GGML_TYPE_F32: {
        إذا (Btype != GGML_TYPE_F32)
            أرجع خطأ;
#إذا defined(__AVX512F__)
        tinyBLAS<16, __m512, __m512, عائم, عائم, عائم> tb{ params,
            k, (ثابت عائم *)A, lda,
            (ثابت عائم *)B, ldb,
            (عائم *)C, ldc};
        أرجع tb.matmul(m, n);
#وإلاإذا defined(__AVX__) || defined(__AVX2__)
        tinyBLAS<8, __m256, __m256, عائم, عائم, عائم> tb{ params,
            k, (ثابت عائم *)A, lda,
            (ثابت عائم *)B, ldb,
            (عائم *)C, ldc};
        أرجع tb.matmul(m, n);
#وإلاإذا defined(__ARM_NEON)
        إذا (n < 4)
            أرجع خطأ;
        tinyBLAS<4, float32x4_t, float32x4_t, عائم, عائم, عائم> tb{ params,
            k, (ثابت عائم *)A, lda,
            (ثابت عائم *)B, ldb,
            (عائم *)C, ldc};
        أرجع tb.matmul(m, n);
#وإلاإذا defined(__VXE__) || defined(__VXE2__)
        إذا (n < 4)
            أرجع خطأ;
        tinyBLAS<4, float32x4_t, float32x4_t, عائم, عائم, عائم> tb{ params,
            k, (ثابت عائم *)A, lda,
            (ثابت عائم *)B, ldb,
            (عائم *)C, ldc};
        أرجع tb.matmul(m, n);
#وإلاإذا defined(__MMA__)
        إذا (k % 8)
            أرجع خطأ;
        tinyBLAS_PPC tb{
            k, (ثابت عائم *)A, lda,
            (ثابت عائم *)B, ldb,
            (عائم *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        أرجع true;
#وإلاإذا defined(__riscv_v_intrinsic)
    #إذا LMUL == 1
        tinyBLAS_RVV<vfloat32m1_t, vfloat32m1_t, عائم, عائم, عائم> tb{ params,
            k, (ثابت عائم *)A, lda,
            (ثابت عائم *)B, ldb,
            (عائم *)C, ldc};
    #وإلاإذا LMUL == 2
        tinyBLAS_RVV<vfloat32m2_t, vfloat32m2_t, عائم, عائم, عائم> tb{ params,
            k, (ثابت عائم *)A, lda,
            (ثابت عائم *)B, ldb,
            (عائم *)C, ldc};
    #وإلا // LMUL = 4
        tinyBLAS_RVV<vfloat32m4_t, vfloat32m4_t, عائم, عائم, عائم> tb{ params,
            k, (ثابت عائم *)A, lda,
            (ثابت عائم *)B, ldb,
            (عائم *)C, ldc};
    #endif
        أرجع tb.matmul(m, n);
#وإلا
        أرجع خطأ;
#endif
    }

    حالة GGML_TYPE_BF16: {
#إذا defined(__AVX512BF16__)
        إذا (Btype == GGML_TYPE_BF16) {
            tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, عائم> tb{ params, k,
                (ثابت ggml_bf16_t *)A, lda,
                (ثابت ggml_bf16_t *)B, ldb,
                (عائم *)C, ldc};
            أرجع tb.matmul(m, n);
        }
#وإلاإذا defined(__AVX512F__)
        إذا (Btype == GGML_TYPE_BF16) {
            tinyBLAS<16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, عائم> tb{ params, k,
                (ثابت ggml_bf16_t *)A, lda,
                (ثابت ggml_bf16_t *)B, ldb,
                (عائم *)C, ldc};
            أرجع tb.matmul(m, n);
        }
#وإلاإذا defined(__AVX2__)
        إذا (Btype == GGML_TYPE_BF16) {
            tinyBLAS<8, __m256, __m256, ggml_bf16_t, ggml_bf16_t, عائم> tb{ params, k,
                (ثابت ggml_bf16_t *)A, lda,
                (ثابت ggml_bf16_t *)B, ldb,
                (عائم *)C, ldc};
            أرجع tb.matmul(m, n);
        }
#وإلاإذا defined(__MMA__)
        إذا (k % 8) {
            أرجع خطأ;
        }

        إذا (Btype == GGML_TYPE_BF16) {
            tinyBLAS_HP16_PPC<ggml_bf16_t, ggml_bf16_t, عائم> tb{ k,
                (ثابت ggml_bf16_t *)A, lda,
                (ثابت ggml_bf16_t *)B, ldb,
                (عائم *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            أرجع true;
        }
#وإلاإذا defined(__riscv_zvfbfwma)
        إذا (Btype == GGML_TYPE_BF16) {
            #إذا LMUL == 1
                tinyBLAS_RVV<vfloat32m1_t, vbfloat16mf2_t, ggml_bf16_t, ggml_bf16_t, عائم> tb{ params,
                    k, (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (عائم *)C, ldc};
            #وإلاإذا LMUL == 2
                tinyBLAS_RVV<vfloat32m2_t, vbfloat16m1_t, ggml_bf16_t, ggml_bf16_t, عائم> tb{ params,
                    k, (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (عائم *)C, ldc};
            #وإلا // LMUL = 4
                tinyBLAS_RVV<vfloat32m4_t, vbfloat16m2_t, ggml_bf16_t, ggml_bf16_t, عائم> tb{ params,
                    k, (ثابت ggml_bf16_t *)A, lda,
                    (ثابت ggml_bf16_t *)B, ldb,
                    (عائم *)C, ldc};
            #endif
                أرجع tb.matmul(m, n);
        }
#endif
        أرجع خطأ;
    }

    حالة GGML_TYPE_F16: {
#إذا defined(__AVX512F__)
        إذا (Btype == GGML_TYPE_F16) {
            tinyBLAS<16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, عائم> tb{ params, k,
                (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (عائم *)C, ldc};
            أرجع tb.matmul(m, n);
        }
#وإلاإذا (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        إذا (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, عائم> tb{ params, k,
                (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (عائم *)C, ldc};
            أرجع tb.matmul(m, n);
        }
#وإلاإذا defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
        إذا (n < 8)
            أرجع خطأ;
        إذا (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, عائم> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (عائم *)C, ldc};
            أرجع tb.matmul(m, n);
        }
#وإلاإذا defined(__ARM_NEON) && !defined(_MSC_VER)
        إذا (Btype == GGML_TYPE_F32) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, عائم, عائم> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت عائم *)B, ldb,
                (عائم *)C, ldc};
            أرجع tb.matmul(m, n);
        }
#وإلاإذا defined(__VXE__) || defined(__VXE2__)
        إذا (n < 4)
            أرجع خطأ;
        إذا (Btype == GGML_TYPE_F16) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, عائم> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (عائم *)C, ldc};
            أرجع tb.matmul(m, n);
        }
#وإلاإذا defined(__riscv_zvfh)
        إذا (Btype == GGML_TYPE_F16) {
        #إذا LMUL == 1
            tinyBLAS_RVV<vfloat32m1_t, vfloat16mf2_t, ggml_fp16_t, ggml_fp16_t, عائم> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (عائم *)C, ldc};
        #وإلاإذا LMUL == 2
            tinyBLAS_RVV<vfloat32m2_t, vfloat16m1_t, ggml_fp16_t, ggml_fp16_t, عائم> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (عائم *)C, ldc};
        #وإلا // LMUL = 4
            tinyBLAS_RVV<vfloat32m4_t, vfloat16m2_t, ggml_fp16_t, ggml_fp16_t, عائم> tb{ params,
                k, (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (عائم *)C, ldc};
        #endif
            أرجع tb.matmul(m, n);
        }
#وإلاإذا defined(__MMA__)
        إذا (k % 8) {
            أرجع خطأ;
        }

        إذا (Btype == GGML_TYPE_F16) {
            tinyBLAS_HP16_PPC<ggml_fp16_t, ggml_fp16_t, عائم> tb{ k,
                (ثابت ggml_fp16_t *)A, lda,
                (ثابت ggml_fp16_t *)B, ldb,
                (عائم *)C, ldc,
                params->ith, params->nth };

            tb.matmul(m, n);
            أرجع true;
        }
#endif
        أرجع خطأ;
    }

    حالة GGML_TYPE_Q8_0: {
        إذا (Btype != GGML_TYPE_Q8_0)
           أرجع خطأ;
#إذا defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, عائم> tb{
            k, (ثابت block_q8_0 *)A, lda,
            (ثابت block_q8_0 *)B, ldb,
            (عائم *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        أرجع true;
#وإلاإذا defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q8_0> tb{
            k, (ثابت block_q8_0 *)A, lda,
            (ثابت block_q8_0 *)B, ldb,
            (عائم *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        أرجع true;
#وإلاإذا defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        إذا (n < 8 && n != 4)
           أرجع خطأ;
        إذا (m < 8 && m != 4)
           أرجع خطأ;
        tinyBLAS_Q0_PPC<block_q8_0> tb{
            k, (ثابت block_q8_0 *)A, lda,
            (ثابت block_q8_0 *)B, ldb,
            (عائم *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        أرجع true;
#وإلا
        أرجع خطأ;
#endif
    }

    حالة GGML_TYPE_Q4_0: {
        إذا (Btype != GGML_TYPE_Q8_0)
            أرجع خطأ;
#إذا defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q4_0, block_q8_0, عائم> tb{
            k, (ثابت block_q4_0 *)A, lda,
            (ثابت block_q8_0 *)B, ldb,
            (عائم *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        أرجع true;
#وإلاإذا defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q4_0> tb{
            k, (ثابت block_q4_0 *)A, lda,
            (ثابت block_q8_0 *)B, ldb,
            (عائم *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        أرجع true;
#وإلاإذا defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        إذا (n < 8 && n != 4)
           أرجع خطأ;
        إذا (m < 8 && m != 4)
           أرجع خطأ;
        tinyBLAS_Q0_PPC<block_q4_0> tb{
            k, (ثابت block_q4_0 *)A, lda,
            (ثابت block_q8_0 *)B, ldb,
            (عائم *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        أرجع true;
#وإلا
        أرجع خطأ;
#endif
    }

    حالة GGML_TYPE_Q5_0: {
        إذا (Btype != GGML_TYPE_Q8_0)
            أرجع خطأ;
#إذا defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q5_0, block_q8_0, عائم> tb{
            k, (ثابت block_q5_0 *)A, lda,
            (ثابت block_q8_0 *)B, ldb,
            (عائم *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        أرجع true;
#وإلا
        أرجع خطأ;
#endif
    }

    حالة GGML_TYPE_IQ4_NL: {
        إذا (Btype != GGML_TYPE_Q8_0)
            أرجع خطأ;
#إذا defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_iq4_nl, block_q8_0, عائم> tb{
            k, (ثابت block_iq4_nl *)A, lda,
            (ثابت block_q8_0 *)B, ldb,
            (عائم *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        أرجع true;
#وإلا
        أرجع خطأ;
#endif
    }

    افتراضي:
        أرجع خطأ;
    }

    (فراغ)params;
    (فراغ)m;
    (فراغ)n;
    (فراغ)k;
    (فراغ)A;
    (فراغ)lda;
    (فراغ)B;
    (فراغ)ldb;
    (فراغ)C;
    (فراغ)ldc;
    (فراغ)Atype;
    (فراغ)Btype;
    (فراغ)Ctype;
}
