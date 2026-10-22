#pragma once

// GGML CPU internal header

#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <string.h> // memcpy
#অন্তর্ভাব <math.h>   // fabsf

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

কাঠামো ggml_compute_params {
    // ith = thread index, nth = number of threads
    পূর্ণসংখ্যা ith, nth;

    // work buffer for all threads
    size_t wsize;
    শূন্য * wdata;

    কাঠামো ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#যদি defined(_MSC_VER)

#define m512bh(p) p
#define m512i(p) p

#অন্যথা

#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)

#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#যদি defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#endif

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#যদি defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ifndef __SSE3__
#define __SSE3__
#endif
#ifndef __SSSE3__
#define __SSSE3__
#endif
#endif

#যদি defined(__s390x__) && defined(__VEC__)
#ifndef __VXE__
#define __VXE__
#endif  // __VXE__
#ifndef __VXE2__
#define __VXE2__
#endif  // __VXE2__
#endif  // __s390x__ && __VEC__

#যদি defined(__ARM_FEATURE_SVE) && defined(__linux__)
#অন্তর্ভাব <sys/prctl.h>
#endif

#যদি defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#যদি defined(_MSC_VER) && !defined(__clang__)
#define ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#অন্যথা
#define ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#endif // _MSC_VER

#যদি !defined(__aarch64__)

// 32-bit ARM compatibility

// vaddlvq_s16
// vpaddq_s16
// vpaddq_s32
// vaddvq_s32
// vaddvq_f32
// vmaxvq_f32
// vcvtnq_s32_f32
// vzip1_u8
// vzip2_u8

অন্তঃরেখ স্থির int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    ফেরত vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

অন্তঃরেখ স্থির int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_নিচের_s16(a), vget_উপরের_s16(a));
    int16x4_t b0 = vpadd_s16(vget_নিচের_s16(b), vget_উপরের_s16(b));
    ফেরত vcombine_s16(a0, b0);
}

অন্তঃরেখ স্থির int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_নিচের_s32(a), vget_উপরের_s32(a));
    int32x2_t b0 = vpadd_s32(vget_নিচের_s32(b), vget_উপরের_s32(b));
    ফেরত vcombine_s32(a0, b0);
}

অন্তঃরেখ স্থির int32_t vaddvq_s32(int32x4_t v) {
    ফেরত vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

অন্তঃরেখ স্থির দশমিক vaddvq_f32(float32x4_t v) {
    ফেরত vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

অন্তঃরেখ স্থির দশমিক vmaxvq_f32(float32x4_t v) {
    ফেরত
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

অন্তঃরেখ স্থির int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    ফেরত res;
}

অন্তঃরেখ স্থির uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    ফেরত res;
}

অন্তঃরেখ স্থির uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    ফেরত res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

প্রকারলেখ কাঠামো ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

অন্তঃরেখ স্থির ggml_int16x8x2_t ggml_vld1q_s16_x2(ধ্রুবক int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    ফেরত res;
}

প্রকারলেখ কাঠামো ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

অন্তঃরেখ স্থির ggml_uint8x16x2_t ggml_vld1q_u8_x2(ধ্রুবক uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    ফেরত res;
}

প্রকারলেখ কাঠামো ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

অন্তঃরেখ স্থির ggml_uint8x16x4_t ggml_vld1q_u8_x4(ধ্রুবক uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    ফেরত res;
}

প্রকারলেখ কাঠামো ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

অন্তঃরেখ স্থির ggml_int8x16x2_t ggml_vld1q_s8_x2(ধ্রুবক int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    ফেরত res;
}

প্রকারলেখ কাঠামো ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

অন্তঃরেখ স্থির ggml_int8x16x4_t ggml_vld1q_s8_x4(ধ্রুবক int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    ফেরত res;
}

// NOTE: not tested
অন্তঃরেখ স্থির int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
    int8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    ফেরত res;
}

// NOTE: not tested
অন্তঃরেখ স্থির uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
    uint8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    ফেরত res;
}

#অন্যথা

#define ggml_int16x8x2_t  int16x8x2_t
#define ggml_uint8x16x2_t uint8x16x2_t
#define ggml_uint8x16x4_t uint8x16x4_t
#define ggml_int8x16x2_t  int8x16x2_t
#define ggml_int8x16x4_t  int8x16x4_t

#define ggml_vld1q_s16_x2 vld1q_s16_x2
#define ggml_vld1q_u8_x2  vld1q_u8_x2
#define ggml_vld1q_u8_x4  vld1q_u8_x4
#define ggml_vld1q_s8_x2  vld1q_s8_x2
#define ggml_vld1q_s8_x4  vld1q_s8_x4
#define ggml_vqtbl1q_s8   vqtbl1q_s8
#define ggml_vqtbl1q_u8   vqtbl1q_u8

#endif // !defined(__aarch64__)

#যদি !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
অন্তঃরেখ স্থির int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ধ্রুবক int16x8_t p0 = vmull_s8(vget_নিচের_s8 (a), vget_নিচের_s8 (b));
    ধ্রুবক int16x8_t p1 = vmull_s8(vget_উপরের_s8(a), vget_উপরের_s8(b));

    ফেরত vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#অন্যথা

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif // !defined(__ARM_FEATURE_DOTPROD)

স্থির অন্তঃরেখ int32x4_t ggml_nvfp4_dot8(ধ্রুবক int8x8_t q4_lo, ধ্রুবক int8x8_t q8_lo,
                                         ধ্রুবক int8x8_t q4_hi, ধ্রুবক int8x8_t q8_hi) {
    ধ্রুবক int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    ধ্রুবক int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    ধ্রুবক int32x4_t যোগ_lo = vpaddlq_s16(p_lo);
    ধ্রুবক int32x4_t যোগ_hi = vpaddlq_s16(p_hi);
    ফেরত vaddq_s32(যোগ_lo, যোগ_hi);
}

#endif // defined(__ARM_NEON)

#ifdef __wasm_simd128__
#অন্তর্ভাব <wasm_simd128.h>
#endif

#ifdef __POWER9_VECTOR__
#অন্তর্ভাব <altivec.h>
#endif

#যদি defined(_MSC_VER) || defined(__MINGW32__)
#অন্তর্ভাব <intrin.h>
#নতুবা_যদি defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#অন্তর্ভাব <immintrin.h>
#endif

#ifdef __riscv_v_intrinsic
#অন্তর্ভাব <riscv_vector.h>
#endif

#যদি defined(__loongarch64)
#যদি defined(__loongarch_asx)
#অন্তর্ভাব <lasxintrin.h>
#endif
#যদি defined(__loongarch_sx)
#অন্তর্ভাব <lsxintrin.h>
#endif
#endif

#যদি defined(__VXE__) || defined(__VXE2__)
#অন্তর্ভাব <vecintrin.h>

#define vec_neg(a)    (-(a))                // Vector Negate
#define vec_add(a, b) ((a) + (b))           // Vector Add
#define vec_sub(a, b) ((a) - (b))           // Vector Subtract
#define vec_mul(a, b) ((a) * (b))           // Vector Multiply
#define vec_div(a, b) ((a) / (b))           // Vector Divide
#define vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#define vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#define vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#define vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#define vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#ifndef vec_and
#define vec_and(a, b) ((a) & (b)) // Vector AND
#endif

#ifndef vec_or
#define vec_or(a, b)  ((a) | (b)) // Vector OR
#endif

#ifndef vec_xor
#define vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#endif

প্রকারলেখ signed   অক্ষর char8x16_t  __attribute__((vector_আকার(16)));
প্রকারলেখ unsigned অক্ষর uchar8x16_t __attribute__((vector_আকার(16)));

প্রকারলেখ int8_t  int8x16_t __attribute__((vector_আকার(16)));
প্রকারলেখ int16_t int16x8_t __attribute__((vector_আকার(16)));
প্রকারলেখ int32_t int32x4_t __attribute__((vector_আকার(16)));
প্রকারলেখ int64_t int64x2_t __attribute__((vector_আকার(16)));

প্রকারলেখ uint8_t  uint8x16_t __attribute__((vector_আকার(16)));
প্রকারলেখ uint16_t uint16x8_t __attribute__((vector_আকার(16)));
প্রকারলেখ uint32_t uint32x4_t __attribute__((vector_আকার(16)));
প্রকারলেখ uint64_t uint64x2_t __attribute__((vector_আকার(16)));

প্রকারলেখ দশমিক  float32x4_t  __attribute__((vector_আকার(16)));
প্রকারলেখ দ্বিগুণ double64x2_t __attribute__((vector_আকার(16)));

প্রকারলেখ কাঠামো ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

অন্তঃরেখ স্থির ggml_uint8x16x2_t ggml_vec_xl_u8x2(ধ্রুবক uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    ফেরত res;
}

প্রকারলেখ কাঠামো ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

অন্তঃরেখ স্থির ggml_uint8x16x4_t ggml_vec_xl_u8x4(ধ্রুবক uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    ফেরত res;
}

প্রকারলেখ কাঠামো ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

অন্তঃরেখ স্থির ggml_int8x16x4_t ggml_vec_xl_s8x4(ধ্রুবক int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    ফেরত res;
}

// load 4 bytes and replicate them into all 4 word lanes
অন্তঃরেখ স্থির int8x16_t ggml_vec_xl_s8x4_splat(ধ্রুবক int8_t * ptr) {
    uint32_t tmp;
    memcpy(&tmp, ptr, আকার(tmp));
    ফেরত (int8x16_t)vec_splats(tmp);
}

প্রকারলেখ কাঠামো ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

অন্তঃরেখ স্থির ggml_int16x8x2_t ggml_vec_xl_s16x2(ধ্রুবক int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    ফেরত res;
}

// faster fp16 -> fp32 conversion using vector intrinsics
// WARNING: fp16 inf converts into 65536.0f. use this only for finite values.
অন্তঃরেখ স্থির float32x4_t ggml_vec_xl_f16x4_finite(ধ্রুবক ggml_fp16_t * x) {
    uint64_t tmp;
    memcpy(&tmp, x, আকার(tmp));

    ধ্রুবক uint64x2_t  v_x    = { tmp, tmp };
    ধ্রুবক uint32x4_t  v_xh   = vec_unpackh((uint16x8_t)v_x);
    ধ্রুবক uint32x4_t  v_bits = vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x7FFF)), 13);
    ধ্রুবক float32x4_t v_abs  = vec_mul((float32x4_t)v_bits, vec_splats(0x1.0p+112f));
    ফেরত (float32x4_t)vec_or((uint32x4_t)v_abs, vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x8000)), 16));
}

অন্তঃরেখ স্থির int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    ধ্রুবক uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    ধ্রুবক int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    ধ্রুবক int16x8_t v_abe = vec_perm(a, b, v_maske);
    ফেরত v_abo + v_abe;
}

অন্তঃরেখ স্থির int32x4_t vec_paddl_s16(int16x8_t a) {
    ধ্রুবক int16x8_t v_ones = vec_splats((int16_t)1);
    ফেরত vec_mule(a, v_ones) + vec_mulo(a, v_ones);
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
অন্তঃরেখ স্থির দশমিক vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_অস্থায়ী = v + vec_reve(v);
    ফেরত v_অস্থায়ী[0] + v_অস্থায়ী[1];
}

অন্তঃরেখ স্থির int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_অস্থায়ী = v + vec_reve(v);
    ফেরত v_অস্থায়ী[0] + v_অস্থায়ী[1];
}

অন্তঃরেখ স্থির int16x8_t ggml_vec_dot_s16(int16x8_t acc, int8x16_t a, int8x16_t b) {
    ফেরত vec_meadd(a, b, vec_moadd(a, b, acc));
}

অন্তঃরেখ স্থির int32x4_t ggml_vec_dot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ধ্রুবক int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    ফেরত acc + (vec_unpackh(p) + vec_unpackl(p));
}

#endif

#যদি defined(__loongarch_sx)
/* float type data load instructions */
স্থির __m128 __lsx_vreplfr2vr_s(ধ্রুবক দশমিক val) {
    v4f32 res = {val, val, val, val};
    ফেরত (__m128)res;
}
#endif

#যদি defined(__loongarch_asx)
স্থির __m256 __lasx_xvreplfr2vr_s(ধ্রুবক দশমিক val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    ফেরত (__m256)res;
}
#endif

// TODO: move to ggml-threading
শূন্য ggml_barrier(কাঠামো ggml_threadpool * tp);

শূন্য ggml_threadpool_chunk_set(কাঠামো ggml_threadpool * tp, পূর্ণসংখ্যা মান);
পূর্ণসংখ্যা  ggml_threadpool_chunk_add(কাঠামো ggml_threadpool * tp, পূর্ণসংখ্যা মান);

#ifdef __cplusplus
}
#endif
