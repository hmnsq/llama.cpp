#pragma once

// GGML CPU internal header

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <string.h> // memcpy
#અન્તર્ભાવ <math.h>   // fabsf

#ifdef __cplusplus
બાહ્ય "C" {
#endif

સંરચના ggml_compute_params {
    // ith = thread index, nth = number of threads
    પૂર્ણાંક ith, nth;

    // work buffer for all threads
    size_t wsize;
    શૂન્ય * wdata;

    સંરચના ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#જો defined(_MSC_VER)

#define m512bh(p) p
#define m512i(p) p

#નહીંતર

#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)

#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#જો defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#endif

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#જો defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ifndef __SSE3__
#define __SSE3__
#endif
#ifndef __SSSE3__
#define __SSSE3__
#endif
#endif

#જો defined(__s390x__) && defined(__VEC__)
#ifndef __VXE__
#define __VXE__
#endif  // __VXE__
#ifndef __VXE2__
#define __VXE2__
#endif  // __VXE2__
#endif  // __s390x__ && __VEC__

#જો defined(__ARM_FEATURE_SVE) && defined(__linux__)
#અન્તર્ભાવ <sys/prctl.h>
#endif

#જો defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#જો defined(_MSC_VER) && !defined(__clang__)
#define ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#નહીંતર
#define ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#endif // _MSC_VER

#જો !defined(__aarch64__)

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

અંતઃરેખ સ્થિર int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    પાછા vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

અંતઃરેખ સ્થિર int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_નીચલું_s16(a), vget_ઉપલું_s16(a));
    int16x4_t b0 = vpadd_s16(vget_નીચલું_s16(b), vget_ઉપલું_s16(b));
    પાછા vcombine_s16(a0, b0);
}

અંતઃરેખ સ્થિર int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_નીચલું_s32(a), vget_ઉપલું_s32(a));
    int32x2_t b0 = vpadd_s32(vget_નીચલું_s32(b), vget_ઉપલું_s32(b));
    પાછા vcombine_s32(a0, b0);
}

અંતઃરેખ સ્થિર int32_t vaddvq_s32(int32x4_t v) {
    પાછા vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

અંતઃરેખ સ્થિર દશાંશ vaddvq_f32(float32x4_t v) {
    પાછા vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

અંતઃરેખ સ્થિર દશાંશ vmaxvq_f32(float32x4_t v) {
    પાછા
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

અંતઃરેખ સ્થિર int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    પાછા res;
}

અંતઃરેખ સ્થિર uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    પાછા res;
}

અંતઃરેખ સ્થિર uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    પાછા res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

પ્રકારવ્યાખ્યા સંરચના ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

અંતઃરેખ સ્થિર ggml_int16x8x2_t ggml_vld1q_s16_x2(અચલ int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    પાછા res;
}

પ્રકારવ્યાખ્યા સંરચના ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

અંતઃરેખ સ્થિર ggml_uint8x16x2_t ggml_vld1q_u8_x2(અચલ uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    પાછા res;
}

પ્રકારવ્યાખ્યા સંરચના ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

અંતઃરેખ સ્થિર ggml_uint8x16x4_t ggml_vld1q_u8_x4(અચલ uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    પાછા res;
}

પ્રકારવ્યાખ્યા સંરચના ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

અંતઃરેખ સ્થિર ggml_int8x16x2_t ggml_vld1q_s8_x2(અચલ int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    પાછા res;
}

પ્રકારવ્યાખ્યા સંરચના ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

અંતઃરેખ સ્થિર ggml_int8x16x4_t ggml_vld1q_s8_x4(અચલ int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    પાછા res;
}

// NOTE: not tested
અંતઃરેખ સ્થિર int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
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

    પાછા res;
}

// NOTE: not tested
અંતઃરેખ સ્થિર uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
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

    પાછા res;
}

#નહીંતર

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

#જો !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
અંતઃરેખ સ્થિર int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    અચલ int16x8_t p0 = vmull_s8(vget_નીચલું_s8 (a), vget_નીચલું_s8 (b));
    અચલ int16x8_t p1 = vmull_s8(vget_ઉપલું_s8(a), vget_ઉપલું_s8(b));

    પાછા vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#નહીંતર

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif // !defined(__ARM_FEATURE_DOTPROD)

સ્થિર અંતઃરેખ int32x4_t ggml_nvfp4_dot8(અચલ int8x8_t q4_lo, અચલ int8x8_t q8_lo,
                                         અચલ int8x8_t q4_hi, અચલ int8x8_t q8_hi) {
    અચલ int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    અચલ int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    અચલ int32x4_t sum_lo = vpaddlq_s16(p_lo);
    અચલ int32x4_t sum_hi = vpaddlq_s16(p_hi);
    પાછા vaddq_s32(sum_lo, sum_hi);
}

#endif // defined(__ARM_NEON)

#ifdef __wasm_simd128__
#અન્તર્ભાવ <wasm_simd128.h>
#endif

#ifdef __POWER9_VECTOR__
#અન્તર્ભાવ <altivec.h>
#endif

#જો defined(_MSC_VER) || defined(__MINGW32__)
#અન્તર્ભાવ <intrin.h>
#અથવાજો defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#અન્તર્ભાવ <immintrin.h>
#endif

#ifdef __riscv_v_intrinsic
#અન્તર્ભાવ <riscv_vector.h>
#endif

#જો defined(__loongarch64)
#જો defined(__loongarch_asx)
#અન્તર્ભાવ <lasxintrin.h>
#endif
#જો defined(__loongarch_sx)
#અન્તર્ભાવ <lsxintrin.h>
#endif
#endif

#જો defined(__VXE__) || defined(__VXE2__)
#અન્તર્ભાવ <vecintrin.h>

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

પ્રકારવ્યાખ્યા signed   અક્ષર char8x16_t  __attribute__((vector_size(16)));
પ્રકારવ્યાખ્યા unsigned અક્ષર uchar8x16_t __attribute__((vector_size(16)));

પ્રકારવ્યાખ્યા int8_t  int8x16_t __attribute__((vector_size(16)));
પ્રકારવ્યાખ્યા int16_t int16x8_t __attribute__((vector_size(16)));
પ્રકારવ્યાખ્યા int32_t int32x4_t __attribute__((vector_size(16)));
પ્રકારવ્યાખ્યા int64_t int64x2_t __attribute__((vector_size(16)));

પ્રકારવ્યાખ્યા uint8_t  uint8x16_t __attribute__((vector_size(16)));
પ્રકારવ્યાખ્યા uint16_t uint16x8_t __attribute__((vector_size(16)));
પ્રકારવ્યાખ્યા uint32_t uint32x4_t __attribute__((vector_size(16)));
પ્રકારવ્યાખ્યા uint64_t uint64x2_t __attribute__((vector_size(16)));

પ્રકારવ્યાખ્યા દશાંશ  float32x4_t  __attribute__((vector_size(16)));
પ્રકારવ્યાખ્યા દ્વિગુણ double64x2_t __attribute__((vector_size(16)));

પ્રકારવ્યાખ્યા સંરચના ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

અંતઃરેખ સ્થિર ggml_uint8x16x2_t ggml_vec_xl_u8x2(અચલ uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    પાછા res;
}

પ્રકારવ્યાખ્યા સંરચના ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

અંતઃરેખ સ્થિર ggml_uint8x16x4_t ggml_vec_xl_u8x4(અચલ uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    પાછા res;
}

પ્રકારવ્યાખ્યા સંરચના ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

અંતઃરેખ સ્થિર ggml_int8x16x4_t ggml_vec_xl_s8x4(અચલ int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    પાછા res;
}

// load 4 bytes and replicate them into all 4 word lanes
અંતઃરેખ સ્થિર int8x16_t ggml_vec_xl_s8x4_splat(અચલ int8_t * ptr) {
    uint32_t tmp;
    memcpy(&tmp, ptr, કદ(tmp));
    પાછા (int8x16_t)vec_splats(tmp);
}

પ્રકારવ્યાખ્યા સંરચના ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

અંતઃરેખ સ્થિર ggml_int16x8x2_t ggml_vec_xl_s16x2(અચલ int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    પાછા res;
}

// faster fp16 -> fp32 conversion using vector intrinsics
// WARNING: fp16 inf converts into 65536.0f. use this only for finite values.
અંતઃરેખ સ્થિર float32x4_t ggml_vec_xl_f16x4_finite(અચલ ggml_fp16_t * x) {
    uint64_t tmp;
    memcpy(&tmp, x, કદ(tmp));

    અચલ uint64x2_t  v_x    = { tmp, tmp };
    અચલ uint32x4_t  v_xh   = vec_unpackh((uint16x8_t)v_x);
    અચલ uint32x4_t  v_bits = vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x7FFF)), 13);
    અચલ float32x4_t v_abs  = vec_mul((float32x4_t)v_bits, vec_splats(0x1.0p+112f));
    પાછા (float32x4_t)vec_or((uint32x4_t)v_abs, vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x8000)), 16));
}

અંતઃરેખ સ્થિર int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    અચલ uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    અચલ int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    અચલ int16x8_t v_abe = vec_perm(a, b, v_maske);
    પાછા v_abo + v_abe;
}

અંતઃરેખ સ્થિર int32x4_t vec_paddl_s16(int16x8_t a) {
    અચલ int16x8_t v_ones = vec_splats((int16_t)1);
    પાછા vec_mule(a, v_ones) + vec_mulo(a, v_ones);
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
અંતઃરેખ સ્થિર દશાંશ vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_temp = v + vec_reve(v);
    પાછા v_temp[0] + v_temp[1];
}

અંતઃરેખ સ્થિર int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_temp = v + vec_reve(v);
    પાછા v_temp[0] + v_temp[1];
}

અંતઃરેખ સ્થિર int16x8_t ggml_vec_dot_s16(int16x8_t acc, int8x16_t a, int8x16_t b) {
    પાછા vec_meadd(a, b, vec_moadd(a, b, acc));
}

અંતઃરેખ સ્થિર int32x4_t ggml_vec_dot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    અચલ int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    પાછા acc + (vec_unpackh(p) + vec_unpackl(p));
}

#endif

#જો defined(__loongarch_sx)
/* float type data load instructions */
સ્થિર __m128 __lsx_vreplfr2vr_s(અચલ દશાંશ val) {
    v4f32 res = {val, val, val, val};
    પાછા (__m128)res;
}
#endif

#જો defined(__loongarch_asx)
સ્થિર __m256 __lasx_xvreplfr2vr_s(અચલ દશાંશ val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    પાછા (__m256)res;
}
#endif

// TODO: move to ggml-threading
શૂન્ય ggml_barrier(સંરચના ggml_threadpool * tp);

શૂન્ય ggml_threadpool_chunk_set(સંરચના ggml_threadpool * tp, પૂર્ણાંક કિંમત);
પૂર્ણાંક  ggml_threadpool_chunk_add(સંરચના ggml_threadpool * tp, પૂર્ણાંક કિંમત);

#ifdef __cplusplus
}
#endif
