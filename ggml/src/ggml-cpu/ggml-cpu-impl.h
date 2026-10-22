#pragma once

// GGML CPU internal header

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <string.h> // memcpy
#ଅନ୍ତର୍ଭାଵ <math.h>   // fabsf

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

ସଂରଚନା ggml_compute_params {
    // ith = thread index, nth = number of threads
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, nth;

    // work buffer for all threads
    size_t wsize;
    ଶୂନ୍ୟ * wdata;

    ସଂରଚନା ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#ଯଦି defined(_MSC_VER)

#define m512bh(p) p
#define m512i(p) p

#ଅନ୍ୟଥା

#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)

#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#ଯଦି defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#endif

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#ଯଦି defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ifndef __SSE3__
#define __SSE3__
#endif
#ifndef __SSSE3__
#define __SSSE3__
#endif
#endif

#ଯଦି defined(__s390x__) && defined(__VEC__)
#ifndef __VXE__
#define __VXE__
#endif  // __VXE__
#ifndef __VXE2__
#define __VXE2__
#endif  // __VXE2__
#endif  // __s390x__ && __VEC__

#ଯଦି defined(__ARM_FEATURE_SVE) && defined(__linux__)
#ଅନ୍ତର୍ଭାଵ <sys/prctl.h>
#endif

#ଯଦି defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#ଯଦି defined(_MSC_VER) && !defined(__clang__)
#define ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#ଅନ୍ୟଥା
#define ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#endif // _MSC_VER

#ଯଦି !defined(__aarch64__)

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

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    ଫେରନ୍ତୁ vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_ତଳ_s16(a), vget_ଉପର_s16(a));
    int16x4_t b0 = vpadd_s16(vget_ତଳ_s16(b), vget_ଉପର_s16(b));
    ଫେରନ୍ତୁ vcombine_s16(a0, b0);
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_ତଳ_s32(a), vget_ଉପର_s32(a));
    int32x2_t b0 = vpadd_s32(vget_ତଳ_s32(b), vget_ଉପର_s32(b));
    ଫେରନ୍ତୁ vcombine_s32(a0, b0);
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int32_t vaddvq_s32(int32x4_t v) {
    ଫେରନ୍ତୁ vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଦଶମିକ vaddvq_f32(float32x4_t v) {
    ଫେରନ୍ତୁ vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଦଶମିକ vmaxvq_f32(float32x4_t v) {
    ଫେରନ୍ତୁ
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    ଫେରନ୍ତୁ res;
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    ଫେରନ୍ତୁ res;
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    ଫେରନ୍ତୁ res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_int16x8x2_t ggml_vld1q_s16_x2(ଅଚଳ int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    ଫେରନ୍ତୁ res;
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_uint8x16x2_t ggml_vld1q_u8_x2(ଅଚଳ uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    ଫେରନ୍ତୁ res;
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_uint8x16x4_t ggml_vld1q_u8_x4(ଅଚଳ uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    ଫେରନ୍ତୁ res;
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_int8x16x2_t ggml_vld1q_s8_x2(ଅଚଳ int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    ଫେରନ୍ତୁ res;
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_int8x16x4_t ggml_vld1q_s8_x4(ଅଚଳ int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    ଫେରନ୍ତୁ res;
}

// NOTE: not tested
ଧାଡ଼ିଭିତରେ ସ୍ଥିର int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
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

    ଫେରନ୍ତୁ res;
}

// NOTE: not tested
ଧାଡ଼ିଭିତରେ ସ୍ଥିର uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
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

    ଫେରନ୍ତୁ res;
}

#ଅନ୍ୟଥା

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

#ଯଦି !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
ଧାଡ଼ିଭିତରେ ସ୍ଥିର int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ଅଚଳ int16x8_t p0 = vmull_s8(vget_ତଳ_s8 (a), vget_ତଳ_s8 (b));
    ଅଚଳ int16x8_t p1 = vmull_s8(vget_ଉପର_s8(a), vget_ଉପର_s8(b));

    ଫେରନ୍ତୁ vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#ଅନ୍ୟଥା

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif // !defined(__ARM_FEATURE_DOTPROD)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int32x4_t ggml_nvfp4_dot8(ଅଚଳ int8x8_t q4_lo, ଅଚଳ int8x8_t q8_lo,
                                         ଅଚଳ int8x8_t q4_hi, ଅଚଳ int8x8_t q8_hi) {
    ଅଚଳ int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    ଅଚଳ int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    ଅଚଳ int32x4_t sum_lo = vpaddlq_s16(p_lo);
    ଅଚଳ int32x4_t sum_hi = vpaddlq_s16(p_hi);
    ଫେରନ୍ତୁ vaddq_s32(sum_lo, sum_hi);
}

#endif // defined(__ARM_NEON)

#ifdef __wasm_simd128__
#ଅନ୍ତର୍ଭାଵ <wasm_simd128.h>
#endif

#ifdef __POWER9_VECTOR__
#ଅନ୍ତର୍ଭାଵ <altivec.h>
#endif

#ଯଦି defined(_MSC_VER) || defined(__MINGW32__)
#ଅନ୍ତର୍ଭାଵ <intrin.h>
#ନଚେତଯଦି defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#ଅନ୍ତର୍ଭାଵ <immintrin.h>
#endif

#ifdef __riscv_v_intrinsic
#ଅନ୍ତର୍ଭାଵ <riscv_vector.h>
#endif

#ଯଦି defined(__loongarch64)
#ଯଦି defined(__loongarch_asx)
#ଅନ୍ତର୍ଭାଵ <lasxintrin.h>
#endif
#ଯଦି defined(__loongarch_sx)
#ଅନ୍ତର୍ଭାଵ <lsxintrin.h>
#endif
#endif

#ଯଦି defined(__VXE__) || defined(__VXE2__)
#ଅନ୍ତର୍ଭାଵ <vecintrin.h>

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

ପ୍ରକାରବ୍ୟାଖ୍ୟା signed   ଅକ୍ଷର char8x16_t  __attribute__((vector_size(16)));
ପ୍ରକାରବ୍ୟାଖ୍ୟା unsigned ଅକ୍ଷର uchar8x16_t __attribute__((vector_size(16)));

ପ୍ରକାରବ୍ୟାଖ୍ୟା int8_t  int8x16_t __attribute__((vector_size(16)));
ପ୍ରକାରବ୍ୟାଖ୍ୟା int16_t int16x8_t __attribute__((vector_size(16)));
ପ୍ରକାରବ୍ୟାଖ୍ୟା int32_t int32x4_t __attribute__((vector_size(16)));
ପ୍ରକାରବ୍ୟାଖ୍ୟା int64_t int64x2_t __attribute__((vector_size(16)));

ପ୍ରକାରବ୍ୟାଖ୍ୟା uint8_t  uint8x16_t __attribute__((vector_size(16)));
ପ୍ରକାରବ୍ୟାଖ୍ୟା uint16_t uint16x8_t __attribute__((vector_size(16)));
ପ୍ରକାରବ୍ୟାଖ୍ୟା uint32_t uint32x4_t __attribute__((vector_size(16)));
ପ୍ରକାରବ୍ୟାଖ୍ୟା uint64_t uint64x2_t __attribute__((vector_size(16)));

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଦଶମିକ  float32x4_t  __attribute__((vector_size(16)));
ପ୍ରକାରବ୍ୟାଖ୍ୟା ଦ୍ୱିଗୁଣ double64x2_t __attribute__((vector_size(16)));

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_uint8x16x2_t ggml_vec_xl_u8x2(ଅଚଳ uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    ଫେରନ୍ତୁ res;
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_uint8x16x4_t ggml_vec_xl_u8x4(ଅଚଳ uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    ଫେରନ୍ତୁ res;
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_int8x16x4_t ggml_vec_xl_s8x4(ଅଚଳ int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    ଫେରନ୍ତୁ res;
}

// load 4 bytes and replicate them into all 4 word lanes
ଧାଡ଼ିଭିତରେ ସ୍ଥିର int8x16_t ggml_vec_xl_s8x4_splat(ଅଚଳ int8_t * ptr) {
    uint32_t tmp;
    memcpy(&tmp, ptr, ଆକାର(tmp));
    ଫେରନ୍ତୁ (int8x16_t)vec_splats(tmp);
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_int16x8x2_t ggml_vec_xl_s16x2(ଅଚଳ int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    ଫେରନ୍ତୁ res;
}

// faster fp16 -> fp32 conversion using vector intrinsics
// WARNING: fp16 inf converts into 65536.0f. use this only for finite values.
ଧାଡ଼ିଭିତରେ ସ୍ଥିର float32x4_t ggml_vec_xl_f16x4_finite(ଅଚଳ ggml_fp16_t * x) {
    uint64_t tmp;
    memcpy(&tmp, x, ଆକାର(tmp));

    ଅଚଳ uint64x2_t  v_x    = { tmp, tmp };
    ଅଚଳ uint32x4_t  v_xh   = vec_unpackh((uint16x8_t)v_x);
    ଅଚଳ uint32x4_t  v_bits = vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x7FFF)), 13);
    ଅଚଳ float32x4_t v_abs  = vec_mul((float32x4_t)v_bits, vec_splats(0x1.0p+112f));
    ଫେରନ୍ତୁ (float32x4_t)vec_or((uint32x4_t)v_abs, vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x8000)), 16));
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    ଅଚଳ uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    ଅଚଳ int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    ଅଚଳ int16x8_t v_abe = vec_perm(a, b, v_maske);
    ଫେରନ୍ତୁ v_abo + v_abe;
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int32x4_t vec_paddl_s16(int16x8_t a) {
    ଅଚଳ int16x8_t v_ones = vec_splats((int16_t)1);
    ଫେରନ୍ତୁ vec_mule(a, v_ones) + vec_mulo(a, v_ones);
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଦଶମିକ vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_temp = v + vec_reve(v);
    ଫେରନ୍ତୁ v_temp[0] + v_temp[1];
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_temp = v + vec_reve(v);
    ଫେରନ୍ତୁ v_temp[0] + v_temp[1];
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int16x8_t ggml_vec_dot_s16(int16x8_t acc, int8x16_t a, int8x16_t b) {
    ଫେରନ୍ତୁ vec_meadd(a, b, vec_moadd(a, b, acc));
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର int32x4_t ggml_vec_dot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ଅଚଳ int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    ଫେରନ୍ତୁ acc + (vec_unpackh(p) + vec_unpackl(p));
}

#endif

#ଯଦି defined(__loongarch_sx)
/* float type data load instructions */
ସ୍ଥିର __m128 __lsx_vreplfr2vr_s(ଅଚଳ ଦଶମିକ val) {
    v4f32 res = {val, val, val, val};
    ଫେରନ୍ତୁ (__m128)res;
}
#endif

#ଯଦି defined(__loongarch_asx)
ସ୍ଥିର __m256 __lasx_xvreplfr2vr_s(ଅଚଳ ଦଶମିକ val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    ଫେରନ୍ତୁ (__m256)res;
}
#endif

// TODO: move to ggml-threading
ଶୂନ୍ୟ ggml_barrier(ସଂରଚନା ggml_threadpool * tp);

ଶୂନ୍ୟ ggml_threadpool_chunk_set(ସଂରଚନା ggml_threadpool * tp, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ମୂଲ୍ୟ);
ପୂର୍ଣ୍ଣସଂଖ୍ୟା  ggml_threadpool_chunk_add(ସଂରଚନା ggml_threadpool * tp, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ମୂଲ୍ୟ);

#ifdef __cplusplus
}
#endif
