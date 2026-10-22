#pragma once

// GGML CPU internal header

#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <string.h> // memcpy
#அந்தர்பாவ <math.h>   // fabsf

#ifdef __cplusplus
வெளி "C" {
#endif

கட்டமைப்பு ggml_compute_params {
    // ith = thread index, nth = number of threads
    முழு ith, nth;

    // work buffer for all threads
    size_t wsize;
    வெறுமை * wdata;

    கட்டமைப்பு ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#எனில் defined(_MSC_VER)

#define m512bh(p) p
#define m512i(p) p

#இல்லையெனில்

#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)

#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#எனில் defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#endif

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#எனில் defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ifndef __SSE3__
#define __SSE3__
#endif
#ifndef __SSSE3__
#define __SSSE3__
#endif
#endif

#எனில் defined(__s390x__) && defined(__VEC__)
#ifndef __VXE__
#define __VXE__
#endif  // __VXE__
#ifndef __VXE2__
#define __VXE2__
#endif  // __VXE2__
#endif  // __s390x__ && __VEC__

#எனில் defined(__ARM_FEATURE_SVE) && defined(__linux__)
#அந்தர்பாவ <sys/prctl.h>
#endif

#எனில் defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#எனில் defined(_MSC_VER) && !defined(__clang__)
#define ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#இல்லையெனில்
#define ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#endif // _MSC_VER

#எனில் !defined(__aarch64__)

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

உள்வரி நிலையான int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    திரும்பு vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

உள்வரி நிலையான int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_கீழ்_s16(a), vget_உயர்வு_s16(a));
    int16x4_t b0 = vpadd_s16(vget_கீழ்_s16(b), vget_உயர்வு_s16(b));
    திரும்பு vcombine_s16(a0, b0);
}

உள்வரி நிலையான int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_கீழ்_s32(a), vget_உயர்வு_s32(a));
    int32x2_t b0 = vpadd_s32(vget_கீழ்_s32(b), vget_உயர்வு_s32(b));
    திரும்பு vcombine_s32(a0, b0);
}

உள்வரி நிலையான int32_t vaddvq_s32(int32x4_t v) {
    திரும்பு vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

உள்வரி நிலையான தசம vaddvq_f32(float32x4_t v) {
    திரும்பு vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

உள்வரி நிலையான தசம vmaxvq_f32(float32x4_t v) {
    திரும்பு
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

உள்வரி நிலையான int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    திரும்பு res;
}

உள்வரி நிலையான uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    திரும்பு res;
}

உள்வரி நிலையான uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    திரும்பு res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

வகைவரை கட்டமைப்பு ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

உள்வரி நிலையான ggml_int16x8x2_t ggml_vld1q_s16_x2(மாறிலி int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    திரும்பு res;
}

வகைவரை கட்டமைப்பு ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

உள்வரி நிலையான ggml_uint8x16x2_t ggml_vld1q_u8_x2(மாறிலி uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    திரும்பு res;
}

வகைவரை கட்டமைப்பு ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

உள்வரி நிலையான ggml_uint8x16x4_t ggml_vld1q_u8_x4(மாறிலி uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    திரும்பு res;
}

வகைவரை கட்டமைப்பு ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

உள்வரி நிலையான ggml_int8x16x2_t ggml_vld1q_s8_x2(மாறிலி int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    திரும்பு res;
}

வகைவரை கட்டமைப்பு ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

உள்வரி நிலையான ggml_int8x16x4_t ggml_vld1q_s8_x4(மாறிலி int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    திரும்பு res;
}

// NOTE: not tested
உள்வரி நிலையான int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
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

    திரும்பு res;
}

// NOTE: not tested
உள்வரி நிலையான uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
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

    திரும்பு res;
}

#இல்லையெனில்

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

#எனில் !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
உள்வரி நிலையான int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    மாறிலி int16x8_t p0 = vmull_s8(vget_கீழ்_s8 (a), vget_கீழ்_s8 (b));
    மாறிலி int16x8_t p1 = vmull_s8(vget_உயர்வு_s8(a), vget_உயர்வு_s8(b));

    திரும்பு vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#இல்லையெனில்

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif // !defined(__ARM_FEATURE_DOTPROD)

நிலையான உள்வரி int32x4_t ggml_nvfp4_dot8(மாறிலி int8x8_t q4_lo, மாறிலி int8x8_t q8_lo,
                                         மாறிலி int8x8_t q4_hi, மாறிலி int8x8_t q8_hi) {
    மாறிலி int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    மாறிலி int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    மாறிலி int32x4_t sum_lo = vpaddlq_s16(p_lo);
    மாறிலி int32x4_t sum_hi = vpaddlq_s16(p_hi);
    திரும்பு vaddq_s32(sum_lo, sum_hi);
}

#endif // defined(__ARM_NEON)

#ifdef __wasm_simd128__
#அந்தர்பாவ <wasm_simd128.h>
#endif

#ifdef __POWER9_VECTOR__
#அந்தர்பாவ <altivec.h>
#endif

#எனில் defined(_MSC_VER) || defined(__MINGW32__)
#அந்தர்பாவ <intrin.h>
#அல்லதெனில் defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#அந்தர்பாவ <immintrin.h>
#endif

#ifdef __riscv_v_intrinsic
#அந்தர்பாவ <riscv_vector.h>
#endif

#எனில் defined(__loongarch64)
#எனில் defined(__loongarch_asx)
#அந்தர்பாவ <lasxintrin.h>
#endif
#எனில் defined(__loongarch_sx)
#அந்தர்பாவ <lsxintrin.h>
#endif
#endif

#எனில் defined(__VXE__) || defined(__VXE2__)
#அந்தர்பாவ <vecintrin.h>

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

வகைவரை signed   எழுத்து char8x16_t  __attribute__((vector_size(16)));
வகைவரை unsigned எழுத்து uchar8x16_t __attribute__((vector_size(16)));

வகைவரை int8_t  int8x16_t __attribute__((vector_size(16)));
வகைவரை int16_t int16x8_t __attribute__((vector_size(16)));
வகைவரை int32_t int32x4_t __attribute__((vector_size(16)));
வகைவரை int64_t int64x2_t __attribute__((vector_size(16)));

வகைவரை uint8_t  uint8x16_t __attribute__((vector_size(16)));
வகைவரை uint16_t uint16x8_t __attribute__((vector_size(16)));
வகைவரை uint32_t uint32x4_t __attribute__((vector_size(16)));
வகைவரை uint64_t uint64x2_t __attribute__((vector_size(16)));

வகைவரை தசம  float32x4_t  __attribute__((vector_size(16)));
வகைவரை இரட்டை double64x2_t __attribute__((vector_size(16)));

வகைவரை கட்டமைப்பு ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

உள்வரி நிலையான ggml_uint8x16x2_t ggml_vec_xl_u8x2(மாறிலி uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    திரும்பு res;
}

வகைவரை கட்டமைப்பு ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

உள்வரி நிலையான ggml_uint8x16x4_t ggml_vec_xl_u8x4(மாறிலி uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    திரும்பு res;
}

வகைவரை கட்டமைப்பு ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

உள்வரி நிலையான ggml_int8x16x4_t ggml_vec_xl_s8x4(மாறிலி int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    திரும்பு res;
}

// load 4 bytes and replicate them into all 4 word lanes
உள்வரி நிலையான int8x16_t ggml_vec_xl_s8x4_splat(மாறிலி int8_t * ptr) {
    uint32_t tmp;
    memcpy(&tmp, ptr, அளவு(tmp));
    திரும்பு (int8x16_t)vec_splats(tmp);
}

வகைவரை கட்டமைப்பு ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

உள்வரி நிலையான ggml_int16x8x2_t ggml_vec_xl_s16x2(மாறிலி int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    திரும்பு res;
}

// faster fp16 -> fp32 conversion using vector intrinsics
// WARNING: fp16 inf converts into 65536.0f. use this only for finite values.
உள்வரி நிலையான float32x4_t ggml_vec_xl_f16x4_finite(மாறிலி ggml_fp16_t * x) {
    uint64_t tmp;
    memcpy(&tmp, x, அளவு(tmp));

    மாறிலி uint64x2_t  v_x    = { tmp, tmp };
    மாறிலி uint32x4_t  v_xh   = vec_unpackh((uint16x8_t)v_x);
    மாறிலி uint32x4_t  v_bits = vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x7FFF)), 13);
    மாறிலி float32x4_t v_abs  = vec_mul((float32x4_t)v_bits, vec_splats(0x1.0p+112f));
    திரும்பு (float32x4_t)vec_or((uint32x4_t)v_abs, vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x8000)), 16));
}

உள்வரி நிலையான int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    மாறிலி uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    மாறிலி int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    மாறிலி int16x8_t v_abe = vec_perm(a, b, v_maske);
    திரும்பு v_abo + v_abe;
}

உள்வரி நிலையான int32x4_t vec_paddl_s16(int16x8_t a) {
    மாறிலி int16x8_t v_ones = vec_splats((int16_t)1);
    திரும்பு vec_mule(a, v_ones) + vec_mulo(a, v_ones);
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
உள்வரி நிலையான தசம vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_temp = v + vec_reve(v);
    திரும்பு v_temp[0] + v_temp[1];
}

உள்வரி நிலையான int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_temp = v + vec_reve(v);
    திரும்பு v_temp[0] + v_temp[1];
}

உள்வரி நிலையான int16x8_t ggml_vec_dot_s16(int16x8_t acc, int8x16_t a, int8x16_t b) {
    திரும்பு vec_meadd(a, b, vec_moadd(a, b, acc));
}

உள்வரி நிலையான int32x4_t ggml_vec_dot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    மாறிலி int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    திரும்பு acc + (vec_unpackh(p) + vec_unpackl(p));
}

#endif

#எனில் defined(__loongarch_sx)
/* float type data load instructions */
நிலையான __m128 __lsx_vreplfr2vr_s(மாறிலி தசம val) {
    v4f32 res = {val, val, val, val};
    திரும்பு (__m128)res;
}
#endif

#எனில் defined(__loongarch_asx)
நிலையான __m256 __lasx_xvreplfr2vr_s(மாறிலி தசம val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    திரும்பு (__m256)res;
}
#endif

// TODO: move to ggml-threading
வெறுமை ggml_barrier(கட்டமைப்பு ggml_threadpool * tp);

வெறுமை ggml_threadpool_chunk_set(கட்டமைப்பு ggml_threadpool * tp, முழு மதிப்பு);
முழு  ggml_threadpool_chunk_add(கட்டமைப்பு ggml_threadpool * tp, முழு மதிப்பு);

#ifdef __cplusplus
}
#endif
