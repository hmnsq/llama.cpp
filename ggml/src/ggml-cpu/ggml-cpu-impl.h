#ہدایت once

// GGML CPU internal header

#شامل "ggml.h"
#شامل "ggml-impl.h"

#شامل <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#شامل <stdbool.h>
#شامل <string.h> // memcpy
#شامل <math.h>   // fabsf

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

ساخت ggml_compute_params {
    // ith = thread index, nth = number of threads
    صحیح ith, nth;

    // work buffer for all threads
    size_t wsize;
    خالی * wdata;

    ساخت ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#اگر defined(_MSC_VER)

#تعریف_کرو m512bh(p) p
#تعریف_کرو m512i(p) p

#ورنہ

#تعریف_کرو m512bh(p) (__m512bh)(p)
#تعریف_کرو m512i(p) (__m512i)(p)

#اگر_ختم

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#اگر defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#اگر_غیر_تعریف_شدہ __FMA__
#تعریف_کرو __FMA__
#اگر_ختم
#اگر_غیر_تعریف_شدہ __F16C__
#تعریف_کرو __F16C__
#اگر_ختم
#اگر_ختم

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#اگر defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#اگر_غیر_تعریف_شدہ __SSE3__
#تعریف_کرو __SSE3__
#اگر_ختم
#اگر_غیر_تعریف_شدہ __SSSE3__
#تعریف_کرو __SSSE3__
#اگر_ختم
#اگر_ختم

#اگر defined(__s390x__) && defined(__VEC__)
#اگر_غیر_تعریف_شدہ __VXE__
#تعریف_کرو __VXE__
#اگر_ختم  // __VXE__
#اگر_غیر_تعریف_شدہ __VXE2__
#تعریف_کرو __VXE2__
#اگر_ختم  // __VXE2__
#اگر_ختم  // __s390x__ && __VEC__

#اگر defined(__ARM_FEATURE_SVE) && defined(__linux__)
#شامل <sys/prctl.h>
#اگر_ختم

#اگر defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#اگر defined(_MSC_VER) && !defined(__clang__)
#تعریف_کرو ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#ورنہ
#تعریف_کرو ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#اگر_ختم // _MSC_VER

#اگر !defined(__aarch64__)

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

اندرونی ساکن int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    واپس vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

اندرونی ساکن int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_کم_s16(a), vget_بلند_s16(a));
    int16x4_t b0 = vpadd_s16(vget_کم_s16(b), vget_بلند_s16(b));
    واپس vcombine_s16(a0, b0);
}

اندرونی ساکن int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_کم_s32(a), vget_بلند_s32(a));
    int32x2_t b0 = vpadd_s32(vget_کم_s32(b), vget_بلند_s32(b));
    واپس vcombine_s32(a0, b0);
}

اندرونی ساکن int32_t vaddvq_s32(int32x4_t v) {
    واپس vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

اندرونی ساکن اعشاری vaddvq_f32(float32x4_t v) {
    واپس vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

اندرونی ساکن اعشاری vmaxvq_f32(float32x4_t v) {
    واپس
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

اندرونی ساکن int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    واپس res;
}

اندرونی ساکن uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    واپس res;
}

اندرونی ساکن uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    واپس res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

قسمتعریف ساخت ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

اندرونی ساکن ggml_int16x8x2_t ggml_vld1q_s16_x2(ثابت int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    واپس res;
}

قسمتعریف ساخت ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

اندرونی ساکن ggml_uint8x16x2_t ggml_vld1q_u8_x2(ثابت uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    واپس res;
}

قسمتعریف ساخت ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

اندرونی ساکن ggml_uint8x16x4_t ggml_vld1q_u8_x4(ثابت uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    واپس res;
}

قسمتعریف ساخت ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

اندرونی ساکن ggml_int8x16x2_t ggml_vld1q_s8_x2(ثابت int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    واپس res;
}

قسمتعریف ساخت ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

اندرونی ساکن ggml_int8x16x4_t ggml_vld1q_s8_x4(ثابت int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    واپس res;
}

// NOTE: not tested
اندرونی ساکن int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
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

    واپس res;
}

// NOTE: not tested
اندرونی ساکن uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
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

    واپس res;
}

#ورنہ

#تعریف_کرو ggml_int16x8x2_t  int16x8x2_t
#تعریف_کرو ggml_uint8x16x2_t uint8x16x2_t
#تعریف_کرو ggml_uint8x16x4_t uint8x16x4_t
#تعریف_کرو ggml_int8x16x2_t  int8x16x2_t
#تعریف_کرو ggml_int8x16x4_t  int8x16x4_t

#تعریف_کرو ggml_vld1q_s16_x2 vld1q_s16_x2
#تعریف_کرو ggml_vld1q_u8_x2  vld1q_u8_x2
#تعریف_کرو ggml_vld1q_u8_x4  vld1q_u8_x4
#تعریف_کرو ggml_vld1q_s8_x2  vld1q_s8_x2
#تعریف_کرو ggml_vld1q_s8_x4  vld1q_s8_x4
#تعریف_کرو ggml_vqtbl1q_s8   vqtbl1q_s8
#تعریف_کرو ggml_vqtbl1q_u8   vqtbl1q_u8

#اگر_ختم // !defined(__aarch64__)

#اگر !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
اندرونی ساکن int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ثابت int16x8_t p0 = vmull_s8(vget_کم_s8 (a), vget_کم_s8 (b));
    ثابت int16x8_t p1 = vmull_s8(vget_بلند_s8(a), vget_بلند_s8(b));

    واپس vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#ورنہ

#تعریف_کرو ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#اگر_ختم // !defined(__ARM_FEATURE_DOTPROD)

ساکن اندرونی int32x4_t ggml_nvfp4_dot8(ثابت int8x8_t q4_lo, ثابت int8x8_t q8_lo,
                                         ثابت int8x8_t q4_hi, ثابت int8x8_t q8_hi) {
    ثابت int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    ثابت int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    ثابت int32x4_t مجموع_lo = vpaddlq_s16(p_lo);
    ثابت int32x4_t مجموع_hi = vpaddlq_s16(p_hi);
    واپس vaddq_s32(مجموع_lo, مجموع_hi);
}

#اگر_ختم // defined(__ARM_NEON)

#اگر_تعریف_شدہ __wasm_simd128__
#شامل <wasm_simd128.h>
#اگر_ختم

#اگر_تعریف_شدہ __POWER9_VECTOR__
#شامل <altivec.h>
#اگر_ختم

#اگر defined(_MSC_VER) || defined(__MINGW32__)
#شامل <intrin.h>
#وگرنہ_اگر defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#شامل <immintrin.h>
#اگر_ختم

#اگر_تعریف_شدہ __riscv_v_intrinsic
#شامل <riscv_vector.h>
#اگر_ختم

#اگر defined(__loongarch64)
#اگر defined(__loongarch_asx)
#شامل <lasxintrin.h>
#اگر_ختم
#اگر defined(__loongarch_sx)
#شامل <lsxintrin.h>
#اگر_ختم
#اگر_ختم

#اگر defined(__VXE__) || defined(__VXE2__)
#شامل <vecintrin.h>

#تعریف_کرو vec_neg(a)    (-(a))                // Vector Negate
#تعریف_کرو vec_جوڑو(a, b) ((a) + (b))           // Vector Add
#تعریف_کرو vec_sub(a, b) ((a) - (b))           // Vector Subtract
#تعریف_کرو vec_mul(a, b) ((a) * (b))           // Vector Multiply
#تعریف_کرو vec_div(a, b) ((a) / (b))           // Vector Divide
#تعریف_کرو vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#تعریف_کرو vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#تعریف_کرو vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#تعریف_کرو vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#تعریف_کرو vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#اگر_غیر_تعریف_شدہ vec_and
#تعریف_کرو vec_and(a, b) ((a) & (b)) // Vector AND
#اگر_ختم

#اگر_غیر_تعریف_شدہ vec_or
#تعریف_کرو vec_or(a, b)  ((a) | (b)) // Vector OR
#اگر_ختم

#اگر_غیر_تعریف_شدہ vec_xor
#تعریف_کرو vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#اگر_ختم

قسمتعریف علامتی   حرف char8x16_t  __attribute__((سمتیہ_سائز(16)));
قسمتعریف غیرعلامتی حرف uchar8x16_t __attribute__((سمتیہ_سائز(16)));

قسمتعریف int8_t  int8x16_t __attribute__((سمتیہ_سائز(16)));
قسمتعریف int16_t int16x8_t __attribute__((سمتیہ_سائز(16)));
قسمتعریف int32_t int32x4_t __attribute__((سمتیہ_سائز(16)));
قسمتعریف int64_t int64x2_t __attribute__((سمتیہ_سائز(16)));

قسمتعریف uint8_t  uint8x16_t __attribute__((سمتیہ_سائز(16)));
قسمتعریف uint16_t uint16x8_t __attribute__((سمتیہ_سائز(16)));
قسمتعریف uint32_t uint32x4_t __attribute__((سمتیہ_سائز(16)));
قسمتعریف uint64_t uint64x2_t __attribute__((سمتیہ_سائز(16)));

قسمتعریف اعشاری  float32x4_t  __attribute__((سمتیہ_سائز(16)));
قسمتعریف دوہرا double64x2_t __attribute__((سمتیہ_سائز(16)));

قسمتعریف ساخت ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

اندرونی ساکن ggml_uint8x16x2_t ggml_vec_xl_u8x2(ثابت uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    واپس res;
}

قسمتعریف ساخت ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

اندرونی ساکن ggml_uint8x16x4_t ggml_vec_xl_u8x4(ثابت uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    واپس res;
}

قسمتعریف ساخت ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

اندرونی ساکن ggml_int8x16x4_t ggml_vec_xl_s8x4(ثابت int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    واپس res;
}

// load 4 bytes and replicate them into all 4 word lanes
اندرونی ساکن int8x16_t ggml_vec_xl_s8x4_splat(ثابت int8_t * ptr) {
    uint32_t tmp;
    حافظہ_نقل(&tmp, ptr, حجم(tmp));
    واپس (int8x16_t)vec_splats(tmp);
}

قسمتعریف ساخت ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

اندرونی ساکن ggml_int16x8x2_t ggml_vec_xl_s16x2(ثابت int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    واپس res;
}

// faster fp16 -> fp32 conversion using vector intrinsics
// WARNING: fp16 inf converts into 65536.0f. use this only for finite values.
اندرونی ساکن float32x4_t ggml_vec_xl_f16x4_finite(ثابت ggml_fp16_t * x) {
    uint64_t tmp;
    حافظہ_نقل(&tmp, x, حجم(tmp));

    ثابت uint64x2_t  v_x    = { tmp, tmp };
    ثابت uint32x4_t  v_xh   = vec_unpackh((uint16x8_t)v_x);
    ثابت uint32x4_t  v_بٹ = vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x7FFF)), 13);
    ثابت float32x4_t v_abs  = vec_mul((float32x4_t)v_بٹ, vec_splats(0x1.0p+112f));
    واپس (float32x4_t)vec_or((uint32x4_t)v_abs, vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x8000)), 16));
}

اندرونی ساکن int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    ثابت uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    ثابت int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    ثابت int16x8_t v_abe = vec_perm(a, b, v_maske);
    واپس v_abo + v_abe;
}

اندرونی ساکن int32x4_t vec_paddl_s16(int16x8_t a) {
    ثابت int16x8_t v_ones = vec_splats((int16_t)1);
    واپس vec_mule(a, v_ones) + vec_mulo(a, v_ones);
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
اندرونی ساکن اعشاری vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_عارضی = v + vec_reve(v);
    واپس v_عارضی[0] + v_عارضی[1];
}

اندرونی ساکن int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_عارضی = v + vec_reve(v);
    واپس v_عارضی[0] + v_عارضی[1];
}

اندرونی ساکن int16x8_t ggml_vec_dot_s16(int16x8_t acc, int8x16_t a, int8x16_t b) {
    واپس vec_meadd(a, b, vec_moadd(a, b, acc));
}

اندرونی ساکن int32x4_t ggml_vec_dot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ثابت int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    واپس acc + (vec_unpackh(p) + vec_unpackl(p));
}

#اگر_ختم

#اگر defined(__loongarch_sx)
/* float type data load instructions */
ساکن __m128 __lsx_vreplfr2vr_s(ثابت اعشاری val) {
    v4f32 res = {val, val, val, val};
    واپس (__m128)res;
}
#اگر_ختم

#اگر defined(__loongarch_asx)
ساکن __m256 __lasx_xvreplfr2vr_s(ثابت اعشاری val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    واپس (__m256)res;
}
#اگر_ختم

// TODO: move to ggml-threading
خالی ggml_barrier(ساخت ggml_threadpool * tp);

خالی ggml_threadpool_chunk_سیٹ(ساخت ggml_threadpool * tp, صحیح قدر);
صحیح  ggml_threadpool_chunk_جوڑو(ساخت ggml_threadpool * tp, صحیح قدر);

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم
