#ਹਦਾਇਤ once

// GGML CPU internal header

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h> // memcpy
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>   // fabsf

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

ਬਣਤਰ ggml_compute_params {
    // ith = thread index, nth = number of threads
    ਪੂਰਨ ith, nth;

    // work buffer for all threads
    size_t wsize;
    ਖਾਲੀ * wdata;

    ਬਣਤਰ ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#ਜੇ defined(_MSC_VER)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ m512bh(p) p
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ m512i(p) p

#ਨਹੀਂਤਾਂ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ m512bh(p) (__m512bh)(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ m512i(p) (__m512i)(p)

#ਜੇ_ਅੰਤ

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#ਜੇ defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ __FMA__
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __FMA__
#ਜੇ_ਅੰਤ
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ __F16C__
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __F16C__
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#ਜੇ defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ __SSE3__
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __SSE3__
#ਜੇ_ਅੰਤ
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ __SSSE3__
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __SSSE3__
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ

#ਜੇ defined(__s390x__) && defined(__VEC__)
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ __VXE__
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __VXE__
#ਜੇ_ਅੰਤ  // __VXE__
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ __VXE2__
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __VXE2__
#ਜੇ_ਅੰਤ  // __VXE2__
#ਜੇ_ਅੰਤ  // __s390x__ && __VEC__

#ਜੇ defined(__ARM_FEATURE_SVE) && defined(__linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/prctl.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#ਜੇ defined(_MSC_VER) && !defined(__clang__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#ਜੇ_ਅੰਤ // _MSC_VER

#ਜੇ !defined(__aarch64__)

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

ਸਤਰਅੰਦਰ ਸਥਿਰ int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    ਵਾਪਸ vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

ਸਤਰਅੰਦਰ ਸਥਿਰ int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_ਨੀਵਾਂ_s16(a), vget_ਉੱਚਾ_s16(a));
    int16x4_t b0 = vpadd_s16(vget_ਨੀਵਾਂ_s16(b), vget_ਉੱਚਾ_s16(b));
    ਵਾਪਸ vcombine_s16(a0, b0);
}

ਸਤਰਅੰਦਰ ਸਥਿਰ int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_ਨੀਵਾਂ_s32(a), vget_ਉੱਚਾ_s32(a));
    int32x2_t b0 = vpadd_s32(vget_ਨੀਵਾਂ_s32(b), vget_ਉੱਚਾ_s32(b));
    ਵਾਪਸ vcombine_s32(a0, b0);
}

ਸਤਰਅੰਦਰ ਸਥਿਰ int32_t vaddvq_s32(int32x4_t v) {
    ਵਾਪਸ vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

ਸਤਰਅੰਦਰ ਸਥਿਰ ਦਸ਼ਮਲਵ vaddvq_f32(float32x4_t v) {
    ਵਾਪਸ vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

ਸਤਰਅੰਦਰ ਸਥਿਰ ਦਸ਼ਮਲਵ vmaxvq_f32(float32x4_t v) {
    ਵਾਪਸ
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

ਸਤਰਅੰਦਰ ਸਥਿਰ int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    ਵਾਪਸ res;
}

ਸਤਰਅੰਦਰ ਸਥਿਰ uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    ਵਾਪਸ res;
}

ਸਤਰਅੰਦਰ ਸਥਿਰ uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    ਵਾਪਸ res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

ਸਤਰਅੰਦਰ ਸਥਿਰ ggml_int16x8x2_t ggml_vld1q_s16_x2(ਅਚਲ int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    ਵਾਪਸ res;
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

ਸਤਰਅੰਦਰ ਸਥਿਰ ggml_uint8x16x2_t ggml_vld1q_u8_x2(ਅਚਲ uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    ਵਾਪਸ res;
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

ਸਤਰਅੰਦਰ ਸਥਿਰ ggml_uint8x16x4_t ggml_vld1q_u8_x4(ਅਚਲ uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    ਵਾਪਸ res;
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

ਸਤਰਅੰਦਰ ਸਥਿਰ ggml_int8x16x2_t ggml_vld1q_s8_x2(ਅਚਲ int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    ਵਾਪਸ res;
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

ਸਤਰਅੰਦਰ ਸਥਿਰ ggml_int8x16x4_t ggml_vld1q_s8_x4(ਅਚਲ int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    ਵਾਪਸ res;
}

// NOTE: not tested
ਸਤਰਅੰਦਰ ਸਥਿਰ int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
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

    ਵਾਪਸ res;
}

// NOTE: not tested
ਸਤਰਅੰਦਰ ਸਥਿਰ uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
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

    ਵਾਪਸ res;
}

#ਨਹੀਂਤਾਂ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_int16x8x2_t  int16x8x2_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_uint8x16x2_t uint8x16x2_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_uint8x16x4_t uint8x16x4_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_int8x16x2_t  int8x16x2_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_int8x16x4_t  int8x16x4_t

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vld1q_s16_x2 vld1q_s16_x2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vld1q_u8_x2  vld1q_u8_x2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vld1q_u8_x4  vld1q_u8_x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vld1q_s8_x2  vld1q_s8_x2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vld1q_s8_x4  vld1q_s8_x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vqtbl1q_s8   vqtbl1q_s8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vqtbl1q_u8   vqtbl1q_u8

#ਜੇ_ਅੰਤ // !defined(__aarch64__)

#ਜੇ !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
ਸਤਰਅੰਦਰ ਸਥਿਰ int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ਅਚਲ int16x8_t p0 = vmull_s8(vget_ਨੀਵਾਂ_s8 (a), vget_ਨੀਵਾਂ_s8 (b));
    ਅਚਲ int16x8_t p1 = vmull_s8(vget_ਉੱਚਾ_s8(a), vget_ਉੱਚਾ_s8(b));

    ਵਾਪਸ vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#ਨਹੀਂਤਾਂ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#ਜੇ_ਅੰਤ // !defined(__ARM_FEATURE_DOTPROD)

ਸਥਿਰ ਸਤਰਅੰਦਰ int32x4_t ggml_nvfp4_dot8(ਅਚਲ int8x8_t q4_lo, ਅਚਲ int8x8_t q8_lo,
                                         ਅਚਲ int8x8_t q4_hi, ਅਚਲ int8x8_t q8_hi) {
    ਅਚਲ int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    ਅਚਲ int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    ਅਚਲ int32x4_t ਜੋੜ_lo = vpaddlq_s16(p_lo);
    ਅਚਲ int32x4_t ਜੋੜ_hi = vpaddlq_s16(p_hi);
    ਵਾਪਸ vaddq_s32(ਜੋੜ_lo, ਜੋੜ_hi);
}

#ਜੇ_ਅੰਤ // defined(__ARM_NEON)

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __wasm_simd128__
#ਅਨ੍ਤਰ੍ਭਾਵ <wasm_simd128.h>
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __POWER9_VECTOR__
#ਅਨ੍ਤਰ੍ਭਾਵ <altivec.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(_MSC_VER) || defined(__MINGW32__)
#ਅਨ੍ਤਰ੍ਭਾਵ <intrin.h>
#ਜਾਂਜੇ defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#ਅਨ੍ਤਰ੍ਭਾਵ <immintrin.h>
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __riscv_v_intrinsic
#ਅਨ੍ਤਰ੍ਭਾਵ <riscv_vector.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(__loongarch64)
#ਜੇ defined(__loongarch_asx)
#ਅਨ੍ਤਰ੍ਭਾਵ <lasxintrin.h>
#ਜੇ_ਅੰਤ
#ਜੇ defined(__loongarch_sx)
#ਅਨ੍ਤਰ੍ਭਾਵ <lsxintrin.h>
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ

#ਜੇ defined(__VXE__) || defined(__VXE2__)
#ਅਨ੍ਤਰ੍ਭਾਵ <vecintrin.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_neg(a)    (-(a))                // Vector Negate
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_ਜੋੜੋ(a, b) ((a) + (b))           // Vector Add
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_sub(a, b) ((a) - (b))           // Vector Subtract
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_mul(a, b) ((a) * (b))           // Vector Multiply
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_div(a, b) ((a) / (b))           // Vector Divide
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ vec_and
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_and(a, b) ((a) & (b)) // Vector AND
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ vec_or
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_or(a, b)  ((a) | (b)) // Vector OR
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ vec_xor
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਨਿਸ਼ਾਨੀਵਾਲਾ   ਅੱਖਰ char8x16_t  __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ uchar8x16_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));

ਕਿਸਮਪਰਿਭਾਸ਼ਾ int8_t  int8x16_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));
ਕਿਸਮਪਰਿਭਾਸ਼ਾ int16_t int16x8_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));
ਕਿਸਮਪਰਿਭਾਸ਼ਾ int32_t int32x4_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));
ਕਿਸਮਪਰਿਭਾਸ਼ਾ int64_t int64x2_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));

ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint8_t  uint8x16_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));
ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint16_t uint16x8_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));
ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint32_t uint32x4_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));
ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint64_t uint64x2_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਦਸ਼ਮਲਵ  float32x4_t  __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਦੁੱਗਣਾ double64x2_t __attribute__((ਸਦਿਸ਼_ਮਾਪ(16)));

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

ਸਤਰਅੰਦਰ ਸਥਿਰ ggml_uint8x16x2_t ggml_vec_xl_u8x2(ਅਚਲ uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    ਵਾਪਸ res;
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

ਸਤਰਅੰਦਰ ਸਥਿਰ ggml_uint8x16x4_t ggml_vec_xl_u8x4(ਅਚਲ uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    ਵਾਪਸ res;
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

ਸਤਰਅੰਦਰ ਸਥਿਰ ggml_int8x16x4_t ggml_vec_xl_s8x4(ਅਚਲ int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    ਵਾਪਸ res;
}

// load 4 bytes and replicate them into all 4 word lanes
ਸਤਰਅੰਦਰ ਸਥਿਰ int8x16_t ggml_vec_xl_s8x4_splat(ਅਚਲ int8_t * ptr) {
    uint32_t tmp;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&tmp, ptr, ਆਕਾਰ(tmp));
    ਵਾਪਸ (int8x16_t)vec_splats(tmp);
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

ਸਤਰਅੰਦਰ ਸਥਿਰ ggml_int16x8x2_t ggml_vec_xl_s16x2(ਅਚਲ int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    ਵਾਪਸ res;
}

// faster fp16 -> fp32 conversion using vector intrinsics
// WARNING: fp16 inf converts into 65536.0f. use this only for finite values.
ਸਤਰਅੰਦਰ ਸਥਿਰ float32x4_t ggml_vec_xl_f16x4_finite(ਅਚਲ ggml_fp16_t * x) {
    uint64_t tmp;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&tmp, x, ਆਕਾਰ(tmp));

    ਅਚਲ uint64x2_t  v_x    = { tmp, tmp };
    ਅਚਲ uint32x4_t  v_xh   = vec_unpackh((uint16x8_t)v_x);
    ਅਚਲ uint32x4_t  v_ਬਿੱਟ = vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x7FFF)), 13);
    ਅਚਲ float32x4_t v_abs  = vec_mul((float32x4_t)v_ਬਿੱਟ, vec_splats(0x1.0p+112f));
    ਵਾਪਸ (float32x4_t)vec_or((uint32x4_t)v_abs, vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x8000)), 16));
}

ਸਤਰਅੰਦਰ ਸਥਿਰ int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    ਅਚਲ uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    ਅਚਲ int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    ਅਚਲ int16x8_t v_abe = vec_perm(a, b, v_maske);
    ਵਾਪਸ v_abo + v_abe;
}

ਸਤਰਅੰਦਰ ਸਥਿਰ int32x4_t vec_paddl_s16(int16x8_t a) {
    ਅਚਲ int16x8_t v_ones = vec_splats((int16_t)1);
    ਵਾਪਸ vec_mule(a, v_ones) + vec_mulo(a, v_ones);
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
ਸਤਰਅੰਦਰ ਸਥਿਰ ਦਸ਼ਮਲਵ vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_ਆਰਜ਼ੀ = v + vec_reve(v);
    ਵਾਪਸ v_ਆਰਜ਼ੀ[0] + v_ਆਰਜ਼ੀ[1];
}

ਸਤਰਅੰਦਰ ਸਥਿਰ int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_ਆਰਜ਼ੀ = v + vec_reve(v);
    ਵਾਪਸ v_ਆਰਜ਼ੀ[0] + v_ਆਰਜ਼ੀ[1];
}

ਸਤਰਅੰਦਰ ਸਥਿਰ int16x8_t ggml_vec_dot_s16(int16x8_t acc, int8x16_t a, int8x16_t b) {
    ਵਾਪਸ vec_meadd(a, b, vec_moadd(a, b, acc));
}

ਸਤਰਅੰਦਰ ਸਥਿਰ int32x4_t ggml_vec_dot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ਅਚਲ int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    ਵਾਪਸ acc + (vec_unpackh(p) + vec_unpackl(p));
}

#ਜੇ_ਅੰਤ

#ਜੇ defined(__loongarch_sx)
/* float type data load instructions */
ਸਥਿਰ __m128 __lsx_vreplfr2vr_s(ਅਚਲ ਦਸ਼ਮਲਵ val) {
    v4f32 res = {val, val, val, val};
    ਵਾਪਸ (__m128)res;
}
#ਜੇ_ਅੰਤ

#ਜੇ defined(__loongarch_asx)
ਸਥਿਰ __m256 __lasx_xvreplfr2vr_s(ਅਚਲ ਦਸ਼ਮਲਵ val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    ਵਾਪਸ (__m256)res;
}
#ਜੇ_ਅੰਤ

// TODO: move to ggml-threading
ਖਾਲੀ ggml_barrier(ਬਣਤਰ ggml_threadpool * tp);

ਖਾਲੀ ggml_threadpool_chunk_ਸੈੱਟ(ਬਣਤਰ ggml_threadpool * tp, ਪੂਰਨ ਮੁੱਲ);
ਪੂਰਨ  ggml_threadpool_chunk_ਜੋੜੋ(ਬਣਤਰ ggml_threadpool * tp, ਪੂਰਨ ਮੁੱਲ);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
