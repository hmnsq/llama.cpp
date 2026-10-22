#ಸೂಚನೆ once

// GGML CPU internal header

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <string.h> // memcpy
#ಅನ್ತರ್ಭಾವ <math.h>   // fabsf

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ ggml_compute_params {
    // ith = thread index, nth = number of threads
    ಪೂರ್ಣಾಂಕ ith, nth;

    // work buffer for all threads
    size_t wsize;
    ಶೂನ್ಯ * wdata;

    ರಚನೆ ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#ಆದರೆ defined(_MSC_VER)

#ವ್ಯಾಖ್ಯಾನಿಸು m512bh(p) p
#ವ್ಯಾಖ್ಯಾನಿಸು m512i(p) p

#ಇಲ್ಲದಿದ್ದರೆ

#ವ್ಯಾಖ್ಯಾನಿಸು m512bh(p) (__m512bh)(p)
#ವ್ಯಾಖ್ಯಾನಿಸು m512i(p) (__m512i)(p)

#ಆದರೆ_ಅಂತ್ಯ

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#ಆದರೆ defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ __FMA__
#ವ್ಯಾಖ್ಯಾನಿಸು __FMA__
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ __F16C__
#ವ್ಯಾಖ್ಯಾನಿಸು __F16C__
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#ಆದರೆ defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ __SSE3__
#ವ್ಯಾಖ್ಯಾನಿಸು __SSE3__
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ __SSSE3__
#ವ್ಯಾಖ್ಯಾನಿಸು __SSSE3__
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__s390x__) && defined(__VEC__)
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ __VXE__
#ವ್ಯಾಖ್ಯಾನಿಸು __VXE__
#ಆದರೆ_ಅಂತ್ಯ  // __VXE__
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ __VXE2__
#ವ್ಯಾಖ್ಯಾನಿಸು __VXE2__
#ಆದರೆ_ಅಂತ್ಯ  // __VXE2__
#ಆದರೆ_ಅಂತ್ಯ  // __s390x__ && __VEC__

#ಆದರೆ defined(__ARM_FEATURE_SVE) && defined(__linux__)
#ಅನ್ತರ್ಭಾವ <sys/prctl.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#ಆದರೆ defined(_MSC_VER) && !defined(__clang__)
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#ಆದರೆ_ಅಂತ್ಯ // _MSC_VER

#ಆದರೆ !defined(__aarch64__)

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

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    ಹಿಂತಿರುಗಿ vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_ಕೆಳಗಿನ_s16(a), vget_ಮೇಲಿನ_s16(a));
    int16x4_t b0 = vpadd_s16(vget_ಕೆಳಗಿನ_s16(b), vget_ಮೇಲಿನ_s16(b));
    ಹಿಂತಿರುಗಿ vcombine_s16(a0, b0);
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_ಕೆಳಗಿನ_s32(a), vget_ಮೇಲಿನ_s32(a));
    int32x2_t b0 = vpadd_s32(vget_ಕೆಳಗಿನ_s32(b), vget_ಮೇಲಿನ_s32(b));
    ಹಿಂತಿರುಗಿ vcombine_s32(a0, b0);
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int32_t vaddvq_s32(int32x4_t v) {
    ಹಿಂತಿರುಗಿ vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ದಶಮಾಂಶ vaddvq_f32(float32x4_t v) {
    ಹಿಂತಿರುಗಿ vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ದಶಮಾಂಶ vmaxvq_f32(float32x4_t v) {
    ಹಿಂತಿರುಗಿ
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    ಹಿಂತಿರುಗಿ res;
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    ಹಿಂತಿರುಗಿ res;
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    ಹಿಂತಿರುಗಿ res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ggml_int16x8x2_t ggml_vld1q_s16_x2(ಸ್ಥಿರಾಂಕ int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    ಹಿಂತಿರುಗಿ res;
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ggml_uint8x16x2_t ggml_vld1q_u8_x2(ಸ್ಥಿರಾಂಕ uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    ಹಿಂತಿರುಗಿ res;
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ggml_uint8x16x4_t ggml_vld1q_u8_x4(ಸ್ಥಿರಾಂಕ uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    ಹಿಂತಿರುಗಿ res;
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ggml_int8x16x2_t ggml_vld1q_s8_x2(ಸ್ಥಿರಾಂಕ int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    ಹಿಂತಿರುಗಿ res;
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ggml_int8x16x4_t ggml_vld1q_s8_x4(ಸ್ಥಿರಾಂಕ int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    ಹಿಂತಿರುಗಿ res;
}

// NOTE: not tested
ಸಾಲೊಳಗೆ ಸ್ಥಿರ int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
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

    ಹಿಂತಿರುಗಿ res;
}

// NOTE: not tested
ಸಾಲೊಳಗೆ ಸ್ಥಿರ uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
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

    ಹಿಂತಿರುಗಿ res;
}

#ಇಲ್ಲದಿದ್ದರೆ

#ವ್ಯಾಖ್ಯಾನಿಸು ggml_int16x8x2_t  int16x8x2_t
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_uint8x16x2_t uint8x16x2_t
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_uint8x16x4_t uint8x16x4_t
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_int8x16x2_t  int8x16x2_t
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_int8x16x4_t  int8x16x4_t

#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vld1q_s16_x2 vld1q_s16_x2
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vld1q_u8_x2  vld1q_u8_x2
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vld1q_u8_x4  vld1q_u8_x4
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vld1q_s8_x2  vld1q_s8_x2
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vld1q_s8_x4  vld1q_s8_x4
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vqtbl1q_s8   vqtbl1q_s8
#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vqtbl1q_u8   vqtbl1q_u8

#ಆದರೆ_ಅಂತ್ಯ // !defined(__aarch64__)

#ಆದರೆ !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
ಸಾಲೊಳಗೆ ಸ್ಥಿರ int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ಸ್ಥಿರಾಂಕ int16x8_t p0 = vmull_s8(vget_ಕೆಳಗಿನ_s8 (a), vget_ಕೆಳಗಿನ_s8 (b));
    ಸ್ಥಿರಾಂಕ int16x8_t p1 = vmull_s8(vget_ಮೇಲಿನ_s8(a), vget_ಮೇಲಿನ_s8(b));

    ಹಿಂತಿರುಗಿ vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#ಇಲ್ಲದಿದ್ದರೆ

#ವ್ಯಾಖ್ಯಾನಿಸು ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#ಆದರೆ_ಅಂತ್ಯ // !defined(__ARM_FEATURE_DOTPROD)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32x4_t ggml_nvfp4_dot8(ಸ್ಥಿರಾಂಕ int8x8_t q4_lo, ಸ್ಥಿರಾಂಕ int8x8_t q8_lo,
                                         ಸ್ಥಿರಾಂಕ int8x8_t q4_hi, ಸ್ಥಿರಾಂಕ int8x8_t q8_hi) {
    ಸ್ಥಿರಾಂಕ int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    ಸ್ಥಿರಾಂಕ int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    ಸ್ಥಿರಾಂಕ int32x4_t ಮೊತ್ತ_lo = vpaddlq_s16(p_lo);
    ಸ್ಥಿರಾಂಕ int32x4_t ಮೊತ್ತ_hi = vpaddlq_s16(p_hi);
    ಹಿಂತಿರುಗಿ vaddq_s32(ಮೊತ್ತ_lo, ಮೊತ್ತ_hi);
}

#ಆದರೆ_ಅಂತ್ಯ // defined(__ARM_NEON)

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __wasm_simd128__
#ಅನ್ತರ್ಭಾವ <wasm_simd128.h>
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __POWER9_VECTOR__
#ಅನ್ತರ್ಭಾವ <altivec.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(_MSC_VER) || defined(__MINGW32__)
#ಅನ್ತರ್ಭಾವ <intrin.h>
#ಅಥವಾಆದರೆ defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#ಅನ್ತರ್ಭಾವ <immintrin.h>
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __riscv_v_intrinsic
#ಅನ್ತರ್ಭಾವ <riscv_vector.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__loongarch64)
#ಆದರೆ defined(__loongarch_asx)
#ಅನ್ತರ್ಭಾವ <lasxintrin.h>
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(__loongarch_sx)
#ಅನ್ತರ್ಭಾವ <lsxintrin.h>
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__VXE__) || defined(__VXE2__)
#ಅನ್ತರ್ಭಾವ <vecintrin.h>

#ವ್ಯಾಖ್ಯಾನಿಸು vec_neg(a)    (-(a))                // Vector Negate
#ವ್ಯಾಖ್ಯಾನಿಸು vec_ಕೂಡಿಸು(a, b) ((a) + (b))           // Vector Add
#ವ್ಯಾಖ್ಯಾನಿಸು vec_sub(a, b) ((a) - (b))           // Vector Subtract
#ವ್ಯಾಖ್ಯಾನಿಸು vec_mul(a, b) ((a) * (b))           // Vector Multiply
#ವ್ಯಾಖ್ಯಾನಿಸು vec_div(a, b) ((a) / (b))           // Vector Divide
#ವ್ಯಾಖ್ಯಾನಿಸು vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#ವ್ಯಾಖ್ಯಾನಿಸು vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#ವ್ಯಾಖ್ಯಾನಿಸು vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#ವ್ಯಾಖ್ಯಾನಿಸು vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#ವ್ಯಾಖ್ಯಾನಿಸು vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ vec_and
#ವ್ಯಾಖ್ಯಾನಿಸು vec_and(a, b) ((a) & (b)) // Vector AND
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ vec_or
#ವ್ಯಾಖ್ಯಾನಿಸು vec_or(a, b)  ((a) | (b)) // Vector OR
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ vec_xor
#ವ್ಯಾಖ್ಯಾನಿಸು vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಚಿಹ್ನೆಯುಳ್ಳ   ಅಕ್ಷರ char8x16_t  __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ uchar8x16_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ int8_t  int8x16_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ int16_t int16x8_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ int32_t int32x4_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ int64_t int64x2_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint8_t  uint8x16_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint16_t uint16x8_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint32_t uint32x4_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint64_t uint64x2_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ದಶಮಾಂಶ  float32x4_t  __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ದ್ವಿಗುಣ double64x2_t __attribute__((ಸದಿಶ_ಗಾತ್ರ(16)));

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ggml_uint8x16x2_t ggml_vec_xl_u8x2(ಸ್ಥಿರಾಂಕ uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    ಹಿಂತಿರುಗಿ res;
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ggml_uint8x16x4_t ggml_vec_xl_u8x4(ಸ್ಥಿರಾಂಕ uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    ಹಿಂತಿರುಗಿ res;
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ggml_int8x16x4_t ggml_vec_xl_s8x4(ಸ್ಥಿರಾಂಕ int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    ಹಿಂತಿರುಗಿ res;
}

// load 4 bytes and replicate them into all 4 word lanes
ಸಾಲೊಳಗೆ ಸ್ಥಿರ int8x16_t ggml_vec_xl_s8x4_splat(ಸ್ಥಿರಾಂಕ int8_t * ptr) {
    uint32_t tmp;
    ಸ್ಮರಣೆ_ನಕಲು(&tmp, ptr, ಗಾತ್ರ(tmp));
    ಹಿಂತಿರುಗಿ (int8x16_t)vec_splats(tmp);
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ggml_int16x8x2_t ggml_vec_xl_s16x2(ಸ್ಥಿರಾಂಕ int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    ಹಿಂತಿರುಗಿ res;
}

// faster fp16 -> fp32 conversion using vector intrinsics
// WARNING: fp16 inf converts into 65536.0f. use this only for finite values.
ಸಾಲೊಳಗೆ ಸ್ಥಿರ float32x4_t ggml_vec_xl_f16x4_finite(ಸ್ಥಿರಾಂಕ ggml_fp16_t * x) {
    uint64_t tmp;
    ಸ್ಮರಣೆ_ನಕಲು(&tmp, x, ಗಾತ್ರ(tmp));

    ಸ್ಥಿರಾಂಕ uint64x2_t  v_x    = { tmp, tmp };
    ಸ್ಥಿರಾಂಕ uint32x4_t  v_xh   = vec_unpackh((uint16x8_t)v_x);
    ಸ್ಥಿರಾಂಕ uint32x4_t  v_ಬಿಟ್ಟುಗಳು = vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x7FFF)), 13);
    ಸ್ಥಿರಾಂಕ float32x4_t v_abs  = vec_mul((float32x4_t)v_ಬಿಟ್ಟುಗಳು, vec_splats(0x1.0p+112f));
    ಹಿಂತಿರುಗಿ (float32x4_t)vec_or((uint32x4_t)v_abs, vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x8000)), 16));
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    ಸ್ಥಿರಾಂಕ uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    ಸ್ಥಿರಾಂಕ int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    ಸ್ಥಿರಾಂಕ int16x8_t v_abe = vec_perm(a, b, v_maske);
    ಹಿಂತಿರುಗಿ v_abo + v_abe;
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int32x4_t vec_paddl_s16(int16x8_t a) {
    ಸ್ಥಿರಾಂಕ int16x8_t v_ones = vec_splats((int16_t)1);
    ಹಿಂತಿರುಗಿ vec_mule(a, v_ones) + vec_mulo(a, v_ones);
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
ಸಾಲೊಳಗೆ ಸ್ಥಿರ ದಶಮಾಂಶ vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_ತಾತ್ಕಾಲಿಕ = v + vec_reve(v);
    ಹಿಂತಿರುಗಿ v_ತಾತ್ಕಾಲಿಕ[0] + v_ತಾತ್ಕಾಲಿಕ[1];
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_ತಾತ್ಕಾಲಿಕ = v + vec_reve(v);
    ಹಿಂತಿರುಗಿ v_ತಾತ್ಕಾಲಿಕ[0] + v_ತಾತ್ಕಾಲಿಕ[1];
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int16x8_t ggml_vec_dot_s16(int16x8_t acc, int8x16_t a, int8x16_t b) {
    ಹಿಂತಿರುಗಿ vec_meadd(a, b, vec_moadd(a, b, acc));
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ int32x4_t ggml_vec_dot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    ಸ್ಥಿರಾಂಕ int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    ಹಿಂತಿರುಗಿ acc + (vec_unpackh(p) + vec_unpackl(p));
}

#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__loongarch_sx)
/* float type data load instructions */
ಸ್ಥಿರ __m128 __lsx_vreplfr2vr_s(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val) {
    v4f32 res = {val, val, val, val};
    ಹಿಂತಿರುಗಿ (__m128)res;
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__loongarch_asx)
ಸ್ಥಿರ __m256 __lasx_xvreplfr2vr_s(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    ಹಿಂತಿರುಗಿ (__m256)res;
}
#ಆದರೆ_ಅಂತ್ಯ

// TODO: move to ggml-threading
ಶೂನ್ಯ ggml_barrier(ರಚನೆ ggml_threadpool * tp);

ಶೂನ್ಯ ggml_threadpool_chunk_ಗಣ(ರಚನೆ ggml_threadpool * tp, ಪೂರ್ಣಾಂಕ ಮೌಲ್ಯ);
ಪೂರ್ಣಾಂಕ  ggml_threadpool_chunk_ಕೂಡಿಸು(ರಚನೆ ggml_threadpool * tp, ಪೂರ್ಣಾಂಕ ಮೌಲ್ಯ);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
