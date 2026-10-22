#आशय once

// GGML CPU internal header

#समावेश "ggml.h"
#समावेश "ggml-impl.h"

#समावेश <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#समावेश <stdbool.h>
#समावेश <string.h> // memcpy
#समावेश <math.h>   // fabsf

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

काष्ठा ggml_compute_params {
    // ith = thread index, nth = number of threads
    पूर्णांक ith, nth;

    // work buffer for all threads
    माप_प्रकार wsize;
    व्योम * wdata;

    काष्ठा ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#अगर defined(_MSC_VER)

#घोषणा m512bh(p) p
#घोषणा m512i(p) p

#अन्यथा

#घोषणा m512bh(p) (__m512bh)(p)
#घोषणा m512i(p) (__m512i)(p)

#पूर्ण_अगर

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#अगर defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#अगर_अघोषित __FMA__
#घोषणा __FMA__
#पूर्ण_अगर
#अगर_अघोषित __F16C__
#घोषणा __F16C__
#पूर्ण_अगर
#पूर्ण_अगर

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#अगर defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#अगर_अघोषित __SSE3__
#घोषणा __SSE3__
#पूर्ण_अगर
#अगर_अघोषित __SSSE3__
#घोषणा __SSSE3__
#पूर्ण_अगर
#पूर्ण_अगर

#अगर defined(__s390x__) && defined(__VEC__)
#अगर_अघोषित __VXE__
#घोषणा __VXE__
#पूर्ण_अगर  // __VXE__
#अगर_अघोषित __VXE2__
#घोषणा __VXE2__
#पूर्ण_अगर  // __VXE2__
#पूर्ण_अगर  // __s390x__ && __VEC__

#अगर defined(__ARM_FEATURE_SVE) && defined(__linux__)
#समावेश <sys/prctl.h>
#पूर्ण_अगर

#अगर defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#अगर defined(_MSC_VER) && !defined(__clang__)
#घोषणा ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#अन्यथा
#घोषणा ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#पूर्ण_अगर // _MSC_VER

#अगर !defined(__aarch64__)

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

अंतरभूत अटल int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    वापस vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

अंतरभूत अटल int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_निचला_s16(a), vget_ऊपरी_s16(a));
    int16x4_t b0 = vpadd_s16(vget_निचला_s16(b), vget_ऊपरी_s16(b));
    वापस vcombine_s16(a0, b0);
}

अंतरभूत अटल int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_निचला_s32(a), vget_ऊपरी_s32(a));
    int32x2_t b0 = vpadd_s32(vget_निचला_s32(b), vget_ऊपरी_s32(b));
    वापस vcombine_s32(a0, b0);
}

अंतरभूत अटल int32_t vaddvq_s32(int32x4_t v) {
    वापस vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

अंतरभूत अटल भग्न vaddvq_f32(float32x4_t v) {
    वापस vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

अंतरभूत अटल भग्न vmaxvq_f32(float32x4_t v) {
    वापस
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

अंतरभूत अटल int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    वापस res;
}

अंतरभूत अटल uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    वापस res;
}

अंतरभूत अटल uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    वापस res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

प्रकार काष्ठा ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

अंतरभूत अटल ggml_int16x8x2_t ggml_vld1q_s16_x2(स्थिर int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    वापस res;
}

प्रकार काष्ठा ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

अंतरभूत अटल ggml_uint8x16x2_t ggml_vld1q_u8_x2(स्थिर uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    वापस res;
}

प्रकार काष्ठा ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

अंतरभूत अटल ggml_uint8x16x4_t ggml_vld1q_u8_x4(स्थिर uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    वापस res;
}

प्रकार काष्ठा ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

अंतरभूत अटल ggml_int8x16x2_t ggml_vld1q_s8_x2(स्थिर int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    वापस res;
}

प्रकार काष्ठा ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

अंतरभूत अटल ggml_int8x16x4_t ggml_vld1q_s8_x4(स्थिर int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    वापस res;
}

// NOTE: not tested
अंतरभूत अटल int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
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

    वापस res;
}

// NOTE: not tested
अंतरभूत अटल uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
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

    वापस res;
}

#अन्यथा

#घोषणा ggml_int16x8x2_t  int16x8x2_t
#घोषणा ggml_uint8x16x2_t uint8x16x2_t
#घोषणा ggml_uint8x16x4_t uint8x16x4_t
#घोषणा ggml_int8x16x2_t  int8x16x2_t
#घोषणा ggml_int8x16x4_t  int8x16x4_t

#घोषणा ggml_vld1q_s16_x2 vld1q_s16_x2
#घोषणा ggml_vld1q_u8_x2  vld1q_u8_x2
#घोषणा ggml_vld1q_u8_x4  vld1q_u8_x4
#घोषणा ggml_vld1q_s8_x2  vld1q_s8_x2
#घोषणा ggml_vld1q_s8_x4  vld1q_s8_x4
#घोषणा ggml_vqtbl1q_s8   vqtbl1q_s8
#घोषणा ggml_vqtbl1q_u8   vqtbl1q_u8

#पूर्ण_अगर // !defined(__aarch64__)

#अगर !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
अंतरभूत अटल int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    स्थिर int16x8_t p0 = vmull_s8(vget_निचला_s8 (a), vget_निचला_s8 (b));
    स्थिर int16x8_t p1 = vmull_s8(vget_ऊपरी_s8(a), vget_ऊपरी_s8(b));

    वापस vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#अन्यथा

#घोषणा ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#पूर्ण_अगर // !defined(__ARM_FEATURE_DOTPROD)

अटल अंतरभूत int32x4_t ggml_nvfp4_dot8(स्थिर int8x8_t q4_lo, स्थिर int8x8_t q8_lo,
                                         स्थिर int8x8_t q4_hi, स्थिर int8x8_t q8_hi) {
    स्थिर int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    स्थिर int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    स्थिर int32x4_t योग_lo = vpaddlq_s16(p_lo);
    स्थिर int32x4_t योग_hi = vpaddlq_s16(p_hi);
    वापस vaddq_s32(योग_lo, योग_hi);
}

#पूर्ण_अगर // defined(__ARM_NEON)

#अगर_घोषित __wasm_simd128__
#समावेश <wasm_simd128.h>
#पूर्ण_अगर

#अगर_घोषित __POWER9_VECTOR__
#समावेश <altivec.h>
#पूर्ण_अगर

#अगर defined(_MSC_VER) || defined(__MINGW32__)
#समावेश <intrin.h>
#या_अगर defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#समावेश <immintrin.h>
#पूर्ण_अगर

#अगर_घोषित __riscv_v_intrinsic
#समावेश <riscv_vector.h>
#पूर्ण_अगर

#अगर defined(__loongarch64)
#अगर defined(__loongarch_asx)
#समावेश <lasxintrin.h>
#पूर्ण_अगर
#अगर defined(__loongarch_sx)
#समावेश <lsxintrin.h>
#पूर्ण_अगर
#पूर्ण_अगर

#अगर defined(__VXE__) || defined(__VXE2__)
#समावेश <vecintrin.h>

#घोषणा vec_neg(a)    (-(a))                // Vector Negate
#घोषणा vec_add(a, b) ((a) + (b))           // Vector Add
#घोषणा vec_sub(a, b) ((a) - (b))           // Vector Subtract
#घोषणा vec_mul(a, b) ((a) * (b))           // Vector Multiply
#घोषणा vec_div(a, b) ((a) / (b))           // Vector Divide
#घोषणा vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#घोषणा vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#घोषणा vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#घोषणा vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#घोषणा vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#अगर_अघोषित vec_and
#घोषणा vec_and(a, b) ((a) & (b)) // Vector AND
#पूर्ण_अगर

#अगर_अघोषित vec_or
#घोषणा vec_or(a, b)  ((a) | (b)) // Vector OR
#पूर्ण_अगर

#अगर_अघोषित vec_xor
#घोषणा vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#पूर्ण_अगर

प्रकार चिन्हित   अक्षर char8x16_t  __attribute__((vector_आकार(16)));
प्रकार अचिन्हित अक्षर uchar8x16_t __attribute__((vector_आकार(16)));

प्रकार int8_t  int8x16_t __attribute__((vector_आकार(16)));
प्रकार int16_t int16x8_t __attribute__((vector_आकार(16)));
प्रकार int32_t int32x4_t __attribute__((vector_आकार(16)));
प्रकार int64_t int64x2_t __attribute__((vector_आकार(16)));

प्रकार uint8_t  uint8x16_t __attribute__((vector_आकार(16)));
प्रकार uint16_t uint16x8_t __attribute__((vector_आकार(16)));
प्रकार uint32_t uint32x4_t __attribute__((vector_आकार(16)));
प्रकार uint64_t uint64x2_t __attribute__((vector_आकार(16)));

प्रकार भग्न  float32x4_t  __attribute__((vector_आकार(16)));
प्रकार द्विगुन double64x2_t __attribute__((vector_आकार(16)));

प्रकार काष्ठा ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

अंतरभूत अटल ggml_uint8x16x2_t ggml_vec_xl_u8x2(स्थिर uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    वापस res;
}

प्रकार काष्ठा ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

अंतरभूत अटल ggml_uint8x16x4_t ggml_vec_xl_u8x4(स्थिर uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    वापस res;
}

प्रकार काष्ठा ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

अंतरभूत अटल ggml_int8x16x4_t ggml_vec_xl_s8x4(स्थिर int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    वापस res;
}

// load 4 bytes and replicate them into all 4 word lanes
अंतरभूत अटल int8x16_t ggml_vec_xl_s8x4_splat(स्थिर int8_t * ptr) {
    uint32_t tmp;
    स_नकल(&tmp, ptr, माप(tmp));
    वापस (int8x16_t)vec_splats(tmp);
}

प्रकार काष्ठा ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

अंतरभूत अटल ggml_int16x8x2_t ggml_vec_xl_s16x2(स्थिर int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    वापस res;
}

// faster fp16 -> fp32 conversion using vector intrinsics
// WARNING: fp16 inf converts into 65536.0f. use this only for finite values.
अंतरभूत अटल float32x4_t ggml_vec_xl_f16x4_finite(स्थिर ggml_fp16_t * x) {
    uint64_t tmp;
    स_नकल(&tmp, x, माप(tmp));

    स्थिर uint64x2_t  v_x    = { tmp, tmp };
    स्थिर uint32x4_t  v_xh   = vec_unpackh((uint16x8_t)v_x);
    स्थिर uint32x4_t  v_bits = vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x7FFF)), 13);
    स्थिर float32x4_t v_abs  = vec_mul((float32x4_t)v_bits, vec_splats(0x1.0p+112f));
    वापस (float32x4_t)vec_or((uint32x4_t)v_abs, vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x8000)), 16));
}

अंतरभूत अटल int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    स्थिर uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    स्थिर int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    स्थिर int16x8_t v_abe = vec_perm(a, b, v_maske);
    वापस v_abo + v_abe;
}

अंतरभूत अटल int32x4_t vec_paddl_s16(int16x8_t a) {
    स्थिर int16x8_t v_ones = vec_splats((int16_t)1);
    वापस vec_mule(a, v_ones) + vec_mulo(a, v_ones);
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
अंतरभूत अटल भग्न vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_अस्थायी = v + vec_reve(v);
    वापस v_अस्थायी[0] + v_अस्थायी[1];
}

अंतरभूत अटल int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_अस्थायी = v + vec_reve(v);
    वापस v_अस्थायी[0] + v_अस्थायी[1];
}

अंतरभूत अटल int16x8_t ggml_vec_dot_s16(int16x8_t acc, int8x16_t a, int8x16_t b) {
    वापस vec_meadd(a, b, vec_moadd(a, b, acc));
}

अंतरभूत अटल int32x4_t ggml_vec_dot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    स्थिर int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    वापस acc + (vec_unpackh(p) + vec_unpackl(p));
}

#पूर्ण_अगर

#अगर defined(__loongarch_sx)
/* float type data load instructions */
अटल __m128 __lsx_vreplfr2vr_s(स्थिर भग्न val) {
    v4f32 res = {val, val, val, val};
    वापस (__m128)res;
}
#पूर्ण_अगर

#अगर defined(__loongarch_asx)
अटल __m256 __lasx_xvreplfr2vr_s(स्थिर भग्न val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    वापस (__m256)res;
}
#पूर्ण_अगर

// TODO: move to ggml-threading
व्योम ggml_barrier(काष्ठा ggml_threadpool * tp);

व्योम ggml_threadpool_chunk_set(काष्ठा ggml_threadpool * tp, पूर्णांक मूल्य);
पूर्णांक  ggml_threadpool_chunk_add(काष्ठा ggml_threadpool * tp, पूर्णांक मूल्य);

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर
