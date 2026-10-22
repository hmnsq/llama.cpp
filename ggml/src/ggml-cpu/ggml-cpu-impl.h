#సూచన once

// GGML CPU internal header

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <string.h> // memcpy
#అన్తర్భావ <math.h>   // fabsf

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

నిర్మాణం ggml_compute_params {
    // ith = thread index, nth = number of threads
    పూర్ణాంకం ith, nth;

    // work buffer for all threads
    size_t wsize;
    శూన్య * wdata;

    నిర్మాణం ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#ఒకవేళ defined(_MSC_VER)

#నిర్వచించు m512bh(p) p
#నిర్వచించు m512i(p) p

#లేకపోతే

#నిర్వచించు m512bh(p) (__m512bh)(p)
#నిర్వచించు m512i(p) (__m512i)(p)

#అయితే_ముగింపు

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#ఒకవేళ defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#నిర్వచితం_కాకపోతే __FMA__
#నిర్వచించు __FMA__
#అయితే_ముగింపు
#నిర్వచితం_కాకపోతే __F16C__
#నిర్వచించు __F16C__
#అయితే_ముగింపు
#అయితే_ముగింపు

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#ఒకవేళ defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#నిర్వచితం_కాకపోతే __SSE3__
#నిర్వచించు __SSE3__
#అయితే_ముగింపు
#నిర్వచితం_కాకపోతే __SSSE3__
#నిర్వచించు __SSSE3__
#అయితే_ముగింపు
#అయితే_ముగింపు

#ఒకవేళ defined(__s390x__) && defined(__VEC__)
#నిర్వచితం_కాకపోతే __VXE__
#నిర్వచించు __VXE__
#అయితే_ముగింపు  // __VXE__
#నిర్వచితం_కాకపోతే __VXE2__
#నిర్వచించు __VXE2__
#అయితే_ముగింపు  // __VXE2__
#అయితే_ముగింపు  // __s390x__ && __VEC__

#ఒకవేళ defined(__ARM_FEATURE_SVE) && defined(__linux__)
#అన్తర్భావ <sys/prctl.h>
#అయితే_ముగింపు

#ఒకవేళ defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#ఒకవేళ defined(_MSC_VER) && !defined(__clang__)
#నిర్వచించు ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#లేకపోతే
#నిర్వచించు ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#అయితే_ముగింపు // _MSC_VER

#ఒకవేళ !defined(__aarch64__)

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

అంతఃవరుస స్థిర int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    తిరిగి vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

అంతఃవరుస స్థిర int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_దిగువ_s16(a), vget_ఎగువ_s16(a));
    int16x4_t b0 = vpadd_s16(vget_దిగువ_s16(b), vget_ఎగువ_s16(b));
    తిరిగి vcombine_s16(a0, b0);
}

అంతఃవరుస స్థిర int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_దిగువ_s32(a), vget_ఎగువ_s32(a));
    int32x2_t b0 = vpadd_s32(vget_దిగువ_s32(b), vget_ఎగువ_s32(b));
    తిరిగి vcombine_s32(a0, b0);
}

అంతఃవరుస స్థిర int32_t vaddvq_s32(int32x4_t v) {
    తిరిగి vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

అంతఃవరుస స్థిర దశాంశం vaddvq_f32(float32x4_t v) {
    తిరిగి vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

అంతఃవరుస స్థిర దశాంశం vmaxvq_f32(float32x4_t v) {
    తిరిగి
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

అంతఃవరుస స్థిర int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    తిరిగి res;
}

అంతఃవరుస స్థిర uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    తిరిగి res;
}

అంతఃవరుస స్థిర uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    తిరిగి res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

రకనిర్వచనం నిర్మాణం ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

అంతఃవరుస స్థిర ggml_int16x8x2_t ggml_vld1q_s16_x2(స్థిరరాశి int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    తిరిగి res;
}

రకనిర్వచనం నిర్మాణం ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

అంతఃవరుస స్థిర ggml_uint8x16x2_t ggml_vld1q_u8_x2(స్థిరరాశి uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    తిరిగి res;
}

రకనిర్వచనం నిర్మాణం ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

అంతఃవరుస స్థిర ggml_uint8x16x4_t ggml_vld1q_u8_x4(స్థిరరాశి uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    తిరిగి res;
}

రకనిర్వచనం నిర్మాణం ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

అంతఃవరుస స్థిర ggml_int8x16x2_t ggml_vld1q_s8_x2(స్థిరరాశి int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    తిరిగి res;
}

రకనిర్వచనం నిర్మాణం ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

అంతఃవరుస స్థిర ggml_int8x16x4_t ggml_vld1q_s8_x4(స్థిరరాశి int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    తిరిగి res;
}

// NOTE: not tested
అంతఃవరుస స్థిర int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
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

    తిరిగి res;
}

// NOTE: not tested
అంతఃవరుస స్థిర uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
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

    తిరిగి res;
}

#లేకపోతే

#నిర్వచించు ggml_int16x8x2_t  int16x8x2_t
#నిర్వచించు ggml_uint8x16x2_t uint8x16x2_t
#నిర్వచించు ggml_uint8x16x4_t uint8x16x4_t
#నిర్వచించు ggml_int8x16x2_t  int8x16x2_t
#నిర్వచించు ggml_int8x16x4_t  int8x16x4_t

#నిర్వచించు ggml_vld1q_s16_x2 vld1q_s16_x2
#నిర్వచించు ggml_vld1q_u8_x2  vld1q_u8_x2
#నిర్వచించు ggml_vld1q_u8_x4  vld1q_u8_x4
#నిర్వచించు ggml_vld1q_s8_x2  vld1q_s8_x2
#నిర్వచించు ggml_vld1q_s8_x4  vld1q_s8_x4
#నిర్వచించు ggml_vqtbl1q_s8   vqtbl1q_s8
#నిర్వచించు ggml_vqtbl1q_u8   vqtbl1q_u8

#అయితే_ముగింపు // !defined(__aarch64__)

#ఒకవేళ !defined(__ARM_FEATURE_DOTPROD)

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
అంతఃవరుస స్థిర int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    స్థిరరాశి int16x8_t p0 = vmull_s8(vget_దిగువ_s8 (a), vget_దిగువ_s8 (b));
    స్థిరరాశి int16x8_t p1 = vmull_s8(vget_ఎగువ_s8(a), vget_ఎగువ_s8(b));

    తిరిగి vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#లేకపోతే

#నిర్వచించు ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#అయితే_ముగింపు // !defined(__ARM_FEATURE_DOTPROD)

స్థిర అంతఃవరుస int32x4_t ggml_nvfp4_dot8(స్థిరరాశి int8x8_t q4_lo, స్థిరరాశి int8x8_t q8_lo,
                                         స్థిరరాశి int8x8_t q4_hi, స్థిరరాశి int8x8_t q8_hi) {
    స్థిరరాశి int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    స్థిరరాశి int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    స్థిరరాశి int32x4_t కూడిక_lo = vpaddlq_s16(p_lo);
    స్థిరరాశి int32x4_t కూడిక_hi = vpaddlq_s16(p_hi);
    తిరిగి vaddq_s32(కూడిక_lo, కూడిక_hi);
}

#అయితే_ముగింపు // defined(__ARM_NEON)

#నిర్వచితమైతే __wasm_simd128__
#అన్తర్భావ <wasm_simd128.h>
#అయితే_ముగింపు

#నిర్వచితమైతే __POWER9_VECTOR__
#అన్తర్భావ <altivec.h>
#అయితే_ముగింపు

#ఒకవేళ defined(_MSC_VER) || defined(__MINGW32__)
#అన్తర్భావ <intrin.h>
#లేదావేళ defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)
#అన్తర్భావ <immintrin.h>
#అయితే_ముగింపు

#నిర్వచితమైతే __riscv_v_intrinsic
#అన్తర్భావ <riscv_vector.h>
#అయితే_ముగింపు

#ఒకవేళ defined(__loongarch64)
#ఒకవేళ defined(__loongarch_asx)
#అన్తర్భావ <lasxintrin.h>
#అయితే_ముగింపు
#ఒకవేళ defined(__loongarch_sx)
#అన్తర్భావ <lsxintrin.h>
#అయితే_ముగింపు
#అయితే_ముగింపు

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
#అన్తర్భావ <vecintrin.h>

#నిర్వచించు vec_neg(a)    (-(a))                // Vector Negate
#నిర్వచించు vec_కూడు(a, b) ((a) + (b))           // Vector Add
#నిర్వచించు vec_sub(a, b) ((a) - (b))           // Vector Subtract
#నిర్వచించు vec_mul(a, b) ((a) * (b))           // Vector Multiply
#నిర్వచించు vec_div(a, b) ((a) / (b))           // Vector Divide
#నిర్వచించు vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#నిర్వచించు vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#నిర్వచించు vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#నిర్వచించు vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#నిర్వచించు vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#నిర్వచితం_కాకపోతే vec_and
#నిర్వచించు vec_and(a, b) ((a) & (b)) // Vector AND
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే vec_or
#నిర్వచించు vec_or(a, b)  ((a) | (b)) // Vector OR
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే vec_xor
#నిర్వచించు vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#అయితే_ముగింపు

రకనిర్వచనం గుర్తుగల   అక్షరం char8x16_t  __attribute__((సదిశ_కొలత(16)));
రకనిర్వచనం గుర్తులేని అక్షరం uchar8x16_t __attribute__((సదిశ_కొలత(16)));

రకనిర్వచనం int8_t  int8x16_t __attribute__((సదిశ_కొలత(16)));
రకనిర్వచనం int16_t int16x8_t __attribute__((సదిశ_కొలత(16)));
రకనిర్వచనం int32_t int32x4_t __attribute__((సదిశ_కొలత(16)));
రకనిర్వచనం int64_t int64x2_t __attribute__((సదిశ_కొలత(16)));

రకనిర్వచనం uint8_t  uint8x16_t __attribute__((సదిశ_కొలత(16)));
రకనిర్వచనం uint16_t uint16x8_t __attribute__((సదిశ_కొలత(16)));
రకనిర్వచనం uint32_t uint32x4_t __attribute__((సదిశ_కొలత(16)));
రకనిర్వచనం uint64_t uint64x2_t __attribute__((సదిశ_కొలత(16)));

రకనిర్వచనం దశాంశం  float32x4_t  __attribute__((సదిశ_కొలత(16)));
రకనిర్వచనం ద్విగుణ double64x2_t __attribute__((సదిశ_కొలత(16)));

రకనిర్వచనం నిర్మాణం ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

అంతఃవరుస స్థిర ggml_uint8x16x2_t ggml_vec_xl_u8x2(స్థిరరాశి uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    తిరిగి res;
}

రకనిర్వచనం నిర్మాణం ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

అంతఃవరుస స్థిర ggml_uint8x16x4_t ggml_vec_xl_u8x4(స్థిరరాశి uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    తిరిగి res;
}

రకనిర్వచనం నిర్మాణం ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

అంతఃవరుస స్థిర ggml_int8x16x4_t ggml_vec_xl_s8x4(స్థిరరాశి int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    తిరిగి res;
}

// load 4 bytes and replicate them into all 4 word lanes
అంతఃవరుస స్థిర int8x16_t ggml_vec_xl_s8x4_splat(స్థిరరాశి int8_t * ptr) {
    uint32_t tmp;
    జ్ఞాపకం_నకలు(&tmp, ptr, పరిమాణం(tmp));
    తిరిగి (int8x16_t)vec_splats(tmp);
}

రకనిర్వచనం నిర్మాణం ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

అంతఃవరుస స్థిర ggml_int16x8x2_t ggml_vec_xl_s16x2(స్థిరరాశి int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    తిరిగి res;
}

// faster fp16 -> fp32 conversion using vector intrinsics
// WARNING: fp16 inf converts into 65536.0f. use this only for finite values.
అంతఃవరుస స్థిర float32x4_t ggml_vec_xl_f16x4_finite(స్థిరరాశి ggml_fp16_t * x) {
    uint64_t tmp;
    జ్ఞాపకం_నకలు(&tmp, x, పరిమాణం(tmp));

    స్థిరరాశి uint64x2_t  v_x    = { tmp, tmp };
    స్థిరరాశి uint32x4_t  v_xh   = vec_unpackh((uint16x8_t)v_x);
    స్థిరరాశి uint32x4_t  v_బిట్లు = vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x7FFF)), 13);
    స్థిరరాశి float32x4_t v_abs  = vec_mul((float32x4_t)v_బిట్లు, vec_splats(0x1.0p+112f));
    తిరిగి (float32x4_t)vec_or((uint32x4_t)v_abs, vec_sl(vec_and(v_xh, vec_splats((uint32_t)0x8000)), 16));
}

అంతఃవరుస స్థిర int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    స్థిరరాశి uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    స్థిరరాశి int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    స్థిరరాశి int16x8_t v_abe = vec_perm(a, b, v_maske);
    తిరిగి v_abo + v_abe;
}

అంతఃవరుస స్థిర int32x4_t vec_paddl_s16(int16x8_t a) {
    స్థిరరాశి int16x8_t v_ones = vec_splats((int16_t)1);
    తిరిగి vec_mule(a, v_ones) + vec_mulo(a, v_ones);
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
అంతఃవరుస స్థిర దశాంశం vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_తాత్కాలిక = v + vec_reve(v);
    తిరిగి v_తాత్కాలిక[0] + v_తాత్కాలిక[1];
}

అంతఃవరుస స్థిర int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_తాత్కాలిక = v + vec_reve(v);
    తిరిగి v_తాత్కాలిక[0] + v_తాత్కాలిక[1];
}

అంతఃవరుస స్థిర int16x8_t ggml_vec_dot_s16(int16x8_t acc, int8x16_t a, int8x16_t b) {
    తిరిగి vec_meadd(a, b, vec_moadd(a, b, acc));
}

అంతఃవరుస స్థిర int32x4_t ggml_vec_dot_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    స్థిరరాశి int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    తిరిగి acc + (vec_unpackh(p) + vec_unpackl(p));
}

#అయితే_ముగింపు

#ఒకవేళ defined(__loongarch_sx)
/* float type data load instructions */
స్థిర __m128 __lsx_vreplfr2vr_s(స్థిరరాశి దశాంశం val) {
    v4f32 res = {val, val, val, val};
    తిరిగి (__m128)res;
}
#అయితే_ముగింపు

#ఒకవేళ defined(__loongarch_asx)
స్థిర __m256 __lasx_xvreplfr2vr_s(స్థిరరాశి దశాంశం val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    తిరిగి (__m256)res;
}
#అయితే_ముగింపు

// TODO: move to ggml-threading
శూన్య ggml_barrier(నిర్మాణం ggml_threadpool * tp);

శూన్య ggml_threadpool_chunk_సమితి(నిర్మాణం ggml_threadpool * tp, పూర్ణాంకం విలువ);
పూర్ణాంకం  ggml_threadpool_chunk_కూడు(నిర్మాణం ggml_threadpool * tp, పూర్ణాంకం విలువ);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
