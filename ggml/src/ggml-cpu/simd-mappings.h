#رهنمود once

#ضمن "ggml-cpu-impl.h"

#اگر_تعریف_شده __ARM_FEATURE_SVE
#ضمن <arm_sve.h>
#پایان_اگر // __ARM_FEATURE_SVE

#إذا defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ضمن <arm_neon.h>
#پایان_اگر

#إذا defined(__riscv_v_intrinsic)
#ضمن <riscv_vector.h>
#پایان_اگر

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

//
// simd mappings
//

// FP16 to FP32 conversion

// 16-bit float
// on Arm, we use __fp16, which requires the IEEE fp16 format: implied on
// AArch64, selected by -mfp16-format=ieee on 32 bit Arm, where the compiler
// may otherwise reject the type
// on x86, we use uint16_t
//
// for old CUDA compilers (<= 11), we use uint16_t: ref https://github.com/ggml-org/llama.cpp/pull/10616
// for     MUSA compilers        , we use uint16_t: ref https://github.com/ggml-org/llama.cpp/pull/11843
//
#إذا defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
    #تعریف_کن GGML_CPU_COMPUTE_FP16_مقصد_FP32(x) neon_compute_fp16_مقصد_fp32(x)
    #تعریف_کن GGML_CPU_COMPUTE_FP32_مقصد_FP16(x) neon_compute_fp32_مقصد_fp16(x)

    #تعریف_کن GGML_CPU_FP16_مقصد_FP32(x) GGML_CPU_COMPUTE_FP16_مقصد_FP32(x)
    #تعریف_کن GGML_CPU_FP32_مقصد_FP16(x) GGML_CPU_COMPUTE_FP32_مقصد_FP16(x)

    ساكن مضمن عائم neon_compute_fp16_مقصد_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        کپی_حافظه(&tmp, &h, حجم(ggml_fp16_t));
        أرجع (عائم)tmp;
    }

    ساكن مضمن ggml_fp16_t neon_compute_fp32_مقصد_fp16(عائم f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        کپی_حافظه(&res, &tmp, حجم(ggml_fp16_t));
        أرجع res;
    }
#وإلاإذا defined(__F16C__)
    #اگر_تعریف_شده _MSC_VER
        #تعریف_کن GGML_CPU_COMPUTE_FP16_مقصد_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #تعریف_کن GGML_CPU_COMPUTE_FP32_مقصد_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(mm_مجموعه_ss(x), 0), 0)
    #وإلا
        #تعریف_کن GGML_CPU_COMPUTE_FP16_مقصد_FP32(x) _cvtsh_ss(x)
        #تعریف_کن GGML_CPU_COMPUTE_FP32_مقصد_FP16(x) _cvtss_sh(x, 0)
    #پایان_اگر
#وإلاإذا defined(__POWER9_VECTOR__)
    #تعریف_کن GGML_CPU_COMPUTE_FP16_مقصد_FP32(x) power_compute_fp16_مقصد_fp32(x)
    #تعریف_کن GGML_CPU_COMPUTE_FP32_مقصد_FP16(x) power_compute_fp32_مقصد_fp16(x)
    /* the inline asm below is about 12% faster than the lookup method */
    #تعریف_کن GGML_CPU_FP16_مقصد_FP32(x) GGML_CPU_COMPUTE_FP16_مقصد_FP32(x)
    #تعریف_کن GGML_CPU_FP32_مقصد_FP16(x) GGML_CPU_COMPUTE_FP32_مقصد_FP16(x)

    ساكن مضمن عائم power_compute_fp16_مقصد_fp32(ggml_fp16_t h) {
        عائم f;
        مزدوج d;
        __asm__(
            "mtfprd %0,%2\n"
            "xscvhpdp %0,%0\n"
            "frsp %1,%0\n" :
            /*موقت*/ "=d"(d),
            /*بیرون*/  "=f"(f):
            /*درون*/   "r"(h));
        أرجع f;
    }

    ساكن مضمن ggml_fp16_t power_compute_fp32_مقصد_fp16(عائم f) {
        مزدوج d;
        ggml_fp16_t r;
        __asm__( /* xscvdphp can work on double or single precision */
            "xscvdphp %0,%2\n"
            "mffprd %1,%0\n" :
            /*موقت*/ "=d"(d),
            /*بیرون*/  "=r"(r):
            /*درون*/   "f"(f));
        أرجع r;
    }
#وإلاإذا defined(__riscv) && defined(__riscv_zfhmin)
    ساكن مضمن عائم riscv_compute_fp16_مقصد_fp32(ggml_fp16_t h) {
        _Float16 hf;
        کپی_حافظه(&hf, &h, حجم(ggml_fp16_t));
        أرجع hf;
    }

    ساكن مضمن ggml_fp16_t riscv_compute_fp32_مقصد_fp16(عائم f) {
        ggml_fp16_t res;
        _Float16 hf = (_Float16)f;
        کپی_حافظه(&res, &hf, حجم(ggml_fp16_t));
        أرجع res;
    }

    #تعریف_کن GGML_CPU_COMPUTE_FP16_مقصد_FP32(x) riscv_compute_fp16_مقصد_fp32(x)
    #تعریف_کن GGML_CPU_COMPUTE_FP32_مقصد_FP16(x) riscv_compute_fp32_مقصد_fp16(x)
    #تعریف_کن GGML_CPU_FP16_مقصد_FP32(x) GGML_CPU_COMPUTE_FP16_مقصد_FP32(x)
    #تعریف_کن GGML_CPU_FP32_مقصد_FP16(x) GGML_CPU_COMPUTE_FP32_مقصد_FP16(x)
#پایان_اگر

// precomputed f32 table for f16 (256 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
خارجي عائم ggml_جدول_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
خارجي عائم ggml_جدول_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
خارجي عائم ggml_جدول_f32_ue4m3[1 << 8];

// Use lookup table for E8M0 on x86 (faster than bit manipulation)
#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#تعریف_کن GGML_CPU_E8_M0_مقصد_FP32_HALF(x) ggml_جدول_f32_e8m0_half[(uint8_t)(x)]
#وإلا
#تعریف_کن GGML_CPU_E8_M0_مقصد_FP32_HALF(x) GGML_E8_M0_مقصد_FP32_HALF(x)
#پایان_اگر

// Use lookup table for UE4M3 on x86 and ARM (faster than bit manipulation)
#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__ARM_NEON)
#تعریف_کن GGML_CPU_UE4_M3_مقصد_FP32(x) ggml_جدول_f32_ue4m3[(uint8_t)(x)]
#وإلا
#تعریف_کن GGML_CPU_UE4_M3_مقصد_FP32(x) ggml_ue4m3_مقصد_fp32(x)
#پایان_اگر

// Pre AVX512 CPUs (AVX2 with F16C, such as i5-8th gen or i7-4790)
// do not produce codegen as checks below don't test for __F16C__
// but this check does.
#إذا defined(__F16C__)
// force FP16 -> FP32 conversion to use LUT instead
// see: https://github.com/ggml-org/llama.cpp/issues/30342
//#define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
#تعریف_کن GGML_CPU_FP32_مقصد_FP16(x) GGML_CPU_COMPUTE_FP32_مقصد_FP16(x)
#پایان_اگر

// a branch that defines a GGML_CPU_COMPUTE_* macro must also select it, or the fallback takes over silently
#إذا defined(GGML_CPU_COMPUTE_FP16_مقصد_FP32) && !defined(GGML_CPU_FP16_مقصد_FP32) && !defined(__F16C__) // see: https://github.com/ggml-org/llama.cpp/issues/30342
#خطا "a branch defined GGML_CPU_COMPUTE_FP16_TO_FP32 without selecting it as GGML_CPU_FP16_TO_FP32"
#پایان_اگر
#إذا defined(GGML_CPU_COMPUTE_FP32_مقصد_FP16) && !defined(GGML_CPU_FP32_مقصد_FP16)
#خطا "a branch defined GGML_CPU_COMPUTE_FP32_TO_FP16 without selecting it as GGML_CPU_FP32_TO_FP16"
#پایان_اگر

// fall back to the lookup table / the portable software conversion for whichever direction no branch selected
#إذا !defined(GGML_CPU_FP16_مقصد_FP32)
مضمن ساكن عائم ggml_lookup_fp16_مقصد_fp32(ggml_fp16_t f) {
    uint16_t s;
    کپی_حافظه(&s, &f, حجم(uint16_t));
    أرجع ggml_جدول_f32_f16[s];
}

#تعریف_کن GGML_CPU_FP16_مقصد_FP32(x) ggml_lookup_fp16_مقصد_fp32(x)
#پایان_اگر

#إذا !defined(GGML_CPU_FP32_مقصد_FP16)
#تعریف_کن GGML_CPU_FP32_مقصد_FP16(x) GGML_COMPUTE_FP32_مقصد_FP16(x)
#پایان_اگر


// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// GGML_F32_STEP / GGML_F16_STEP
//   number of elements to process in a single step
//
// GGML_F32_EPR / GGML_F16_EPR
//   number of elements to fit in a single register
//

#إذا defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_FMA)

#تعریف_کن GGML_SIMD

// F32 SVE
#تعریف_کن GGML_F32_EPR 8
#تعریف_کن DEFAULT_PG svptrue_b32()

#تعریف_کن GGML_F32xt                        svfloat32_t
#تعریف_کن GGML_F32xt_ZERO                   svdup_n_f32(0.0f)
#تعریف_کن GGML_F32xt_SET1(x)                svdup_n_f32(x)
#تعریف_کن GGML_F32xt_بارگیری_IMPL(pg, a)       svld1_f32(pg, a)
#تعریف_کن GGML_F32xt_بارگیری(a)                GGML_F32xt_بارگیری_IMPL(DEFAULT_PG, a)
#تعریف_کن GGML_F32xt_STORE_IMPL(pg, a, b)   svst1_f32(pg, a, b)
#تعریف_کن GGML_F32xt_STORE(a, b)            GGML_F32xt_STORE_IMPL(DEFAULT_PG, a, b)
#تعریف_کن GGML_F32xt_FMA_IMPL(pg, a, b, c)  svmad_f32_m(pg, b, c, a)
#تعریف_کن GGML_F32xt_FMA(a, b, c)           GGML_F32xt_FMA_IMPL(DEFAULT_PG, a, b, c)
#تعریف_کن GGML_F32xt_جمع_کن_IMPL(pg, a, b)     svadd_f32_m(pg, a, b)
#تعریف_کن GGML_F32xt_جمع_کن(a, b)              GGML_F32xt_جمع_کن_IMPL(DEFAULT_PG, a, b)
#تعریف_کن GGML_F32xt_MUL_IMPL(pg, a, b)     svmul_f32_m(pg, a, b)
#تعریف_کن GGML_F32xt_MUL(a, b)              GGML_F32xt_MUL_IMPL(DEFAULT_PG, a, b)
#تعریف_کن GGML_F32xt_REDUCE_ONE_IMPL(pg, a) svaddv(pg, a)
#تعریف_کن GGML_F32xt_REDUCE_ONE(a)          GGML_F32xt_REDUCE_ONE_IMPL(DEFAULT_PG, a)
#تعریف_کن GGML_F32xt_REDUCE_IMPL(pg, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
{                                                      \
    sum1 = svadd_f32_m(DEFAULT_PG, sum1, sum2);        \
    sum3 = svadd_f32_m(DEFAULT_PG, sum3, sum4);        \
    sum5 = svadd_f32_m(DEFAULT_PG, sum5, sum6);        \
    sum7 = svadd_f32_m(DEFAULT_PG, sum7, sum8);        \
    sum1 = svadd_f32_m(DEFAULT_PG, sum1, sum3);        \
    sum5 = svadd_f32_m(DEFAULT_PG, sum5, sum7);        \
    sum1 = svadd_f32_m(DEFAULT_PG, sum1, sum5);        \
    (res) = (ggml_float) GGML_F32xt_REDUCE_ONE(sum1);  \
}
#تعریف_کن GGML_F32xt_REDUCE(res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
        GGML_F32xt_REDUCE_IMPL(DEFAULT_PG, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)

#تعریف_کن GGML_F32_VEC        GGML_F32xt
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32xt_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32xt_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32xt_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32xt_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32xt_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32xt_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32xt_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32xt_REDUCE

// F16 SVE
#تعریف_کن DEFAULT_PG32    svptrue_b32()
#تعریف_کن DEFAULT_PG16    svptrue_b16()

#تعریف_کن GGML_F32Cxt                         svfloat16_t
#تعریف_کن GGML_F32Cxt_ZERO                    svdup_n_f16(0.0f)
#تعریف_کن GGML_F32Cxt_SET1(x)                 svdup_n_f16(x)
#تعریف_کن GGML_F32_Cxt_بارگیری(p)                 svld1_f16(DEFAULT_PG16, (ثابت __fp16 *)(p))
#تعریف_کن GGML_F32Cxt_STORE(dst_ptr, src_vec) svst1_f16(DEFAULT_PG16, (__fp16 *)(dst_ptr), (src_vec))

#تعریف_کن GGML_F32Cxt_FMA_IMPL(pg, a, b, c)   svmad_f16_x(pg, b, c, a)
#تعریف_کن GGML_F32Cxt_FMA(a, b, c)            GGML_F32Cxt_FMA_IMPL(DEFAULT_PG16, a, b, c)
#تعریف_کن GGML_F32_Cxt_جمع_کن_IMPL(pg, a, b)      svadd_f16_x(pg, a, b)
#تعریف_کن GGML_F32_Cxt_جمع_کن(a, b)               GGML_F32_Cxt_جمع_کن_IMPL(DEFAULT_PG16, a, b)
#تعریف_کن GGML_F32Cxt_MUL_IMPL(pg, a, b)      svmul_f16_x(pg, a, b)
#تعریف_کن GGML_F32Cxt_MUL(a, b)               GGML_F32Cxt_MUL_IMPL(DEFAULT_PG16, a, b)
#تعریف_کن GGML_F32Cxt_REDUCE                  GGML_F16xt_REDUCE_MIXED

#تعریف_کن GGML_F16x_VEC                GGML_F32Cxt
#تعریف_کن GGML_F16x_VEC_ZERO           GGML_F32Cxt_ZERO
#تعریف_کن GGML_F16x_VEC_SET1           GGML_F32Cxt_SET1
#تعریف_کن GGML_F16x_VEC_بارگیری(p, i)     GGML_F32_Cxt_بارگیری(p)
#تعریف_کن GGML_F16x_VEC_STORE(p, r, i) GGML_F32Cxt_STORE((__fp16 *)(p), r)
#تعریف_کن GGML_F16x_VEC_FMA            GGML_F32Cxt_FMA
#تعریف_کن GGML_F16x_VEC_جمع_کن            GGML_F32_Cxt_جمع_کن
#تعریف_کن GGML_F16x_VEC_MUL            GGML_F32Cxt_MUL
#تعریف_کن GGML_F16x_VEC_REDUCE         GGML_F32Cxt_REDUCE

#تعریف_کن GGML_F16xt_REDUCE_ONE_IMPL(pg, a) svaddv_f16(pg, a)
#تعریف_کن GGML_F16xt_REDUCE_ONE(a)          GGML_F16xt_REDUCE_ONE_IMPL(DEFAULT_PG16, a)

#تعریف_کن GGML_F16xt_REDUCE_MIXED_IMPL(pg16, res, sum1, sum2, sum3, sum4)  \
{                                                      \
    sum1 = svadd_f16_x(pg16, sum1, sum2);              \
    sum3 = svadd_f16_x(pg16, sum3, sum4);              \
    sum1 = svadd_f16_x(pg16, sum1, sum3);              \
    __fp16 جمع_f16 = svaddv_f16(pg16, sum1);           \
    (res) = (ggml_float) جمع_f16;                      \
}
#تعریف_کن GGML_F16xt_REDUCE_MIXED(res, sum1, sum2, sum3, sum4)  \
        GGML_F16xt_REDUCE_MIXED_IMPL(DEFAULT_PG16, res, sum1, sum2, sum3, sum4)

// F16 NEON

#إذا defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #تعریف_کن GGML_F16_گام 32
    #تعریف_کن GGML_F16_EPR  8

    #تعریف_کن GGML_F16x8              float16x8_t
    #تعریف_کن GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #تعریف_کن GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #تعریف_کن GGML_F16x8_بارگیری(x)      vld1q_f16((ثابت __fp16 *)(x))
    #تعریف_کن GGML_F16x8_STORE        vst1q_f16
    #تعریف_کن GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #تعریف_کن GGML_F16x8_جمع_کن          vaddq_f16
    #تعریف_کن GGML_F16x8_MUL          vmulq_f16
    #تعریف_کن GGML_F16x8_REDUCE(res, x)                               \
    افعل {                                                            \
        صحيح جابه_جایی = GGML_F16_ARR >> 1;                             \
        لكل (صحيح i = 0; i < جابه_جایی; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[جابه_جایی+i]);              \
        }                                                           \
        جابه_جایی >>= 1;                                               \
        لكل (صحيح i = 0; i < جابه_جایی; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[جابه_جایی+i]);              \
        }                                                           \
        جابه_جایی >>= 1;                                               \
        لكل (صحيح i = 0; i < جابه_جایی; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[جابه_جایی+i]);              \
        }                                                           \
        ثابت float32x4_t t0 = vcvt_f32_f16(vget_کم_f16 ((x)[0])); \
        ثابت float32x4_t t1 = vcvt_f32_f16(vget_بالا_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } طالما (0)

    #تعریف_کن GGML_F16_VEC                GGML_F16x8
    #تعریف_کن GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #تعریف_کن GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #تعریف_کن GGML_F16_VEC_بارگیری(p, i)     GGML_F16x8_بارگیری(p)
    #تعریف_کن GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #تعریف_کن GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F16x8_جمع_کن
    #تعریف_کن GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #تعریف_کن GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#وإلا
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #تعریف_کن GGML_F16_گام 16
    #تعریف_کن GGML_F16_EPR  4

    #تعریف_کن GGML_F32Cx4              float32x4_t
    #تعریف_کن GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #تعریف_کن GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #تعریف_کن GGML_F32_Cx4_بارگیری(x)      vcvt_f32_f16(vld1_f16((ثابت __fp16 *)(x)))
    #تعریف_کن GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #تعریف_کن GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #تعریف_کن GGML_F32_Cx4_جمع_کن          vaddq_f32
    #تعریف_کن GGML_F32Cx4_MUL          vmulq_f32
    #تعریف_کن GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #تعریف_کن GGML_F16_VEC                GGML_F32Cx4
    #تعریف_کن GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #تعریف_کن GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #تعریف_کن GGML_F16_VEC_بارگیری(p, i)     GGML_F32_Cx4_بارگیری(p)
    #تعریف_کن GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #تعریف_کن GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F32_Cx4_جمع_کن
    #تعریف_کن GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #تعریف_کن GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#پایان_اگر

#وإلاإذا defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)

#تعریف_کن GGML_SIMD

// F32 NEON

#تعریف_کن GGML_F32_گام 16
#تعریف_کن GGML_F32_EPR  4

#تعریف_کن GGML_F32x4              float32x4_t
#تعریف_کن GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#تعریف_کن GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#تعریف_کن GGML_F32x4_بارگیری         vld1q_f32
#تعریف_کن GGML_F32x4_STORE        vst1q_f32
#تعریف_کن GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#تعریف_کن GGML_F32x4_جمع_کن          vaddq_f32
#تعریف_کن GGML_F32x4_MUL          vmulq_f32
#تعریف_کن GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#تعریف_کن GGML_F32x4_REDUCE(res, x)                       \
{                                                       \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;                     \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[جابه_جایی+i]);      \
    }                                                   \
    جابه_جایی >>= 1;                                       \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[جابه_جایی+i]);      \
    }                                                   \
    جابه_جایی >>= 1;                                       \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[جابه_جایی+i]);      \
    }                                                   \
    (res) = (ggml_float) GGML_F32x4_REDUCE_ONE((x)[0]); \
}

#تعریف_کن GGML_F32_VEC        GGML_F32x4
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x4_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x4_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x4_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x4_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x4_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x4_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#إذا defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #تعریف_کن GGML_F16_گام 32
    #تعریف_کن GGML_F16_EPR  8

    #تعریف_کن GGML_F16x8              float16x8_t
    #تعریف_کن GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #تعریف_کن GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #تعریف_کن GGML_F16x8_بارگیری(x)      vld1q_f16((ثابت __fp16 *)(x))
    #تعریف_کن GGML_F16x8_STORE        vst1q_f16
    #تعریف_کن GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #تعریف_کن GGML_F16x8_جمع_کن          vaddq_f16
    #تعریف_کن GGML_F16x8_MUL          vmulq_f16
    #تعریف_کن GGML_F16x8_REDUCE(res, x)                               \
    افعل {                                                            \
        صحيح جابه_جایی = GGML_F16_ARR >> 1;                             \
        لكل (صحيح i = 0; i < جابه_جایی; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[جابه_جایی+i]);              \
        }                                                           \
        جابه_جایی >>= 1;                                               \
        لكل (صحيح i = 0; i < جابه_جایی; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[جابه_جایی+i]);              \
        }                                                           \
        جابه_جایی >>= 1;                                               \
        لكل (صحيح i = 0; i < جابه_جایی; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[جابه_جایی+i]);              \
        }                                                           \
        ثابت float32x4_t t0 = vcvt_f32_f16(vget_کم_f16 ((x)[0])); \
        ثابت float32x4_t t1 = vcvt_f32_f16(vget_بالا_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } طالما (0)

    #تعریف_کن GGML_F16_VEC                GGML_F16x8
    #تعریف_کن GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #تعریف_کن GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #تعریف_کن GGML_F16_VEC_بارگیری(p, i)     GGML_F16x8_بارگیری(p)
    #تعریف_کن GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #تعریف_کن GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F16x8_جمع_کن
    #تعریف_کن GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #تعریف_کن GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#وإلا
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #تعریف_کن GGML_F16_گام 16
    #تعریف_کن GGML_F16_EPR  4

    #تعریف_کن GGML_F32Cx4              float32x4_t
    #تعریف_کن GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #تعریف_کن GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #تعریف_کن GGML_F32_Cx4_بارگیری(x)      vcvt_f32_f16(vld1_f16((ثابت __fp16 *)(x)))
    #تعریف_کن GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #تعریف_کن GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #تعریف_کن GGML_F32_Cx4_جمع_کن          vaddq_f32
    #تعریف_کن GGML_F32Cx4_MUL          vmulq_f32
    #تعریف_کن GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #تعریف_کن GGML_F16_VEC                GGML_F32Cx4
    #تعریف_کن GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #تعریف_کن GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #تعریف_کن GGML_F16_VEC_بارگیری(p, i)     GGML_F32_Cx4_بارگیری(p)
    #تعریف_کن GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #تعریف_کن GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F32_Cx4_جمع_کن
    #تعریف_کن GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #تعریف_کن GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#پایان_اگر

#وإلاإذا defined(__AVX512F__)

#تعریف_کن GGML_SIMD

// F32 AVX512

#تعریف_کن GGML_F32_گام 64
#تعریف_کن GGML_F32_EPR  16

#تعریف_کن GGML_F32x16         __m512
#تعریف_کن GGML_F32x16_ZERO    _mm512_setzero_ps()
#تعریف_کن GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#تعریف_کن GGML_F32x16_بارگیری    _mm512_loadu_ps
#تعریف_کن GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#تعریف_کن GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#تعریف_کن GGML_F32x16_جمع_کن     mm512_جمع_کن_ps
#تعریف_کن GGML_F32x16_MUL     _mm512_mul_ps
#تعریف_کن GGML_F32x16_REDUCE(res, x)                                    \
افعل {                                                                  \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;                                   \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                                \
        x[i] = mm512_جمع_کن_ps(x[i], x[جابه_جایی+i]);                      \
    }                                                                 \
    جابه_جایی >>= 1;                                                     \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                                \
        x[i] = mm512_جمع_کن_ps(x[i], x[جابه_جایی+i]);                      \
    }                                                                 \
    جابه_جایی >>= 1;                                                     \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                                \
        x[i] = mm512_جمع_کن_ps(x[i], x[جابه_جایی+i]);                      \
    }                                                                 \
    res = (ggml_float) mm512_reduce_جمع_کن_ps(x[0]);                    \
} طالما (0)

// TODO: is this optimal ?

#تعریف_کن GGML_F32_VEC        GGML_F32x16
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x16_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x16_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x16_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x16_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x16_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x16_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x16_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512

#إذا defined(__AVX512FP16__)

#تعریف_کن GGML_F16_گام 128
#تعریف_کن GGML_F16_EPR  32

#تعریف_کن GGML_F16x32              __m512h
#تعریف_کن GGML_F16x32_ZERO         _mm512_setzero_ph()
#تعریف_کن GGML_F16x32_SET1(x)      _mm512_set1_ph(__extension__(_Float16)(x))
#تعریف_کن GGML_F16x32_بارگیری(x)      _mm512_loadu_ph(x)
#تعریف_کن GGML_F16x32_STORE(x, y)  _mm512_storeu_ph(x, y)
#تعریف_کن GGML_F16x32_FMA(a, b, c) _mm512_fmadd_ph(b, c, a)
#تعریف_کن GGML_F16x32_جمع_کن          mm512_جمع_کن_ph
#تعریف_کن GGML_F16x32_MUL          _mm512_mul_ph
#تعریف_کن GGML_F16x32_REDUCE(res, x)                                     \
افعل {                                                                   \
    صحيح جابه_جایی = GGML_F16_ARR >> 1;                                    \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                                 \
        x[i] = mm512_جمع_کن_ph(x[i], x[جابه_جایی+i]);                       \
    }                                                                  \
    جابه_جایی >>= 1;                                                      \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                                 \
        x[i] = mm512_جمع_کن_ph(x[i], x[جابه_جایی+i]);                       \
    }                                                                  \
    جابه_جایی >>= 1;                                                      \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                                 \
        x[i] = mm512_جمع_کن_ph(x[i], x[جابه_جایی+i]);                       \
    }                                                                  \
    res = (ggml_float) mm512_reduce_جمع_کن_ph(x[0]);                     \
} طالما (0)

#تعریف_کن GGML_F16_VEC                GGML_F16x32
#تعریف_کن GGML_F16_VEC_ZERO           GGML_F16x32_ZERO
#تعریف_کن GGML_F16_VEC_SET1           GGML_F16x32_SET1
#تعریف_کن GGML_F16_VEC_بارگیری(p, i)     GGML_F16x32_بارگیری(p)
#تعریف_کن GGML_F16_VEC_STORE(p, r, i) GGML_F16x32_STORE(p, r[i])
#تعریف_کن GGML_F16_VEC_FMA            GGML_F16x32_FMA
#تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F16x32_جمع_کن
#تعریف_کن GGML_F16_VEC_MUL            GGML_F16x32_MUL
#تعریف_کن GGML_F16_VEC_REDUCE         GGML_F16x32_REDUCE

#وإلا // Fallback FP16 <-> FP32

#تعریف_کن GGML_F16_گام 64
#تعریف_کن GGML_F16_EPR  16

#تعریف_کن GGML_F32Cx16             __m512
#تعریف_کن GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#تعریف_کن GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#تعریف_کن GGML_F32_Cx16_بارگیری(x)     _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(x)))
#تعریف_کن GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#تعریف_کن GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#تعریف_کن GGML_F32_Cx16_جمع_کن         mm512_جمع_کن_ps
#تعریف_کن GGML_F32Cx16_MUL         _mm512_mul_ps
#تعریف_کن GGML_F32Cx16_REDUCE(res, x)                               \
افعل {                                                              \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;                               \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = mm512_جمع_کن_ps(x[i], x[جابه_جایی+i]);                  \
    }                                                             \
    جابه_جایی >>= 1;                                                 \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = mm512_جمع_کن_ps(x[i], x[جابه_جایی+i]);                  \
    }                                                             \
    جابه_جایی >>= 1;                                                 \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = mm512_جمع_کن_ps(x[i], x[جابه_جایی+i]);                  \
    }                                                             \
    res = (ggml_float) mm512_reduce_جمع_کن_ps(x[0]);                \
} طالما (0)

#تعریف_کن GGML_F16_VEC                GGML_F32Cx16
#تعریف_کن GGML_F16_VEC_ZERO           GGML_F32Cx16_ZERO
#تعریف_کن GGML_F16_VEC_SET1           GGML_F32Cx16_SET1
#تعریف_کن GGML_F16_VEC_بارگیری(p, i)     GGML_F32_Cx16_بارگیری(p)
#تعریف_کن GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#تعریف_کن GGML_F16_VEC_FMA            GGML_F32Cx16_FMA
#تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F32_Cx16_جمع_کن
#تعریف_کن GGML_F16_VEC_MUL            GGML_F32Cx16_MUL

#تعریف_کن GGML_F16_VEC_REDUCE         GGML_F32Cx16_REDUCE

#پایان_اگر // __AVX512FP16__
#وإلاإذا defined(__AVX__)

#تعریف_کن GGML_SIMD

// F32 AVX

#تعریف_کن GGML_F32_گام 32
#تعریف_کن GGML_F32_EPR  8

#تعریف_کن GGML_F32x8         __m256
#تعریف_کن GGML_F32x8_ZERO    _mm256_setzero_ps()
#تعریف_کن GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#تعریف_کن GGML_F32x8_بارگیری    _mm256_loadu_ps
#تعریف_کن GGML_F32x8_STORE   _mm256_storeu_ps
#إذا defined(__FMA__)
    #تعریف_کن GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#وإلا
    #تعریف_کن GGML_F32x8_FMA(a, b, c) mm256_جمع_کن_ps(_mm256_mul_ps(b, c), a)
#پایان_اگر
#تعریف_کن GGML_F32x8_جمع_کن     mm256_جمع_کن_ps
#تعریف_کن GGML_F32x8_MUL     _mm256_mul_ps
#تعریف_کن GGML_F32x8_REDUCE(res, x)                                 \
افعل {                                                              \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;                               \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = mm256_جمع_کن_ps(x[i], x[جابه_جایی+i]);                  \
    }                                                             \
    جابه_جایی >>= 1;                                                 \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = mm256_جمع_کن_ps(x[i], x[جابه_جایی+i]);                  \
    }                                                             \
    جابه_جایی >>= 1;                                                 \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = mm256_جمع_کن_ps(x[i], x[جابه_جایی+i]);                  \
    }                                                             \
    ثابت __m128 t0 = mm_جمع_کن_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    ثابت __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} طالما (0)
// TODO: is this optimal ?

#تعریف_کن GGML_F32_VEC        GGML_F32x8
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x8_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x8_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x8_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x8_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x8_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x8_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 AVX

#تعریف_کن GGML_F16_گام 32
#تعریف_کن GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#تعریف_کن GGML_F32Cx8             __m256
#تعریف_کن GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#تعریف_کن GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#إذا defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#تعریف_کن GGML_F32_Cx8_بارگیری(x)     _mm256_cvtph_ps(_mm_loadu_si128((ثابت __m128i *)(x)))
#تعریف_کن GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#وإلا
ساكن مضمن __m256 __avx_f32cx8_load(ثابت ggml_fp16_t * x) {
    عائم tmp[8];

    لكل (صحيح i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_مقصد_FP32(x[i]);
    }

    أرجع _mm256_loadu_ps(tmp);
}
ساكن مضمن فراغ __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    عائم arr[8];

    _mm256_storeu_ps(arr, y);

    لكل (صحيح i = 0; i < 8; i++)
        x[i] = GGML_CPU_FP32_مقصد_FP16(arr[i]);
}
#تعریف_کن GGML_F32_Cx8_بارگیری(x)     __avx_f32cx8_load(x)
#تعریف_کن GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#پایان_اگر

#تعریف_کن GGML_F32Cx8_FMA         GGML_F32x8_FMA
#تعریف_کن GGML_F32_Cx8_جمع_کن         mm256_جمع_کن_ps
#تعریف_کن GGML_F32Cx8_MUL         _mm256_mul_ps
#تعریف_کن GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#تعریف_کن GGML_F16_VEC                GGML_F32Cx8
#تعریف_کن GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#تعریف_کن GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#تعریف_کن GGML_F16_VEC_بارگیری(p, i)     GGML_F32_Cx8_بارگیری(p)
#تعریف_کن GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#تعریف_کن GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F32_Cx8_جمع_کن
#تعریف_کن GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#تعریف_کن GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#وإلاإذا defined(__POWER9_VECTOR__)

#تعریف_کن GGML_SIMD

// F32 POWER9

#تعریف_کن GGML_F32_گام 32
#تعریف_کن GGML_F32_EPR  4

#تعریف_کن GGML_F32x4              وکتور عائم
#تعریف_کن GGML_F32x4_ZERO         {0.0f}
#تعریف_کن GGML_F32x4_SET1         vec_splats
#تعریف_کن GGML_F32x4_بارگیری(p)      vec_xl(0, p)
#تعریف_کن GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#تعریف_کن GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#تعریف_کن GGML_F32x4_جمع_کن          vec_جمع_کن
#تعریف_کن GGML_F32x4_MUL          vec_mul
#تعریف_کن GGML_F32x4_REDUCE(res, x)              \
{                                              \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;            \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {         \
        x[i] = vec_جمع_کن(x[i], x[جابه_جایی+i]);     \
    }                                          \
    جابه_جایی >>= 1;                              \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {         \
        x[i] = vec_جمع_کن(x[i], x[جابه_جایی+i]);     \
    }                                          \
    جابه_جایی >>= 1;                              \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {         \
        x[i] = vec_جمع_کن(x[i], x[جابه_جایی+i]);     \
    }                                          \
    res = vec_استخراج(x[0], 0) +               \
          vec_استخراج(x[0], 1) +               \
          vec_استخراج(x[0], 2) +               \
          vec_استخراج(x[0], 3);                \
}
#تعریف_کن GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3)        \
{                                                       \
    وکتور عائم v = vec_جمع_کن(vec_جمع_کن(s0, s1),           \
                             vec_جمع_کن(s2, s3));          \
    v = vec_جمع_کن(v, vec_sld(v, v, 8));                   \
    v = vec_جمع_کن(v, vec_sld(v, v, 4));                   \
    res += (ggml_float) vec_استخراج(v, 0);              \
}

#تعریف_کن GGML_F32_VEC        GGML_F32x4
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x4_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x4_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x4_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x4_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x4_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x4_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 POWER9
#تعریف_کن GGML_F16_گام       GGML_F32_گام
#تعریف_کن GGML_F16_EPR        GGML_F32_EPR
#تعریف_کن GGML_F16_VEC        GGML_F32x4
#تعریف_کن GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#تعریف_کن GGML_F16_VEC_SET1   GGML_F32x4_SET1
#تعریف_کن GGML_F16_VEC_FMA    GGML_F32x4_FMA
#تعریف_کن GGML_F16_VEC_جمع_کن    GGML_F32x4_جمع_کن
#تعریف_کن GGML_F16_VEC_MUL    GGML_F32x4_MUL
#تعریف_کن GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#تعریف_کن GGML_F16_VEC_بارگیری(p, i) (i & 0x1) ?                   \
  vec_استخراج_fp32_from_shorth(vec_xl(0, p - GGML_F16_EPR)) : \
  vec_استخراج_fp32_from_shortl(vec_xl(0, p))
ساكن مضمن بیعلامت حرف ggml_endian_byte(صحيح i) {
       uint16_t tmp_val = 1;
       أرجع ((بیعلامت حرف *)&tmp_val)[i];
}
#تعریف_کن GGML_ENDIAN_BYTE(i) ggml_endian_byte(i)
#تعریف_کن GGML_F16_VEC_STORE(p, r, i)                             \
  إذا (i & 0x1)                                                  \
    vec_xst(vec_pack_مقصد_short_fp32(r[i - GGML_ENDIAN_BYTE(1)],  \
                                   r[i - GGML_ENDIAN_BYTE(0)]), \
            0, p - GGML_F16_EPR)

//BF16 POWER9
#تعریف_کن GGML_BF16_گام 16
#تعریف_کن GGML_BF16_EPR  8

#تعریف_کن GGML_BF16x8         وکتور بیعلامت کوتاه
#تعریف_کن GGML_BF16x8_ZERO    vec_splats((بیعلامت کوتاه)0)
#تعریف_کن GGML_BF16x8_بارگیری(p) vec_xl(0, (ثابت بیعلامت کوتاه *)(p))

#تعریف_کن GGML_BF16_VEC          GGML_BF16x8
#تعریف_کن GGML_BF16_VEC_ZERO     GGML_BF16x8_ZERO
#تعریف_کن GGML_BF16_VEC_بارگیری     GGML_BF16x8_بارگیری
#إذا defined(__LITTLE_ENDIAN__)
#تعریف_کن GGML_BF16_مقصد_F32_LO(v) ((وکتور عائم) vec_mergel(GGML_BF16_VEC_ZERO, (v)))
#تعریف_کن GGML_BF16_مقصد_F32_HI(v) ((وکتور عائم) vec_mergeh(GGML_BF16_VEC_ZERO, (v)))
#وإلا
#تعریف_کن GGML_BF16_مقصد_F32_LO(v) ((وکتور عائم) vec_mergel((v), GGML_BF16_VEC_ZERO))
#تعریف_کن GGML_BF16_مقصد_F32_HI(v) ((وکتور عائم) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#پایان_اگر
#تعریف_کن GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_مقصد_F32_LO(x), GGML_BF16_مقصد_F32_LO(y))
#تعریف_کن GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_مقصد_F32_HI(x), GGML_BF16_مقصد_F32_HI(y))

#وإلاإذا defined(__wasm_simd128__)

#تعریف_کن GGML_SIMD

// F32 WASM

#تعریف_کن GGML_F32_گام 16
#تعریف_کن GGML_F32_EPR  4

#تعریف_کن GGML_F32x4              v128_t
#تعریف_کن GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#تعریف_کن GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#تعریف_کن GGML_F32x4_بارگیری         wasm_v128_بارگیری
#تعریف_کن GGML_F32x4_STORE        wasm_v128_store
#تعریف_کن GGML_F32x4_FMA(a, b, c) wasm_f32x4_جمع_کن(wasm_f32x4_mul(b, c), a)
#تعریف_کن GGML_F32x4_جمع_کن          wasm_f32x4_جمع_کن
#تعریف_کن GGML_F32x4_MUL          wasm_f32x4_mul
#تعریف_کن GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;                \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {             \
        x[i] = wasm_f32x4_جمع_کن(x[i], x[جابه_جایی+i]);  \
    }                                              \
    جابه_جایی >>= 1;                                  \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {             \
        x[i] = wasm_f32x4_جمع_کن(x[i], x[جابه_جایی+i]);  \
    }                                              \
    جابه_جایی >>= 1;                                  \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {             \
        x[i] = wasm_f32x4_جمع_کن(x[i], x[جابه_جایی+i]);  \
    }                                              \
    res = wasm_f32x4_استخراج_lane(x[0], 0) +       \
          wasm_f32x4_استخراج_lane(x[0], 1) +       \
          wasm_f32x4_استخراج_lane(x[0], 2) +       \
          wasm_f32x4_استخراج_lane(x[0], 3);        \
}

#تعریف_کن GGML_F32_VEC        GGML_F32x4
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x4_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x4_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x4_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x4_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x4_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x4_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 WASM

#تعریف_کن GGML_F16_گام 16
#تعریف_کن GGML_F16_EPR  4

مضمن ساكن v128_t __wasm_f16x4_load(ثابت ggml_fp16_t * p) {
    عائم tmp[4];

    tmp[0] = GGML_CPU_FP16_مقصد_FP32(p[0]);
    tmp[1] = GGML_CPU_FP16_مقصد_FP32(p[1]);
    tmp[2] = GGML_CPU_FP16_مقصد_FP32(p[2]);
    tmp[3] = GGML_CPU_FP16_مقصد_FP32(p[3]);

    أرجع wasm_v128_بارگیری(tmp);
}

مضمن ساكن فراغ __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
    عائم tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = GGML_CPU_FP32_مقصد_FP16(tmp[0]);
    p[1] = GGML_CPU_FP32_مقصد_FP16(tmp[1]);
    p[2] = GGML_CPU_FP32_مقصد_FP16(tmp[2]);
    p[3] = GGML_CPU_FP32_مقصد_FP16(tmp[3]);
}

#تعریف_کن GGML_F16x4             v128_t
#تعریف_کن GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#تعریف_کن GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#تعریف_کن GGML_F16x4_بارگیری(x)     __wasm_f16x4_load(x)
#تعریف_کن GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#تعریف_کن GGML_F16x4_FMA         GGML_F32x4_FMA
#تعریف_کن GGML_F16x4_جمع_کن         wasm_f32x4_جمع_کن
#تعریف_کن GGML_F16x4_MUL         wasm_f32x4_mul
#تعریف_کن GGML_F16x4_REDUCE(res, x)                           \
{                                                           \
    صحيح جابه_جایی = GGML_F16_ARR >> 1;                         \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                      \
        x[i] = wasm_f32x4_جمع_کن(x[i], x[جابه_جایی+i]);           \
    }                                                       \
    جابه_جایی >>= 1;                                           \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                      \
        x[i] = wasm_f32x4_جمع_کن(x[i], x[جابه_جایی+i]);           \
    }                                                       \
    جابه_جایی >>= 1;                                           \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                      \
        x[i] = wasm_f32x4_جمع_کن(x[i], x[جابه_جایی+i]);           \
    }                                                       \
    res = (ggml_float) (wasm_f32x4_استخراج_lane(x[0], 0) +  \
          wasm_f32x4_استخراج_lane(x[0], 1) +                \
          wasm_f32x4_استخراج_lane(x[0], 2) +                \
          wasm_f32x4_استخراج_lane(x[0], 3));                \
}

#تعریف_کن GGML_F16_VEC                GGML_F16x4
#تعریف_کن GGML_F16_VEC_ZERO           GGML_F16x4_ZERO
#تعریف_کن GGML_F16_VEC_SET1           GGML_F16x4_SET1
#تعریف_کن GGML_F16_VEC_بارگیری(p, i)     GGML_F16x4_بارگیری(p)
#تعریف_کن GGML_F16_VEC_STORE(p, r, i) GGML_F16x4_STORE(p, r[i])
#تعریف_کن GGML_F16_VEC_FMA            GGML_F16x4_FMA
#تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F16x4_جمع_کن
#تعریف_کن GGML_F16_VEC_MUL            GGML_F16x4_MUL
#تعریف_کن GGML_F16_VEC_REDUCE         GGML_F16x4_REDUCE

#وإلاإذا defined(__SSE3__)

#تعریف_کن GGML_SIMD

// F32 SSE

#تعریف_کن GGML_F32_گام 32
#تعریف_کن GGML_F32_EPR  4

#تعریف_کن GGML_F32x4         __m128
#تعریف_کن GGML_F32x4_ZERO    _mm_setzero_ps()
#تعریف_کن GGML_F32x4_SET1(x) _mm_set1_ps(x)
#تعریف_کن GGML_F32x4_بارگیری    _mm_loadu_ps
#تعریف_کن GGML_F32x4_STORE   _mm_storeu_ps
#إذا defined(__FMA__)
    // TODO: Does this work?
    #تعریف_کن GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#وإلا
    #تعریف_کن GGML_F32x4_FMA(a, b, c) mm_جمع_کن_ps(_mm_mul_ps(b, c), a)
#پایان_اگر
#تعریف_کن GGML_F32x4_جمع_کن     mm_جمع_کن_ps
#تعریف_کن GGML_F32x4_MUL     _mm_mul_ps
#تعریف_کن GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;                               \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = mm_جمع_کن_ps(x[i], x[جابه_جایی+i]);                     \
    }                                                             \
    جابه_جایی >>= 1;                                                 \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = mm_جمع_کن_ps(x[i], x[جابه_جایی+i]);                     \
    }                                                             \
    جابه_جایی >>= 1;                                                 \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = mm_جمع_کن_ps(x[i], x[جابه_جایی+i]);                     \
    }                                                             \
    ثابت __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#تعریف_کن GGML_F32_VEC        GGML_F32x4
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x4_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x4_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x4_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x4_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x4_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x4_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 SSE

#تعریف_کن GGML_F16_گام 32
#تعریف_کن GGML_F16_EPR  4

ساكن مضمن __m128 __sse_f16x4_load(ثابت ggml_fp16_t * x) {
    عائم tmp[4];

    tmp[0] = GGML_CPU_FP16_مقصد_FP32(x[0]);
    tmp[1] = GGML_CPU_FP16_مقصد_FP32(x[1]);
    tmp[2] = GGML_CPU_FP16_مقصد_FP32(x[2]);
    tmp[3] = GGML_CPU_FP16_مقصد_FP32(x[3]);

    أرجع _mm_loadu_ps(tmp);
}

ساكن مضمن فراغ __sse_f16x4_store(ggml_fp16_t * x, __m128 y) {
    عائم arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = GGML_CPU_FP32_مقصد_FP16(arr[0]);
    x[1] = GGML_CPU_FP32_مقصد_FP16(arr[1]);
    x[2] = GGML_CPU_FP32_مقصد_FP16(arr[2]);
    x[3] = GGML_CPU_FP32_مقصد_FP16(arr[3]);
}

#تعریف_کن GGML_F32Cx4             __m128
#تعریف_کن GGML_F32Cx4_ZERO        _mm_setzero_ps()
#تعریف_کن GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#تعریف_کن GGML_F32_Cx4_بارگیری(x)     __sse_f16x4_load(x)
#تعریف_کن GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#تعریف_کن GGML_F32Cx4_FMA         GGML_F32x4_FMA
#تعریف_کن GGML_F32_Cx4_جمع_کن         mm_جمع_کن_ps
#تعریف_کن GGML_F32Cx4_MUL         _mm_mul_ps
#تعریف_کن GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#تعریف_کن GGML_F16_VEC                 GGML_F32Cx4
#تعریف_کن GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#تعریف_کن GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#تعریف_کن GGML_F16_VEC_بارگیری(p, i)      GGML_F32_Cx4_بارگیری(p)
#تعریف_کن GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#تعریف_کن GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#تعریف_کن GGML_F16_VEC_جمع_کن             GGML_F32_Cx4_جمع_کن
#تعریف_کن GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#تعریف_کن GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#وإلاإذا defined(__loongarch_asx)

#تعریف_کن GGML_SIMD

// F32 LASX
#تعریف_کن GGML_F32_گام 32
#تعریف_کن GGML_F32_EPR  8

#تعریف_کن GGML_F32x8         __m256
#تعریف_کن GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#تعریف_کن GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#تعریف_کن GGML_F32x8_بارگیری(x) (__m256)__lasx_xvld((x), 0)
#تعریف_کن GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#تعریف_کن GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#تعریف_کن GGML_F32x8_جمع_کن     __lasx_xvfadd_s
#تعریف_کن GGML_F32x8_MUL     __lasx_xvfmul_s
#تعریف_کن GGML_F32x8_REDUCE(res, x)                                 \
افعل {                                                              \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;                               \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[جابه_جایی+i]);                  \
    }                                                             \
    جابه_جایی >>= 1;                                                 \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[جابه_جایی+i]);                  \
    }                                                             \
    جابه_جایی >>= 1;                                                 \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[جابه_جایی+i]);                  \
    }                                                             \
    عائم *tmp_p = (عائم *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} طالما (0)
// TODO: is this optimal ?

#تعریف_کن GGML_F32_VEC        GGML_F32x8
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x8_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x8_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x8_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x8_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x8_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x8_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 LASX

#تعریف_کن GGML_F16_گام 32
#تعریف_کن GGML_F16_EPR  8

// F16 arithmetic is not supported by LASX, so we use F32 instead

#تعریف_کن GGML_F32Cx8          __m256
#تعریف_کن GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#تعریف_کن GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))

ساكن مضمن __m256 __lasx_f32cx8_load(ثابت ggml_fp16_t * x) {
    __m256i a;
    کپی_حافظه(&a, x, حجم(ggml_fp16_t) * 8);
    a = __lasx_xvpermi_d(a, 0 | (1 << 4));
    أرجع __lasx_xvfcvtl_s_h(a);
}

ساكن مضمن فراغ __lasx_f32cx8_store(ggml_fp16_t * x, __m256 y) {
    __m256i a = __lasx_xvfcvt_h_s(y, y);
    a = __lasx_xvpermi_d(a, 0 | (2 << 2));
    کپی_حافظه(x, &a, حجم(ggml_fp16_t) * 8);
}
#تعریف_کن GGML_F32_Cx8_بارگیری(x)     __lasx_f32cx8_load(x)
#تعریف_کن GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#تعریف_کن GGML_F32Cx8_FMA         GGML_F32x8_FMA
#تعریف_کن GGML_F32_Cx8_جمع_کن         __lasx_xvfadd_s
#تعریف_کن GGML_F32Cx8_MUL         __lasx_xvfmul_s
#تعریف_کن GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#تعریف_کن GGML_F16_VEC                GGML_F32Cx8
#تعریف_کن GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#تعریف_کن GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#تعریف_کن GGML_F16_VEC_بارگیری(p, i)     GGML_F32_Cx8_بارگیری(p)
#تعریف_کن GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#تعریف_کن GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F32_Cx8_جمع_کن
#تعریف_کن GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#تعریف_کن GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#وإلاإذا defined(__loongarch_sx)

#تعریف_کن GGML_SIMD

// F32 LSX

#تعریف_کن GGML_F32_گام 32
#تعریف_کن GGML_F32_EPR  4

#تعریف_کن GGML_F32x4         __m128
#تعریف_کن GGML_F32x4_ZERO    (__m128)__lsx_vldi(0)
#تعریف_کن GGML_F32x4_SET1(x) (__m128)__lsx_vreplfr2vr_s((x))
#تعریف_کن GGML_F32x4_بارگیری(x) (__m128)__lsx_vld((x), 0)
#تعریف_کن GGML_F32x4_STORE(x, y)   __lsx_vst(y, x, 0)
#تعریف_کن GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#تعریف_کن GGML_F32x4_جمع_کن     __lsx_vfadd_s
#تعریف_کن GGML_F32x4_MUL     __lsx_vfmul_s

#تعریف_کن GGML_F32x4_REDUCE(res, x)                               \
{                                                               \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;                             \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[جابه_جایی+i]);                \
    }                                                           \
    جابه_جایی >>= 1;                                               \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[جابه_جایی+i]);                \
    }                                                           \
    جابه_جایی >>= 1;                                               \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[جابه_جایی+i]);                \
    }                                                           \
    __m128i t0 = __lsx_vpickev_w((__m128i)x[0], (__m128i)x[0]); \
    __m128i t1 = __lsx_vpickod_w((__m128i)x[0], (__m128i)x[0]); \
    __m128 t2 = __lsx_vfadd_s((__m128)t0, (__m128)t1);          \
    __m128i t3 = __lsx_vpickev_w((__m128i)t2, (__m128i)t2);     \
    __m128i t4 = __lsx_vpickod_w((__m128i)t2, (__m128i)t2);     \
    __m128 t5 = __lsx_vfadd_s((__m128)t3, (__m128)t4);          \
    res = (ggml_float) ((v4f32)t5)[0];                          \
}

#تعریف_کن GGML_F32_VEC        GGML_F32x4
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x4_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x4_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x4_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x4_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x4_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x4_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 LSX

#تعریف_کن GGML_F16_گام 32
#تعریف_کن GGML_F16_EPR  4

ساكن مضمن __m128 __lsx_f16x4_load(ثابت ggml_fp16_t * x) {
    أرجع __lsx_vfcvtl_s_h(__lsx_vld((ثابت فراغ *)x, 0));
}

ساكن مضمن فراغ __lsx_f16x4_store(ggml_fp16_t * x, __m128 y) {
    __m128i a = __lsx_vfcvt_h_s(y, y);
    کپی_حافظه(x, &a, حجم(ggml_fp16_t) * 4);
}

#تعریف_کن GGML_F32Cx4             __m128
#تعریف_کن GGML_F32Cx4_ZERO        (__m128)__lsx_vldi(0)
#تعریف_کن GGML_F32Cx4_SET1(x)     (__m128)__lsx_vreplfr2vr_s((x))
#تعریف_کن GGML_F32_Cx4_بارگیری(x)     (__m128)__lsx_f16x4_load(x)
#تعریف_کن GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#تعریف_کن GGML_F32Cx4_FMA         GGML_F32x4_FMA
#تعریف_کن GGML_F32_Cx4_جمع_کن         __lsx_vfadd_s
#تعریف_کن GGML_F32Cx4_MUL         __lsx_vfmul_s
#تعریف_کن GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#تعریف_کن GGML_F16_VEC                 GGML_F32Cx4
#تعریف_کن GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#تعریف_کن GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#تعریف_کن GGML_F16_VEC_بارگیری(p, i)      GGML_F32_Cx4_بارگیری(p)
#تعریف_کن GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#تعریف_کن GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#تعریف_کن GGML_F16_VEC_جمع_کن             GGML_F32_Cx4_جمع_کن
#تعریف_کن GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#تعریف_کن GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#وإلاإذا defined(__VXE__) || defined(__VXE2__)

#تعریف_کن GGML_SIMD

// F32 s390x

#تعریف_کن GGML_F32_گام 32
#تعریف_کن GGML_F32_EPR  4

#تعریف_کن GGML_F32x4              float32x4_t
#تعریف_کن GGML_F32x4_ZERO         vec_splats(0.0f)
#تعریف_کن GGML_F32x4_SET1         vec_splats
#تعریف_کن GGML_F32x4_بارگیری(p)      vec_xl(0, p)
#تعریف_کن GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#تعریف_کن GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#تعریف_کن GGML_F32x4_جمع_کن          vec_جمع_کن
#تعریف_کن GGML_F32x4_MUL          vec_mul
#تعریف_کن GGML_F32x4_REDUCE(res, x)                   \
{                                                   \
    صحيح جابه_جایی = GGML_F32_ARR >> 1;                 \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {              \
        x[i] = vec_جمع_کن(x[i], x[جابه_جایی + i]);        \
    }                                               \
    جابه_جایی >>= 1;                                   \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {              \
        x[i] = vec_جمع_کن(x[i], x[جابه_جایی + i]);        \
    }                                               \
    جابه_جایی >>= 1;                                   \
    لكل (صحيح i = 0; i < جابه_جایی; ++i) {              \
        x[i] = vec_جمع_کن(x[i], x[جابه_جایی + i]);        \
    }                                               \
    float32x4_t tmp = x[0] + vec_reve(x[0]);        \
    res = tmp[0] + tmp[1];                          \
}
#تعریف_کن GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3) \
{                                                \
    float32x4_t v = vec_جمع_کن(vec_جمع_کن(s0, s1),     \
                            vec_جمع_کن(s2, s3));    \
    v = vec_جمع_کن(v, vec_sld(v, v, 8));            \
    v = vec_جمع_کن(v, vec_sld(v, v, 4));            \
    res += (ggml_float)vec_استخراج(v, 0);        \
}

#تعریف_کن GGML_F32_VEC        GGML_F32x4
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x4_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x4_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x4_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x4_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x4_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x4_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 s390x
#تعریف_کن GGML_F16_گام GGML_F32_گام
#تعریف_کن GGML_F16_EPR  GGML_F32_EPR

ساكن مضمن uint32x4_t __lzs_f32cx4_to_f16(float32x4_t v_f) {
    float32x4_t v_پایه = vec_mul(vec_mul(vec_abs(v_f), vec_splats(0x1.0p+112f)), vec_splats(0x1.0p-110f));

    ثابت uint32x4_t v_w      = (uint32x4_t)v_f;
    ثابت uint32x4_t v_shl1_w = vec_جمع_کن(v_w, v_w);
    ثابت uint32x4_t v_sign   = vec_and(v_w, vec_splats(UINT32_C(0x80000000)));
    ثابت uint32x4_t v_bias   = vec_max(vec_and(v_shl1_w, vec_splats(UINT32_C(0xFF000000))), vec_splats(UINT32_C(0x71000000)));

    v_پایه = vec_جمع_کن((float32x4_t)vec_جمع_کن(vec_sr(v_bias, 1), vec_splats(UINT32_C(0x07800000))), v_پایه);

    ثابت uint32x4_t v_بیتها    = (uint32x4_t)v_پایه;
    ثابت uint32x4_t v_nonsign = vec_جمع_کن(vec_and(vec_sr(v_بیتها, 13), vec_splats(UINT32_C(0x00007C00))),
                                         vec_and(v_بیتها, vec_splats(UINT32_C(0x00000FFF))));
    ثابت uint32x4_t v_is_nan  = (uint32x4_t)vec_cmpgt(v_shl1_w, vec_splats(UINT32_C(0xFF000000)));

    أرجع vec_or(vec_sr(v_sign, 16), vec_sel(v_nonsign, vec_splats(UINT32_C(0x7E00)), v_is_nan));
}

ساكن مضمن float32x4_t __lzs_f16cx4_load(ثابت ggml_fp16_t * x) {
    عائم tmp[4];

    لكل (صحيح i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_مقصد_FP32(x[i]);
    }

    // note: keep type-cast here to prevent compiler bugs
    // see: https://github.com/ggml-org/llama.cpp/issues/12846
    أرجع vec_xl(0, (ثابت عائم *)(tmp));
}

ساكن مضمن فراغ __lzs_f16cx4_store(ggml_fp16_t * x, float32x4_t v_y) {
    ثابت uint32x4_t v_h = __lzs_f32cx4_to_f16(v_y);
    ثابت uint64_t   tmp = ((uint64x2_t)vec_pack(v_h, v_h))[0];
    کپی_حافظه(x, &tmp, حجم(tmp));
}

#تعریف_کن GGML_F16_VEC                GGML_F32x4
#تعریف_کن GGML_F16_VEC_ZERO           GGML_F32x4_ZERO
#تعریف_کن GGML_F16_VEC_SET1           GGML_F32x4_SET1
#تعریف_کن GGML_F16_VEC_بارگیری(p, i)     __lzs_f16cx4_load(p)
#تعریف_کن GGML_F16_VEC_STORE(p, r, i) __lzs_f16cx4_store(p, r[i])
#تعریف_کن GGML_F16_VEC_FMA            GGML_F32x4_FMA
#تعریف_کن GGML_F16_VEC_جمع_کن            GGML_F32x4_جمع_کن
#تعریف_کن GGML_F16_VEC_MUL            GGML_F32x4_MUL
#تعریف_کن GGML_F16_VEC_REDUCE         GGML_F32x4_REDUCE

// BF16 s390x
#تعریف_کن GGML_BF16_گام 16
#تعریف_کن GGML_BF16_EPR  8

#تعریف_کن GGML_BF16x8         __vector بیعلامت کوتاه
#تعریف_کن GGML_BF16x8_ZERO    vec_splats((بیعلامت کوتاه)0)
#تعریف_کن GGML_BF16x8_بارگیری(p) vec_xl(0, (ثابت بیعلامت کوتاه *)(p))

#تعریف_کن GGML_BF16_VEC      GGML_BF16x8
#تعریف_کن GGML_BF16_VEC_ZERO GGML_BF16x8_ZERO
#تعریف_کن GGML_BF16_VEC_بارگیری GGML_BF16x8_بارگیری
#تعریف_کن GGML_BF16_مقصد_F32_LO(v) ((float32x4_t) vec_mergel((v), GGML_BF16_VEC_ZERO))
#تعریف_کن GGML_BF16_مقصد_F32_HI(v) ((float32x4_t) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#تعریف_کن GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_مقصد_F32_LO(x), GGML_BF16_مقصد_F32_LO(y))
#تعریف_کن GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_مقصد_F32_HI(x), GGML_BF16_مقصد_F32_HI(y))

#وإلاإذا defined(__riscv_v_intrinsic)

// compatible with vlen >= 128

#تعریف_کن GGML_SIMD

// F32

#تعریف_کن GGML_F32_گام 16
#تعریف_کن GGML_F32_EPR  4

#تعریف_کن GGML_F32x4              vfloat32m1_t
#تعریف_کن GGML_F32x4_ZERO         __riscv_vfmv_v_f_f32m1(0.0f, GGML_F32_EPR)
#تعریف_کن GGML_F32x4_SET1(x)      __riscv_vfmv_v_f_f32m1(x, GGML_F32_EPR)
#تعریف_کن GGML_F32x4_بارگیری(x)      __riscv_vle32_v_f32m1(x, GGML_F32_EPR)
#تعریف_کن GGML_F32x4_STORE(b, v)  __riscv_vse32_v_f32m1(b, v, GGML_F32_EPR)
#تعریف_کن GGML_F32x4_FMA(a, b, c) __riscv_vfmacc_vv_f32m1(a, b, c, GGML_F32_EPR)
#تعریف_کن GGML_F32x4_جمع_کن(a, b)    __riscv_vfadd_vv_f32m1(a, b, GGML_F32_EPR)
#تعریف_کن GGML_F32x4_MUL(a, b)    __riscv_vfmul_vv_f32m1(a, b, GGML_F32_EPR)

#تعریف_کن GGML_F32_VEC        GGML_F32x4
#تعریف_کن GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#تعریف_کن GGML_F32_VEC_SET1   GGML_F32x4_SET1
#تعریف_کن GGML_F32_VEC_بارگیری   GGML_F32x4_بارگیری
#تعریف_کن GGML_F32_VEC_STORE  GGML_F32x4_STORE
#تعریف_کن GGML_F32_VEC_FMA    GGML_F32x4_FMA
#تعریف_کن GGML_F32_VEC_جمع_کن    GGML_F32x4_جمع_کن
#تعریف_کن GGML_F32_VEC_MUL    GGML_F32x4_MUL
#تعریف_کن GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

#پایان_اگر

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#اگر_تعریف_شده GGML_SIMD
#تعریف_کن GGML_F32_ARR (GGML_F32_گام/GGML_F32_EPR)
#تعریف_کن GGML_F16_ARR (GGML_F16_گام/GGML_F16_EPR)
#پایان_اگر

// GGML_F16_DOT_*
// like GGML_F16_* but for dot products which need F32 accumulation on AVX512-FP16

#إذا defined(__AVX512FP16__)

#تعریف_کن GGML_F16_DOT_گام           GGML_F32_گام
#تعریف_کن GGML_F16_DOT_EPR            GGML_F32_EPR
#تعریف_کن GGML_F16_DOT_ARR            GGML_F32_ARR
#تعریف_کن GGML_F16_DOT_VEC            GGML_F32x16
#تعریف_کن GGML_F16_DOT_VEC_ZERO       GGML_F32x16_ZERO
#تعریف_کن GGML_F16_DOT_VEC_بارگیری(p, i) _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(p)))
#تعریف_کن GGML_F16_DOT_VEC_FMA        GGML_F32x16_FMA
#تعریف_کن GGML_F16_DOT_VEC_REDUCE     GGML_F32x16_REDUCE

#وإلا

#تعریف_کن GGML_F16_DOT_گام           GGML_F16_گام
#تعریف_کن GGML_F16_DOT_EPR            GGML_F16_EPR
#تعریف_کن GGML_F16_DOT_ARR            GGML_F16_ARR
#تعریف_کن GGML_F16_DOT_VEC            GGML_F16_VEC
#تعریف_کن GGML_F16_DOT_VEC_ZERO       GGML_F16_VEC_ZERO
#تعریف_کن GGML_F16_DOT_VEC_بارگیری       GGML_F16_VEC_بارگیری
#تعریف_کن GGML_F16_DOT_VEC_FMA        GGML_F16_VEC_FMA
#تعریف_کن GGML_F16_DOT_VEC_REDUCE     GGML_F16_VEC_REDUCE

#پایان_اگر // defined(__AVX512FP16__)

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر
