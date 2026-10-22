#निर्देशः once

#अन्तर्भाव "ggml-cpu-impl.h"

#यदि_परिभाषितम् __ARM_FEATURE_SVE
#अन्तर्भाव <arm_sve.h>
#यदि_अन्तः // __ARM_FEATURE_SVE

#यदि defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#अन्तर्भाव <arm_neon.h>
#यदि_अन्तः

#यदि defined(__riscv_v_intrinsic)
#अन्तर्भाव <riscv_vector.h>
#यदि_अन्तः

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

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
#यदि defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
    #परिभाषय GGML_CPU_COMPUTE_FP16_गन्तव्यम्_FP32(x) neon_compute_fp16_गन्तव्यम्_fp32(x)
    #परिभाषय GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16(x) neon_compute_fp32_गन्तव्यम्_fp16(x)

    #परिभाषय GGML_CPU_FP16_गन्तव्यम्_FP32(x) GGML_CPU_COMPUTE_FP16_गन्तव्यम्_FP32(x)
    #परिभाषय GGML_CPU_FP32_गन्तव्यम्_FP16(x) GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16(x)

    स्थिर अन्तःस्थ भिन्न neon_compute_fp16_गन्तव्यम्_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        स्मृतिं_प्रतिलिख(&tmp, &h, परिमाण(ggml_fp16_t));
        प्रत्यागम (भिन्न)tmp;
    }

    स्थिर अन्तःस्थ ggml_fp16_t neon_compute_fp32_गन्तव्यम्_fp16(भिन्न f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        स्मृतिं_प्रतिलिख(&res, &tmp, परिमाण(ggml_fp16_t));
        प्रत्यागम res;
    }
#नोचेद्यदि defined(__F16C__)
    #यदि_परिभाषितम् _MSC_VER
        #परिभाषय GGML_CPU_COMPUTE_FP16_गन्तव्यम्_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #परिभाषय GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(mm_समुच्चयः_ss(x), 0), 0)
    #अन्यथा
        #परिभाषय GGML_CPU_COMPUTE_FP16_गन्तव्यम्_FP32(x) _cvtsh_ss(x)
        #परिभाषय GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16(x) _cvtss_sh(x, 0)
    #यदि_अन्तः
#नोचेद्यदि defined(__POWER9_VECTOR__)
    #परिभाषय GGML_CPU_COMPUTE_FP16_गन्तव्यम्_FP32(x) power_compute_fp16_गन्तव्यम्_fp32(x)
    #परिभाषय GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16(x) power_compute_fp32_गन्तव्यम्_fp16(x)
    /* the inline asm below is about 12% faster than the lookup method */
    #परिभाषय GGML_CPU_FP16_गन्तव्यम्_FP32(x) GGML_CPU_COMPUTE_FP16_गन्तव्यम्_FP32(x)
    #परिभाषय GGML_CPU_FP32_गन्तव्यम्_FP16(x) GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16(x)

    स्थिर अन्तःस्थ भिन्न power_compute_fp16_गन्तव्यम्_fp32(ggml_fp16_t h) {
        भिन्न f;
        द्विगुण d;
        __asm__(
            "mtfprd %0,%2\n"
            "xscvhpdp %0,%0\n"
            "frsp %1,%0\n" :
            /*अस्थायी*/ "=d"(d),
            /*बहिर्गमनम्*/  "=f"(f):
            /*अन्तर्*/   "r"(h));
        प्रत्यागम f;
    }

    स्थिर अन्तःस्थ ggml_fp16_t power_compute_fp32_गन्तव्यम्_fp16(भिन्न f) {
        द्विगुण d;
        ggml_fp16_t r;
        __asm__( /* xscvdphp can work on double or single precision */
            "xscvdphp %0,%2\n"
            "mffprd %1,%0\n" :
            /*अस्थायी*/ "=d"(d),
            /*बहिर्गमनम्*/  "=r"(r):
            /*अन्तर्*/   "f"(f));
        प्रत्यागम r;
    }
#नोचेद्यदि defined(__riscv) && defined(__riscv_zfhmin)
    स्थिर अन्तःस्थ भिन्न riscv_compute_fp16_गन्तव्यम्_fp32(ggml_fp16_t h) {
        _Float16 hf;
        स्मृतिं_प्रतिलिख(&hf, &h, परिमाण(ggml_fp16_t));
        प्रत्यागम hf;
    }

    स्थिर अन्तःस्थ ggml_fp16_t riscv_compute_fp32_गन्तव्यम्_fp16(भिन्न f) {
        ggml_fp16_t res;
        _Float16 hf = (_Float16)f;
        स्मृतिं_प्रतिलिख(&res, &hf, परिमाण(ggml_fp16_t));
        प्रत्यागम res;
    }

    #परिभाषय GGML_CPU_COMPUTE_FP16_गन्तव्यम्_FP32(x) riscv_compute_fp16_गन्तव्यम्_fp32(x)
    #परिभाषय GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16(x) riscv_compute_fp32_गन्तव्यम्_fp16(x)
    #परिभाषय GGML_CPU_FP16_गन्तव्यम्_FP32(x) GGML_CPU_COMPUTE_FP16_गन्तव्यम्_FP32(x)
    #परिभाषय GGML_CPU_FP32_गन्तव्यम्_FP16(x) GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16(x)
#यदि_अन्तः

// precomputed f32 table for f16 (256 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
बाह्य भिन्न ggml_सारणी_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
बाह्य भिन्न ggml_सारणी_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
बाह्य भिन्न ggml_सारणी_f32_ue4m3[1 << 8];

// Use lookup table for E8M0 on x86 (faster than bit manipulation)
#यदि defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#परिभाषय GGML_CPU_E8_M0_गन्तव्यम्_FP32_HALF(x) ggml_सारणी_f32_e8m0_half[(uint8_t)(x)]
#अन्यथा
#परिभाषय GGML_CPU_E8_M0_गन्तव्यम्_FP32_HALF(x) GGML_E8_M0_गन्तव्यम्_FP32_HALF(x)
#यदि_अन्तः

// Use lookup table for UE4M3 on x86 and ARM (faster than bit manipulation)
#यदि defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__ARM_NEON)
#परिभाषय GGML_CPU_UE4_M3_गन्तव्यम्_FP32(x) ggml_सारणी_f32_ue4m3[(uint8_t)(x)]
#अन्यथा
#परिभाषय GGML_CPU_UE4_M3_गन्तव्यम्_FP32(x) ggml_ue4m3_गन्तव्यम्_fp32(x)
#यदि_अन्तः

// Pre AVX512 CPUs (AVX2 with F16C, such as i5-8th gen or i7-4790)
// do not produce codegen as checks below don't test for __F16C__
// but this check does.
#यदि defined(__F16C__)
// force FP16 -> FP32 conversion to use LUT instead
// see: https://github.com/ggml-org/llama.cpp/issues/30342
//#define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
#परिभाषय GGML_CPU_FP32_गन्तव्यम्_FP16(x) GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16(x)
#यदि_अन्तः

// a branch that defines a GGML_CPU_COMPUTE_* macro must also select it, or the fallback takes over silently
#यदि defined(GGML_CPU_COMPUTE_FP16_गन्तव्यम्_FP32) && !defined(GGML_CPU_FP16_गन्तव्यम्_FP32) && !defined(__F16C__) // see: https://github.com/ggml-org/llama.cpp/issues/30342
#दोषः "a branch defined GGML_CPU_COMPUTE_FP16_TO_FP32 without selecting it as GGML_CPU_FP16_TO_FP32"
#यदि_अन्तः
#यदि defined(GGML_CPU_COMPUTE_FP32_गन्तव्यम्_FP16) && !defined(GGML_CPU_FP32_गन्तव्यम्_FP16)
#दोषः "a branch defined GGML_CPU_COMPUTE_FP32_TO_FP16 without selecting it as GGML_CPU_FP32_TO_FP16"
#यदि_अन्तः

// fall back to the lookup table / the portable software conversion for whichever direction no branch selected
#यदि !defined(GGML_CPU_FP16_गन्तव्यम्_FP32)
अन्तःस्थ स्थिर भिन्न ggml_lookup_fp16_गन्तव्यम्_fp32(ggml_fp16_t f) {
    uint16_t s;
    स्मृतिं_प्रतिलिख(&s, &f, परिमाण(uint16_t));
    प्रत्यागम ggml_सारणी_f32_f16[s];
}

#परिभाषय GGML_CPU_FP16_गन्तव्यम्_FP32(x) ggml_lookup_fp16_गन्तव्यम्_fp32(x)
#यदि_अन्तः

#यदि !defined(GGML_CPU_FP32_गन्तव्यम्_FP16)
#परिभाषय GGML_CPU_FP32_गन्तव्यम्_FP16(x) GGML_COMPUTE_FP32_गन्तव्यम्_FP16(x)
#यदि_अन्तः


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

#यदि defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_FMA)

#परिभाषय GGML_SIMD

// F32 SVE
#परिभाषय GGML_F32_EPR 8
#परिभाषय DEFAULT_PG svptrue_b32()

#परिभाषय GGML_F32xt                        svfloat32_t
#परिभाषय GGML_F32xt_ZERO                   svdup_n_f32(0.0f)
#परिभाषय GGML_F32xt_SET1(x)                svdup_n_f32(x)
#परिभाषय GGML_F32xt_आरोपय_IMPL(pg, a)       svld1_f32(pg, a)
#परिभाषय GGML_F32xt_आरोपय(a)                GGML_F32xt_आरोपय_IMPL(DEFAULT_PG, a)
#परिभाषय GGML_F32xt_STORE_IMPL(pg, a, b)   svst1_f32(pg, a, b)
#परिभाषय GGML_F32xt_STORE(a, b)            GGML_F32xt_STORE_IMPL(DEFAULT_PG, a, b)
#परिभाषय GGML_F32xt_FMA_IMPL(pg, a, b, c)  svmad_f32_m(pg, b, c, a)
#परिभाषय GGML_F32xt_FMA(a, b, c)           GGML_F32xt_FMA_IMPL(DEFAULT_PG, a, b, c)
#परिभाषय GGML_F32xt_सङ्कलय_IMPL(pg, a, b)     svadd_f32_m(pg, a, b)
#परिभाषय GGML_F32xt_सङ्कलय(a, b)              GGML_F32xt_सङ्कलय_IMPL(DEFAULT_PG, a, b)
#परिभाषय GGML_F32xt_MUL_IMPL(pg, a, b)     svmul_f32_m(pg, a, b)
#परिभाषय GGML_F32xt_MUL(a, b)              GGML_F32xt_MUL_IMPL(DEFAULT_PG, a, b)
#परिभाषय GGML_F32xt_REDUCE_ONE_IMPL(pg, a) svaddv(pg, a)
#परिभाषय GGML_F32xt_REDUCE_ONE(a)          GGML_F32xt_REDUCE_ONE_IMPL(DEFAULT_PG, a)
#परिभाषय GGML_F32xt_REDUCE_IMPL(pg, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
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
#परिभाषय GGML_F32xt_REDUCE(res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
        GGML_F32xt_REDUCE_IMPL(DEFAULT_PG, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)

#परिभाषय GGML_F32_VEC        GGML_F32xt
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32xt_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32xt_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32xt_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32xt_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32xt_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32xt_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32xt_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32xt_REDUCE

// F16 SVE
#परिभाषय DEFAULT_PG32    svptrue_b32()
#परिभाषय DEFAULT_PG16    svptrue_b16()

#परिभाषय GGML_F32Cxt                         svfloat16_t
#परिभाषय GGML_F32Cxt_ZERO                    svdup_n_f16(0.0f)
#परिभाषय GGML_F32Cxt_SET1(x)                 svdup_n_f16(x)
#परिभाषय GGML_F32_Cxt_आरोपय(p)                 svld1_f16(DEFAULT_PG16, (ध्रुव __fp16 *)(p))
#परिभाषय GGML_F32Cxt_STORE(dst_ptr, src_vec) svst1_f16(DEFAULT_PG16, (__fp16 *)(dst_ptr), (src_vec))

#परिभाषय GGML_F32Cxt_FMA_IMPL(pg, a, b, c)   svmad_f16_x(pg, b, c, a)
#परिभाषय GGML_F32Cxt_FMA(a, b, c)            GGML_F32Cxt_FMA_IMPL(DEFAULT_PG16, a, b, c)
#परिभाषय GGML_F32_Cxt_सङ्कलय_IMPL(pg, a, b)      svadd_f16_x(pg, a, b)
#परिभाषय GGML_F32_Cxt_सङ्कलय(a, b)               GGML_F32_Cxt_सङ्कलय_IMPL(DEFAULT_PG16, a, b)
#परिभाषय GGML_F32Cxt_MUL_IMPL(pg, a, b)      svmul_f16_x(pg, a, b)
#परिभाषय GGML_F32Cxt_MUL(a, b)               GGML_F32Cxt_MUL_IMPL(DEFAULT_PG16, a, b)
#परिभाषय GGML_F32Cxt_REDUCE                  GGML_F16xt_REDUCE_MIXED

#परिभाषय GGML_F16x_VEC                GGML_F32Cxt
#परिभाषय GGML_F16x_VEC_ZERO           GGML_F32Cxt_ZERO
#परिभाषय GGML_F16x_VEC_SET1           GGML_F32Cxt_SET1
#परिभाषय GGML_F16x_VEC_आरोपय(p, i)     GGML_F32_Cxt_आरोपय(p)
#परिभाषय GGML_F16x_VEC_STORE(p, r, i) GGML_F32Cxt_STORE((__fp16 *)(p), r)
#परिभाषय GGML_F16x_VEC_FMA            GGML_F32Cxt_FMA
#परिभाषय GGML_F16x_VEC_सङ्कलय            GGML_F32_Cxt_सङ्कलय
#परिभाषय GGML_F16x_VEC_MUL            GGML_F32Cxt_MUL
#परिभाषय GGML_F16x_VEC_REDUCE         GGML_F32Cxt_REDUCE

#परिभाषय GGML_F16xt_REDUCE_ONE_IMPL(pg, a) svaddv_f16(pg, a)
#परिभाषय GGML_F16xt_REDUCE_ONE(a)          GGML_F16xt_REDUCE_ONE_IMPL(DEFAULT_PG16, a)

#परिभाषय GGML_F16xt_REDUCE_MIXED_IMPL(pg16, res, sum1, sum2, sum3, sum4)  \
{                                                      \
    sum1 = svadd_f16_x(pg16, sum1, sum2);              \
    sum3 = svadd_f16_x(pg16, sum3, sum4);              \
    sum1 = svadd_f16_x(pg16, sum1, sum3);              \
    __fp16 योग_f16 = svaddv_f16(pg16, sum1);           \
    (res) = (ggml_float) योग_f16;                      \
}
#परिभाषय GGML_F16xt_REDUCE_MIXED(res, sum1, sum2, sum3, sum4)  \
        GGML_F16xt_REDUCE_MIXED_IMPL(DEFAULT_PG16, res, sum1, sum2, sum3, sum4)

// F16 NEON

#यदि defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #परिभाषय GGML_F16_चरणः 32
    #परिभाषय GGML_F16_EPR  8

    #परिभाषय GGML_F16x8              float16x8_t
    #परिभाषय GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #परिभाषय GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #परिभाषय GGML_F16x8_आरोपय(x)      vld1q_f16((ध्रुव __fp16 *)(x))
    #परिभाषय GGML_F16x8_STORE        vst1q_f16
    #परिभाषय GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #परिभाषय GGML_F16x8_सङ्कलय          vaddq_f16
    #परिभाषय GGML_F16x8_MUL          vmulq_f16
    #परिभाषय GGML_F16x8_REDUCE(res, x)                               \
    कुरु {                                                            \
        पूर्णाङ्क विस्थापनम् = GGML_F16_ARR >> 1;                             \
        प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[विस्थापनम्+i]);              \
        }                                                           \
        विस्थापनम् >>= 1;                                               \
        प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[विस्थापनम्+i]);              \
        }                                                           \
        विस्थापनम् >>= 1;                                               \
        प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[विस्थापनम्+i]);              \
        }                                                           \
        ध्रुव float32x4_t t0 = vcvt_f32_f16(vget_नीचः_f16 ((x)[0])); \
        ध्रुव float32x4_t t1 = vcvt_f32_f16(vget_उच्चः_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } यावत् (0)

    #परिभाषय GGML_F16_VEC                GGML_F16x8
    #परिभाषय GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #परिभाषय GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #परिभाषय GGML_F16_VEC_आरोपय(p, i)     GGML_F16x8_आरोपय(p)
    #परिभाषय GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #परिभाषय GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F16x8_सङ्कलय
    #परिभाषय GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #परिभाषय GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#अन्यथा
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #परिभाषय GGML_F16_चरणः 16
    #परिभाषय GGML_F16_EPR  4

    #परिभाषय GGML_F32Cx4              float32x4_t
    #परिभाषय GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #परिभाषय GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #परिभाषय GGML_F32_Cx4_आरोपय(x)      vcvt_f32_f16(vld1_f16((ध्रुव __fp16 *)(x)))
    #परिभाषय GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #परिभाषय GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #परिभाषय GGML_F32_Cx4_सङ्कलय          vaddq_f32
    #परिभाषय GGML_F32Cx4_MUL          vmulq_f32
    #परिभाषय GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #परिभाषय GGML_F16_VEC                GGML_F32Cx4
    #परिभाषय GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #परिभाषय GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #परिभाषय GGML_F16_VEC_आरोपय(p, i)     GGML_F32_Cx4_आरोपय(p)
    #परिभाषय GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #परिभाषय GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F32_Cx4_सङ्कलय
    #परिभाषय GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #परिभाषय GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#यदि_अन्तः

#नोचेद्यदि defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)

#परिभाषय GGML_SIMD

// F32 NEON

#परिभाषय GGML_F32_चरणः 16
#परिभाषय GGML_F32_EPR  4

#परिभाषय GGML_F32x4              float32x4_t
#परिभाषय GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#परिभाषय GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#परिभाषय GGML_F32x4_आरोपय         vld1q_f32
#परिभाषय GGML_F32x4_STORE        vst1q_f32
#परिभाषय GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#परिभाषय GGML_F32x4_सङ्कलय          vaddq_f32
#परिभाषय GGML_F32x4_MUL          vmulq_f32
#परिभाषय GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#परिभाषय GGML_F32x4_REDUCE(res, x)                       \
{                                                       \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;                     \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[विस्थापनम्+i]);      \
    }                                                   \
    विस्थापनम् >>= 1;                                       \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[विस्थापनम्+i]);      \
    }                                                   \
    विस्थापनम् >>= 1;                                       \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[विस्थापनम्+i]);      \
    }                                                   \
    (res) = (ggml_float) GGML_F32x4_REDUCE_ONE((x)[0]); \
}

#परिभाषय GGML_F32_VEC        GGML_F32x4
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x4_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x4_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x4_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x4_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x4_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x4_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#यदि defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #परिभाषय GGML_F16_चरणः 32
    #परिभाषय GGML_F16_EPR  8

    #परिभाषय GGML_F16x8              float16x8_t
    #परिभाषय GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #परिभाषय GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #परिभाषय GGML_F16x8_आरोपय(x)      vld1q_f16((ध्रुव __fp16 *)(x))
    #परिभाषय GGML_F16x8_STORE        vst1q_f16
    #परिभाषय GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #परिभाषय GGML_F16x8_सङ्कलय          vaddq_f16
    #परिभाषय GGML_F16x8_MUL          vmulq_f16
    #परिभाषय GGML_F16x8_REDUCE(res, x)                               \
    कुरु {                                                            \
        पूर्णाङ्क विस्थापनम् = GGML_F16_ARR >> 1;                             \
        प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[विस्थापनम्+i]);              \
        }                                                           \
        विस्थापनम् >>= 1;                                               \
        प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[विस्थापनम्+i]);              \
        }                                                           \
        विस्थापनम् >>= 1;                                               \
        प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[विस्थापनम्+i]);              \
        }                                                           \
        ध्रुव float32x4_t t0 = vcvt_f32_f16(vget_नीचः_f16 ((x)[0])); \
        ध्रुव float32x4_t t1 = vcvt_f32_f16(vget_उच्चः_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } यावत् (0)

    #परिभाषय GGML_F16_VEC                GGML_F16x8
    #परिभाषय GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #परिभाषय GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #परिभाषय GGML_F16_VEC_आरोपय(p, i)     GGML_F16x8_आरोपय(p)
    #परिभाषय GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #परिभाषय GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F16x8_सङ्कलय
    #परिभाषय GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #परिभाषय GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#अन्यथा
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #परिभाषय GGML_F16_चरणः 16
    #परिभाषय GGML_F16_EPR  4

    #परिभाषय GGML_F32Cx4              float32x4_t
    #परिभाषय GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #परिभाषय GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #परिभाषय GGML_F32_Cx4_आरोपय(x)      vcvt_f32_f16(vld1_f16((ध्रुव __fp16 *)(x)))
    #परिभाषय GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #परिभाषय GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #परिभाषय GGML_F32_Cx4_सङ्कलय          vaddq_f32
    #परिभाषय GGML_F32Cx4_MUL          vmulq_f32
    #परिभाषय GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #परिभाषय GGML_F16_VEC                GGML_F32Cx4
    #परिभाषय GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #परिभाषय GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #परिभाषय GGML_F16_VEC_आरोपय(p, i)     GGML_F32_Cx4_आरोपय(p)
    #परिभाषय GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #परिभाषय GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F32_Cx4_सङ्कलय
    #परिभाषय GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #परिभाषय GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#यदि_अन्तः

#नोचेद्यदि defined(__AVX512F__)

#परिभाषय GGML_SIMD

// F32 AVX512

#परिभाषय GGML_F32_चरणः 64
#परिभाषय GGML_F32_EPR  16

#परिभाषय GGML_F32x16         __m512
#परिभाषय GGML_F32x16_ZERO    _mm512_setzero_ps()
#परिभाषय GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#परिभाषय GGML_F32x16_आरोपय    _mm512_loadu_ps
#परिभाषय GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#परिभाषय GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#परिभाषय GGML_F32x16_सङ्कलय     mm512_सङ्कलय_ps
#परिभाषय GGML_F32x16_MUL     _mm512_mul_ps
#परिभाषय GGML_F32x16_REDUCE(res, x)                                    \
कुरु {                                                                  \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;                                   \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                                \
        x[i] = mm512_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                      \
    }                                                                 \
    विस्थापनम् >>= 1;                                                     \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                                \
        x[i] = mm512_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                      \
    }                                                                 \
    विस्थापनम् >>= 1;                                                     \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                                \
        x[i] = mm512_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                      \
    }                                                                 \
    res = (ggml_float) mm512_reduce_सङ्कलय_ps(x[0]);                    \
} यावत् (0)

// TODO: is this optimal ?

#परिभाषय GGML_F32_VEC        GGML_F32x16
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x16_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x16_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x16_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x16_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x16_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x16_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x16_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512

#यदि defined(__AVX512FP16__)

#परिभाषय GGML_F16_चरणः 128
#परिभाषय GGML_F16_EPR  32

#परिभाषय GGML_F16x32              __m512h
#परिभाषय GGML_F16x32_ZERO         _mm512_setzero_ph()
#परिभाषय GGML_F16x32_SET1(x)      _mm512_set1_ph(__extension__(_Float16)(x))
#परिभाषय GGML_F16x32_आरोपय(x)      _mm512_loadu_ph(x)
#परिभाषय GGML_F16x32_STORE(x, y)  _mm512_storeu_ph(x, y)
#परिभाषय GGML_F16x32_FMA(a, b, c) _mm512_fmadd_ph(b, c, a)
#परिभाषय GGML_F16x32_सङ्कलय          mm512_सङ्कलय_ph
#परिभाषय GGML_F16x32_MUL          _mm512_mul_ph
#परिभाषय GGML_F16x32_REDUCE(res, x)                                     \
कुरु {                                                                   \
    पूर्णाङ्क विस्थापनम् = GGML_F16_ARR >> 1;                                    \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                                 \
        x[i] = mm512_सङ्कलय_ph(x[i], x[विस्थापनम्+i]);                       \
    }                                                                  \
    विस्थापनम् >>= 1;                                                      \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                                 \
        x[i] = mm512_सङ्कलय_ph(x[i], x[विस्थापनम्+i]);                       \
    }                                                                  \
    विस्थापनम् >>= 1;                                                      \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                                 \
        x[i] = mm512_सङ्कलय_ph(x[i], x[विस्थापनम्+i]);                       \
    }                                                                  \
    res = (ggml_float) mm512_reduce_सङ्कलय_ph(x[0]);                     \
} यावत् (0)

#परिभाषय GGML_F16_VEC                GGML_F16x32
#परिभाषय GGML_F16_VEC_ZERO           GGML_F16x32_ZERO
#परिभाषय GGML_F16_VEC_SET1           GGML_F16x32_SET1
#परिभाषय GGML_F16_VEC_आरोपय(p, i)     GGML_F16x32_आरोपय(p)
#परिभाषय GGML_F16_VEC_STORE(p, r, i) GGML_F16x32_STORE(p, r[i])
#परिभाषय GGML_F16_VEC_FMA            GGML_F16x32_FMA
#परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F16x32_सङ्कलय
#परिभाषय GGML_F16_VEC_MUL            GGML_F16x32_MUL
#परिभाषय GGML_F16_VEC_REDUCE         GGML_F16x32_REDUCE

#अन्यथा // Fallback FP16 <-> FP32

#परिभाषय GGML_F16_चरणः 64
#परिभाषय GGML_F16_EPR  16

#परिभाषय GGML_F32Cx16             __m512
#परिभाषय GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#परिभाषय GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#परिभाषय GGML_F32_Cx16_आरोपय(x)     _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(x)))
#परिभाषय GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#परिभाषय GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#परिभाषय GGML_F32_Cx16_सङ्कलय         mm512_सङ्कलय_ps
#परिभाषय GGML_F32Cx16_MUL         _mm512_mul_ps
#परिभाषय GGML_F32Cx16_REDUCE(res, x)                               \
कुरु {                                                              \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;                               \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = mm512_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                  \
    }                                                             \
    विस्थापनम् >>= 1;                                                 \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = mm512_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                  \
    }                                                             \
    विस्थापनम् >>= 1;                                                 \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = mm512_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                  \
    }                                                             \
    res = (ggml_float) mm512_reduce_सङ्कलय_ps(x[0]);                \
} यावत् (0)

#परिभाषय GGML_F16_VEC                GGML_F32Cx16
#परिभाषय GGML_F16_VEC_ZERO           GGML_F32Cx16_ZERO
#परिभाषय GGML_F16_VEC_SET1           GGML_F32Cx16_SET1
#परिभाषय GGML_F16_VEC_आरोपय(p, i)     GGML_F32_Cx16_आरोपय(p)
#परिभाषय GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#परिभाषय GGML_F16_VEC_FMA            GGML_F32Cx16_FMA
#परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F32_Cx16_सङ्कलय
#परिभाषय GGML_F16_VEC_MUL            GGML_F32Cx16_MUL

#परिभाषय GGML_F16_VEC_REDUCE         GGML_F32Cx16_REDUCE

#यदि_अन्तः // __AVX512FP16__
#नोचेद्यदि defined(__AVX__)

#परिभाषय GGML_SIMD

// F32 AVX

#परिभाषय GGML_F32_चरणः 32
#परिभाषय GGML_F32_EPR  8

#परिभाषय GGML_F32x8         __m256
#परिभाषय GGML_F32x8_ZERO    _mm256_setzero_ps()
#परिभाषय GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#परिभाषय GGML_F32x8_आरोपय    _mm256_loadu_ps
#परिभाषय GGML_F32x8_STORE   _mm256_storeu_ps
#यदि defined(__FMA__)
    #परिभाषय GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#अन्यथा
    #परिभाषय GGML_F32x8_FMA(a, b, c) mm256_सङ्कलय_ps(_mm256_mul_ps(b, c), a)
#यदि_अन्तः
#परिभाषय GGML_F32x8_सङ्कलय     mm256_सङ्कलय_ps
#परिभाषय GGML_F32x8_MUL     _mm256_mul_ps
#परिभाषय GGML_F32x8_REDUCE(res, x)                                 \
कुरु {                                                              \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;                               \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = mm256_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                  \
    }                                                             \
    विस्थापनम् >>= 1;                                                 \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = mm256_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                  \
    }                                                             \
    विस्थापनम् >>= 1;                                                 \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = mm256_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                  \
    }                                                             \
    ध्रुव __m128 t0 = mm_सङ्कलय_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    ध्रुव __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} यावत् (0)
// TODO: is this optimal ?

#परिभाषय GGML_F32_VEC        GGML_F32x8
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x8_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x8_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x8_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x8_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x8_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x8_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 AVX

#परिभाषय GGML_F16_चरणः 32
#परिभाषय GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#परिभाषय GGML_F32Cx8             __m256
#परिभाषय GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#परिभाषय GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#यदि defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#परिभाषय GGML_F32_Cx8_आरोपय(x)     _mm256_cvtph_ps(_mm_loadu_si128((ध्रुव __m128i *)(x)))
#परिभाषय GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#अन्यथा
स्थिर अन्तःस्थ __m256 __avx_f32cx8_load(ध्रुव ggml_fp16_t * x) {
    भिन्न tmp[8];

    प्रति (पूर्णाङ्क i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_गन्तव्यम्_FP32(x[i]);
    }

    प्रत्यागम _mm256_loadu_ps(tmp);
}
स्थिर अन्तःस्थ शून्य __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    भिन्न arr[8];

    _mm256_storeu_ps(arr, y);

    प्रति (पूर्णाङ्क i = 0; i < 8; i++)
        x[i] = GGML_CPU_FP32_गन्तव्यम्_FP16(arr[i]);
}
#परिभाषय GGML_F32_Cx8_आरोपय(x)     __avx_f32cx8_load(x)
#परिभाषय GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#यदि_अन्तः

#परिभाषय GGML_F32Cx8_FMA         GGML_F32x8_FMA
#परिभाषय GGML_F32_Cx8_सङ्कलय         mm256_सङ्कलय_ps
#परिभाषय GGML_F32Cx8_MUL         _mm256_mul_ps
#परिभाषय GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#परिभाषय GGML_F16_VEC                GGML_F32Cx8
#परिभाषय GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#परिभाषय GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#परिभाषय GGML_F16_VEC_आरोपय(p, i)     GGML_F32_Cx8_आरोपय(p)
#परिभाषय GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#परिभाषय GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F32_Cx8_सङ्कलय
#परिभाषय GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#परिभाषय GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#नोचेद्यदि defined(__POWER9_VECTOR__)

#परिभाषय GGML_SIMD

// F32 POWER9

#परिभाषय GGML_F32_चरणः 32
#परिभाषय GGML_F32_EPR  4

#परिभाषय GGML_F32x4              सदिशम् भिन्न
#परिभाषय GGML_F32x4_ZERO         {0.0f}
#परिभाषय GGML_F32x4_SET1         vec_splats
#परिभाषय GGML_F32x4_आरोपय(p)      vec_xl(0, p)
#परिभाषय GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#परिभाषय GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#परिभाषय GGML_F32x4_सङ्कलय          vec_सङ्कलय
#परिभाषय GGML_F32x4_MUL          vec_mul
#परिभाषय GGML_F32x4_REDUCE(res, x)              \
{                                              \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;            \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {         \
        x[i] = vec_सङ्कलय(x[i], x[विस्थापनम्+i]);     \
    }                                          \
    विस्थापनम् >>= 1;                              \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {         \
        x[i] = vec_सङ्कलय(x[i], x[विस्थापनम्+i]);     \
    }                                          \
    विस्थापनम् >>= 1;                              \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {         \
        x[i] = vec_सङ्कलय(x[i], x[विस्थापनम्+i]);     \
    }                                          \
    res = vec_extract(x[0], 0) +               \
          vec_extract(x[0], 1) +               \
          vec_extract(x[0], 2) +               \
          vec_extract(x[0], 3);                \
}
#परिभाषय GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3)        \
{                                                       \
    सदिशम् भिन्न v = vec_सङ्कलय(vec_सङ्कलय(s0, s1),           \
                             vec_सङ्कलय(s2, s3));          \
    v = vec_सङ्कलय(v, vec_sld(v, v, 8));                   \
    v = vec_सङ्कलय(v, vec_sld(v, v, 4));                   \
    res += (ggml_float) vec_extract(v, 0);              \
}

#परिभाषय GGML_F32_VEC        GGML_F32x4
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x4_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x4_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x4_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x4_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x4_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x4_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 POWER9
#परिभाषय GGML_F16_चरणः       GGML_F32_चरणः
#परिभाषय GGML_F16_EPR        GGML_F32_EPR
#परिभाषय GGML_F16_VEC        GGML_F32x4
#परिभाषय GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#परिभाषय GGML_F16_VEC_SET1   GGML_F32x4_SET1
#परिभाषय GGML_F16_VEC_FMA    GGML_F32x4_FMA
#परिभाषय GGML_F16_VEC_सङ्कलय    GGML_F32x4_सङ्कलय
#परिभाषय GGML_F16_VEC_MUL    GGML_F32x4_MUL
#परिभाषय GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#परिभाषय GGML_F16_VEC_आरोपय(p, i) (i & 0x1) ?                   \
  vec_extract_fp32_from_shorth(vec_xl(0, p - GGML_F16_EPR)) : \
  vec_extract_fp32_from_shortl(vec_xl(0, p))
स्थिर अन्तःस्थ अचिह्नित वर्ण ggml_endian_byte(पूर्णाङ्क i) {
       uint16_t tmp_val = 1;
       प्रत्यागम ((अचिह्नित वर्ण *)&tmp_val)[i];
}
#परिभाषय GGML_ENDIAN_BYTE(i) ggml_endian_byte(i)
#परिभाषय GGML_F16_VEC_STORE(p, r, i)                             \
  यदि (i & 0x1)                                                  \
    vec_xst(vec_pack_गन्तव्यम्_short_fp32(r[i - GGML_ENDIAN_BYTE(1)],  \
                                   r[i - GGML_ENDIAN_BYTE(0)]), \
            0, p - GGML_F16_EPR)

//BF16 POWER9
#परिभाषय GGML_BF16_चरणः 16
#परिभाषय GGML_BF16_EPR  8

#परिभाषय GGML_BF16x8         सदिशम् अचिह्नित ह्रस्व
#परिभाषय GGML_BF16x8_ZERO    vec_splats((अचिह्नित ह्रस्व)0)
#परिभाषय GGML_BF16x8_आरोपय(p) vec_xl(0, (ध्रुव अचिह्नित ह्रस्व *)(p))

#परिभाषय GGML_BF16_VEC          GGML_BF16x8
#परिभाषय GGML_BF16_VEC_ZERO     GGML_BF16x8_ZERO
#परिभाषय GGML_BF16_VEC_आरोपय     GGML_BF16x8_आरोपय
#यदि defined(__LITTLE_ENDIAN__)
#परिभाषय GGML_BF16_गन्तव्यम्_F32_LO(v) ((सदिशम् भिन्न) vec_mergel(GGML_BF16_VEC_ZERO, (v)))
#परिभाषय GGML_BF16_गन्तव्यम्_F32_HI(v) ((सदिशम् भिन्न) vec_mergeh(GGML_BF16_VEC_ZERO, (v)))
#अन्यथा
#परिभाषय GGML_BF16_गन्तव्यम्_F32_LO(v) ((सदिशम् भिन्न) vec_mergel((v), GGML_BF16_VEC_ZERO))
#परिभाषय GGML_BF16_गन्तव्यम्_F32_HI(v) ((सदिशम् भिन्न) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#यदि_अन्तः
#परिभाषय GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_गन्तव्यम्_F32_LO(x), GGML_BF16_गन्तव्यम्_F32_LO(y))
#परिभाषय GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_गन्तव्यम्_F32_HI(x), GGML_BF16_गन्तव्यम्_F32_HI(y))

#नोचेद्यदि defined(__wasm_simd128__)

#परिभाषय GGML_SIMD

// F32 WASM

#परिभाषय GGML_F32_चरणः 16
#परिभाषय GGML_F32_EPR  4

#परिभाषय GGML_F32x4              v128_t
#परिभाषय GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#परिभाषय GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#परिभाषय GGML_F32x4_आरोपय         wasm_v128_आरोपय
#परिभाषय GGML_F32x4_STORE        wasm_v128_store
#परिभाषय GGML_F32x4_FMA(a, b, c) wasm_f32x4_सङ्कलय(wasm_f32x4_mul(b, c), a)
#परिभाषय GGML_F32x4_सङ्कलय          wasm_f32x4_सङ्कलय
#परिभाषय GGML_F32x4_MUL          wasm_f32x4_mul
#परिभाषय GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;                \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {             \
        x[i] = wasm_f32x4_सङ्कलय(x[i], x[विस्थापनम्+i]);  \
    }                                              \
    विस्थापनम् >>= 1;                                  \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {             \
        x[i] = wasm_f32x4_सङ्कलय(x[i], x[विस्थापनम्+i]);  \
    }                                              \
    विस्थापनम् >>= 1;                                  \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {             \
        x[i] = wasm_f32x4_सङ्कलय(x[i], x[विस्थापनम्+i]);  \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#परिभाषय GGML_F32_VEC        GGML_F32x4
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x4_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x4_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x4_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x4_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x4_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x4_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 WASM

#परिभाषय GGML_F16_चरणः 16
#परिभाषय GGML_F16_EPR  4

अन्तःस्थ स्थिर v128_t __wasm_f16x4_load(ध्रुव ggml_fp16_t * p) {
    भिन्न tmp[4];

    tmp[0] = GGML_CPU_FP16_गन्तव्यम्_FP32(p[0]);
    tmp[1] = GGML_CPU_FP16_गन्तव्यम्_FP32(p[1]);
    tmp[2] = GGML_CPU_FP16_गन्तव्यम्_FP32(p[2]);
    tmp[3] = GGML_CPU_FP16_गन्तव्यम्_FP32(p[3]);

    प्रत्यागम wasm_v128_आरोपय(tmp);
}

अन्तःस्थ स्थिर शून्य __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
    भिन्न tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = GGML_CPU_FP32_गन्तव्यम्_FP16(tmp[0]);
    p[1] = GGML_CPU_FP32_गन्तव्यम्_FP16(tmp[1]);
    p[2] = GGML_CPU_FP32_गन्तव्यम्_FP16(tmp[2]);
    p[3] = GGML_CPU_FP32_गन्तव्यम्_FP16(tmp[3]);
}

#परिभाषय GGML_F16x4             v128_t
#परिभाषय GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#परिभाषय GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#परिभाषय GGML_F16x4_आरोपय(x)     __wasm_f16x4_load(x)
#परिभाषय GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#परिभाषय GGML_F16x4_FMA         GGML_F32x4_FMA
#परिभाषय GGML_F16x4_सङ्कलय         wasm_f32x4_सङ्कलय
#परिभाषय GGML_F16x4_MUL         wasm_f32x4_mul
#परिभाषय GGML_F16x4_REDUCE(res, x)                           \
{                                                           \
    पूर्णाङ्क विस्थापनम् = GGML_F16_ARR >> 1;                         \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                      \
        x[i] = wasm_f32x4_सङ्कलय(x[i], x[विस्थापनम्+i]);           \
    }                                                       \
    विस्थापनम् >>= 1;                                           \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                      \
        x[i] = wasm_f32x4_सङ्कलय(x[i], x[विस्थापनम्+i]);           \
    }                                                       \
    विस्थापनम् >>= 1;                                           \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                      \
        x[i] = wasm_f32x4_सङ्कलय(x[i], x[विस्थापनम्+i]);           \
    }                                                       \
    res = (ggml_float) (wasm_f32x4_extract_lane(x[0], 0) +  \
          wasm_f32x4_extract_lane(x[0], 1) +                \
          wasm_f32x4_extract_lane(x[0], 2) +                \
          wasm_f32x4_extract_lane(x[0], 3));                \
}

#परिभाषय GGML_F16_VEC                GGML_F16x4
#परिभाषय GGML_F16_VEC_ZERO           GGML_F16x4_ZERO
#परिभाषय GGML_F16_VEC_SET1           GGML_F16x4_SET1
#परिभाषय GGML_F16_VEC_आरोपय(p, i)     GGML_F16x4_आरोपय(p)
#परिभाषय GGML_F16_VEC_STORE(p, r, i) GGML_F16x4_STORE(p, r[i])
#परिभाषय GGML_F16_VEC_FMA            GGML_F16x4_FMA
#परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F16x4_सङ्कलय
#परिभाषय GGML_F16_VEC_MUL            GGML_F16x4_MUL
#परिभाषय GGML_F16_VEC_REDUCE         GGML_F16x4_REDUCE

#नोचेद्यदि defined(__SSE3__)

#परिभाषय GGML_SIMD

// F32 SSE

#परिभाषय GGML_F32_चरणः 32
#परिभाषय GGML_F32_EPR  4

#परिभाषय GGML_F32x4         __m128
#परिभाषय GGML_F32x4_ZERO    _mm_setzero_ps()
#परिभाषय GGML_F32x4_SET1(x) _mm_set1_ps(x)
#परिभाषय GGML_F32x4_आरोपय    _mm_loadu_ps
#परिभाषय GGML_F32x4_STORE   _mm_storeu_ps
#यदि defined(__FMA__)
    // TODO: Does this work?
    #परिभाषय GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#अन्यथा
    #परिभाषय GGML_F32x4_FMA(a, b, c) mm_सङ्कलय_ps(_mm_mul_ps(b, c), a)
#यदि_अन्तः
#परिभाषय GGML_F32x4_सङ्कलय     mm_सङ्कलय_ps
#परिभाषय GGML_F32x4_MUL     _mm_mul_ps
#परिभाषय GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;                               \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = mm_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                     \
    }                                                             \
    विस्थापनम् >>= 1;                                                 \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = mm_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                     \
    }                                                             \
    विस्थापनम् >>= 1;                                                 \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = mm_सङ्कलय_ps(x[i], x[विस्थापनम्+i]);                     \
    }                                                             \
    ध्रुव __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#परिभाषय GGML_F32_VEC        GGML_F32x4
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x4_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x4_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x4_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x4_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x4_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x4_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 SSE

#परिभाषय GGML_F16_चरणः 32
#परिभाषय GGML_F16_EPR  4

स्थिर अन्तःस्थ __m128 __sse_f16x4_load(ध्रुव ggml_fp16_t * x) {
    भिन्न tmp[4];

    tmp[0] = GGML_CPU_FP16_गन्तव्यम्_FP32(x[0]);
    tmp[1] = GGML_CPU_FP16_गन्तव्यम्_FP32(x[1]);
    tmp[2] = GGML_CPU_FP16_गन्तव्यम्_FP32(x[2]);
    tmp[3] = GGML_CPU_FP16_गन्तव्यम्_FP32(x[3]);

    प्रत्यागम _mm_loadu_ps(tmp);
}

स्थिर अन्तःस्थ शून्य __sse_f16x4_store(ggml_fp16_t * x, __m128 y) {
    भिन्न arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = GGML_CPU_FP32_गन्तव्यम्_FP16(arr[0]);
    x[1] = GGML_CPU_FP32_गन्तव्यम्_FP16(arr[1]);
    x[2] = GGML_CPU_FP32_गन्तव्यम्_FP16(arr[2]);
    x[3] = GGML_CPU_FP32_गन्तव्यम्_FP16(arr[3]);
}

#परिभाषय GGML_F32Cx4             __m128
#परिभाषय GGML_F32Cx4_ZERO        _mm_setzero_ps()
#परिभाषय GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#परिभाषय GGML_F32_Cx4_आरोपय(x)     __sse_f16x4_load(x)
#परिभाषय GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#परिभाषय GGML_F32Cx4_FMA         GGML_F32x4_FMA
#परिभाषय GGML_F32_Cx4_सङ्कलय         mm_सङ्कलय_ps
#परिभाषय GGML_F32Cx4_MUL         _mm_mul_ps
#परिभाषय GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#परिभाषय GGML_F16_VEC                 GGML_F32Cx4
#परिभाषय GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#परिभाषय GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#परिभाषय GGML_F16_VEC_आरोपय(p, i)      GGML_F32_Cx4_आरोपय(p)
#परिभाषय GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#परिभाषय GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#परिभाषय GGML_F16_VEC_सङ्कलय             GGML_F32_Cx4_सङ्कलय
#परिभाषय GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#परिभाषय GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#नोचेद्यदि defined(__loongarch_asx)

#परिभाषय GGML_SIMD

// F32 LASX
#परिभाषय GGML_F32_चरणः 32
#परिभाषय GGML_F32_EPR  8

#परिभाषय GGML_F32x8         __m256
#परिभाषय GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#परिभाषय GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#परिभाषय GGML_F32x8_आरोपय(x) (__m256)__lasx_xvld((x), 0)
#परिभाषय GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#परिभाषय GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#परिभाषय GGML_F32x8_सङ्कलय     __lasx_xvfadd_s
#परिभाषय GGML_F32x8_MUL     __lasx_xvfmul_s
#परिभाषय GGML_F32x8_REDUCE(res, x)                                 \
कुरु {                                                              \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;                               \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[विस्थापनम्+i]);                  \
    }                                                             \
    विस्थापनम् >>= 1;                                                 \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[विस्थापनम्+i]);                  \
    }                                                             \
    विस्थापनम् >>= 1;                                                 \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[विस्थापनम्+i]);                  \
    }                                                             \
    भिन्न *tmp_p = (भिन्न *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} यावत् (0)
// TODO: is this optimal ?

#परिभाषय GGML_F32_VEC        GGML_F32x8
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x8_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x8_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x8_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x8_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x8_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x8_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 LASX

#परिभाषय GGML_F16_चरणः 32
#परिभाषय GGML_F16_EPR  8

// F16 arithmetic is not supported by LASX, so we use F32 instead

#परिभाषय GGML_F32Cx8          __m256
#परिभाषय GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#परिभाषय GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))

स्थिर अन्तःस्थ __m256 __lasx_f32cx8_load(ध्रुव ggml_fp16_t * x) {
    __m256i a;
    स्मृतिं_प्रतिलिख(&a, x, परिमाण(ggml_fp16_t) * 8);
    a = __lasx_xvpermi_d(a, 0 | (1 << 4));
    प्रत्यागम __lasx_xvfcvtl_s_h(a);
}

स्थिर अन्तःस्थ शून्य __lasx_f32cx8_store(ggml_fp16_t * x, __m256 y) {
    __m256i a = __lasx_xvfcvt_h_s(y, y);
    a = __lasx_xvpermi_d(a, 0 | (2 << 2));
    स्मृतिं_प्रतिलिख(x, &a, परिमाण(ggml_fp16_t) * 8);
}
#परिभाषय GGML_F32_Cx8_आरोपय(x)     __lasx_f32cx8_load(x)
#परिभाषय GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#परिभाषय GGML_F32Cx8_FMA         GGML_F32x8_FMA
#परिभाषय GGML_F32_Cx8_सङ्कलय         __lasx_xvfadd_s
#परिभाषय GGML_F32Cx8_MUL         __lasx_xvfmul_s
#परिभाषय GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#परिभाषय GGML_F16_VEC                GGML_F32Cx8
#परिभाषय GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#परिभाषय GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#परिभाषय GGML_F16_VEC_आरोपय(p, i)     GGML_F32_Cx8_आरोपय(p)
#परिभाषय GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#परिभाषय GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F32_Cx8_सङ्कलय
#परिभाषय GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#परिभाषय GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#नोचेद्यदि defined(__loongarch_sx)

#परिभाषय GGML_SIMD

// F32 LSX

#परिभाषय GGML_F32_चरणः 32
#परिभाषय GGML_F32_EPR  4

#परिभाषय GGML_F32x4         __m128
#परिभाषय GGML_F32x4_ZERO    (__m128)__lsx_vldi(0)
#परिभाषय GGML_F32x4_SET1(x) (__m128)__lsx_vreplfr2vr_s((x))
#परिभाषय GGML_F32x4_आरोपय(x) (__m128)__lsx_vld((x), 0)
#परिभाषय GGML_F32x4_STORE(x, y)   __lsx_vst(y, x, 0)
#परिभाषय GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#परिभाषय GGML_F32x4_सङ्कलय     __lsx_vfadd_s
#परिभाषय GGML_F32x4_MUL     __lsx_vfmul_s

#परिभाषय GGML_F32x4_REDUCE(res, x)                               \
{                                                               \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;                             \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[विस्थापनम्+i]);                \
    }                                                           \
    विस्थापनम् >>= 1;                                               \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[विस्थापनम्+i]);                \
    }                                                           \
    विस्थापनम् >>= 1;                                               \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[विस्थापनम्+i]);                \
    }                                                           \
    __m128i t0 = __lsx_vpickev_w((__m128i)x[0], (__m128i)x[0]); \
    __m128i t1 = __lsx_vpickod_w((__m128i)x[0], (__m128i)x[0]); \
    __m128 t2 = __lsx_vfadd_s((__m128)t0, (__m128)t1);          \
    __m128i t3 = __lsx_vpickev_w((__m128i)t2, (__m128i)t2);     \
    __m128i t4 = __lsx_vpickod_w((__m128i)t2, (__m128i)t2);     \
    __m128 t5 = __lsx_vfadd_s((__m128)t3, (__m128)t4);          \
    res = (ggml_float) ((v4f32)t5)[0];                          \
}

#परिभाषय GGML_F32_VEC        GGML_F32x4
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x4_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x4_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x4_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x4_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x4_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x4_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 LSX

#परिभाषय GGML_F16_चरणः 32
#परिभाषय GGML_F16_EPR  4

स्थिर अन्तःस्थ __m128 __lsx_f16x4_load(ध्रुव ggml_fp16_t * x) {
    प्रत्यागम __lsx_vfcvtl_s_h(__lsx_vld((ध्रुव शून्य *)x, 0));
}

स्थिर अन्तःस्थ शून्य __lsx_f16x4_store(ggml_fp16_t * x, __m128 y) {
    __m128i a = __lsx_vfcvt_h_s(y, y);
    स्मृतिं_प्रतिलिख(x, &a, परिमाण(ggml_fp16_t) * 4);
}

#परिभाषय GGML_F32Cx4             __m128
#परिभाषय GGML_F32Cx4_ZERO        (__m128)__lsx_vldi(0)
#परिभाषय GGML_F32Cx4_SET1(x)     (__m128)__lsx_vreplfr2vr_s((x))
#परिभाषय GGML_F32_Cx4_आरोपय(x)     (__m128)__lsx_f16x4_load(x)
#परिभाषय GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#परिभाषय GGML_F32Cx4_FMA         GGML_F32x4_FMA
#परिभाषय GGML_F32_Cx4_सङ्कलय         __lsx_vfadd_s
#परिभाषय GGML_F32Cx4_MUL         __lsx_vfmul_s
#परिभाषय GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#परिभाषय GGML_F16_VEC                 GGML_F32Cx4
#परिभाषय GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#परिभाषय GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#परिभाषय GGML_F16_VEC_आरोपय(p, i)      GGML_F32_Cx4_आरोपय(p)
#परिभाषय GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#परिभाषय GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#परिभाषय GGML_F16_VEC_सङ्कलय             GGML_F32_Cx4_सङ्कलय
#परिभाषय GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#परिभाषय GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#नोचेद्यदि defined(__VXE__) || defined(__VXE2__)

#परिभाषय GGML_SIMD

// F32 s390x

#परिभाषय GGML_F32_चरणः 32
#परिभाषय GGML_F32_EPR  4

#परिभाषय GGML_F32x4              float32x4_t
#परिभाषय GGML_F32x4_ZERO         vec_splats(0.0f)
#परिभाषय GGML_F32x4_SET1         vec_splats
#परिभाषय GGML_F32x4_आरोपय(p)      vec_xl(0, p)
#परिभाषय GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#परिभाषय GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#परिभाषय GGML_F32x4_सङ्कलय          vec_सङ्कलय
#परिभाषय GGML_F32x4_MUL          vec_mul
#परिभाषय GGML_F32x4_REDUCE(res, x)                   \
{                                                   \
    पूर्णाङ्क विस्थापनम् = GGML_F32_ARR >> 1;                 \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {              \
        x[i] = vec_सङ्कलय(x[i], x[विस्थापनम् + i]);        \
    }                                               \
    विस्थापनम् >>= 1;                                   \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {              \
        x[i] = vec_सङ्कलय(x[i], x[विस्थापनम् + i]);        \
    }                                               \
    विस्थापनम् >>= 1;                                   \
    प्रति (पूर्णाङ्क i = 0; i < विस्थापनम्; ++i) {              \
        x[i] = vec_सङ्कलय(x[i], x[विस्थापनम् + i]);        \
    }                                               \
    float32x4_t tmp = x[0] + vec_reve(x[0]);        \
    res = tmp[0] + tmp[1];                          \
}
#परिभाषय GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3) \
{                                                \
    float32x4_t v = vec_सङ्कलय(vec_सङ्कलय(s0, s1),     \
                            vec_सङ्कलय(s2, s3));    \
    v = vec_सङ्कलय(v, vec_sld(v, v, 8));            \
    v = vec_सङ्कलय(v, vec_sld(v, v, 4));            \
    res += (ggml_float)vec_extract(v, 0);        \
}

#परिभाषय GGML_F32_VEC        GGML_F32x4
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x4_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x4_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x4_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x4_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x4_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x4_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 s390x
#परिभाषय GGML_F16_चरणः GGML_F32_चरणः
#परिभाषय GGML_F16_EPR  GGML_F32_EPR

स्थिर अन्तःस्थ uint32x4_t __lzs_f32cx4_to_f16(float32x4_t v_f) {
    float32x4_t v_आधारः = vec_mul(vec_mul(vec_abs(v_f), vec_splats(0x1.0p+112f)), vec_splats(0x1.0p-110f));

    ध्रुव uint32x4_t v_w      = (uint32x4_t)v_f;
    ध्रुव uint32x4_t v_shl1_w = vec_सङ्कलय(v_w, v_w);
    ध्रुव uint32x4_t v_sign   = vec_and(v_w, vec_splats(UINT32_C(0x80000000)));
    ध्रुव uint32x4_t v_bias   = vec_max(vec_and(v_shl1_w, vec_splats(UINT32_C(0xFF000000))), vec_splats(UINT32_C(0x71000000)));

    v_आधारः = vec_सङ्कलय((float32x4_t)vec_सङ्कलय(vec_sr(v_bias, 1), vec_splats(UINT32_C(0x07800000))), v_आधारः);

    ध्रुव uint32x4_t v_द्वयङ्काः    = (uint32x4_t)v_आधारः;
    ध्रुव uint32x4_t v_nonsign = vec_सङ्कलय(vec_and(vec_sr(v_द्वयङ्काः, 13), vec_splats(UINT32_C(0x00007C00))),
                                         vec_and(v_द्वयङ्काः, vec_splats(UINT32_C(0x00000FFF))));
    ध्रुव uint32x4_t v_is_nan  = (uint32x4_t)vec_cmpgt(v_shl1_w, vec_splats(UINT32_C(0xFF000000)));

    प्रत्यागम vec_or(vec_sr(v_sign, 16), vec_sel(v_nonsign, vec_splats(UINT32_C(0x7E00)), v_is_nan));
}

स्थिर अन्तःस्थ float32x4_t __lzs_f16cx4_load(ध्रुव ggml_fp16_t * x) {
    भिन्न tmp[4];

    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_गन्तव्यम्_FP32(x[i]);
    }

    // note: keep type-cast here to prevent compiler bugs
    // see: https://github.com/ggml-org/llama.cpp/issues/12846
    प्रत्यागम vec_xl(0, (ध्रुव भिन्न *)(tmp));
}

स्थिर अन्तःस्थ शून्य __lzs_f16cx4_store(ggml_fp16_t * x, float32x4_t v_y) {
    ध्रुव uint32x4_t v_h = __lzs_f32cx4_to_f16(v_y);
    ध्रुव uint64_t   tmp = ((uint64x2_t)vec_pack(v_h, v_h))[0];
    स्मृतिं_प्रतिलिख(x, &tmp, परिमाण(tmp));
}

#परिभाषय GGML_F16_VEC                GGML_F32x4
#परिभाषय GGML_F16_VEC_ZERO           GGML_F32x4_ZERO
#परिभाषय GGML_F16_VEC_SET1           GGML_F32x4_SET1
#परिभाषय GGML_F16_VEC_आरोपय(p, i)     __lzs_f16cx4_load(p)
#परिभाषय GGML_F16_VEC_STORE(p, r, i) __lzs_f16cx4_store(p, r[i])
#परिभाषय GGML_F16_VEC_FMA            GGML_F32x4_FMA
#परिभाषय GGML_F16_VEC_सङ्कलय            GGML_F32x4_सङ्कलय
#परिभाषय GGML_F16_VEC_MUL            GGML_F32x4_MUL
#परिभाषय GGML_F16_VEC_REDUCE         GGML_F32x4_REDUCE

// BF16 s390x
#परिभाषय GGML_BF16_चरणः 16
#परिभाषय GGML_BF16_EPR  8

#परिभाषय GGML_BF16x8         __vector अचिह्नित ह्रस्व
#परिभाषय GGML_BF16x8_ZERO    vec_splats((अचिह्नित ह्रस्व)0)
#परिभाषय GGML_BF16x8_आरोपय(p) vec_xl(0, (ध्रुव अचिह्नित ह्रस्व *)(p))

#परिभाषय GGML_BF16_VEC      GGML_BF16x8
#परिभाषय GGML_BF16_VEC_ZERO GGML_BF16x8_ZERO
#परिभाषय GGML_BF16_VEC_आरोपय GGML_BF16x8_आरोपय
#परिभाषय GGML_BF16_गन्तव्यम्_F32_LO(v) ((float32x4_t) vec_mergel((v), GGML_BF16_VEC_ZERO))
#परिभाषय GGML_BF16_गन्तव्यम्_F32_HI(v) ((float32x4_t) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#परिभाषय GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_गन्तव्यम्_F32_LO(x), GGML_BF16_गन्तव्यम्_F32_LO(y))
#परिभाषय GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_गन्तव्यम्_F32_HI(x), GGML_BF16_गन्तव्यम्_F32_HI(y))

#नोचेद्यदि defined(__riscv_v_intrinsic)

// compatible with vlen >= 128

#परिभाषय GGML_SIMD

// F32

#परिभाषय GGML_F32_चरणः 16
#परिभाषय GGML_F32_EPR  4

#परिभाषय GGML_F32x4              vfloat32m1_t
#परिभाषय GGML_F32x4_ZERO         __riscv_vfmv_v_f_f32m1(0.0f, GGML_F32_EPR)
#परिभाषय GGML_F32x4_SET1(x)      __riscv_vfmv_v_f_f32m1(x, GGML_F32_EPR)
#परिभाषय GGML_F32x4_आरोपय(x)      __riscv_vle32_v_f32m1(x, GGML_F32_EPR)
#परिभाषय GGML_F32x4_STORE(b, v)  __riscv_vse32_v_f32m1(b, v, GGML_F32_EPR)
#परिभाषय GGML_F32x4_FMA(a, b, c) __riscv_vfmacc_vv_f32m1(a, b, c, GGML_F32_EPR)
#परिभाषय GGML_F32x4_सङ्कलय(a, b)    __riscv_vfadd_vv_f32m1(a, b, GGML_F32_EPR)
#परिभाषय GGML_F32x4_MUL(a, b)    __riscv_vfmul_vv_f32m1(a, b, GGML_F32_EPR)

#परिभाषय GGML_F32_VEC        GGML_F32x4
#परिभाषय GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#परिभाषय GGML_F32_VEC_SET1   GGML_F32x4_SET1
#परिभाषय GGML_F32_VEC_आरोपय   GGML_F32x4_आरोपय
#परिभाषय GGML_F32_VEC_STORE  GGML_F32x4_STORE
#परिभाषय GGML_F32_VEC_FMA    GGML_F32x4_FMA
#परिभाषय GGML_F32_VEC_सङ्कलय    GGML_F32x4_सङ्कलय
#परिभाषय GGML_F32_VEC_MUL    GGML_F32x4_MUL
#परिभाषय GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

#यदि_अन्तः

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#यदि_परिभाषितम् GGML_SIMD
#परिभाषय GGML_F32_ARR (GGML_F32_चरणः/GGML_F32_EPR)
#परिभाषय GGML_F16_ARR (GGML_F16_चरणः/GGML_F16_EPR)
#यदि_अन्तः

// GGML_F16_DOT_*
// like GGML_F16_* but for dot products which need F32 accumulation on AVX512-FP16

#यदि defined(__AVX512FP16__)

#परिभाषय GGML_F16_DOT_चरणः           GGML_F32_चरणः
#परिभाषय GGML_F16_DOT_EPR            GGML_F32_EPR
#परिभाषय GGML_F16_DOT_ARR            GGML_F32_ARR
#परिभाषय GGML_F16_DOT_VEC            GGML_F32x16
#परिभाषय GGML_F16_DOT_VEC_ZERO       GGML_F32x16_ZERO
#परिभाषय GGML_F16_DOT_VEC_आरोपय(p, i) _mm512_cvtph_ps(_mm256_loadu_si256((ध्रुव __m256i *)(p)))
#परिभाषय GGML_F16_DOT_VEC_FMA        GGML_F32x16_FMA
#परिभाषय GGML_F16_DOT_VEC_REDUCE     GGML_F32x16_REDUCE

#अन्यथा

#परिभाषय GGML_F16_DOT_चरणः           GGML_F16_चरणः
#परिभाषय GGML_F16_DOT_EPR            GGML_F16_EPR
#परिभाषय GGML_F16_DOT_ARR            GGML_F16_ARR
#परिभाषय GGML_F16_DOT_VEC            GGML_F16_VEC
#परिभाषय GGML_F16_DOT_VEC_ZERO       GGML_F16_VEC_ZERO
#परिभाषय GGML_F16_DOT_VEC_आरोपय       GGML_F16_VEC_आरोपय
#परिभाषय GGML_F16_DOT_VEC_FMA        GGML_F16_VEC_FMA
#परिभाषय GGML_F16_DOT_VEC_REDUCE     GGML_F16_VEC_REDUCE

#यदि_अन्तः // defined(__AVX512FP16__)

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः
