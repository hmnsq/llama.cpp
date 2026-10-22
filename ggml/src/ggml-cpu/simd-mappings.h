#आशय once

#समावेश "ggml-cpu-impl.h"

#अगर_घोषित __ARM_FEATURE_SVE
#समावेश <arm_sve.h>
#पूर्ण_अगर // __ARM_FEATURE_SVE

#अगर defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#समावेश <arm_neon.h>
#पूर्ण_अगर

#अगर defined(__riscv_v_intrinsic)
#समावेश <riscv_vector.h>
#पूर्ण_अगर

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

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
#अगर defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
    #घोषणा GGML_CPU_COMPUTE_FP16_TO_FP32(x) neon_compute_fp16_to_fp32(x)
    #घोषणा GGML_CPU_COMPUTE_FP32_TO_FP16(x) neon_compute_fp32_to_fp16(x)

    #घोषणा GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
    #घोषणा GGML_CPU_FP32_TO_FP16(x) GGML_CPU_COMPUTE_FP32_TO_FP16(x)

    अटल अंतरभूत भग्न neon_compute_fp16_to_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        स_नकल(&tmp, &h, माप(ggml_fp16_t));
        वापस (भग्न)tmp;
    }

    अटल अंतरभूत ggml_fp16_t neon_compute_fp32_to_fp16(भग्न f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        स_नकल(&res, &tmp, माप(ggml_fp16_t));
        वापस res;
    }
#या_अगर defined(__F16C__)
    #अगर_घोषित _MSC_VER
        #घोषणा GGML_CPU_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #घोषणा GGML_CPU_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
    #अन्यथा
        #घोषणा GGML_CPU_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
        #घोषणा GGML_CPU_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
    #पूर्ण_अगर
#या_अगर defined(__POWER9_VECTOR__)
    #घोषणा GGML_CPU_COMPUTE_FP16_TO_FP32(x) power_compute_fp16_to_fp32(x)
    #घोषणा GGML_CPU_COMPUTE_FP32_TO_FP16(x) power_compute_fp32_to_fp16(x)
    /* the inline asm below is about 12% faster than the lookup method */
    #घोषणा GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
    #घोषणा GGML_CPU_FP32_TO_FP16(x) GGML_CPU_COMPUTE_FP32_TO_FP16(x)

    अटल अंतरभूत भग्न power_compute_fp16_to_fp32(ggml_fp16_t h) {
        भग्न f;
        द्विगुन d;
        __asm__(
            "mtfprd %0,%2\n"
            "xscvhpdp %0,%0\n"
            "frsp %1,%0\n" :
            /*अस्थायी*/ "=d"(d),
            /* out */  "=f"(f):
            /* in */   "r"(h));
        वापस f;
    }

    अटल अंतरभूत ggml_fp16_t power_compute_fp32_to_fp16(भग्न f) {
        द्विगुन d;
        ggml_fp16_t r;
        __asm__( /* xscvdphp can work on double or single precision */
            "xscvdphp %0,%2\n"
            "mffprd %1,%0\n" :
            /*अस्थायी*/ "=d"(d),
            /* out */  "=r"(r):
            /* in */   "f"(f));
        वापस r;
    }
#या_अगर defined(__riscv) && defined(__riscv_zfhmin)
    अटल अंतरभूत भग्न riscv_compute_fp16_to_fp32(ggml_fp16_t h) {
        _Float16 hf;
        स_नकल(&hf, &h, माप(ggml_fp16_t));
        वापस hf;
    }

    अटल अंतरभूत ggml_fp16_t riscv_compute_fp32_to_fp16(भग्न f) {
        ggml_fp16_t res;
        _Float16 hf = (_Float16)f;
        स_नकल(&res, &hf, माप(ggml_fp16_t));
        वापस res;
    }

    #घोषणा GGML_CPU_COMPUTE_FP16_TO_FP32(x) riscv_compute_fp16_to_fp32(x)
    #घोषणा GGML_CPU_COMPUTE_FP32_TO_FP16(x) riscv_compute_fp32_to_fp16(x)
    #घोषणा GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
    #घोषणा GGML_CPU_FP32_TO_FP16(x) GGML_CPU_COMPUTE_FP32_TO_FP16(x)
#पूर्ण_अगर

// precomputed f32 table for f16 (256 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
बाह्य भग्न ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
बाह्य भग्न ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
बाह्य भग्न ggml_table_f32_ue4m3[1 << 8];

// Use lookup table for E8M0 on x86 (faster than bit manipulation)
#अगर defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#घोषणा GGML_CPU_E8M0_TO_FP32_HALF(x) ggml_table_f32_e8m0_half[(uint8_t)(x)]
#अन्यथा
#घोषणा GGML_CPU_E8M0_TO_FP32_HALF(x) GGML_E8M0_TO_FP32_HALF(x)
#पूर्ण_अगर

// Use lookup table for UE4M3 on x86 and ARM (faster than bit manipulation)
#अगर defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__ARM_NEON)
#घोषणा GGML_CPU_UE4M3_TO_FP32(x) ggml_table_f32_ue4m3[(uint8_t)(x)]
#अन्यथा
#घोषणा GGML_CPU_UE4M3_TO_FP32(x) ggml_ue4m3_to_fp32(x)
#पूर्ण_अगर

// Pre AVX512 CPUs (AVX2 with F16C, such as i5-8th gen or i7-4790)
// do not produce codegen as checks below don't test for __F16C__
// but this check does.
#अगर defined(__F16C__)
// force FP16 -> FP32 conversion to use LUT instead
// see: https://github.com/ggml-org/llama.cpp/issues/30342
//#define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
#घोषणा GGML_CPU_FP32_TO_FP16(x) GGML_CPU_COMPUTE_FP32_TO_FP16(x)
#पूर्ण_अगर

// a branch that defines a GGML_CPU_COMPUTE_* macro must also select it, or the fallback takes over silently
#अगर defined(GGML_CPU_COMPUTE_FP16_TO_FP32) && !defined(GGML_CPU_FP16_TO_FP32) && !defined(__F16C__) // see: https://github.com/ggml-org/llama.cpp/issues/30342
#त्रुटि "a branch defined GGML_CPU_COMPUTE_FP16_TO_FP32 without selecting it as GGML_CPU_FP16_TO_FP32"
#पूर्ण_अगर
#अगर defined(GGML_CPU_COMPUTE_FP32_TO_FP16) && !defined(GGML_CPU_FP32_TO_FP16)
#त्रुटि "a branch defined GGML_CPU_COMPUTE_FP32_TO_FP16 without selecting it as GGML_CPU_FP32_TO_FP16"
#पूर्ण_अगर

// fall back to the lookup table / the portable software conversion for whichever direction no branch selected
#अगर !defined(GGML_CPU_FP16_TO_FP32)
अंतरभूत अटल भग्न ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    स_नकल(&s, &f, माप(uint16_t));
    वापस ggml_table_f32_f16[s];
}

#घोषणा GGML_CPU_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#पूर्ण_अगर

#अगर !defined(GGML_CPU_FP32_TO_FP16)
#घोषणा GGML_CPU_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
#पूर्ण_अगर


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

#अगर defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_FMA)

#घोषणा GGML_SIMD

// F32 SVE
#घोषणा GGML_F32_EPR 8
#घोषणा DEFAULT_PG svptrue_b32()

#घोषणा GGML_F32xt                        svfloat32_t
#घोषणा GGML_F32xt_ZERO                   svdup_n_f32(0.0f)
#घोषणा GGML_F32xt_SET1(x)                svdup_n_f32(x)
#घोषणा GGML_F32xt_LOAD_IMPL(pg, a)       svld1_f32(pg, a)
#घोषणा GGML_F32xt_LOAD(a)                GGML_F32xt_LOAD_IMPL(DEFAULT_PG, a)
#घोषणा GGML_F32xt_STORE_IMPL(pg, a, b)   svst1_f32(pg, a, b)
#घोषणा GGML_F32xt_STORE(a, b)            GGML_F32xt_STORE_IMPL(DEFAULT_PG, a, b)
#घोषणा GGML_F32xt_FMA_IMPL(pg, a, b, c)  svmad_f32_m(pg, b, c, a)
#घोषणा GGML_F32xt_FMA(a, b, c)           GGML_F32xt_FMA_IMPL(DEFAULT_PG, a, b, c)
#घोषणा GGML_F32xt_ADD_IMPL(pg, a, b)     svadd_f32_m(pg, a, b)
#घोषणा GGML_F32xt_ADD(a, b)              GGML_F32xt_ADD_IMPL(DEFAULT_PG, a, b)
#घोषणा GGML_F32xt_MUL_IMPL(pg, a, b)     svmul_f32_m(pg, a, b)
#घोषणा GGML_F32xt_MUL(a, b)              GGML_F32xt_MUL_IMPL(DEFAULT_PG, a, b)
#घोषणा GGML_F32xt_REDUCE_ONE_IMPL(pg, a) svaddv(pg, a)
#घोषणा GGML_F32xt_REDUCE_ONE(a)          GGML_F32xt_REDUCE_ONE_IMPL(DEFAULT_PG, a)
#घोषणा GGML_F32xt_REDUCE_IMPL(pg, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
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
#घोषणा GGML_F32xt_REDUCE(res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
        GGML_F32xt_REDUCE_IMPL(DEFAULT_PG, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)

#घोषणा GGML_F32_VEC        GGML_F32xt
#घोषणा GGML_F32_VEC_ZERO   GGML_F32xt_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32xt_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32xt_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32xt_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32xt_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32xt_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32xt_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32xt_REDUCE

// F16 SVE
#घोषणा DEFAULT_PG32    svptrue_b32()
#घोषणा DEFAULT_PG16    svptrue_b16()

#घोषणा GGML_F32Cxt                         svfloat16_t
#घोषणा GGML_F32Cxt_ZERO                    svdup_n_f16(0.0f)
#घोषणा GGML_F32Cxt_SET1(x)                 svdup_n_f16(x)
#घोषणा GGML_F32Cxt_LOAD(p)                 svld1_f16(DEFAULT_PG16, (स्थिर __fp16 *)(p))
#घोषणा GGML_F32Cxt_STORE(dst_ptr, src_vec) svst1_f16(DEFAULT_PG16, (__fp16 *)(dst_ptr), (src_vec))

#घोषणा GGML_F32Cxt_FMA_IMPL(pg, a, b, c)   svmad_f16_x(pg, b, c, a)
#घोषणा GGML_F32Cxt_FMA(a, b, c)            GGML_F32Cxt_FMA_IMPL(DEFAULT_PG16, a, b, c)
#घोषणा GGML_F32Cxt_ADD_IMPL(pg, a, b)      svadd_f16_x(pg, a, b)
#घोषणा GGML_F32Cxt_ADD(a, b)               GGML_F32Cxt_ADD_IMPL(DEFAULT_PG16, a, b)
#घोषणा GGML_F32Cxt_MUL_IMPL(pg, a, b)      svmul_f16_x(pg, a, b)
#घोषणा GGML_F32Cxt_MUL(a, b)               GGML_F32Cxt_MUL_IMPL(DEFAULT_PG16, a, b)
#घोषणा GGML_F32Cxt_REDUCE                  GGML_F16xt_REDUCE_MIXED

#घोषणा GGML_F16x_VEC                GGML_F32Cxt
#घोषणा GGML_F16x_VEC_ZERO           GGML_F32Cxt_ZERO
#घोषणा GGML_F16x_VEC_SET1           GGML_F32Cxt_SET1
#घोषणा GGML_F16x_VEC_LOAD(p, i)     GGML_F32Cxt_LOAD(p)
#घोषणा GGML_F16x_VEC_STORE(p, r, i) GGML_F32Cxt_STORE((__fp16 *)(p), r)
#घोषणा GGML_F16x_VEC_FMA            GGML_F32Cxt_FMA
#घोषणा GGML_F16x_VEC_ADD            GGML_F32Cxt_ADD
#घोषणा GGML_F16x_VEC_MUL            GGML_F32Cxt_MUL
#घोषणा GGML_F16x_VEC_REDUCE         GGML_F32Cxt_REDUCE

#घोषणा GGML_F16xt_REDUCE_ONE_IMPL(pg, a) svaddv_f16(pg, a)
#घोषणा GGML_F16xt_REDUCE_ONE(a)          GGML_F16xt_REDUCE_ONE_IMPL(DEFAULT_PG16, a)

#घोषणा GGML_F16xt_REDUCE_MIXED_IMPL(pg16, res, sum1, sum2, sum3, sum4)  \
{                                                      \
    sum1 = svadd_f16_x(pg16, sum1, sum2);              \
    sum3 = svadd_f16_x(pg16, sum3, sum4);              \
    sum1 = svadd_f16_x(pg16, sum1, sum3);              \
    __fp16 योग_f16 = svaddv_f16(pg16, sum1);           \
    (res) = (ggml_float) योग_f16;                      \
}
#घोषणा GGML_F16xt_REDUCE_MIXED(res, sum1, sum2, sum3, sum4)  \
        GGML_F16xt_REDUCE_MIXED_IMPL(DEFAULT_PG16, res, sum1, sum2, sum3, sum4)

// F16 NEON

#अगर defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #घोषणा GGML_F16_STEP 32
    #घोषणा GGML_F16_EPR  8

    #घोषणा GGML_F16x8              float16x8_t
    #घोषणा GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #घोषणा GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #घोषणा GGML_F16x8_LOAD(x)      vld1q_f16((स्थिर __fp16 *)(x))
    #घोषणा GGML_F16x8_STORE        vst1q_f16
    #घोषणा GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #घोषणा GGML_F16x8_ADD          vaddq_f16
    #घोषणा GGML_F16x8_MUL          vmulq_f16
    #घोषणा GGML_F16x8_REDUCE(res, x)                               \
    करो {                                                            \
        पूर्णांक offset = GGML_F16_ARR >> 1;                             \
        क्रम (पूर्णांक i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        क्रम (पूर्णांक i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        क्रम (पूर्णांक i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        स्थिर float32x4_t t0 = vcvt_f32_f16(vget_निचला_f16 ((x)[0])); \
        स्थिर float32x4_t t1 = vcvt_f32_f16(vget_ऊपरी_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } जबतक (0)

    #घोषणा GGML_F16_VEC                GGML_F16x8
    #घोषणा GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #घोषणा GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #घोषणा GGML_F16_VEC_LOAD(p, i)     GGML_F16x8_LOAD(p)
    #घोषणा GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #घोषणा GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #घोषणा GGML_F16_VEC_ADD            GGML_F16x8_ADD
    #घोषणा GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #घोषणा GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#अन्यथा
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #घोषणा GGML_F16_STEP 16
    #घोषणा GGML_F16_EPR  4

    #घोषणा GGML_F32Cx4              float32x4_t
    #घोषणा GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #घोषणा GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #घोषणा GGML_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16((स्थिर __fp16 *)(x)))
    #घोषणा GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #घोषणा GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #घोषणा GGML_F32Cx4_ADD          vaddq_f32
    #घोषणा GGML_F32Cx4_MUL          vmulq_f32
    #घोषणा GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #घोषणा GGML_F16_VEC                GGML_F32Cx4
    #घोषणा GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #घोषणा GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #घोषणा GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx4_LOAD(p)
    #घोषणा GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #घोषणा GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #घोषणा GGML_F16_VEC_ADD            GGML_F32Cx4_ADD
    #घोषणा GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #घोषणा GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#पूर्ण_अगर

#या_अगर defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)

#घोषणा GGML_SIMD

// F32 NEON

#घोषणा GGML_F32_STEP 16
#घोषणा GGML_F32_EPR  4

#घोषणा GGML_F32x4              float32x4_t
#घोषणा GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#घोषणा GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#घोषणा GGML_F32x4_LOAD         vld1q_f32
#घोषणा GGML_F32x4_STORE        vst1q_f32
#घोषणा GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#घोषणा GGML_F32x4_ADD          vaddq_f32
#घोषणा GGML_F32x4_MUL          vmulq_f32
#घोषणा GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#घोषणा GGML_F32x4_REDUCE(res, x)                       \
{                                                       \
    पूर्णांक offset = GGML_F32_ARR >> 1;                     \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
    }                                                   \
    offset >>= 1;                                       \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
    }                                                   \
    offset >>= 1;                                       \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
    }                                                   \
    (res) = (ggml_float) GGML_F32x4_REDUCE_ONE((x)[0]); \
}

#घोषणा GGML_F32_VEC        GGML_F32x4
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x4_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x4_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x4_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x4_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x4_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#अगर defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #घोषणा GGML_F16_STEP 32
    #घोषणा GGML_F16_EPR  8

    #घोषणा GGML_F16x8              float16x8_t
    #घोषणा GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #घोषणा GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #घोषणा GGML_F16x8_LOAD(x)      vld1q_f16((स्थिर __fp16 *)(x))
    #घोषणा GGML_F16x8_STORE        vst1q_f16
    #घोषणा GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #घोषणा GGML_F16x8_ADD          vaddq_f16
    #घोषणा GGML_F16x8_MUL          vmulq_f16
    #घोषणा GGML_F16x8_REDUCE(res, x)                               \
    करो {                                                            \
        पूर्णांक offset = GGML_F16_ARR >> 1;                             \
        क्रम (पूर्णांक i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        क्रम (पूर्णांक i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        क्रम (पूर्णांक i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        स्थिर float32x4_t t0 = vcvt_f32_f16(vget_निचला_f16 ((x)[0])); \
        स्थिर float32x4_t t1 = vcvt_f32_f16(vget_ऊपरी_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } जबतक (0)

    #घोषणा GGML_F16_VEC                GGML_F16x8
    #घोषणा GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #घोषणा GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #घोषणा GGML_F16_VEC_LOAD(p, i)     GGML_F16x8_LOAD(p)
    #घोषणा GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #घोषणा GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #घोषणा GGML_F16_VEC_ADD            GGML_F16x8_ADD
    #घोषणा GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #घोषणा GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#अन्यथा
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #घोषणा GGML_F16_STEP 16
    #घोषणा GGML_F16_EPR  4

    #घोषणा GGML_F32Cx4              float32x4_t
    #घोषणा GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #घोषणा GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #घोषणा GGML_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16((स्थिर __fp16 *)(x)))
    #घोषणा GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #घोषणा GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #घोषणा GGML_F32Cx4_ADD          vaddq_f32
    #घोषणा GGML_F32Cx4_MUL          vmulq_f32
    #घोषणा GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #घोषणा GGML_F16_VEC                GGML_F32Cx4
    #घोषणा GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #घोषणा GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #घोषणा GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx4_LOAD(p)
    #घोषणा GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #घोषणा GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #घोषणा GGML_F16_VEC_ADD            GGML_F32Cx4_ADD
    #घोषणा GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #घोषणा GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#पूर्ण_अगर

#या_अगर defined(__AVX512F__)

#घोषणा GGML_SIMD

// F32 AVX512

#घोषणा GGML_F32_STEP 64
#घोषणा GGML_F32_EPR  16

#घोषणा GGML_F32x16         __m512
#घोषणा GGML_F32x16_ZERO    _mm512_setzero_ps()
#घोषणा GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#घोषणा GGML_F32x16_LOAD    _mm512_loadu_ps
#घोषणा GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#घोषणा GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#घोषणा GGML_F32x16_ADD     _mm512_add_ps
#घोषणा GGML_F32x16_MUL     _mm512_mul_ps
#घोषणा GGML_F32x16_REDUCE(res, x)                                    \
करो {                                                                  \
    पूर्णांक offset = GGML_F32_ARR >> 1;                                   \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    res = (ggml_float) _mm512_reduce_add_ps(x[0]);                    \
} जबतक (0)

// TODO: is this optimal ?

#घोषणा GGML_F32_VEC        GGML_F32x16
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x16_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x16_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x16_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x16_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x16_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x16_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x16_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512

#अगर defined(__AVX512FP16__)

#घोषणा GGML_F16_STEP 128
#घोषणा GGML_F16_EPR  32

#घोषणा GGML_F16x32              __m512h
#घोषणा GGML_F16x32_ZERO         _mm512_setzero_ph()
#घोषणा GGML_F16x32_SET1(x)      _mm512_set1_ph(__extension__(_Float16)(x))
#घोषणा GGML_F16x32_LOAD(x)      _mm512_loadu_ph(x)
#घोषणा GGML_F16x32_STORE(x, y)  _mm512_storeu_ph(x, y)
#घोषणा GGML_F16x32_FMA(a, b, c) _mm512_fmadd_ph(b, c, a)
#घोषणा GGML_F16x32_ADD          _mm512_add_ph
#घोषणा GGML_F16x32_MUL          _mm512_mul_ph
#घोषणा GGML_F16x32_REDUCE(res, x)                                     \
करो {                                                                   \
    पूर्णांक offset = GGML_F16_ARR >> 1;                                    \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                                 \
        x[i] = _mm512_add_ph(x[i], x[offset+i]);                       \
    }                                                                  \
    offset >>= 1;                                                      \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                                 \
        x[i] = _mm512_add_ph(x[i], x[offset+i]);                       \
    }                                                                  \
    offset >>= 1;                                                      \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                                 \
        x[i] = _mm512_add_ph(x[i], x[offset+i]);                       \
    }                                                                  \
    res = (ggml_float) _mm512_reduce_add_ph(x[0]);                     \
} जबतक (0)

#घोषणा GGML_F16_VEC                GGML_F16x32
#घोषणा GGML_F16_VEC_ZERO           GGML_F16x32_ZERO
#घोषणा GGML_F16_VEC_SET1           GGML_F16x32_SET1
#घोषणा GGML_F16_VEC_LOAD(p, i)     GGML_F16x32_LOAD(p)
#घोषणा GGML_F16_VEC_STORE(p, r, i) GGML_F16x32_STORE(p, r[i])
#घोषणा GGML_F16_VEC_FMA            GGML_F16x32_FMA
#घोषणा GGML_F16_VEC_ADD            GGML_F16x32_ADD
#घोषणा GGML_F16_VEC_MUL            GGML_F16x32_MUL
#घोषणा GGML_F16_VEC_REDUCE         GGML_F16x32_REDUCE

#अन्यथा // Fallback FP16 <-> FP32

#घोषणा GGML_F16_STEP 64
#घोषणा GGML_F16_EPR  16

#घोषणा GGML_F32Cx16             __m512
#घोषणा GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#घोषणा GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#घोषणा GGML_F32Cx16_LOAD(x)     _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(x)))
#घोषणा GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#घोषणा GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#घोषणा GGML_F32Cx16_ADD         _mm512_add_ps
#घोषणा GGML_F32Cx16_MUL         _mm512_mul_ps
#घोषणा GGML_F32Cx16_REDUCE(res, x)                               \
करो {                                                              \
    पूर्णांक offset = GGML_F32_ARR >> 1;                               \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    res = (ggml_float) _mm512_reduce_add_ps(x[0]);                \
} जबतक (0)

#घोषणा GGML_F16_VEC                GGML_F32Cx16
#घोषणा GGML_F16_VEC_ZERO           GGML_F32Cx16_ZERO
#घोषणा GGML_F16_VEC_SET1           GGML_F32Cx16_SET1
#घोषणा GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx16_LOAD(p)
#घोषणा GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#घोषणा GGML_F16_VEC_FMA            GGML_F32Cx16_FMA
#घोषणा GGML_F16_VEC_ADD            GGML_F32Cx16_ADD
#घोषणा GGML_F16_VEC_MUL            GGML_F32Cx16_MUL

#घोषणा GGML_F16_VEC_REDUCE         GGML_F32Cx16_REDUCE

#पूर्ण_अगर // __AVX512FP16__
#या_अगर defined(__AVX__)

#घोषणा GGML_SIMD

// F32 AVX

#घोषणा GGML_F32_STEP 32
#घोषणा GGML_F32_EPR  8

#घोषणा GGML_F32x8         __m256
#घोषणा GGML_F32x8_ZERO    _mm256_setzero_ps()
#घोषणा GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#घोषणा GGML_F32x8_LOAD    _mm256_loadu_ps
#घोषणा GGML_F32x8_STORE   _mm256_storeu_ps
#अगर defined(__FMA__)
    #घोषणा GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#अन्यथा
    #घोषणा GGML_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#पूर्ण_अगर
#घोषणा GGML_F32x8_ADD     _mm256_add_ps
#घोषणा GGML_F32x8_MUL     _mm256_mul_ps
#घोषणा GGML_F32x8_REDUCE(res, x)                                 \
करो {                                                              \
    पूर्णांक offset = GGML_F32_ARR >> 1;                               \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    स्थिर __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    स्थिर __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} जबतक (0)
// TODO: is this optimal ?

#घोषणा GGML_F32_VEC        GGML_F32x8
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x8_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x8_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x8_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x8_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x8_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x8_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 AVX

#घोषणा GGML_F16_STEP 32
#घोषणा GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#घोषणा GGML_F32Cx8             __m256
#घोषणा GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#घोषणा GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#अगर defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#घोषणा GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((स्थिर __m128i *)(x)))
#घोषणा GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#अन्यथा
अटल अंतरभूत __m256 __avx_f32cx8_load(स्थिर ggml_fp16_t * x) {
    भग्न tmp[8];

    क्रम (पूर्णांक i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    वापस _mm256_loadu_ps(tmp);
}
अटल अंतरभूत व्योम __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    भग्न arr[8];

    _mm256_storeu_ps(arr, y);

    क्रम (पूर्णांक i = 0; i < 8; i++)
        x[i] = GGML_CPU_FP32_TO_FP16(arr[i]);
}
#घोषणा GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#घोषणा GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#पूर्ण_अगर

#घोषणा GGML_F32Cx8_FMA         GGML_F32x8_FMA
#घोषणा GGML_F32Cx8_ADD         _mm256_add_ps
#घोषणा GGML_F32Cx8_MUL         _mm256_mul_ps
#घोषणा GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#घोषणा GGML_F16_VEC                GGML_F32Cx8
#घोषणा GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#घोषणा GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#घोषणा GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx8_LOAD(p)
#घोषणा GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#घोषणा GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#घोषणा GGML_F16_VEC_ADD            GGML_F32Cx8_ADD
#घोषणा GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#घोषणा GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#या_अगर defined(__POWER9_VECTOR__)

#घोषणा GGML_SIMD

// F32 POWER9

#घोषणा GGML_F32_STEP 32
#घोषणा GGML_F32_EPR  4

#घोषणा GGML_F32x4              vector भग्न
#घोषणा GGML_F32x4_ZERO         {0.0f}
#घोषणा GGML_F32x4_SET1         vec_splats
#घोषणा GGML_F32x4_LOAD(p)      vec_xl(0, p)
#घोषणा GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#घोषणा GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#घोषणा GGML_F32x4_ADD          vec_add
#घोषणा GGML_F32x4_MUL          vec_mul
#घोषणा GGML_F32x4_REDUCE(res, x)              \
{                                              \
    पूर्णांक offset = GGML_F32_ARR >> 1;            \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    offset >>= 1;                              \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    offset >>= 1;                              \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    res = vec_extract(x[0], 0) +               \
          vec_extract(x[0], 1) +               \
          vec_extract(x[0], 2) +               \
          vec_extract(x[0], 3);                \
}
#घोषणा GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3)        \
{                                                       \
    vector भग्न v = vec_add(vec_add(s0, s1),           \
                             vec_add(s2, s3));          \
    v = vec_add(v, vec_sld(v, v, 8));                   \
    v = vec_add(v, vec_sld(v, v, 4));                   \
    res += (ggml_float) vec_extract(v, 0);              \
}

#घोषणा GGML_F32_VEC        GGML_F32x4
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x4_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x4_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x4_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x4_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x4_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 POWER9
#घोषणा GGML_F16_STEP       GGML_F32_STEP
#घोषणा GGML_F16_EPR        GGML_F32_EPR
#घोषणा GGML_F16_VEC        GGML_F32x4
#घोषणा GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#घोषणा GGML_F16_VEC_SET1   GGML_F32x4_SET1
#घोषणा GGML_F16_VEC_FMA    GGML_F32x4_FMA
#घोषणा GGML_F16_VEC_ADD    GGML_F32x4_ADD
#घोषणा GGML_F16_VEC_MUL    GGML_F32x4_MUL
#घोषणा GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#घोषणा GGML_F16_VEC_LOAD(p, i) (i & 0x1) ?                   \
  vec_extract_fp32_from_shorth(vec_xl(0, p - GGML_F16_EPR)) : \
  vec_extract_fp32_from_shortl(vec_xl(0, p))
अटल अंतरभूत अचिन्हित अक्षर ggml_endian_byte(पूर्णांक i) {
       uint16_t tmp_val = 1;
       वापस ((अचिन्हित अक्षर *)&tmp_val)[i];
}
#घोषणा GGML_ENDIAN_BYTE(i) ggml_endian_byte(i)
#घोषणा GGML_F16_VEC_STORE(p, r, i)                             \
  अगर (i & 0x1)                                                  \
    vec_xst(vec_pack_to_short_fp32(r[i - GGML_ENDIAN_BYTE(1)],  \
                                   r[i - GGML_ENDIAN_BYTE(0)]), \
            0, p - GGML_F16_EPR)

//BF16 POWER9
#घोषणा GGML_BF16_STEP 16
#घोषणा GGML_BF16_EPR  8

#घोषणा GGML_BF16x8         vector अचिन्हित लघु
#घोषणा GGML_BF16x8_ZERO    vec_splats((अचिन्हित लघु)0)
#घोषणा GGML_BF16x8_LOAD(p) vec_xl(0, (स्थिर अचिन्हित लघु *)(p))

#घोषणा GGML_BF16_VEC          GGML_BF16x8
#घोषणा GGML_BF16_VEC_ZERO     GGML_BF16x8_ZERO
#घोषणा GGML_BF16_VEC_LOAD     GGML_BF16x8_LOAD
#अगर defined(__LITTLE_ENDIAN__)
#घोषणा GGML_BF16_TO_F32_LO(v) ((vector भग्न) vec_mergel(GGML_BF16_VEC_ZERO, (v)))
#घोषणा GGML_BF16_TO_F32_HI(v) ((vector भग्न) vec_mergeh(GGML_BF16_VEC_ZERO, (v)))
#अन्यथा
#घोषणा GGML_BF16_TO_F32_LO(v) ((vector भग्न) vec_mergel((v), GGML_BF16_VEC_ZERO))
#घोषणा GGML_BF16_TO_F32_HI(v) ((vector भग्न) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#पूर्ण_अगर
#घोषणा GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_TO_F32_LO(x), GGML_BF16_TO_F32_LO(y))
#घोषणा GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_TO_F32_HI(x), GGML_BF16_TO_F32_HI(y))

#या_अगर defined(__wasm_simd128__)

#घोषणा GGML_SIMD

// F32 WASM

#घोषणा GGML_F32_STEP 16
#घोषणा GGML_F32_EPR  4

#घोषणा GGML_F32x4              v128_t
#घोषणा GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#घोषणा GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#घोषणा GGML_F32x4_LOAD         wasm_v128_load
#घोषणा GGML_F32x4_STORE        wasm_v128_store
#घोषणा GGML_F32x4_FMA(a, b, c) wasm_f32x4_add(wasm_f32x4_mul(b, c), a)
#घोषणा GGML_F32x4_ADD          wasm_f32x4_add
#घोषणा GGML_F32x4_MUL          wasm_f32x4_mul
#घोषणा GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    पूर्णांक offset = GGML_F32_ARR >> 1;                \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#घोषणा GGML_F32_VEC        GGML_F32x4
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x4_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x4_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x4_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x4_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x4_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 WASM

#घोषणा GGML_F16_STEP 16
#घोषणा GGML_F16_EPR  4

अंतरभूत अटल v128_t __wasm_f16x4_load(स्थिर ggml_fp16_t * p) {
    भग्न tmp[4];

    tmp[0] = GGML_CPU_FP16_TO_FP32(p[0]);
    tmp[1] = GGML_CPU_FP16_TO_FP32(p[1]);
    tmp[2] = GGML_CPU_FP16_TO_FP32(p[2]);
    tmp[3] = GGML_CPU_FP16_TO_FP32(p[3]);

    वापस wasm_v128_load(tmp);
}

अंतरभूत अटल व्योम __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
    भग्न tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = GGML_CPU_FP32_TO_FP16(tmp[0]);
    p[1] = GGML_CPU_FP32_TO_FP16(tmp[1]);
    p[2] = GGML_CPU_FP32_TO_FP16(tmp[2]);
    p[3] = GGML_CPU_FP32_TO_FP16(tmp[3]);
}

#घोषणा GGML_F16x4             v128_t
#घोषणा GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#घोषणा GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#घोषणा GGML_F16x4_LOAD(x)     __wasm_f16x4_load(x)
#घोषणा GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#घोषणा GGML_F16x4_FMA         GGML_F32x4_FMA
#घोषणा GGML_F16x4_ADD         wasm_f32x4_add
#घोषणा GGML_F16x4_MUL         wasm_f32x4_mul
#घोषणा GGML_F16x4_REDUCE(res, x)                           \
{                                                           \
    पूर्णांक offset = GGML_F16_ARR >> 1;                         \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                      \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);           \
    }                                                       \
    offset >>= 1;                                           \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                      \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);           \
    }                                                       \
    offset >>= 1;                                           \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                      \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);           \
    }                                                       \
    res = (ggml_float) (wasm_f32x4_extract_lane(x[0], 0) +  \
          wasm_f32x4_extract_lane(x[0], 1) +                \
          wasm_f32x4_extract_lane(x[0], 2) +                \
          wasm_f32x4_extract_lane(x[0], 3));                \
}

#घोषणा GGML_F16_VEC                GGML_F16x4
#घोषणा GGML_F16_VEC_ZERO           GGML_F16x4_ZERO
#घोषणा GGML_F16_VEC_SET1           GGML_F16x4_SET1
#घोषणा GGML_F16_VEC_LOAD(p, i)     GGML_F16x4_LOAD(p)
#घोषणा GGML_F16_VEC_STORE(p, r, i) GGML_F16x4_STORE(p, r[i])
#घोषणा GGML_F16_VEC_FMA            GGML_F16x4_FMA
#घोषणा GGML_F16_VEC_ADD            GGML_F16x4_ADD
#घोषणा GGML_F16_VEC_MUL            GGML_F16x4_MUL
#घोषणा GGML_F16_VEC_REDUCE         GGML_F16x4_REDUCE

#या_अगर defined(__SSE3__)

#घोषणा GGML_SIMD

// F32 SSE

#घोषणा GGML_F32_STEP 32
#घोषणा GGML_F32_EPR  4

#घोषणा GGML_F32x4         __m128
#घोषणा GGML_F32x4_ZERO    _mm_setzero_ps()
#घोषणा GGML_F32x4_SET1(x) _mm_set1_ps(x)
#घोषणा GGML_F32x4_LOAD    _mm_loadu_ps
#घोषणा GGML_F32x4_STORE   _mm_storeu_ps
#अगर defined(__FMA__)
    // TODO: Does this work?
    #घोषणा GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#अन्यथा
    #घोषणा GGML_F32x4_FMA(a, b, c) _mm_add_ps(_mm_mul_ps(b, c), a)
#पूर्ण_अगर
#घोषणा GGML_F32x4_ADD     _mm_add_ps
#घोषणा GGML_F32x4_MUL     _mm_mul_ps
#घोषणा GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    पूर्णांक offset = GGML_F32_ARR >> 1;                               \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    स्थिर __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#घोषणा GGML_F32_VEC        GGML_F32x4
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x4_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x4_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x4_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x4_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x4_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 SSE

#घोषणा GGML_F16_STEP 32
#घोषणा GGML_F16_EPR  4

अटल अंतरभूत __m128 __sse_f16x4_load(स्थिर ggml_fp16_t * x) {
    भग्न tmp[4];

    tmp[0] = GGML_CPU_FP16_TO_FP32(x[0]);
    tmp[1] = GGML_CPU_FP16_TO_FP32(x[1]);
    tmp[2] = GGML_CPU_FP16_TO_FP32(x[2]);
    tmp[3] = GGML_CPU_FP16_TO_FP32(x[3]);

    वापस _mm_loadu_ps(tmp);
}

अटल अंतरभूत व्योम __sse_f16x4_store(ggml_fp16_t * x, __m128 y) {
    भग्न arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = GGML_CPU_FP32_TO_FP16(arr[0]);
    x[1] = GGML_CPU_FP32_TO_FP16(arr[1]);
    x[2] = GGML_CPU_FP32_TO_FP16(arr[2]);
    x[3] = GGML_CPU_FP32_TO_FP16(arr[3]);
}

#घोषणा GGML_F32Cx4             __m128
#घोषणा GGML_F32Cx4_ZERO        _mm_setzero_ps()
#घोषणा GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#घोषणा GGML_F32Cx4_LOAD(x)     __sse_f16x4_load(x)
#घोषणा GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#घोषणा GGML_F32Cx4_FMA         GGML_F32x4_FMA
#घोषणा GGML_F32Cx4_ADD         _mm_add_ps
#घोषणा GGML_F32Cx4_MUL         _mm_mul_ps
#घोषणा GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#घोषणा GGML_F16_VEC                 GGML_F32Cx4
#घोषणा GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#घोषणा GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#घोषणा GGML_F16_VEC_LOAD(p, i)      GGML_F32Cx4_LOAD(p)
#घोषणा GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#घोषणा GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#घोषणा GGML_F16_VEC_ADD             GGML_F32Cx4_ADD
#घोषणा GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#घोषणा GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#या_अगर defined(__loongarch_asx)

#घोषणा GGML_SIMD

// F32 LASX
#घोषणा GGML_F32_STEP 32
#घोषणा GGML_F32_EPR  8

#घोषणा GGML_F32x8         __m256
#घोषणा GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#घोषणा GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#घोषणा GGML_F32x8_LOAD(x) (__m256)__lasx_xvld((x), 0)
#घोषणा GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#घोषणा GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#घोषणा GGML_F32x8_ADD     __lasx_xvfadd_s
#घोषणा GGML_F32x8_MUL     __lasx_xvfmul_s
#घोषणा GGML_F32x8_REDUCE(res, x)                                 \
करो {                                                              \
    पूर्णांक offset = GGML_F32_ARR >> 1;                               \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    भग्न *tmp_p = (भग्न *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} जबतक (0)
// TODO: is this optimal ?

#घोषणा GGML_F32_VEC        GGML_F32x8
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x8_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x8_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x8_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x8_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x8_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x8_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 LASX

#घोषणा GGML_F16_STEP 32
#घोषणा GGML_F16_EPR  8

// F16 arithmetic is not supported by LASX, so we use F32 instead

#घोषणा GGML_F32Cx8          __m256
#घोषणा GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#घोषणा GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))

अटल अंतरभूत __m256 __lasx_f32cx8_load(स्थिर ggml_fp16_t * x) {
    __m256i a;
    स_नकल(&a, x, माप(ggml_fp16_t) * 8);
    a = __lasx_xvpermi_d(a, 0 | (1 << 4));
    वापस __lasx_xvfcvtl_s_h(a);
}

अटल अंतरभूत व्योम __lasx_f32cx8_store(ggml_fp16_t * x, __m256 y) {
    __m256i a = __lasx_xvfcvt_h_s(y, y);
    a = __lasx_xvpermi_d(a, 0 | (2 << 2));
    स_नकल(x, &a, माप(ggml_fp16_t) * 8);
}
#घोषणा GGML_F32Cx8_LOAD(x)     __lasx_f32cx8_load(x)
#घोषणा GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#घोषणा GGML_F32Cx8_FMA         GGML_F32x8_FMA
#घोषणा GGML_F32Cx8_ADD         __lasx_xvfadd_s
#घोषणा GGML_F32Cx8_MUL         __lasx_xvfmul_s
#घोषणा GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#घोषणा GGML_F16_VEC                GGML_F32Cx8
#घोषणा GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#घोषणा GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#घोषणा GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx8_LOAD(p)
#घोषणा GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#घोषणा GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#घोषणा GGML_F16_VEC_ADD            GGML_F32Cx8_ADD
#घोषणा GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#घोषणा GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#या_अगर defined(__loongarch_sx)

#घोषणा GGML_SIMD

// F32 LSX

#घोषणा GGML_F32_STEP 32
#घोषणा GGML_F32_EPR  4

#घोषणा GGML_F32x4         __m128
#घोषणा GGML_F32x4_ZERO    (__m128)__lsx_vldi(0)
#घोषणा GGML_F32x4_SET1(x) (__m128)__lsx_vreplfr2vr_s((x))
#घोषणा GGML_F32x4_LOAD(x) (__m128)__lsx_vld((x), 0)
#घोषणा GGML_F32x4_STORE(x, y)   __lsx_vst(y, x, 0)
#घोषणा GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#घोषणा GGML_F32x4_ADD     __lsx_vfadd_s
#घोषणा GGML_F32x4_MUL     __lsx_vfmul_s

#घोषणा GGML_F32x4_REDUCE(res, x)                               \
{                                                               \
    पूर्णांक offset = GGML_F32_ARR >> 1;                             \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[offset+i]);                \
    }                                                           \
    offset >>= 1;                                               \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[offset+i]);                \
    }                                                           \
    offset >>= 1;                                               \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[offset+i]);                \
    }                                                           \
    __m128i t0 = __lsx_vpickev_w((__m128i)x[0], (__m128i)x[0]); \
    __m128i t1 = __lsx_vpickod_w((__m128i)x[0], (__m128i)x[0]); \
    __m128 t2 = __lsx_vfadd_s((__m128)t0, (__m128)t1);          \
    __m128i t3 = __lsx_vpickev_w((__m128i)t2, (__m128i)t2);     \
    __m128i t4 = __lsx_vpickod_w((__m128i)t2, (__m128i)t2);     \
    __m128 t5 = __lsx_vfadd_s((__m128)t3, (__m128)t4);          \
    res = (ggml_float) ((v4f32)t5)[0];                          \
}

#घोषणा GGML_F32_VEC        GGML_F32x4
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x4_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x4_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x4_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x4_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x4_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 LSX

#घोषणा GGML_F16_STEP 32
#घोषणा GGML_F16_EPR  4

अटल अंतरभूत __m128 __lsx_f16x4_load(स्थिर ggml_fp16_t * x) {
    वापस __lsx_vfcvtl_s_h(__lsx_vld((स्थिर व्योम *)x, 0));
}

अटल अंतरभूत व्योम __lsx_f16x4_store(ggml_fp16_t * x, __m128 y) {
    __m128i a = __lsx_vfcvt_h_s(y, y);
    स_नकल(x, &a, माप(ggml_fp16_t) * 4);
}

#घोषणा GGML_F32Cx4             __m128
#घोषणा GGML_F32Cx4_ZERO        (__m128)__lsx_vldi(0)
#घोषणा GGML_F32Cx4_SET1(x)     (__m128)__lsx_vreplfr2vr_s((x))
#घोषणा GGML_F32Cx4_LOAD(x)     (__m128)__lsx_f16x4_load(x)
#घोषणा GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#घोषणा GGML_F32Cx4_FMA         GGML_F32x4_FMA
#घोषणा GGML_F32Cx4_ADD         __lsx_vfadd_s
#घोषणा GGML_F32Cx4_MUL         __lsx_vfmul_s
#घोषणा GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#घोषणा GGML_F16_VEC                 GGML_F32Cx4
#घोषणा GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#घोषणा GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#घोषणा GGML_F16_VEC_LOAD(p, i)      GGML_F32Cx4_LOAD(p)
#घोषणा GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#घोषणा GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#घोषणा GGML_F16_VEC_ADD             GGML_F32Cx4_ADD
#घोषणा GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#घोषणा GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#या_अगर defined(__VXE__) || defined(__VXE2__)

#घोषणा GGML_SIMD

// F32 s390x

#घोषणा GGML_F32_STEP 32
#घोषणा GGML_F32_EPR  4

#घोषणा GGML_F32x4              float32x4_t
#घोषणा GGML_F32x4_ZERO         vec_splats(0.0f)
#घोषणा GGML_F32x4_SET1         vec_splats
#घोषणा GGML_F32x4_LOAD(p)      vec_xl(0, p)
#घोषणा GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#घोषणा GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#घोषणा GGML_F32x4_ADD          vec_add
#घोषणा GGML_F32x4_MUL          vec_mul
#घोषणा GGML_F32x4_REDUCE(res, x)                   \
{                                                   \
    पूर्णांक offset = GGML_F32_ARR >> 1;                 \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {              \
        x[i] = vec_add(x[i], x[offset + i]);        \
    }                                               \
    offset >>= 1;                                   \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {              \
        x[i] = vec_add(x[i], x[offset + i]);        \
    }                                               \
    offset >>= 1;                                   \
    क्रम (पूर्णांक i = 0; i < offset; ++i) {              \
        x[i] = vec_add(x[i], x[offset + i]);        \
    }                                               \
    float32x4_t tmp = x[0] + vec_reve(x[0]);        \
    res = tmp[0] + tmp[1];                          \
}
#घोषणा GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3) \
{                                                \
    float32x4_t v = vec_add(vec_add(s0, s1),     \
                            vec_add(s2, s3));    \
    v = vec_add(v, vec_sld(v, v, 8));            \
    v = vec_add(v, vec_sld(v, v, 4));            \
    res += (ggml_float)vec_extract(v, 0);        \
}

#घोषणा GGML_F32_VEC        GGML_F32x4
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x4_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x4_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x4_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x4_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x4_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 s390x
#घोषणा GGML_F16_STEP GGML_F32_STEP
#घोषणा GGML_F16_EPR  GGML_F32_EPR

अटल अंतरभूत uint32x4_t __lzs_f32cx4_to_f16(float32x4_t v_f) {
    float32x4_t v_base = vec_mul(vec_mul(vec_abs(v_f), vec_splats(0x1.0p+112f)), vec_splats(0x1.0p-110f));

    स्थिर uint32x4_t v_w      = (uint32x4_t)v_f;
    स्थिर uint32x4_t v_shl1_w = vec_add(v_w, v_w);
    स्थिर uint32x4_t v_sign   = vec_and(v_w, vec_splats(UINT32_C(0x80000000)));
    स्थिर uint32x4_t v_bias   = vec_max(vec_and(v_shl1_w, vec_splats(UINT32_C(0xFF000000))), vec_splats(UINT32_C(0x71000000)));

    v_base = vec_add((float32x4_t)vec_add(vec_sr(v_bias, 1), vec_splats(UINT32_C(0x07800000))), v_base);

    स्थिर uint32x4_t v_bits    = (uint32x4_t)v_base;
    स्थिर uint32x4_t v_nonsign = vec_add(vec_and(vec_sr(v_bits, 13), vec_splats(UINT32_C(0x00007C00))),
                                         vec_and(v_bits, vec_splats(UINT32_C(0x00000FFF))));
    स्थिर uint32x4_t v_is_nan  = (uint32x4_t)vec_cmpgt(v_shl1_w, vec_splats(UINT32_C(0xFF000000)));

    वापस vec_or(vec_sr(v_sign, 16), vec_sel(v_nonsign, vec_splats(UINT32_C(0x7E00)), v_is_nan));
}

अटल अंतरभूत float32x4_t __lzs_f16cx4_load(स्थिर ggml_fp16_t * x) {
    भग्न tmp[4];

    क्रम (पूर्णांक i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    // note: keep type-cast here to prevent compiler bugs
    // see: https://github.com/ggml-org/llama.cpp/issues/12846
    वापस vec_xl(0, (स्थिर भग्न *)(tmp));
}

अटल अंतरभूत व्योम __lzs_f16cx4_store(ggml_fp16_t * x, float32x4_t v_y) {
    स्थिर uint32x4_t v_h = __lzs_f32cx4_to_f16(v_y);
    स्थिर uint64_t   tmp = ((uint64x2_t)vec_pack(v_h, v_h))[0];
    स_नकल(x, &tmp, माप(tmp));
}

#घोषणा GGML_F16_VEC                GGML_F32x4
#घोषणा GGML_F16_VEC_ZERO           GGML_F32x4_ZERO
#घोषणा GGML_F16_VEC_SET1           GGML_F32x4_SET1
#घोषणा GGML_F16_VEC_LOAD(p, i)     __lzs_f16cx4_load(p)
#घोषणा GGML_F16_VEC_STORE(p, r, i) __lzs_f16cx4_store(p, r[i])
#घोषणा GGML_F16_VEC_FMA            GGML_F32x4_FMA
#घोषणा GGML_F16_VEC_ADD            GGML_F32x4_ADD
#घोषणा GGML_F16_VEC_MUL            GGML_F32x4_MUL
#घोषणा GGML_F16_VEC_REDUCE         GGML_F32x4_REDUCE

// BF16 s390x
#घोषणा GGML_BF16_STEP 16
#घोषणा GGML_BF16_EPR  8

#घोषणा GGML_BF16x8         __vector अचिन्हित लघु
#घोषणा GGML_BF16x8_ZERO    vec_splats((अचिन्हित लघु)0)
#घोषणा GGML_BF16x8_LOAD(p) vec_xl(0, (स्थिर अचिन्हित लघु *)(p))

#घोषणा GGML_BF16_VEC      GGML_BF16x8
#घोषणा GGML_BF16_VEC_ZERO GGML_BF16x8_ZERO
#घोषणा GGML_BF16_VEC_LOAD GGML_BF16x8_LOAD
#घोषणा GGML_BF16_TO_F32_LO(v) ((float32x4_t) vec_mergel((v), GGML_BF16_VEC_ZERO))
#घोषणा GGML_BF16_TO_F32_HI(v) ((float32x4_t) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#घोषणा GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_TO_F32_LO(x), GGML_BF16_TO_F32_LO(y))
#घोषणा GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_TO_F32_HI(x), GGML_BF16_TO_F32_HI(y))

#या_अगर defined(__riscv_v_intrinsic)

// compatible with vlen >= 128

#घोषणा GGML_SIMD

// F32

#घोषणा GGML_F32_STEP 16
#घोषणा GGML_F32_EPR  4

#घोषणा GGML_F32x4              vfloat32m1_t
#घोषणा GGML_F32x4_ZERO         __riscv_vfmv_v_f_f32m1(0.0f, GGML_F32_EPR)
#घोषणा GGML_F32x4_SET1(x)      __riscv_vfmv_v_f_f32m1(x, GGML_F32_EPR)
#घोषणा GGML_F32x4_LOAD(x)      __riscv_vle32_v_f32m1(x, GGML_F32_EPR)
#घोषणा GGML_F32x4_STORE(b, v)  __riscv_vse32_v_f32m1(b, v, GGML_F32_EPR)
#घोषणा GGML_F32x4_FMA(a, b, c) __riscv_vfmacc_vv_f32m1(a, b, c, GGML_F32_EPR)
#घोषणा GGML_F32x4_ADD(a, b)    __riscv_vfadd_vv_f32m1(a, b, GGML_F32_EPR)
#घोषणा GGML_F32x4_MUL(a, b)    __riscv_vfmul_vv_f32m1(a, b, GGML_F32_EPR)

#घोषणा GGML_F32_VEC        GGML_F32x4
#घोषणा GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#घोषणा GGML_F32_VEC_SET1   GGML_F32x4_SET1
#घोषणा GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#घोषणा GGML_F32_VEC_STORE  GGML_F32x4_STORE
#घोषणा GGML_F32_VEC_FMA    GGML_F32x4_FMA
#घोषणा GGML_F32_VEC_ADD    GGML_F32x4_ADD
#घोषणा GGML_F32_VEC_MUL    GGML_F32x4_MUL
#घोषणा GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

#पूर्ण_अगर

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#अगर_घोषित GGML_SIMD
#घोषणा GGML_F32_ARR (GGML_F32_STEP/GGML_F32_EPR)
#घोषणा GGML_F16_ARR (GGML_F16_STEP/GGML_F16_EPR)
#पूर्ण_अगर

// GGML_F16_DOT_*
// like GGML_F16_* but for dot products which need F32 accumulation on AVX512-FP16

#अगर defined(__AVX512FP16__)

#घोषणा GGML_F16_DOT_STEP           GGML_F32_STEP
#घोषणा GGML_F16_DOT_EPR            GGML_F32_EPR
#घोषणा GGML_F16_DOT_ARR            GGML_F32_ARR
#घोषणा GGML_F16_DOT_VEC            GGML_F32x16
#घोषणा GGML_F16_DOT_VEC_ZERO       GGML_F32x16_ZERO
#घोषणा GGML_F16_DOT_VEC_LOAD(p, i) _mm512_cvtph_ps(_mm256_loadu_si256((स्थिर __m256i *)(p)))
#घोषणा GGML_F16_DOT_VEC_FMA        GGML_F32x16_FMA
#घोषणा GGML_F16_DOT_VEC_REDUCE     GGML_F32x16_REDUCE

#अन्यथा

#घोषणा GGML_F16_DOT_STEP           GGML_F16_STEP
#घोषणा GGML_F16_DOT_EPR            GGML_F16_EPR
#घोषणा GGML_F16_DOT_ARR            GGML_F16_ARR
#घोषणा GGML_F16_DOT_VEC            GGML_F16_VEC
#घोषणा GGML_F16_DOT_VEC_ZERO       GGML_F16_VEC_ZERO
#घोषणा GGML_F16_DOT_VEC_LOAD       GGML_F16_VEC_LOAD
#घोषणा GGML_F16_DOT_VEC_FMA        GGML_F16_VEC_FMA
#घोषणा GGML_F16_DOT_VEC_REDUCE     GGML_F16_VEC_REDUCE

#पूर्ण_अगर // defined(__AVX512FP16__)

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर
