#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __ARM_FEATURE_SVE
#ಅನ್ತರ್ಭಾವ <arm_sve.h>
#ಆದರೆ_ಅಂತ್ಯ // __ARM_FEATURE_SVE

#ಆದರೆ defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ಅನ್ತರ್ಭಾವ <arm_neon.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__riscv_v_intrinsic)
#ಅನ್ತರ್ಭಾವ <riscv_vector.h>
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

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
#ಆದರೆ defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP16_ಗಮ್ಯ_FP32(x) neon_compute_fp16_ಗಮ್ಯ_fp32(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16(x) neon_compute_fp32_ಗಮ್ಯ_fp16(x)

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_FP16_ಗಮ್ಯ_FP32(x) GGML_CPU_COMPUTE_FP16_ಗಮ್ಯ_FP32(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_FP32_ಗಮ್ಯ_FP16(x) GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16(x)

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ neon_compute_fp16_ಗಮ್ಯ_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        ಸ್ಮರಣೆ_ನಕಲು(&tmp, &h, ಗಾತ್ರ(ggml_fp16_t));
        ಹಿಂತಿರುಗಿ (ದಶಮಾಂಶ)tmp;
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_fp16_t neon_compute_fp32_ಗಮ್ಯ_fp16(ದಶಮಾಂಶ f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        ಸ್ಮರಣೆ_ನಕಲು(&res, &tmp, ಗಾತ್ರ(ggml_fp16_t));
        ಹಿಂತಿರುಗಿ res;
    }
#ಅಥವಾಆದರೆ defined(__F16C__)
    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _MSC_VER
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP16_ಗಮ್ಯ_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(mm_ಗಣ_ss(x), 0), 0)
    #ಇಲ್ಲದಿದ್ದರೆ
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP16_ಗಮ್ಯ_FP32(x) _cvtsh_ss(x)
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16(x) _cvtss_sh(x, 0)
    #ಆದರೆ_ಅಂತ್ಯ
#ಅಥವಾಆದರೆ defined(__POWER9_VECTOR__)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP16_ಗಮ್ಯ_FP32(x) power_compute_fp16_ಗಮ್ಯ_fp32(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16(x) power_compute_fp32_ಗಮ್ಯ_fp16(x)
    /* the inline asm below is about 12% faster than the lookup method */
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_FP16_ಗಮ್ಯ_FP32(x) GGML_CPU_COMPUTE_FP16_ಗಮ್ಯ_FP32(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_FP32_ಗಮ್ಯ_FP16(x) GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16(x)

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ power_compute_fp16_ಗಮ್ಯ_fp32(ggml_fp16_t h) {
        ದಶಮಾಂಶ f;
        ದ್ವಿಗುಣ d;
        __asm__(
            "mtfprd %0,%2\n"
            "xscvhpdp %0,%0\n"
            "frsp %1,%0\n" :
            /*ತಾತ್ಕಾಲಿಕ*/ "=d"(d),
            /*ಹೊರಹೋಗು*/  "=f"(f):
            /*ಒಳ*/   "r"(h));
        ಹಿಂತಿರುಗಿ f;
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_fp16_t power_compute_fp32_ಗಮ್ಯ_fp16(ದಶಮಾಂಶ f) {
        ದ್ವಿಗುಣ d;
        ggml_fp16_t r;
        __asm__( /* xscvdphp can work on double or single precision */
            "xscvdphp %0,%2\n"
            "mffprd %1,%0\n" :
            /*ತಾತ್ಕಾಲಿಕ*/ "=d"(d),
            /*ಹೊರಹೋಗು*/  "=r"(r):
            /*ಒಳ*/   "f"(f));
        ಹಿಂತಿರುಗಿ r;
    }
#ಅಥವಾಆದರೆ defined(__riscv) && defined(__riscv_zfhmin)
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ riscv_compute_fp16_ಗಮ್ಯ_fp32(ggml_fp16_t h) {
        _Float16 hf;
        ಸ್ಮರಣೆ_ನಕಲು(&hf, &h, ಗಾತ್ರ(ggml_fp16_t));
        ಹಿಂತಿರುಗಿ hf;
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_fp16_t riscv_compute_fp32_ಗಮ್ಯ_fp16(ದಶಮಾಂಶ f) {
        ggml_fp16_t res;
        _Float16 hf = (_Float16)f;
        ಸ್ಮರಣೆ_ನಕಲು(&res, &hf, ಗಾತ್ರ(ggml_fp16_t));
        ಹಿಂತಿರುಗಿ res;
    }

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP16_ಗಮ್ಯ_FP32(x) riscv_compute_fp16_ಗಮ್ಯ_fp32(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16(x) riscv_compute_fp32_ಗಮ್ಯ_fp16(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_FP16_ಗಮ್ಯ_FP32(x) GGML_CPU_COMPUTE_FP16_ಗಮ್ಯ_FP32(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_FP32_ಗಮ್ಯ_FP16(x) GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16(x)
#ಆದರೆ_ಅಂತ್ಯ

// precomputed f32 table for f16 (256 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
ಬಾಹ್ಯ ದಶಮಾಂಶ ggml_ಕೋಷ್ಟಕ_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
ಬಾಹ್ಯ ದಶಮಾಂಶ ggml_ಕೋಷ್ಟಕ_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
ಬಾಹ್ಯ ದಶಮಾಂಶ ggml_ಕೋಷ್ಟಕ_f32_ue4m3[1 << 8];

// Use lookup table for E8M0 on x86 (faster than bit manipulation)
#ಆದರೆ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_E8_M0_ಗಮ್ಯ_FP32_HALF(x) ggml_ಕೋಷ್ಟಕ_f32_e8m0_half[(uint8_t)(x)]
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_E8_M0_ಗಮ್ಯ_FP32_HALF(x) GGML_E8_M0_ಗಮ್ಯ_FP32_HALF(x)
#ಆದರೆ_ಅಂತ್ಯ

// Use lookup table for UE4M3 on x86 and ARM (faster than bit manipulation)
#ಆದರೆ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__ARM_NEON)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_UE4_M3_ಗಮ್ಯ_FP32(x) ggml_ಕೋಷ್ಟಕ_f32_ue4m3[(uint8_t)(x)]
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_UE4_M3_ಗಮ್ಯ_FP32(x) ggml_ue4m3_ಗಮ್ಯ_fp32(x)
#ಆದರೆ_ಅಂತ್ಯ

// Pre AVX512 CPUs (AVX2 with F16C, such as i5-8th gen or i7-4790)
// do not produce codegen as checks below don't test for __F16C__
// but this check does.
#ಆದರೆ defined(__F16C__)
// force FP16 -> FP32 conversion to use LUT instead
// see: https://github.com/ggml-org/llama.cpp/issues/30342
//#define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_FP32_ಗಮ್ಯ_FP16(x) GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16(x)
#ಆದರೆ_ಅಂತ್ಯ

// a branch that defines a GGML_CPU_COMPUTE_* macro must also select it, or the fallback takes over silently
#ಆದರೆ defined(GGML_CPU_COMPUTE_FP16_ಗಮ್ಯ_FP32) && !defined(GGML_CPU_FP16_ಗಮ್ಯ_FP32) && !defined(__F16C__) // see: https://github.com/ggml-org/llama.cpp/issues/30342
#ದೋಷ "a branch defined GGML_CPU_COMPUTE_FP16_TO_FP32 without selecting it as GGML_CPU_FP16_TO_FP32"
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_CPU_COMPUTE_FP32_ಗಮ್ಯ_FP16) && !defined(GGML_CPU_FP32_ಗಮ್ಯ_FP16)
#ದೋಷ "a branch defined GGML_CPU_COMPUTE_FP32_TO_FP16 without selecting it as GGML_CPU_FP32_TO_FP16"
#ಆದರೆ_ಅಂತ್ಯ

// fall back to the lookup table / the portable software conversion for whichever direction no branch selected
#ಆದರೆ !defined(GGML_CPU_FP16_ಗಮ್ಯ_FP32)
ಸಾಲೊಳಗೆ ಸ್ಥಿರ ದಶಮಾಂಶ ggml_lookup_fp16_ಗಮ್ಯ_fp32(ggml_fp16_t f) {
    uint16_t s;
    ಸ್ಮರಣೆ_ನಕಲು(&s, &f, ಗಾತ್ರ(uint16_t));
    ಹಿಂತಿರುಗಿ ggml_ಕೋಷ್ಟಕ_f32_f16[s];
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_FP16_ಗಮ್ಯ_FP32(x) ggml_lookup_fp16_ಗಮ್ಯ_fp32(x)
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ !defined(GGML_CPU_FP32_ಗಮ್ಯ_FP16)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CPU_FP32_ಗಮ್ಯ_FP16(x) GGML_COMPUTE_FP32_ಗಮ್ಯ_FP16(x)
#ಆದರೆ_ಅಂತ್ಯ


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

#ಆದರೆ defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_FMA)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 SVE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR 8
#ವ್ಯಾಖ್ಯಾನಿಸು DEFAULT_PG svptrue_b32()

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt                        svfloat32_t
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_ZERO                   svdup_n_f32(0.0f)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_SET1(x)                svdup_n_f32(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_ತುಂಬು_IMPL(pg, a)       svld1_f32(pg, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_ತುಂಬು(a)                GGML_F32xt_ತುಂಬು_IMPL(DEFAULT_PG, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_STORE_IMPL(pg, a, b)   svst1_f32(pg, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_STORE(a, b)            GGML_F32xt_STORE_IMPL(DEFAULT_PG, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_FMA_IMPL(pg, a, b, c)  svmad_f32_m(pg, b, c, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_FMA(a, b, c)           GGML_F32xt_FMA_IMPL(DEFAULT_PG, a, b, c)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_ಕೂಡಿಸು_IMPL(pg, a, b)     svadd_f32_m(pg, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_ಕೂಡಿಸು(a, b)              GGML_F32xt_ಕೂಡಿಸು_IMPL(DEFAULT_PG, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_MUL_IMPL(pg, a, b)     svmul_f32_m(pg, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_MUL(a, b)              GGML_F32xt_MUL_IMPL(DEFAULT_PG, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_REDUCE_ONE_IMPL(pg, a) svaddv(pg, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_REDUCE_ONE(a)          GGML_F32xt_REDUCE_ONE_IMPL(DEFAULT_PG, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_REDUCE_IMPL(pg, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
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
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32xt_REDUCE(res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
        GGML_F32xt_REDUCE_IMPL(DEFAULT_PG, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32xt
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32xt_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32xt_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32xt_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32xt_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32xt_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32xt_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32xt_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32xt_REDUCE

// F16 SVE
#ವ್ಯಾಖ್ಯಾನಿಸು DEFAULT_PG32    svptrue_b32()
#ವ್ಯಾಖ್ಯಾನಿಸು DEFAULT_PG16    svptrue_b16()

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cxt                         svfloat16_t
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cxt_ZERO                    svdup_n_f16(0.0f)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cxt_SET1(x)                 svdup_n_f16(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cxt_ತುಂಬು(p)                 svld1_f16(DEFAULT_PG16, (ಸ್ಥಿರಾಂಕ __fp16 *)(p))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cxt_STORE(dst_ptr, src_vec) svst1_f16(DEFAULT_PG16, (__fp16 *)(dst_ptr), (src_vec))

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cxt_FMA_IMPL(pg, a, b, c)   svmad_f16_x(pg, b, c, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cxt_FMA(a, b, c)            GGML_F32Cxt_FMA_IMPL(DEFAULT_PG16, a, b, c)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cxt_ಕೂಡಿಸು_IMPL(pg, a, b)      svadd_f16_x(pg, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cxt_ಕೂಡಿಸು(a, b)               GGML_F32_Cxt_ಕೂಡಿಸು_IMPL(DEFAULT_PG16, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cxt_MUL_IMPL(pg, a, b)      svmul_f16_x(pg, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cxt_MUL(a, b)               GGML_F32Cxt_MUL_IMPL(DEFAULT_PG16, a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cxt_REDUCE                  GGML_F16xt_REDUCE_MIXED

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x_VEC                GGML_F32Cxt
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x_VEC_ZERO           GGML_F32Cxt_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x_VEC_SET1           GGML_F32Cxt_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x_VEC_ತುಂಬು(p, i)     GGML_F32_Cxt_ತುಂಬು(p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x_VEC_STORE(p, r, i) GGML_F32Cxt_STORE((__fp16 *)(p), r)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x_VEC_FMA            GGML_F32Cxt_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x_VEC_ಕೂಡಿಸು            GGML_F32_Cxt_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x_VEC_MUL            GGML_F32Cxt_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x_VEC_REDUCE         GGML_F32Cxt_REDUCE

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16xt_REDUCE_ONE_IMPL(pg, a) svaddv_f16(pg, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16xt_REDUCE_ONE(a)          GGML_F16xt_REDUCE_ONE_IMPL(DEFAULT_PG16, a)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16xt_REDUCE_MIXED_IMPL(pg16, res, sum1, sum2, sum3, sum4)  \
{                                                      \
    sum1 = svadd_f16_x(pg16, sum1, sum2);              \
    sum3 = svadd_f16_x(pg16, sum3, sum4);              \
    sum1 = svadd_f16_x(pg16, sum1, sum3);              \
    __fp16 ಮೊತ್ತ_f16 = svaddv_f16(pg16, sum1);           \
    (res) = (ggml_float) ಮೊತ್ತ_f16;                      \
}
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16xt_REDUCE_MIXED(res, sum1, sum2, sum3, sum4)  \
        GGML_F16xt_REDUCE_MIXED_IMPL(DEFAULT_PG16, res, sum1, sum2, sum3, sum4)

// F16 NEON

#ಆದರೆ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 32
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  8

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8              float16x8_t
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_ತುಂಬು(x)      vld1q_f16((ಸ್ಥಿರಾಂಕ __fp16 *)(x))
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_STORE        vst1q_f16
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_ಕೂಡಿಸು          vaddq_f16
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_MUL          vmulq_f16
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_REDUCE(res, x)                               \
    ಮಾಡು {                                                            \
        ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F16_ARR >> 1;                             \
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ಸ್ಥಾನಾಂತರ+i]);              \
        }                                                           \
        ಸ್ಥಾನಾಂತರ >>= 1;                                               \
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ಸ್ಥಾನಾಂತರ+i]);              \
        }                                                           \
        ಸ್ಥಾನಾಂತರ >>= 1;                                               \
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ಸ್ಥಾನಾಂತರ+i]);              \
        }                                                           \
        ಸ್ಥಿರಾಂಕ float32x4_t t0 = vcvt_f32_f16(vget_ಕೆಳಗಿನ_f16 ((x)[0])); \
        ಸ್ಥಿರಾಂಕ float32x4_t t1 = vcvt_f32_f16(vget_ಮೇಲಿನ_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } ಆಗ (0)

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F16x8
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     GGML_F16x8_ತುಂಬು(p)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F16x8_ಕೂಡಿಸು
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#ಇಲ್ಲದಿದ್ದರೆ
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 16
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  4

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4              float32x4_t
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx4_ತುಂಬು(x)      vcvt_f32_f16(vld1_f16((ಸ್ಥಿರಾಂಕ __fp16 *)(x)))
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx4_ಕೂಡಿಸು          vaddq_f32
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_MUL          vmulq_f32
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F32Cx4
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     GGML_F32_Cx4_ತುಂಬು(p)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F32_Cx4_ಕೂಡಿಸು
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#ಆದರೆ_ಅಂತ್ಯ

#ಅಥವಾಆದರೆ defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 NEON

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 16
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  4

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4              float32x4_t
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ತುಂಬು         vld1q_f32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_STORE        vst1q_f32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ಕೂಡಿಸು          vaddq_f32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_MUL          vmulq_f32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_REDUCE(res, x)                       \
{                                                       \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;                     \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[ಸ್ಥಾನಾಂತರ+i]);      \
    }                                                   \
    ಸ್ಥಾನಾಂತರ >>= 1;                                       \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[ಸ್ಥಾನಾಂತರ+i]);      \
    }                                                   \
    ಸ್ಥಾನಾಂತರ >>= 1;                                       \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[ಸ್ಥಾನಾಂತರ+i]);      \
    }                                                   \
    (res) = (ggml_float) GGML_F32x4_REDUCE_ONE((x)[0]); \
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x4_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#ಆದರೆ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 32
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  8

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8              float16x8_t
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_ತುಂಬು(x)      vld1q_f16((ಸ್ಥಿರಾಂಕ __fp16 *)(x))
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_STORE        vst1q_f16
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_ಕೂಡಿಸು          vaddq_f16
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_MUL          vmulq_f16
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x8_REDUCE(res, x)                               \
    ಮಾಡು {                                                            \
        ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F16_ARR >> 1;                             \
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ಸ್ಥಾನಾಂತರ+i]);              \
        }                                                           \
        ಸ್ಥಾನಾಂತರ >>= 1;                                               \
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ಸ್ಥಾನಾಂತರ+i]);              \
        }                                                           \
        ಸ್ಥಾನಾಂತರ >>= 1;                                               \
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ಸ್ಥಾನಾಂತರ+i]);              \
        }                                                           \
        ಸ್ಥಿರಾಂಕ float32x4_t t0 = vcvt_f32_f16(vget_ಕೆಳಗಿನ_f16 ((x)[0])); \
        ಸ್ಥಿರಾಂಕ float32x4_t t1 = vcvt_f32_f16(vget_ಮೇಲಿನ_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } ಆಗ (0)

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F16x8
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     GGML_F16x8_ತುಂಬು(p)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F16x8_ಕೂಡಿಸು
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#ಇಲ್ಲದಿದ್ದರೆ
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 16
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  4

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4              float32x4_t
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx4_ತುಂಬು(x)      vcvt_f32_f16(vld1_f16((ಸ್ಥಿರಾಂಕ __fp16 *)(x)))
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx4_ಕೂಡಿಸು          vaddq_f32
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_MUL          vmulq_f32
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F32Cx4
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     GGML_F32_Cx4_ತುಂಬು(p)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F32_Cx4_ಕೂಡಿಸು
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#ಆದರೆ_ಅಂತ್ಯ

#ಅಥವಾಆದರೆ defined(__AVX512F__)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 AVX512

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 64
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  16

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x16         __m512
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x16_ZERO    _mm512_setzero_ps()
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x16_ತುಂಬು    _mm512_loadu_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x16_ಕೂಡಿಸು     mm512_ಕೂಡಿಸು_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x16_MUL     _mm512_mul_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x16_REDUCE(res, x)                                    \
ಮಾಡು {                                                                  \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;                                   \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                                \
        x[i] = mm512_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                      \
    }                                                                 \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                     \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                                \
        x[i] = mm512_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                      \
    }                                                                 \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                     \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                                \
        x[i] = mm512_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                      \
    }                                                                 \
    res = (ggml_float) mm512_reduce_ಕೂಡಿಸು_ps(x[0]);                    \
} ಆಗ (0)

// TODO: is this optimal ?

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x16
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x16_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x16_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x16_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x16_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x16_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x16_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x16_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512

#ಆದರೆ defined(__AVX512FP16__)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 128
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  32

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x32              __m512h
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x32_ZERO         _mm512_setzero_ph()
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x32_SET1(x)      _mm512_set1_ph(__extension__(_Float16)(x))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x32_ತುಂಬು(x)      _mm512_loadu_ph(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x32_STORE(x, y)  _mm512_storeu_ph(x, y)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x32_FMA(a, b, c) _mm512_fmadd_ph(b, c, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x32_ಕೂಡಿಸು          mm512_ಕೂಡಿಸು_ph
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x32_MUL          _mm512_mul_ph
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x32_REDUCE(res, x)                                     \
ಮಾಡು {                                                                   \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F16_ARR >> 1;                                    \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                                 \
        x[i] = mm512_ಕೂಡಿಸು_ph(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                       \
    }                                                                  \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                      \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                                 \
        x[i] = mm512_ಕೂಡಿಸು_ph(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                       \
    }                                                                  \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                      \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                                 \
        x[i] = mm512_ಕೂಡಿಸು_ph(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                       \
    }                                                                  \
    res = (ggml_float) mm512_reduce_ಕೂಡಿಸು_ph(x[0]);                     \
} ಆಗ (0)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F16x32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F16x32_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F16x32_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     GGML_F16x32_ತುಂಬು(p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) GGML_F16x32_STORE(p, r[i])
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F16x32_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F16x32_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F16x32_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F16x32_REDUCE

#ಇಲ್ಲದಿದ್ದರೆ // Fallback FP16 <-> FP32

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 64
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  16

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx16             __m512
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx16_ತುಂಬು(x)     _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(x)))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx16_ಕೂಡಿಸು         mm512_ಕೂಡಿಸು_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx16_MUL         _mm512_mul_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx16_REDUCE(res, x)                               \
ಮಾಡು {                                                              \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;                               \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = mm512_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                  \
    }                                                             \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                 \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = mm512_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                  \
    }                                                             \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                 \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = mm512_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                  \
    }                                                             \
    res = (ggml_float) mm512_reduce_ಕೂಡಿಸು_ps(x[0]);                \
} ಆಗ (0)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F32Cx16
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F32Cx16_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F32Cx16_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     GGML_F32_Cx16_ತುಂಬು(p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F32Cx16_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F32_Cx16_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F32Cx16_MUL

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F32Cx16_REDUCE

#ಆದರೆ_ಅಂತ್ಯ // __AVX512FP16__
#ಅಥವಾಆದರೆ defined(__AVX__)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 AVX

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  8

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8         __m256
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_ZERO    _mm256_setzero_ps()
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_ತುಂಬು    _mm256_loadu_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_STORE   _mm256_storeu_ps
#ಆದರೆ defined(__FMA__)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#ಇಲ್ಲದಿದ್ದರೆ
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_FMA(a, b, c) mm256_ಕೂಡಿಸು_ps(_mm256_mul_ps(b, c), a)
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_ಕೂಡಿಸು     mm256_ಕೂಡಿಸು_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_MUL     _mm256_mul_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_REDUCE(res, x)                                 \
ಮಾಡು {                                                              \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;                               \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = mm256_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                  \
    }                                                             \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                 \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = mm256_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                  \
    }                                                             \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                 \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = mm256_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                  \
    }                                                             \
    ಸ್ಥಿರಾಂಕ __m128 t0 = mm_ಕೂಡಿಸು_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    ಸ್ಥಿರಾಂಕ __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} ಆಗ (0)
// TODO: is this optimal ?

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x8
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x8_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x8_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x8_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x8_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x8_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x8_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 AVX

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8             __m256
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#ಆದರೆ defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx8_ತುಂಬು(x)     _mm256_cvtph_ps(_mm_loadu_si128((ಸ್ಥಿರಾಂಕ __m128i *)(x)))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಸಾಲೊಳಗೆ __m256 __avx_f32cx8_load(ಸ್ಥಿರಾಂಕ ggml_fp16_t * x) {
    ದಶಮಾಂಶ tmp[8];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i]);
    }

    ಹಿಂತಿರುಗಿ _mm256_loadu_ps(tmp);
}
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    ದಶಮಾಂಶ arr[8];

    _mm256_storeu_ps(arr, y);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 8; i++)
        x[i] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(arr[i]);
}
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx8_ತುಂಬು(x)     __avx_f32cx8_load(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_FMA         GGML_F32x8_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx8_ಕೂಡಿಸು         mm256_ಕೂಡಿಸು_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_MUL         _mm256_mul_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F32Cx8
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     GGML_F32_Cx8_ತುಂಬು(p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F32_Cx8_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#ಅಥವಾಆದರೆ defined(__POWER9_VECTOR__)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 POWER9

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  4

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4              ಸದಿಶ ದಶಮಾಂಶ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ZERO         {0.0f}
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_SET1         vec_splats
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ತುಂಬು(p)      vec_xl(0, p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ಕೂಡಿಸು          vec_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_MUL          vec_mul
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_REDUCE(res, x)              \
{                                              \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;            \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {         \
        x[i] = vec_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ+i]);     \
    }                                          \
    ಸ್ಥಾನಾಂತರ >>= 1;                              \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {         \
        x[i] = vec_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ+i]);     \
    }                                          \
    ಸ್ಥಾನಾಂತರ >>= 1;                              \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {         \
        x[i] = vec_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ+i]);     \
    }                                          \
    res = vec_ಹೊರತೆಗೆ(x[0], 0) +               \
          vec_ಹೊರತೆಗೆ(x[0], 1) +               \
          vec_ಹೊರತೆಗೆ(x[0], 2) +               \
          vec_ಹೊರತೆಗೆ(x[0], 3);                \
}
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3)        \
{                                                       \
    ಸದಿಶ ದಶಮಾಂಶ v = vec_ಕೂಡಿಸು(vec_ಕೂಡಿಸು(s0, s1),           \
                             vec_ಕೂಡಿಸು(s2, s3));          \
    v = vec_ಕೂಡಿಸು(v, vec_sld(v, v, 8));                   \
    v = vec_ಕೂಡಿಸು(v, vec_sld(v, v, 4));                   \
    res += (ggml_float) vec_ಹೊರತೆಗೆ(v, 0);              \
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x4_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 POWER9
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ       GGML_F32_ಹೆಜ್ಜೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR        GGML_F32_EPR
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC        GGML_F32x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1   GGML_F32x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA    GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು    GGML_F32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL    GGML_F32x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i) (i & 0x1) ?                   \
  vec_ಹೊರತೆಗೆ_fp32_from_shorth(vec_xl(0, p - GGML_F16_EPR)) : \
  vec_ಹೊರತೆಗೆ_fp32_from_shortl(vec_xl(0, p))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ ggml_endian_byte(ಪೂರ್ಣಾಂಕ i) {
       uint16_t tmp_val = 1;
       ಹಿಂತಿರುಗಿ ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *)&tmp_val)[i];
}
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ENDIAN_BYTE(i) ggml_endian_byte(i)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i)                             \
  ಆದರೆ (i & 0x1)                                                  \
    vec_xst(vec_pack_ಗಮ್ಯ_short_fp32(r[i - GGML_ENDIAN_BYTE(1)],  \
                                   r[i - GGML_ENDIAN_BYTE(0)]), \
            0, p - GGML_F16_EPR)

//BF16 POWER9
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_ಹೆಜ್ಜೆ 16
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_EPR  8

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16x8         ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16x8_ZERO    vec_splats((ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ)0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16x8_ತುಂಬು(p) vec_xl(0, (ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ *)(p))

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_VEC          GGML_BF16x8
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_VEC_ZERO     GGML_BF16x8_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_VEC_ತುಂಬು     GGML_BF16x8_ತುಂಬು
#ಆದರೆ defined(__LITTLE_ENDIAN__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_ಗಮ್ಯ_F32_LO(v) ((ಸದಿಶ ದಶಮಾಂಶ) vec_mergel(GGML_BF16_VEC_ZERO, (v)))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_ಗಮ್ಯ_F32_HI(v) ((ಸದಿಶ ದಶಮಾಂಶ) vec_mergeh(GGML_BF16_VEC_ZERO, (v)))
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_ಗಮ್ಯ_F32_LO(v) ((ಸದಿಶ ದಶಮಾಂಶ) vec_mergel((v), GGML_BF16_VEC_ZERO))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_ಗಮ್ಯ_F32_HI(v) ((ಸದಿಶ ದಶಮಾಂಶ) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_ಗಮ್ಯ_F32_LO(x), GGML_BF16_ಗಮ್ಯ_F32_LO(y))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_ಗಮ್ಯ_F32_HI(x), GGML_BF16_ಗಮ್ಯ_F32_HI(y))

#ಅಥವಾಆದರೆ defined(__wasm_simd128__)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 WASM

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 16
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  4

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4              v128_t
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ತುಂಬು         wasm_v128_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_STORE        wasm_v128_store
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_FMA(a, b, c) wasm_f32x4_ಕೂಡಿಸು(wasm_f32x4_mul(b, c), a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ಕೂಡಿಸು          wasm_f32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_MUL          wasm_f32x4_mul
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;                \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {             \
        x[i] = wasm_f32x4_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ+i]);  \
    }                                              \
    ಸ್ಥಾನಾಂತರ >>= 1;                                  \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {             \
        x[i] = wasm_f32x4_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ+i]);  \
    }                                              \
    ಸ್ಥಾನಾಂತರ >>= 1;                                  \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {             \
        x[i] = wasm_f32x4_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ+i]);  \
    }                                              \
    res = wasm_f32x4_ಹೊರತೆಗೆ_lane(x[0], 0) +       \
          wasm_f32x4_ಹೊರತೆಗೆ_lane(x[0], 1) +       \
          wasm_f32x4_ಹೊರತೆಗೆ_lane(x[0], 2) +       \
          wasm_f32x4_ಹೊರತೆಗೆ_lane(x[0], 3);        \
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x4_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 WASM

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 16
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  4

ಸಾಲೊಳಗೆ ಸ್ಥಿರ v128_t __wasm_f16x4_load(ಸ್ಥಿರಾಂಕ ggml_fp16_t * p) {
    ದಶಮಾಂಶ tmp[4];

    tmp[0] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(p[0]);
    tmp[1] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(p[1]);
    tmp[2] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(p[2]);
    tmp[3] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(p[3]);

    ಹಿಂತಿರುಗಿ wasm_v128_ತುಂಬು(tmp);
}

ಸಾಲೊಳಗೆ ಸ್ಥಿರ ಶೂನ್ಯ __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
    ದಶಮಾಂಶ tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(tmp[0]);
    p[1] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(tmp[1]);
    p[2] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(tmp[2]);
    p[3] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(tmp[3]);
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x4             v128_t
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x4_ತುಂಬು(x)     __wasm_f16x4_load(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x4_FMA         GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x4_ಕೂಡಿಸು         wasm_f32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x4_MUL         wasm_f32x4_mul
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16x4_REDUCE(res, x)                           \
{                                                           \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F16_ARR >> 1;                         \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                      \
        x[i] = wasm_f32x4_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ+i]);           \
    }                                                       \
    ಸ್ಥಾನಾಂತರ >>= 1;                                           \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                      \
        x[i] = wasm_f32x4_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ+i]);           \
    }                                                       \
    ಸ್ಥಾನಾಂತರ >>= 1;                                           \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                      \
        x[i] = wasm_f32x4_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ+i]);           \
    }                                                       \
    res = (ggml_float) (wasm_f32x4_ಹೊರತೆಗೆ_lane(x[0], 0) +  \
          wasm_f32x4_ಹೊರತೆಗೆ_lane(x[0], 1) +                \
          wasm_f32x4_ಹೊರತೆಗೆ_lane(x[0], 2) +                \
          wasm_f32x4_ಹೊರತೆಗೆ_lane(x[0], 3));                \
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F16x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F16x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F16x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     GGML_F16x4_ತುಂಬು(p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) GGML_F16x4_STORE(p, r[i])
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F16x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F16x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F16x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F16x4_REDUCE

#ಅಥವಾಆದರೆ defined(__SSE3__)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 SSE

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  4

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4         __m128
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ZERO    _mm_setzero_ps()
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_SET1(x) _mm_set1_ps(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ತುಂಬು    _mm_loadu_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_STORE   _mm_storeu_ps
#ಆದರೆ defined(__FMA__)
    // TODO: Does this work?
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#ಇಲ್ಲದಿದ್ದರೆ
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_FMA(a, b, c) mm_ಕೂಡಿಸು_ps(_mm_mul_ps(b, c), a)
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ಕೂಡಿಸು     mm_ಕೂಡಿಸು_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_MUL     _mm_mul_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;                               \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = mm_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                     \
    }                                                             \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                 \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = mm_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                     \
    }                                                             \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                 \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = mm_ಕೂಡಿಸು_ps(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                     \
    }                                                             \
    ಸ್ಥಿರಾಂಕ __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x4_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 SSE

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  4

ಸ್ಥಿರ ಸಾಲೊಳಗೆ __m128 __sse_f16x4_load(ಸ್ಥಿರಾಂಕ ggml_fp16_t * x) {
    ದಶಮಾಂಶ tmp[4];

    tmp[0] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[0]);
    tmp[1] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[1]);
    tmp[2] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[2]);
    tmp[3] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[3]);

    ಹಿಂತಿರುಗಿ _mm_loadu_ps(tmp);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __sse_f16x4_store(ggml_fp16_t * x, __m128 y) {
    ದಶಮಾಂಶ arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(arr[0]);
    x[1] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(arr[1]);
    x[2] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(arr[2]);
    x[3] = GGML_CPU_FP32_ಗಮ್ಯ_FP16(arr[3]);
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4             __m128
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_ZERO        _mm_setzero_ps()
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx4_ತುಂಬು(x)     __sse_f16x4_load(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_FMA         GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx4_ಕೂಡಿಸು         mm_ಕೂಡಿಸು_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_MUL         _mm_mul_ps
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                 GGML_F32Cx4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)      GGML_F32_Cx4_ತುಂಬು(p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು             GGML_F32_Cx4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#ಅಥವಾಆದರೆ defined(__loongarch_asx)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 LASX
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  8

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8         __m256
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_ತುಂಬು(x) (__m256)__lasx_xvld((x), 0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_ಕೂಡಿಸು     __lasx_xvfadd_s
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_MUL     __lasx_xvfmul_s
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x8_REDUCE(res, x)                                 \
ಮಾಡು {                                                              \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;                               \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                  \
    }                                                             \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                 \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                  \
    }                                                             \
    ಸ್ಥಾನಾಂತರ >>= 1;                                                 \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                  \
    }                                                             \
    ದಶಮಾಂಶ *tmp_p = (ದಶಮಾಂಶ *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} ಆಗ (0)
// TODO: is this optimal ?

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x8
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x8_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x8_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x8_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x8_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x8_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x8_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 LASX

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  8

// F16 arithmetic is not supported by LASX, so we use F32 instead

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8          __m256
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))

ಸ್ಥಿರ ಸಾಲೊಳಗೆ __m256 __lasx_f32cx8_load(ಸ್ಥಿರಾಂಕ ggml_fp16_t * x) {
    __m256i a;
    ಸ್ಮರಣೆ_ನಕಲು(&a, x, ಗಾತ್ರ(ggml_fp16_t) * 8);
    a = __lasx_xvpermi_d(a, 0 | (1 << 4));
    ಹಿಂತಿರುಗಿ __lasx_xvfcvtl_s_h(a);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __lasx_f32cx8_store(ggml_fp16_t * x, __m256 y) {
    __m256i a = __lasx_xvfcvt_h_s(y, y);
    a = __lasx_xvpermi_d(a, 0 | (2 << 2));
    ಸ್ಮರಣೆ_ನಕಲು(x, &a, ಗಾತ್ರ(ggml_fp16_t) * 8);
}
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx8_ತುಂಬು(x)     __lasx_f32cx8_load(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_FMA         GGML_F32x8_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx8_ಕೂಡಿಸು         __lasx_xvfadd_s
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_MUL         __lasx_xvfmul_s
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F32Cx8
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     GGML_F32_Cx8_ತುಂಬು(p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F32_Cx8_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#ಅಥವಾಆದರೆ defined(__loongarch_sx)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 LSX

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  4

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4         __m128
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ZERO    (__m128)__lsx_vldi(0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_SET1(x) (__m128)__lsx_vreplfr2vr_s((x))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ತುಂಬು(x) (__m128)__lsx_vld((x), 0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_STORE(x, y)   __lsx_vst(y, x, 0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ಕೂಡಿಸು     __lsx_vfadd_s
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_MUL     __lsx_vfmul_s

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_REDUCE(res, x)                               \
{                                                               \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;                             \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                \
    }                                                           \
    ಸ್ಥಾನಾಂತರ >>= 1;                                               \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                \
    }                                                           \
    ಸ್ಥಾನಾಂತರ >>= 1;                                               \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[ಸ್ಥಾನಾಂತರ+i]);                \
    }                                                           \
    __m128i t0 = __lsx_vpickev_w((__m128i)x[0], (__m128i)x[0]); \
    __m128i t1 = __lsx_vpickod_w((__m128i)x[0], (__m128i)x[0]); \
    __m128 t2 = __lsx_vfadd_s((__m128)t0, (__m128)t1);          \
    __m128i t3 = __lsx_vpickev_w((__m128i)t2, (__m128i)t2);     \
    __m128i t4 = __lsx_vpickod_w((__m128i)t2, (__m128i)t2);     \
    __m128 t5 = __lsx_vfadd_s((__m128)t3, (__m128)t4);          \
    res = (ggml_float) ((v4f32)t5)[0];                          \
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x4_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 LSX

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  4

ಸ್ಥಿರ ಸಾಲೊಳಗೆ __m128 __lsx_f16x4_load(ಸ್ಥಿರಾಂಕ ggml_fp16_t * x) {
    ಹಿಂತಿರುಗಿ __lsx_vfcvtl_s_h(__lsx_vld((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)x, 0));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __lsx_f16x4_store(ggml_fp16_t * x, __m128 y) {
    __m128i a = __lsx_vfcvt_h_s(y, y);
    ಸ್ಮರಣೆ_ನಕಲು(x, &a, ಗಾತ್ರ(ggml_fp16_t) * 4);
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4             __m128
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_ZERO        (__m128)__lsx_vldi(0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_SET1(x)     (__m128)__lsx_vreplfr2vr_s((x))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx4_ತುಂಬು(x)     (__m128)__lsx_f16x4_load(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_FMA         GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_Cx4_ಕೂಡಿಸು         __lsx_vfadd_s
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_MUL         __lsx_vfmul_s
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                 GGML_F32Cx4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)      GGML_F32_Cx4_ತುಂಬು(p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು             GGML_F32_Cx4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#ಅಥವಾಆದರೆ defined(__VXE__) || defined(__VXE2__)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32 s390x

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  4

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4              float32x4_t
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ZERO         vec_splats(0.0f)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_SET1         vec_splats
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ತುಂಬು(p)      vec_xl(0, p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ಕೂಡಿಸು          vec_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_MUL          vec_mul
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_REDUCE(res, x)                   \
{                                                   \
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = GGML_F32_ARR >> 1;                 \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {              \
        x[i] = vec_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ + i]);        \
    }                                               \
    ಸ್ಥಾನಾಂತರ >>= 1;                                   \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {              \
        x[i] = vec_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ + i]);        \
    }                                               \
    ಸ್ಥಾನಾಂತರ >>= 1;                                   \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಸ್ಥಾನಾಂತರ; ++i) {              \
        x[i] = vec_ಕೂಡಿಸು(x[i], x[ಸ್ಥಾನಾಂತರ + i]);        \
    }                                               \
    float32x4_t tmp = x[0] + vec_reve(x[0]);        \
    res = tmp[0] + tmp[1];                          \
}
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3) \
{                                                \
    float32x4_t v = vec_ಕೂಡಿಸು(vec_ಕೂಡಿಸು(s0, s1),     \
                            vec_ಕೂಡಿಸು(s2, s3));    \
    v = vec_ಕೂಡಿಸು(v, vec_sld(v, v, 8));            \
    v = vec_ಕೂಡಿಸು(v, vec_sld(v, v, 4));            \
    res += (ggml_float)vec_ಹೊರತೆಗೆ(v, 0);        \
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x4_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 s390x
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ಹೆಜ್ಜೆ GGML_F32_ಹೆಜ್ಜೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_EPR  GGML_F32_EPR

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32x4_t __lzs_f32cx4_to_f16(float32x4_t v_f) {
    float32x4_t v_ತಳ = vec_mul(vec_mul(vec_abs(v_f), vec_splats(0x1.0p+112f)), vec_splats(0x1.0p-110f));

    ಸ್ಥಿರಾಂಕ uint32x4_t v_w      = (uint32x4_t)v_f;
    ಸ್ಥಿರಾಂಕ uint32x4_t v_shl1_w = vec_ಕೂಡಿಸು(v_w, v_w);
    ಸ್ಥಿರಾಂಕ uint32x4_t v_sign   = vec_and(v_w, vec_splats(UINT32_C(0x80000000)));
    ಸ್ಥಿರಾಂಕ uint32x4_t v_bias   = vec_max(vec_and(v_shl1_w, vec_splats(UINT32_C(0xFF000000))), vec_splats(UINT32_C(0x71000000)));

    v_ತಳ = vec_ಕೂಡಿಸು((float32x4_t)vec_ಕೂಡಿಸು(vec_sr(v_bias, 1), vec_splats(UINT32_C(0x07800000))), v_ತಳ);

    ಸ್ಥಿರಾಂಕ uint32x4_t v_ಬಿಟ್ಟುಗಳು    = (uint32x4_t)v_ತಳ;
    ಸ್ಥಿರಾಂಕ uint32x4_t v_nonsign = vec_ಕೂಡಿಸು(vec_and(vec_sr(v_ಬಿಟ್ಟುಗಳು, 13), vec_splats(UINT32_C(0x00007C00))),
                                         vec_and(v_ಬಿಟ್ಟುಗಳು, vec_splats(UINT32_C(0x00000FFF))));
    ಸ್ಥಿರಾಂಕ uint32x4_t v_is_nan  = (uint32x4_t)vec_cmpgt(v_shl1_w, vec_splats(UINT32_C(0xFF000000)));

    ಹಿಂತಿರುಗಿ vec_or(vec_sr(v_sign, 16), vec_sel(v_nonsign, vec_splats(UINT32_C(0x7E00)), v_is_nan));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ float32x4_t __lzs_f16cx4_load(ಸ್ಥಿರಾಂಕ ggml_fp16_t * x) {
    ದಶಮಾಂಶ tmp[4];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i]);
    }

    // note: keep type-cast here to prevent compiler bugs
    // see: https://github.com/ggml-org/llama.cpp/issues/12846
    ಹಿಂತಿರುಗಿ vec_xl(0, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(tmp));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __lzs_f16cx4_store(ggml_fp16_t * x, float32x4_t v_y) {
    ಸ್ಥಿರಾಂಕ uint32x4_t v_h = __lzs_f32cx4_to_f16(v_y);
    ಸ್ಥಿರಾಂಕ uint64_t   tmp = ((uint64x2_t)vec_pack(v_h, v_h))[0];
    ಸ್ಮರಣೆ_ನಕಲು(x, &tmp, ಗಾತ್ರ(tmp));
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC                GGML_F32x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ZERO           GGML_F32x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_SET1           GGML_F32x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ತುಂಬು(p, i)     __lzs_f16cx4_load(p)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_STORE(p, r, i) __lzs_f16cx4_store(p, r[i])
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_FMA            GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_ಕೂಡಿಸು            GGML_F32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_MUL            GGML_F32x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_VEC_REDUCE         GGML_F32x4_REDUCE

// BF16 s390x
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_ಹೆಜ್ಜೆ 16
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_EPR  8

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16x8         __vector ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16x8_ZERO    vec_splats((ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ)0)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16x8_ತುಂಬು(p) vec_xl(0, (ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ *)(p))

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_VEC      GGML_BF16x8
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_VEC_ZERO GGML_BF16x8_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_VEC_ತುಂಬು GGML_BF16x8_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_ಗಮ್ಯ_F32_LO(v) ((float32x4_t) vec_mergel((v), GGML_BF16_VEC_ZERO))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_ಗಮ್ಯ_F32_HI(v) ((float32x4_t) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_ಗಮ್ಯ_F32_LO(x), GGML_BF16_ಗಮ್ಯ_F32_LO(y))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_ಗಮ್ಯ_F32_HI(x), GGML_BF16_ಗಮ್ಯ_F32_HI(y))

#ಅಥವಾಆದರೆ defined(__riscv_v_intrinsic)

// compatible with vlen >= 128

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SIMD

// F32

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ಹೆಜ್ಜೆ 16
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_EPR  4

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4              vfloat32m1_t
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ZERO         __riscv_vfmv_v_f_f32m1(0.0f, GGML_F32_EPR)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_SET1(x)      __riscv_vfmv_v_f_f32m1(x, GGML_F32_EPR)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ತುಂಬು(x)      __riscv_vle32_v_f32m1(x, GGML_F32_EPR)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_STORE(b, v)  __riscv_vse32_v_f32m1(b, v, GGML_F32_EPR)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_FMA(a, b, c) __riscv_vfmacc_vv_f32m1(a, b, c, GGML_F32_EPR)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_ಕೂಡಿಸು(a, b)    __riscv_vfadd_vv_f32m1(a, b, GGML_F32_EPR)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32x4_MUL(a, b)    __riscv_vfmul_vv_f32m1(a, b, GGML_F32_EPR)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC        GGML_F32x4
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ತುಂಬು   GGML_F32x4_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_ಕೂಡಿಸು    GGML_F32x4_ಕೂಡಿಸು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

#ಆದರೆ_ಅಂತ್ಯ

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SIMD
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F32_ARR (GGML_F32_ಹೆಜ್ಜೆ/GGML_F32_EPR)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_ARR (GGML_F16_ಹೆಜ್ಜೆ/GGML_F16_EPR)
#ಆದರೆ_ಅಂತ್ಯ

// GGML_F16_DOT_*
// like GGML_F16_* but for dot products which need F32 accumulation on AVX512-FP16

#ಆದರೆ defined(__AVX512FP16__)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_ಹೆಜ್ಜೆ           GGML_F32_ಹೆಜ್ಜೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_EPR            GGML_F32_EPR
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_ARR            GGML_F32_ARR
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC            GGML_F32x16
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC_ZERO       GGML_F32x16_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC_ತುಂಬು(p, i) _mm512_cvtph_ps(_mm256_loadu_si256((ಸ್ಥಿರಾಂಕ __m256i *)(p)))
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC_FMA        GGML_F32x16_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC_REDUCE     GGML_F32x16_REDUCE

#ಇಲ್ಲದಿದ್ದರೆ

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_ಹೆಜ್ಜೆ           GGML_F16_ಹೆಜ್ಜೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_EPR            GGML_F16_EPR
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_ARR            GGML_F16_ARR
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC            GGML_F16_VEC
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC_ZERO       GGML_F16_VEC_ZERO
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC_ತುಂಬು       GGML_F16_VEC_ತುಂಬು
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC_FMA        GGML_F16_VEC_FMA
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_F16_DOT_VEC_REDUCE     GGML_F16_VEC_REDUCE

#ಆದರೆ_ಅಂತ್ಯ // defined(__AVX512FP16__)

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
