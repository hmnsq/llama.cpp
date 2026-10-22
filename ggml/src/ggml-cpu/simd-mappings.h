#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __ARM_FEATURE_SVE
#ਅਨ੍ਤਰ੍ਭਾਵ <arm_sve.h>
#ਜੇ_ਅੰਤ // __ARM_FEATURE_SVE

#ਜੇ defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ਅਨ੍ਤਰ੍ਭਾਵ <arm_neon.h>
#ਜੇ_ਅੰਤ

#ਜੇ defined(__riscv_v_intrinsic)
#ਅਨ੍ਤਰ੍ਭਾਵ <riscv_vector.h>
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

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
#ਜੇ defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x) neon_compute_fp16_ਮੰਜ਼ਿਲ_fp32(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x) neon_compute_fp32_ਮੰਜ਼ਿਲ_fp16(x)

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x) GGML_CPU_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(x) GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x)

    ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ neon_compute_fp16_ਮੰਜ਼ਿਲ_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&tmp, &h, ਆਕਾਰ(ggml_fp16_t));
        ਵਾਪਸ (ਦਸ਼ਮਲਵ)tmp;
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_fp16_t neon_compute_fp32_ਮੰਜ਼ਿਲ_fp16(ਦਸ਼ਮਲਵ f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&res, &tmp, ਆਕਾਰ(ggml_fp16_t));
        ਵਾਪਸ res;
    }
#ਜਾਂਜੇ defined(__F16C__)
    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _MSC_VER
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(mm_ਸੈੱਟ_ss(x), 0), 0)
    #ਨਹੀਂਤਾਂ
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x) _cvtsh_ss(x)
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x) _cvtss_sh(x, 0)
    #ਜੇ_ਅੰਤ
#ਜਾਂਜੇ defined(__POWER9_VECTOR__)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x) power_compute_fp16_ਮੰਜ਼ਿਲ_fp32(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x) power_compute_fp32_ਮੰਜ਼ਿਲ_fp16(x)
    /* the inline asm below is about 12% faster than the lookup method */
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x) GGML_CPU_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(x) GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x)

    ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ power_compute_fp16_ਮੰਜ਼ਿਲ_fp32(ggml_fp16_t h) {
        ਦਸ਼ਮਲਵ f;
        ਦੁੱਗਣਾ d;
        __asm__(
            "mtfprd %0,%2\n"
            "xscvhpdp %0,%0\n"
            "frsp %1,%0\n" :
            /*ਆਰਜ਼ੀ*/ "=d"(d),
            /*ਬਾਹਰਲਾ*/  "=f"(f):
            /*ਵਿੱਚ*/   "r"(h));
        ਵਾਪਸ f;
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_fp16_t power_compute_fp32_ਮੰਜ਼ਿਲ_fp16(ਦਸ਼ਮਲਵ f) {
        ਦੁੱਗਣਾ d;
        ggml_fp16_t r;
        __asm__( /* xscvdphp can work on double or single precision */
            "xscvdphp %0,%2\n"
            "mffprd %1,%0\n" :
            /*ਆਰਜ਼ੀ*/ "=d"(d),
            /*ਬਾਹਰਲਾ*/  "=r"(r):
            /*ਵਿੱਚ*/   "f"(f));
        ਵਾਪਸ r;
    }
#ਜਾਂਜੇ defined(__riscv) && defined(__riscv_zfhmin)
    ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ riscv_compute_fp16_ਮੰਜ਼ਿਲ_fp32(ggml_fp16_t h) {
        _Float16 hf;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&hf, &h, ਆਕਾਰ(ggml_fp16_t));
        ਵਾਪਸ hf;
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_fp16_t riscv_compute_fp32_ਮੰਜ਼ਿਲ_fp16(ਦਸ਼ਮਲਵ f) {
        ggml_fp16_t res;
        _Float16 hf = (_Float16)f;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&res, &hf, ਆਕਾਰ(ggml_fp16_t));
        ਵਾਪਸ res;
    }

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x) riscv_compute_fp16_ਮੰਜ਼ਿਲ_fp32(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x) riscv_compute_fp32_ਮੰਜ਼ਿਲ_fp16(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x) GGML_CPU_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(x) GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x)
#ਜੇ_ਅੰਤ

// precomputed f32 table for f16 (256 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
ਬਾਹਰੀ ਦਸ਼ਮਲਵ ggml_ਸਾਰਣੀ_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
ਬਾਹਰੀ ਦਸ਼ਮਲਵ ggml_ਸਾਰਣੀ_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
ਬਾਹਰੀ ਦਸ਼ਮਲਵ ggml_ਸਾਰਣੀ_f32_ue4m3[1 << 8];

// Use lookup table for E8M0 on x86 (faster than bit manipulation)
#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(x) ggml_ਸਾਰਣੀ_f32_e8m0_half[(uint8_t)(x)]
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(x) GGML_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(x)
#ਜੇ_ਅੰਤ

// Use lookup table for UE4M3 on x86 and ARM (faster than bit manipulation)
#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__ARM_NEON)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_UE4_M3_ਮੰਜ਼ਿਲ_FP32(x) ggml_ਸਾਰਣੀ_f32_ue4m3[(uint8_t)(x)]
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_UE4_M3_ਮੰਜ਼ਿਲ_FP32(x) ggml_ue4m3_ਮੰਜ਼ਿਲ_fp32(x)
#ਜੇ_ਅੰਤ

// Pre AVX512 CPUs (AVX2 with F16C, such as i5-8th gen or i7-4790)
// do not produce codegen as checks below don't test for __F16C__
// but this check does.
#ਜੇ defined(__F16C__)
// force FP16 -> FP32 conversion to use LUT instead
// see: https://github.com/ggml-org/llama.cpp/issues/30342
//#define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(x) GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x)
#ਜੇ_ਅੰਤ

// a branch that defines a GGML_CPU_COMPUTE_* macro must also select it, or the fallback takes over silently
#ਜੇ defined(GGML_CPU_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32) && !defined(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32) && !defined(__F16C__) // see: https://github.com/ggml-org/llama.cpp/issues/30342
#ਗਲਤੀ "a branch defined GGML_CPU_COMPUTE_FP16_TO_FP32 without selecting it as GGML_CPU_FP16_TO_FP32"
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_CPU_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16) && !defined(GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16)
#ਗਲਤੀ "a branch defined GGML_CPU_COMPUTE_FP32_TO_FP16 without selecting it as GGML_CPU_FP32_TO_FP16"
#ਜੇ_ਅੰਤ

// fall back to the lookup table / the portable software conversion for whichever direction no branch selected
#ਜੇ !defined(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32)
ਸਤਰਅੰਦਰ ਸਥਿਰ ਦਸ਼ਮਲਵ ggml_lookup_fp16_ਮੰਜ਼ਿਲ_fp32(ggml_fp16_t f) {
    uint16_t s;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&s, &f, ਆਕਾਰ(uint16_t));
    ਵਾਪਸ ggml_ਸਾਰਣੀ_f32_f16[s];
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x) ggml_lookup_fp16_ਮੰਜ਼ਿਲ_fp32(x)
#ਜੇ_ਅੰਤ

#ਜੇ !defined(GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(x) GGML_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x)
#ਜੇ_ਅੰਤ


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

#ਜੇ defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_FMA)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 SVE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR 8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFAULT_PG svptrue_b32()

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt                        svfloat32_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_ZERO                   svdup_n_f32(0.0f)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_SET1(x)                svdup_n_f32(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_ਲੱਦੋ_IMPL(pg, a)       svld1_f32(pg, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_ਲੱਦੋ(a)                GGML_F32xt_ਲੱਦੋ_IMPL(DEFAULT_PG, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_STORE_IMPL(pg, a, b)   svst1_f32(pg, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_STORE(a, b)            GGML_F32xt_STORE_IMPL(DEFAULT_PG, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_FMA_IMPL(pg, a, b, c)  svmad_f32_m(pg, b, c, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_FMA(a, b, c)           GGML_F32xt_FMA_IMPL(DEFAULT_PG, a, b, c)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_ਜੋੜੋ_IMPL(pg, a, b)     svadd_f32_m(pg, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_ਜੋੜੋ(a, b)              GGML_F32xt_ਜੋੜੋ_IMPL(DEFAULT_PG, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_MUL_IMPL(pg, a, b)     svmul_f32_m(pg, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_MUL(a, b)              GGML_F32xt_MUL_IMPL(DEFAULT_PG, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_REDUCE_ONE_IMPL(pg, a) svaddv(pg, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_REDUCE_ONE(a)          GGML_F32xt_REDUCE_ONE_IMPL(DEFAULT_PG, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_REDUCE_IMPL(pg, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
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
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32xt_REDUCE(res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
        GGML_F32xt_REDUCE_IMPL(DEFAULT_PG, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32xt
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32xt_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32xt_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32xt_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32xt_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32xt_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32xt_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32xt_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32xt_REDUCE

// F16 SVE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFAULT_PG32    svptrue_b32()
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFAULT_PG16    svptrue_b16()

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cxt                         svfloat16_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cxt_ZERO                    svdup_n_f16(0.0f)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cxt_SET1(x)                 svdup_n_f16(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cxt_ਲੱਦੋ(p)                 svld1_f16(DEFAULT_PG16, (ਅਚਲ __fp16 *)(p))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cxt_STORE(dst_ptr, src_vec) svst1_f16(DEFAULT_PG16, (__fp16 *)(dst_ptr), (src_vec))

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cxt_FMA_IMPL(pg, a, b, c)   svmad_f16_x(pg, b, c, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cxt_FMA(a, b, c)            GGML_F32Cxt_FMA_IMPL(DEFAULT_PG16, a, b, c)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cxt_ਜੋੜੋ_IMPL(pg, a, b)      svadd_f16_x(pg, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cxt_ਜੋੜੋ(a, b)               GGML_F32_Cxt_ਜੋੜੋ_IMPL(DEFAULT_PG16, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cxt_MUL_IMPL(pg, a, b)      svmul_f16_x(pg, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cxt_MUL(a, b)               GGML_F32Cxt_MUL_IMPL(DEFAULT_PG16, a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cxt_REDUCE                  GGML_F16xt_REDUCE_MIXED

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x_VEC                GGML_F32Cxt
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x_VEC_ZERO           GGML_F32Cxt_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x_VEC_SET1           GGML_F32Cxt_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x_VEC_ਲੱਦੋ(p, i)     GGML_F32_Cxt_ਲੱਦੋ(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x_VEC_STORE(p, r, i) GGML_F32Cxt_STORE((__fp16 *)(p), r)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x_VEC_FMA            GGML_F32Cxt_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x_VEC_ਜੋੜੋ            GGML_F32_Cxt_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x_VEC_MUL            GGML_F32Cxt_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x_VEC_REDUCE         GGML_F32Cxt_REDUCE

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16xt_REDUCE_ONE_IMPL(pg, a) svaddv_f16(pg, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16xt_REDUCE_ONE(a)          GGML_F16xt_REDUCE_ONE_IMPL(DEFAULT_PG16, a)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16xt_REDUCE_MIXED_IMPL(pg16, res, sum1, sum2, sum3, sum4)  \
{                                                      \
    sum1 = svadd_f16_x(pg16, sum1, sum2);              \
    sum3 = svadd_f16_x(pg16, sum3, sum4);              \
    sum1 = svadd_f16_x(pg16, sum1, sum3);              \
    __fp16 ਜੋੜ_f16 = svaddv_f16(pg16, sum1);           \
    (res) = (ggml_float) ਜੋੜ_f16;                      \
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16xt_REDUCE_MIXED(res, sum1, sum2, sum3, sum4)  \
        GGML_F16xt_REDUCE_MIXED_IMPL(DEFAULT_PG16, res, sum1, sum2, sum3, sum4)

// F16 NEON

#ਜੇ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 32
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  8

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8              float16x8_t
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_ਲੱਦੋ(x)      vld1q_f16((ਅਚਲ __fp16 *)(x))
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_STORE        vst1q_f16
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_ਜੋੜੋ          vaddq_f16
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_MUL          vmulq_f16
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_REDUCE(res, x)                               \
    ਕਰੋ {                                                            \
        ਪੂਰਨ ਵਿੱਥ = GGML_F16_ARR >> 1;                             \
        ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ਵਿੱਥ+i]);              \
        }                                                           \
        ਵਿੱਥ >>= 1;                                               \
        ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ਵਿੱਥ+i]);              \
        }                                                           \
        ਵਿੱਥ >>= 1;                                               \
        ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ਵਿੱਥ+i]);              \
        }                                                           \
        ਅਚਲ float32x4_t t0 = vcvt_f32_f16(vget_ਨੀਵਾਂ_f16 ((x)[0])); \
        ਅਚਲ float32x4_t t1 = vcvt_f32_f16(vget_ਉੱਚਾ_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } ਜਦੋਂ (0)

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F16x8
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     GGML_F16x8_ਲੱਦੋ(p)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F16x8_ਜੋੜੋ
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#ਨਹੀਂਤਾਂ
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 16
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  4

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4              float32x4_t
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx4_ਲੱਦੋ(x)      vcvt_f32_f16(vld1_f16((ਅਚਲ __fp16 *)(x)))
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx4_ਜੋੜੋ          vaddq_f32
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_MUL          vmulq_f32
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F32Cx4
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     GGML_F32_Cx4_ਲੱਦੋ(p)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F32_Cx4_ਜੋੜੋ
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#ਜੇ_ਅੰਤ

#ਜਾਂਜੇ defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 NEON

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4              float32x4_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਲੱਦੋ         vld1q_f32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_STORE        vst1q_f32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਜੋੜੋ          vaddq_f32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_MUL          vmulq_f32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_REDUCE(res, x)                       \
{                                                       \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;                     \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[ਵਿੱਥ+i]);      \
    }                                                   \
    ਵਿੱਥ >>= 1;                                       \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[ਵਿੱਥ+i]);      \
    }                                                   \
    ਵਿੱਥ >>= 1;                                       \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[ਵਿੱਥ+i]);      \
    }                                                   \
    (res) = (ggml_float) GGML_F32x4_REDUCE_ONE((x)[0]); \
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x4_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#ਜੇ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 32
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  8

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8              float16x8_t
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_ਲੱਦੋ(x)      vld1q_f16((ਅਚਲ __fp16 *)(x))
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_STORE        vst1q_f16
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_ਜੋੜੋ          vaddq_f16
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_MUL          vmulq_f16
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x8_REDUCE(res, x)                               \
    ਕਰੋ {                                                            \
        ਪੂਰਨ ਵਿੱਥ = GGML_F16_ARR >> 1;                             \
        ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ਵਿੱਥ+i]);              \
        }                                                           \
        ਵਿੱਥ >>= 1;                                               \
        ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ਵਿੱਥ+i]);              \
        }                                                           \
        ਵਿੱਥ >>= 1;                                               \
        ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[ਵਿੱਥ+i]);              \
        }                                                           \
        ਅਚਲ float32x4_t t0 = vcvt_f32_f16(vget_ਨੀਵਾਂ_f16 ((x)[0])); \
        ਅਚਲ float32x4_t t1 = vcvt_f32_f16(vget_ਉੱਚਾ_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } ਜਦੋਂ (0)

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F16x8
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     GGML_F16x8_ਲੱਦੋ(p)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F16x8_ਜੋੜੋ
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#ਨਹੀਂਤਾਂ
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 16
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  4

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4              float32x4_t
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx4_ਲੱਦੋ(x)      vcvt_f32_f16(vld1_f16((ਅਚਲ __fp16 *)(x)))
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx4_ਜੋੜੋ          vaddq_f32
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_MUL          vmulq_f32
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F32Cx4
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     GGML_F32_Cx4_ਲੱਦੋ(p)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F32_Cx4_ਜੋੜੋ
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#ਜੇ_ਅੰਤ

#ਜਾਂਜੇ defined(__AVX512F__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 AVX512

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  16

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x16         __m512
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x16_ZERO    _mm512_setzero_ps()
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x16_ਲੱਦੋ    _mm512_loadu_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x16_ਜੋੜੋ     mm512_ਜੋੜੋ_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x16_MUL     _mm512_mul_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x16_REDUCE(res, x)                                    \
ਕਰੋ {                                                                  \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;                                   \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                                \
        x[i] = mm512_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                      \
    }                                                                 \
    ਵਿੱਥ >>= 1;                                                     \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                                \
        x[i] = mm512_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                      \
    }                                                                 \
    ਵਿੱਥ >>= 1;                                                     \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                                \
        x[i] = mm512_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                      \
    }                                                                 \
    res = (ggml_float) mm512_reduce_ਜੋੜੋ_ps(x[0]);                    \
} ਜਦੋਂ (0)

// TODO: is this optimal ?

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x16_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x16_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x16_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x16_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x16_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x16_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x16_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512

#ਜੇ defined(__AVX512FP16__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x32              __m512h
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x32_ZERO         _mm512_setzero_ph()
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x32_SET1(x)      _mm512_set1_ph(__extension__(_Float16)(x))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x32_ਲੱਦੋ(x)      _mm512_loadu_ph(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x32_STORE(x, y)  _mm512_storeu_ph(x, y)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x32_FMA(a, b, c) _mm512_fmadd_ph(b, c, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x32_ਜੋੜੋ          mm512_ਜੋੜੋ_ph
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x32_MUL          _mm512_mul_ph
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x32_REDUCE(res, x)                                     \
ਕਰੋ {                                                                   \
    ਪੂਰਨ ਵਿੱਥ = GGML_F16_ARR >> 1;                                    \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                                 \
        x[i] = mm512_ਜੋੜੋ_ph(x[i], x[ਵਿੱਥ+i]);                       \
    }                                                                  \
    ਵਿੱਥ >>= 1;                                                      \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                                 \
        x[i] = mm512_ਜੋੜੋ_ph(x[i], x[ਵਿੱਥ+i]);                       \
    }                                                                  \
    ਵਿੱਥ >>= 1;                                                      \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                                 \
        x[i] = mm512_ਜੋੜੋ_ph(x[i], x[ਵਿੱਥ+i]);                       \
    }                                                                  \
    res = (ggml_float) mm512_reduce_ਜੋੜੋ_ph(x[0]);                     \
} ਜਦੋਂ (0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F16x32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F16x32_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F16x32_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     GGML_F16x32_ਲੱਦੋ(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) GGML_F16x32_STORE(p, r[i])
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F16x32_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F16x32_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F16x32_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F16x32_REDUCE

#ਨਹੀਂਤਾਂ // Fallback FP16 <-> FP32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  16

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx16             __m512
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx16_ਲੱਦੋ(x)     _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(x)))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx16_ਜੋੜੋ         mm512_ਜੋੜੋ_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx16_MUL         _mm512_mul_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx16_REDUCE(res, x)                               \
ਕਰੋ {                                                              \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;                               \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = mm512_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                  \
    }                                                             \
    ਵਿੱਥ >>= 1;                                                 \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = mm512_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                  \
    }                                                             \
    ਵਿੱਥ >>= 1;                                                 \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = mm512_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                  \
    }                                                             \
    res = (ggml_float) mm512_reduce_ਜੋੜੋ_ps(x[0]);                \
} ਜਦੋਂ (0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F32Cx16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F32Cx16_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F32Cx16_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     GGML_F32_Cx16_ਲੱਦੋ(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F32Cx16_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F32_Cx16_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F32Cx16_MUL

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F32Cx16_REDUCE

#ਜੇ_ਅੰਤ // __AVX512FP16__
#ਜਾਂਜੇ defined(__AVX__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 AVX

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8         __m256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_ZERO    _mm256_setzero_ps()
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_ਲੱਦੋ    _mm256_loadu_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_STORE   _mm256_storeu_ps
#ਜੇ defined(__FMA__)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#ਨਹੀਂਤਾਂ
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_FMA(a, b, c) mm256_ਜੋੜੋ_ps(_mm256_mul_ps(b, c), a)
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_ਜੋੜੋ     mm256_ਜੋੜੋ_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_MUL     _mm256_mul_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_REDUCE(res, x)                                 \
ਕਰੋ {                                                              \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;                               \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = mm256_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                  \
    }                                                             \
    ਵਿੱਥ >>= 1;                                                 \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = mm256_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                  \
    }                                                             \
    ਵਿੱਥ >>= 1;                                                 \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = mm256_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                  \
    }                                                             \
    ਅਚਲ __m128 t0 = mm_ਜੋੜੋ_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    ਅਚਲ __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} ਜਦੋਂ (0)
// TODO: is this optimal ?

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x8_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x8_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x8_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x8_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x8_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x8_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 AVX

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8             __m256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#ਜੇ defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx8_ਲੱਦੋ(x)     _mm256_cvtph_ps(_mm_loadu_si128((ਅਚਲ __m128i *)(x)))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਸਤਰਅੰਦਰ __m256 __avx_f32cx8_load(ਅਚਲ ggml_fp16_t * x) {
    ਦਸ਼ਮਲਵ tmp[8];

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i]);
    }

    ਵਾਪਸ _mm256_loadu_ps(tmp);
}
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    ਦਸ਼ਮਲਵ arr[8];

    _mm256_storeu_ps(arr, y);

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++)
        x[i] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(arr[i]);
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx8_ਲੱਦੋ(x)     __avx_f32cx8_load(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_FMA         GGML_F32x8_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx8_ਜੋੜੋ         mm256_ਜੋੜੋ_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_MUL         _mm256_mul_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F32Cx8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     GGML_F32_Cx8_ਲੱਦੋ(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F32_Cx8_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#ਜਾਂਜੇ defined(__POWER9_VECTOR__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 POWER9

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4              ਸਦਿਸ਼ ਦਸ਼ਮਲਵ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ZERO         {0.0f}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_SET1         vec_splats
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਲੱਦੋ(p)      vec_xl(0, p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਜੋੜੋ          vec_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_MUL          vec_mul
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_REDUCE(res, x)              \
{                                              \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;            \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {         \
        x[i] = vec_ਜੋੜੋ(x[i], x[ਵਿੱਥ+i]);     \
    }                                          \
    ਵਿੱਥ >>= 1;                              \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {         \
        x[i] = vec_ਜੋੜੋ(x[i], x[ਵਿੱਥ+i]);     \
    }                                          \
    ਵਿੱਥ >>= 1;                              \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {         \
        x[i] = vec_ਜੋੜੋ(x[i], x[ਵਿੱਥ+i]);     \
    }                                          \
    res = vec_ਕੱਢੋ(x[0], 0) +               \
          vec_ਕੱਢੋ(x[0], 1) +               \
          vec_ਕੱਢੋ(x[0], 2) +               \
          vec_ਕੱਢੋ(x[0], 3);                \
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3)        \
{                                                       \
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ v = vec_ਜੋੜੋ(vec_ਜੋੜੋ(s0, s1),           \
                             vec_ਜੋੜੋ(s2, s3));          \
    v = vec_ਜੋੜੋ(v, vec_sld(v, v, 8));                   \
    v = vec_ਜੋੜੋ(v, vec_sld(v, v, 4));                   \
    res += (ggml_float) vec_ਕੱਢੋ(v, 0);              \
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x4_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 POWER9
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ       GGML_F32_ਕਦਮ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR        GGML_F32_EPR
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC        GGML_F32x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1   GGML_F32x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA    GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ    GGML_F32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL    GGML_F32x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i) (i & 0x1) ?                   \
  vec_ਕੱਢੋ_fp32_from_shorth(vec_xl(0, p - GGML_F16_EPR)) : \
  vec_ਕੱਢੋ_fp32_from_shortl(vec_xl(0, p))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ ggml_endian_byte(ਪੂਰਨ i) {
       uint16_t tmp_val = 1;
       ਵਾਪਸ ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *)&tmp_val)[i];
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ENDIAN_BYTE(i) ggml_endian_byte(i)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i)                             \
  ਜੇ (i & 0x1)                                                  \
    vec_xst(vec_pack_ਮੰਜ਼ਿਲ_short_fp32(r[i - GGML_ENDIAN_BYTE(1)],  \
                                   r[i - GGML_ENDIAN_BYTE(0)]), \
            0, p - GGML_F16_EPR)

//BF16 POWER9
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_ਕਦਮ 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_EPR  8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16x8         ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16x8_ZERO    vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ)0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16x8_ਲੱਦੋ(p) vec_xl(0, (ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ *)(p))

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_VEC          GGML_BF16x8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_VEC_ZERO     GGML_BF16x8_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_VEC_ਲੱਦੋ     GGML_BF16x8_ਲੱਦੋ
#ਜੇ defined(__LITTLE_ENDIAN__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_ਮੰਜ਼ਿਲ_F32_LO(v) ((ਸਦਿਸ਼ ਦਸ਼ਮਲਵ) vec_mergel(GGML_BF16_VEC_ZERO, (v)))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_ਮੰਜ਼ਿਲ_F32_HI(v) ((ਸਦਿਸ਼ ਦਸ਼ਮਲਵ) vec_mergeh(GGML_BF16_VEC_ZERO, (v)))
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_ਮੰਜ਼ਿਲ_F32_LO(v) ((ਸਦਿਸ਼ ਦਸ਼ਮਲਵ) vec_mergel((v), GGML_BF16_VEC_ZERO))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_ਮੰਜ਼ਿਲ_F32_HI(v) ((ਸਦਿਸ਼ ਦਸ਼ਮਲਵ) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_ਮੰਜ਼ਿਲ_F32_LO(x), GGML_BF16_ਮੰਜ਼ਿਲ_F32_LO(y))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_ਮੰਜ਼ਿਲ_F32_HI(x), GGML_BF16_ਮੰਜ਼ਿਲ_F32_HI(y))

#ਜਾਂਜੇ defined(__wasm_simd128__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 WASM

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4              v128_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਲੱਦੋ         wasm_v128_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_STORE        wasm_v128_store
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_FMA(a, b, c) wasm_f32x4_ਜੋੜੋ(wasm_f32x4_mul(b, c), a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਜੋੜੋ          wasm_f32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_MUL          wasm_f32x4_mul
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;                \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {             \
        x[i] = wasm_f32x4_ਜੋੜੋ(x[i], x[ਵਿੱਥ+i]);  \
    }                                              \
    ਵਿੱਥ >>= 1;                                  \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {             \
        x[i] = wasm_f32x4_ਜੋੜੋ(x[i], x[ਵਿੱਥ+i]);  \
    }                                              \
    ਵਿੱਥ >>= 1;                                  \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {             \
        x[i] = wasm_f32x4_ਜੋੜੋ(x[i], x[ਵਿੱਥ+i]);  \
    }                                              \
    res = wasm_f32x4_ਕੱਢੋ_lane(x[0], 0) +       \
          wasm_f32x4_ਕੱਢੋ_lane(x[0], 1) +       \
          wasm_f32x4_ਕੱਢੋ_lane(x[0], 2) +       \
          wasm_f32x4_ਕੱਢੋ_lane(x[0], 3);        \
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x4_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 WASM

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  4

ਸਤਰਅੰਦਰ ਸਥਿਰ v128_t __wasm_f16x4_load(ਅਚਲ ggml_fp16_t * p) {
    ਦਸ਼ਮਲਵ tmp[4];

    tmp[0] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(p[0]);
    tmp[1] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(p[1]);
    tmp[2] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(p[2]);
    tmp[3] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(p[3]);

    ਵਾਪਸ wasm_v128_ਲੱਦੋ(tmp);
}

ਸਤਰਅੰਦਰ ਸਥਿਰ ਖਾਲੀ __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
    ਦਸ਼ਮਲਵ tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(tmp[0]);
    p[1] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(tmp[1]);
    p[2] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(tmp[2]);
    p[3] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(tmp[3]);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x4             v128_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x4_ਲੱਦੋ(x)     __wasm_f16x4_load(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x4_FMA         GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x4_ਜੋੜੋ         wasm_f32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x4_MUL         wasm_f32x4_mul
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16x4_REDUCE(res, x)                           \
{                                                           \
    ਪੂਰਨ ਵਿੱਥ = GGML_F16_ARR >> 1;                         \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                      \
        x[i] = wasm_f32x4_ਜੋੜੋ(x[i], x[ਵਿੱਥ+i]);           \
    }                                                       \
    ਵਿੱਥ >>= 1;                                           \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                      \
        x[i] = wasm_f32x4_ਜੋੜੋ(x[i], x[ਵਿੱਥ+i]);           \
    }                                                       \
    ਵਿੱਥ >>= 1;                                           \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                      \
        x[i] = wasm_f32x4_ਜੋੜੋ(x[i], x[ਵਿੱਥ+i]);           \
    }                                                       \
    res = (ggml_float) (wasm_f32x4_ਕੱਢੋ_lane(x[0], 0) +  \
          wasm_f32x4_ਕੱਢੋ_lane(x[0], 1) +                \
          wasm_f32x4_ਕੱਢੋ_lane(x[0], 2) +                \
          wasm_f32x4_ਕੱਢੋ_lane(x[0], 3));                \
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F16x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F16x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F16x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     GGML_F16x4_ਲੱਦੋ(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) GGML_F16x4_STORE(p, r[i])
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F16x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F16x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F16x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F16x4_REDUCE

#ਜਾਂਜੇ defined(__SSE3__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 SSE

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4         __m128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ZERO    _mm_setzero_ps()
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_SET1(x) _mm_set1_ps(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਲੱਦੋ    _mm_loadu_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_STORE   _mm_storeu_ps
#ਜੇ defined(__FMA__)
    // TODO: Does this work?
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#ਨਹੀਂਤਾਂ
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_FMA(a, b, c) mm_ਜੋੜੋ_ps(_mm_mul_ps(b, c), a)
#ਜੇ_ਅੰਤ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਜੋੜੋ     mm_ਜੋੜੋ_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_MUL     _mm_mul_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;                               \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = mm_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                     \
    }                                                             \
    ਵਿੱਥ >>= 1;                                                 \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = mm_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                     \
    }                                                             \
    ਵਿੱਥ >>= 1;                                                 \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = mm_ਜੋੜੋ_ps(x[i], x[ਵਿੱਥ+i]);                     \
    }                                                             \
    ਅਚਲ __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x4_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 SSE

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  4

ਸਥਿਰ ਸਤਰਅੰਦਰ __m128 __sse_f16x4_load(ਅਚਲ ggml_fp16_t * x) {
    ਦਸ਼ਮਲਵ tmp[4];

    tmp[0] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[0]);
    tmp[1] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[1]);
    tmp[2] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[2]);
    tmp[3] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[3]);

    ਵਾਪਸ _mm_loadu_ps(tmp);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __sse_f16x4_store(ggml_fp16_t * x, __m128 y) {
    ਦਸ਼ਮਲਵ arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(arr[0]);
    x[1] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(arr[1]);
    x[2] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(arr[2]);
    x[3] = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(arr[3]);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4             __m128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_ZERO        _mm_setzero_ps()
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx4_ਲੱਦੋ(x)     __sse_f16x4_load(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_FMA         GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx4_ਜੋੜੋ         mm_ਜੋੜੋ_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_MUL         _mm_mul_ps
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                 GGML_F32Cx4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)      GGML_F32_Cx4_ਲੱਦੋ(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ             GGML_F32_Cx4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#ਜਾਂਜੇ defined(__loongarch_asx)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 LASX
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8         __m256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_ਲੱਦੋ(x) (__m256)__lasx_xvld((x), 0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_ਜੋੜੋ     __lasx_xvfadd_s
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_MUL     __lasx_xvfmul_s
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x8_REDUCE(res, x)                                 \
ਕਰੋ {                                                              \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;                               \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[ਵਿੱਥ+i]);                  \
    }                                                             \
    ਵਿੱਥ >>= 1;                                                 \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[ਵਿੱਥ+i]);                  \
    }                                                             \
    ਵਿੱਥ >>= 1;                                                 \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[ਵਿੱਥ+i]);                  \
    }                                                             \
    ਦਸ਼ਮਲਵ *tmp_p = (ਦਸ਼ਮਲਵ *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} ਜਦੋਂ (0)
// TODO: is this optimal ?

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x8_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x8_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x8_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x8_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x8_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x8_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 LASX

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  8

// F16 arithmetic is not supported by LASX, so we use F32 instead

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8          __m256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))

ਸਥਿਰ ਸਤਰਅੰਦਰ __m256 __lasx_f32cx8_load(ਅਚਲ ggml_fp16_t * x) {
    __m256i a;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&a, x, ਆਕਾਰ(ggml_fp16_t) * 8);
    a = __lasx_xvpermi_d(a, 0 | (1 << 4));
    ਵਾਪਸ __lasx_xvfcvtl_s_h(a);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __lasx_f32cx8_store(ggml_fp16_t * x, __m256 y) {
    __m256i a = __lasx_xvfcvt_h_s(y, y);
    a = __lasx_xvpermi_d(a, 0 | (2 << 2));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(x, &a, ਆਕਾਰ(ggml_fp16_t) * 8);
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx8_ਲੱਦੋ(x)     __lasx_f32cx8_load(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_FMA         GGML_F32x8_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx8_ਜੋੜੋ         __lasx_xvfadd_s
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_MUL         __lasx_xvfmul_s
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F32Cx8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     GGML_F32_Cx8_ਲੱਦੋ(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F32_Cx8_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#ਜਾਂਜੇ defined(__loongarch_sx)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 LSX

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4         __m128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ZERO    (__m128)__lsx_vldi(0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_SET1(x) (__m128)__lsx_vreplfr2vr_s((x))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਲੱਦੋ(x) (__m128)__lsx_vld((x), 0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_STORE(x, y)   __lsx_vst(y, x, 0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਜੋੜੋ     __lsx_vfadd_s
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_MUL     __lsx_vfmul_s

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_REDUCE(res, x)                               \
{                                                               \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;                             \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[ਵਿੱਥ+i]);                \
    }                                                           \
    ਵਿੱਥ >>= 1;                                               \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[ਵਿੱਥ+i]);                \
    }                                                           \
    ਵਿੱਥ >>= 1;                                               \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[ਵਿੱਥ+i]);                \
    }                                                           \
    __m128i t0 = __lsx_vpickev_w((__m128i)x[0], (__m128i)x[0]); \
    __m128i t1 = __lsx_vpickod_w((__m128i)x[0], (__m128i)x[0]); \
    __m128 t2 = __lsx_vfadd_s((__m128)t0, (__m128)t1);          \
    __m128i t3 = __lsx_vpickev_w((__m128i)t2, (__m128i)t2);     \
    __m128i t4 = __lsx_vpickod_w((__m128i)t2, (__m128i)t2);     \
    __m128 t5 = __lsx_vfadd_s((__m128)t3, (__m128)t4);          \
    res = (ggml_float) ((v4f32)t5)[0];                          \
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x4_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 LSX

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  4

ਸਥਿਰ ਸਤਰਅੰਦਰ __m128 __lsx_f16x4_load(ਅਚਲ ggml_fp16_t * x) {
    ਵਾਪਸ __lsx_vfcvtl_s_h(__lsx_vld((ਅਚਲ ਖਾਲੀ *)x, 0));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __lsx_f16x4_store(ggml_fp16_t * x, __m128 y) {
    __m128i a = __lsx_vfcvt_h_s(y, y);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(x, &a, ਆਕਾਰ(ggml_fp16_t) * 4);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4             __m128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_ZERO        (__m128)__lsx_vldi(0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_SET1(x)     (__m128)__lsx_vreplfr2vr_s((x))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx4_ਲੱਦੋ(x)     (__m128)__lsx_f16x4_load(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_FMA         GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_Cx4_ਜੋੜੋ         __lsx_vfadd_s
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_MUL         __lsx_vfmul_s
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                 GGML_F32Cx4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)      GGML_F32_Cx4_ਲੱਦੋ(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ             GGML_F32_Cx4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32 s390x

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4              float32x4_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ZERO         vec_splats(0.0f)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_SET1         vec_splats
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਲੱਦੋ(p)      vec_xl(0, p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਜੋੜੋ          vec_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_MUL          vec_mul
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_REDUCE(res, x)                   \
{                                                   \
    ਪੂਰਨ ਵਿੱਥ = GGML_F32_ARR >> 1;                 \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {              \
        x[i] = vec_ਜੋੜੋ(x[i], x[ਵਿੱਥ + i]);        \
    }                                               \
    ਵਿੱਥ >>= 1;                                   \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {              \
        x[i] = vec_ਜੋੜੋ(x[i], x[ਵਿੱਥ + i]);        \
    }                                               \
    ਵਿੱਥ >>= 1;                                   \
    ਲਈ (ਪੂਰਨ i = 0; i < ਵਿੱਥ; ++i) {              \
        x[i] = vec_ਜੋੜੋ(x[i], x[ਵਿੱਥ + i]);        \
    }                                               \
    float32x4_t tmp = x[0] + vec_reve(x[0]);        \
    res = tmp[0] + tmp[1];                          \
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3) \
{                                                \
    float32x4_t v = vec_ਜੋੜੋ(vec_ਜੋੜੋ(s0, s1),     \
                            vec_ਜੋੜੋ(s2, s3));    \
    v = vec_ਜੋੜੋ(v, vec_sld(v, v, 8));            \
    v = vec_ਜੋੜੋ(v, vec_sld(v, v, 4));            \
    res += (ggml_float)vec_ਕੱਢੋ(v, 0);        \
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x4_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 s390x
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ਕਦਮ GGML_F32_ਕਦਮ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_EPR  GGML_F32_EPR

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32x4_t __lzs_f32cx4_to_f16(float32x4_t v_f) {
    float32x4_t v_ਅਧਾਰ = vec_mul(vec_mul(vec_abs(v_f), vec_splats(0x1.0p+112f)), vec_splats(0x1.0p-110f));

    ਅਚਲ uint32x4_t v_w      = (uint32x4_t)v_f;
    ਅਚਲ uint32x4_t v_shl1_w = vec_ਜੋੜੋ(v_w, v_w);
    ਅਚਲ uint32x4_t v_sign   = vec_and(v_w, vec_splats(UINT32_C(0x80000000)));
    ਅਚਲ uint32x4_t v_bias   = vec_max(vec_and(v_shl1_w, vec_splats(UINT32_C(0xFF000000))), vec_splats(UINT32_C(0x71000000)));

    v_ਅਧਾਰ = vec_ਜੋੜੋ((float32x4_t)vec_ਜੋੜੋ(vec_sr(v_bias, 1), vec_splats(UINT32_C(0x07800000))), v_ਅਧਾਰ);

    ਅਚਲ uint32x4_t v_ਬਿੱਟ    = (uint32x4_t)v_ਅਧਾਰ;
    ਅਚਲ uint32x4_t v_nonsign = vec_ਜੋੜੋ(vec_and(vec_sr(v_ਬਿੱਟ, 13), vec_splats(UINT32_C(0x00007C00))),
                                         vec_and(v_ਬਿੱਟ, vec_splats(UINT32_C(0x00000FFF))));
    ਅਚਲ uint32x4_t v_is_nan  = (uint32x4_t)vec_cmpgt(v_shl1_w, vec_splats(UINT32_C(0xFF000000)));

    ਵਾਪਸ vec_or(vec_sr(v_sign, 16), vec_sel(v_nonsign, vec_splats(UINT32_C(0x7E00)), v_is_nan));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ float32x4_t __lzs_f16cx4_load(ਅਚਲ ggml_fp16_t * x) {
    ਦਸ਼ਮਲਵ tmp[4];

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i]);
    }

    // note: keep type-cast here to prevent compiler bugs
    // see: https://github.com/ggml-org/llama.cpp/issues/12846
    ਵਾਪਸ vec_xl(0, (ਅਚਲ ਦਸ਼ਮਲਵ *)(tmp));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __lzs_f16cx4_store(ggml_fp16_t * x, float32x4_t v_y) {
    ਅਚਲ uint32x4_t v_h = __lzs_f32cx4_to_f16(v_y);
    ਅਚਲ uint64_t   tmp = ((uint64x2_t)vec_pack(v_h, v_h))[0];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(x, &tmp, ਆਕਾਰ(tmp));
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC                GGML_F32x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ZERO           GGML_F32x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_SET1           GGML_F32x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਲੱਦੋ(p, i)     __lzs_f16cx4_load(p)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_STORE(p, r, i) __lzs_f16cx4_store(p, r[i])
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_FMA            GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_ਜੋੜੋ            GGML_F32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_MUL            GGML_F32x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_VEC_REDUCE         GGML_F32x4_REDUCE

// BF16 s390x
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_ਕਦਮ 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_EPR  8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16x8         __vector ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16x8_ZERO    vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ)0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16x8_ਲੱਦੋ(p) vec_xl(0, (ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ *)(p))

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_VEC      GGML_BF16x8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_VEC_ZERO GGML_BF16x8_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_VEC_ਲੱਦੋ GGML_BF16x8_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_ਮੰਜ਼ਿਲ_F32_LO(v) ((float32x4_t) vec_mergel((v), GGML_BF16_VEC_ZERO))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_ਮੰਜ਼ਿਲ_F32_HI(v) ((float32x4_t) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_ਮੰਜ਼ਿਲ_F32_LO(x), GGML_BF16_ਮੰਜ਼ਿਲ_F32_LO(y))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_ਮੰਜ਼ਿਲ_F32_HI(x), GGML_BF16_ਮੰਜ਼ਿਲ_F32_HI(y))

#ਜਾਂਜੇ defined(__riscv_v_intrinsic)

// compatible with vlen >= 128

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SIMD

// F32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ਕਦਮ 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_EPR  4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4              vfloat32m1_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ZERO         __riscv_vfmv_v_f_f32m1(0.0f, GGML_F32_EPR)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_SET1(x)      __riscv_vfmv_v_f_f32m1(x, GGML_F32_EPR)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਲੱਦੋ(x)      __riscv_vle32_v_f32m1(x, GGML_F32_EPR)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_STORE(b, v)  __riscv_vse32_v_f32m1(b, v, GGML_F32_EPR)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_FMA(a, b, c) __riscv_vfmacc_vv_f32m1(a, b, c, GGML_F32_EPR)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_ਜੋੜੋ(a, b)    __riscv_vfadd_vv_f32m1(a, b, GGML_F32_EPR)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32x4_MUL(a, b)    __riscv_vfmul_vv_f32m1(a, b, GGML_F32_EPR)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC        GGML_F32x4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_SET1   GGML_F32x4_SET1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਲੱਦੋ   GGML_F32x4_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_STORE  GGML_F32x4_STORE
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_FMA    GGML_F32x4_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_ਜੋੜੋ    GGML_F32x4_ਜੋੜੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_MUL    GGML_F32x4_MUL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

#ਜੇ_ਅੰਤ

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SIMD
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F32_ARR (GGML_F32_ਕਦਮ/GGML_F32_EPR)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_ARR (GGML_F16_ਕਦਮ/GGML_F16_EPR)
#ਜੇ_ਅੰਤ

// GGML_F16_DOT_*
// like GGML_F16_* but for dot products which need F32 accumulation on AVX512-FP16

#ਜੇ defined(__AVX512FP16__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_ਕਦਮ           GGML_F32_ਕਦਮ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_EPR            GGML_F32_EPR
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_ARR            GGML_F32_ARR
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC            GGML_F32x16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC_ZERO       GGML_F32x16_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC_ਲੱਦੋ(p, i) _mm512_cvtph_ps(_mm256_loadu_si256((ਅਚਲ __m256i *)(p)))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC_FMA        GGML_F32x16_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC_REDUCE     GGML_F32x16_REDUCE

#ਨਹੀਂਤਾਂ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_ਕਦਮ           GGML_F16_ਕਦਮ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_EPR            GGML_F16_EPR
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_ARR            GGML_F16_ARR
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC            GGML_F16_VEC
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC_ZERO       GGML_F16_VEC_ZERO
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC_ਲੱਦੋ       GGML_F16_VEC_ਲੱਦੋ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC_FMA        GGML_F16_VEC_FMA
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_F16_DOT_VEC_REDUCE     GGML_F16_VEC_REDUCE

#ਜੇ_ਅੰਤ // defined(__AVX512FP16__)

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
