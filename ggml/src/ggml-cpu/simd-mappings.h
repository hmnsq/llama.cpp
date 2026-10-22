#సూచన once

#అన్తర్భావ "ggml-cpu-impl.h"

#నిర్వచితమైతే __ARM_FEATURE_SVE
#అన్తర్భావ <arm_sve.h>
#అయితే_ముగింపు // __ARM_FEATURE_SVE

#ఒకవేళ defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#అన్తర్భావ <arm_neon.h>
#అయితే_ముగింపు

#ఒకవేళ defined(__riscv_v_intrinsic)
#అన్తర్భావ <riscv_vector.h>
#అయితే_ముగింపు

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

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
#ఒకవేళ defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
    #నిర్వచించు GGML_CPU_COMPUTE_FP16_గమ్యం_FP32(x) neon_compute_fp16_గమ్యం_fp32(x)
    #నిర్వచించు GGML_CPU_COMPUTE_FP32_గమ్యం_FP16(x) neon_compute_fp32_గమ్యం_fp16(x)

    #నిర్వచించు GGML_CPU_FP16_గమ్యం_FP32(x) GGML_CPU_COMPUTE_FP16_గమ్యం_FP32(x)
    #నిర్వచించు GGML_CPU_FP32_గమ్యం_FP16(x) GGML_CPU_COMPUTE_FP32_గమ్యం_FP16(x)

    స్థిర అంతఃవరుస దశాంశం neon_compute_fp16_గమ్యం_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        జ్ఞాపకం_నకలు(&tmp, &h, పరిమాణం(ggml_fp16_t));
        తిరిగి (దశాంశం)tmp;
    }

    స్థిర అంతఃవరుస ggml_fp16_t neon_compute_fp32_గమ్యం_fp16(దశాంశం f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        జ్ఞాపకం_నకలు(&res, &tmp, పరిమాణం(ggml_fp16_t));
        తిరిగి res;
    }
#లేదావేళ defined(__F16C__)
    #నిర్వచితమైతే _MSC_VER
        #నిర్వచించు GGML_CPU_COMPUTE_FP16_గమ్యం_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #నిర్వచించు GGML_CPU_COMPUTE_FP32_గమ్యం_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(mm_సమితి_ss(x), 0), 0)
    #లేకపోతే
        #నిర్వచించు GGML_CPU_COMPUTE_FP16_గమ్యం_FP32(x) _cvtsh_ss(x)
        #నిర్వచించు GGML_CPU_COMPUTE_FP32_గమ్యం_FP16(x) _cvtss_sh(x, 0)
    #అయితే_ముగింపు
#లేదావేళ defined(__POWER9_VECTOR__)
    #నిర్వచించు GGML_CPU_COMPUTE_FP16_గమ్యం_FP32(x) power_compute_fp16_గమ్యం_fp32(x)
    #నిర్వచించు GGML_CPU_COMPUTE_FP32_గమ్యం_FP16(x) power_compute_fp32_గమ్యం_fp16(x)
    /* the inline asm below is about 12% faster than the lookup method */
    #నిర్వచించు GGML_CPU_FP16_గమ్యం_FP32(x) GGML_CPU_COMPUTE_FP16_గమ్యం_FP32(x)
    #నిర్వచించు GGML_CPU_FP32_గమ్యం_FP16(x) GGML_CPU_COMPUTE_FP32_గమ్యం_FP16(x)

    స్థిర అంతఃవరుస దశాంశం power_compute_fp16_గమ్యం_fp32(ggml_fp16_t h) {
        దశాంశం f;
        ద్విగుణ d;
        __asm__(
            "mtfprd %0,%2\n"
            "xscvhpdp %0,%0\n"
            "frsp %1,%0\n" :
            /*తాత్కాలిక*/ "=d"(d),
            /*బహిర్గతం*/  "=f"(f):
            /*లోపల*/   "r"(h));
        తిరిగి f;
    }

    స్థిర అంతఃవరుస ggml_fp16_t power_compute_fp32_గమ్యం_fp16(దశాంశం f) {
        ద్విగుణ d;
        ggml_fp16_t r;
        __asm__( /* xscvdphp can work on double or single precision */
            "xscvdphp %0,%2\n"
            "mffprd %1,%0\n" :
            /*తాత్కాలిక*/ "=d"(d),
            /*బహిర్గతం*/  "=r"(r):
            /*లోపల*/   "f"(f));
        తిరిగి r;
    }
#లేదావేళ defined(__riscv) && defined(__riscv_zfhmin)
    స్థిర అంతఃవరుస దశాంశం riscv_compute_fp16_గమ్యం_fp32(ggml_fp16_t h) {
        _Float16 hf;
        జ్ఞాపకం_నకలు(&hf, &h, పరిమాణం(ggml_fp16_t));
        తిరిగి hf;
    }

    స్థిర అంతఃవరుస ggml_fp16_t riscv_compute_fp32_గమ్యం_fp16(దశాంశం f) {
        ggml_fp16_t res;
        _Float16 hf = (_Float16)f;
        జ్ఞాపకం_నకలు(&res, &hf, పరిమాణం(ggml_fp16_t));
        తిరిగి res;
    }

    #నిర్వచించు GGML_CPU_COMPUTE_FP16_గమ్యం_FP32(x) riscv_compute_fp16_గమ్యం_fp32(x)
    #నిర్వచించు GGML_CPU_COMPUTE_FP32_గమ్యం_FP16(x) riscv_compute_fp32_గమ్యం_fp16(x)
    #నిర్వచించు GGML_CPU_FP16_గమ్యం_FP32(x) GGML_CPU_COMPUTE_FP16_గమ్యం_FP32(x)
    #నిర్వచించు GGML_CPU_FP32_గమ్యం_FP16(x) GGML_CPU_COMPUTE_FP32_గమ్యం_FP16(x)
#అయితే_ముగింపు

// precomputed f32 table for f16 (256 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
బాహ్య దశాంశం ggml_పట్టిక_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
బాహ్య దశాంశం ggml_పట్టిక_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
బాహ్య దశాంశం ggml_పట్టిక_f32_ue4m3[1 << 8];

// Use lookup table for E8M0 on x86 (faster than bit manipulation)
#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#నిర్వచించు GGML_CPU_E8_M0_గమ్యం_FP32_HALF(x) ggml_పట్టిక_f32_e8m0_half[(uint8_t)(x)]
#లేకపోతే
#నిర్వచించు GGML_CPU_E8_M0_గమ్యం_FP32_HALF(x) GGML_E8_M0_గమ్యం_FP32_HALF(x)
#అయితే_ముగింపు

// Use lookup table for UE4M3 on x86 and ARM (faster than bit manipulation)
#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__ARM_NEON)
#నిర్వచించు GGML_CPU_UE4_M3_గమ్యం_FP32(x) ggml_పట్టిక_f32_ue4m3[(uint8_t)(x)]
#లేకపోతే
#నిర్వచించు GGML_CPU_UE4_M3_గమ్యం_FP32(x) ggml_ue4m3_గమ్యం_fp32(x)
#అయితే_ముగింపు

// Pre AVX512 CPUs (AVX2 with F16C, such as i5-8th gen or i7-4790)
// do not produce codegen as checks below don't test for __F16C__
// but this check does.
#ఒకవేళ defined(__F16C__)
// force FP16 -> FP32 conversion to use LUT instead
// see: https://github.com/ggml-org/llama.cpp/issues/30342
//#define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
#నిర్వచించు GGML_CPU_FP32_గమ్యం_FP16(x) GGML_CPU_COMPUTE_FP32_గమ్యం_FP16(x)
#అయితే_ముగింపు

// a branch that defines a GGML_CPU_COMPUTE_* macro must also select it, or the fallback takes over silently
#ఒకవేళ defined(GGML_CPU_COMPUTE_FP16_గమ్యం_FP32) && !defined(GGML_CPU_FP16_గమ్యం_FP32) && !defined(__F16C__) // see: https://github.com/ggml-org/llama.cpp/issues/30342
#దోషం "a branch defined GGML_CPU_COMPUTE_FP16_TO_FP32 without selecting it as GGML_CPU_FP16_TO_FP32"
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_CPU_COMPUTE_FP32_గమ్యం_FP16) && !defined(GGML_CPU_FP32_గమ్యం_FP16)
#దోషం "a branch defined GGML_CPU_COMPUTE_FP32_TO_FP16 without selecting it as GGML_CPU_FP32_TO_FP16"
#అయితే_ముగింపు

// fall back to the lookup table / the portable software conversion for whichever direction no branch selected
#ఒకవేళ !defined(GGML_CPU_FP16_గమ్యం_FP32)
అంతఃవరుస స్థిర దశాంశం ggml_lookup_fp16_గమ్యం_fp32(ggml_fp16_t f) {
    uint16_t s;
    జ్ఞాపకం_నకలు(&s, &f, పరిమాణం(uint16_t));
    తిరిగి ggml_పట్టిక_f32_f16[s];
}

#నిర్వచించు GGML_CPU_FP16_గమ్యం_FP32(x) ggml_lookup_fp16_గమ్యం_fp32(x)
#అయితే_ముగింపు

#ఒకవేళ !defined(GGML_CPU_FP32_గమ్యం_FP16)
#నిర్వచించు GGML_CPU_FP32_గమ్యం_FP16(x) GGML_COMPUTE_FP32_గమ్యం_FP16(x)
#అయితే_ముగింపు


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

#ఒకవేళ defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_FMA)

#నిర్వచించు GGML_SIMD

// F32 SVE
#నిర్వచించు GGML_F32_EPR 8
#నిర్వచించు DEFAULT_PG svptrue_b32()

#నిర్వచించు GGML_F32xt                        svfloat32_t
#నిర్వచించు GGML_F32xt_ZERO                   svdup_n_f32(0.0f)
#నిర్వచించు GGML_F32xt_SET1(x)                svdup_n_f32(x)
#నిర్వచించు GGML_F32xt_ఎక్కించు_IMPL(pg, a)       svld1_f32(pg, a)
#నిర్వచించు GGML_F32xt_ఎక్కించు(a)                GGML_F32xt_ఎక్కించు_IMPL(DEFAULT_PG, a)
#నిర్వచించు GGML_F32xt_STORE_IMPL(pg, a, b)   svst1_f32(pg, a, b)
#నిర్వచించు GGML_F32xt_STORE(a, b)            GGML_F32xt_STORE_IMPL(DEFAULT_PG, a, b)
#నిర్వచించు GGML_F32xt_FMA_IMPL(pg, a, b, c)  svmad_f32_m(pg, b, c, a)
#నిర్వచించు GGML_F32xt_FMA(a, b, c)           GGML_F32xt_FMA_IMPL(DEFAULT_PG, a, b, c)
#నిర్వచించు GGML_F32xt_కూడు_IMPL(pg, a, b)     svadd_f32_m(pg, a, b)
#నిర్వచించు GGML_F32xt_కూడు(a, b)              GGML_F32xt_కూడు_IMPL(DEFAULT_PG, a, b)
#నిర్వచించు GGML_F32xt_MUL_IMPL(pg, a, b)     svmul_f32_m(pg, a, b)
#నిర్వచించు GGML_F32xt_MUL(a, b)              GGML_F32xt_MUL_IMPL(DEFAULT_PG, a, b)
#నిర్వచించు GGML_F32xt_REDUCE_ONE_IMPL(pg, a) svaddv(pg, a)
#నిర్వచించు GGML_F32xt_REDUCE_ONE(a)          GGML_F32xt_REDUCE_ONE_IMPL(DEFAULT_PG, a)
#నిర్వచించు GGML_F32xt_REDUCE_IMPL(pg, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
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
#నిర్వచించు GGML_F32xt_REDUCE(res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
        GGML_F32xt_REDUCE_IMPL(DEFAULT_PG, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)

#నిర్వచించు GGML_F32_VEC        GGML_F32xt
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32xt_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32xt_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32xt_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32xt_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32xt_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32xt_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32xt_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32xt_REDUCE

// F16 SVE
#నిర్వచించు DEFAULT_PG32    svptrue_b32()
#నిర్వచించు DEFAULT_PG16    svptrue_b16()

#నిర్వచించు GGML_F32Cxt                         svfloat16_t
#నిర్వచించు GGML_F32Cxt_ZERO                    svdup_n_f16(0.0f)
#నిర్వచించు GGML_F32Cxt_SET1(x)                 svdup_n_f16(x)
#నిర్వచించు GGML_F32_Cxt_ఎక్కించు(p)                 svld1_f16(DEFAULT_PG16, (స్థిరరాశి __fp16 *)(p))
#నిర్వచించు GGML_F32Cxt_STORE(dst_ptr, src_vec) svst1_f16(DEFAULT_PG16, (__fp16 *)(dst_ptr), (src_vec))

#నిర్వచించు GGML_F32Cxt_FMA_IMPL(pg, a, b, c)   svmad_f16_x(pg, b, c, a)
#నిర్వచించు GGML_F32Cxt_FMA(a, b, c)            GGML_F32Cxt_FMA_IMPL(DEFAULT_PG16, a, b, c)
#నిర్వచించు GGML_F32_Cxt_కూడు_IMPL(pg, a, b)      svadd_f16_x(pg, a, b)
#నిర్వచించు GGML_F32_Cxt_కూడు(a, b)               GGML_F32_Cxt_కూడు_IMPL(DEFAULT_PG16, a, b)
#నిర్వచించు GGML_F32Cxt_MUL_IMPL(pg, a, b)      svmul_f16_x(pg, a, b)
#నిర్వచించు GGML_F32Cxt_MUL(a, b)               GGML_F32Cxt_MUL_IMPL(DEFAULT_PG16, a, b)
#నిర్వచించు GGML_F32Cxt_REDUCE                  GGML_F16xt_REDUCE_MIXED

#నిర్వచించు GGML_F16x_VEC                GGML_F32Cxt
#నిర్వచించు GGML_F16x_VEC_ZERO           GGML_F32Cxt_ZERO
#నిర్వచించు GGML_F16x_VEC_SET1           GGML_F32Cxt_SET1
#నిర్వచించు GGML_F16x_VEC_ఎక్కించు(p, i)     GGML_F32_Cxt_ఎక్కించు(p)
#నిర్వచించు GGML_F16x_VEC_STORE(p, r, i) GGML_F32Cxt_STORE((__fp16 *)(p), r)
#నిర్వచించు GGML_F16x_VEC_FMA            GGML_F32Cxt_FMA
#నిర్వచించు GGML_F16x_VEC_కూడు            GGML_F32_Cxt_కూడు
#నిర్వచించు GGML_F16x_VEC_MUL            GGML_F32Cxt_MUL
#నిర్వచించు GGML_F16x_VEC_REDUCE         GGML_F32Cxt_REDUCE

#నిర్వచించు GGML_F16xt_REDUCE_ONE_IMPL(pg, a) svaddv_f16(pg, a)
#నిర్వచించు GGML_F16xt_REDUCE_ONE(a)          GGML_F16xt_REDUCE_ONE_IMPL(DEFAULT_PG16, a)

#నిర్వచించు GGML_F16xt_REDUCE_MIXED_IMPL(pg16, res, sum1, sum2, sum3, sum4)  \
{                                                      \
    sum1 = svadd_f16_x(pg16, sum1, sum2);              \
    sum3 = svadd_f16_x(pg16, sum3, sum4);              \
    sum1 = svadd_f16_x(pg16, sum1, sum3);              \
    __fp16 కూడిక_f16 = svaddv_f16(pg16, sum1);           \
    (res) = (ggml_float) కూడిక_f16;                      \
}
#నిర్వచించు GGML_F16xt_REDUCE_MIXED(res, sum1, sum2, sum3, sum4)  \
        GGML_F16xt_REDUCE_MIXED_IMPL(DEFAULT_PG16, res, sum1, sum2, sum3, sum4)

// F16 NEON

#ఒకవేళ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #నిర్వచించు GGML_F16_అడుగు 32
    #నిర్వచించు GGML_F16_EPR  8

    #నిర్వచించు GGML_F16x8              float16x8_t
    #నిర్వచించు GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #నిర్వచించు GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #నిర్వచించు GGML_F16x8_ఎక్కించు(x)      vld1q_f16((స్థిరరాశి __fp16 *)(x))
    #నిర్వచించు GGML_F16x8_STORE        vst1q_f16
    #నిర్వచించు GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #నిర్వచించు GGML_F16x8_కూడు          vaddq_f16
    #నిర్వచించు GGML_F16x8_MUL          vmulq_f16
    #నిర్వచించు GGML_F16x8_REDUCE(res, x)                               \
    చేయి {                                                            \
        పూర్ణాంకం స్థానభ్రంశం = GGML_F16_ARR >> 1;                             \
        కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[స్థానభ్రంశం+i]);              \
        }                                                           \
        స్థానభ్రంశం >>= 1;                                               \
        కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[స్థానభ్రంశం+i]);              \
        }                                                           \
        స్థానభ్రంశం >>= 1;                                               \
        కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[స్థానభ్రంశం+i]);              \
        }                                                           \
        స్థిరరాశి float32x4_t t0 = vcvt_f32_f16(vget_దిగువ_f16 ((x)[0])); \
        స్థిరరాశి float32x4_t t1 = vcvt_f32_f16(vget_ఎగువ_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } అయితే (0)

    #నిర్వచించు GGML_F16_VEC                GGML_F16x8
    #నిర్వచించు GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #నిర్వచించు GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     GGML_F16x8_ఎక్కించు(p)
    #నిర్వచించు GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #నిర్వచించు GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #నిర్వచించు GGML_F16_VEC_కూడు            GGML_F16x8_కూడు
    #నిర్వచించు GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#లేకపోతే
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #నిర్వచించు GGML_F16_అడుగు 16
    #నిర్వచించు GGML_F16_EPR  4

    #నిర్వచించు GGML_F32Cx4              float32x4_t
    #నిర్వచించు GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #నిర్వచించు GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #నిర్వచించు GGML_F32_Cx4_ఎక్కించు(x)      vcvt_f32_f16(vld1_f16((స్థిరరాశి __fp16 *)(x)))
    #నిర్వచించు GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #నిర్వచించు GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #నిర్వచించు GGML_F32_Cx4_కూడు          vaddq_f32
    #నిర్వచించు GGML_F32Cx4_MUL          vmulq_f32
    #నిర్వచించు GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #నిర్వచించు GGML_F16_VEC                GGML_F32Cx4
    #నిర్వచించు GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #నిర్వచించు GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     GGML_F32_Cx4_ఎక్కించు(p)
    #నిర్వచించు GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #నిర్వచించు GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #నిర్వచించు GGML_F16_VEC_కూడు            GGML_F32_Cx4_కూడు
    #నిర్వచించు GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#అయితే_ముగింపు

#లేదావేళ defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)

#నిర్వచించు GGML_SIMD

// F32 NEON

#నిర్వచించు GGML_F32_అడుగు 16
#నిర్వచించు GGML_F32_EPR  4

#నిర్వచించు GGML_F32x4              float32x4_t
#నిర్వచించు GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#నిర్వచించు GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#నిర్వచించు GGML_F32x4_ఎక్కించు         vld1q_f32
#నిర్వచించు GGML_F32x4_STORE        vst1q_f32
#నిర్వచించు GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#నిర్వచించు GGML_F32x4_కూడు          vaddq_f32
#నిర్వచించు GGML_F32x4_MUL          vmulq_f32
#నిర్వచించు GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#నిర్వచించు GGML_F32x4_REDUCE(res, x)                       \
{                                                       \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;                     \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[స్థానభ్రంశం+i]);      \
    }                                                   \
    స్థానభ్రంశం >>= 1;                                       \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[స్థానభ్రంశం+i]);      \
    }                                                   \
    స్థానభ్రంశం >>= 1;                                       \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[స్థానభ్రంశం+i]);      \
    }                                                   \
    (res) = (ggml_float) GGML_F32x4_REDUCE_ONE((x)[0]); \
}

#నిర్వచించు GGML_F32_VEC        GGML_F32x4
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x4_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x4_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x4_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x4_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x4_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x4_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#ఒకవేళ defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #నిర్వచించు GGML_F16_అడుగు 32
    #నిర్వచించు GGML_F16_EPR  8

    #నిర్వచించు GGML_F16x8              float16x8_t
    #నిర్వచించు GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #నిర్వచించు GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #నిర్వచించు GGML_F16x8_ఎక్కించు(x)      vld1q_f16((స్థిరరాశి __fp16 *)(x))
    #నిర్వచించు GGML_F16x8_STORE        vst1q_f16
    #నిర్వచించు GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #నిర్వచించు GGML_F16x8_కూడు          vaddq_f16
    #నిర్వచించు GGML_F16x8_MUL          vmulq_f16
    #నిర్వచించు GGML_F16x8_REDUCE(res, x)                               \
    చేయి {                                                            \
        పూర్ణాంకం స్థానభ్రంశం = GGML_F16_ARR >> 1;                             \
        కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[స్థానభ్రంశం+i]);              \
        }                                                           \
        స్థానభ్రంశం >>= 1;                                               \
        కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[స్థానభ్రంశం+i]);              \
        }                                                           \
        స్థానభ్రంశం >>= 1;                                               \
        కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[స్థానభ్రంశం+i]);              \
        }                                                           \
        స్థిరరాశి float32x4_t t0 = vcvt_f32_f16(vget_దిగువ_f16 ((x)[0])); \
        స్థిరరాశి float32x4_t t1 = vcvt_f32_f16(vget_ఎగువ_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } అయితే (0)

    #నిర్వచించు GGML_F16_VEC                GGML_F16x8
    #నిర్వచించు GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #నిర్వచించు GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     GGML_F16x8_ఎక్కించు(p)
    #నిర్వచించు GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #నిర్వచించు GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #నిర్వచించు GGML_F16_VEC_కూడు            GGML_F16x8_కూడు
    #నిర్వచించు GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#లేకపోతే
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #నిర్వచించు GGML_F16_అడుగు 16
    #నిర్వచించు GGML_F16_EPR  4

    #నిర్వచించు GGML_F32Cx4              float32x4_t
    #నిర్వచించు GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #నిర్వచించు GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #నిర్వచించు GGML_F32_Cx4_ఎక్కించు(x)      vcvt_f32_f16(vld1_f16((స్థిరరాశి __fp16 *)(x)))
    #నిర్వచించు GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #నిర్వచించు GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #నిర్వచించు GGML_F32_Cx4_కూడు          vaddq_f32
    #నిర్వచించు GGML_F32Cx4_MUL          vmulq_f32
    #నిర్వచించు GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #నిర్వచించు GGML_F16_VEC                GGML_F32Cx4
    #నిర్వచించు GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #నిర్వచించు GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     GGML_F32_Cx4_ఎక్కించు(p)
    #నిర్వచించు GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #నిర్వచించు GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #నిర్వచించు GGML_F16_VEC_కూడు            GGML_F32_Cx4_కూడు
    #నిర్వచించు GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#అయితే_ముగింపు

#లేదావేళ defined(__AVX512F__)

#నిర్వచించు GGML_SIMD

// F32 AVX512

#నిర్వచించు GGML_F32_అడుగు 64
#నిర్వచించు GGML_F32_EPR  16

#నిర్వచించు GGML_F32x16         __m512
#నిర్వచించు GGML_F32x16_ZERO    _mm512_setzero_ps()
#నిర్వచించు GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#నిర్వచించు GGML_F32x16_ఎక్కించు    _mm512_loadu_ps
#నిర్వచించు GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#నిర్వచించు GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#నిర్వచించు GGML_F32x16_కూడు     mm512_కూడు_ps
#నిర్వచించు GGML_F32x16_MUL     _mm512_mul_ps
#నిర్వచించు GGML_F32x16_REDUCE(res, x)                                    \
చేయి {                                                                  \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;                                   \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                                \
        x[i] = mm512_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                      \
    }                                                                 \
    స్థానభ్రంశం >>= 1;                                                     \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                                \
        x[i] = mm512_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                      \
    }                                                                 \
    స్థానభ్రంశం >>= 1;                                                     \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                                \
        x[i] = mm512_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                      \
    }                                                                 \
    res = (ggml_float) mm512_reduce_కూడు_ps(x[0]);                    \
} అయితే (0)

// TODO: is this optimal ?

#నిర్వచించు GGML_F32_VEC        GGML_F32x16
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x16_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x16_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x16_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x16_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x16_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x16_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x16_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512

#ఒకవేళ defined(__AVX512FP16__)

#నిర్వచించు GGML_F16_అడుగు 128
#నిర్వచించు GGML_F16_EPR  32

#నిర్వచించు GGML_F16x32              __m512h
#నిర్వచించు GGML_F16x32_ZERO         _mm512_setzero_ph()
#నిర్వచించు GGML_F16x32_SET1(x)      _mm512_set1_ph(__extension__(_Float16)(x))
#నిర్వచించు GGML_F16x32_ఎక్కించు(x)      _mm512_loadu_ph(x)
#నిర్వచించు GGML_F16x32_STORE(x, y)  _mm512_storeu_ph(x, y)
#నిర్వచించు GGML_F16x32_FMA(a, b, c) _mm512_fmadd_ph(b, c, a)
#నిర్వచించు GGML_F16x32_కూడు          mm512_కూడు_ph
#నిర్వచించు GGML_F16x32_MUL          _mm512_mul_ph
#నిర్వచించు GGML_F16x32_REDUCE(res, x)                                     \
చేయి {                                                                   \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F16_ARR >> 1;                                    \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                                 \
        x[i] = mm512_కూడు_ph(x[i], x[స్థానభ్రంశం+i]);                       \
    }                                                                  \
    స్థానభ్రంశం >>= 1;                                                      \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                                 \
        x[i] = mm512_కూడు_ph(x[i], x[స్థానభ్రంశం+i]);                       \
    }                                                                  \
    స్థానభ్రంశం >>= 1;                                                      \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                                 \
        x[i] = mm512_కూడు_ph(x[i], x[స్థానభ్రంశం+i]);                       \
    }                                                                  \
    res = (ggml_float) mm512_reduce_కూడు_ph(x[0]);                     \
} అయితే (0)

#నిర్వచించు GGML_F16_VEC                GGML_F16x32
#నిర్వచించు GGML_F16_VEC_ZERO           GGML_F16x32_ZERO
#నిర్వచించు GGML_F16_VEC_SET1           GGML_F16x32_SET1
#నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     GGML_F16x32_ఎక్కించు(p)
#నిర్వచించు GGML_F16_VEC_STORE(p, r, i) GGML_F16x32_STORE(p, r[i])
#నిర్వచించు GGML_F16_VEC_FMA            GGML_F16x32_FMA
#నిర్వచించు GGML_F16_VEC_కూడు            GGML_F16x32_కూడు
#నిర్వచించు GGML_F16_VEC_MUL            GGML_F16x32_MUL
#నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F16x32_REDUCE

#లేకపోతే // Fallback FP16 <-> FP32

#నిర్వచించు GGML_F16_అడుగు 64
#నిర్వచించు GGML_F16_EPR  16

#నిర్వచించు GGML_F32Cx16             __m512
#నిర్వచించు GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#నిర్వచించు GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#నిర్వచించు GGML_F32_Cx16_ఎక్కించు(x)     _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(x)))
#నిర్వచించు GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#నిర్వచించు GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#నిర్వచించు GGML_F32_Cx16_కూడు         mm512_కూడు_ps
#నిర్వచించు GGML_F32Cx16_MUL         _mm512_mul_ps
#నిర్వచించు GGML_F32Cx16_REDUCE(res, x)                               \
చేయి {                                                              \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;                               \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = mm512_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                  \
    }                                                             \
    స్థానభ్రంశం >>= 1;                                                 \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = mm512_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                  \
    }                                                             \
    స్థానభ్రంశం >>= 1;                                                 \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = mm512_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                  \
    }                                                             \
    res = (ggml_float) mm512_reduce_కూడు_ps(x[0]);                \
} అయితే (0)

#నిర్వచించు GGML_F16_VEC                GGML_F32Cx16
#నిర్వచించు GGML_F16_VEC_ZERO           GGML_F32Cx16_ZERO
#నిర్వచించు GGML_F16_VEC_SET1           GGML_F32Cx16_SET1
#నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     GGML_F32_Cx16_ఎక్కించు(p)
#నిర్వచించు GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#నిర్వచించు GGML_F16_VEC_FMA            GGML_F32Cx16_FMA
#నిర్వచించు GGML_F16_VEC_కూడు            GGML_F32_Cx16_కూడు
#నిర్వచించు GGML_F16_VEC_MUL            GGML_F32Cx16_MUL

#నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F32Cx16_REDUCE

#అయితే_ముగింపు // __AVX512FP16__
#లేదావేళ defined(__AVX__)

#నిర్వచించు GGML_SIMD

// F32 AVX

#నిర్వచించు GGML_F32_అడుగు 32
#నిర్వచించు GGML_F32_EPR  8

#నిర్వచించు GGML_F32x8         __m256
#నిర్వచించు GGML_F32x8_ZERO    _mm256_setzero_ps()
#నిర్వచించు GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#నిర్వచించు GGML_F32x8_ఎక్కించు    _mm256_loadu_ps
#నిర్వచించు GGML_F32x8_STORE   _mm256_storeu_ps
#ఒకవేళ defined(__FMA__)
    #నిర్వచించు GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#లేకపోతే
    #నిర్వచించు GGML_F32x8_FMA(a, b, c) mm256_కూడు_ps(_mm256_mul_ps(b, c), a)
#అయితే_ముగింపు
#నిర్వచించు GGML_F32x8_కూడు     mm256_కూడు_ps
#నిర్వచించు GGML_F32x8_MUL     _mm256_mul_ps
#నిర్వచించు GGML_F32x8_REDUCE(res, x)                                 \
చేయి {                                                              \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;                               \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = mm256_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                  \
    }                                                             \
    స్థానభ్రంశం >>= 1;                                                 \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = mm256_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                  \
    }                                                             \
    స్థానభ్రంశం >>= 1;                                                 \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = mm256_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                  \
    }                                                             \
    స్థిరరాశి __m128 t0 = mm_కూడు_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    స్థిరరాశి __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} అయితే (0)
// TODO: is this optimal ?

#నిర్వచించు GGML_F32_VEC        GGML_F32x8
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x8_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x8_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x8_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x8_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x8_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x8_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 AVX

#నిర్వచించు GGML_F16_అడుగు 32
#నిర్వచించు GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#నిర్వచించు GGML_F32Cx8             __m256
#నిర్వచించు GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#నిర్వచించు GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#ఒకవేళ defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#నిర్వచించు GGML_F32_Cx8_ఎక్కించు(x)     _mm256_cvtph_ps(_mm_loadu_si128((స్థిరరాశి __m128i *)(x)))
#నిర్వచించు GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#లేకపోతే
స్థిర అంతఃవరుస __m256 __avx_f32cx8_load(స్థిరరాశి ggml_fp16_t * x) {
    దశాంశం tmp[8];

    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_గమ్యం_FP32(x[i]);
    }

    తిరిగి _mm256_loadu_ps(tmp);
}
స్థిర అంతఃవరుస శూన్య __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    దశాంశం arr[8];

    _mm256_storeu_ps(arr, y);

    కొరకు (పూర్ణాంకం i = 0; i < 8; i++)
        x[i] = GGML_CPU_FP32_గమ్యం_FP16(arr[i]);
}
#నిర్వచించు GGML_F32_Cx8_ఎక్కించు(x)     __avx_f32cx8_load(x)
#నిర్వచించు GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#అయితే_ముగింపు

#నిర్వచించు GGML_F32Cx8_FMA         GGML_F32x8_FMA
#నిర్వచించు GGML_F32_Cx8_కూడు         mm256_కూడు_ps
#నిర్వచించు GGML_F32Cx8_MUL         _mm256_mul_ps
#నిర్వచించు GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#నిర్వచించు GGML_F16_VEC                GGML_F32Cx8
#నిర్వచించు GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#నిర్వచించు GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     GGML_F32_Cx8_ఎక్కించు(p)
#నిర్వచించు GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#నిర్వచించు GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#నిర్వచించు GGML_F16_VEC_కూడు            GGML_F32_Cx8_కూడు
#నిర్వచించు GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#లేదావేళ defined(__POWER9_VECTOR__)

#నిర్వచించు GGML_SIMD

// F32 POWER9

#నిర్వచించు GGML_F32_అడుగు 32
#నిర్వచించు GGML_F32_EPR  4

#నిర్వచించు GGML_F32x4              సదిశ దశాంశం
#నిర్వచించు GGML_F32x4_ZERO         {0.0f}
#నిర్వచించు GGML_F32x4_SET1         vec_splats
#నిర్వచించు GGML_F32x4_ఎక్కించు(p)      vec_xl(0, p)
#నిర్వచించు GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#నిర్వచించు GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#నిర్వచించు GGML_F32x4_కూడు          vec_కూడు
#నిర్వచించు GGML_F32x4_MUL          vec_mul
#నిర్వచించు GGML_F32x4_REDUCE(res, x)              \
{                                              \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;            \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {         \
        x[i] = vec_కూడు(x[i], x[స్థానభ్రంశం+i]);     \
    }                                          \
    స్థానభ్రంశం >>= 1;                              \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {         \
        x[i] = vec_కూడు(x[i], x[స్థానభ్రంశం+i]);     \
    }                                          \
    స్థానభ్రంశం >>= 1;                              \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {         \
        x[i] = vec_కూడు(x[i], x[స్థానభ్రంశం+i]);     \
    }                                          \
    res = vec_వెలికితీయి(x[0], 0) +               \
          vec_వెలికితీయి(x[0], 1) +               \
          vec_వెలికితీయి(x[0], 2) +               \
          vec_వెలికితీయి(x[0], 3);                \
}
#నిర్వచించు GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3)        \
{                                                       \
    సదిశ దశాంశం v = vec_కూడు(vec_కూడు(s0, s1),           \
                             vec_కూడు(s2, s3));          \
    v = vec_కూడు(v, vec_sld(v, v, 8));                   \
    v = vec_కూడు(v, vec_sld(v, v, 4));                   \
    res += (ggml_float) vec_వెలికితీయి(v, 0);              \
}

#నిర్వచించు GGML_F32_VEC        GGML_F32x4
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x4_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x4_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x4_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x4_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x4_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x4_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 POWER9
#నిర్వచించు GGML_F16_అడుగు       GGML_F32_అడుగు
#నిర్వచించు GGML_F16_EPR        GGML_F32_EPR
#నిర్వచించు GGML_F16_VEC        GGML_F32x4
#నిర్వచించు GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#నిర్వచించు GGML_F16_VEC_SET1   GGML_F32x4_SET1
#నిర్వచించు GGML_F16_VEC_FMA    GGML_F32x4_FMA
#నిర్వచించు GGML_F16_VEC_కూడు    GGML_F32x4_కూడు
#నిర్వచించు GGML_F16_VEC_MUL    GGML_F32x4_MUL
#నిర్వచించు GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i) (i & 0x1) ?                   \
  vec_వెలికితీయి_fp32_from_shorth(vec_xl(0, p - GGML_F16_EPR)) : \
  vec_వెలికితీయి_fp32_from_shortl(vec_xl(0, p))
స్థిర అంతఃవరుస గుర్తులేని అక్షరం ggml_endian_byte(పూర్ణాంకం i) {
       uint16_t tmp_val = 1;
       తిరిగి ((గుర్తులేని అక్షరం *)&tmp_val)[i];
}
#నిర్వచించు GGML_ENDIAN_BYTE(i) ggml_endian_byte(i)
#నిర్వచించు GGML_F16_VEC_STORE(p, r, i)                             \
  ఒకవేళ (i & 0x1)                                                  \
    vec_xst(vec_pack_గమ్యం_short_fp32(r[i - GGML_ENDIAN_BYTE(1)],  \
                                   r[i - GGML_ENDIAN_BYTE(0)]), \
            0, p - GGML_F16_EPR)

//BF16 POWER9
#నిర్వచించు GGML_BF16_అడుగు 16
#నిర్వచించు GGML_BF16_EPR  8

#నిర్వచించు GGML_BF16x8         సదిశ గుర్తులేని పొట్టి
#నిర్వచించు GGML_BF16x8_ZERO    vec_splats((గుర్తులేని పొట్టి)0)
#నిర్వచించు GGML_BF16x8_ఎక్కించు(p) vec_xl(0, (స్థిరరాశి గుర్తులేని పొట్టి *)(p))

#నిర్వచించు GGML_BF16_VEC          GGML_BF16x8
#నిర్వచించు GGML_BF16_VEC_ZERO     GGML_BF16x8_ZERO
#నిర్వచించు GGML_BF16_VEC_ఎక్కించు     GGML_BF16x8_ఎక్కించు
#ఒకవేళ defined(__LITTLE_ENDIAN__)
#నిర్వచించు GGML_BF16_గమ్యం_F32_LO(v) ((సదిశ దశాంశం) vec_mergel(GGML_BF16_VEC_ZERO, (v)))
#నిర్వచించు GGML_BF16_గమ్యం_F32_HI(v) ((సదిశ దశాంశం) vec_mergeh(GGML_BF16_VEC_ZERO, (v)))
#లేకపోతే
#నిర్వచించు GGML_BF16_గమ్యం_F32_LO(v) ((సదిశ దశాంశం) vec_mergel((v), GGML_BF16_VEC_ZERO))
#నిర్వచించు GGML_BF16_గమ్యం_F32_HI(v) ((సదిశ దశాంశం) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#అయితే_ముగింపు
#నిర్వచించు GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_గమ్యం_F32_LO(x), GGML_BF16_గమ్యం_F32_LO(y))
#నిర్వచించు GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_గమ్యం_F32_HI(x), GGML_BF16_గమ్యం_F32_HI(y))

#లేదావేళ defined(__wasm_simd128__)

#నిర్వచించు GGML_SIMD

// F32 WASM

#నిర్వచించు GGML_F32_అడుగు 16
#నిర్వచించు GGML_F32_EPR  4

#నిర్వచించు GGML_F32x4              v128_t
#నిర్వచించు GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#నిర్వచించు GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#నిర్వచించు GGML_F32x4_ఎక్కించు         wasm_v128_ఎక్కించు
#నిర్వచించు GGML_F32x4_STORE        wasm_v128_store
#నిర్వచించు GGML_F32x4_FMA(a, b, c) wasm_f32x4_కూడు(wasm_f32x4_mul(b, c), a)
#నిర్వచించు GGML_F32x4_కూడు          wasm_f32x4_కూడు
#నిర్వచించు GGML_F32x4_MUL          wasm_f32x4_mul
#నిర్వచించు GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;                \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {             \
        x[i] = wasm_f32x4_కూడు(x[i], x[స్థానభ్రంశం+i]);  \
    }                                              \
    స్థానభ్రంశం >>= 1;                                  \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {             \
        x[i] = wasm_f32x4_కూడు(x[i], x[స్థానభ్రంశం+i]);  \
    }                                              \
    స్థానభ్రంశం >>= 1;                                  \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {             \
        x[i] = wasm_f32x4_కూడు(x[i], x[స్థానభ్రంశం+i]);  \
    }                                              \
    res = wasm_f32x4_వెలికితీయి_lane(x[0], 0) +       \
          wasm_f32x4_వెలికితీయి_lane(x[0], 1) +       \
          wasm_f32x4_వెలికితీయి_lane(x[0], 2) +       \
          wasm_f32x4_వెలికితీయి_lane(x[0], 3);        \
}

#నిర్వచించు GGML_F32_VEC        GGML_F32x4
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x4_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x4_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x4_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x4_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x4_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x4_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 WASM

#నిర్వచించు GGML_F16_అడుగు 16
#నిర్వచించు GGML_F16_EPR  4

అంతఃవరుస స్థిర v128_t __wasm_f16x4_load(స్థిరరాశి ggml_fp16_t * p) {
    దశాంశం tmp[4];

    tmp[0] = GGML_CPU_FP16_గమ్యం_FP32(p[0]);
    tmp[1] = GGML_CPU_FP16_గమ్యం_FP32(p[1]);
    tmp[2] = GGML_CPU_FP16_గమ్యం_FP32(p[2]);
    tmp[3] = GGML_CPU_FP16_గమ్యం_FP32(p[3]);

    తిరిగి wasm_v128_ఎక్కించు(tmp);
}

అంతఃవరుస స్థిర శూన్య __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
    దశాంశం tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = GGML_CPU_FP32_గమ్యం_FP16(tmp[0]);
    p[1] = GGML_CPU_FP32_గమ్యం_FP16(tmp[1]);
    p[2] = GGML_CPU_FP32_గమ్యం_FP16(tmp[2]);
    p[3] = GGML_CPU_FP32_గమ్యం_FP16(tmp[3]);
}

#నిర్వచించు GGML_F16x4             v128_t
#నిర్వచించు GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#నిర్వచించు GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#నిర్వచించు GGML_F16x4_ఎక్కించు(x)     __wasm_f16x4_load(x)
#నిర్వచించు GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#నిర్వచించు GGML_F16x4_FMA         GGML_F32x4_FMA
#నిర్వచించు GGML_F16x4_కూడు         wasm_f32x4_కూడు
#నిర్వచించు GGML_F16x4_MUL         wasm_f32x4_mul
#నిర్వచించు GGML_F16x4_REDUCE(res, x)                           \
{                                                           \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F16_ARR >> 1;                         \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                      \
        x[i] = wasm_f32x4_కూడు(x[i], x[స్థానభ్రంశం+i]);           \
    }                                                       \
    స్థానభ్రంశం >>= 1;                                           \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                      \
        x[i] = wasm_f32x4_కూడు(x[i], x[స్థానభ్రంశం+i]);           \
    }                                                       \
    స్థానభ్రంశం >>= 1;                                           \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                      \
        x[i] = wasm_f32x4_కూడు(x[i], x[స్థానభ్రంశం+i]);           \
    }                                                       \
    res = (ggml_float) (wasm_f32x4_వెలికితీయి_lane(x[0], 0) +  \
          wasm_f32x4_వెలికితీయి_lane(x[0], 1) +                \
          wasm_f32x4_వెలికితీయి_lane(x[0], 2) +                \
          wasm_f32x4_వెలికితీయి_lane(x[0], 3));                \
}

#నిర్వచించు GGML_F16_VEC                GGML_F16x4
#నిర్వచించు GGML_F16_VEC_ZERO           GGML_F16x4_ZERO
#నిర్వచించు GGML_F16_VEC_SET1           GGML_F16x4_SET1
#నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     GGML_F16x4_ఎక్కించు(p)
#నిర్వచించు GGML_F16_VEC_STORE(p, r, i) GGML_F16x4_STORE(p, r[i])
#నిర్వచించు GGML_F16_VEC_FMA            GGML_F16x4_FMA
#నిర్వచించు GGML_F16_VEC_కూడు            GGML_F16x4_కూడు
#నిర్వచించు GGML_F16_VEC_MUL            GGML_F16x4_MUL
#నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F16x4_REDUCE

#లేదావేళ defined(__SSE3__)

#నిర్వచించు GGML_SIMD

// F32 SSE

#నిర్వచించు GGML_F32_అడుగు 32
#నిర్వచించు GGML_F32_EPR  4

#నిర్వచించు GGML_F32x4         __m128
#నిర్వచించు GGML_F32x4_ZERO    _mm_setzero_ps()
#నిర్వచించు GGML_F32x4_SET1(x) _mm_set1_ps(x)
#నిర్వచించు GGML_F32x4_ఎక్కించు    _mm_loadu_ps
#నిర్వచించు GGML_F32x4_STORE   _mm_storeu_ps
#ఒకవేళ defined(__FMA__)
    // TODO: Does this work?
    #నిర్వచించు GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#లేకపోతే
    #నిర్వచించు GGML_F32x4_FMA(a, b, c) mm_కూడు_ps(_mm_mul_ps(b, c), a)
#అయితే_ముగింపు
#నిర్వచించు GGML_F32x4_కూడు     mm_కూడు_ps
#నిర్వచించు GGML_F32x4_MUL     _mm_mul_ps
#నిర్వచించు GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;                               \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = mm_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                     \
    }                                                             \
    స్థానభ్రంశం >>= 1;                                                 \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = mm_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                     \
    }                                                             \
    స్థానభ్రంశం >>= 1;                                                 \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = mm_కూడు_ps(x[i], x[స్థానభ్రంశం+i]);                     \
    }                                                             \
    స్థిరరాశి __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#నిర్వచించు GGML_F32_VEC        GGML_F32x4
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x4_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x4_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x4_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x4_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x4_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x4_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 SSE

#నిర్వచించు GGML_F16_అడుగు 32
#నిర్వచించు GGML_F16_EPR  4

స్థిర అంతఃవరుస __m128 __sse_f16x4_load(స్థిరరాశి ggml_fp16_t * x) {
    దశాంశం tmp[4];

    tmp[0] = GGML_CPU_FP16_గమ్యం_FP32(x[0]);
    tmp[1] = GGML_CPU_FP16_గమ్యం_FP32(x[1]);
    tmp[2] = GGML_CPU_FP16_గమ్యం_FP32(x[2]);
    tmp[3] = GGML_CPU_FP16_గమ్యం_FP32(x[3]);

    తిరిగి _mm_loadu_ps(tmp);
}

స్థిర అంతఃవరుస శూన్య __sse_f16x4_store(ggml_fp16_t * x, __m128 y) {
    దశాంశం arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = GGML_CPU_FP32_గమ్యం_FP16(arr[0]);
    x[1] = GGML_CPU_FP32_గమ్యం_FP16(arr[1]);
    x[2] = GGML_CPU_FP32_గమ్యం_FP16(arr[2]);
    x[3] = GGML_CPU_FP32_గమ్యం_FP16(arr[3]);
}

#నిర్వచించు GGML_F32Cx4             __m128
#నిర్వచించు GGML_F32Cx4_ZERO        _mm_setzero_ps()
#నిర్వచించు GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#నిర్వచించు GGML_F32_Cx4_ఎక్కించు(x)     __sse_f16x4_load(x)
#నిర్వచించు GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#నిర్వచించు GGML_F32Cx4_FMA         GGML_F32x4_FMA
#నిర్వచించు GGML_F32_Cx4_కూడు         mm_కూడు_ps
#నిర్వచించు GGML_F32Cx4_MUL         _mm_mul_ps
#నిర్వచించు GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#నిర్వచించు GGML_F16_VEC                 GGML_F32Cx4
#నిర్వచించు GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#నిర్వచించు GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)      GGML_F32_Cx4_ఎక్కించు(p)
#నిర్వచించు GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#నిర్వచించు GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#నిర్వచించు GGML_F16_VEC_కూడు             GGML_F32_Cx4_కూడు
#నిర్వచించు GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#నిర్వచించు GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#లేదావేళ defined(__loongarch_asx)

#నిర్వచించు GGML_SIMD

// F32 LASX
#నిర్వచించు GGML_F32_అడుగు 32
#నిర్వచించు GGML_F32_EPR  8

#నిర్వచించు GGML_F32x8         __m256
#నిర్వచించు GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#నిర్వచించు GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#నిర్వచించు GGML_F32x8_ఎక్కించు(x) (__m256)__lasx_xvld((x), 0)
#నిర్వచించు GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#నిర్వచించు GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#నిర్వచించు GGML_F32x8_కూడు     __lasx_xvfadd_s
#నిర్వచించు GGML_F32x8_MUL     __lasx_xvfmul_s
#నిర్వచించు GGML_F32x8_REDUCE(res, x)                                 \
చేయి {                                                              \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;                               \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[స్థానభ్రంశం+i]);                  \
    }                                                             \
    స్థానభ్రంశం >>= 1;                                                 \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[స్థానభ్రంశం+i]);                  \
    }                                                             \
    స్థానభ్రంశం >>= 1;                                                 \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[స్థానభ్రంశం+i]);                  \
    }                                                             \
    దశాంశం *tmp_p = (దశాంశం *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} అయితే (0)
// TODO: is this optimal ?

#నిర్వచించు GGML_F32_VEC        GGML_F32x8
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x8_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x8_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x8_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x8_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x8_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x8_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 LASX

#నిర్వచించు GGML_F16_అడుగు 32
#నిర్వచించు GGML_F16_EPR  8

// F16 arithmetic is not supported by LASX, so we use F32 instead

#నిర్వచించు GGML_F32Cx8          __m256
#నిర్వచించు GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#నిర్వచించు GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))

స్థిర అంతఃవరుస __m256 __lasx_f32cx8_load(స్థిరరాశి ggml_fp16_t * x) {
    __m256i a;
    జ్ఞాపకం_నకలు(&a, x, పరిమాణం(ggml_fp16_t) * 8);
    a = __lasx_xvpermi_d(a, 0 | (1 << 4));
    తిరిగి __lasx_xvfcvtl_s_h(a);
}

స్థిర అంతఃవరుస శూన్య __lasx_f32cx8_store(ggml_fp16_t * x, __m256 y) {
    __m256i a = __lasx_xvfcvt_h_s(y, y);
    a = __lasx_xvpermi_d(a, 0 | (2 << 2));
    జ్ఞాపకం_నకలు(x, &a, పరిమాణం(ggml_fp16_t) * 8);
}
#నిర్వచించు GGML_F32_Cx8_ఎక్కించు(x)     __lasx_f32cx8_load(x)
#నిర్వచించు GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#నిర్వచించు GGML_F32Cx8_FMA         GGML_F32x8_FMA
#నిర్వచించు GGML_F32_Cx8_కూడు         __lasx_xvfadd_s
#నిర్వచించు GGML_F32Cx8_MUL         __lasx_xvfmul_s
#నిర్వచించు GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#నిర్వచించు GGML_F16_VEC                GGML_F32Cx8
#నిర్వచించు GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#నిర్వచించు GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     GGML_F32_Cx8_ఎక్కించు(p)
#నిర్వచించు GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#నిర్వచించు GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#నిర్వచించు GGML_F16_VEC_కూడు            GGML_F32_Cx8_కూడు
#నిర్వచించు GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#లేదావేళ defined(__loongarch_sx)

#నిర్వచించు GGML_SIMD

// F32 LSX

#నిర్వచించు GGML_F32_అడుగు 32
#నిర్వచించు GGML_F32_EPR  4

#నిర్వచించు GGML_F32x4         __m128
#నిర్వచించు GGML_F32x4_ZERO    (__m128)__lsx_vldi(0)
#నిర్వచించు GGML_F32x4_SET1(x) (__m128)__lsx_vreplfr2vr_s((x))
#నిర్వచించు GGML_F32x4_ఎక్కించు(x) (__m128)__lsx_vld((x), 0)
#నిర్వచించు GGML_F32x4_STORE(x, y)   __lsx_vst(y, x, 0)
#నిర్వచించు GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#నిర్వచించు GGML_F32x4_కూడు     __lsx_vfadd_s
#నిర్వచించు GGML_F32x4_MUL     __lsx_vfmul_s

#నిర్వచించు GGML_F32x4_REDUCE(res, x)                               \
{                                                               \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;                             \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[స్థానభ్రంశం+i]);                \
    }                                                           \
    స్థానభ్రంశం >>= 1;                                               \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[స్థానభ్రంశం+i]);                \
    }                                                           \
    స్థానభ్రంశం >>= 1;                                               \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[స్థానభ్రంశం+i]);                \
    }                                                           \
    __m128i t0 = __lsx_vpickev_w((__m128i)x[0], (__m128i)x[0]); \
    __m128i t1 = __lsx_vpickod_w((__m128i)x[0], (__m128i)x[0]); \
    __m128 t2 = __lsx_vfadd_s((__m128)t0, (__m128)t1);          \
    __m128i t3 = __lsx_vpickev_w((__m128i)t2, (__m128i)t2);     \
    __m128i t4 = __lsx_vpickod_w((__m128i)t2, (__m128i)t2);     \
    __m128 t5 = __lsx_vfadd_s((__m128)t3, (__m128)t4);          \
    res = (ggml_float) ((v4f32)t5)[0];                          \
}

#నిర్వచించు GGML_F32_VEC        GGML_F32x4
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x4_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x4_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x4_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x4_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x4_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x4_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 LSX

#నిర్వచించు GGML_F16_అడుగు 32
#నిర్వచించు GGML_F16_EPR  4

స్థిర అంతఃవరుస __m128 __lsx_f16x4_load(స్థిరరాశి ggml_fp16_t * x) {
    తిరిగి __lsx_vfcvtl_s_h(__lsx_vld((స్థిరరాశి శూన్య *)x, 0));
}

స్థిర అంతఃవరుస శూన్య __lsx_f16x4_store(ggml_fp16_t * x, __m128 y) {
    __m128i a = __lsx_vfcvt_h_s(y, y);
    జ్ఞాపకం_నకలు(x, &a, పరిమాణం(ggml_fp16_t) * 4);
}

#నిర్వచించు GGML_F32Cx4             __m128
#నిర్వచించు GGML_F32Cx4_ZERO        (__m128)__lsx_vldi(0)
#నిర్వచించు GGML_F32Cx4_SET1(x)     (__m128)__lsx_vreplfr2vr_s((x))
#నిర్వచించు GGML_F32_Cx4_ఎక్కించు(x)     (__m128)__lsx_f16x4_load(x)
#నిర్వచించు GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#నిర్వచించు GGML_F32Cx4_FMA         GGML_F32x4_FMA
#నిర్వచించు GGML_F32_Cx4_కూడు         __lsx_vfadd_s
#నిర్వచించు GGML_F32Cx4_MUL         __lsx_vfmul_s
#నిర్వచించు GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#నిర్వచించు GGML_F16_VEC                 GGML_F32Cx4
#నిర్వచించు GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#నిర్వచించు GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)      GGML_F32_Cx4_ఎక్కించు(p)
#నిర్వచించు GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#నిర్వచించు GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#నిర్వచించు GGML_F16_VEC_కూడు             GGML_F32_Cx4_కూడు
#నిర్వచించు GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#నిర్వచించు GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#లేదావేళ defined(__VXE__) || defined(__VXE2__)

#నిర్వచించు GGML_SIMD

// F32 s390x

#నిర్వచించు GGML_F32_అడుగు 32
#నిర్వచించు GGML_F32_EPR  4

#నిర్వచించు GGML_F32x4              float32x4_t
#నిర్వచించు GGML_F32x4_ZERO         vec_splats(0.0f)
#నిర్వచించు GGML_F32x4_SET1         vec_splats
#నిర్వచించు GGML_F32x4_ఎక్కించు(p)      vec_xl(0, p)
#నిర్వచించు GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#నిర్వచించు GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#నిర్వచించు GGML_F32x4_కూడు          vec_కూడు
#నిర్వచించు GGML_F32x4_MUL          vec_mul
#నిర్వచించు GGML_F32x4_REDUCE(res, x)                   \
{                                                   \
    పూర్ణాంకం స్థానభ్రంశం = GGML_F32_ARR >> 1;                 \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {              \
        x[i] = vec_కూడు(x[i], x[స్థానభ్రంశం + i]);        \
    }                                               \
    స్థానభ్రంశం >>= 1;                                   \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {              \
        x[i] = vec_కూడు(x[i], x[స్థానభ్రంశం + i]);        \
    }                                               \
    స్థానభ్రంశం >>= 1;                                   \
    కొరకు (పూర్ణాంకం i = 0; i < స్థానభ్రంశం; ++i) {              \
        x[i] = vec_కూడు(x[i], x[స్థానభ్రంశం + i]);        \
    }                                               \
    float32x4_t tmp = x[0] + vec_reve(x[0]);        \
    res = tmp[0] + tmp[1];                          \
}
#నిర్వచించు GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3) \
{                                                \
    float32x4_t v = vec_కూడు(vec_కూడు(s0, s1),     \
                            vec_కూడు(s2, s3));    \
    v = vec_కూడు(v, vec_sld(v, v, 8));            \
    v = vec_కూడు(v, vec_sld(v, v, 4));            \
    res += (ggml_float)vec_వెలికితీయి(v, 0);        \
}

#నిర్వచించు GGML_F32_VEC        GGML_F32x4
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x4_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x4_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x4_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x4_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x4_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x4_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 s390x
#నిర్వచించు GGML_F16_అడుగు GGML_F32_అడుగు
#నిర్వచించు GGML_F16_EPR  GGML_F32_EPR

స్థిర అంతఃవరుస uint32x4_t __lzs_f32cx4_to_f16(float32x4_t v_f) {
    float32x4_t v_పునాది = vec_mul(vec_mul(vec_abs(v_f), vec_splats(0x1.0p+112f)), vec_splats(0x1.0p-110f));

    స్థిరరాశి uint32x4_t v_w      = (uint32x4_t)v_f;
    స్థిరరాశి uint32x4_t v_shl1_w = vec_కూడు(v_w, v_w);
    స్థిరరాశి uint32x4_t v_sign   = vec_and(v_w, vec_splats(UINT32_C(0x80000000)));
    స్థిరరాశి uint32x4_t v_bias   = vec_max(vec_and(v_shl1_w, vec_splats(UINT32_C(0xFF000000))), vec_splats(UINT32_C(0x71000000)));

    v_పునాది = vec_కూడు((float32x4_t)vec_కూడు(vec_sr(v_bias, 1), vec_splats(UINT32_C(0x07800000))), v_పునాది);

    స్థిరరాశి uint32x4_t v_బిట్లు    = (uint32x4_t)v_పునాది;
    స్థిరరాశి uint32x4_t v_nonsign = vec_కూడు(vec_and(vec_sr(v_బిట్లు, 13), vec_splats(UINT32_C(0x00007C00))),
                                         vec_and(v_బిట్లు, vec_splats(UINT32_C(0x00000FFF))));
    స్థిరరాశి uint32x4_t v_is_nan  = (uint32x4_t)vec_cmpgt(v_shl1_w, vec_splats(UINT32_C(0xFF000000)));

    తిరిగి vec_or(vec_sr(v_sign, 16), vec_sel(v_nonsign, vec_splats(UINT32_C(0x7E00)), v_is_nan));
}

స్థిర అంతఃవరుస float32x4_t __lzs_f16cx4_load(స్థిరరాశి ggml_fp16_t * x) {
    దశాంశం tmp[4];

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_గమ్యం_FP32(x[i]);
    }

    // note: keep type-cast here to prevent compiler bugs
    // see: https://github.com/ggml-org/llama.cpp/issues/12846
    తిరిగి vec_xl(0, (స్థిరరాశి దశాంశం *)(tmp));
}

స్థిర అంతఃవరుస శూన్య __lzs_f16cx4_store(ggml_fp16_t * x, float32x4_t v_y) {
    స్థిరరాశి uint32x4_t v_h = __lzs_f32cx4_to_f16(v_y);
    స్థిరరాశి uint64_t   tmp = ((uint64x2_t)vec_pack(v_h, v_h))[0];
    జ్ఞాపకం_నకలు(x, &tmp, పరిమాణం(tmp));
}

#నిర్వచించు GGML_F16_VEC                GGML_F32x4
#నిర్వచించు GGML_F16_VEC_ZERO           GGML_F32x4_ZERO
#నిర్వచించు GGML_F16_VEC_SET1           GGML_F32x4_SET1
#నిర్వచించు GGML_F16_VEC_ఎక్కించు(p, i)     __lzs_f16cx4_load(p)
#నిర్వచించు GGML_F16_VEC_STORE(p, r, i) __lzs_f16cx4_store(p, r[i])
#నిర్వచించు GGML_F16_VEC_FMA            GGML_F32x4_FMA
#నిర్వచించు GGML_F16_VEC_కూడు            GGML_F32x4_కూడు
#నిర్వచించు GGML_F16_VEC_MUL            GGML_F32x4_MUL
#నిర్వచించు GGML_F16_VEC_REDUCE         GGML_F32x4_REDUCE

// BF16 s390x
#నిర్వచించు GGML_BF16_అడుగు 16
#నిర్వచించు GGML_BF16_EPR  8

#నిర్వచించు GGML_BF16x8         __vector గుర్తులేని పొట్టి
#నిర్వచించు GGML_BF16x8_ZERO    vec_splats((గుర్తులేని పొట్టి)0)
#నిర్వచించు GGML_BF16x8_ఎక్కించు(p) vec_xl(0, (స్థిరరాశి గుర్తులేని పొట్టి *)(p))

#నిర్వచించు GGML_BF16_VEC      GGML_BF16x8
#నిర్వచించు GGML_BF16_VEC_ZERO GGML_BF16x8_ZERO
#నిర్వచించు GGML_BF16_VEC_ఎక్కించు GGML_BF16x8_ఎక్కించు
#నిర్వచించు GGML_BF16_గమ్యం_F32_LO(v) ((float32x4_t) vec_mergel((v), GGML_BF16_VEC_ZERO))
#నిర్వచించు GGML_BF16_గమ్యం_F32_HI(v) ((float32x4_t) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#నిర్వచించు GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_గమ్యం_F32_LO(x), GGML_BF16_గమ్యం_F32_LO(y))
#నిర్వచించు GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_గమ్యం_F32_HI(x), GGML_BF16_గమ్యం_F32_HI(y))

#లేదావేళ defined(__riscv_v_intrinsic)

// compatible with vlen >= 128

#నిర్వచించు GGML_SIMD

// F32

#నిర్వచించు GGML_F32_అడుగు 16
#నిర్వచించు GGML_F32_EPR  4

#నిర్వచించు GGML_F32x4              vfloat32m1_t
#నిర్వచించు GGML_F32x4_ZERO         __riscv_vfmv_v_f_f32m1(0.0f, GGML_F32_EPR)
#నిర్వచించు GGML_F32x4_SET1(x)      __riscv_vfmv_v_f_f32m1(x, GGML_F32_EPR)
#నిర్వచించు GGML_F32x4_ఎక్కించు(x)      __riscv_vle32_v_f32m1(x, GGML_F32_EPR)
#నిర్వచించు GGML_F32x4_STORE(b, v)  __riscv_vse32_v_f32m1(b, v, GGML_F32_EPR)
#నిర్వచించు GGML_F32x4_FMA(a, b, c) __riscv_vfmacc_vv_f32m1(a, b, c, GGML_F32_EPR)
#నిర్వచించు GGML_F32x4_కూడు(a, b)    __riscv_vfadd_vv_f32m1(a, b, GGML_F32_EPR)
#నిర్వచించు GGML_F32x4_MUL(a, b)    __riscv_vfmul_vv_f32m1(a, b, GGML_F32_EPR)

#నిర్వచించు GGML_F32_VEC        GGML_F32x4
#నిర్వచించు GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#నిర్వచించు GGML_F32_VEC_SET1   GGML_F32x4_SET1
#నిర్వచించు GGML_F32_VEC_ఎక్కించు   GGML_F32x4_ఎక్కించు
#నిర్వచించు GGML_F32_VEC_STORE  GGML_F32x4_STORE
#నిర్వచించు GGML_F32_VEC_FMA    GGML_F32x4_FMA
#నిర్వచించు GGML_F32_VEC_కూడు    GGML_F32x4_కూడు
#నిర్వచించు GGML_F32_VEC_MUL    GGML_F32x4_MUL
#నిర్వచించు GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

#అయితే_ముగింపు

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#నిర్వచితమైతే GGML_SIMD
#నిర్వచించు GGML_F32_ARR (GGML_F32_అడుగు/GGML_F32_EPR)
#నిర్వచించు GGML_F16_ARR (GGML_F16_అడుగు/GGML_F16_EPR)
#అయితే_ముగింపు

// GGML_F16_DOT_*
// like GGML_F16_* but for dot products which need F32 accumulation on AVX512-FP16

#ఒకవేళ defined(__AVX512FP16__)

#నిర్వచించు GGML_F16_DOT_అడుగు           GGML_F32_అడుగు
#నిర్వచించు GGML_F16_DOT_EPR            GGML_F32_EPR
#నిర్వచించు GGML_F16_DOT_ARR            GGML_F32_ARR
#నిర్వచించు GGML_F16_DOT_VEC            GGML_F32x16
#నిర్వచించు GGML_F16_DOT_VEC_ZERO       GGML_F32x16_ZERO
#నిర్వచించు GGML_F16_DOT_VEC_ఎక్కించు(p, i) _mm512_cvtph_ps(_mm256_loadu_si256((స్థిరరాశి __m256i *)(p)))
#నిర్వచించు GGML_F16_DOT_VEC_FMA        GGML_F32x16_FMA
#నిర్వచించు GGML_F16_DOT_VEC_REDUCE     GGML_F32x16_REDUCE

#లేకపోతే

#నిర్వచించు GGML_F16_DOT_అడుగు           GGML_F16_అడుగు
#నిర్వచించు GGML_F16_DOT_EPR            GGML_F16_EPR
#నిర్వచించు GGML_F16_DOT_ARR            GGML_F16_ARR
#నిర్వచించు GGML_F16_DOT_VEC            GGML_F16_VEC
#నిర్వచించు GGML_F16_DOT_VEC_ZERO       GGML_F16_VEC_ZERO
#నిర్వచించు GGML_F16_DOT_VEC_ఎక్కించు       GGML_F16_VEC_ఎక్కించు
#నిర్వచించు GGML_F16_DOT_VEC_FMA        GGML_F16_VEC_FMA
#నిర్వచించు GGML_F16_DOT_VEC_REDUCE     GGML_F16_VEC_REDUCE

#అయితే_ముగింపు // defined(__AVX512FP16__)

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
