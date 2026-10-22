#هدايت once

#ضمن "ggml-cpu-impl.h"

#جيڪڏهن_وصف_ٿيل __ARM_FEATURE_SVE
#ضمن <arm_sve.h>
#جيڪڏهن_پورو // __ARM_FEATURE_SVE

#إذا defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ضمن <arm_neon.h>
#جيڪڏهن_پورو

#إذا defined(__riscv_v_intrinsic)
#ضمن <riscv_vector.h>
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل __cplusplus
خارجي "C" {
#جيڪڏهن_پورو

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
    #وصف_ڪريو GGML_CPU_COMPUTE_FP16_منزل_FP32(x) neon_compute_fp16_منزل_fp32(x)
    #وصف_ڪريو GGML_CPU_COMPUTE_FP32_منزل_FP16(x) neon_compute_fp32_منزل_fp16(x)

    #وصف_ڪريو GGML_CPU_FP16_منزل_FP32(x) GGML_CPU_COMPUTE_FP16_منزل_FP32(x)
    #وصف_ڪريو GGML_CPU_FP32_منزل_FP16(x) GGML_CPU_COMPUTE_FP32_منزل_FP16(x)

    ساكن مضمن عائم neon_compute_fp16_منزل_fp32(ggml_fp16_t h) {
        __fp16 tmp;
        يادگيري_نقل(&tmp, &h, حجم(ggml_fp16_t));
        أرجع (عائم)tmp;
    }

    ساكن مضمن ggml_fp16_t neon_compute_fp32_منزل_fp16(عائم f) {
        ggml_fp16_t res;
        __fp16 tmp = f;
        يادگيري_نقل(&res, &tmp, حجم(ggml_fp16_t));
        أرجع res;
    }
#وإلاإذا defined(__F16C__)
    #جيڪڏهن_وصف_ٿيل _MSC_VER
        #وصف_ڪريو GGML_CPU_COMPUTE_FP16_منزل_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
        #وصف_ڪريو GGML_CPU_COMPUTE_FP32_منزل_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(mm_سيٽ_ss(x), 0), 0)
    #وإلا
        #وصف_ڪريو GGML_CPU_COMPUTE_FP16_منزل_FP32(x) _cvtsh_ss(x)
        #وصف_ڪريو GGML_CPU_COMPUTE_FP32_منزل_FP16(x) _cvtss_sh(x, 0)
    #جيڪڏهن_پورو
#وإلاإذا defined(__POWER9_VECTOR__)
    #وصف_ڪريو GGML_CPU_COMPUTE_FP16_منزل_FP32(x) power_compute_fp16_منزل_fp32(x)
    #وصف_ڪريو GGML_CPU_COMPUTE_FP32_منزل_FP16(x) power_compute_fp32_منزل_fp16(x)
    /* the inline asm below is about 12% faster than the lookup method */
    #وصف_ڪريو GGML_CPU_FP16_منزل_FP32(x) GGML_CPU_COMPUTE_FP16_منزل_FP32(x)
    #وصف_ڪريو GGML_CPU_FP32_منزل_FP16(x) GGML_CPU_COMPUTE_FP32_منزل_FP16(x)

    ساكن مضمن عائم power_compute_fp16_منزل_fp32(ggml_fp16_t h) {
        عائم f;
        مزدوج d;
        __asm__(
            "mtfprd %0,%2\n"
            "xscvhpdp %0,%0\n"
            "frsp %1,%0\n" :
            /*عارضي*/ "=d"(d),
            /*ٻاهريون*/  "=f"(f):
            /*اندريون*/   "r"(h));
        أرجع f;
    }

    ساكن مضمن ggml_fp16_t power_compute_fp32_منزل_fp16(عائم f) {
        مزدوج d;
        ggml_fp16_t r;
        __asm__( /* xscvdphp can work on double or single precision */
            "xscvdphp %0,%2\n"
            "mffprd %1,%0\n" :
            /*عارضي*/ "=d"(d),
            /*ٻاهريون*/  "=r"(r):
            /*اندريون*/   "f"(f));
        أرجع r;
    }
#وإلاإذا defined(__riscv) && defined(__riscv_zfhmin)
    ساكن مضمن عائم riscv_compute_fp16_منزل_fp32(ggml_fp16_t h) {
        _Float16 hf;
        يادگيري_نقل(&hf, &h, حجم(ggml_fp16_t));
        أرجع hf;
    }

    ساكن مضمن ggml_fp16_t riscv_compute_fp32_منزل_fp16(عائم f) {
        ggml_fp16_t res;
        _Float16 hf = (_Float16)f;
        يادگيري_نقل(&res, &hf, حجم(ggml_fp16_t));
        أرجع res;
    }

    #وصف_ڪريو GGML_CPU_COMPUTE_FP16_منزل_FP32(x) riscv_compute_fp16_منزل_fp32(x)
    #وصف_ڪريو GGML_CPU_COMPUTE_FP32_منزل_FP16(x) riscv_compute_fp32_منزل_fp16(x)
    #وصف_ڪريو GGML_CPU_FP16_منزل_FP32(x) GGML_CPU_COMPUTE_FP16_منزل_FP32(x)
    #وصف_ڪريو GGML_CPU_FP32_منزل_FP16(x) GGML_CPU_COMPUTE_FP32_منزل_FP16(x)
#جيڪڏهن_پورو

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
#وصف_ڪريو GGML_CPU_E8_M0_منزل_FP32_HALF(x) ggml_جدول_f32_e8m0_half[(uint8_t)(x)]
#وإلا
#وصف_ڪريو GGML_CPU_E8_M0_منزل_FP32_HALF(x) GGML_E8_M0_منزل_FP32_HALF(x)
#جيڪڏهن_پورو

// Use lookup table for UE4M3 on x86 and ARM (faster than bit manipulation)
#إذا defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__ARM_NEON)
#وصف_ڪريو GGML_CPU_UE4_M3_منزل_FP32(x) ggml_جدول_f32_ue4m3[(uint8_t)(x)]
#وإلا
#وصف_ڪريو GGML_CPU_UE4_M3_منزل_FP32(x) ggml_ue4m3_منزل_fp32(x)
#جيڪڏهن_پورو

// Pre AVX512 CPUs (AVX2 with F16C, such as i5-8th gen or i7-4790)
// do not produce codegen as checks below don't test for __F16C__
// but this check does.
#إذا defined(__F16C__)
// force FP16 -> FP32 conversion to use LUT instead
// see: https://github.com/ggml-org/llama.cpp/issues/30342
//#define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
#وصف_ڪريو GGML_CPU_FP32_منزل_FP16(x) GGML_CPU_COMPUTE_FP32_منزل_FP16(x)
#جيڪڏهن_پورو

// a branch that defines a GGML_CPU_COMPUTE_* macro must also select it, or the fallback takes over silently
#إذا defined(GGML_CPU_COMPUTE_FP16_منزل_FP32) && !defined(GGML_CPU_FP16_منزل_FP32) && !defined(__F16C__) // see: https://github.com/ggml-org/llama.cpp/issues/30342
#غلطي "a branch defined GGML_CPU_COMPUTE_FP16_TO_FP32 without selecting it as GGML_CPU_FP16_TO_FP32"
#جيڪڏهن_پورو
#إذا defined(GGML_CPU_COMPUTE_FP32_منزل_FP16) && !defined(GGML_CPU_FP32_منزل_FP16)
#غلطي "a branch defined GGML_CPU_COMPUTE_FP32_TO_FP16 without selecting it as GGML_CPU_FP32_TO_FP16"
#جيڪڏهن_پورو

// fall back to the lookup table / the portable software conversion for whichever direction no branch selected
#إذا !defined(GGML_CPU_FP16_منزل_FP32)
مضمن ساكن عائم ggml_lookup_fp16_منزل_fp32(ggml_fp16_t f) {
    uint16_t s;
    يادگيري_نقل(&s, &f, حجم(uint16_t));
    أرجع ggml_جدول_f32_f16[s];
}

#وصف_ڪريو GGML_CPU_FP16_منزل_FP32(x) ggml_lookup_fp16_منزل_fp32(x)
#جيڪڏهن_پورو

#إذا !defined(GGML_CPU_FP32_منزل_FP16)
#وصف_ڪريو GGML_CPU_FP32_منزل_FP16(x) GGML_COMPUTE_FP32_منزل_FP16(x)
#جيڪڏهن_پورو


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

#وصف_ڪريو GGML_SIMD

// F32 SVE
#وصف_ڪريو GGML_F32_EPR 8
#وصف_ڪريو DEFAULT_PG svptrue_b32()

#وصف_ڪريو GGML_F32xt                        svfloat32_t
#وصف_ڪريو GGML_F32xt_ZERO                   svdup_n_f32(0.0f)
#وصف_ڪريو GGML_F32xt_SET1(x)                svdup_n_f32(x)
#وصف_ڪريو GGML_F32xt_ڀريو_IMPL(pg, a)       svld1_f32(pg, a)
#وصف_ڪريو GGML_F32xt_ڀريو(a)                GGML_F32xt_ڀريو_IMPL(DEFAULT_PG, a)
#وصف_ڪريو GGML_F32xt_STORE_IMPL(pg, a, b)   svst1_f32(pg, a, b)
#وصف_ڪريو GGML_F32xt_STORE(a, b)            GGML_F32xt_STORE_IMPL(DEFAULT_PG, a, b)
#وصف_ڪريو GGML_F32xt_FMA_IMPL(pg, a, b, c)  svmad_f32_m(pg, b, c, a)
#وصف_ڪريو GGML_F32xt_FMA(a, b, c)           GGML_F32xt_FMA_IMPL(DEFAULT_PG, a, b, c)
#وصف_ڪريو GGML_F32xt_جوڙيو_IMPL(pg, a, b)     svadd_f32_m(pg, a, b)
#وصف_ڪريو GGML_F32xt_جوڙيو(a, b)              GGML_F32xt_جوڙيو_IMPL(DEFAULT_PG, a, b)
#وصف_ڪريو GGML_F32xt_MUL_IMPL(pg, a, b)     svmul_f32_m(pg, a, b)
#وصف_ڪريو GGML_F32xt_MUL(a, b)              GGML_F32xt_MUL_IMPL(DEFAULT_PG, a, b)
#وصف_ڪريو GGML_F32xt_REDUCE_ONE_IMPL(pg, a) svaddv(pg, a)
#وصف_ڪريو GGML_F32xt_REDUCE_ONE(a)          GGML_F32xt_REDUCE_ONE_IMPL(DEFAULT_PG, a)
#وصف_ڪريو GGML_F32xt_REDUCE_IMPL(pg, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
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
#وصف_ڪريو GGML_F32xt_REDUCE(res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)  \
        GGML_F32xt_REDUCE_IMPL(DEFAULT_PG, res, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8)

#وصف_ڪريو GGML_F32_VEC        GGML_F32xt
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32xt_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32xt_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32xt_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32xt_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32xt_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32xt_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32xt_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32xt_REDUCE

// F16 SVE
#وصف_ڪريو DEFAULT_PG32    svptrue_b32()
#وصف_ڪريو DEFAULT_PG16    svptrue_b16()

#وصف_ڪريو GGML_F32Cxt                         svfloat16_t
#وصف_ڪريو GGML_F32Cxt_ZERO                    svdup_n_f16(0.0f)
#وصف_ڪريو GGML_F32Cxt_SET1(x)                 svdup_n_f16(x)
#وصف_ڪريو GGML_F32_Cxt_ڀريو(p)                 svld1_f16(DEFAULT_PG16, (ثابت __fp16 *)(p))
#وصف_ڪريو GGML_F32Cxt_STORE(dst_ptr, src_vec) svst1_f16(DEFAULT_PG16, (__fp16 *)(dst_ptr), (src_vec))

#وصف_ڪريو GGML_F32Cxt_FMA_IMPL(pg, a, b, c)   svmad_f16_x(pg, b, c, a)
#وصف_ڪريو GGML_F32Cxt_FMA(a, b, c)            GGML_F32Cxt_FMA_IMPL(DEFAULT_PG16, a, b, c)
#وصف_ڪريو GGML_F32_Cxt_جوڙيو_IMPL(pg, a, b)      svadd_f16_x(pg, a, b)
#وصف_ڪريو GGML_F32_Cxt_جوڙيو(a, b)               GGML_F32_Cxt_جوڙيو_IMPL(DEFAULT_PG16, a, b)
#وصف_ڪريو GGML_F32Cxt_MUL_IMPL(pg, a, b)      svmul_f16_x(pg, a, b)
#وصف_ڪريو GGML_F32Cxt_MUL(a, b)               GGML_F32Cxt_MUL_IMPL(DEFAULT_PG16, a, b)
#وصف_ڪريو GGML_F32Cxt_REDUCE                  GGML_F16xt_REDUCE_MIXED

#وصف_ڪريو GGML_F16x_VEC                GGML_F32Cxt
#وصف_ڪريو GGML_F16x_VEC_ZERO           GGML_F32Cxt_ZERO
#وصف_ڪريو GGML_F16x_VEC_SET1           GGML_F32Cxt_SET1
#وصف_ڪريو GGML_F16x_VEC_ڀريو(p, i)     GGML_F32_Cxt_ڀريو(p)
#وصف_ڪريو GGML_F16x_VEC_STORE(p, r, i) GGML_F32Cxt_STORE((__fp16 *)(p), r)
#وصف_ڪريو GGML_F16x_VEC_FMA            GGML_F32Cxt_FMA
#وصف_ڪريو GGML_F16x_VEC_جوڙيو            GGML_F32_Cxt_جوڙيو
#وصف_ڪريو GGML_F16x_VEC_MUL            GGML_F32Cxt_MUL
#وصف_ڪريو GGML_F16x_VEC_REDUCE         GGML_F32Cxt_REDUCE

#وصف_ڪريو GGML_F16xt_REDUCE_ONE_IMPL(pg, a) svaddv_f16(pg, a)
#وصف_ڪريو GGML_F16xt_REDUCE_ONE(a)          GGML_F16xt_REDUCE_ONE_IMPL(DEFAULT_PG16, a)

#وصف_ڪريو GGML_F16xt_REDUCE_MIXED_IMPL(pg16, res, sum1, sum2, sum3, sum4)  \
{                                                      \
    sum1 = svadd_f16_x(pg16, sum1, sum2);              \
    sum3 = svadd_f16_x(pg16, sum3, sum4);              \
    sum1 = svadd_f16_x(pg16, sum1, sum3);              \
    __fp16 جوڙ_f16 = svaddv_f16(pg16, sum1);           \
    (res) = (ggml_float) جوڙ_f16;                      \
}
#وصف_ڪريو GGML_F16xt_REDUCE_MIXED(res, sum1, sum2, sum3, sum4)  \
        GGML_F16xt_REDUCE_MIXED_IMPL(DEFAULT_PG16, res, sum1, sum2, sum3, sum4)

// F16 NEON

#إذا defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #وصف_ڪريو GGML_F16_قدم 32
    #وصف_ڪريو GGML_F16_EPR  8

    #وصف_ڪريو GGML_F16x8              float16x8_t
    #وصف_ڪريو GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #وصف_ڪريو GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #وصف_ڪريو GGML_F16x8_ڀريو(x)      vld1q_f16((ثابت __fp16 *)(x))
    #وصف_ڪريو GGML_F16x8_STORE        vst1q_f16
    #وصف_ڪريو GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #وصف_ڪريو GGML_F16x8_جوڙيو          vaddq_f16
    #وصف_ڪريو GGML_F16x8_MUL          vmulq_f16
    #وصف_ڪريو GGML_F16x8_REDUCE(res, x)                               \
    افعل {                                                            \
        صحيح وٿي = GGML_F16_ARR >> 1;                             \
        لكل (صحيح i = 0; i < وٿي; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[وٿي+i]);              \
        }                                                           \
        وٿي >>= 1;                                               \
        لكل (صحيح i = 0; i < وٿي; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[وٿي+i]);              \
        }                                                           \
        وٿي >>= 1;                                               \
        لكل (صحيح i = 0; i < وٿي; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[وٿي+i]);              \
        }                                                           \
        ثابت float32x4_t t0 = vcvt_f32_f16(vget_هيٺيون_f16 ((x)[0])); \
        ثابت float32x4_t t1 = vcvt_f32_f16(vget_مٿاهون_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } طالما (0)

    #وصف_ڪريو GGML_F16_VEC                GGML_F16x8
    #وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #وصف_ڪريو GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     GGML_F16x8_ڀريو(p)
    #وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #وصف_ڪريو GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F16x8_جوڙيو
    #وصف_ڪريو GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#وإلا
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #وصف_ڪريو GGML_F16_قدم 16
    #وصف_ڪريو GGML_F16_EPR  4

    #وصف_ڪريو GGML_F32Cx4              float32x4_t
    #وصف_ڪريو GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #وصف_ڪريو GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #وصف_ڪريو GGML_F32_Cx4_ڀريو(x)      vcvt_f32_f16(vld1_f16((ثابت __fp16 *)(x)))
    #وصف_ڪريو GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #وصف_ڪريو GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #وصف_ڪريو GGML_F32_Cx4_جوڙيو          vaddq_f32
    #وصف_ڪريو GGML_F32Cx4_MUL          vmulq_f32
    #وصف_ڪريو GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #وصف_ڪريو GGML_F16_VEC                GGML_F32Cx4
    #وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #وصف_ڪريو GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     GGML_F32_Cx4_ڀريو(p)
    #وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #وصف_ڪريو GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F32_Cx4_جوڙيو
    #وصف_ڪريو GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#جيڪڏهن_پورو

#وإلاإذا defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && defined(__ARM_FP16_FORMAT_IEEE)

#وصف_ڪريو GGML_SIMD

// F32 NEON

#وصف_ڪريو GGML_F32_قدم 16
#وصف_ڪريو GGML_F32_EPR  4

#وصف_ڪريو GGML_F32x4              float32x4_t
#وصف_ڪريو GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#وصف_ڪريو GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#وصف_ڪريو GGML_F32x4_ڀريو         vld1q_f32
#وصف_ڪريو GGML_F32x4_STORE        vst1q_f32
#وصف_ڪريو GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#وصف_ڪريو GGML_F32x4_جوڙيو          vaddq_f32
#وصف_ڪريو GGML_F32x4_MUL          vmulq_f32
#وصف_ڪريو GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#وصف_ڪريو GGML_F32x4_REDUCE(res, x)                       \
{                                                       \
    صحيح وٿي = GGML_F32_ARR >> 1;                     \
    لكل (صحيح i = 0; i < وٿي; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[وٿي+i]);      \
    }                                                   \
    وٿي >>= 1;                                       \
    لكل (صحيح i = 0; i < وٿي; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[وٿي+i]);      \
    }                                                   \
    وٿي >>= 1;                                       \
    لكل (صحيح i = 0; i < وٿي; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[وٿي+i]);      \
    }                                                   \
    (res) = (ggml_float) GGML_F32x4_REDUCE_ONE((x)[0]); \
}

#وصف_ڪريو GGML_F32_VEC        GGML_F32x4
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x4_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x4_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x4_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x4_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x4_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x4_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#إذا defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #وصف_ڪريو GGML_F16_قدم 32
    #وصف_ڪريو GGML_F16_EPR  8

    #وصف_ڪريو GGML_F16x8              float16x8_t
    #وصف_ڪريو GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #وصف_ڪريو GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #وصف_ڪريو GGML_F16x8_ڀريو(x)      vld1q_f16((ثابت __fp16 *)(x))
    #وصف_ڪريو GGML_F16x8_STORE        vst1q_f16
    #وصف_ڪريو GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #وصف_ڪريو GGML_F16x8_جوڙيو          vaddq_f16
    #وصف_ڪريو GGML_F16x8_MUL          vmulq_f16
    #وصف_ڪريو GGML_F16x8_REDUCE(res, x)                               \
    افعل {                                                            \
        صحيح وٿي = GGML_F16_ARR >> 1;                             \
        لكل (صحيح i = 0; i < وٿي; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[وٿي+i]);              \
        }                                                           \
        وٿي >>= 1;                                               \
        لكل (صحيح i = 0; i < وٿي; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[وٿي+i]);              \
        }                                                           \
        وٿي >>= 1;                                               \
        لكل (صحيح i = 0; i < وٿي; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[وٿي+i]);              \
        }                                                           \
        ثابت float32x4_t t0 = vcvt_f32_f16(vget_هيٺيون_f16 ((x)[0])); \
        ثابت float32x4_t t1 = vcvt_f32_f16(vget_مٿاهون_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } طالما (0)

    #وصف_ڪريو GGML_F16_VEC                GGML_F16x8
    #وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #وصف_ڪريو GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     GGML_F16x8_ڀريو(p)
    #وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #وصف_ڪريو GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F16x8_جوڙيو
    #وصف_ڪريو GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#وإلا
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #وصف_ڪريو GGML_F16_قدم 16
    #وصف_ڪريو GGML_F16_EPR  4

    #وصف_ڪريو GGML_F32Cx4              float32x4_t
    #وصف_ڪريو GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #وصف_ڪريو GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #وصف_ڪريو GGML_F32_Cx4_ڀريو(x)      vcvt_f32_f16(vld1_f16((ثابت __fp16 *)(x)))
    #وصف_ڪريو GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #وصف_ڪريو GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #وصف_ڪريو GGML_F32_Cx4_جوڙيو          vaddq_f32
    #وصف_ڪريو GGML_F32Cx4_MUL          vmulq_f32
    #وصف_ڪريو GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #وصف_ڪريو GGML_F16_VEC                GGML_F32Cx4
    #وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #وصف_ڪريو GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     GGML_F32_Cx4_ڀريو(p)
    #وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #وصف_ڪريو GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F32_Cx4_جوڙيو
    #وصف_ڪريو GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#جيڪڏهن_پورو

#وإلاإذا defined(__AVX512F__)

#وصف_ڪريو GGML_SIMD

// F32 AVX512

#وصف_ڪريو GGML_F32_قدم 64
#وصف_ڪريو GGML_F32_EPR  16

#وصف_ڪريو GGML_F32x16         __m512
#وصف_ڪريو GGML_F32x16_ZERO    _mm512_setzero_ps()
#وصف_ڪريو GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#وصف_ڪريو GGML_F32x16_ڀريو    _mm512_loadu_ps
#وصف_ڪريو GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#وصف_ڪريو GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#وصف_ڪريو GGML_F32x16_جوڙيو     mm512_جوڙيو_ps
#وصف_ڪريو GGML_F32x16_MUL     _mm512_mul_ps
#وصف_ڪريو GGML_F32x16_REDUCE(res, x)                                    \
افعل {                                                                  \
    صحيح وٿي = GGML_F32_ARR >> 1;                                   \
    لكل (صحيح i = 0; i < وٿي; ++i) {                                \
        x[i] = mm512_جوڙيو_ps(x[i], x[وٿي+i]);                      \
    }                                                                 \
    وٿي >>= 1;                                                     \
    لكل (صحيح i = 0; i < وٿي; ++i) {                                \
        x[i] = mm512_جوڙيو_ps(x[i], x[وٿي+i]);                      \
    }                                                                 \
    وٿي >>= 1;                                                     \
    لكل (صحيح i = 0; i < وٿي; ++i) {                                \
        x[i] = mm512_جوڙيو_ps(x[i], x[وٿي+i]);                      \
    }                                                                 \
    res = (ggml_float) mm512_reduce_جوڙيو_ps(x[0]);                    \
} طالما (0)

// TODO: is this optimal ?

#وصف_ڪريو GGML_F32_VEC        GGML_F32x16
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x16_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x16_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x16_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x16_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x16_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x16_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x16_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512

#إذا defined(__AVX512FP16__)

#وصف_ڪريو GGML_F16_قدم 128
#وصف_ڪريو GGML_F16_EPR  32

#وصف_ڪريو GGML_F16x32              __m512h
#وصف_ڪريو GGML_F16x32_ZERO         _mm512_setzero_ph()
#وصف_ڪريو GGML_F16x32_SET1(x)      _mm512_set1_ph(__extension__(_Float16)(x))
#وصف_ڪريو GGML_F16x32_ڀريو(x)      _mm512_loadu_ph(x)
#وصف_ڪريو GGML_F16x32_STORE(x, y)  _mm512_storeu_ph(x, y)
#وصف_ڪريو GGML_F16x32_FMA(a, b, c) _mm512_fmadd_ph(b, c, a)
#وصف_ڪريو GGML_F16x32_جوڙيو          mm512_جوڙيو_ph
#وصف_ڪريو GGML_F16x32_MUL          _mm512_mul_ph
#وصف_ڪريو GGML_F16x32_REDUCE(res, x)                                     \
افعل {                                                                   \
    صحيح وٿي = GGML_F16_ARR >> 1;                                    \
    لكل (صحيح i = 0; i < وٿي; ++i) {                                 \
        x[i] = mm512_جوڙيو_ph(x[i], x[وٿي+i]);                       \
    }                                                                  \
    وٿي >>= 1;                                                      \
    لكل (صحيح i = 0; i < وٿي; ++i) {                                 \
        x[i] = mm512_جوڙيو_ph(x[i], x[وٿي+i]);                       \
    }                                                                  \
    وٿي >>= 1;                                                      \
    لكل (صحيح i = 0; i < وٿي; ++i) {                                 \
        x[i] = mm512_جوڙيو_ph(x[i], x[وٿي+i]);                       \
    }                                                                  \
    res = (ggml_float) mm512_reduce_جوڙيو_ph(x[0]);                     \
} طالما (0)

#وصف_ڪريو GGML_F16_VEC                GGML_F16x32
#وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F16x32_ZERO
#وصف_ڪريو GGML_F16_VEC_SET1           GGML_F16x32_SET1
#وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     GGML_F16x32_ڀريو(p)
#وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) GGML_F16x32_STORE(p, r[i])
#وصف_ڪريو GGML_F16_VEC_FMA            GGML_F16x32_FMA
#وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F16x32_جوڙيو
#وصف_ڪريو GGML_F16_VEC_MUL            GGML_F16x32_MUL
#وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F16x32_REDUCE

#وإلا // Fallback FP16 <-> FP32

#وصف_ڪريو GGML_F16_قدم 64
#وصف_ڪريو GGML_F16_EPR  16

#وصف_ڪريو GGML_F32Cx16             __m512
#وصف_ڪريو GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#وصف_ڪريو GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#وصف_ڪريو GGML_F32_Cx16_ڀريو(x)     _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(x)))
#وصف_ڪريو GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#وصف_ڪريو GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#وصف_ڪريو GGML_F32_Cx16_جوڙيو         mm512_جوڙيو_ps
#وصف_ڪريو GGML_F32Cx16_MUL         _mm512_mul_ps
#وصف_ڪريو GGML_F32Cx16_REDUCE(res, x)                               \
افعل {                                                              \
    صحيح وٿي = GGML_F32_ARR >> 1;                               \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = mm512_جوڙيو_ps(x[i], x[وٿي+i]);                  \
    }                                                             \
    وٿي >>= 1;                                                 \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = mm512_جوڙيو_ps(x[i], x[وٿي+i]);                  \
    }                                                             \
    وٿي >>= 1;                                                 \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = mm512_جوڙيو_ps(x[i], x[وٿي+i]);                  \
    }                                                             \
    res = (ggml_float) mm512_reduce_جوڙيو_ps(x[0]);                \
} طالما (0)

#وصف_ڪريو GGML_F16_VEC                GGML_F32Cx16
#وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F32Cx16_ZERO
#وصف_ڪريو GGML_F16_VEC_SET1           GGML_F32Cx16_SET1
#وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     GGML_F32_Cx16_ڀريو(p)
#وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#وصف_ڪريو GGML_F16_VEC_FMA            GGML_F32Cx16_FMA
#وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F32_Cx16_جوڙيو
#وصف_ڪريو GGML_F16_VEC_MUL            GGML_F32Cx16_MUL

#وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F32Cx16_REDUCE

#جيڪڏهن_پورو // __AVX512FP16__
#وإلاإذا defined(__AVX__)

#وصف_ڪريو GGML_SIMD

// F32 AVX

#وصف_ڪريو GGML_F32_قدم 32
#وصف_ڪريو GGML_F32_EPR  8

#وصف_ڪريو GGML_F32x8         __m256
#وصف_ڪريو GGML_F32x8_ZERO    _mm256_setzero_ps()
#وصف_ڪريو GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#وصف_ڪريو GGML_F32x8_ڀريو    _mm256_loadu_ps
#وصف_ڪريو GGML_F32x8_STORE   _mm256_storeu_ps
#إذا defined(__FMA__)
    #وصف_ڪريو GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#وإلا
    #وصف_ڪريو GGML_F32x8_FMA(a, b, c) mm256_جوڙيو_ps(_mm256_mul_ps(b, c), a)
#جيڪڏهن_پورو
#وصف_ڪريو GGML_F32x8_جوڙيو     mm256_جوڙيو_ps
#وصف_ڪريو GGML_F32x8_MUL     _mm256_mul_ps
#وصف_ڪريو GGML_F32x8_REDUCE(res, x)                                 \
افعل {                                                              \
    صحيح وٿي = GGML_F32_ARR >> 1;                               \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = mm256_جوڙيو_ps(x[i], x[وٿي+i]);                  \
    }                                                             \
    وٿي >>= 1;                                                 \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = mm256_جوڙيو_ps(x[i], x[وٿي+i]);                  \
    }                                                             \
    وٿي >>= 1;                                                 \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = mm256_جوڙيو_ps(x[i], x[وٿي+i]);                  \
    }                                                             \
    ثابت __m128 t0 = mm_جوڙيو_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    ثابت __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} طالما (0)
// TODO: is this optimal ?

#وصف_ڪريو GGML_F32_VEC        GGML_F32x8
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x8_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x8_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x8_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x8_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x8_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x8_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 AVX

#وصف_ڪريو GGML_F16_قدم 32
#وصف_ڪريو GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#وصف_ڪريو GGML_F32Cx8             __m256
#وصف_ڪريو GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#وصف_ڪريو GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#إذا defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#وصف_ڪريو GGML_F32_Cx8_ڀريو(x)     _mm256_cvtph_ps(_mm_loadu_si128((ثابت __m128i *)(x)))
#وصف_ڪريو GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#وإلا
ساكن مضمن __m256 __avx_f32cx8_load(ثابت ggml_fp16_t * x) {
    عائم tmp[8];

    لكل (صحيح i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_منزل_FP32(x[i]);
    }

    أرجع _mm256_loadu_ps(tmp);
}
ساكن مضمن فراغ __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    عائم arr[8];

    _mm256_storeu_ps(arr, y);

    لكل (صحيح i = 0; i < 8; i++)
        x[i] = GGML_CPU_FP32_منزل_FP16(arr[i]);
}
#وصف_ڪريو GGML_F32_Cx8_ڀريو(x)     __avx_f32cx8_load(x)
#وصف_ڪريو GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#جيڪڏهن_پورو

#وصف_ڪريو GGML_F32Cx8_FMA         GGML_F32x8_FMA
#وصف_ڪريو GGML_F32_Cx8_جوڙيو         mm256_جوڙيو_ps
#وصف_ڪريو GGML_F32Cx8_MUL         _mm256_mul_ps
#وصف_ڪريو GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#وصف_ڪريو GGML_F16_VEC                GGML_F32Cx8
#وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#وصف_ڪريو GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     GGML_F32_Cx8_ڀريو(p)
#وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#وصف_ڪريو GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F32_Cx8_جوڙيو
#وصف_ڪريو GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#وإلاإذا defined(__POWER9_VECTOR__)

#وصف_ڪريو GGML_SIMD

// F32 POWER9

#وصف_ڪريو GGML_F32_قدم 32
#وصف_ڪريو GGML_F32_EPR  4

#وصف_ڪريو GGML_F32x4              ويڪٽر عائم
#وصف_ڪريو GGML_F32x4_ZERO         {0.0f}
#وصف_ڪريو GGML_F32x4_SET1         vec_splats
#وصف_ڪريو GGML_F32x4_ڀريو(p)      vec_xl(0, p)
#وصف_ڪريو GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#وصف_ڪريو GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#وصف_ڪريو GGML_F32x4_جوڙيو          vec_جوڙيو
#وصف_ڪريو GGML_F32x4_MUL          vec_mul
#وصف_ڪريو GGML_F32x4_REDUCE(res, x)              \
{                                              \
    صحيح وٿي = GGML_F32_ARR >> 1;            \
    لكل (صحيح i = 0; i < وٿي; ++i) {         \
        x[i] = vec_جوڙيو(x[i], x[وٿي+i]);     \
    }                                          \
    وٿي >>= 1;                              \
    لكل (صحيح i = 0; i < وٿي; ++i) {         \
        x[i] = vec_جوڙيو(x[i], x[وٿي+i]);     \
    }                                          \
    وٿي >>= 1;                              \
    لكل (صحيح i = 0; i < وٿي; ++i) {         \
        x[i] = vec_جوڙيو(x[i], x[وٿي+i]);     \
    }                                          \
    res = vec_ڪڍو(x[0], 0) +               \
          vec_ڪڍو(x[0], 1) +               \
          vec_ڪڍو(x[0], 2) +               \
          vec_ڪڍو(x[0], 3);                \
}
#وصف_ڪريو GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3)        \
{                                                       \
    ويڪٽر عائم v = vec_جوڙيو(vec_جوڙيو(s0, s1),           \
                             vec_جوڙيو(s2, s3));          \
    v = vec_جوڙيو(v, vec_sld(v, v, 8));                   \
    v = vec_جوڙيو(v, vec_sld(v, v, 4));                   \
    res += (ggml_float) vec_ڪڍو(v, 0);              \
}

#وصف_ڪريو GGML_F32_VEC        GGML_F32x4
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x4_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x4_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x4_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x4_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x4_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x4_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 POWER9
#وصف_ڪريو GGML_F16_قدم       GGML_F32_قدم
#وصف_ڪريو GGML_F16_EPR        GGML_F32_EPR
#وصف_ڪريو GGML_F16_VEC        GGML_F32x4
#وصف_ڪريو GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#وصف_ڪريو GGML_F16_VEC_SET1   GGML_F32x4_SET1
#وصف_ڪريو GGML_F16_VEC_FMA    GGML_F32x4_FMA
#وصف_ڪريو GGML_F16_VEC_جوڙيو    GGML_F32x4_جوڙيو
#وصف_ڪريو GGML_F16_VEC_MUL    GGML_F32x4_MUL
#وصف_ڪريو GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#وصف_ڪريو GGML_F16_VEC_ڀريو(p, i) (i & 0x1) ?                   \
  vec_ڪڍو_fp32_from_shorth(vec_xl(0, p - GGML_F16_EPR)) : \
  vec_ڪڍو_fp32_from_shortl(vec_xl(0, p))
ساكن مضمن بنانشان حرف ggml_endian_byte(صحيح i) {
       uint16_t tmp_val = 1;
       أرجع ((بنانشان حرف *)&tmp_val)[i];
}
#وصف_ڪريو GGML_ENDIAN_BYTE(i) ggml_endian_byte(i)
#وصف_ڪريو GGML_F16_VEC_STORE(p, r, i)                             \
  إذا (i & 0x1)                                                  \
    vec_xst(vec_pack_منزل_short_fp32(r[i - GGML_ENDIAN_BYTE(1)],  \
                                   r[i - GGML_ENDIAN_BYTE(0)]), \
            0, p - GGML_F16_EPR)

//BF16 POWER9
#وصف_ڪريو GGML_BF16_قدم 16
#وصف_ڪريو GGML_BF16_EPR  8

#وصف_ڪريو GGML_BF16x8         ويڪٽر بنانشان ننڍو
#وصف_ڪريو GGML_BF16x8_ZERO    vec_splats((بنانشان ننڍو)0)
#وصف_ڪريو GGML_BF16x8_ڀريو(p) vec_xl(0, (ثابت بنانشان ننڍو *)(p))

#وصف_ڪريو GGML_BF16_VEC          GGML_BF16x8
#وصف_ڪريو GGML_BF16_VEC_ZERO     GGML_BF16x8_ZERO
#وصف_ڪريو GGML_BF16_VEC_ڀريو     GGML_BF16x8_ڀريو
#إذا defined(__LITTLE_ENDIAN__)
#وصف_ڪريو GGML_BF16_منزل_F32_LO(v) ((ويڪٽر عائم) vec_mergel(GGML_BF16_VEC_ZERO, (v)))
#وصف_ڪريو GGML_BF16_منزل_F32_HI(v) ((ويڪٽر عائم) vec_mergeh(GGML_BF16_VEC_ZERO, (v)))
#وإلا
#وصف_ڪريو GGML_BF16_منزل_F32_LO(v) ((ويڪٽر عائم) vec_mergel((v), GGML_BF16_VEC_ZERO))
#وصف_ڪريو GGML_BF16_منزل_F32_HI(v) ((ويڪٽر عائم) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#جيڪڏهن_پورو
#وصف_ڪريو GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_منزل_F32_LO(x), GGML_BF16_منزل_F32_LO(y))
#وصف_ڪريو GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_منزل_F32_HI(x), GGML_BF16_منزل_F32_HI(y))

#وإلاإذا defined(__wasm_simd128__)

#وصف_ڪريو GGML_SIMD

// F32 WASM

#وصف_ڪريو GGML_F32_قدم 16
#وصف_ڪريو GGML_F32_EPR  4

#وصف_ڪريو GGML_F32x4              v128_t
#وصف_ڪريو GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#وصف_ڪريو GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#وصف_ڪريو GGML_F32x4_ڀريو         wasm_v128_ڀريو
#وصف_ڪريو GGML_F32x4_STORE        wasm_v128_store
#وصف_ڪريو GGML_F32x4_FMA(a, b, c) wasm_f32x4_جوڙيو(wasm_f32x4_mul(b, c), a)
#وصف_ڪريو GGML_F32x4_جوڙيو          wasm_f32x4_جوڙيو
#وصف_ڪريو GGML_F32x4_MUL          wasm_f32x4_mul
#وصف_ڪريو GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    صحيح وٿي = GGML_F32_ARR >> 1;                \
    لكل (صحيح i = 0; i < وٿي; ++i) {             \
        x[i] = wasm_f32x4_جوڙيو(x[i], x[وٿي+i]);  \
    }                                              \
    وٿي >>= 1;                                  \
    لكل (صحيح i = 0; i < وٿي; ++i) {             \
        x[i] = wasm_f32x4_جوڙيو(x[i], x[وٿي+i]);  \
    }                                              \
    وٿي >>= 1;                                  \
    لكل (صحيح i = 0; i < وٿي; ++i) {             \
        x[i] = wasm_f32x4_جوڙيو(x[i], x[وٿي+i]);  \
    }                                              \
    res = wasm_f32x4_ڪڍو_lane(x[0], 0) +       \
          wasm_f32x4_ڪڍو_lane(x[0], 1) +       \
          wasm_f32x4_ڪڍو_lane(x[0], 2) +       \
          wasm_f32x4_ڪڍو_lane(x[0], 3);        \
}

#وصف_ڪريو GGML_F32_VEC        GGML_F32x4
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x4_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x4_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x4_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x4_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x4_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x4_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 WASM

#وصف_ڪريو GGML_F16_قدم 16
#وصف_ڪريو GGML_F16_EPR  4

مضمن ساكن v128_t __wasm_f16x4_load(ثابت ggml_fp16_t * p) {
    عائم tmp[4];

    tmp[0] = GGML_CPU_FP16_منزل_FP32(p[0]);
    tmp[1] = GGML_CPU_FP16_منزل_FP32(p[1]);
    tmp[2] = GGML_CPU_FP16_منزل_FP32(p[2]);
    tmp[3] = GGML_CPU_FP16_منزل_FP32(p[3]);

    أرجع wasm_v128_ڀريو(tmp);
}

مضمن ساكن فراغ __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
    عائم tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = GGML_CPU_FP32_منزل_FP16(tmp[0]);
    p[1] = GGML_CPU_FP32_منزل_FP16(tmp[1]);
    p[2] = GGML_CPU_FP32_منزل_FP16(tmp[2]);
    p[3] = GGML_CPU_FP32_منزل_FP16(tmp[3]);
}

#وصف_ڪريو GGML_F16x4             v128_t
#وصف_ڪريو GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#وصف_ڪريو GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#وصف_ڪريو GGML_F16x4_ڀريو(x)     __wasm_f16x4_load(x)
#وصف_ڪريو GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#وصف_ڪريو GGML_F16x4_FMA         GGML_F32x4_FMA
#وصف_ڪريو GGML_F16x4_جوڙيو         wasm_f32x4_جوڙيو
#وصف_ڪريو GGML_F16x4_MUL         wasm_f32x4_mul
#وصف_ڪريو GGML_F16x4_REDUCE(res, x)                           \
{                                                           \
    صحيح وٿي = GGML_F16_ARR >> 1;                         \
    لكل (صحيح i = 0; i < وٿي; ++i) {                      \
        x[i] = wasm_f32x4_جوڙيو(x[i], x[وٿي+i]);           \
    }                                                       \
    وٿي >>= 1;                                           \
    لكل (صحيح i = 0; i < وٿي; ++i) {                      \
        x[i] = wasm_f32x4_جوڙيو(x[i], x[وٿي+i]);           \
    }                                                       \
    وٿي >>= 1;                                           \
    لكل (صحيح i = 0; i < وٿي; ++i) {                      \
        x[i] = wasm_f32x4_جوڙيو(x[i], x[وٿي+i]);           \
    }                                                       \
    res = (ggml_float) (wasm_f32x4_ڪڍو_lane(x[0], 0) +  \
          wasm_f32x4_ڪڍو_lane(x[0], 1) +                \
          wasm_f32x4_ڪڍو_lane(x[0], 2) +                \
          wasm_f32x4_ڪڍو_lane(x[0], 3));                \
}

#وصف_ڪريو GGML_F16_VEC                GGML_F16x4
#وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F16x4_ZERO
#وصف_ڪريو GGML_F16_VEC_SET1           GGML_F16x4_SET1
#وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     GGML_F16x4_ڀريو(p)
#وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) GGML_F16x4_STORE(p, r[i])
#وصف_ڪريو GGML_F16_VEC_FMA            GGML_F16x4_FMA
#وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F16x4_جوڙيو
#وصف_ڪريو GGML_F16_VEC_MUL            GGML_F16x4_MUL
#وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F16x4_REDUCE

#وإلاإذا defined(__SSE3__)

#وصف_ڪريو GGML_SIMD

// F32 SSE

#وصف_ڪريو GGML_F32_قدم 32
#وصف_ڪريو GGML_F32_EPR  4

#وصف_ڪريو GGML_F32x4         __m128
#وصف_ڪريو GGML_F32x4_ZERO    _mm_setzero_ps()
#وصف_ڪريو GGML_F32x4_SET1(x) _mm_set1_ps(x)
#وصف_ڪريو GGML_F32x4_ڀريو    _mm_loadu_ps
#وصف_ڪريو GGML_F32x4_STORE   _mm_storeu_ps
#إذا defined(__FMA__)
    // TODO: Does this work?
    #وصف_ڪريو GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#وإلا
    #وصف_ڪريو GGML_F32x4_FMA(a, b, c) mm_جوڙيو_ps(_mm_mul_ps(b, c), a)
#جيڪڏهن_پورو
#وصف_ڪريو GGML_F32x4_جوڙيو     mm_جوڙيو_ps
#وصف_ڪريو GGML_F32x4_MUL     _mm_mul_ps
#وصف_ڪريو GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    صحيح وٿي = GGML_F32_ARR >> 1;                               \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = mm_جوڙيو_ps(x[i], x[وٿي+i]);                     \
    }                                                             \
    وٿي >>= 1;                                                 \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = mm_جوڙيو_ps(x[i], x[وٿي+i]);                     \
    }                                                             \
    وٿي >>= 1;                                                 \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = mm_جوڙيو_ps(x[i], x[وٿي+i]);                     \
    }                                                             \
    ثابت __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#وصف_ڪريو GGML_F32_VEC        GGML_F32x4
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x4_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x4_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x4_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x4_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x4_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x4_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 SSE

#وصف_ڪريو GGML_F16_قدم 32
#وصف_ڪريو GGML_F16_EPR  4

ساكن مضمن __m128 __sse_f16x4_load(ثابت ggml_fp16_t * x) {
    عائم tmp[4];

    tmp[0] = GGML_CPU_FP16_منزل_FP32(x[0]);
    tmp[1] = GGML_CPU_FP16_منزل_FP32(x[1]);
    tmp[2] = GGML_CPU_FP16_منزل_FP32(x[2]);
    tmp[3] = GGML_CPU_FP16_منزل_FP32(x[3]);

    أرجع _mm_loadu_ps(tmp);
}

ساكن مضمن فراغ __sse_f16x4_store(ggml_fp16_t * x, __m128 y) {
    عائم arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = GGML_CPU_FP32_منزل_FP16(arr[0]);
    x[1] = GGML_CPU_FP32_منزل_FP16(arr[1]);
    x[2] = GGML_CPU_FP32_منزل_FP16(arr[2]);
    x[3] = GGML_CPU_FP32_منزل_FP16(arr[3]);
}

#وصف_ڪريو GGML_F32Cx4             __m128
#وصف_ڪريو GGML_F32Cx4_ZERO        _mm_setzero_ps()
#وصف_ڪريو GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#وصف_ڪريو GGML_F32_Cx4_ڀريو(x)     __sse_f16x4_load(x)
#وصف_ڪريو GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#وصف_ڪريو GGML_F32Cx4_FMA         GGML_F32x4_FMA
#وصف_ڪريو GGML_F32_Cx4_جوڙيو         mm_جوڙيو_ps
#وصف_ڪريو GGML_F32Cx4_MUL         _mm_mul_ps
#وصف_ڪريو GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#وصف_ڪريو GGML_F16_VEC                 GGML_F32Cx4
#وصف_ڪريو GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#وصف_ڪريو GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)      GGML_F32_Cx4_ڀريو(p)
#وصف_ڪريو GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#وصف_ڪريو GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#وصف_ڪريو GGML_F16_VEC_جوڙيو             GGML_F32_Cx4_جوڙيو
#وصف_ڪريو GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#وصف_ڪريو GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#وإلاإذا defined(__loongarch_asx)

#وصف_ڪريو GGML_SIMD

// F32 LASX
#وصف_ڪريو GGML_F32_قدم 32
#وصف_ڪريو GGML_F32_EPR  8

#وصف_ڪريو GGML_F32x8         __m256
#وصف_ڪريو GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#وصف_ڪريو GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#وصف_ڪريو GGML_F32x8_ڀريو(x) (__m256)__lasx_xvld((x), 0)
#وصف_ڪريو GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#وصف_ڪريو GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#وصف_ڪريو GGML_F32x8_جوڙيو     __lasx_xvfadd_s
#وصف_ڪريو GGML_F32x8_MUL     __lasx_xvfmul_s
#وصف_ڪريو GGML_F32x8_REDUCE(res, x)                                 \
افعل {                                                              \
    صحيح وٿي = GGML_F32_ARR >> 1;                               \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[وٿي+i]);                  \
    }                                                             \
    وٿي >>= 1;                                                 \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[وٿي+i]);                  \
    }                                                             \
    وٿي >>= 1;                                                 \
    لكل (صحيح i = 0; i < وٿي; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[وٿي+i]);                  \
    }                                                             \
    عائم *tmp_p = (عائم *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} طالما (0)
// TODO: is this optimal ?

#وصف_ڪريو GGML_F32_VEC        GGML_F32x8
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x8_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x8_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x8_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x8_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x8_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x8_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 LASX

#وصف_ڪريو GGML_F16_قدم 32
#وصف_ڪريو GGML_F16_EPR  8

// F16 arithmetic is not supported by LASX, so we use F32 instead

#وصف_ڪريو GGML_F32Cx8          __m256
#وصف_ڪريو GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#وصف_ڪريو GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))

ساكن مضمن __m256 __lasx_f32cx8_load(ثابت ggml_fp16_t * x) {
    __m256i a;
    يادگيري_نقل(&a, x, حجم(ggml_fp16_t) * 8);
    a = __lasx_xvpermi_d(a, 0 | (1 << 4));
    أرجع __lasx_xvfcvtl_s_h(a);
}

ساكن مضمن فراغ __lasx_f32cx8_store(ggml_fp16_t * x, __m256 y) {
    __m256i a = __lasx_xvfcvt_h_s(y, y);
    a = __lasx_xvpermi_d(a, 0 | (2 << 2));
    يادگيري_نقل(x, &a, حجم(ggml_fp16_t) * 8);
}
#وصف_ڪريو GGML_F32_Cx8_ڀريو(x)     __lasx_f32cx8_load(x)
#وصف_ڪريو GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#وصف_ڪريو GGML_F32Cx8_FMA         GGML_F32x8_FMA
#وصف_ڪريو GGML_F32_Cx8_جوڙيو         __lasx_xvfadd_s
#وصف_ڪريو GGML_F32Cx8_MUL         __lasx_xvfmul_s
#وصف_ڪريو GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#وصف_ڪريو GGML_F16_VEC                GGML_F32Cx8
#وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#وصف_ڪريو GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     GGML_F32_Cx8_ڀريو(p)
#وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#وصف_ڪريو GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F32_Cx8_جوڙيو
#وصف_ڪريو GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#وإلاإذا defined(__loongarch_sx)

#وصف_ڪريو GGML_SIMD

// F32 LSX

#وصف_ڪريو GGML_F32_قدم 32
#وصف_ڪريو GGML_F32_EPR  4

#وصف_ڪريو GGML_F32x4         __m128
#وصف_ڪريو GGML_F32x4_ZERO    (__m128)__lsx_vldi(0)
#وصف_ڪريو GGML_F32x4_SET1(x) (__m128)__lsx_vreplfr2vr_s((x))
#وصف_ڪريو GGML_F32x4_ڀريو(x) (__m128)__lsx_vld((x), 0)
#وصف_ڪريو GGML_F32x4_STORE(x, y)   __lsx_vst(y, x, 0)
#وصف_ڪريو GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#وصف_ڪريو GGML_F32x4_جوڙيو     __lsx_vfadd_s
#وصف_ڪريو GGML_F32x4_MUL     __lsx_vfmul_s

#وصف_ڪريو GGML_F32x4_REDUCE(res, x)                               \
{                                                               \
    صحيح وٿي = GGML_F32_ARR >> 1;                             \
    لكل (صحيح i = 0; i < وٿي; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[وٿي+i]);                \
    }                                                           \
    وٿي >>= 1;                                               \
    لكل (صحيح i = 0; i < وٿي; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[وٿي+i]);                \
    }                                                           \
    وٿي >>= 1;                                               \
    لكل (صحيح i = 0; i < وٿي; ++i) {                          \
        x[i] = __lsx_vfadd_s(x[i], x[وٿي+i]);                \
    }                                                           \
    __m128i t0 = __lsx_vpickev_w((__m128i)x[0], (__m128i)x[0]); \
    __m128i t1 = __lsx_vpickod_w((__m128i)x[0], (__m128i)x[0]); \
    __m128 t2 = __lsx_vfadd_s((__m128)t0, (__m128)t1);          \
    __m128i t3 = __lsx_vpickev_w((__m128i)t2, (__m128i)t2);     \
    __m128i t4 = __lsx_vpickod_w((__m128i)t2, (__m128i)t2);     \
    __m128 t5 = __lsx_vfadd_s((__m128)t3, (__m128)t4);          \
    res = (ggml_float) ((v4f32)t5)[0];                          \
}

#وصف_ڪريو GGML_F32_VEC        GGML_F32x4
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x4_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x4_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x4_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x4_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x4_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x4_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 LSX

#وصف_ڪريو GGML_F16_قدم 32
#وصف_ڪريو GGML_F16_EPR  4

ساكن مضمن __m128 __lsx_f16x4_load(ثابت ggml_fp16_t * x) {
    أرجع __lsx_vfcvtl_s_h(__lsx_vld((ثابت فراغ *)x, 0));
}

ساكن مضمن فراغ __lsx_f16x4_store(ggml_fp16_t * x, __m128 y) {
    __m128i a = __lsx_vfcvt_h_s(y, y);
    يادگيري_نقل(x, &a, حجم(ggml_fp16_t) * 4);
}

#وصف_ڪريو GGML_F32Cx4             __m128
#وصف_ڪريو GGML_F32Cx4_ZERO        (__m128)__lsx_vldi(0)
#وصف_ڪريو GGML_F32Cx4_SET1(x)     (__m128)__lsx_vreplfr2vr_s((x))
#وصف_ڪريو GGML_F32_Cx4_ڀريو(x)     (__m128)__lsx_f16x4_load(x)
#وصف_ڪريو GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#وصف_ڪريو GGML_F32Cx4_FMA         GGML_F32x4_FMA
#وصف_ڪريو GGML_F32_Cx4_جوڙيو         __lsx_vfadd_s
#وصف_ڪريو GGML_F32Cx4_MUL         __lsx_vfmul_s
#وصف_ڪريو GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#وصف_ڪريو GGML_F16_VEC                 GGML_F32Cx4
#وصف_ڪريو GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#وصف_ڪريو GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)      GGML_F32_Cx4_ڀريو(p)
#وصف_ڪريو GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#وصف_ڪريو GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#وصف_ڪريو GGML_F16_VEC_جوڙيو             GGML_F32_Cx4_جوڙيو
#وصف_ڪريو GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#وصف_ڪريو GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#وإلاإذا defined(__VXE__) || defined(__VXE2__)

#وصف_ڪريو GGML_SIMD

// F32 s390x

#وصف_ڪريو GGML_F32_قدم 32
#وصف_ڪريو GGML_F32_EPR  4

#وصف_ڪريو GGML_F32x4              float32x4_t
#وصف_ڪريو GGML_F32x4_ZERO         vec_splats(0.0f)
#وصف_ڪريو GGML_F32x4_SET1         vec_splats
#وصف_ڪريو GGML_F32x4_ڀريو(p)      vec_xl(0, p)
#وصف_ڪريو GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#وصف_ڪريو GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#وصف_ڪريو GGML_F32x4_جوڙيو          vec_جوڙيو
#وصف_ڪريو GGML_F32x4_MUL          vec_mul
#وصف_ڪريو GGML_F32x4_REDUCE(res, x)                   \
{                                                   \
    صحيح وٿي = GGML_F32_ARR >> 1;                 \
    لكل (صحيح i = 0; i < وٿي; ++i) {              \
        x[i] = vec_جوڙيو(x[i], x[وٿي + i]);        \
    }                                               \
    وٿي >>= 1;                                   \
    لكل (صحيح i = 0; i < وٿي; ++i) {              \
        x[i] = vec_جوڙيو(x[i], x[وٿي + i]);        \
    }                                               \
    وٿي >>= 1;                                   \
    لكل (صحيح i = 0; i < وٿي; ++i) {              \
        x[i] = vec_جوڙيو(x[i], x[وٿي + i]);        \
    }                                               \
    float32x4_t tmp = x[0] + vec_reve(x[0]);        \
    res = tmp[0] + tmp[1];                          \
}
#وصف_ڪريو GGML_F32x4_REDUCE_4(res, s0, s1, s2, s3) \
{                                                \
    float32x4_t v = vec_جوڙيو(vec_جوڙيو(s0, s1),     \
                            vec_جوڙيو(s2, s3));    \
    v = vec_جوڙيو(v, vec_sld(v, v, 8));            \
    v = vec_جوڙيو(v, vec_sld(v, v, 4));            \
    res += (ggml_float)vec_ڪڍو(v, 0);        \
}

#وصف_ڪريو GGML_F32_VEC        GGML_F32x4
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x4_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x4_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x4_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x4_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x4_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x4_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 s390x
#وصف_ڪريو GGML_F16_قدم GGML_F32_قدم
#وصف_ڪريو GGML_F16_EPR  GGML_F32_EPR

ساكن مضمن uint32x4_t __lzs_f32cx4_to_f16(float32x4_t v_f) {
    float32x4_t v_بنياد = vec_mul(vec_mul(vec_abs(v_f), vec_splats(0x1.0p+112f)), vec_splats(0x1.0p-110f));

    ثابت uint32x4_t v_w      = (uint32x4_t)v_f;
    ثابت uint32x4_t v_shl1_w = vec_جوڙيو(v_w, v_w);
    ثابت uint32x4_t v_sign   = vec_and(v_w, vec_splats(UINT32_C(0x80000000)));
    ثابت uint32x4_t v_bias   = vec_max(vec_and(v_shl1_w, vec_splats(UINT32_C(0xFF000000))), vec_splats(UINT32_C(0x71000000)));

    v_بنياد = vec_جوڙيو((float32x4_t)vec_جوڙيو(vec_sr(v_bias, 1), vec_splats(UINT32_C(0x07800000))), v_بنياد);

    ثابت uint32x4_t v_بٽ    = (uint32x4_t)v_بنياد;
    ثابت uint32x4_t v_nonsign = vec_جوڙيو(vec_and(vec_sr(v_بٽ, 13), vec_splats(UINT32_C(0x00007C00))),
                                         vec_and(v_بٽ, vec_splats(UINT32_C(0x00000FFF))));
    ثابت uint32x4_t v_is_nan  = (uint32x4_t)vec_cmpgt(v_shl1_w, vec_splats(UINT32_C(0xFF000000)));

    أرجع vec_or(vec_sr(v_sign, 16), vec_sel(v_nonsign, vec_splats(UINT32_C(0x7E00)), v_is_nan));
}

ساكن مضمن float32x4_t __lzs_f16cx4_load(ثابت ggml_fp16_t * x) {
    عائم tmp[4];

    لكل (صحيح i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_منزل_FP32(x[i]);
    }

    // note: keep type-cast here to prevent compiler bugs
    // see: https://github.com/ggml-org/llama.cpp/issues/12846
    أرجع vec_xl(0, (ثابت عائم *)(tmp));
}

ساكن مضمن فراغ __lzs_f16cx4_store(ggml_fp16_t * x, float32x4_t v_y) {
    ثابت uint32x4_t v_h = __lzs_f32cx4_to_f16(v_y);
    ثابت uint64_t   tmp = ((uint64x2_t)vec_pack(v_h, v_h))[0];
    يادگيري_نقل(x, &tmp, حجم(tmp));
}

#وصف_ڪريو GGML_F16_VEC                GGML_F32x4
#وصف_ڪريو GGML_F16_VEC_ZERO           GGML_F32x4_ZERO
#وصف_ڪريو GGML_F16_VEC_SET1           GGML_F32x4_SET1
#وصف_ڪريو GGML_F16_VEC_ڀريو(p, i)     __lzs_f16cx4_load(p)
#وصف_ڪريو GGML_F16_VEC_STORE(p, r, i) __lzs_f16cx4_store(p, r[i])
#وصف_ڪريو GGML_F16_VEC_FMA            GGML_F32x4_FMA
#وصف_ڪريو GGML_F16_VEC_جوڙيو            GGML_F32x4_جوڙيو
#وصف_ڪريو GGML_F16_VEC_MUL            GGML_F32x4_MUL
#وصف_ڪريو GGML_F16_VEC_REDUCE         GGML_F32x4_REDUCE

// BF16 s390x
#وصف_ڪريو GGML_BF16_قدم 16
#وصف_ڪريو GGML_BF16_EPR  8

#وصف_ڪريو GGML_BF16x8         __vector بنانشان ننڍو
#وصف_ڪريو GGML_BF16x8_ZERO    vec_splats((بنانشان ننڍو)0)
#وصف_ڪريو GGML_BF16x8_ڀريو(p) vec_xl(0, (ثابت بنانشان ننڍو *)(p))

#وصف_ڪريو GGML_BF16_VEC      GGML_BF16x8
#وصف_ڪريو GGML_BF16_VEC_ZERO GGML_BF16x8_ZERO
#وصف_ڪريو GGML_BF16_VEC_ڀريو GGML_BF16x8_ڀريو
#وصف_ڪريو GGML_BF16_منزل_F32_LO(v) ((float32x4_t) vec_mergel((v), GGML_BF16_VEC_ZERO))
#وصف_ڪريو GGML_BF16_منزل_F32_HI(v) ((float32x4_t) vec_mergeh((v), GGML_BF16_VEC_ZERO))
#وصف_ڪريو GGML_BF16_FMA_LO(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_منزل_F32_LO(x), GGML_BF16_منزل_F32_LO(y))
#وصف_ڪريو GGML_BF16_FMA_HI(acc, x, y) \
    (acc) = GGML_F32x4_FMA((acc), GGML_BF16_منزل_F32_HI(x), GGML_BF16_منزل_F32_HI(y))

#وإلاإذا defined(__riscv_v_intrinsic)

// compatible with vlen >= 128

#وصف_ڪريو GGML_SIMD

// F32

#وصف_ڪريو GGML_F32_قدم 16
#وصف_ڪريو GGML_F32_EPR  4

#وصف_ڪريو GGML_F32x4              vfloat32m1_t
#وصف_ڪريو GGML_F32x4_ZERO         __riscv_vfmv_v_f_f32m1(0.0f, GGML_F32_EPR)
#وصف_ڪريو GGML_F32x4_SET1(x)      __riscv_vfmv_v_f_f32m1(x, GGML_F32_EPR)
#وصف_ڪريو GGML_F32x4_ڀريو(x)      __riscv_vle32_v_f32m1(x, GGML_F32_EPR)
#وصف_ڪريو GGML_F32x4_STORE(b, v)  __riscv_vse32_v_f32m1(b, v, GGML_F32_EPR)
#وصف_ڪريو GGML_F32x4_FMA(a, b, c) __riscv_vfmacc_vv_f32m1(a, b, c, GGML_F32_EPR)
#وصف_ڪريو GGML_F32x4_جوڙيو(a, b)    __riscv_vfadd_vv_f32m1(a, b, GGML_F32_EPR)
#وصف_ڪريو GGML_F32x4_MUL(a, b)    __riscv_vfmul_vv_f32m1(a, b, GGML_F32_EPR)

#وصف_ڪريو GGML_F32_VEC        GGML_F32x4
#وصف_ڪريو GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#وصف_ڪريو GGML_F32_VEC_SET1   GGML_F32x4_SET1
#وصف_ڪريو GGML_F32_VEC_ڀريو   GGML_F32x4_ڀريو
#وصف_ڪريو GGML_F32_VEC_STORE  GGML_F32x4_STORE
#وصف_ڪريو GGML_F32_VEC_FMA    GGML_F32x4_FMA
#وصف_ڪريو GGML_F32_VEC_جوڙيو    GGML_F32x4_جوڙيو
#وصف_ڪريو GGML_F32_VEC_MUL    GGML_F32x4_MUL
#وصف_ڪريو GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

#جيڪڏهن_پورو

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#جيڪڏهن_وصف_ٿيل GGML_SIMD
#وصف_ڪريو GGML_F32_ARR (GGML_F32_قدم/GGML_F32_EPR)
#وصف_ڪريو GGML_F16_ARR (GGML_F16_قدم/GGML_F16_EPR)
#جيڪڏهن_پورو

// GGML_F16_DOT_*
// like GGML_F16_* but for dot products which need F32 accumulation on AVX512-FP16

#إذا defined(__AVX512FP16__)

#وصف_ڪريو GGML_F16_DOT_قدم           GGML_F32_قدم
#وصف_ڪريو GGML_F16_DOT_EPR            GGML_F32_EPR
#وصف_ڪريو GGML_F16_DOT_ARR            GGML_F32_ARR
#وصف_ڪريو GGML_F16_DOT_VEC            GGML_F32x16
#وصف_ڪريو GGML_F16_DOT_VEC_ZERO       GGML_F32x16_ZERO
#وصف_ڪريو GGML_F16_DOT_VEC_ڀريو(p, i) _mm512_cvtph_ps(_mm256_loadu_si256((ثابت __m256i *)(p)))
#وصف_ڪريو GGML_F16_DOT_VEC_FMA        GGML_F32x16_FMA
#وصف_ڪريو GGML_F16_DOT_VEC_REDUCE     GGML_F32x16_REDUCE

#وإلا

#وصف_ڪريو GGML_F16_DOT_قدم           GGML_F16_قدم
#وصف_ڪريو GGML_F16_DOT_EPR            GGML_F16_EPR
#وصف_ڪريو GGML_F16_DOT_ARR            GGML_F16_ARR
#وصف_ڪريو GGML_F16_DOT_VEC            GGML_F16_VEC
#وصف_ڪريو GGML_F16_DOT_VEC_ZERO       GGML_F16_VEC_ZERO
#وصف_ڪريو GGML_F16_DOT_VEC_ڀريو       GGML_F16_VEC_ڀريو
#وصف_ڪريو GGML_F16_DOT_VEC_FMA        GGML_F16_VEC_FMA
#وصف_ڪريو GGML_F16_DOT_VEC_REDUCE     GGML_F16_VEC_REDUCE

#جيڪڏهن_پورو // defined(__AVX512FP16__)

#جيڪڏهن_وصف_ٿيل __cplusplus
}
#جيڪڏهن_پورو
