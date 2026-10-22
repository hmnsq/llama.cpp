// Vectorized functions for fundamental operations

#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "simd-mappings.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu.h"

#ଯଦି defined(GGML_USE_ACCELERATE)
#ଅନ୍ତର୍ଭାଵ <Accelerate/Accelerate.h>
#endif

// floating point type used to accumulate sums
ପ୍ରକାରବ୍ୟାଖ୍ୟା ଦ୍ୱିଗୁଣ ggml_float;

#ଯଦି defined(__ARM_FEATURE_SVE)
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_sve_f16_fma_widened(
        svfloat32_t * acc_lo,
        svfloat32_t * acc_hi,
        svfloat16_t x,
        svfloat16_t y) {
#ଯଦି defined(__ARM_FEATURE_SVE2)
    *acc_lo = svmlalb_f32(*acc_lo, x, y);
    *acc_hi = svmlalt_f32(*acc_hi, x, y);
#ଅନ୍ୟଥା
    // Plain SVE fallback path if SVE2 instructions not available
    svfloat16_t x_even = svtrn1_f16(x, x);
    svfloat16_t x_odd = svtrn2_f16(x, x);

    svfloat16_t y_even = svtrn1_f16(y, y);
    svfloat16_t y_odd = svtrn2_f16(y, y);

    svbool_t pg = svptrue_b32();

    *acc_lo = svmla_f32_x(pg, *acc_lo, svcvt_f32_f16_x(pg, x_even), svcvt_f32_f16_x(pg, y_even));
    *acc_hi = svmla_f32_x(pg, *acc_hi, svcvt_f32_f16_x(pg, x_odd), svcvt_f32_f16_x(pg, y_odd));
#endif
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_float ggml_sve_sum_f32x2(svfloat32_t sum_lo, svfloat32_t sum_hi) {
    ଫେରନ୍ତୁ (ggml_float) (svaddv_f32(svptrue_b32(), sum_lo) + svaddv_f32(svptrue_b32(), sum_hi));
}
#endif

#define GGML_GELU_FP16

#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL  2
#define GGML_VEC_MAD_UNROLL  32

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

//
// global data
//

// precomputed gelu table for f16 (128 KB)
ବାହ୍ୟ ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ବାହ୍ୟ ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

//
// fundamental operations
//

ଶୂନ୍ୟ ggml_vec_dot_f32(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, size_t bx, ଅଚଳ ଦଶମିକ * GGML_RESTRICT y, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_bf16(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);
ଶୂନ୍ୟ ggml_vec_dot_f16(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrc);

ଶୂନ୍ୟ ggml_vec_silu_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x);
ggml_float ggml_vec_cvar_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ mean); //it will also center y ( y = y - mean )
ggml_float ggml_vec_soft_max_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଦଶମିକ max);
ggml_float ggml_vec_log_soft_max_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଦଶମିକ max);

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_set_i8(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, int8_t * x, ଅଚଳ int8_t v) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) x[i] = v; }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_set_i16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, int16_t * x, ଅଚଳ int16_t v) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) x[i] = v; }

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_set_i32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, int32_t * x, ଅଚଳ int32_t   v) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) x[i] = v;    }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_cpy_i32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, int32_t * y, ଅଚଳ int32_t * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = x[i]; }

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_set_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * x, ଅଚଳ ggml_fp16_t v) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) x[i] = v; }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_set_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * x, ଅଚଳ ggml_bf16_t v) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) x[i] = v; }

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_add_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * z, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * y) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0;
#ଯଦି defined(__AVX2__)
    ପାଇଁ (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vz = _mm256_add_ps(vx, vy);
        _mm256_storeu_ps(z + i, vz);
    }
#endif
    ପାଇଁ (; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_add_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * z, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * y) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) + GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_add1_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * z, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ   v) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) z[i]  = x[i] + v;    }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_acc_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x)                  { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] += x[i];        }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_acc1_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ   v)                  { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] += v;           }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sub_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * z, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * y) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sub_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * z, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * y) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) - GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_set_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * x, ଅଚଳ ଦଶମିକ   v)                  { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) x[i]  = v;           }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_cpy_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x)                  { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i]  = x[i];        }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_neg_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x)                  { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i]  = -x[i];       }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_neg_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(-GGML_CPU_FP16_TO_FP32(x[i]));
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_mul_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * z, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * y) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_mul_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * z, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * y) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) * GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_div_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * z, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * y) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_div_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * z, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * y) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) / GGML_CPU_FP16_TO_FP32(y[i]));
    }
}

// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_dot_f16_unroll(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା xs, ଦଶମିକ * GGML_RESTRICT s, ଶୂନ୍ୟ * GGML_RESTRICT xv, ggml_fp16_t * GGML_RESTRICT y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * GGML_RESTRICT x[GGML_VEC_DOT_UNROLL];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((ଅକ୍ଷର *) xv + i*xs);
    }

#ଯଦି defined(GGML_SIMD)
    #ଯଦି defined(__ARM_FEATURE_SVE)

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f16_epr = svcnth();
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f16_step = 2 * ggml_f16_epr;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = n - (n % ggml_f16_step);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା np2 = n - (n % ggml_f16_epr);

        svfloat32_t sum_0_0_lo = svdup_n_f32(0.0f);
        svfloat32_t sum_0_0_hi = svdup_n_f32(0.0f);
        svfloat32_t sum_0_1_lo = svdup_n_f32(0.0f);
        svfloat32_t sum_0_1_hi = svdup_n_f32(0.0f);
        svfloat32_t sum_1_0_lo = svdup_n_f32(0.0f);
        svfloat32_t sum_1_0_hi = svdup_n_f32(0.0f);
        svfloat32_t sum_1_1_lo = svdup_n_f32(0.0f);
        svfloat32_t sum_1_1_hi = svdup_n_f32(0.0f);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += ggml_f16_step) {
            ଅଚଳ svfloat16_t ay0 = GGML_F16x_VEC_LOAD(y + i, 0);
            ଅଚଳ svfloat16_t ax00 = GGML_F16x_VEC_LOAD(x[0] + i, 0);
            ଅଚଳ svfloat16_t ax01 = GGML_F16x_VEC_LOAD(x[1] + i, 0);

            ggml_sve_f16_fma_widened(&sum_0_0_lo, &sum_0_0_hi, ax00, ay0);
            ggml_sve_f16_fma_widened(&sum_1_0_lo, &sum_1_0_hi, ax01, ay0);

            ଅଚଳ svfloat16_t ay1 = GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 0);
            ଅଚଳ svfloat16_t ax10 = GGML_F16x_VEC_LOAD(x[0] + i + 1 * ggml_f16_epr, 0);
            ଅଚଳ svfloat16_t ax11 = GGML_F16x_VEC_LOAD(x[1] + i + 1 * ggml_f16_epr, 0);

            ggml_sve_f16_fma_widened(&sum_0_1_lo, &sum_0_1_hi, ax10, ay1);
            ggml_sve_f16_fma_widened(&sum_1_1_lo, &sum_1_1_hi, ax11, ay1);
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < np2; i += ggml_f16_epr) {
            ଅଚଳ svfloat16_t ry = GGML_F16x_VEC_LOAD(y + i, 0);
            ଅଚଳ svfloat16_t rx0 = GGML_F16x_VEC_LOAD(x[0] + i, 0);
            ଅଚଳ svfloat16_t rx1 = GGML_F16x_VEC_LOAD(x[1] + i, 0);

            ggml_sve_f16_fma_widened(&sum_0_0_lo, &sum_0_0_hi, rx0, ry);
            ggml_sve_f16_fma_widened(&sum_1_0_lo, &sum_1_0_hi, rx1, ry);
        }

        ଯଦି (np2 < n) {
            ଅଚଳ svbool_t pg = svwhilelt_b16(np2, n);
            ଅଚଳ svfloat16_t ay = svld1_f16(pg, (ଅଚଳ __fp16 *)(y + np2));
            ଅଚଳ svfloat16_t ax0 = svld1_f16(pg, (ଅଚଳ __fp16 *)(x[0] + np2));
            ଅଚଳ svfloat16_t ax1 = svld1_f16(pg, (ଅଚଳ __fp16 *)(x[1] + np2));

            ggml_sve_f16_fma_widened(&sum_0_0_lo, &sum_0_0_hi, ax0, ay);
            ggml_sve_f16_fma_widened(&sum_1_0_lo, &sum_1_0_hi, ax1, ay);
        }

        svfloat32_t sum_0_lo = svadd_f32_x(DEFAULT_PG32, sum_0_0_lo, sum_0_1_lo);
        svfloat32_t sum_0_hi = svadd_f32_x(DEFAULT_PG32, sum_0_0_hi, sum_0_1_hi);
        svfloat32_t sum_1_lo = svadd_f32_x(DEFAULT_PG32, sum_1_0_lo, sum_1_1_lo);
        svfloat32_t sum_1_hi = svadd_f32_x(DEFAULT_PG32, sum_1_0_hi, sum_1_1_hi);
        sumf[0] = ggml_sve_sum_f32x2(sum_0_lo, sum_0_hi);
        sumf[1] = ggml_sve_sum_f32x2(sum_1_lo, sum_1_hi);
        np = n;
    #ନଚେତଯଦି defined(__riscv_v_intrinsic)
        #ଯଦି defined(__riscv_zvfh)
            size_t vl = __riscv_vsetvlmax_e32m4();

            // initialize accumulators to all zeroes
            vfloat32m4_t vsum0_0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vsum0_1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vsum1_0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vsum1_1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            // calculate step size
            ଅଚଳ size_t epr = __riscv_vsetvlmax_e16m2();
            ଅଚଳ size_t step = epr * 2;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(step - 1));

            // unroll by 2 along the row dimension
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += step) {
                vfloat16m2_t ay0 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16 *)(y + i), epr);
                vfloat16m2_t ax0_0 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16 *)(x[0] + i), epr);
                vfloat16m2_t ax1_0 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16 *)(x[1] + i), epr);
                vsum0_0 = __riscv_vfwmacc_vv_f32m4(vsum0_0, ax0_0, ay0, epr);
                vsum1_0 = __riscv_vfwmacc_vv_f32m4(vsum1_0, ax1_0, ay0, epr);

                vfloat16m2_t ay1 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16 *)(y + i + epr), epr);
                vfloat16m2_t ax0_1 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16 *)(x[0] + i + epr), epr);
                vfloat16m2_t ax1_1 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16 *)(x[1] + i + epr), epr);
                vsum0_1 = __riscv_vfwmacc_vv_f32m4(vsum0_1, ax0_1, ay1, epr);
                vsum1_1 = __riscv_vfwmacc_vv_f32m4(vsum1_1, ax1_1, ay1, epr);
            }

            vfloat32m4_t vsum0 = __riscv_vfadd_vv_f32m4(vsum0_0, vsum0_1, vl);
            vfloat32m4_t vsum1 = __riscv_vfadd_vv_f32m4(vsum1_0, vsum1_1, vl);

            // leftovers
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; i += vl) {
                vl = __riscv_vsetvl_e16m2(n - i);
                vfloat16m2_t ay = __riscv_vle16_v_f16m2((ଅଚଳ _Float16 *)(y + i), vl);
                vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16 *)(x[0] + i), vl);
                vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((ଅଚଳ _Float16 *)(x[1] + i), vl);

                vsum0 = __riscv_vfwmacc_vv_f32m4(vsum0, ax0, ay, vl);
                vsum1 = __riscv_vfwmacc_vv_f32m4(vsum1, ax1, ay, vl);
            }

            // reduce
            vl = __riscv_vsetvlmax_e32m2();
            vfloat32m2_t acc0_0 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(vsum0, 0),
                                        __riscv_vget_v_f32m4_f32m2(vsum0, 1), vl);
            vl = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t acc0_1 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(acc0_0, 0),
            __riscv_vget_v_f32m2_f32m1(acc0_0, 1), vl);
            vfloat32m1_t redsum0 = __riscv_vfredusum_vs_f32m1_f32m1(
                                        acc0_1, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);

            vl = __riscv_vsetvlmax_e32m2();
            vfloat32m2_t acc1_0 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(vsum1, 0),
                                        __riscv_vget_v_f32m4_f32m2(vsum1, 1), vl);
            vl = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t acc1_1 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(acc1_0, 0),
                                        __riscv_vget_v_f32m2_f32m1(acc1_0, 1), vl);
            vfloat32m1_t redsum1 = __riscv_vfredusum_vs_f32m1_f32m1(
                                        acc1_1, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
            sumf[0] = __riscv_vfmv_f_s_f32m1_f32(redsum0);
            sumf[1] = __riscv_vfmv_f_s_f32m1_f32(redsum1);
            np = n;
        #ଅନ୍ୟଥା
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = 0;
        #endif
    #ଅନ୍ୟଥା
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(GGML_F16_DOT_STEP - 1));

        GGML_F16_DOT_VEC sum[GGML_VEC_DOT_UNROLL][GGML_F16_DOT_ARR] = { { GGML_F16_DOT_VEC_ZERO } };

        GGML_F16_DOT_VEC ax[GGML_F16_DOT_ARR];
        GGML_F16_DOT_VEC ay[GGML_F16_DOT_ARR];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += GGML_F16_DOT_STEP) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_F16_DOT_ARR; j++) {
                ay[j] = GGML_F16_DOT_VEC_LOAD(y + i + j*GGML_F16_DOT_EPR, j);

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
                    ax[j] = GGML_F16_DOT_VEC_LOAD(x[k] + i + j*GGML_F16_DOT_EPR, j);

                    sum[k][j] = GGML_F16_DOT_VEC_FMA(sum[k][j], ax[j], ay[j]);
                }
            }
        }

        // reduce sum0..sum3 to sum0
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
            GGML_F16_DOT_VEC_REDUCE(sumf[k], sum[k]);
        }
    #endif
#ଅନ୍ୟଥା
    // scalar path
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = 0;
#endif
    // scalar and leftovers
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; ++i) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[j][i])*GGML_CPU_FP16_TO_FP32(y[i]));
        }
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = (ଦଶମିକ)sumf[i];
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_mad_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * GGML_RESTRICT y, ଅଚଳ ଦଶମିକ * GGML_RESTRICT x, ଅଚଳ ଦଶମିକ v) {
#ଯଦି defined(GGML_SIMD)
    #ଯଦି defined(__ARM_FEATURE_SVE)

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f32_step = 8 * ggml_f32_epr; // choose 8 SVE registers
        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(ggml_f32_step - 1));
        svfloat32_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat32_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += ggml_f32_step) {

            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_FMA(ay1, ax1, vx);

            GGML_F32_VEC_STORE(y + i, ay1);

            ax2 = GGML_F32_VEC_LOAD(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_FMA(ay2, ax2, vx);

            GGML_F32_VEC_STORE(y + i + 1*ggml_f32_epr, ay2);

            ax3 = GGML_F32_VEC_LOAD(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_LOAD(y + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_FMA(ay3, ax3, vx);

            GGML_F32_VEC_STORE(y + i + 2*ggml_f32_epr, ay3);

            ax4 = GGML_F32_VEC_LOAD(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_LOAD(y + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_FMA(ay4, ax4, vx);

            GGML_F32_VEC_STORE(y + i + 3*ggml_f32_epr, ay4);

            ax5 = GGML_F32_VEC_LOAD(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_LOAD(y + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_FMA(ay5, ax5, vx);

            GGML_F32_VEC_STORE(y + i + 4*ggml_f32_epr, ay5);

            ax6 = GGML_F32_VEC_LOAD(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_LOAD(y + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_FMA(ay6, ax6, vx);

            GGML_F32_VEC_STORE(y + i + 5*ggml_f32_epr, ay6);

            ax7 = GGML_F32_VEC_LOAD(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_LOAD(y + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_FMA(ay7, ax7, vx);

            GGML_F32_VEC_STORE(y + i + 6*ggml_f32_epr, ay7);

            ax8 = GGML_F32_VEC_LOAD(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_LOAD(y + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_FMA(ay8, ax8, vx);

            GGML_F32_VEC_STORE(y + i + 7*ggml_f32_epr, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np2 = (n & ~(ggml_f32_epr - 1));
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_FMA(ay1, ax1, vx);

            GGML_F32_VEC_STORE(y + i, ay1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        ଯଦି (np2 < n) {
            svbool_t pg =svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            ay1 = svmad_f32_m(pg, ax1, vx, ay1);

            svst1_f32(pg, y + np2, ay1);
        }
    #ନଚେତଯଦି defined(__riscv_v_intrinsic)
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[i], avl);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(ax, v, ay, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #ଅନ୍ୟଥା
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += GGML_F32_STEP) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; ++i) {
            y[i] += x[i]*v;
        }
    #endif
#ଅନ୍ୟଥା
    // scalar
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_mad_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * GGML_RESTRICT y, ଅଚଳ ggml_fp16_t * GGML_RESTRICT x, ଅଚଳ ଦଶମିକ v) {
#ଯଦି defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sve_register_length = svcntb() * 8;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f16_epr = sve_register_length / 16;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f16_step = 8 * ggml_f16_epr;

    GGML_F16x_VEC vx = GGML_F16x_VEC_SET1(v);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(ggml_f16_step - 1));

    svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
    svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += ggml_f16_step) {
        ax1 = GGML_F16x_VEC_LOAD(x + i + 0 * ggml_f16_epr, 0);
        ay1 = GGML_F16x_VEC_LOAD(y + i + 0 * ggml_f16_epr, 0);
        ay1 = GGML_F16x_VEC_FMA(ay1, ax1, vx);

        GGML_F16x_VEC_STORE(y + i + 0 * ggml_f16_epr, ay1, 0);

        ax2 = GGML_F16x_VEC_LOAD(x + i + 1 * ggml_f16_epr, 1);
        ay2 = GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 1);
        ay2 = GGML_F16x_VEC_FMA(ay2, ax2, vx);

        GGML_F16x_VEC_STORE(y + i + 1 * ggml_f16_epr, ay2, 1);

        ax3 = GGML_F16x_VEC_LOAD(x + i + 2 * ggml_f16_epr, 2);
        ay3 = GGML_F16x_VEC_LOAD(y + i + 2 * ggml_f16_epr, 2);
        ay3 = GGML_F16x_VEC_FMA(ay3, ax3, vx);

        GGML_F16x_VEC_STORE(y + i + 2 * ggml_f16_epr, ay3, 2);

        ax4 = GGML_F16x_VEC_LOAD(x + i + 3 * ggml_f16_epr, 3);
        ay4 = GGML_F16x_VEC_LOAD(y + i + 3 * ggml_f16_epr, 3);
        ay4 = GGML_F16x_VEC_FMA(ay4, ax4, vx);

        GGML_F16x_VEC_STORE(y + i + 3 * ggml_f16_epr, ay4, 3);

        ax5 = GGML_F16x_VEC_LOAD(x + i + 4 * ggml_f16_epr, 4);
        ay5 = GGML_F16x_VEC_LOAD(y + i + 4 * ggml_f16_epr, 4);
        ay5 = GGML_F16x_VEC_FMA(ay5, ax5, vx);

        GGML_F16x_VEC_STORE(y + i + 4 * ggml_f16_epr, ay5, 4);

        ax6 = GGML_F16x_VEC_LOAD(x + i + 5 * ggml_f16_epr, 5);
        ay6 = GGML_F16x_VEC_LOAD(y + i + 5 * ggml_f16_epr, 5);
        ay6 = GGML_F16x_VEC_FMA(ay6, ax6, vx);

        GGML_F16x_VEC_STORE(y + i + 5 * ggml_f16_epr, ay6, 5);

        ax7 = GGML_F16x_VEC_LOAD(x + i + 6 * ggml_f16_epr, 6);
        ay7 = GGML_F16x_VEC_LOAD(y + i + 6 * ggml_f16_epr, 6);
        ay7 = GGML_F16x_VEC_FMA(ay7, ax7, vx);

        GGML_F16x_VEC_STORE(y + i + 6 * ggml_f16_epr, ay7, 6);

        ax8 = GGML_F16x_VEC_LOAD(x + i + 7 * ggml_f16_epr, 7);
        ay8 = GGML_F16x_VEC_LOAD(y + i + 7 * ggml_f16_epr, 7);
        ay8 = GGML_F16x_VEC_FMA(ay8, ax8, vx);

        GGML_F16x_VEC_STORE(y + i + 7 * ggml_f16_epr, ay8, 7);
    }
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np2 = (n & ~(ggml_f16_epr - 1));
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = np; k < np2; k += ggml_f16_epr) {
        svfloat16_t rx = GGML_F16x_VEC_LOAD(x + k, 0);
        svfloat16_t ry = GGML_F16x_VEC_LOAD(y + k, 0);
        ry = GGML_F16x_VEC_FMA(ry, rx, vx);

        GGML_F16x_VEC_STORE(y + k, ry, 0);
    }

    ଯଦି (np2 < n) {
        svbool_t pg = svwhilelt_b16(np2, n);
        svfloat16_t hx = svld1_f16(pg, (ଅଚଳ __fp16 *)(x + np2));
        svfloat16_t hy = svld1_f16(pg, (ଅଚଳ __fp16 *)(y + np2));
        hy = svmad_f16_x(pg, hx, vx, hy);
        svst1_f16(pg, (__fp16 *)(y + np2), hy);
    }
    np = n;
#ନଚେତଯଦି defined(__riscv_v_intrinsic) // implies __riscv_v_intrinsic
    #ଯଦି defined (__riscv_zvfh)
        ଅଚଳ ggml_fp16_t s = GGML_CPU_FP32_TO_FP16(v);
        ଅଚଳ _Float16 scale = *(ଅଚଳ _Float16*)(&s);

        // calculate step size
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା epr = __riscv_vsetvlmax_e16m4();
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = epr * 2;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(step - 1));

        // unroll by 2
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += step) {
            vfloat16m4_t ax0 = __riscv_vle16_v_f16m4((ଅଚଳ _Float16*)x + i, epr);
            vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((ଅଚଳ _Float16*)y + i, epr);
            ay0 = __riscv_vfmacc_vf_f16m4(ay0, scale, ax0, epr);
            __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, epr);
            __asm__ __volatile__ ("" ::: "memory");

            vfloat16m4_t ax1 = __riscv_vle16_v_f16m4((ଅଚଳ _Float16*)x + i + epr, epr);
            vfloat16m4_t ay1 = __riscv_vle16_v_f16m4((ଅଚଳ _Float16*)y + i + epr, epr);
            ay1 = __riscv_vfmacc_vf_f16m4(ay1, scale, ax1, epr);
            __riscv_vse16_v_f16m4((_Float16*)y + i + epr, ay1, epr);
            __asm__ __volatile__ ("" ::: "memory");
        }

        // leftovers
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା vl;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m4(n - i);
            vfloat16m4_t ax0 = __riscv_vle16_v_f16m4((ଅଚଳ _Float16*)x + i, vl);
            vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((ଅଚଳ _Float16*)y + i, vl);
            ay0 = __riscv_vfmacc_vf_f16m4(ay0, scale, ax0, vl);
            __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, vl);
        }
        np = n;
    #ଅନ୍ୟଥା
        // fall to scalar path
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = 0;
    #endif
#ନଚେତଯଦି defined(GGML_SIMD)
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += GGML_F16_STEP) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_FMA(ay[j], ax[j], vx);

            GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
        }
    }
#ଅନ୍ୟଥା
    // scalar path
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = 0;
#endif

    // scalar and leftovers
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i]) + GGML_CPU_FP16_TO_FP32(x[i])*v);
    }
}

// xs and vs are byte strides of x and v
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_mad_f32_unroll(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା xs, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା vs, ଦଶମିକ * GGML_RESTRICT y, ଅଚଳ ଦଶମିକ * GGML_RESTRICT xv, ଅଚଳ ଦଶମିକ * GGML_RESTRICT vv) {

    ଅଚଳ ଦଶମିକ * GGML_RESTRICT x[GGML_VEC_MAD_UNROLL];
    ଅଚଳ ଦଶମିକ * GGML_RESTRICT v[GGML_VEC_MAD_UNROLL];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_VEC_MAD_UNROLL; ++i) {
        x[i] = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) xv + i*xs);
        v[i] = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) vv + i*vs);
    }

#ଯଦି defined(GGML_SIMD)
    #ଯଦି defined(__ARM_FEATURE_SVE)
        // scalar Route to scalar implementation       //TODO: Write SVE code
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
                y[i] += x[k][i]*v[k][0];
            }
        }
    #ନଚେତଯଦି defined(__riscv_v_intrinsic)
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < GGML_VEC_MAD_UNROLL; k++) {
                vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[k][i], avl);
                ay = __riscv_vfmadd_vf_f32m8(ax, v[k][0], ay, avl);
            }
            __riscv_vse32_v_f32m8(&y[i], ay, avl);
        }
    #ଅନ୍ୟଥା
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx[GGML_VEC_MAD_UNROLL];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            vx[k] = GGML_F32_VEC_SET1(v[k][0]);
        }

        GGML_F32_VEC ax[GGML_VEC_MAD_UNROLL][GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += GGML_F32_STEP) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
                    ax[k][j] = GGML_F32_VEC_LOAD(x[k] + i + j*GGML_F32_EPR);
                    ay[j] = GGML_F32_VEC_FMA(ay[j], ax[k][j], vx[k]);
                }

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; ++i) {
                y[i] += x[k][i]*v[k][0];
            }
        }
    #endif
#ଅନ୍ୟଥା
    // scalar
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#endif
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_mad1_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ s, ଅଚଳ ଦଶମିକ b) {
#ଯଦି defined(GGML_USE_ACCELERATE)
    vDSP_vsmsa(x, 1, &s, &b, y, 1, n);
#ନଚେତଯଦି defined(GGML_SIMD)
    #ଯଦି defined(__ARM_FEATURE_SVE)
        // scalar ; TODO: Write SVE code
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
            y[i] = x[i]*s + b;
        }
    #ନଚେତଯଦି defined(__riscv_v_intrinsic)
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[i], avl);
            vfloat32m8_t vb = __riscv_vfmv_v_f_f32m8(b, avl);
            vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(ax, s, vb, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #ଅନ୍ୟଥା
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vs = GGML_F32_VEC_SET1(s);
        GGML_F32_VEC vb = GGML_F32_VEC_SET1(b);

        GGML_F32_VEC ay[GGML_F32_ARR];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += GGML_F32_STEP) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_FMA(vb, ay[j], vs);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; ++i) {
            y[i] = x[i]*s + b;
        }
    #endif
#ଅନ୍ୟଥା
    // scalar
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = x[i]*s + b;
    }
#endif
}

//inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_scale_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ   v) {
#ଯଦି defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#ନଚେତଯଦି defined(GGML_SIMD)
    #ଯଦି defined(__ARM_FEATURE_SVE)
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f32_step = 2 * ggml_f32_epr;

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(ggml_f32_step - 1));
        svfloat32_t ay1;
        svfloat32_t ay2;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += ggml_f32_step) {
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_MUL(ay1, vx);
            GGML_F32_VEC_STORE(y + i, ay1);

            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_MUL(ay2, vx);
            GGML_F32_VEC_STORE(y + i + 1*ggml_f32_epr, ay2);
        }
        // leftovers
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; i += ggml_f32_epr) {
            svbool_t pg = svwhilelt_b32(i, n);
            ay1 = svld1_f32(pg, y + i);
            ay1 = svmul_f32_m(pg, ay1, vx);
            svst1_f32(pg, y + i, ay1);
        }
    #ନଚେତଯଦି defined(__riscv_v_intrinsic)
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(ay, v, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #ଅନ୍ୟଥା
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        GGML_F32_VEC ay[GGML_F32_ARR];

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += GGML_F32_STEP) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; ++i) {
            y[i] *= v;
        }
    #endif
#ଅନ୍ୟଥା
    // scalar
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_scale_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ଦଶମିକ v) {
#ଯଦି defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sve_register_length = svcntb() * 8;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f16_epr = sve_register_length / 16;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_f16_step = 2 * ggml_f16_epr;

    GGML_F16x_VEC vx =  GGML_F16x_VEC_SET1(v);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(ggml_f16_step - 1));
    svfloat16_t ay1, ay2;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += ggml_f16_step) {
        ay1 = GGML_F16x_VEC_LOAD(y + i + 0*ggml_f16_epr, 0);
        ay1 = GGML_F16x_VEC_MUL(ay1, vx);
        GGML_F16x_VEC_STORE(y + i + 0*ggml_f16_epr, ay1, 0);

        ay2 = GGML_F16x_VEC_LOAD(y + i + 1*ggml_f16_epr, 1);
        ay2 = GGML_F16x_VEC_MUL(ay2, vx);
        GGML_F16x_VEC_STORE(y + i + 1*ggml_f16_epr, ay2, 1);
    }
    // leftovers
    // maximum number of leftover elements will be less that ggmlF_16x_epr. Apply predicated svmad on available elements only
    ଯଦି (np < n) {
        svbool_t pg = svwhilelt_b16(np, n);
        svfloat16_t hy = svld1_f16(pg, (__fp16 *)(y + np));
        svfloat16_t out = svmul_f16_m(pg, hy, vx);
        svst1_f16(pg, (__fp16 *)(y + np), out);
    }
    np = n;
#ନଚେତଯଦି defined(__riscv_v_intrinsic)
    #ଯଦି defined(__riscv_zvfh)
        ଅଚଳ ggml_fp16_t s = GGML_CPU_FP32_TO_FP16(v);
        ଅଚଳ _Float16 scale = *(ଅଚଳ _Float16*)(&s);

        // calculate step size
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା epr = __riscv_vsetvlmax_e16m4();
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା step = epr * 2;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(step - 1));

        // unroll by 2
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += step) {
            vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((ଅଚଳ _Float16*)y + i, epr);
            ay0 = __riscv_vfmul_vf_f16m4(ay0, scale, epr);
            __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, epr);
            __asm__ __volatile__ ("" ::: "memory");

            vfloat16m4_t ay1 = __riscv_vle16_v_f16m4((ଅଚଳ _Float16*)y + i + epr, epr);
            ay1 = __riscv_vfmul_vf_f16m4(ay1, scale, epr);
            __riscv_vse16_v_f16m4((_Float16*)y + i + epr, ay1, epr);
            __asm__ __volatile__ ("" ::: "memory");
        }

        // leftovers
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା vl;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m4(n - i);
            vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((ଅଚଳ _Float16*)y + i, vl);
            ay0 = __riscv_vfmul_vf_f16m4(ay0, scale, vl);
            __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, vl);
        }
        np = n;
    #ଅନ୍ୟଥା
        // fall to scalar path
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = 0;
    #endif
#ନଚେତଯଦି defined(GGML_SIMD)
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

    GGML_F16_VEC ay[GGML_F16_ARR];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < np; i += GGML_F16_STEP) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_F16_ARR; j++) {
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_MUL(ay[j], vx);

            GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
        }
    }
#ଅନ୍ୟଥା
    // scalar path
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା np = 0;
#endif
    // scalar and leftovers
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = np; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i])*v);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_norm_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * s, ଅଚଳ ଦଶମିକ * x) { ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);   }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sqr_f32  (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sqr_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v*v);
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sqrt_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sqrt_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sqrtf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_log_f32  (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = logf(x[i]);  }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_log_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(logf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sin_f32  (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = sinf(x[i]);  }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sin_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sinf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_cos_f32  (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = cosf(x[i]);  }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_cos_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(cosf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_abs_f32  (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_abs_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(fabsf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sgn_f32  (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sgn_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_step_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_step_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16((GGML_CPU_FP16_TO_FP32(x[i]) > 0.f) ? 1.f : 0.f);
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_tanh_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = tanhf(x[i]);  }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_tanh_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(tanhf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_elu_f32  (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : expm1f(x[i]); }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_elu_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଅଚଳ ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v : expm1f(v));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_relu_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_relu_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v : 0.f);
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_leaky_relu_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ ns) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = ((x[i] > 0.f) ? x[i] : 0.f) + ns * ((x[i] < 0.0f) ? x[i] : 0.f); }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_leaky_relu_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ଦଶମିକ ns) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(((v > 0.f) ? v : 0.f) + ns * ((v < 0.0f) ? v : 0.f));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sigmoid_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sigmoid_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(1.f / (1.f + expf(-GGML_CPU_FP16_TO_FP32(x[i]))));
    }
}
// TODO: optimize performance
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_hardswish_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_hardswish_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v * fminf(1.0f, fmaxf(0.0f, (v + 3.0f) / 6.0f)));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_hardsigmoid_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_hardsigmoid_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(fminf(1.0f, fmaxf(0.0f, (GGML_CPU_FP16_TO_FP32(x[i]) + 3.0f) / 6.0f)));
    }
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_exp_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) { ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) y[i] = expf(x[i]); }
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_exp_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(expf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}

ସ୍ଥିର ଅଚଳ ଦଶମିକ GELU_COEF_A     = 0.044715f;
ସ୍ଥିର ଅଚଳ ଦଶମିକ GELU_QUICK_COEF = -1.702f;
ସ୍ଥିର ଅଚଳ ଦଶମିକ SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
ସ୍ଥିର ଅଚଳ ଦଶମିକ SQRT_2_INV      = 0.70710678118654752440084436210484f;

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଦଶମିକ ggml_gelu_f32(ଦଶମିକ x) {
    ଫେରନ୍ତୁ 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ଅଚଳ uint16_t * i16 = (ଅଚଳ uint16_t *) x;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = ggml_table_gelu_f16[i16[i]];
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * y, ଅଚଳ ggml_bf16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(ggml_gelu_f32(GGML_BF16_TO_FP32(x[i])));
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_erf_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ xi = GGML_CPU_FP16_TO_FP32(x[i]);
        ଦଶମିକ res = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
        y[i] = GGML_CPU_FP32_TO_FP16(res);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_erf_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * y, ଅଚଳ ggml_bf16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ xi = GGML_BF16_TO_FP32(x[i]);
        y[i] = GGML_FP32_TO_BF16(0.5f*xi*(1.0f + erff(xi*SQRT_2_INV)));
    }
}

#ifdef GGML_GELU_FP16
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) {
    uint16_t t;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଯଦି (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } ଅନ୍ୟଥା ଯଦି (x[i] >= 10.0f) {
            y[i] = x[i];
        } ଅନ୍ୟଥା {
            ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, ଆକାର(uint16_t));
            y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]);
        }
    }
}
#ଅନ୍ୟଥା
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]);
    }
}
#endif

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_erf_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ xi = x[i];
        y[i] = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଦଶମିକ ggml_gelu_quick_f32(ଦଶମିକ x) {
    ଫେରନ୍ତୁ x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)));
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_quick_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ଅଚଳ uint16_t * i16 = (ଅଚଳ uint16_t *) x;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = ggml_table_gelu_quick_f16[i16[i]];
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_quick_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * y, ଅଚଳ ggml_bf16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(ggml_gelu_quick_f32(GGML_BF16_TO_FP32(x[i])));
    }
}

#ifdef GGML_GELU_QUICK_FP16
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_gelu_quick_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) {
    uint16_t t;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, ଆକାର(uint16_t));
        y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[t]);
    }
}
#ଅନ୍ୟଥା
ଶୂନ୍ୟ ggml_vec_gelu_quick_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x);
#endif

// Sigmoid Linear Unit (SiLU) function
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଦଶମିକ ggml_silu_f32(ଦଶମିକ x) {
    ଫେରନ୍ତୁ x/(1.0f + expf(-x));
}
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_fp16_t ggml_silu_f16(ggml_fp16_t x) {
    ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x);
    ଫେରନ୍ତୁ GGML_CPU_FP32_TO_FP16(v/(1.0f + expf(-v)));
}

#ଯଦି __FINITE_MATH_ONLY__
#error "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
#error "ref: https://github.com/ggml-org/llama.cpp/pull/7154#issuecomment-2143844461"
#endif

/* Below function was borrowed from the GitHub repository:
https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/common.hpp */
#ଯଦି defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    ଧାଡ଼ିଭିତରେ ସ୍ଥିର svfloat32_t exp_ps_sve(svbool_t pg, svfloat32_t src) {
        // Constants
        ଅଚଳ svfloat32_t log2_e = svdup_n_f32(1.4426950409f);
        ଅଚଳ svfloat32_t ln2 = svdup_n_f32(0.6931473921f);
        ଅଚଳ svfloat32_t half_ln2_sq = svdup_n_f32(0.2413862043f);
        ଅଚଳ svuint32_t not_mask17 = svdup_n_u32(~((1u << 17) - 1));
        ଅଚଳ svfloat32_t one = svdup_n_f32(1.0f);
        ଅଚଳ svfloat32_t inactive1 = svdup_n_f32(0.0f);
        ଅଚଳ svint32_t inactive2 = svdup_n_s32(0);

        // Algorithm starts here
        svfloat32_t t0 = svmul_f32_m(pg, src, log2_e);  // y = x * log2(e)
        svfloat32_t t1 = svrintm_f32_m(inactive1, pg, t0);         // rount to int (float)
        svint32_t t2 = svcvt_s32_f32_m(inactive2, pg, t1);         // n

        t1 = svsub_f32_m(pg, t0, t1);   // a = y - floor(y)
        t1 = svadd_f32_m(pg, t1, one);  // b = a + 1

        svuint32_t t3 = svlsr_n_u32_m(pg, svreinterpret_u32_f32(t1), 17);  // v = b >> 17 (u32)
        svfloat32_t t4 = svexpa_f32(t3);                                   // c = fexpa(v)
        t4 = svscale_f32_m(pg, t4, t2);                                    // fexpa(v) * 2^(n)

        // and_(t2.d, t1.d, not_mask17.d)
        svfloat32_t t5 = svreinterpret_f32_u32(svand_u32_m(pg, svreinterpret_u32_f32(t1), not_mask17));
        t5 = svsub_f32_m(pg, t1, t5);                // z
        t0 = svmla_f32_m(pg, ln2, t5, half_ln2_sq);  // ln2 + half_ln2_sq * z
        t0 = svmla_f32_m(pg, one, t5, t0);           // 1 + (ln2 * z) + (half_ln2_sq * z * z)
        t0 = svmul_f32_m(pg, t0, t4);                // Final result

        ଫେରନ୍ତୁ t0;
    }
#endif

#ଯଦି defined(__ARM_FEATURE_SVE) && defined(__aarch64__)

ଧାଡ଼ିଭିତରେ ସ୍ଥିର svfloat32_t ggml_v_expf(svbool_t pg, svfloat32_t x) {
    ଅଚଳ svfloat32_t r = svdup_n_f32_x(pg, 0x1.8p23f);
    ଅଚଳ svfloat32_t z = svmla_n_f32_x(pg, r, x, 0x1.715476p+0f);
    ଅଚଳ svfloat32_t n = svsub_f32_x(pg, z, r);
    ଅଚଳ svfloat32_t b = svmls_n_f32_x(pg, svmls_n_f32_x(pg, x, n, 0x1.62e4p-1f), n, 0x1.7f7d1cp-20f);
    ଅଚଳ svuint32_t e = svlsl_n_u32_x(pg, svreinterpret_u32_f32(z), 23);
    ଅଚଳ svfloat32_t k = svreinterpret_f32_u32(svadd_u32_x(pg, e, svreinterpret_u32_f32(svdup_n_f32_x(pg, 1))));
    ଅଚଳ svbool_t c = svacgt_n_f32(pg, n, 126);
    ଅଚଳ svfloat32_t u = svmul_f32_x(pg, b, b);
    ଅଚଳ svfloat32_t j = svmla_f32_x(pg,
        svmul_n_f32_x(pg, b, 0x1.ffffecp-1f),
        svmla_f32_x(pg, svmla_f32_x(pg, svdup_n_f32_x(pg, 0x1.fffdb6p-2f), svdup_n_f32_x(pg, 0x1.555e66p-3f), b),
                        svmla_f32_x(pg, svdup_n_f32_x(pg, 0x1.573e2ep-5f), svdup_n_f32_x(pg, 0x1.0e4020p-7f), b), u), u);
    ଅଚଳ svuint32_t d = svdup_n_u32_z(svcmple_n_f32(pg, n, 0.0), 0x82000000);
    ଅଚଳ svfloat32_t s1 = svreinterpret_f32_u32(svadd_n_u32_x(pg, d, 0x7f000000));
    ଅଚଳ svfloat32_t s2 = svreinterpret_f32_u32(svsub_u32_x(pg, e, d));
    ଫେରନ୍ତୁ svsel_f32(svacgt_f32(pg, n, svdup_n_f32_x(pg, 192)), svmul_f32_x(pg, s1, s1),
                     svsel_f32(c, svmul_f32_x(pg, svmla_f32_x(pg, s2, s2, j), s1), svmla_f32_x(pg, k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
ଧାଡ଼ିଭିତରେ ସ୍ଥିର svfloat32_t ggml_v_silu(svbool_t pg, svfloat32_t x) {
    ଅଚଳ svfloat32_t one = svdup_n_f32_x(pg, 1.0f);
    ଅଚଳ svfloat32_t zero = svdup_n_f32_x(pg, 0.0f);
    ଅଚଳ svfloat32_t neg_x = svsub_f32_x(pg, zero, x);
    ଅଚଳ svfloat32_t exp_neg_x = ggml_v_expf(pg, neg_x);
    ଅଚଳ svfloat32_t one_plus_exp_neg_x = svadd_f32_x(pg, one, exp_neg_x);
    ଫେରନ୍ତୁ svdiv_f32_x(pg, x, one_plus_exp_neg_x);
}

// gelu_quick(x) = x / (1 + exp(-1.702 x)); x stays the numerator so it cannot overflow
ଧାଡ଼ିଭିତରେ ସ୍ଥିର svfloat32_t ggml_v_gelu_quick(svbool_t pg, svfloat32_t x) {
    ଅଚଳ svfloat32_t one = svdup_n_f32_x(pg, 1.0f);
    ଅଚଳ svfloat32_t exp_cx = ggml_v_expf(pg, svmul_n_f32_x(pg, x, GELU_QUICK_COEF));
    ଫେରନ୍ତୁ svdiv_f32_x(pg, x, svadd_f32_x(pg, one, exp_cx));
}

#ନଚେତଯଦି defined(__ARM_NEON) && defined(__aarch64__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
ଧାଡ଼ିଭିତରେ ସ୍ଥିର float32x4_t ggml_v_expf(float32x4_t x) {
    ଅଚଳ float32x4_t r = vdupq_n_f32(0x1.8p23f);
    ଅଚଳ float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    ଅଚଳ float32x4_t n = vsubq_f32(z, r);
    ଅଚଳ float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    ଅଚଳ uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    ଅଚଳ float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    ଅଚଳ uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    ଅଚଳ float32x4_t u = vmulq_f32(b, b);
    ଅଚଳ float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    ଯଦି (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        ଫେରନ୍ତୁ vfmaq_f32(k, j, k);
    ଅଚଳ uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    ଅଚଳ float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    ଅଚଳ float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    ଫେରନ୍ତୁ vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
ଧାଡ଼ିଭିତରେ ସ୍ଥିର float32x4_t ggml_v_silu(float32x4_t x) {
    ଅଚଳ float32x4_t one = vdupq_n_f32(1.0f);
    ଅଚଳ float32x4_t zero = vdupq_n_f32(0.0f);
    ଅଚଳ float32x4_t neg_x = vsubq_f32(zero, x);
    ଅଚଳ float32x4_t exp_neg_x = ggml_v_expf(neg_x);
    ଅଚଳ float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    ଫେରନ୍ତୁ vdivq_f32(x, one_plus_exp_neg_x);
}

// gelu_quick(x) = x / (1 + exp(-1.702 x)); x stays the numerator so it cannot overflow
ଧାଡ଼ିଭିତରେ ସ୍ଥିର float32x4_t ggml_v_gelu_quick(float32x4_t x) {
    ଅଚଳ float32x4_t one = vdupq_n_f32(1.0f);
    ଅଚଳ float32x4_t exp_cx = ggml_v_expf(vmulq_n_f32(x, GELU_QUICK_COEF));
    ଫେରନ୍ତୁ vdivq_f32(x, vaddq_f32(one, exp_cx));
}

#ନଚେତଯଦି defined(__AVX512F__) && defined(__AVX512DQ__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
ଧାଡ଼ିଭିତରେ ସ୍ଥିର __m512 ggml_v_expf(__m512 x) {
  ଅଚଳ __m512 r = _mm512_set1_ps(0x1.8p23f);
  ଅଚଳ __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  ଅଚଳ __m512 n = _mm512_sub_ps(z, r);
  ଅଚଳ __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                       _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  ଅଚଳ __mmask16 d =
      _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  ଅଚଳ __m512 u = _mm512_mul_ps(b, b);
  ଅଚଳ __m512 j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                      _mm512_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                      _mm512_set1_ps(0x1.fffdb6p-2f))),
      u,
      _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  ଅଚଳ __m512 res = _mm512_scalef_ps(j, n);
  ଯଦି (_mm512_kortestz(d, d))
    ଫେରନ୍ତୁ res;
  ଅଚଳ __m512 zero = _mm512_setzero_ps();
  ଅଚଳ __m512 alt = _mm512_mask_blend_ps(
      _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  ଫେରନ୍ତୁ _mm512_mask_blend_ps(d, res, alt);
}

// computes silu x/(1+exp(-x)) in single precision vector
ଧାଡ଼ିଭିତରେ ସ୍ଥିର __m512 ggml_v_silu(__m512 x) {
    ଅଚଳ __m512 one = _mm512_set1_ps(1);
    ଅଚଳ __m512 zero = _mm512_setzero_ps();
    ଅଚଳ __m512 neg_x = _mm512_sub_ps(zero, x);
    ଅଚଳ __m512 exp_neg_x = ggml_v_expf(neg_x);
    ଅଚଳ __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    ଫେରନ୍ତୁ _mm512_div_ps(x, one_plus_exp_neg_x);
}

// gelu_quick(x) = x / (1 + exp(-1.702 x)); x stays the numerator so it cannot overflow
ଧାଡ଼ିଭିତରେ ସ୍ଥିର __m512 ggml_v_gelu_quick(__m512 x) {
    ଅଚଳ __m512 one = _mm512_set1_ps(1);
    ଅଚଳ __m512 exp_cx = ggml_v_expf(_mm512_mul_ps(x, _mm512_set1_ps(GELU_QUICK_COEF)));
    ଫେରନ୍ତୁ _mm512_div_ps(x, _mm512_add_ps(one, exp_cx));
}

#ନଚେତଯଦି defined(__AVX2__) && defined(__FMA__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
ଧାଡ଼ିଭିତରେ ସ୍ଥିର __m256 ggml_v_expf(__m256 x) {
  ଅଚଳ __m256 r = _mm256_set1_ps(0x1.8p23f);
  ଅଚଳ __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  ଅଚଳ __m256 n = _mm256_sub_ps(z, r);
  ଅଚଳ __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  ଅଚଳ __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  ଅଚଳ __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  ଅଚଳ __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  ଅଚଳ __m256 u = _mm256_mul_ps(b, b);
  ଅଚଳ __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  ଯଦି (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    ଫେରନ୍ତୁ _mm256_fmadd_ps(j, k, k);
  ଅଚଳ __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  ଅଚଳ __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  ଅଚଳ __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  ଅଚଳ __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  ଫେରନ୍ତୁ _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
ଧାଡ଼ିଭିତରେ ସ୍ଥିର __m256 ggml_v_silu(__m256 x) {
    ଅଚଳ __m256 one = _mm256_set1_ps(1);
    ଅଚଳ __m256 zero = _mm256_setzero_ps();
    ଅଚଳ __m256 neg_x = _mm256_sub_ps(zero, x);
    ଅଚଳ __m256 exp_neg_x = ggml_v_expf(neg_x);
    ଅଚଳ __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    ଫେରନ୍ତୁ _mm256_div_ps(x, one_plus_exp_neg_x);
}

// gelu_quick(x) = x / (1 + exp(-1.702 x)); x stays the numerator so it cannot overflow
ଧାଡ଼ିଭିତରେ ସ୍ଥିର __m256 ggml_v_gelu_quick(__m256 x) {
    ଅଚଳ __m256 one = _mm256_set1_ps(1);
    ଅଚଳ __m256 exp_cx = ggml_v_expf(_mm256_mul_ps(x, _mm256_set1_ps(GELU_QUICK_COEF)));
    ଫେରନ୍ତୁ _mm256_div_ps(x, _mm256_add_ps(one, exp_cx));
}

#ନଚେତଯଦି defined(__SSE2__) // __AVX2__ / __ARM_NEON

#ଯଦି defined(__FMA__)
#define MADD128(x, y, z) _mm_fmadd_ps(x, y, z)
#define NMADD128(x, y, z) _mm_fnmadd_ps(x, y, z)
#ଅନ୍ୟଥା
#define MADD128(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#define NMADD128(x, y, z) _mm_sub_ps(z, _mm_mul_ps(x, y))
#endif

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
ଧାଡ଼ିଭିତରେ ସ୍ଥିର __m128 ggml_v_expf(__m128 x) {
    ଅଚଳ __m128 r = _mm_set1_ps(0x1.8p23f);
    ଅଚଳ __m128 z = MADD128(x, _mm_set1_ps(0x1.715476p+0f), r);
    ଅଚଳ __m128 n = _mm_sub_ps(z, r);
    ଅଚଳ __m128 b =
        NMADD128(n, _mm_set1_ps(0x1.7f7d1cp-20f), NMADD128(n, _mm_set1_ps(0x1.62e4p-1f), x));
    ଅଚଳ __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    ଅଚଳ __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    ଅଚଳ __m128i c =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    ଅଚଳ __m128 u = _mm_mul_ps(b, b);
    ଅଚଳ __m128 j =
        MADD128(MADD128(MADD128(_mm_set1_ps(0x1.0e4020p-7f), b, _mm_set1_ps(0x1.573e2ep-5f)), u,
                        MADD128(_mm_set1_ps(0x1.555e66p-3f), b, _mm_set1_ps(0x1.fffdb6p-2f))),
                u, _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    ଯଦି (!_mm_movemask_epi8(c))
        ଫେରନ୍ତୁ MADD128(j, k, k);
    ଅଚଳ __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),
                                    _mm_set1_epi32(0x82000000u));
    ଅଚଳ __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
    ଅଚଳ __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    ଅଚଳ __m128i d =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    ଫେରନ୍ତୁ _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(_mm_castsi128_ps(d),
                      _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(MADD128(s2, j, s2), s1)),
                                _mm_andnot_ps(_mm_castsi128_ps(c), MADD128(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
ଧାଡ଼ିଭିତରେ ସ୍ଥିର __m128 ggml_v_silu(__m128 x) {
    ଅଚଳ __m128 one = _mm_set1_ps(1);
    ଅଚଳ __m128 zero = _mm_setzero_ps();
    ଅଚଳ __m128 neg_x = _mm_sub_ps(zero, x);
    ଅଚଳ __m128 exp_neg_x = ggml_v_expf(neg_x);
    ଅଚଳ __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
    ଫେରନ୍ତୁ _mm_div_ps(x, one_plus_exp_neg_x);
}

// gelu_quick(x) = x / (1 + exp(-1.702 x)); x stays the numerator so it cannot overflow
ଧାଡ଼ିଭିତରେ ସ୍ଥିର __m128 ggml_v_gelu_quick(__m128 x) {
    ଅଚଳ __m128 one = _mm_set1_ps(1);
    ଅଚଳ __m128 exp_cx = ggml_v_expf(_mm_mul_ps(x, _mm_set1_ps(GELU_QUICK_COEF)));
    ଫେରନ୍ତୁ _mm_div_ps(x, _mm_add_ps(one, exp_cx));
}

#ନଚେତଯଦି defined(__VXE__) || defined(__VXE2__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
ଧାଡ଼ିଭିତରେ ସ୍ଥିର float32x4_t ggml_v_expf(float32x4_t v_x) {
    ଅଚଳ float32x4_t v_r = vec_splats(0x1.8p23f);
    ଅଚଳ float32x4_t v_z = vec_madd(v_x, vec_splats(0x1.715476p+0f), v_r);
    ଅଚଳ float32x4_t v_n = vec_sub(v_z, v_r);
    ଅଚଳ float32x4_t v_b = vec_nmsub(v_n, vec_splats(0x1.7f7d1cp-20f),
                                      vec_nmsub(v_n, vec_splats(0x1.62e4p-1f), v_x));
    ଅଚଳ uint32x4_t  v_e = vec_sl((uint32x4_t)v_z, 23);
    ଅଚଳ float32x4_t v_k = (float32x4_t)vec_add(v_e, (uint32x4_t)vec_splats(1.0f));
    ଅଚଳ uint32x4_t  v_c = (uint32x4_t)vec_cmpgt(vec_abs(v_n), vec_splats(126.0f));
    ଅଚଳ float32x4_t v_u = vec_mul(v_b, v_b);
    ଅଚଳ float32x4_t v_j = vec_madd(
        vec_madd(vec_madd(vec_splats(0x1.0e4020p-7f), v_b, vec_splats(0x1.573e2ep-5f)), v_u,
                 vec_madd(vec_splats(0x1.555e66p-3f), v_b, vec_splats(0x1.fffdb6p-2f))),
        v_u, vec_mul(vec_splats(0x1.ffffecp-1f), v_b));
    ଯଦି (vec_all_eq(v_c, vec_splats(UINT32_C(0)))) {
        ଫେରନ୍ତୁ vec_madd(v_j, v_k, v_k);
    }
    ଅଚଳ uint32x4_t  v_d  = vec_and((uint32x4_t)vec_cmple(v_n, vec_splats(0.0f)), vec_splats(UINT32_C(0x82000000)));
    ଅଚଳ float32x4_t v_s1 = (float32x4_t)vec_add(v_d, vec_splats(UINT32_C(0x7f000000)));
    ଅଚଳ float32x4_t v_s2 = (float32x4_t)vec_sub(v_e, v_d);
    ଫେରନ୍ତୁ vec_sel(vec_sel(vec_madd(v_k, v_j, v_k), vec_mul(vec_madd(v_s2, v_j, v_s2), v_s1), v_c),
                   vec_mul(v_s1, v_s1), (uint32x4_t)vec_cmpgt(vec_abs(v_n), vec_splats(192.0f)));
}

// computes silu x/(1+exp(-x)) in single precision vector
ଧାଡ଼ିଭିତରେ ସ୍ଥିର float32x4_t ggml_v_silu(float32x4_t v_x) {
    ଅଚଳ float32x4_t v_one = vec_splats(1.0f);
    ଅଚଳ float32x4_t v_zero = vec_splats(0.0f);
    ଅଚଳ float32x4_t v_neg_x = vec_sub(v_zero, v_x);
    ଅଚଳ float32x4_t v_exp_neg_x = ggml_v_expf(v_neg_x);
    ଅଚଳ float32x4_t v_one_plus_exp_neg_x = vec_add(v_one, v_exp_neg_x);
    ଫେରନ୍ତୁ vec_div(v_x, v_one_plus_exp_neg_x);
}

// gelu_quick(x) = x / (1 + exp(-1.702 x)); x stays the numerator so it cannot overflow
ଧାଡ଼ିଭିତରେ ସ୍ଥିର float32x4_t ggml_v_gelu_quick(float32x4_t x) {
    ଅଚଳ float32x4_t one = vec_splats(1.0f);
    ଅଚଳ float32x4_t exp_cx = ggml_v_expf(vec_mul(x, vec_splats(GELU_QUICK_COEF)));
    ଫେରନ୍ତୁ vec_div(x, vec_add(one, exp_cx));
}

#ନଚେତଯଦି defined(__riscv_v_intrinsic)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
ଧାଡ଼ିଭିତରେ ସ୍ଥିର vfloat32m2_t ggml_v_expf_m2(vfloat32m2_t x, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vl) {
    ଅଚଳ vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
#ifdef __riscv_xtheadvector
    // workaround for compiler bug (gcc 14.3.0: Error: unrecognized opcode `th.vmv1r.v v2,v4')
    vfloat32m2_t z = __riscv_vfadd_vf_f32m2(r, 0.0f, vl);
    z = __riscv_vfmacc_vf_f32m2(z, 0x1.715476p+0f, x, vl);
#ଅନ୍ୟଥା
    ଅଚଳ vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
#endif
    ଅଚଳ vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    ଅଚଳ vfloat32m2_t b = __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl),
                                                    0x1.7f7d1cp-20f, n, vl);
    ଅଚଳ vuint32m2_t e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    ଅଚଳ vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl)); // 1.0f
    ଅଚଳ vbool16_t c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    ଅଚଳ vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    ଅଚଳ vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl),
            u, vl), u, vl);
    ଯଦି (!__riscv_vcpop_m_b16(c, vl))
        ଫେରନ୍ତୁ __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    ଅଚଳ vbool16_t  dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    ଅଚଳ vuint32m2_t d = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    ଅଚଳ vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    ଅଚଳ vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    ଅଚଳ vfloat32m2_t r1 = __riscv_vmerge_vvm_f32m2(
        __riscv_vfmacc_vv_f32m2(k, k, j, vl),
        __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl),
        c, vl);
    ଫେରନ୍ତୁ __riscv_vmerge_vvm_f32m2(
        r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
        __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl),
        vl);
}

// computes silu x/(1+exp(-x)) in single precision vector
ଧାଡ଼ିଭିତରେ ସ୍ଥିର vfloat32m2_t ggml_v_silu_m2(vfloat32m2_t x, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vl) {
    ଅଚଳ vfloat32m2_t neg_x = __riscv_vfneg_v_f32m2(x, vl);
    ଅଚଳ vfloat32m2_t exp_neg_x = ggml_v_expf_m2(neg_x, vl);
    ଅଚଳ vfloat32m2_t one_plus_exp_neg_x = __riscv_vfadd_vf_f32m2(exp_neg_x, 1.0f, vl);
    ଫେରନ୍ତୁ __riscv_vfdiv_vv_f32m2(x, one_plus_exp_neg_x, vl);
}

// gelu_quick(x) = x / (1 + exp(-1.702 x)); x stays the numerator so it cannot overflow
ଧାଡ଼ିଭିତରେ ସ୍ଥିର vfloat32m2_t ggml_v_gelu_quick_m2(vfloat32m2_t x, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vl) {
    ଅଚଳ vfloat32m2_t exp_cx = ggml_v_expf_m2(__riscv_vfmul_vf_f32m2(x, GELU_QUICK_COEF, vl), vl);
    ଫେରନ୍ତୁ __riscv_vfdiv_vv_f32m2(x, __riscv_vfadd_vf_f32m2(exp_cx, 1.0f, vl), vl);
}

#endif // __ARM_NEON / __AVX2__ / __SSE2__ / __VXE__ / __riscv_v_intrinsic

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_silu_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = ggml_silu_f16(x[i]);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_silu_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * y, ଅଚଳ ggml_bf16_t * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(ggml_silu_f32(GGML_BF16_TO_FP32(x[i])));
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଦଶମିକ ggml_silu_backward_f32(ଦଶମିକ x, ଦଶମିକ dy) {
    ଅଚଳ ଦଶମିକ s = 1.0f/(1.0f + expf(-x));
    ଫେରନ୍ତୁ dy*s*(1.0f + x*(1.0f - s));
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ggml_fp16_t ggml_silu_backward_f16(ggml_fp16_t x, ggml_fp16_t dy) {
    ଅଚଳ ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x);
    ଅଚଳ ଦଶମିକ s = 1.0f/(1.0f + expf(-v));
    ଫେରନ୍ତୁ GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(dy)*s*(1.0f + v*(1.0f - s)));
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_silu_backward_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * dx, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * dy) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        dx[i] = ggml_silu_backward_f32(x[i], dy[i]);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_silu_backward_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * dx, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * dy) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        dx[i] = ggml_silu_backward_f16(x[i], dy[i]);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_reglu_f32 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = (x[i] > 0.f) ? x[i] * g[i] : 0.f;
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_reglu_f16 (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v * GGML_CPU_FP16_TO_FP32(g[i]) : 0.f);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_reglu_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * y, ଅଚଳ ggml_bf16_t * x, ଅଚଳ ggml_bf16_t * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ v = GGML_BF16_TO_FP32(x[i]);
        y[i] = GGML_FP32_TO_BF16((v > 0.f) ? v * GGML_BF16_TO_FP32(g[i]) : 0.f);
    }
}

#ifdef GGML_GELU_FP16
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * g) {
    uint16_t t;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଯଦି (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } ଅନ୍ୟଥା ଯଦି (x[i] >= 10.0f) {
            y[i] = x[i] * g[i];
        } ଅନ୍ୟଥା {
            ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, ଆକାର(uint16_t));
            y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]) * g[i];
        }
    }
}
#ଅନ୍ୟଥା
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]) * g[i];
    }
}
#endif

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * g) {
    ଅଚଳ uint16_t * i16 = (ଅଚଳ uint16_t *) x;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[i16[i]]) * v);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * y, ଅଚଳ ggml_bf16_t * x, ଅଚଳ ggml_bf16_t * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(ggml_gelu_f32(GGML_BF16_TO_FP32(x[i])) * GGML_BF16_TO_FP32(g[i]));
    }
}

ଶୂନ୍ୟ ggml_vec_swiglu_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * g);

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_swiglu_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ xi = GGML_CPU_FP16_TO_FP32(x[i]);
        ଦଶମିକ gi = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((xi/(1.0f + expf(-xi))) * gi);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_swiglu_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * y, ଅଚଳ ggml_bf16_t * x, ଅଚଳ ggml_bf16_t * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ xi = GGML_BF16_TO_FP32(x[i]);
        ଦଶମିକ gi = GGML_BF16_TO_FP32(g[i]);
        y[i] = GGML_FP32_TO_BF16((xi/(1.0f + expf(-xi))) * gi);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_erf_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ xi = x[i];
        y[i] = 0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * g[i];
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_erf_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ xi = GGML_CPU_FP16_TO_FP32(x[i]);
        ଦଶମିକ gi = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * gi);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_erf_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * y, ଅଚଳ ggml_bf16_t * x, ଅଚଳ ggml_bf16_t * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ xi = GGML_BF16_TO_FP32(x[i]);
        ଦଶମିକ gi = GGML_BF16_TO_FP32(g[i]);
        y[i] = GGML_FP32_TO_BF16(0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * gi);
    }
}

#ifdef GGML_GELU_QUICK_FP16
ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_quick_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * g) {
    uint16_t t;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, ଆକାର(uint16_t));
        y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[t]) * g[i];
    }
}
#ଅନ୍ୟଥା
ଶୂନ୍ୟ ggml_vec_geglu_quick_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x, ଅଚଳ ଦଶମିକ * g);
#endif

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_quick_f16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_fp16_t * y, ଅଚଳ ggml_fp16_t * x, ଅଚଳ ggml_fp16_t * g) {
    ଅଚଳ uint16_t * i16 = (ଅଚଳ uint16_t *) x;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଦଶମିକ v = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[i16[i]]) * v);
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_geglu_quick_bf16(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_bf16_t * y, ଅଚଳ ggml_bf16_t * x, ଅଚଳ ggml_bf16_t * g) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(ggml_gelu_quick_f32(GGML_BF16_TO_FP32(x[i])) * GGML_BF16_TO_FP32(g[i]));
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sum_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * s, ଅଚଳ ଦଶମିକ * x) {
#ifndef GGML_USE_ACCELERATE
    ggml_float sum = 0.0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        sum += (ggml_float)x[i];
    }
    *s = (ଦଶମିକ)sum;
#ଅନ୍ୟଥା
    vDSP_sve(x, 1, s, n);
#endif
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_cumsum_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * y, ଅଚଳ ଦଶମିକ * x) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        ଯଦି (i == 0) {
            y[i] = x[i];
        } ଅନ୍ୟଥା {
            y[i] = y[i - 1] + x[i];
        }
    }
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sum_f32_ggf(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ggml_float * s, ଅଚଳ ଦଶମିକ * x) {
    ggml_float sum = 0.0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        sum += (ggml_float)x[i];
    }
    *s = sum;
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sum_f16_ggf(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * s, ଅଚଳ ggml_fp16_t * x) {
    ଦଶମିକ sum = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        sum += GGML_CPU_FP16_TO_FP32(x[i]);
    }
    *s = sum;
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_sum_bf16_ggf(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * s, ଅଚଳ ggml_bf16_t * x) {
    ଦଶମିକ sum = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        sum += GGML_BF16_TO_FP32(x[i]);
    }
    *s = sum;
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_max_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * s, ଅଚଳ ଦଶମିକ * x) {
#ifndef GGML_USE_ACCELERATE
    ଦଶମିକ max = -INFINITY;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
#ଅନ୍ୟଥା
    vDSP_maxv(x, 1, s, n);
#endif
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_norm_inv_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ * s, ଅଚଳ ଦଶମିକ * x) {
    ggml_vec_norm_f32(n, s, x);
    *s = 1.f/(*s);
}

ଧାଡ଼ିଭିତରେ ସ୍ଥିର ଶୂନ୍ୟ ggml_vec_argmax_f32(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା * s, ଅଚଳ ଦଶମିକ * x) {
    ଦଶମିକ max = -INFINITY;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
        ଯଦି (max == x[i]) { idx = i; }
    }
    *s = idx;
}

#ifdef __cplusplus
}
#endif
