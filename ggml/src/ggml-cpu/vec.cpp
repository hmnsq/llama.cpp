#అన్తర్భావ "vec.h"

#అన్తర్భావ <cassert>

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_పట్టిక_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_పట్టిక_gelu_శీఘ్ర_f16[1 << 16];

శూన్య ggml_vec_dot_f32(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి దశాంశం * GGML_RESTRICT x, size_t bx, స్థిరరాశి దశాంశం * GGML_RESTRICT y, size_t by, పూర్ణాంకం nrc) {
   ప్రతిజ్ఞా(nrc == 1);
   GGML_వాడని(nrc);
   GGML_వాడని(bx);
   GGML_వాడని(by);
   GGML_వాడని(bs);

#ఒకవేళ defined(GGML_SIMD)
    దశాంశం sumf = 0.0f;

    #ఒకవేళ defined(__ARM_FEATURE_SVE)
        స్థిరరాశి పూర్ణాంకం sve_రిజిస్టరు_నిడివి = ggml_cpu_get_sve_cnt() * 8;
        స్థిరరాశి పూర్ణాంకం ggml_f32_epr = sve_రిజిస్టరు_నిడివి / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        స్థిరరాశి పూర్ణాంకం ggml_f32_అడుగు = 8 * ggml_f32_epr; // choose 8 SVE registers

        స్థిరరాశి పూర్ణాంకం np = (n & ~(ggml_f32_అడుగు - 1));
        svfloat32_t sum1 = svdup_n_f32(0.0f);
        svfloat32_t sum2 = svdup_n_f32(0.0f);
        svfloat32_t sum3 = svdup_n_f32(0.0f);
        svfloat32_t sum4 = svdup_n_f32(0.0f);
        svfloat32_t sum5 = svdup_n_f32(0.0f);
        svfloat32_t sum6 = svdup_n_f32(0.0f);
        svfloat32_t sum7 = svdup_n_f32(0.0f);
        svfloat32_t sum8 = svdup_n_f32(0.0f);
        svfloat32_t ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8;
        svfloat32_t ay1,ay2,ay3,ay4,ay5,ay6,ay7,ay8;
        కొరకు (పూర్ణాంకం i = 0; i < np; i += ggml_f32_అడుగు) {
            ax1 = GGML_F32_VEC_ఎక్కించు(x + i);
            ay1 = GGML_F32_VEC_ఎక్కించు(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);

            ax2 = GGML_F32_VEC_ఎక్కించు(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_ఎక్కించు(y + i + 1*ggml_f32_epr);
            sum2 = GGML_F32_VEC_FMA(sum2, ax2, ay2);

            ax3 = GGML_F32_VEC_ఎక్కించు(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_ఎక్కించు(y + i + 2*ggml_f32_epr);
            sum3 = GGML_F32_VEC_FMA(sum3, ax3, ay3);

            ax4 = GGML_F32_VEC_ఎక్కించు(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_ఎక్కించు(y + i + 3*ggml_f32_epr);
            sum4 = GGML_F32_VEC_FMA(sum4, ax4, ay4);

            ax5 = GGML_F32_VEC_ఎక్కించు(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_ఎక్కించు(y + i + 4*ggml_f32_epr);
            sum5 = GGML_F32_VEC_FMA(sum5, ax5, ay5);

            ax6 = GGML_F32_VEC_ఎక్కించు(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_ఎక్కించు(y + i + 5*ggml_f32_epr);
            sum6 = GGML_F32_VEC_FMA(sum6, ax6, ay6);

            ax7 = GGML_F32_VEC_ఎక్కించు(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_ఎక్కించు(y + i + 6*ggml_f32_epr);
            sum7 = GGML_F32_VEC_FMA(sum7, ax7, ay7);

            ax8 = GGML_F32_VEC_ఎక్కించు(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_ఎక్కించు(y + i + 7*ggml_f32_epr);
            sum8 = GGML_F32_VEC_FMA(sum8, ax8, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        స్థిరరాశి పూర్ణాంకం np2 = (n & ~(ggml_f32_epr - 1));
        కొరకు (పూర్ణాంకం i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_ఎక్కించు(x + i);
            ay1 = GGML_F32_VEC_ఎక్కించు(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmla on available elements only
        ఒకవేళ (np2 < n) {
            svbool_t pg = svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            sum1 = svmla_f32_m(pg, sum1, ax1, ay1);
        }
        // reduce sum1,sum2 to sum1
        GGML_F32_VEC_REDUCE(sumf, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
    #లేదావేళ defined(__riscv_v_intrinsic)
        పూర్ణాంకం vl = __riscv_vsetvlmax_e32m8();
        vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m8_t vsum;
        vfloat32m8_t ax;
        vfloat32m8_t ay;
        vsum = __riscv_vfmv_v_f_f32m8_tu(vsum, 0.0f, vl);
        కొరకు (పూర్ణాంకం i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            ax = __riscv_vle32_v_f32m8_tu(ax, &x[i], vl);
            ay = __riscv_vle32_v_f32m8_tu(ay, &y[i], vl);
            vsum = __riscv_vfmacc_vv_f32m8_tu(vsum, ax, ay, vl);
        }
        vl = __riscv_vsetvlmax_e32m8();
        vs = __riscv_vfredusum_vs_f32m8_f32m1(vsum, vs, vl);
        sumf += __riscv_vfmv_f_s_f32m1_f32(vs);
    #లేకపోతే
        స్థిరరాశి పూర్ణాంకం np = (n & ~(GGML_F32_అడుగు - 1));

        GGML_F32_VEC కూడిక[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        కొరకు (పూర్ణాంకం i = 0; i < np; i += GGML_F32_అడుగు) {
            కొరకు (పూర్ణాంకం j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_ఎక్కించు(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_ఎక్కించు(y + i + j*GGML_F32_EPR);

                కూడిక[j] = GGML_F32_VEC_FMA(కూడిక[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F32_VEC_REDUCE(sumf, కూడిక);

        // leftovers
        కొరకు (పూర్ణాంకం i = np; i < n; ++i) {
            sumf += x[i]*y[i];
        }
    #అయితే_ముగింపు
#లేకపోతే
    // scalar
    ggml_float sumf = 0.0;
    కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#అయితే_ముగింపు

    *s = sumf;
}

శూన్య ggml_vec_dot_bf16(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(nrc == 1);
    GGML_వాడని(nrc);
    GGML_వాడని(bx);
    GGML_వాడని(by);
    GGML_వాడని(bs);
    పూర్ణాంకం i = 0;
    ggml_float sumf = 0;

#ఒకవేళ defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    కొరకు (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)mm512_reduce_కూడు_ps(c1);
    sumf += (ggml_float)mm512_reduce_కూడు_ps(c2);

#లేదావేళ defined(__AVX512F__)
#నిర్వచించు ఎక్కించు_2(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((స్థిరరాశి __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    కొరకు (; i + 32 <= n; i += 32) {
        c1 = mm512_కూడు_ps(_mm512_mul_ps(ఎక్కించు_2(x + i), ఎక్కించు_2(y + i)), c1);
        c2 = mm512_కూడు_ps(_mm512_mul_ps(ఎక్కించు_2(x + i + 16), ఎక్కించు_2(y + i + 16)), c2);
    }
    sumf += (ggml_float)mm512_reduce_కూడు_ps(c1);
    sumf += (ggml_float)mm512_reduce_కూడు_ps(c2);

#నిర్వచనం_రద్దు ఎక్కించు_2
#లేదావేళ defined(__AVX2__) || defined(__AVX__)
#ఒకవేళ defined(__AVX2__)
#నిర్వచించు ఎక్కించు_2(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((స్థిరరాశి __m128i *)(p))), 16))
#లేకపోతే
#నిర్వచించు ఎక్కించు_2(p) _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((స్థిరరాశి __m128i *)(p))), 16)), (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((స్థిరరాశి __m128i *)(p)), 8)), 16)), 1))
#అయితే_ముగింపు
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    కొరకు (; i + 32 <= n; i += 32) {
        c1 = mm256_కూడు_ps(_mm256_mul_ps(ఎక్కించు_2(x + i), ఎక్కించు_2(y + i)), c1);
        c2 = mm256_కూడు_ps(_mm256_mul_ps(ఎక్కించు_2(x + i + 8), ఎక్కించు_2(y + i + 8)), c2);
        c3 = mm256_కూడు_ps(_mm256_mul_ps(ఎక్కించు_2(x + i + 16), ఎక్కించు_2(y + i + 16)), c3);
        c4 = mm256_కూడు_ps(_mm256_mul_ps(ఎక్కించు_2(x + i + 24), ఎక్కించు_2(y + i + 24)), c4);
    }
    __m128 g;
    c1 = mm256_కూడు_ps(mm256_కూడు_ps(c1, c3),
                       mm256_కూడు_ps(c2, c4));
    g = mm_కూడు_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = mm_కూడు_ps(g, _mm_movehl_ps(g, g));
    g = mm_కూడు_ss(g, _mm_movehdup_ps(g));
    sumf += (ggml_float)_mm_cvtss_f32(g);

#నిర్వచనం_రద్దు ఎక్కించు_2
#లేదావేళ defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfwma)
    size_t vl = __riscv_vsetvlmax_e32m4();

    // initialize accumulators to all zeroes
    vfloat32m4_t vsum0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vsum1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    // calculate step size
    స్థిరరాశి size_t epr = __riscv_vsetvlmax_e16m2();
    స్థిరరాశి size_t అడుగు = epr * 2;
    స్థిరరాశి పూర్ణాంకం np = (n & ~(అడుగు - 1));

    // unroll by 2
    కొరకు (; i < np; i += అడుగు) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((స్థిరరాశి __bf16 *)&x[i], epr);
        vbfloat16m2_t ay0 = __riscv_vle16_v_bf16m2((స్థిరరాశి __bf16 *)&y[i], epr);
        vsum0 = __riscv_vfwmaccbf16_vv_f32m4(vsum0, ax0, ay0, epr);
        __asm__ __volatile__ ("" ::: "memory");

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((స్థిరరాశి __bf16 *)&x[i + epr], epr);
        vbfloat16m2_t ay1 = __riscv_vle16_v_bf16m2((స్థిరరాశి __bf16 *)&y[i + epr], epr);
        vsum1 = __riscv_vfwmaccbf16_vv_f32m4(vsum1, ax1, ay1, epr);
        __asm__ __volatile__ ("" ::: "memory");
    }

    // accumulate in 1 register
    vsum0 = __riscv_vfadd_vv_f32m4(vsum0, vsum1, vl);

    // leftovers
    కొరకు (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((స్థిరరాశి __bf16 *)&x[i], vl);
        vbfloat16m2_t ay0 = __riscv_vle16_v_bf16m2((స్థిరరాశి __bf16 *)&y[i], vl);
        vsum0 = __riscv_vfwmaccbf16_vv_f32m4(vsum0, ax0, ay0, vl);
    }

    // reduce
    vl = __riscv_vsetvlmax_e32m4();
    vfloat32m1_t redsum = __riscv_vfredusum_vs_f32m4_f32m1(vsum0, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
    sumf += __riscv_vfmv_f_s_f32m1_f32(redsum);

#లేదావేళ defined(__POWER9_VECTOR__) || defined(__VXE__) || defined(__VXE2__)
    స్థిరరాశి పూర్ణాంకం np = (n & ~(GGML_BF16_అడుగు - 1));
    ఒకవేళ (np > 0) {
        GGML_F32_VEC కూడిక[4] = {GGML_F32_VEC_ZERO};
        కొరకు (; i < np; i += GGML_BF16_అడుగు) {
            GGML_BF16_VEC vx0 = GGML_BF16_VEC_ఎక్కించు(x + i);
            GGML_BF16_VEC vx1 = GGML_BF16_VEC_ఎక్కించు(x + i + 8);
            GGML_BF16_VEC vy0 = GGML_BF16_VEC_ఎక్కించు(y + i);
            GGML_BF16_VEC vy1 = GGML_BF16_VEC_ఎక్కించు(y + i + 8);
            GGML_BF16_FMA_LO(కూడిక[0], vx0, vy0);
            GGML_BF16_FMA_HI(కూడిక[1], vx0, vy0);
            GGML_BF16_FMA_LO(కూడిక[2], vx1, vy1);
            GGML_BF16_FMA_HI(కూడిక[3], vx1, vy1);
        }
        GGML_F32x4_REDUCE_4(sumf, కూడిక[0], కూడిక[1], కూడిక[2], కూడిక[3]);
    }
#అయితే_ముగింపు

    కొరకు (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_గమ్యం_FP32(x[i]) *
                             GGML_BF16_గమ్యం_FP32(y[i]));
    }
    *s = sumf;
}

శూన్య ggml_vec_dot_f16(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(nrc == 1);
    GGML_వాడని(nrc);
    GGML_వాడని(bx);
    GGML_వాడని(by);
    GGML_వాడని(bs);

    ggml_float sumf = 0.0;


#ఒకవేళ defined(GGML_SIMD)
    #ఒకవేళ defined(__ARM_FEATURE_SVE)
        స్థిరరాశి పూర్ణాంకం ggml_f16_epr = svcnth();
        స్థిరరాశి పూర్ణాంకం ggml_f16_అడుగు = 8 * ggml_f16_epr;
        స్థిరరాశి పూర్ణాంకం np = n - (n % ggml_f16_అడుగు);
        స్థిరరాశి పూర్ణాంకం np2 = n - (n % ggml_f16_epr);

        svfloat32_t sum1_lo = svdup_n_f32(0.0f);
        svfloat32_t sum1_hi = svdup_n_f32(0.0f);
        svfloat32_t sum2_lo = svdup_n_f32(0.0f);
        svfloat32_t sum2_hi = svdup_n_f32(0.0f);
        svfloat32_t sum3_lo = svdup_n_f32(0.0f);
        svfloat32_t sum3_hi = svdup_n_f32(0.0f);
        svfloat32_t sum4_lo = svdup_n_f32(0.0f);
        svfloat32_t sum4_hi = svdup_n_f32(0.0f);

        కొరకు (పూర్ణాంకం i = 0; i < np; i += ggml_f16_అడుగు) {
            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_ఎక్కించు(x + i + 0 * ggml_f16_epr, 0), GGML_F16x_VEC_ఎక్కించు(y + i + 0 * ggml_f16_epr, 0));
            ggml_sve_f16_fma_widened(&sum2_lo, &sum2_hi, GGML_F16x_VEC_ఎక్కించు(x + i + 1 * ggml_f16_epr, 1), GGML_F16x_VEC_ఎక్కించు(y + i + 1 * ggml_f16_epr, 1));
            ggml_sve_f16_fma_widened(&sum3_lo, &sum3_hi, GGML_F16x_VEC_ఎక్కించు(x + i + 2 * ggml_f16_epr, 2), GGML_F16x_VEC_ఎక్కించు(y + i + 2 * ggml_f16_epr, 2));
            ggml_sve_f16_fma_widened(&sum4_lo, &sum4_hi, GGML_F16x_VEC_ఎక్కించు(x + i + 3 * ggml_f16_epr, 3), GGML_F16x_VEC_ఎక్కించు(y + i + 3 * ggml_f16_epr, 3));
            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_ఎక్కించు(x + i + 4 * ggml_f16_epr, 4), GGML_F16x_VEC_ఎక్కించు(y + i + 4 * ggml_f16_epr, 4));
            ggml_sve_f16_fma_widened(&sum2_lo, &sum2_hi, GGML_F16x_VEC_ఎక్కించు(x + i + 5 * ggml_f16_epr, 5), GGML_F16x_VEC_ఎక్కించు(y + i + 5 * ggml_f16_epr, 5));
            ggml_sve_f16_fma_widened(&sum3_lo, &sum3_hi, GGML_F16x_VEC_ఎక్కించు(x + i + 6 * ggml_f16_epr, 6), GGML_F16x_VEC_ఎక్కించు(y + i + 6 * ggml_f16_epr, 6));
            ggml_sve_f16_fma_widened(&sum4_lo, &sum4_hi, GGML_F16x_VEC_ఎక్కించు(x + i + 7 * ggml_f16_epr, 7), GGML_F16x_VEC_ఎక్కించు(y + i + 7 * ggml_f16_epr, 7));
        }

        కొరకు (పూర్ణాంకం i = np; i < np2; i += ggml_f16_epr) {
            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_ఎక్కించు(x + i, 0), GGML_F16x_VEC_ఎక్కించు(y + i, 0));
        }

        ఒకవేళ (np2 < n) {
            స్థిరరాశి svbool_t pg = svwhilelt_b16(np2, n);
            స్థిరరాశి svfloat16_t rx = svld1_f16(pg, (స్థిరరాశి __fp16 *)(x + np2));
            స్థిరరాశి svfloat16_t ry = svld1_f16(pg, (స్థిరరాశి __fp16 *)(y + np2));

            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, rx, ry);
        }

        sum1_lo = svadd_f32_m(DEFAULT_PG32, sum1_lo, sum2_lo);
        sum1_hi = svadd_f32_m(DEFAULT_PG32, sum1_hi, sum2_hi);
        sum3_lo = svadd_f32_m(DEFAULT_PG32, sum3_lo, sum4_lo);
        sum3_hi = svadd_f32_m(DEFAULT_PG32, sum3_hi, sum4_hi);
        sum1_lo = svadd_f32_m(DEFAULT_PG32, sum1_lo, sum3_lo);
        sum1_hi = svadd_f32_m(DEFAULT_PG32, sum1_hi, sum3_hi);

        sumf = ggml_sve_కూడిక_f32x2(sum1_lo, sum1_hi);
    #లేదావేళ defined(__riscv_v_intrinsic)
        #ఒకవేళ defined(__riscv_zvfh)
            పూర్ణాంకం vl = __riscv_vsetvlmax_e32m2();
            vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m2_t vsum;
            vfloat16m1_t ax;
            vfloat16m1_t ay;
            vsum = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vmv_v_x_u32m2(0, vl));
            కొరకు (పూర్ణాంకం i = 0; i < n; i += vl) {
                vl = __riscv_vsetvl_e16m1(n - i);
                ax = __riscv_vle16_v_f16m1_tu(ax, (స్థిరరాశి _Float16 *)&x[i], vl);
                ay = __riscv_vle16_v_f16m1_tu(ay, (స్థిరరాశి _Float16 *)&y[i], vl);
                vsum = __riscv_vfwmacc_vv_f32m2_tu(vsum, ax, ay, vl);
            }
            vl = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t ac0 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(vsum, 0), __riscv_vget_v_f32m2_f32m1(vsum, 1), vl);
            vs = __riscv_vfredusum_vs_f32m1_f32m1(ac0, vs, vl);
            sumf += __riscv_vfmv_f_s_f32m1_f32(vs);
        #లేకపోతే
            కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
                sumf += (ggml_float)(GGML_CPU_FP16_గమ్యం_FP32(x[i])*GGML_CPU_FP16_గమ్యం_FP32(y[i]));
            }
        #అయితే_ముగింపు // __riscv_zvfh
    #లేకపోతే
        స్థిరరాశి పూర్ణాంకం np = (n & ~(GGML_F16_DOT_అడుగు - 1));

        GGML_F16_DOT_VEC కూడిక[GGML_F16_DOT_ARR] = { GGML_F16_DOT_VEC_ZERO };

        GGML_F16_DOT_VEC ax[GGML_F16_DOT_ARR];
        GGML_F16_DOT_VEC ay[GGML_F16_DOT_ARR];

        కొరకు (పూర్ణాంకం i = 0; i < np; i += GGML_F16_DOT_అడుగు) {
            కొరకు (పూర్ణాంకం j = 0; j < GGML_F16_DOT_ARR; j++) {
                ax[j] = GGML_F16_DOT_VEC_ఎక్కించు(x + i + j*GGML_F16_DOT_EPR, j);
                ay[j] = GGML_F16_DOT_VEC_ఎక్కించు(y + i + j*GGML_F16_DOT_EPR, j);

                కూడిక[j] = GGML_F16_DOT_VEC_FMA(కూడిక[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F16_DOT_VEC_REDUCE(sumf, కూడిక);

        // leftovers
        కొరకు (పూర్ణాంకం i = np; i < n; ++i) {
            sumf += (ggml_float)(GGML_CPU_FP16_గమ్యం_FP32(x[i])*GGML_CPU_FP16_గమ్యం_FP32(y[i]));
        }
        // if you hit this, you are likely running outside the FP range
        ప్రతిజ్ఞా(!isnan(sumf) && !isinf(sumf));
    #అయితే_ముగింపు
#లేకపోతే
    కొరకు (పూర్ణాంకం i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_CPU_FP16_గమ్యం_FP32(x[i])*GGML_CPU_FP16_గమ్యం_FP32(y[i]));
    }
#అయితే_ముగింపు // GGML_SIMD

    *s = sumf;
}

శూన్య ggml_vec_silu_f32(స్థిరరాశి పూర్ణాంకం n, దశాంశం * y, స్థిరరాశి దశాంశం * x) {
    పూర్ణాంకం i = 0;
#ఒకవేళ defined(__AVX512F__) && defined(__AVX512DQ__)
    కొరకు (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_silu(_mm512_loadu_ps(x + i)));
    }
#లేదావేళ defined(__AVX2__) && defined(__FMA__)
    కొరకు (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_silu(_mm256_loadu_ps(x + i)));
    }
#లేదావేళ defined(__SSE2__)
    కొరకు (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, ggml_v_silu(_mm_loadu_ps(x + i)));
    }
#లేదావేళ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    స్థిరరాశి పూర్ణాంకం vlen = svcntw();
    కొరకు (; i < n; i += vlen) {
        స్థిరరాశి svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, ggml_v_silu(pg, svld1_f32(pg, x + i)));
    }
#లేదావేళ defined(__ARM_NEON) && defined(__aarch64__)
    కొరకు (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, ggml_v_silu(vld1q_f32(x + i)));
    }
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (; i + 3 < n; i += 4) {
        vec_xst(ggml_v_silu(vec_xl(0, x + i)), 0, y + i);
    }
#లేదావేళ defined(__riscv_v_intrinsic)
    కొరకు (పూర్ణాంకం vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vy = ggml_v_silu_m2(vx, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#అయితే_ముగింపు
    కొరకు (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

శూన్య ggml_vec_swiglu_f32(స్థిరరాశి పూర్ణాంకం n, దశాంశం * y, స్థిరరాశి దశాంశం * x, స్థిరరాశి దశాంశం * g) {
    పూర్ణాంకం i = 0;
#ఒకవేళ defined(__AVX512F__) && defined(__AVX512DQ__)
    కొరకు (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_mul_ps(ggml_v_silu(_mm512_loadu_ps(x + i)), _mm512_loadu_ps(g + i)));
    }
#లేదావేళ defined(__AVX2__) && defined(__FMA__)
    కొరకు (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(ggml_v_silu(_mm256_loadu_ps(x + i)), _mm256_loadu_ps(g + i)));
    }
#లేదావేళ defined(__SSE2__)
    కొరకు (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, _mm_mul_ps(ggml_v_silu(_mm_loadu_ps(x + i)), _mm_loadu_ps(g + i)));
    }
#లేదావేళ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    స్థిరరాశి పూర్ణాంకం vlen = svcntw();
    కొరకు (; i < n; i += vlen) {
        స్థిరరాశి svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, svmul_f32_x(pg, ggml_v_silu(pg, svld1_f32(pg, x + i)), svld1_f32(pg, g + i)));
    }
#లేదావేళ defined(__ARM_NEON) && defined(__aarch64__)
    కొరకు (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(ggml_v_silu(vld1q_f32(x + i)), vld1q_f32(g + i)));
    }
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (; i + 3 < n; i += 4) {
        vec_xst(vec_mul(ggml_v_silu(vec_xl(0, x + i)), vec_xl(0, g + i)), 0, y + i);
    }
#లేదావేళ defined(__riscv_v_intrinsic)
    కొరకు (పూర్ణాంకం vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vg = __riscv_vle32_v_f32m2(&g[i], vl);
        vfloat32m2_t vy = __riscv_vfmul_vv_f32m2(ggml_v_silu_m2(vx, vl), vg, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#అయితే_ముగింపు
    కొరకు (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]) * g[i];
    }
}

#నిర్వచితం_కాకపోతే GGML_GELU_శీఘ్ర_FP16
// computed in f32 without the fp16 table, which loses precision on f32 input
శూన్య ggml_vec_gelu_శీఘ్ర_f32(స్థిరరాశి పూర్ణాంకం n, దశాంశం * y, స్థిరరాశి దశాంశం * x) {
    పూర్ణాంకం i = 0;
#ఒకవేళ defined(__AVX512F__) && defined(__AVX512DQ__)
    కొరకు (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_gelu_శీఘ్ర(_mm512_loadu_ps(x + i)));
    }
#లేదావేళ defined(__AVX2__) && defined(__FMA__)
    కొరకు (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_gelu_శీఘ్ర(_mm256_loadu_ps(x + i)));
    }
#లేదావేళ defined(__SSE2__)
    కొరకు (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, ggml_v_gelu_శీఘ్ర(_mm_loadu_ps(x + i)));
    }
#లేదావేళ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    స్థిరరాశి పూర్ణాంకం vlen = svcntw();
    కొరకు (; i < n; i += vlen) {
        స్థిరరాశి svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, ggml_v_gelu_శీఘ్ర(pg, svld1_f32(pg, x + i)));
    }
#లేదావేళ defined(__ARM_NEON) && defined(__aarch64__)
    కొరకు (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, ggml_v_gelu_శీఘ్ర(vld1q_f32(x + i)));
    }
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (; i + 3 < n; i += 4) {
        vec_xst(ggml_v_gelu_శీఘ్ర(vec_xl(0, x + i)), 0, y + i);
    }
#లేదావేళ defined(__riscv_v_intrinsic)
    కొరకు (పూర్ణాంకం vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vy = ggml_v_gelu_శీఘ్ర_m2(vx, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#అయితే_ముగింపు
    కొరకు (; i < n; ++i) {
        y[i] = ggml_gelu_శీఘ్ర_f32(x[i]);
    }
}

శూన్య ggml_vec_geglu_శీఘ్ర_f32(స్థిరరాశి పూర్ణాంకం n, దశాంశం * y, స్థిరరాశి దశాంశం * x, స్థిరరాశి దశాంశం * g) {
    పూర్ణాంకం i = 0;
#ఒకవేళ defined(__AVX512F__) && defined(__AVX512DQ__)
    కొరకు (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_mul_ps(ggml_v_gelu_శీఘ్ర(_mm512_loadu_ps(x + i)), _mm512_loadu_ps(g + i)));
    }
#లేదావేళ defined(__AVX2__) && defined(__FMA__)
    కొరకు (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(ggml_v_gelu_శీఘ్ర(_mm256_loadu_ps(x + i)), _mm256_loadu_ps(g + i)));
    }
#లేదావేళ defined(__SSE2__)
    కొరకు (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, _mm_mul_ps(ggml_v_gelu_శీఘ్ర(_mm_loadu_ps(x + i)), _mm_loadu_ps(g + i)));
    }
#లేదావేళ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    స్థిరరాశి పూర్ణాంకం vlen = svcntw();
    కొరకు (; i < n; i += vlen) {
        స్థిరరాశి svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, svmul_f32_x(pg, ggml_v_gelu_శీఘ్ర(pg, svld1_f32(pg, x + i)), svld1_f32(pg, g + i)));
    }
#లేదావేళ defined(__ARM_NEON) && defined(__aarch64__)
    కొరకు (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(ggml_v_gelu_శీఘ్ర(vld1q_f32(x + i)), vld1q_f32(g + i)));
    }
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (; i + 3 < n; i += 4) {
        vec_xst(vec_mul(ggml_v_gelu_శీఘ్ర(vec_xl(0, x + i)), vec_xl(0, g + i)), 0, y + i);
    }
#లేదావేళ defined(__riscv_v_intrinsic)
    కొరకు (పూర్ణాంకం vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vg = __riscv_vle32_v_f32m2(&g[i], vl);
        vfloat32m2_t vy = __riscv_vfmul_vv_f32m2(ggml_v_gelu_శీఘ్ర_m2(vx, vl), vg, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#అయితే_ముగింపు
    కొరకు (; i < n; ++i) {
        y[i] = ggml_gelu_శీఘ్ర_f32(x[i]) * g[i];
    }
}
#అయితే_ముగింపు

ggml_float ggml_vec_cvar_f32(స్థిరరాశి పూర్ణాంకం n, దశాంశం * y, స్థిరరాశి దశాంశం * x, స్థిరరాశి దశాంశం సగటు_2) {
    పూర్ణాంకం i = 0;
    ggml_float కూడిక = 0;
// TODO: optimize to process the remaining elements in groups using the smaller vector sizes from AVX2 and SSE
// ref: https://github.com/ggml-org/llama.cpp/pull/15953#pullrequestreview-3310928344
#ఒకవేళ defined(__AVX512F__) && defined(__AVX512DQ__)
    కొరకు (; i + 15 < n; i += 16) {
        __m512 val = _mm512_sub_ps(_mm512_loadu_ps(x + i),
                                   _mm512_set1_ps(సగటు_2));
        _mm512_storeu_ps(y + i, val);
        కూడిక += (ggml_float)mm512_reduce_కూడు_ps(_mm512_mul_ps(val, val));
    }
#లేదావేళ defined(__AVX2__) && defined(__FMA__)
    కొరకు (; i + 7 < n; i += 8) {
        __m256 val = _mm256_sub_ps(_mm256_loadu_ps(x + i),
                                   _mm256_set1_ps(సగటు_2));
        _mm256_storeu_ps(y + i, val);
        val = _mm256_mul_ps(val,val);
        __m128 val2 = mm_కూడు_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = mm_కూడు_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = mm_కూడు_ss(val2, _mm_movehdup_ps(val2));
        కూడిక += (ggml_float)_mm_cvtss_f32(val2);
    }
#లేదావేళ defined(__SSE2__)
    కొరకు (; i + 3 < n; i += 4) {
        __m128 val = _mm_sub_ps(_mm_loadu_ps(x + i),
                                _mm_set1_ps(సగటు_2));
        _mm_storeu_ps(y + i, val);
        val = _mm_mul_ps(val, val);
#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = mm_కూడు_ps(val, _mm_movehl_ps(val, val));
        val = mm_కూడు_ss(val, _mm_movehdup_ps(val));
#లేకపోతే
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = mm_కూడు_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = mm_కూడు_ss(val, tmp);
#అయితే_ముగింపు  // __AVX__ || __AVX2__ || __AVX512F__
        కూడిక += (ggml_float)_mm_cvtss_f32(val);
    }
#లేదావేళ defined(__ARM_NEON) && defined(__aarch64__)
    కొరకు (; i + 3 < n; i += 4) {
        float32x4_t val = vsubq_f32(vld1q_f32(x + i),
                                    vdupq_n_f32(సగటు_2));
        vst1q_f32(y + i, val);
        val = vmulq_f32(val, val);
        కూడిక += (ggml_float)vaddvq_f32(val);
    }
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (; i + 3 < n; i += 4) {
        float32x4_t val = vec_sub(vec_xl(0, x + i), vec_splats(సగటు_2));
        vec_xst(val, 0, y + i);
        val = vec_mul(val, val);
        కూడిక += (ggml_float)vec_hsum_f32x4(val);
    }
#లేదావేళ defined(__riscv_v_intrinsic)
    vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
    కొరకు (పూర్ణాంకం vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t val = __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&x[i], vl), సగటు_2, vl);
        __riscv_vse32_v_f32m2(&y[i], val, vl);
        val = __riscv_vfmul_vv_f32m2(val, val, vl);
        vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, vl);
    }
    కూడిక = (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
#అయితే_ముగింపు
    కొరకు (; i < n; ++i) {
        దశాంశం val = x[i] - సగటు_2;
        y[i] = val;
        val *= val;
        కూడిక += (ggml_float)val;
    }
    తిరిగి కూడిక/n;
}

ggml_float ggml_vec_soft_max_f32(స్థిరరాశి పూర్ణాంకం n, దశాంశం * y, స్థిరరాశి దశాంశం * x, దశాంశం max) {
    పూర్ణాంకం i = 0;
    ggml_float కూడిక = 0;
#ఒకవేళ defined(__AVX512F__) && defined(__AVX512DQ__)
    కొరకు (; i + 15 < n; i += 16) {
        __m512 val = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        కూడిక += (ggml_float)mm512_reduce_కూడు_ps(val);
    }
#లేదావేళ defined(__AVX2__) && defined(__FMA__)
    కొరకు (; i + 7 < n; i += 8) {
        __m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = mm_కూడు_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = mm_కూడు_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = mm_కూడు_ss(val2, _mm_movehdup_ps(val2));
        కూడిక += (ggml_float)_mm_cvtss_f32(val2);
    }
#లేదావేళ defined(__SSE2__)
    కొరకు (; i + 3 < n; i += 4) {
        __m128 val = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#ఒకవేళ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = mm_కూడు_ps(val, _mm_movehl_ps(val, val));
        val = mm_కూడు_ss(val, _mm_movehdup_ps(val));
#లేకపోతే
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = mm_కూడు_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = mm_కూడు_ss(val, tmp);
#అయితే_ముగింపు
        కూడిక += (ggml_float)_mm_cvtss_f32(val);
    }
#లేదావేళ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    స్థిరరాశి పూర్ణాంకం vlen = svcntw();
    కొరకు (; i < n; i += vlen) {
        స్థిరరాశి svbool_t pg = svwhilelt_b32_s32(i, n);
        svfloat32_t val = ggml_v_expf(pg, svsub_f32_x(pg, svld1_f32(pg, x + i),
                                                svdup_n_f32_x(pg, max)));
        svst1_f32(pg, y + i, val);
        కూడిక += (ggml_float)svaddv_f32(pg, val);
    }
#లేదావేళ defined(__ARM_NEON) && defined(__aarch64__)
    కొరకు (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        కూడిక += (ggml_float)vaddvq_f32(val);
    }
#లేదావేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (; i + 3 < n; i += 4) {
        float32x4_t v_val = ggml_v_expf(vec_sub(vec_xl(0, x + i), vec_splats(max)));
        vec_xst(v_val, 0, y + i);
        కూడిక += (ggml_float)vec_hsum_f32x4(v_val);
    }
#లేదావేళ defined(__riscv_v_intrinsic)
    vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
    కొరకు (పూర్ణాంకం avl; i < n; i += avl) {
        avl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t val = ggml_v_expf_m2(__riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&x[i], avl), max, avl), avl);
        __riscv_vse32_v_f32m2(&y[i], val, avl);
        vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, avl);
    }
    తిరిగి (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
#అయితే_ముగింపు
    కొరకు (; i < n; ++i) {
        దశాంశం val = expf(x[i] - max);
        కూడిక += (ggml_float)val;
        y[i] = val;
    }
    తిరిగి కూడిక;
}

ggml_float ggml_vec_log_soft_max_f32(స్థిరరాశి పూర్ణాంకం n, దశాంశం * y, స్థిరరాశి దశాంశం * x, దశాంశం max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    పూర్ణాంకం i = 0;
    ggml_float కూడిక = 0;
    కొరకు (; i < n; ++i) {
        దశాంశం val = x[i] - max;
        y[i] = val;
        కూడిక += (ggml_float)expf(val);
    }
    తిరిగి కూడిక = (ggml_float)logf(కూడిక);
}
