#ਅਨ੍ਤਰ੍ਭਾਵ "vec.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_ਸਾਰਣੀ_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_ਸਾਰਣੀ_gelu_ਤੇਜ਼_f16[1 << 16];

ਖਾਲੀ ggml_vec_dot_f32(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, size_t bx, ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT y, size_t by, ਪੂਰਨ nrc) {
   ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
   GGML_ਅਣਵਰਤਿਆ(nrc);
   GGML_ਅਣਵਰਤਿਆ(bx);
   GGML_ਅਣਵਰਤਿਆ(by);
   GGML_ਅਣਵਰਤਿਆ(bs);

#ਜੇ defined(GGML_SIMD)
    ਦਸ਼ਮਲਵ sumf = 0.0f;

    #ਜੇ defined(__ARM_FEATURE_SVE)
        ਅਚਲ ਪੂਰਨ sve_ਰਜਿਸਟਰ_ਲੰਮਾਈ = ggml_cpu_get_sve_cnt() * 8;
        ਅਚਲ ਪੂਰਨ ggml_f32_epr = sve_ਰਜਿਸਟਰ_ਲੰਮਾਈ / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        ਅਚਲ ਪੂਰਨ ggml_f32_ਕਦਮ = 8 * ggml_f32_epr; // choose 8 SVE registers

        ਅਚਲ ਪੂਰਨ np = (n & ~(ggml_f32_ਕਦਮ - 1));
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
        ਲਈ (ਪੂਰਨ i = 0; i < np; i += ggml_f32_ਕਦਮ) {
            ax1 = GGML_F32_VEC_ਲੱਦੋ(x + i);
            ay1 = GGML_F32_VEC_ਲੱਦੋ(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);

            ax2 = GGML_F32_VEC_ਲੱਦੋ(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_ਲੱਦੋ(y + i + 1*ggml_f32_epr);
            sum2 = GGML_F32_VEC_FMA(sum2, ax2, ay2);

            ax3 = GGML_F32_VEC_ਲੱਦੋ(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_ਲੱਦੋ(y + i + 2*ggml_f32_epr);
            sum3 = GGML_F32_VEC_FMA(sum3, ax3, ay3);

            ax4 = GGML_F32_VEC_ਲੱਦੋ(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_ਲੱਦੋ(y + i + 3*ggml_f32_epr);
            sum4 = GGML_F32_VEC_FMA(sum4, ax4, ay4);

            ax5 = GGML_F32_VEC_ਲੱਦੋ(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_ਲੱਦੋ(y + i + 4*ggml_f32_epr);
            sum5 = GGML_F32_VEC_FMA(sum5, ax5, ay5);

            ax6 = GGML_F32_VEC_ਲੱਦੋ(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_ਲੱਦੋ(y + i + 5*ggml_f32_epr);
            sum6 = GGML_F32_VEC_FMA(sum6, ax6, ay6);

            ax7 = GGML_F32_VEC_ਲੱਦੋ(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_ਲੱਦੋ(y + i + 6*ggml_f32_epr);
            sum7 = GGML_F32_VEC_FMA(sum7, ax7, ay7);

            ax8 = GGML_F32_VEC_ਲੱਦੋ(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_ਲੱਦੋ(y + i + 7*ggml_f32_epr);
            sum8 = GGML_F32_VEC_FMA(sum8, ax8, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        ਅਚਲ ਪੂਰਨ np2 = (n & ~(ggml_f32_epr - 1));
        ਲਈ (ਪੂਰਨ i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_ਲੱਦੋ(x + i);
            ay1 = GGML_F32_VEC_ਲੱਦੋ(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmla on available elements only
        ਜੇ (np2 < n) {
            svbool_t pg = svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            sum1 = svmla_f32_m(pg, sum1, ax1, ay1);
        }
        // reduce sum1,sum2 to sum1
        GGML_F32_VEC_REDUCE(sumf, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
    #ਜਾਂਜੇ defined(__riscv_v_intrinsic)
        ਪੂਰਨ vl = __riscv_vsetvlmax_e32m8();
        vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m8_t vsum;
        vfloat32m8_t ax;
        vfloat32m8_t ay;
        vsum = __riscv_vfmv_v_f_f32m8_tu(vsum, 0.0f, vl);
        ਲਈ (ਪੂਰਨ i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            ax = __riscv_vle32_v_f32m8_tu(ax, &x[i], vl);
            ay = __riscv_vle32_v_f32m8_tu(ay, &y[i], vl);
            vsum = __riscv_vfmacc_vv_f32m8_tu(vsum, ax, ay, vl);
        }
        vl = __riscv_vsetvlmax_e32m8();
        vs = __riscv_vfredusum_vs_f32m8_f32m1(vsum, vs, vl);
        sumf += __riscv_vfmv_f_s_f32m1_f32(vs);
    #ਨਹੀਂਤਾਂ
        ਅਚਲ ਪੂਰਨ np = (n & ~(GGML_F32_ਕਦਮ - 1));

        GGML_F32_VEC ਜੋੜ[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        ਲਈ (ਪੂਰਨ i = 0; i < np; i += GGML_F32_ਕਦਮ) {
            ਲਈ (ਪੂਰਨ j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_ਲੱਦੋ(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_ਲੱਦੋ(y + i + j*GGML_F32_EPR);

                ਜੋੜ[j] = GGML_F32_VEC_FMA(ਜੋੜ[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F32_VEC_REDUCE(sumf, ਜੋੜ);

        // leftovers
        ਲਈ (ਪੂਰਨ i = np; i < n; ++i) {
            sumf += x[i]*y[i];
        }
    #ਜੇ_ਅੰਤ
#ਨਹੀਂਤਾਂ
    // scalar
    ggml_float sumf = 0.0;
    ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#ਜੇ_ਅੰਤ

    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_bf16(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    GGML_ਅਣਵਰਤਿਆ(nrc);
    GGML_ਅਣਵਰਤਿਆ(bx);
    GGML_ਅਣਵਰਤਿਆ(by);
    GGML_ਅਣਵਰਤਿਆ(bs);
    ਪੂਰਨ i = 0;
    ggml_float sumf = 0;

#ਜੇ defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    ਲਈ (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)mm512_reduce_ਜੋੜੋ_ps(c1);
    sumf += (ggml_float)mm512_reduce_ਜੋੜੋ_ps(c2);

#ਜਾਂਜੇ defined(__AVX512F__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਲੱਦੋ_2(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((ਅਚਲ __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    ਲਈ (; i + 32 <= n; i += 32) {
        c1 = mm512_ਜੋੜੋ_ps(_mm512_mul_ps(ਲੱਦੋ_2(x + i), ਲੱਦੋ_2(y + i)), c1);
        c2 = mm512_ਜੋੜੋ_ps(_mm512_mul_ps(ਲੱਦੋ_2(x + i + 16), ਲੱਦੋ_2(y + i + 16)), c2);
    }
    sumf += (ggml_float)mm512_reduce_ਜੋੜੋ_ps(c1);
    sumf += (ggml_float)mm512_reduce_ਜੋੜੋ_ps(c2);

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ ਲੱਦੋ_2
#ਜਾਂਜੇ defined(__AVX2__) || defined(__AVX__)
#ਜੇ defined(__AVX2__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਲੱਦੋ_2(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((ਅਚਲ __m128i *)(p))), 16))
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਲੱਦੋ_2(p) _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((ਅਚਲ __m128i *)(p))), 16)), (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((ਅਚਲ __m128i *)(p)), 8)), 16)), 1))
#ਜੇ_ਅੰਤ
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    ਲਈ (; i + 32 <= n; i += 32) {
        c1 = mm256_ਜੋੜੋ_ps(_mm256_mul_ps(ਲੱਦੋ_2(x + i), ਲੱਦੋ_2(y + i)), c1);
        c2 = mm256_ਜੋੜੋ_ps(_mm256_mul_ps(ਲੱਦੋ_2(x + i + 8), ਲੱਦੋ_2(y + i + 8)), c2);
        c3 = mm256_ਜੋੜੋ_ps(_mm256_mul_ps(ਲੱਦੋ_2(x + i + 16), ਲੱਦੋ_2(y + i + 16)), c3);
        c4 = mm256_ਜੋੜੋ_ps(_mm256_mul_ps(ਲੱਦੋ_2(x + i + 24), ਲੱਦੋ_2(y + i + 24)), c4);
    }
    __m128 g;
    c1 = mm256_ਜੋੜੋ_ps(mm256_ਜੋੜੋ_ps(c1, c3),
                       mm256_ਜੋੜੋ_ps(c2, c4));
    g = mm_ਜੋੜੋ_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = mm_ਜੋੜੋ_ps(g, _mm_movehl_ps(g, g));
    g = mm_ਜੋੜੋ_ss(g, _mm_movehdup_ps(g));
    sumf += (ggml_float)_mm_cvtss_f32(g);

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ ਲੱਦੋ_2
#ਜਾਂਜੇ defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfwma)
    size_t vl = __riscv_vsetvlmax_e32m4();

    // initialize accumulators to all zeroes
    vfloat32m4_t vsum0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vsum1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    // calculate step size
    ਅਚਲ size_t epr = __riscv_vsetvlmax_e16m2();
    ਅਚਲ size_t ਕਦਮ = epr * 2;
    ਅਚਲ ਪੂਰਨ np = (n & ~(ਕਦਮ - 1));

    // unroll by 2
    ਲਈ (; i < np; i += ਕਦਮ) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ਅਚਲ __bf16 *)&x[i], epr);
        vbfloat16m2_t ay0 = __riscv_vle16_v_bf16m2((ਅਚਲ __bf16 *)&y[i], epr);
        vsum0 = __riscv_vfwmaccbf16_vv_f32m4(vsum0, ax0, ay0, epr);
        __asm__ __volatile__ ("" ::: "memory");

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((ਅਚਲ __bf16 *)&x[i + epr], epr);
        vbfloat16m2_t ay1 = __riscv_vle16_v_bf16m2((ਅਚਲ __bf16 *)&y[i + epr], epr);
        vsum1 = __riscv_vfwmaccbf16_vv_f32m4(vsum1, ax1, ay1, epr);
        __asm__ __volatile__ ("" ::: "memory");
    }

    // accumulate in 1 register
    vsum0 = __riscv_vfadd_vv_f32m4(vsum0, vsum1, vl);

    // leftovers
    ਲਈ (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((ਅਚਲ __bf16 *)&x[i], vl);
        vbfloat16m2_t ay0 = __riscv_vle16_v_bf16m2((ਅਚਲ __bf16 *)&y[i], vl);
        vsum0 = __riscv_vfwmaccbf16_vv_f32m4(vsum0, ax0, ay0, vl);
    }

    // reduce
    vl = __riscv_vsetvlmax_e32m4();
    vfloat32m1_t redsum = __riscv_vfredusum_vs_f32m4_f32m1(vsum0, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
    sumf += __riscv_vfmv_f_s_f32m1_f32(redsum);

#ਜਾਂਜੇ defined(__POWER9_VECTOR__) || defined(__VXE__) || defined(__VXE2__)
    ਅਚਲ ਪੂਰਨ np = (n & ~(GGML_BF16_ਕਦਮ - 1));
    ਜੇ (np > 0) {
        GGML_F32_VEC ਜੋੜ[4] = {GGML_F32_VEC_ZERO};
        ਲਈ (; i < np; i += GGML_BF16_ਕਦਮ) {
            GGML_BF16_VEC vx0 = GGML_BF16_VEC_ਲੱਦੋ(x + i);
            GGML_BF16_VEC vx1 = GGML_BF16_VEC_ਲੱਦੋ(x + i + 8);
            GGML_BF16_VEC vy0 = GGML_BF16_VEC_ਲੱਦੋ(y + i);
            GGML_BF16_VEC vy1 = GGML_BF16_VEC_ਲੱਦੋ(y + i + 8);
            GGML_BF16_FMA_LO(ਜੋੜ[0], vx0, vy0);
            GGML_BF16_FMA_HI(ਜੋੜ[1], vx0, vy0);
            GGML_BF16_FMA_LO(ਜੋੜ[2], vx1, vy1);
            GGML_BF16_FMA_HI(ਜੋੜ[3], vx1, vy1);
        }
        GGML_F32x4_REDUCE_4(sumf, ਜੋੜ[0], ਜੋੜ[1], ਜੋੜ[2], ਜੋੜ[3]);
    }
#ਜੇ_ਅੰਤ

    ਲਈ (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_ਮੰਜ਼ਿਲ_FP32(x[i]) *
                             GGML_BF16_ਮੰਜ਼ਿਲ_FP32(y[i]));
    }
    *s = sumf;
}

ਖਾਲੀ ggml_vec_dot_f16(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    GGML_ਅਣਵਰਤਿਆ(nrc);
    GGML_ਅਣਵਰਤਿਆ(bx);
    GGML_ਅਣਵਰਤਿਆ(by);
    GGML_ਅਣਵਰਤਿਆ(bs);

    ggml_float sumf = 0.0;


#ਜੇ defined(GGML_SIMD)
    #ਜੇ defined(__ARM_FEATURE_SVE)
        ਅਚਲ ਪੂਰਨ ggml_f16_epr = svcnth();
        ਅਚਲ ਪੂਰਨ ggml_f16_ਕਦਮ = 8 * ggml_f16_epr;
        ਅਚਲ ਪੂਰਨ np = n - (n % ggml_f16_ਕਦਮ);
        ਅਚਲ ਪੂਰਨ np2 = n - (n % ggml_f16_epr);

        svfloat32_t sum1_lo = svdup_n_f32(0.0f);
        svfloat32_t sum1_hi = svdup_n_f32(0.0f);
        svfloat32_t sum2_lo = svdup_n_f32(0.0f);
        svfloat32_t sum2_hi = svdup_n_f32(0.0f);
        svfloat32_t sum3_lo = svdup_n_f32(0.0f);
        svfloat32_t sum3_hi = svdup_n_f32(0.0f);
        svfloat32_t sum4_lo = svdup_n_f32(0.0f);
        svfloat32_t sum4_hi = svdup_n_f32(0.0f);

        ਲਈ (ਪੂਰਨ i = 0; i < np; i += ggml_f16_ਕਦਮ) {
            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_ਲੱਦੋ(x + i + 0 * ggml_f16_epr, 0), GGML_F16x_VEC_ਲੱਦੋ(y + i + 0 * ggml_f16_epr, 0));
            ggml_sve_f16_fma_widened(&sum2_lo, &sum2_hi, GGML_F16x_VEC_ਲੱਦੋ(x + i + 1 * ggml_f16_epr, 1), GGML_F16x_VEC_ਲੱਦੋ(y + i + 1 * ggml_f16_epr, 1));
            ggml_sve_f16_fma_widened(&sum3_lo, &sum3_hi, GGML_F16x_VEC_ਲੱਦੋ(x + i + 2 * ggml_f16_epr, 2), GGML_F16x_VEC_ਲੱਦੋ(y + i + 2 * ggml_f16_epr, 2));
            ggml_sve_f16_fma_widened(&sum4_lo, &sum4_hi, GGML_F16x_VEC_ਲੱਦੋ(x + i + 3 * ggml_f16_epr, 3), GGML_F16x_VEC_ਲੱਦੋ(y + i + 3 * ggml_f16_epr, 3));
            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_ਲੱਦੋ(x + i + 4 * ggml_f16_epr, 4), GGML_F16x_VEC_ਲੱਦੋ(y + i + 4 * ggml_f16_epr, 4));
            ggml_sve_f16_fma_widened(&sum2_lo, &sum2_hi, GGML_F16x_VEC_ਲੱਦੋ(x + i + 5 * ggml_f16_epr, 5), GGML_F16x_VEC_ਲੱਦੋ(y + i + 5 * ggml_f16_epr, 5));
            ggml_sve_f16_fma_widened(&sum3_lo, &sum3_hi, GGML_F16x_VEC_ਲੱਦੋ(x + i + 6 * ggml_f16_epr, 6), GGML_F16x_VEC_ਲੱਦੋ(y + i + 6 * ggml_f16_epr, 6));
            ggml_sve_f16_fma_widened(&sum4_lo, &sum4_hi, GGML_F16x_VEC_ਲੱਦੋ(x + i + 7 * ggml_f16_epr, 7), GGML_F16x_VEC_ਲੱਦੋ(y + i + 7 * ggml_f16_epr, 7));
        }

        ਲਈ (ਪੂਰਨ i = np; i < np2; i += ggml_f16_epr) {
            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, GGML_F16x_VEC_ਲੱਦੋ(x + i, 0), GGML_F16x_VEC_ਲੱਦੋ(y + i, 0));
        }

        ਜੇ (np2 < n) {
            ਅਚਲ svbool_t pg = svwhilelt_b16(np2, n);
            ਅਚਲ svfloat16_t rx = svld1_f16(pg, (ਅਚਲ __fp16 *)(x + np2));
            ਅਚਲ svfloat16_t ry = svld1_f16(pg, (ਅਚਲ __fp16 *)(y + np2));

            ggml_sve_f16_fma_widened(&sum1_lo, &sum1_hi, rx, ry);
        }

        sum1_lo = svadd_f32_m(DEFAULT_PG32, sum1_lo, sum2_lo);
        sum1_hi = svadd_f32_m(DEFAULT_PG32, sum1_hi, sum2_hi);
        sum3_lo = svadd_f32_m(DEFAULT_PG32, sum3_lo, sum4_lo);
        sum3_hi = svadd_f32_m(DEFAULT_PG32, sum3_hi, sum4_hi);
        sum1_lo = svadd_f32_m(DEFAULT_PG32, sum1_lo, sum3_lo);
        sum1_hi = svadd_f32_m(DEFAULT_PG32, sum1_hi, sum3_hi);

        sumf = ggml_sve_ਜੋੜ_f32x2(sum1_lo, sum1_hi);
    #ਜਾਂਜੇ defined(__riscv_v_intrinsic)
        #ਜੇ defined(__riscv_zvfh)
            ਪੂਰਨ vl = __riscv_vsetvlmax_e32m2();
            vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m2_t vsum;
            vfloat16m1_t ax;
            vfloat16m1_t ay;
            vsum = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vmv_v_x_u32m2(0, vl));
            ਲਈ (ਪੂਰਨ i = 0; i < n; i += vl) {
                vl = __riscv_vsetvl_e16m1(n - i);
                ax = __riscv_vle16_v_f16m1_tu(ax, (ਅਚਲ _Float16 *)&x[i], vl);
                ay = __riscv_vle16_v_f16m1_tu(ay, (ਅਚਲ _Float16 *)&y[i], vl);
                vsum = __riscv_vfwmacc_vv_f32m2_tu(vsum, ax, ay, vl);
            }
            vl = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t ac0 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(vsum, 0), __riscv_vget_v_f32m2_f32m1(vsum, 1), vl);
            vs = __riscv_vfredusum_vs_f32m1_f32m1(ac0, vs, vl);
            sumf += __riscv_vfmv_f_s_f32m1_f32(vs);
        #ਨਹੀਂਤਾਂ
            ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
                sumf += (ggml_float)(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i])*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[i]));
            }
        #ਜੇ_ਅੰਤ // __riscv_zvfh
    #ਨਹੀਂਤਾਂ
        ਅਚਲ ਪੂਰਨ np = (n & ~(GGML_F16_DOT_ਕਦਮ - 1));

        GGML_F16_DOT_VEC ਜੋੜ[GGML_F16_DOT_ARR] = { GGML_F16_DOT_VEC_ZERO };

        GGML_F16_DOT_VEC ax[GGML_F16_DOT_ARR];
        GGML_F16_DOT_VEC ay[GGML_F16_DOT_ARR];

        ਲਈ (ਪੂਰਨ i = 0; i < np; i += GGML_F16_DOT_ਕਦਮ) {
            ਲਈ (ਪੂਰਨ j = 0; j < GGML_F16_DOT_ARR; j++) {
                ax[j] = GGML_F16_DOT_VEC_ਲੱਦੋ(x + i + j*GGML_F16_DOT_EPR, j);
                ay[j] = GGML_F16_DOT_VEC_ਲੱਦੋ(y + i + j*GGML_F16_DOT_EPR, j);

                ਜੋੜ[j] = GGML_F16_DOT_VEC_FMA(ਜੋੜ[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F16_DOT_VEC_REDUCE(sumf, ਜੋੜ);

        // leftovers
        ਲਈ (ਪੂਰਨ i = np; i < n; ++i) {
            sumf += (ggml_float)(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i])*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[i]));
        }
        // if you hit this, you are likely running outside the FP range
        ਪ੍ਰਤਿਜ੍ਞਾ(!isnan(sumf) && !isinf(sumf));
    #ਜੇ_ਅੰਤ
#ਨਹੀਂਤਾਂ
    ਲਈ (ਪੂਰਨ i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i])*GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[i]));
    }
#ਜੇ_ਅੰਤ // GGML_SIMD

    *s = sumf;
}

ਖਾਲੀ ggml_vec_silu_f32(ਅਚਲ ਪੂਰਨ n, ਦਸ਼ਮਲਵ * y, ਅਚਲ ਦਸ਼ਮਲਵ * x) {
    ਪੂਰਨ i = 0;
#ਜੇ defined(__AVX512F__) && defined(__AVX512DQ__)
    ਲਈ (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_silu(_mm512_loadu_ps(x + i)));
    }
#ਜਾਂਜੇ defined(__AVX2__) && defined(__FMA__)
    ਲਈ (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_silu(_mm256_loadu_ps(x + i)));
    }
#ਜਾਂਜੇ defined(__SSE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, ggml_v_silu(_mm_loadu_ps(x + i)));
    }
#ਜਾਂਜੇ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    ਅਚਲ ਪੂਰਨ vlen = svcntw();
    ਲਈ (; i < n; i += vlen) {
        ਅਚਲ svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, ggml_v_silu(pg, svld1_f32(pg, x + i)));
    }
#ਜਾਂਜੇ defined(__ARM_NEON) && defined(__aarch64__)
    ਲਈ (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, ggml_v_silu(vld1q_f32(x + i)));
    }
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        vec_xst(ggml_v_silu(vec_xl(0, x + i)), 0, y + i);
    }
#ਜਾਂਜੇ defined(__riscv_v_intrinsic)
    ਲਈ (ਪੂਰਨ vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vy = ggml_v_silu_m2(vx, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#ਜੇ_ਅੰਤ
    ਲਈ (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

ਖਾਲੀ ggml_vec_swiglu_f32(ਅਚਲ ਪੂਰਨ n, ਦਸ਼ਮਲਵ * y, ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅਚਲ ਦਸ਼ਮਲਵ * g) {
    ਪੂਰਨ i = 0;
#ਜੇ defined(__AVX512F__) && defined(__AVX512DQ__)
    ਲਈ (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_mul_ps(ggml_v_silu(_mm512_loadu_ps(x + i)), _mm512_loadu_ps(g + i)));
    }
#ਜਾਂਜੇ defined(__AVX2__) && defined(__FMA__)
    ਲਈ (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(ggml_v_silu(_mm256_loadu_ps(x + i)), _mm256_loadu_ps(g + i)));
    }
#ਜਾਂਜੇ defined(__SSE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, _mm_mul_ps(ggml_v_silu(_mm_loadu_ps(x + i)), _mm_loadu_ps(g + i)));
    }
#ਜਾਂਜੇ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    ਅਚਲ ਪੂਰਨ vlen = svcntw();
    ਲਈ (; i < n; i += vlen) {
        ਅਚਲ svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, svmul_f32_x(pg, ggml_v_silu(pg, svld1_f32(pg, x + i)), svld1_f32(pg, g + i)));
    }
#ਜਾਂਜੇ defined(__ARM_NEON) && defined(__aarch64__)
    ਲਈ (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(ggml_v_silu(vld1q_f32(x + i)), vld1q_f32(g + i)));
    }
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        vec_xst(vec_mul(ggml_v_silu(vec_xl(0, x + i)), vec_xl(0, g + i)), 0, y + i);
    }
#ਜਾਂਜੇ defined(__riscv_v_intrinsic)
    ਲਈ (ਪੂਰਨ vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vg = __riscv_vle32_v_f32m2(&g[i], vl);
        vfloat32m2_t vy = __riscv_vfmul_vv_f32m2(ggml_v_silu_m2(vx, vl), vg, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#ਜੇ_ਅੰਤ
    ਲਈ (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]) * g[i];
    }
}

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_GELU_ਤੇਜ਼_FP16
// computed in f32 without the fp16 table, which loses precision on f32 input
ਖਾਲੀ ggml_vec_gelu_ਤੇਜ਼_f32(ਅਚਲ ਪੂਰਨ n, ਦਸ਼ਮਲਵ * y, ਅਚਲ ਦਸ਼ਮਲਵ * x) {
    ਪੂਰਨ i = 0;
#ਜੇ defined(__AVX512F__) && defined(__AVX512DQ__)
    ਲਈ (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_gelu_ਤੇਜ਼(_mm512_loadu_ps(x + i)));
    }
#ਜਾਂਜੇ defined(__AVX2__) && defined(__FMA__)
    ਲਈ (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_gelu_ਤੇਜ਼(_mm256_loadu_ps(x + i)));
    }
#ਜਾਂਜੇ defined(__SSE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, ggml_v_gelu_ਤੇਜ਼(_mm_loadu_ps(x + i)));
    }
#ਜਾਂਜੇ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    ਅਚਲ ਪੂਰਨ vlen = svcntw();
    ਲਈ (; i < n; i += vlen) {
        ਅਚਲ svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, ggml_v_gelu_ਤੇਜ਼(pg, svld1_f32(pg, x + i)));
    }
#ਜਾਂਜੇ defined(__ARM_NEON) && defined(__aarch64__)
    ਲਈ (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, ggml_v_gelu_ਤੇਜ਼(vld1q_f32(x + i)));
    }
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        vec_xst(ggml_v_gelu_ਤੇਜ਼(vec_xl(0, x + i)), 0, y + i);
    }
#ਜਾਂਜੇ defined(__riscv_v_intrinsic)
    ਲਈ (ਪੂਰਨ vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vy = ggml_v_gelu_ਤੇਜ਼_m2(vx, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#ਜੇ_ਅੰਤ
    ਲਈ (; i < n; ++i) {
        y[i] = ggml_gelu_ਤੇਜ਼_f32(x[i]);
    }
}

ਖਾਲੀ ggml_vec_geglu_ਤੇਜ਼_f32(ਅਚਲ ਪੂਰਨ n, ਦਸ਼ਮਲਵ * y, ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅਚਲ ਦਸ਼ਮਲਵ * g) {
    ਪੂਰਨ i = 0;
#ਜੇ defined(__AVX512F__) && defined(__AVX512DQ__)
    ਲਈ (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_mul_ps(ggml_v_gelu_ਤੇਜ਼(_mm512_loadu_ps(x + i)), _mm512_loadu_ps(g + i)));
    }
#ਜਾਂਜੇ defined(__AVX2__) && defined(__FMA__)
    ਲਈ (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(ggml_v_gelu_ਤੇਜ਼(_mm256_loadu_ps(x + i)), _mm256_loadu_ps(g + i)));
    }
#ਜਾਂਜੇ defined(__SSE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, _mm_mul_ps(ggml_v_gelu_ਤੇਜ਼(_mm_loadu_ps(x + i)), _mm_loadu_ps(g + i)));
    }
#ਜਾਂਜੇ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    ਅਚਲ ਪੂਰਨ vlen = svcntw();
    ਲਈ (; i < n; i += vlen) {
        ਅਚਲ svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, svmul_f32_x(pg, ggml_v_gelu_ਤੇਜ਼(pg, svld1_f32(pg, x + i)), svld1_f32(pg, g + i)));
    }
#ਜਾਂਜੇ defined(__ARM_NEON) && defined(__aarch64__)
    ਲਈ (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(ggml_v_gelu_ਤੇਜ਼(vld1q_f32(x + i)), vld1q_f32(g + i)));
    }
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        vec_xst(vec_mul(ggml_v_gelu_ਤੇਜ਼(vec_xl(0, x + i)), vec_xl(0, g + i)), 0, y + i);
    }
#ਜਾਂਜੇ defined(__riscv_v_intrinsic)
    ਲਈ (ਪੂਰਨ vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vg = __riscv_vle32_v_f32m2(&g[i], vl);
        vfloat32m2_t vy = __riscv_vfmul_vv_f32m2(ggml_v_gelu_ਤੇਜ਼_m2(vx, vl), vg, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#ਜੇ_ਅੰਤ
    ਲਈ (; i < n; ++i) {
        y[i] = ggml_gelu_ਤੇਜ਼_f32(x[i]) * g[i];
    }
}
#ਜੇ_ਅੰਤ

ggml_float ggml_vec_cvar_f32(ਅਚਲ ਪੂਰਨ n, ਦਸ਼ਮਲਵ * y, ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅਚਲ ਦਸ਼ਮਲਵ ਔਸਤ_2) {
    ਪੂਰਨ i = 0;
    ggml_float ਜੋੜ = 0;
// TODO: optimize to process the remaining elements in groups using the smaller vector sizes from AVX2 and SSE
// ref: https://github.com/ggml-org/llama.cpp/pull/15953#pullrequestreview-3310928344
#ਜੇ defined(__AVX512F__) && defined(__AVX512DQ__)
    ਲਈ (; i + 15 < n; i += 16) {
        __m512 val = _mm512_sub_ps(_mm512_loadu_ps(x + i),
                                   _mm512_set1_ps(ਔਸਤ_2));
        _mm512_storeu_ps(y + i, val);
        ਜੋੜ += (ggml_float)mm512_reduce_ਜੋੜੋ_ps(_mm512_mul_ps(val, val));
    }
#ਜਾਂਜੇ defined(__AVX2__) && defined(__FMA__)
    ਲਈ (; i + 7 < n; i += 8) {
        __m256 val = _mm256_sub_ps(_mm256_loadu_ps(x + i),
                                   _mm256_set1_ps(ਔਸਤ_2));
        _mm256_storeu_ps(y + i, val);
        val = _mm256_mul_ps(val,val);
        __m128 val2 = mm_ਜੋੜੋ_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = mm_ਜੋੜੋ_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = mm_ਜੋੜੋ_ss(val2, _mm_movehdup_ps(val2));
        ਜੋੜ += (ggml_float)_mm_cvtss_f32(val2);
    }
#ਜਾਂਜੇ defined(__SSE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        __m128 val = _mm_sub_ps(_mm_loadu_ps(x + i),
                                _mm_set1_ps(ਔਸਤ_2));
        _mm_storeu_ps(y + i, val);
        val = _mm_mul_ps(val, val);
#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = mm_ਜੋੜੋ_ps(val, _mm_movehl_ps(val, val));
        val = mm_ਜੋੜੋ_ss(val, _mm_movehdup_ps(val));
#ਨਹੀਂਤਾਂ
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = mm_ਜੋੜੋ_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = mm_ਜੋੜੋ_ss(val, tmp);
#ਜੇ_ਅੰਤ  // __AVX__ || __AVX2__ || __AVX512F__
        ਜੋੜ += (ggml_float)_mm_cvtss_f32(val);
    }
#ਜਾਂਜੇ defined(__ARM_NEON) && defined(__aarch64__)
    ਲਈ (; i + 3 < n; i += 4) {
        float32x4_t val = vsubq_f32(vld1q_f32(x + i),
                                    vdupq_n_f32(ਔਸਤ_2));
        vst1q_f32(y + i, val);
        val = vmulq_f32(val, val);
        ਜੋੜ += (ggml_float)vaddvq_f32(val);
    }
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        float32x4_t val = vec_sub(vec_xl(0, x + i), vec_splats(ਔਸਤ_2));
        vec_xst(val, 0, y + i);
        val = vec_mul(val, val);
        ਜੋੜ += (ggml_float)vec_hsum_f32x4(val);
    }
#ਜਾਂਜੇ defined(__riscv_v_intrinsic)
    vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
    ਲਈ (ਪੂਰਨ vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t val = __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&x[i], vl), ਔਸਤ_2, vl);
        __riscv_vse32_v_f32m2(&y[i], val, vl);
        val = __riscv_vfmul_vv_f32m2(val, val, vl);
        vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, vl);
    }
    ਜੋੜ = (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
#ਜੇ_ਅੰਤ
    ਲਈ (; i < n; ++i) {
        ਦਸ਼ਮਲਵ val = x[i] - ਔਸਤ_2;
        y[i] = val;
        val *= val;
        ਜੋੜ += (ggml_float)val;
    }
    ਵਾਪਸ ਜੋੜ/n;
}

ggml_float ggml_vec_soft_max_f32(ਅਚਲ ਪੂਰਨ n, ਦਸ਼ਮਲਵ * y, ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ max) {
    ਪੂਰਨ i = 0;
    ggml_float ਜੋੜ = 0;
#ਜੇ defined(__AVX512F__) && defined(__AVX512DQ__)
    ਲਈ (; i + 15 < n; i += 16) {
        __m512 val = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        ਜੋੜ += (ggml_float)mm512_reduce_ਜੋੜੋ_ps(val);
    }
#ਜਾਂਜੇ defined(__AVX2__) && defined(__FMA__)
    ਲਈ (; i + 7 < n; i += 8) {
        __m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = mm_ਜੋੜੋ_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = mm_ਜੋੜੋ_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = mm_ਜੋੜੋ_ss(val2, _mm_movehdup_ps(val2));
        ਜੋੜ += (ggml_float)_mm_cvtss_f32(val2);
    }
#ਜਾਂਜੇ defined(__SSE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        __m128 val = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#ਜੇ defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = mm_ਜੋੜੋ_ps(val, _mm_movehl_ps(val, val));
        val = mm_ਜੋੜੋ_ss(val, _mm_movehdup_ps(val));
#ਨਹੀਂਤਾਂ
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = mm_ਜੋੜੋ_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = mm_ਜੋੜੋ_ss(val, tmp);
#ਜੇ_ਅੰਤ
        ਜੋੜ += (ggml_float)_mm_cvtss_f32(val);
    }
#ਜਾਂਜੇ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    ਅਚਲ ਪੂਰਨ vlen = svcntw();
    ਲਈ (; i < n; i += vlen) {
        ਅਚਲ svbool_t pg = svwhilelt_b32_s32(i, n);
        svfloat32_t val = ggml_v_expf(pg, svsub_f32_x(pg, svld1_f32(pg, x + i),
                                                svdup_n_f32_x(pg, max)));
        svst1_f32(pg, y + i, val);
        ਜੋੜ += (ggml_float)svaddv_f32(pg, val);
    }
#ਜਾਂਜੇ defined(__ARM_NEON) && defined(__aarch64__)
    ਲਈ (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        ਜੋੜ += (ggml_float)vaddvq_f32(val);
    }
#ਜਾਂਜੇ defined(__VXE__) || defined(__VXE2__)
    ਲਈ (; i + 3 < n; i += 4) {
        float32x4_t v_val = ggml_v_expf(vec_sub(vec_xl(0, x + i), vec_splats(max)));
        vec_xst(v_val, 0, y + i);
        ਜੋੜ += (ggml_float)vec_hsum_f32x4(v_val);
    }
#ਜਾਂਜੇ defined(__riscv_v_intrinsic)
    vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
    ਲਈ (ਪੂਰਨ avl; i < n; i += avl) {
        avl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t val = ggml_v_expf_m2(__riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&x[i], avl), max, avl), avl);
        __riscv_vse32_v_f32m2(&y[i], val, avl);
        vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, avl);
    }
    ਵਾਪਸ (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
#ਜੇ_ਅੰਤ
    ਲਈ (; i < n; ++i) {
        ਦਸ਼ਮਲਵ val = expf(x[i] - max);
        ਜੋੜ += (ggml_float)val;
        y[i] = val;
    }
    ਵਾਪਸ ਜੋੜ;
}

ggml_float ggml_vec_log_soft_max_f32(ਅਚਲ ਪੂਰਨ n, ਦਸ਼ਮਲਵ * y, ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    ਪੂਰਨ i = 0;
    ggml_float ਜੋੜ = 0;
    ਲਈ (; i < n; ++i) {
        ਦਸ਼ਮਲਵ val = x[i] - max;
        y[i] = val;
        ਜੋੜ += (ggml_float)expf(val);
    }
    ਵਾਪਸ ਜੋੜ = (ggml_float)logf(ਜੋੜ);
}
