#నిర్వచించు GGML_COMMON_IMPL_CPP
#నిర్వచించు GGML_COMMON_DECL_CPP
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml-backend-impl.h"

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ggml-cpu-impl.h"
#అన్తర్భావ "simd-mappings.h"
#అన్తర్భావ "traits.h"

#అన్తర్భావ <cmath>
#అన్తర్భావ <cstring>
#అన్తర్భావ <cassert>

#నిర్వచించు GGML_CPU_CLANG_WORKAROUND
#అన్తర్భావ "../../repack.h"

#నిర్వచించు వాడని GGML_వాడని

శూన్య ggml_quantize_mat_q8_0_4x4(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK8_0 == 32);
    ప్రతిజ్ఞా(k % QK8_0 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK8_0;

    ఖండం_q8_0x4 * GGML_RESTRICT y = (ఖండం_q8_0x4 *) vy;

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    float32x4_t v_src[4][8];
    దశాంశం       id[4];

    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        float32x4_t v_asrc[8];
        float32x4_t v_amax[8];

        కొరకు (పూర్ణాంకం అడ్డువరుస_iter = 0; అడ్డువరుస_iter < 4; అడ్డువరుస_iter++) {
            కొరకు (పూర్ణాంకం j = 0; j < 8; j++) v_src[అడ్డువరుస_iter][j] = vec_xl(0, x + అడ్డువరుస_iter * k + i * 32 + 4 * j);
            కొరకు (పూర్ణాంకం j = 0; j < 8; j++) v_asrc[j] = vec_abs(v_src[అడ్డువరుస_iter][j]);

            కొరకు (పూర్ణాంకం j = 0; j < 4; j++) v_amax[2 * j] = vec_max(v_asrc[2 * j], v_asrc[2 * j + 1]);
            కొరకు (పూర్ణాంకం j = 0; j < 2; j++) v_amax[4 * j] = vec_max(v_amax[4 * j], v_amax[4 * j + 2]);
            కొరకు (పూర్ణాంకం j = 0; j < 1; j++) v_amax[8 * j] = vec_max(v_amax[8 * j], v_amax[8 * j + 4]);

            స్థిరరాశి దశాంశం amax = MAX(MAX(vec_వెలికితీయి(v_amax[0], 0), vec_వెలికితీయి(v_amax[0], 1)),
                                   MAX(vec_వెలికితీయి(v_amax[0], 2), vec_వెలికితీయి(v_amax[0], 3)));

            స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
            id[అడ్డువరుస_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[అడ్డువరుస_iter] = GGML_CPU_FP32_గమ్యం_FP16(d);
        }

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            /* Uses non-default rounding for vec_signed or vec_round */
            స్థిరరాశి int32x4_t v_qs0 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[0][j], id[0]), 4, 1));
            స్థిరరాశి int32x4_t v_qs1 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[1][j], id[1]), 4, 1));
            స్థిరరాశి int32x4_t v_qs2 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[2][j], id[2]), 4, 1));
            స్థిరరాశి int32x4_t v_qs3 = vec_signed(__builtin_s390_vfisb(vec_mul(v_src[3][j], id[3]), 4, 1));

            స్థిరరాశి int16x8_t v_qs01 = vec_packs(v_qs0, v_qs1);
            స్థిరరాశి int16x8_t v_qs23 = vec_packs(v_qs2, v_qs3);

            vec_xst(vec_packs(v_qs01, v_qs23), 0, y[i].qs + 16 * j);
        }
    }
#లేకపోతే
    వాడని(nb);
    వాడని(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#అయితే_ముగింపు
}

శూన్య ggml_gemv_q4_0_4x4_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    దశాంశం * res_ptr = s;

    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q4_0x4 * b_ptr = (స్థిరరాశి ఖండం_q4_0x4 *) vx + (x * nb);

        float32x4_t v_sumf = vec_splats(0.0f);

        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            స్థిరరాశి int8_t * x_qs = b_ptr[l].qs;

            స్థిరరాశి int8x16_t v_x0 = vec_xl( 0, x_qs);
            స్థిరరాశి int8x16_t v_x1 = vec_xl(16, x_qs);
            స్థిరరాశి int8x16_t v_x2 = vec_xl(32, x_qs);
            స్థిరరాశి int8x16_t v_x3 = vec_xl(48, x_qs);

            స్థిరరాశి int8x16_t v_x0l = vec_sra(vec_sl(v_x0, 4), 4);
            స్థిరరాశి int8x16_t v_x1l = vec_sra(vec_sl(v_x1, 4), 4);
            స్థిరరాశి int8x16_t v_x2l = vec_sra(vec_sl(v_x2, 4), 4);
            స్థిరరాశి int8x16_t v_x3l = vec_sra(vec_sl(v_x3, 4), 4);

            స్థిరరాశి int8x16_t v_x0h = vec_sra(v_x0, 4);
            స్థిరరాశి int8x16_t v_x1h = vec_sra(v_x1, 4);
            స్థిరరాశి int8x16_t v_x2h = vec_sra(v_x2, 4);
            స్థిరరాశి int8x16_t v_x3h = vec_sra(v_x3, 4);

            స్థిరరాశి int8_t * y_lo = a_ptr[l].qs;
            స్థిరరాశి int8_t * y_hi = y_lo + qk / 2;

            int16x8_t v_sumi = vec_splats((int16_t)0);

            v_sumi = ggml_vec_dot_s16(v_sumi, v_x0l, ggml_vec_xl_s8x4_splat(y_lo +  0));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x1l, ggml_vec_xl_s8x4_splat(y_lo +  4));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x2l, ggml_vec_xl_s8x4_splat(y_lo +  8));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x3l, ggml_vec_xl_s8x4_splat(y_lo + 12));

            v_sumi = ggml_vec_dot_s16(v_sumi, v_x0h, ggml_vec_xl_s8x4_splat(y_hi +  0));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x1h, ggml_vec_xl_s8x4_splat(y_hi +  4));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x2h, ggml_vec_xl_s8x4_splat(y_hi +  8));
            v_sumi = ggml_vec_dot_s16(v_sumi, v_x3h, ggml_vec_xl_s8x4_splat(y_hi + 12));

            స్థిరరాశి float32x4_t v_yd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d));
            స్థిరరాశి float32x4_t v_xd = ggml_vec_xl_f16x4_finite(b_ptr[l].d);
            స్థిరరాశి float32x4_t v_d  = vec_mul(v_yd, v_xd);

            v_sumf = vec_madd(vec_float(vec_paddl_s16(v_sumi)), v_d, v_sumf);
        }

        vec_xst(v_sumf, 0, res_ptr + x * ncols_interleaved);
    }
#లేకపోతే
    వాడని(nb);
    వాడని(ncols_interleaved);
    ggml_gemv_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#అయితే_ముగింపు
}

శూన్య ggml_gemm_q4_0_4x4_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;

    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);

        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q4_0x4 * b_ptr = (స్థిరరాశి ఖండం_q4_0x4 *) vx + (x * nb);

            float32x4_t v_sumf[4];
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                v_sumf[m] = vec_splats(0.0f);
            }

            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                int16x8_t v_sumi0 = vec_splats((int16_t)0);
                int16x8_t v_sumi1 = vec_splats((int16_t)0);
                int16x8_t v_sumi2 = vec_splats((int16_t)0);
                int16x8_t v_sumi3 = vec_splats((int16_t)0);

                కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                    స్థిరరాశి int8x16_t v_x  = vec_xl(0, b_ptr[l].qs + 16 * k);
                    స్థిరరాశి int8x16_t v_xl = vec_sra(vec_sl(v_x, 4), 4);
                    స్థిరరాశి int8x16_t v_xh = vec_sra(v_x, 4);

                    స్థిరరాశి int8_t * y_lo = a_ptr[l].qs + 16 * k;
                    స్థిరరాశి int8_t * y_hi = y_lo + qk / 2 * 4;

                    v_sumi0 = ggml_vec_dot_s16(v_sumi0, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  0));
                    v_sumi1 = ggml_vec_dot_s16(v_sumi1, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  4));
                    v_sumi2 = ggml_vec_dot_s16(v_sumi2, v_xl, ggml_vec_xl_s8x4_splat(y_lo +  8));
                    v_sumi3 = ggml_vec_dot_s16(v_sumi3, v_xl, ggml_vec_xl_s8x4_splat(y_lo + 12));

                    v_sumi0 = ggml_vec_dot_s16(v_sumi0, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  0));
                    v_sumi1 = ggml_vec_dot_s16(v_sumi1, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  4));
                    v_sumi2 = ggml_vec_dot_s16(v_sumi2, v_xh, ggml_vec_xl_s8x4_splat(y_hi +  8));
                    v_sumi3 = ggml_vec_dot_s16(v_sumi3, v_xh, ggml_vec_xl_s8x4_splat(y_hi + 12));
                }

                స్థిరరాశి float32x4_t v_yd = __lzs_f16cx4_load(a_ptr[l].d);
                స్థిరరాశి float32x4_t v_xd = ggml_vec_xl_f16x4_finite(b_ptr[l].d);

                v_sumf[0] = vec_madd(vec_float(vec_paddl_s16(v_sumi0)), vec_mul(v_xd, vec_splat(v_yd, 0)), v_sumf[0]);
                v_sumf[1] = vec_madd(vec_float(vec_paddl_s16(v_sumi1)), vec_mul(v_xd, vec_splat(v_yd, 1)), v_sumf[1]);
                v_sumf[2] = vec_madd(vec_float(vec_paddl_s16(v_sumi2)), vec_mul(v_xd, vec_splat(v_yd, 2)), v_sumf[2]);
                v_sumf[3] = vec_madd(vec_float(vec_paddl_s16(v_sumi3)), vec_mul(v_xd, vec_splat(v_yd, 3)), v_sumf[3]);
            }

            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                vec_xst(v_sumf[m], 0, s + (y * 4 + m) * bs + x * ncols_interleaved);
            }
        }
    }
#లేకపోతే
    వాడని(nb);
    వాడని(ncols_interleaved);
    ggml_gemm_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
#అయితే_ముగింపు
}
