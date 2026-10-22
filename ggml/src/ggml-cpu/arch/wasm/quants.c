#నిర్వచించు GGML_COMMON_IMPL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml-quants.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "simd-mappings.h"

#అన్తర్భావ "../../quants.h"
#అన్తర్భావ "../../ggml-cpu-impl.h"

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <assert.h>
#అన్తర్భావ <float.h>
#అన్తర్భావ <stdlib.h> // for qsort
#అన్తర్భావ <stdio.h>  // for GGML_ASSERT

#నిర్వచించు సమూహం_MAX_EPS 1e-15f
#నిర్వచించు సమూహం_MAX_EPS_IQ3_XXS 1e-8f
#నిర్వచించు సమూహం_MAX_EPS_IQ2_S 1e-8f
#నిర్వచించు సమూహం_MAX_EPS_IQ1_M 1e-7f
#నిర్వచించు సమూహం_MAX_EPS_IQ1_S 1e-12f

#నిర్వచించు వాడని GGML_వాడని

#ఒకవేళ defined(__wasm_simd128__)
#నిర్వచించు B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#నిర్వచించు B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#నిర్వచించు B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#నిర్వచించు B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#నిర్వచించు B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#నిర్వచించు B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#నిర్వచించు B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#నిర్వచించు B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
స్థిర స్థిరరాశి uint64_t పట్టిక_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
స్థిర స్థిరరాశి uint64_t పట్టిక_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#అయితే_ముగింపు

శూన్య quantize_అడ్డువరుస_q8_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK8_0 == 32);
    ప్రతిజ్ఞా(k % QK8_0 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK8_0;

    ఖండం_q8_0 * GGML_RESTRICT y = vy;

#ఒకవేళ defined __wasm_simd128__
    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) srcv[j]  = wasm_v128_ఎక్కించు(x + i*32 + 4*j);
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        కొరకు (పూర్ణాంకం j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        కొరకు (పూర్ణాంకం j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        కొరకు (పూర్ణాంకం j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        స్థిరరాశి దశాంశం amax = MAX(MAX(wasm_f32x4_వెలికితీయి_lane(amaxv[0], 0),
                                   wasm_f32x4_వెలికితీయి_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_వెలికితీయి_lane(amaxv[0], 2),
                                   wasm_f32x4_వెలికితీయి_lane(amaxv[0], 3)));

        స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
        స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_గమ్యం_FP16(d);

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            స్థిరరాశి v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            స్థిరరాశి v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_వెలికితీయి_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_వెలికితీయి_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_వెలికితీయి_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_వెలికితీయి_lane(vi, 3);
        }
    }
#లేకపోతే
    GGML_వాడని(nb);
    // scalar
    quantize_అడ్డువరుస_q8_0_ref(x, y, k);
#అయితే_ముగింపు
}

శూన్య quantize_అడ్డువరుస_q8_1(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(k % QK8_1 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK8_1;

    ఖండం_q8_1 * GGML_RESTRICT y = vy;
#ఒకవేళ defined __wasm_simd128__
    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) srcv[j]  = wasm_v128_ఎక్కించు(x + i*32 + 4*j);
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        కొరకు (పూర్ణాంకం j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        కొరకు (పూర్ణాంకం j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        కొరకు (పూర్ణాంకం j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        స్థిరరాశి దశాంశం amax = MAX(MAX(wasm_f32x4_వెలికితీయి_lane(amaxv[0], 0),
                                   wasm_f32x4_వెలికితీయి_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_వెలికితీయి_lane(amaxv[0], 2),
                                   wasm_f32x4_వెలికితీయి_lane(amaxv[0], 3)));

        స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
        స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_గమ్యం_FP16(d);

        v128_t accv = wasm_i32x4_splat(0);

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            స్థిరరాశి v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            స్థిరరాశి v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_వెలికితీయి_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_వెలికితీయి_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_వెలికితీయి_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_వెలికితీయి_lane(vi, 3);

            accv = wasm_i32x4_కూడు(accv, vi);
        }

        y[i].s = GGML_CPU_FP32_గమ్యం_FP16(
                d * (wasm_i32x4_వెలికితీయి_lane(accv, 0) +
                     wasm_i32x4_వెలికితీయి_lane(accv, 1) +
                     wasm_i32x4_వెలికితీయి_lane(accv, 2) +
                     wasm_i32x4_వెలికితీయి_lane(accv, 3)));
    }
#లేకపోతే
    GGML_వాడని(nb);
    // scalar
    quantize_అడ్డువరుస_q8_1_ref(x, y, k);
#అయితే_ముగింపు
}

//===================================== Q8_K ==============================================

శూన్య quantize_అడ్డువరుస_q8_K(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT y, int64_t k) {
#నిర్వచితమైతే __wasm_simd128__
    ప్రతిజ్ఞా(k % QK_K == 0);
    స్థిరరాశి int64_t nb = k / QK_K;
    ఖండం_q8_K * GGML_RESTRICT yc = y; // Cast to proper type

    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        స్థిరరాశి దశాంశం * x_ఖండం = x + i * QK_K;

        v128_t min_vec = wasm_v128_ఎక్కించు(x_ఖండం);
        v128_t max_vec = min_vec;

        కొరకు (పూర్ణాంకం j = 4; j < QK_K; j += 4) {
            v128_t x_vec = wasm_v128_ఎక్కించు(x_ఖండం + j);
            max_vec = wasm_f32x4_pmax(max_vec, x_vec);
            min_vec = wasm_f32x4_pmin(min_vec, x_vec);
        }
        max_vec = wasm_f32x4_pmax(max_vec, wasm_i32x4_shuffle(max_vec, max_vec, 2, 3, 0, 1));
        max_vec = wasm_f32x4_pmax(max_vec, wasm_i32x4_shuffle(max_vec, max_vec, 1, 0, 3, 2));
        min_vec = wasm_f32x4_pmin(min_vec, wasm_i32x4_shuffle(min_vec, min_vec, 2, 3, 0, 1));
        min_vec = wasm_f32x4_pmin(min_vec, wasm_i32x4_shuffle(min_vec, min_vec, 1, 0, 3, 2));
        దశాంశం max = wasm_f32x4_వెలికితీయి_lane(max_vec, 0);
        దశాంశం min = wasm_f32x4_వెలికితీయి_lane(min_vec, 0);
        దశాంశం amax = -min > max ? min : max;

        ఒకవేళ (amax == 0.0f) {
            yc[i].d = 0.0f;
            స్థిరరాశి v128_t zero = wasm_i8x16_splat(0);
            కొరకు (పూర్ణాంకం j = 0; j < QK_K; j += 16) {
                wasm_v128_store(yc[i].qs + j, zero);
            }
            కొనసాగు;
        }

        స్థిరరాశి దశాంశం iscale = -127.0f / amax;
        స్థిరరాశి v128_t scale_vec = wasm_f32x4_splat(iscale);

        // Process 16 elements per iteration
        కొరకు (పూర్ణాంకం j = 0, jb = 0; j < QK_K; j += 16, jb++) {
            // Load and quantize 16 floats
            v128_t x0 = wasm_v128_ఎక్కించు(x_ఖండం + j);
            v128_t x1 = wasm_v128_ఎక్కించు(x_ఖండం + j + 4);
            v128_t x2 = wasm_v128_ఎక్కించు(x_ఖండం + j + 8);
            v128_t x3 = wasm_v128_ఎక్కించు(x_ఖండం + j + 12);

            v128_t q0 = wasm_f32x4_సమీప(wasm_f32x4_mul(x0, scale_vec));
            v128_t q1 = wasm_f32x4_సమీప(wasm_f32x4_mul(x1, scale_vec));
            v128_t q2 = wasm_f32x4_సమీప(wasm_f32x4_mul(x2, scale_vec));
            v128_t q3 = wasm_f32x4_సమీప(wasm_f32x4_mul(x3, scale_vec));

            // Convert to i32 with saturation
            v128_t i0 = wasm_i32x4_trunc_sat_f32x4(q0);
            v128_t i1 = wasm_i32x4_trunc_sat_f32x4(q1);
            v128_t i2 = wasm_i32x4_trunc_sat_f32x4(q2);
            v128_t i3 = wasm_i32x4_trunc_sat_f32x4(q3);

            // Pack into 16 i8 values
            v128_t i8 = wasm_i8x16_narrow_i16x8(
                wasm_i16x8_narrow_i32x4(i0, i1),
                wasm_i16x8_narrow_i32x4(i2, i3)
            );
            wasm_v128_store(yc[i].qs + j, i8);

            // Calculate bsums using SIMD
            v128_t sum16 = wasm_i16x8_కూడు(
                wasm_i16x8_extend_దిగువ_i8x16(i8),
                wasm_i16x8_extend_ఎగువ_i8x16(i8)
            );
            v128_t sum32 = wasm_i32x4_కూడు(
                wasm_i32x4_extend_దిగువ_i16x8(sum16),
                wasm_i32x4_extend_ఎగువ_i16x8(sum16)
            );
            sum32 = wasm_i32x4_కూడు(sum32, wasm_i32x4_shuffle(sum32, sum32, 2, 3, 0, 1));
            sum32 = wasm_i32x4_కూడు(sum32, wasm_i32x4_shuffle(sum32, sum32, 1, 0, 3, 2));
            yc[i].bsums[jb] = wasm_i32x4_వెలికితీయి_lane(sum32, 0);
        }

        yc[i].d = 1.0f / iscale;
    }
#లేకపోతే
    quantize_అడ్డువరుస_q8_K_ref(x, y, k);
#అయితే_ముగింపు
}


//===================================== Dot products =================================

శూన్య ggml_vec_dot_q4_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q4_0 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y = vy;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

#ఒకవేళ defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);

    స్థిరరాశి v128_t m4b = wasm_i8x16_splat(0x0F);
    స్థిరరాశి v128_t s8b = wasm_i8x16_splat(0x8);

    కొరకు (; ib + 1 < nb; ib += 2) {
        స్థిరరాశి ఖండం_q4_0 * GGML_RESTRICT x0 = &x[ib];
        స్థిరరాశి ఖండం_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
        స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y0 = &y[ib];
        స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

        // Load and process x0
        v128_t v0_0 = wasm_v128_ఎక్కించు(x0->qs);
        v128_t v0_0l = wasm_v128_and(v0_0, m4b);
        v128_t v0_0h = wasm_u8x16_shr(v0_0, 4);
        v128_t v0_0ls = wasm_i8x16_sub(v0_0l, s8b);
        v128_t v0_0hs = wasm_i8x16_sub(v0_0h, s8b);

        // Load y0 vectors
        v128_t y0_l = wasm_v128_ఎక్కించు(y0->qs);
        v128_t y0_h = wasm_v128_ఎక్కించు(y0->qs + 16);

        // Extend to i16x8 and compute dot products
        v128_t dx0l = wasm_i16x8_extend_దిగువ_i8x16(v0_0ls);
        v128_t dx0h = wasm_i16x8_extend_ఎగువ_i8x16(v0_0ls);
        v128_t dx0hl = wasm_i16x8_extend_దిగువ_i8x16(v0_0hs);
        v128_t dx0hh = wasm_i16x8_extend_ఎగువ_i8x16(v0_0hs);

        v128_t dy0ll = wasm_i16x8_extend_దిగువ_i8x16(y0_l);
        v128_t dy0lh = wasm_i16x8_extend_ఎగువ_i8x16(y0_l);
        v128_t dy0hl = wasm_i16x8_extend_దిగువ_i8x16(y0_h);
        v128_t dy0hh = wasm_i16x8_extend_ఎగువ_i8x16(y0_h);

        v128_t dp0 = wasm_i32x4_కూడు(
            wasm_i32x4_కూడు(
                wasm_i32x4_dot_i16x8(dx0l, dy0ll),
                wasm_i32x4_dot_i16x8(dx0h, dy0lh)
            ),
            wasm_i32x4_కూడు(
                wasm_i32x4_dot_i16x8(dx0hl, dy0hl),
                wasm_i32x4_dot_i16x8(dx0hh, dy0hh)
            )
        );

        // Load and process x1
        v128_t v0_1 = wasm_v128_ఎక్కించు(x1->qs);
        v128_t v0_1l = wasm_v128_and(v0_1, m4b);
        v128_t v0_1h = wasm_u8x16_shr(v0_1, 4);
        v128_t v0_1ls = wasm_i8x16_sub(v0_1l, s8b);
        v128_t v0_1hs = wasm_i8x16_sub(v0_1h, s8b);

        // Load y1 vectors
        v128_t y1_l = wasm_v128_ఎక్కించు(y1->qs);
        v128_t y1_h = wasm_v128_ఎక్కించు(y1->qs + 16);

        // Extend to i16x8 and compute dot products
        v128_t dx1l = wasm_i16x8_extend_దిగువ_i8x16(v0_1ls);
        v128_t dx1h = wasm_i16x8_extend_ఎగువ_i8x16(v0_1ls);
        v128_t dx1hl = wasm_i16x8_extend_దిగువ_i8x16(v0_1hs);
        v128_t dx1hh = wasm_i16x8_extend_ఎగువ_i8x16(v0_1hs);

        v128_t dy1ll = wasm_i16x8_extend_దిగువ_i8x16(y1_l);
        v128_t dy1lh = wasm_i16x8_extend_ఎగువ_i8x16(y1_l);
        v128_t dy1hl = wasm_i16x8_extend_దిగువ_i8x16(y1_h);
        v128_t dy1hh = wasm_i16x8_extend_ఎగువ_i8x16(y1_h);

        v128_t dp1 = wasm_i32x4_కూడు(
            wasm_i32x4_కూడు(
                wasm_i32x4_dot_i16x8(dx1l, dy1ll),
                wasm_i32x4_dot_i16x8(dx1h, dy1lh)
            ),
            wasm_i32x4_కూడు(
                wasm_i32x4_dot_i16x8(dx1hl, dy1hl),
                wasm_i32x4_dot_i16x8(dx1hh, dy1hh)
            )
        );

        // Accumulate results with scaling
        దశాంశం scale0 = GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d);
        దశాంశం scale1 = GGML_CPU_FP16_గమ్యం_FP32(x1->d) * GGML_CPU_FP16_గమ్యం_FP32(y1->d);

        sumv = wasm_f32x4_కూడు(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(dp0), wasm_f32x4_splat(scale0)));
        sumv = wasm_f32x4_కూడు(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(dp1), wasm_f32x4_splat(scale1)));
    }

    sumf = wasm_f32x4_వెలికితీయి_lane(sumv, 0) + wasm_f32x4_వెలికితీయి_lane(sumv, 1) +
           wasm_f32x4_వెలికితీయి_lane(sumv, 2) + wasm_f32x4_వెలికితీయి_lane(sumv, 3);

#అయితే_ముగింపు
    కొరకు (; ib < nb; ++ib) {
        పూర్ణాంకం sumi0 = 0;
        పూర్ణాంకం sumi1 = 0;

        కొరకు (పూర్ణాంకం j = 0; j < qk/2; ++j) {
            స్థిరరాశి పూర్ణాంకం v0 = (x[ib].qs[j] & 0x0F) - 8;
            స్థిరరాశి పూర్ణాంకం v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        పూర్ణాంకం sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_గమ్యం_FP32(x[ib].d)*GGML_CPU_FP16_గమ్యం_FP32(y[ib].d);
    }

    *s = sumf;
}

శూన్య ggml_vec_dot_q4_1_q8_1(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_1;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q4_1 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y = vy;

    దశాంశం sumf = 0;

#ఒకవేళ defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);
    దశాంశం summs = 0.0f;

    కొరకు (పూర్ణాంకం ib = 0; ib < nb; ++ib) {
        స్థిరరాశి ఖండం_q4_1 * GGML_RESTRICT x0 = &x[ib];
        స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y0 = &y[ib];

        summs += GGML_CPU_FP16_గమ్యం_FP32(x0->m) * GGML_CPU_FP16_గమ్యం_FP32(y0->s);

        స్థిరరాశి v128_t raw  = wasm_v128_ఎక్కించు(x0->qs);
        స్థిరరాశి v128_t v0s  = wasm_v128_and(raw, wasm_i8x16_splat(0x0F));
        స్థిరరాశి v128_t v1s  = wasm_u8x16_shr(raw, 4);

        స్థిరరాశి v128_t ys_lo = wasm_v128_ఎక్కించు(y0->qs);
        స్థిరరాశి v128_t ys_hi = wasm_v128_ఎక్కించు(y0->qs + 16);

        స్థిరరాశి v128_t v0s_l = wasm_u16x8_extend_దిగువ_u8x16(v0s);
        స్థిరరాశి v128_t v0s_h = wasm_u16x8_extend_ఎగువ_u8x16(v0s);
        స్థిరరాశి v128_t ylo_l = wasm_i16x8_extend_దిగువ_i8x16(ys_lo);
        స్థిరరాశి v128_t ylo_h = wasm_i16x8_extend_ఎగువ_i8x16(ys_lo);
        స్థిరరాశి v128_t v1s_l = wasm_u16x8_extend_దిగువ_u8x16(v1s);
        స్థిరరాశి v128_t v1s_h = wasm_u16x8_extend_ఎగువ_u8x16(v1s);
        స్థిరరాశి v128_t yhi_l = wasm_i16x8_extend_దిగువ_i8x16(ys_hi);
        స్థిరరాశి v128_t yhi_h = wasm_i16x8_extend_ఎగువ_i8x16(ys_hi);

        స్థిరరాశి v128_t acc = wasm_i32x4_కూడు(
            wasm_i32x4_కూడు(
                wasm_i32x4_dot_i16x8(v0s_l, ylo_l),
                wasm_i32x4_dot_i16x8(v0s_h, ylo_h)),
            wasm_i32x4_కూడు(
                wasm_i32x4_dot_i16x8(v1s_l, yhi_l),
                wasm_i32x4_dot_i16x8(v1s_h, yhi_h)));

        sumv = wasm_f32x4_కూడు(sumv,
            wasm_f32x4_mul(
                wasm_f32x4_convert_i32x4(acc),
                wasm_f32x4_splat(GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d))));
    }

    sumf = wasm_f32x4_వెలికితీయి_lane(sumv, 0) + wasm_f32x4_వెలికితీయి_lane(sumv, 1) +
           wasm_f32x4_వెలికితీయి_lane(sumv, 2) + wasm_f32x4_వెలికితీయి_lane(sumv, 3) + summs;

    *s = sumf;

#లేకపోతే
    వాడని(nb);
    వాడని(x);
    వాడని(y);
    వాడని(sumf);

    ggml_vec_dot_q4_1_q8_1_generic(
        n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q5_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(qk == QK5_0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q5_0 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y = vy;

#ఒకవేళ defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);

    uint32_t qh_;
    uint64_t tmp[4];

    // TODO: check if unrolling this is better
    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి ఖండం_q5_0 * GGML_RESTRICT x0 = &x[ib];
        స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y0 = &y[ib];

        స్థిరరాశి v128_t m4b  = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        జ్ఞాపకం_నకలు(&qh_, x0->qh, పరిమాణం(qh_));

        tmp[0] = పట్టిక_b2b_1[(qh_ >>  0) & 0xFF];
        tmp[1] = పట్టిక_b2b_1[(qh_ >>  8) & 0xFF];
        tmp[2] = పట్టిక_b2b_1[(qh_ >> 16) & 0xFF];
        tmp[3] = పట్టిక_b2b_1[(qh_ >> 24)       ];

        స్థిరరాశి v128_t qhl = wasm_v128_ఎక్కించు(tmp + 0);
        స్థిరరాశి v128_t qhh = wasm_v128_ఎక్కించు(tmp + 2);

        స్థిరరాశి v128_t v0 = wasm_v128_ఎక్కించు(x0->qs);

        // 4-bit -> 8-bit
        స్థిరరాశి v128_t v0l = wasm_v128_and (v0, m4b);
        స్థిరరాశి v128_t v0h = wasm_u8x16_shr(v0, 4);

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        స్థిరరాశి v128_t v0lf = wasm_i8x16_sub(v0l, qhl);
        స్థిరరాశి v128_t v0hf = wasm_i8x16_sub(v0h, qhh);

        // load y
        స్థిరరాశి v128_t v1l = wasm_v128_ఎక్కించు(y0->qs);
        స్థిరరాశి v128_t v1h = wasm_v128_ఎక్కించు(y0->qs + 16);

        // int8x16 -> int16x8
        స్థిరరాశి v128_t v0lfl = wasm_i16x8_extend_దిగువ_i8x16 (v0lf);
        స్థిరరాశి v128_t v0lfh = wasm_i16x8_extend_ఎగువ_i8x16(v0lf);
        స్థిరరాశి v128_t v0hfl = wasm_i16x8_extend_దిగువ_i8x16 (v0hf);
        స్థిరరాశి v128_t v0hfh = wasm_i16x8_extend_ఎగువ_i8x16(v0hf);

        స్థిరరాశి v128_t v1ll = wasm_i16x8_extend_దిగువ_i8x16 (v1l);
        స్థిరరాశి v128_t v1lh = wasm_i16x8_extend_ఎగువ_i8x16(v1l);
        స్థిరరాశి v128_t v1hl = wasm_i16x8_extend_దిగువ_i8x16 (v1h);
        స్థిరరాశి v128_t v1hh = wasm_i16x8_extend_ఎగువ_i8x16(v1h);

        // dot product
        sumv = wasm_f32x4_కూడు(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(
                        wasm_i32x4_కూడు(
                            wasm_i32x4_కూడు(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_కూడు(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))),
                    wasm_f32x4_splat(GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d))));
    }

    sumf = wasm_f32x4_వెలికితీయి_lane(sumv, 0) + wasm_f32x4_వెలికితీయి_lane(sumv, 1) +
           wasm_f32x4_వెలికితీయి_lane(sumv, 2) + wasm_f32x4_వెలికితీయి_lane(sumv, 3);

    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(ib);
    వాడని(sumf);
    వాడని(x);
    వాడని(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q5_1_q8_1(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_1;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(qk == QK5_1);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q5_1 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y = vy;

#ఒకవేళ defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);

    దశాంశం summs = 0.0f;

    uint32_t qh_;
    uint64_t tmp[4];

    // TODO: check if unrolling this is better
    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి ఖండం_q5_1 * GGML_RESTRICT x0 = &x[ib];
        స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y0 = &y[ib];

        summs += GGML_CPU_FP16_గమ్యం_FP32(x0->m) * GGML_CPU_FP16_గమ్యం_FP32(y0->s);

        స్థిరరాశి v128_t m4b = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        జ్ఞాపకం_నకలు(&qh_, x0->qh, పరిమాణం(qh_));

        tmp[0] = పట్టిక_b2b_0[(qh_ >>  0) & 0xFF];
        tmp[1] = పట్టిక_b2b_0[(qh_ >>  8) & 0xFF];
        tmp[2] = పట్టిక_b2b_0[(qh_ >> 16) & 0xFF];
        tmp[3] = పట్టిక_b2b_0[(qh_ >> 24)       ];

        స్థిరరాశి v128_t qhl = wasm_v128_ఎక్కించు(tmp + 0);
        స్థిరరాశి v128_t qhh = wasm_v128_ఎక్కించు(tmp + 2);

        స్థిరరాశి v128_t v0 = wasm_v128_ఎక్కించు(x0->qs);

        // 4-bit -> 8-bit
        స్థిరరాశి v128_t v0l = wasm_v128_and (v0, m4b);
        స్థిరరాశి v128_t v0h = wasm_u8x16_shr(v0, 4);

        // add high bit
        స్థిరరాశి v128_t v0lf = wasm_v128_or(v0l, qhl);
        స్థిరరాశి v128_t v0hf = wasm_v128_or(v0h, qhh);

        // load y
        స్థిరరాశి v128_t v1l = wasm_v128_ఎక్కించు(y0->qs);
        స్థిరరాశి v128_t v1h = wasm_v128_ఎక్కించు(y0->qs + 16);

        // int8x16 -> int16x8
        స్థిరరాశి v128_t v0lfl = wasm_i16x8_extend_దిగువ_i8x16 (v0lf);
        స్థిరరాశి v128_t v0lfh = wasm_i16x8_extend_ఎగువ_i8x16(v0lf);
        స్థిరరాశి v128_t v0hfl = wasm_i16x8_extend_దిగువ_i8x16 (v0hf);
        స్థిరరాశి v128_t v0hfh = wasm_i16x8_extend_ఎగువ_i8x16(v0hf);

        స్థిరరాశి v128_t v1ll = wasm_i16x8_extend_దిగువ_i8x16 (v1l);
        స్థిరరాశి v128_t v1lh = wasm_i16x8_extend_ఎగువ_i8x16(v1l);
        స్థిరరాశి v128_t v1hl = wasm_i16x8_extend_దిగువ_i8x16 (v1h);
        స్థిరరాశి v128_t v1hh = wasm_i16x8_extend_ఎగువ_i8x16(v1h);

        // dot product
        sumv = wasm_f32x4_కూడు(sumv,
                wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_కూడు(
                            wasm_i32x4_కూడు(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_కూడు(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))),
                    wasm_f32x4_splat(GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d))));
    }

    sumf = wasm_f32x4_వెలికితీయి_lane(sumv, 0) + wasm_f32x4_వెలికితీయి_lane(sumv, 1) +
           wasm_f32x4_వెలికితీయి_lane(sumv, 2) + wasm_f32x4_వెలికితీయి_lane(sumv, 3) + summs;

    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(ib);
    వాడని(sumf);
    వాడని(x);
    వాడని(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q8_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y = vy;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

#ఒకవేళ defined __wasm_simd128__
    v128_t sumv = wasm_f32x4_splat(0.0f);

    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT x0 = &x[ib];
        స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y0 = &y[ib];

        స్థిరరాశి v128_t x0_0 = wasm_v128_ఎక్కించు(x0->qs);
        స్థిరరాశి v128_t x0_1 = wasm_v128_ఎక్కించు(x0->qs + 16);
        స్థిరరాశి v128_t y0_0 = wasm_v128_ఎక్కించు(y0->qs);
        స్థిరరాశి v128_t y0_1 = wasm_v128_ఎక్కించు(y0->qs + 16);

        // Extend 8-bit to 16-bit
        స్థిరరాశి v128_t x0_0l = wasm_i16x8_extend_దిగువ_i8x16(x0_0);
        స్థిరరాశి v128_t x0_0h = wasm_i16x8_extend_ఎగువ_i8x16(x0_0);
        స్థిరరాశి v128_t x0_1l = wasm_i16x8_extend_దిగువ_i8x16(x0_1);
        స్థిరరాశి v128_t x0_1h = wasm_i16x8_extend_ఎగువ_i8x16(x0_1);

        స్థిరరాశి v128_t y0_0l = wasm_i16x8_extend_దిగువ_i8x16(y0_0);
        స్థిరరాశి v128_t y0_0h = wasm_i16x8_extend_ఎగువ_i8x16(y0_0);
        స్థిరరాశి v128_t y0_1l = wasm_i16x8_extend_దిగువ_i8x16(y0_1);
        స్థిరరాశి v128_t y0_1h = wasm_i16x8_extend_ఎగువ_i8x16(y0_1);

        // Compute dot products
        స్థిరరాశి v128_t dx0_0 = wasm_i32x4_dot_i16x8(x0_0l, y0_0l);
        స్థిరరాశి v128_t dx0_1 = wasm_i32x4_dot_i16x8(x0_0h, y0_0h);
        స్థిరరాశి v128_t dx1_0 = wasm_i32x4_dot_i16x8(x0_1l, y0_1l);
        స్థిరరాశి v128_t dx1_1 = wasm_i32x4_dot_i16x8(x0_1h, y0_1h);

        // Sum all dot products
        స్థిరరాశి v128_t కూడిక_dots = wasm_i32x4_కూడు(wasm_i32x4_కూడు(dx0_0, dx0_1), wasm_i32x4_కూడు(dx1_0, dx1_1));

        // Convert to float and accumulate
        స్థిరరాశి దశాంశం scale = GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d);
        sumv = wasm_f32x4_కూడు(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(కూడిక_dots), wasm_f32x4_splat(scale)));
    }

    sumf = wasm_f32x4_వెలికితీయి_lane(sumv, 0) + wasm_f32x4_వెలికితీయి_lane(sumv, 1) +
           wasm_f32x4_వెలికితీయి_lane(sumv, 2) + wasm_f32x4_వెలికితీయి_lane(sumv, 3);

    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(x);
    వాడని(y);
    వాడని(ib);
    వాడని(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q2_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q2_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined __wasm_simd128__
    దశాంశం sumf = 0;

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి uint8_t * q2 = x[i].qs;
        స్థిరరాశి int8_t * q8 = y[i].qs;
        స్థిరరాశి uint8_t * sc = x[i].scales;

        // Vectorized summs calculation
        v128_t summs_vec = wasm_i32x4_splat(0);
        {
            v128_t sc_vec = wasm_v128_ఎక్కించు(sc);
            v128_t sc_పై = wasm_u8x16_shr(sc_vec, 4);

            v128_t sc_దిగువ = wasm_u16x8_extend_దిగువ_u8x16(sc_పై);
            v128_t sc_ఎగువ = wasm_u16x8_extend_ఎగువ_u8x16(sc_పై);

            v128_t bsums1 = wasm_v128_ఎక్కించు(&y[i].bsums[0]);
            v128_t bsums2 = wasm_v128_ఎక్కించు(&y[i].bsums[8]);

            summs_vec = wasm_i32x4_కూడు(
                wasm_i32x4_కూడు(wasm_i32x4_dot_i16x8(sc_దిగువ, bsums1),
                               wasm_i32x4_dot_i16x8(sc_ఎగువ, bsums2)),
                summs_vec
            );

            summs_vec = wasm_i32x4_కూడు(summs_vec, wasm_i32x4_shuffle(summs_vec, summs_vec, 2, 3, 0, 1));
            summs_vec = wasm_i32x4_కూడు(summs_vec, wasm_i32x4_shuffle(summs_vec, summs_vec, 1, 0, 3, 2));
        }
        int32_t summs = wasm_i32x4_వెలికితీయి_lane(summs_vec, 0);

        // Vectorized isum calculation
        int32_t isum = 0;
        స్థిరరాశి uint8_t * sc_ptr = sc;
        స్థిరరాశి పూర్ణాంకం k_iters = QK_K/128;

        కొరకు (పూర్ణాంకం k = 0; k < k_iters; ++k) {
            v128_t isum_vec = wasm_i32x4_splat(0);
            పూర్ణాంకం జరుపు = 0;

            కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
                స్థిరరాశి పూర్ణాంకం d0 = (sc_ptr[0] & 0xF);
                స్థిరరాశి పూర్ణాంకం d1 = (sc_ptr[1] & 0xF);
                sc_ptr += 2;

                // Process first 16 elements
                v128_t q2_0 = wasm_v128_ఎక్కించు(q2);
                v128_t q8_0 = wasm_v128_ఎక్కించు(q8);
                v128_t q2_జరుపు_0 = wasm_u8x16_shr(q2_0, జరుపు);
                v128_t q2_బిట్లు_0 = wasm_v128_and(q2_జరుపు_0, wasm_i8x16_splat(0x03));

                // Process next 16 elements
                v128_t q2_1 = wasm_v128_ఎక్కించు(q2 + 16);
                v128_t q8_1 = wasm_v128_ఎక్కించు(q8 + 16);
                v128_t q2_జరుపు_1 = wasm_u8x16_shr(q2_1, జరుపు);
                v128_t q2_బిట్లు_1 = wasm_v128_and(q2_జరుపు_1, wasm_i8x16_splat(0x03));

                // Calculate dot products
                v128_t p0 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_దిగువ_i8x16(q8_0),
                    wasm_i16x8_extend_దిగువ_i8x16(q2_బిట్లు_0)
                );
                v128_t p1 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_ఎగువ_i8x16(q8_0),
                    wasm_i16x8_extend_ఎగువ_i8x16(q2_బిట్లు_0)
                );
                v128_t p2 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_దిగువ_i8x16(q8_1),
                    wasm_i16x8_extend_దిగువ_i8x16(q2_బిట్లు_1)
                );
                v128_t p3 = wasm_i32x4_dot_i16x8(
                    wasm_i16x8_extend_ఎగువ_i8x16(q8_1),
                    wasm_i16x8_extend_ఎగువ_i8x16(q2_బిట్లు_1)
                );

                // Accumulate scaled results
                v128_t scaled = wasm_i32x4_కూడు(
                    wasm_i32x4_mul(wasm_i32x4_కూడు(p0, p1), wasm_i32x4_splat(d0)),
                    wasm_i32x4_mul(wasm_i32x4_కూడు(p2, p3), wasm_i32x4_splat(d1))
                );

                isum_vec = wasm_i32x4_కూడు(isum_vec, scaled);
                q8 += 32;
                జరుపు += 2;
            }
            q2 += 32;

            // Horizontal sum of isum_vec
            isum_vec = wasm_i32x4_కూడు(isum_vec, wasm_i32x4_shuffle(isum_vec, isum_vec, 2, 3, 0, 1));
            isum_vec = wasm_i32x4_కూడు(isum_vec, wasm_i32x4_shuffle(isum_vec, isum_vec, 1, 0, 3, 2));
            isum += wasm_i32x4_వెలికితీయి_lane(isum_vec, 0);
        }

        స్థిరరాశి దశాంశం dall = GGML_CPU_FP16_గమ్యం_FP32(x[i].d) * y[i].d;
        స్థిరరాశి దశాంశం dmin = GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin) * y[i].d;
        sumf += dall * isum - dmin * summs;
    }

    *s = sumf;

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q3_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి uint32_t kmask1 = 0x03030303;
    స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;

    స్థిరరాశి ఖండం_q3_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined __wasm_simd128__
    int8_t  aux8[QK_K];
    దశాంశం   sums[8] = {0};
    uint32_t auxs[4];

    దశాంశం sumf = 0;
    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి uint8_t * GGML_RESTRICT q3 = x[i].qs;
        స్థిరరాశి uint8_t * GGML_RESTRICT hm = x[i].hmask;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Process blocks with SIMD
        int8_t * a = aux8;
        uint8_t m = 1;
        కొరకు (పూర్ణాంకం j = 0; j < QK_K; j += 128) {
            కొరకు (పూర్ణాంకం జరుపు = 0; జరుపు <= 6; జరుపు += 2) {
                v128_t v_m = wasm_i8x16_splat(m);
                కొరకు (పూర్ణాంకం l = 0; l < 32; l += 16) {
                    v128_t v_q3 = wasm_v128_ఎక్కించు(q3 + l);
                    v128_t v_జరుపు = wasm_i8x16_shr(v_q3, జరుపు);
                    v128_t v_low2 = wasm_v128_and(v_జరుపు, wasm_i8x16_splat(0x03));

                    v128_t v_hm = wasm_v128_ఎక్కించు(hm + l);
                    v128_t v_ముసుగు = wasm_v128_and(v_hm, v_m);
                    v_ముసుగు = wasm_i8x16_ne(v_ముసుగు, wasm_i8x16_splat(0));

                    v_low2 = wasm_i8x16_sub(v_low2, wasm_v128_and(wasm_i8x16_splat(4), wasm_v128_not(v_ముసుగు)));
                    wasm_v128_store(a + l, v_low2);
                }
                a += 32;
                m <<= 1;
            }
            q3 += 32;
        }

        // Extract scales
        జ్ఞాపకం_నకలు(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        స్థిరరాశి int8_t * scales = (స్థిరరాశి int8_t *)auxs;

        // SIMD dot product with register accumulators
        v128_t v_acc0 = wasm_i32x4_splat(0);
        v128_t v_acc1 = wasm_i32x4_splat(0);
        a = aux8;
        కొరకు (పూర్ణాంకం j = 0; j < QK_K/16; ++j) {
            స్థిరరాశి v128_t v_scale = wasm_i16x8_splat(scales[j] - 32);

            // Process 16 elements per iteration
            కొరకు (పూర్ణాంకం k = 0; k < 2; ++k) {
                స్థిరరాశి v128_t v_q8 = wasm_i16x8_load8x8(q8);
                స్థిరరాశి v128_t v_a = wasm_i16x8_load8x8(a);

                v128_t v_prod = wasm_i16x8_mul(v_q8, v_a);
                v_prod = wasm_i16x8_mul(v_prod, v_scale);

                v_acc0 = wasm_i32x4_కూడు(v_acc0, wasm_i32x4_extend_దిగువ_i16x8(v_prod));
                v_acc1 = wasm_i32x4_కూడు(v_acc1, wasm_i32x4_extend_ఎగువ_i16x8(v_prod));

                q8 += 8;
                a += 8;
            }
        }

        // Accumulate results
        స్థిరరాశి దశాంశం d = GGML_CPU_FP16_గమ్యం_FP32(x[i].d) * y[i].d;
        స్థిరరాశి v128_t v_d = wasm_f32x4_splat(d);
        v128_t v_కూడిక = wasm_f32x4_కూడు(
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(v_acc0), v_d),
            wasm_f32x4_mul(wasm_f32x4_convert_i32x4(v_acc1), v_d)
        );

        // Accumulate into sums vector
        wasm_v128_store(sums, wasm_f32x4_కూడు(wasm_v128_ఎక్కించు(sums), v_కూడిక));
    }

    // Horizontal sum
    v128_t v_కూడిక = wasm_f32x4_కూడు(wasm_v128_ఎక్కించు(sums), wasm_v128_ఎక్కించు(sums + 4));
    sumf = wasm_f32x4_వెలికితీయి_lane(v_కూడిక, 0) +
           wasm_f32x4_వెలికితీయి_lane(v_కూడిక, 1) +
           wasm_f32x4_వెలికితీయి_lane(v_కూడిక, 2) +
           wasm_f32x4_వెలికితీయి_lane(v_కూడిక, 3);

    *s = sumf;

#లేకపోతే
    వాడని(kmask1);
    వాడని(kmask2);
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు

}

శూన్య ggml_vec_dot_q4_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q4_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ఒకవేళ defined __wasm_simd128__
    స్థిరరాశి uint8_t * scales = (స్థిరరాశి uint8_t*)&utmp[0];
    దశాంశం sumf = 0;

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);
        స్థిరరాశి దశాంశం dmin = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin); // Corrected sign

        స్థిరరాశి uint8_t * GGML_RESTRICT q4 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Process scales and mins
        జ్ఞాపకం_నకలు(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        స్థిరరాశి uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // Sum mins * q8sums
        int32_t sumi = 0;
        స్థిరరాశి int16_t * GGML_RESTRICT q8sums = y[i].bsums;
        స్థిరరాశి uint8_t * m = (స్థిరరాశి uint8_t *)&utmp[2];
        కొరకు (పూర్ణాంకం j = 0; j < 16; j += 2) {
            sumi += (q8sums[j] + q8sums[j+1]) * m[j/2];
        }
        sumf -= dmin * sumi;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/64; ++j) {
            // Load 64 4-bit weights (32 bytes)
            స్థిరరాశి v128_t q4x0 = wasm_v128_ఎక్కించు(q4);
            స్థిరరాశి v128_t q4x1 = wasm_v128_ఎక్కించు(q4 + 16);
            q4 += 32;

            // Split into low/high nibbles
            స్థిరరాశి v128_t q4l0 = wasm_v128_and(q4x0, wasm_i8x16_splat(0x0F));
            స్థిరరాశి v128_t q4h0 = wasm_u8x16_shr(q4x0, 4);
            స్థిరరాశి v128_t q4l1 = wasm_v128_and(q4x1, wasm_i8x16_splat(0x0F));
            స్థిరరాశి v128_t q4h1 = wasm_u8x16_shr(q4x1, 4);

            // Load 64 8-bit values (64 bytes)
            స్థిరరాశి v128_t q8x0 = wasm_v128_ఎక్కించు(q8);
            స్థిరరాశి v128_t q8x1 = wasm_v128_ఎక్కించు(q8 + 16);
            స్థిరరాశి v128_t q8x2 = wasm_v128_ఎక్కించు(q8 + 32);
            స్థిరరాశి v128_t q8x3 = wasm_v128_ఎక్కించు(q8 + 48);
            q8 += 64;

            // Low nibble products
            v128_t vacc1 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_దిగువ_i8x16(q4l0),
                wasm_i16x8_extend_దిగువ_i8x16(q8x0)
            );
            vacc1 = wasm_i32x4_కూడు(vacc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_ఎగువ_i8x16(q4l0),
                wasm_i16x8_extend_ఎగువ_i8x16(q8x0)
            ));
            vacc1 = wasm_i32x4_కూడు(vacc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_దిగువ_i8x16(q4l1),
                wasm_i16x8_extend_దిగువ_i8x16(q8x1)
            ));
            vacc1 = wasm_i32x4_కూడు(vacc1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_ఎగువ_i8x16(q4l1),
                wasm_i16x8_extend_ఎగువ_i8x16(q8x1)
            ));

            // High nibble products
            v128_t vacc2 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_దిగువ_i8x16(q4h0),
                wasm_i16x8_extend_దిగువ_i8x16(q8x2)
            );
            vacc2 = wasm_i32x4_కూడు(vacc2, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_ఎగువ_i8x16(q4h0),
                wasm_i16x8_extend_ఎగువ_i8x16(q8x2)
            ));
            vacc2 = wasm_i32x4_కూడు(vacc2, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_దిగువ_i8x16(q4h1),
                wasm_i16x8_extend_దిగువ_i8x16(q8x3)
            ));
            vacc2 = wasm_i32x4_కూడు(vacc2, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_ఎగువ_i8x16(q4h1),
                wasm_i16x8_extend_ఎగువ_i8x16(q8x3)
            ));

            // Accumulate scaled results
            int32_t vacc1_కూడిక = wasm_i32x4_వెలికితీయి_lane(vacc1, 0) + wasm_i32x4_వెలికితీయి_lane(vacc1, 1) +
                                wasm_i32x4_వెలికితీయి_lane(vacc1, 2) + wasm_i32x4_వెలికితీయి_lane(vacc1, 3);
            sumi1 += vacc1_కూడిక * scales[2*j];

            int32_t vacc2_కూడిక = wasm_i32x4_వెలికితీయి_lane(vacc2, 0) + wasm_i32x4_వెలికితీయి_lane(vacc2, 1) +
                                wasm_i32x4_వెలికితీయి_lane(vacc2, 2) + wasm_i32x4_వెలికితీయి_lane(vacc2, 3);
            sumi2 += vacc2_కూడిక * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf;

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    వాడని(kmask1);
    వాడని(kmask2);
    వాడని(kmask3);
    వాడని(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q5_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy,  size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q5_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ఒకవేళ defined __wasm_simd128__
    //const uint8_t * scales = (const uint8_t*)&utmp[0];
    దశాంశం sumf = 0;

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);
        స్థిరరాశి దశాంశం dmin = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin); // Fixed sign

        స్థిరరాశి uint8_t * GGML_RESTRICT q5 = x[i].qs;
        స్థిరరాశి uint8_t * GGML_RESTRICT qh = x[i].qh;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Process scales and mins
        జ్ఞాపకం_నకలు(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        స్థిరరాశి uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // Sum mins * q8sums
        int32_t sumi_mins = 0;
        స్థిరరాశి int16_t * GGML_RESTRICT q8sums = y[i].bsums;
        స్థిరరాశి uint8_t * m = (స్థిరరాశి uint8_t *)&utmp[2];
        కొరకు (పూర్ణాంకం j = 0; j < 16; j += 2) {
            sumi_mins += (q8sums[j] + q8sums[j+1]) * m[j/2];
        }
        sumf -= dmin * sumi_mins; // Correct subtraction

        v128_t qh0 = wasm_v128_ఎక్కించు(qh);
        v128_t qh1 = wasm_v128_ఎక్కించు(qh + 16);
        స్థిరరాశి uint8_t * sc = (స్థిరరాశి uint8_t *)utmp;

        int32_t sumi = 0;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/64; ++j) {
            స్థిరరాశి పూర్ణాంకం జరుపు = j * 2;
            v128_t qh_shift0 = wasm_u8x16_shr(qh0, జరుపు);
            v128_t qh_shift1 = wasm_u8x16_shr(qh1, జరుపు);

            v128_t qh_low0 = wasm_i8x16_shl(wasm_v128_and(qh_shift0, wasm_i8x16_splat(0x01)), 4);
            v128_t qh_high0 = wasm_i8x16_shl(wasm_v128_and(qh_shift0, wasm_i8x16_splat(0x02)), 3);
            v128_t qh_low1 = wasm_i8x16_shl(wasm_v128_and(qh_shift1, wasm_i8x16_splat(0x01)), 4);
            v128_t qh_high1 = wasm_i8x16_shl(wasm_v128_and(qh_shift1, wasm_i8x16_splat(0x02)), 3);

            v128_t q5_0 = wasm_v128_ఎక్కించు(q5);
            v128_t q5_1 = wasm_v128_ఎక్కించు(q5 + 16);
            q5 += 32;

            v128_t q5l_0 = wasm_v128_or(wasm_v128_and(q5_0, wasm_i8x16_splat(0x0F)), qh_low0);
            v128_t q5h_0 = wasm_v128_or(wasm_u8x16_shr(q5_0, 4), qh_high0);
            v128_t q5l_1 = wasm_v128_or(wasm_v128_and(q5_1, wasm_i8x16_splat(0x0F)), qh_low1);
            v128_t q5h_1 = wasm_v128_or(wasm_u8x16_shr(q5_1, 4), qh_high1);

            v128_t q8_0 = wasm_v128_ఎక్కించు(q8);
            v128_t q8_1 = wasm_v128_ఎక్కించు(q8 + 16);
            v128_t q8_2 = wasm_v128_ఎక్కించు(q8 + 32);
            v128_t q8_3 = wasm_v128_ఎక్కించు(q8 + 48);
            q8 += 64;

            // Process low quants
            v128_t pl0 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_దిగువ_i8x16(q5l_0),
                wasm_i16x8_extend_దిగువ_i8x16(q8_0)
            );
            pl0 = wasm_i32x4_కూడు(pl0, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_ఎగువ_i8x16(q5l_0),
                wasm_i16x8_extend_ఎగువ_i8x16(q8_0)
            ));
            v128_t pl1 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_దిగువ_i8x16(q5l_1),
                wasm_i16x8_extend_దిగువ_i8x16(q8_1)
            );
            pl1 = wasm_i32x4_కూడు(pl1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_ఎగువ_i8x16(q5l_1),
                wasm_i16x8_extend_ఎగువ_i8x16(q8_1)
            ));
            v128_t కూడిక_దిగువ = wasm_i32x4_కూడు(pl0, pl1);

            // Process high quants
            v128_t ph0 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_దిగువ_i8x16(q5h_0),
                wasm_i16x8_extend_దిగువ_i8x16(q8_2)
            );
            ph0 = wasm_i32x4_కూడు(ph0, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_ఎగువ_i8x16(q5h_0),
                wasm_i16x8_extend_ఎగువ_i8x16(q8_2)
            ));
            v128_t ph1 = wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_దిగువ_i8x16(q5h_1),
                wasm_i16x8_extend_దిగువ_i8x16(q8_3)
            );
            ph1 = wasm_i32x4_కూడు(ph1, wasm_i32x4_dot_i16x8(
                wasm_i16x8_extend_ఎగువ_i8x16(q5h_1),
                wasm_i16x8_extend_ఎగువ_i8x16(q8_3)
            ));
            v128_t కూడిక_ఎగువ = wasm_i32x4_కూడు(ph0, ph1);

            // Accumulate with scale factors
            int32_t sl = wasm_i32x4_వెలికితీయి_lane(కూడిక_దిగువ, 0) + wasm_i32x4_వెలికితీయి_lane(కూడిక_దిగువ, 1) +
                        wasm_i32x4_వెలికితీయి_lane(కూడిక_దిగువ, 2) + wasm_i32x4_వెలికితీయి_lane(కూడిక_దిగువ, 3);
            int32_t sh = wasm_i32x4_వెలికితీయి_lane(కూడిక_ఎగువ, 0) + wasm_i32x4_వెలికితీయి_lane(కూడిక_ఎగువ, 1) +
                        wasm_i32x4_వెలికితీయి_lane(కూడిక_ఎగువ, 2) + wasm_i32x4_వెలికితీయి_lane(కూడిక_ఎగువ, 3);

            sumi += sl * sc[2*j] + sh * sc[2*j+1];
        }

        sumf += d * sumi;
    }

    *s = sumf;

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    వాడని(kmask1);
    వాడని(kmask2);
    వాడని(kmask3);
    వాడని(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q6_K_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q6_K * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined __wasm_simd128__
    int8_t aux8[QK_K] __attribute__((aligned(16)));
    int32_t aux32[8] __attribute__((aligned(16))) = {0};
    దశాంశం sums[8] __attribute__((aligned(16))) = {0};

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        // Unpack 6-bit quantized data into aux8 (unchanged)
        స్థిరరాశి uint8_t * GGML_RESTRICT q4 = x[i].ql;
        స్థిరరాశి uint8_t * GGML_RESTRICT qh = x[i].qh;
        int8_t * a = aux8;
        కొరకు (పూర్ణాంకం j = 0; j < QK_K; j += 128) {
            కొరకు (పూర్ణాంకం l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }

        స్థిరరాశి int8_t * GGML_RESTRICT a_ptr = aux8;
        స్థిరరాశి int8_t * GGML_RESTRICT q8 = y[i].qs;
        v128_t acc0 = wasm_i32x4_splat(0);
        v128_t acc1 = wasm_i32x4_splat(0);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/16; ++j) {
            స్థిరరాశి పూర్ణాంకం scale = x[i].scales[j];
            స్థిరరాశి v128_t vscale = wasm_i32x4_splat(scale);

            // Load 16 elements from a and q8
            స్థిరరాశి v128_t a_vec = wasm_v128_ఎక్కించు(a_ptr);
            స్థిరరాశి v128_t q8_vec = wasm_v128_ఎక్కించు(q8);

            // Process low 8 elements
            v128_t a_దిగువ = wasm_i16x8_extend_దిగువ_i8x16(a_vec);
            v128_t q8_దిగువ = wasm_i16x8_extend_దిగువ_i8x16(q8_vec);
            v128_t prod_దిగువ = wasm_i16x8_mul(a_దిగువ, q8_దిగువ);
            v128_t prod_lo_lo = wasm_i32x4_extend_దిగువ_i16x8(prod_దిగువ);
            v128_t prod_lo_hi = wasm_i32x4_extend_ఎగువ_i16x8(prod_దిగువ);

            // Process high 8 elements
            v128_t a_ఎగువ = wasm_i16x8_extend_ఎగువ_i8x16(a_vec);
            v128_t q8_ఎగువ = wasm_i16x8_extend_ఎగువ_i8x16(q8_vec);
            v128_t prod_ఎగువ = wasm_i16x8_mul(a_ఎగువ, q8_ఎగువ);
            v128_t prod_hi_lo = wasm_i32x4_extend_దిగువ_i16x8(prod_ఎగువ);
            v128_t prod_hi_hi = wasm_i32x4_extend_ఎగువ_i16x8(prod_ఎగువ);

            // Scale and accumulate
            prod_lo_lo = wasm_i32x4_mul(prod_lo_lo, vscale);
            prod_lo_hi = wasm_i32x4_mul(prod_lo_hi, vscale);
            prod_hi_lo = wasm_i32x4_mul(prod_hi_lo, vscale);
            prod_hi_hi = wasm_i32x4_mul(prod_hi_hi, vscale);

            acc0 = wasm_i32x4_కూడు(acc0, wasm_i32x4_కూడు(prod_lo_lo, prod_hi_lo));
            acc1 = wasm_i32x4_కూడు(acc1, wasm_i32x4_కూడు(prod_lo_hi, prod_hi_hi));

            a_ptr += 16;
            q8 += 16;
        }

        // Store accumulated results
        wasm_v128_store(&aux32[0], acc0);
        wasm_v128_store(&aux32[4], acc1);

        స్థిరరాశి దశాంశం d = GGML_CPU_FP16_గమ్యం_FP32(x[i].d) * y[i].d;
        కొరకు (పూర్ణాంకం l = 0; l < 8; ++l) {
            sums[l] += d * aux32[l];
        }
    }

    // Sum final results
    దశాంశం sumf = 0;
    కొరకు (పూర్ణాంకం l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    *s = sumf;

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}
