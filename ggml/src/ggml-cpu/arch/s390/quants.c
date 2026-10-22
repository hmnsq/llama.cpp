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

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
#నిర్వచించు B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#నిర్వచించు B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#నిర్వచించు B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#నిర్వచించు B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#నిర్వచించు B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#నిర్వచించు B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#నిర్వచించు B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#నిర్వచించు B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
స్థిర స్థిరరాశి __attribute__((aligned(16))) uint64_t పట్టిక_b2b_0[1 << 8] = { B8(00, 10) }; // ( b ) << 4
స్థిర స్థిరరాశి __attribute__((aligned(16))) uint64_t పట్టిక_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4

// permute mask for byteswapping
స్థిర స్థిరరాశి uint8x16_t v_kperm = (స్థిరరాశి uint8x16_t){
     7,  6,  5,  4,  3,  2, 1, 0,
    15, 14, 13, 12, 11, 10, 9, 8
};
#అయితే_ముగింపు

శూన్య quantize_అడ్డువరుస_q8_0(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK8_0 == 32);
    ప్రతిజ్ఞా(k % QK8_0 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK8_0;

    ఖండం_q8_0 * GGML_RESTRICT y = vy;

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) srcv[j] = vec_xl(0, x + i*32 + 4*j);
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);
        కొరకు (పూర్ణాంకం j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        కొరకు (పూర్ణాంకం j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        కొరకు (పూర్ణాంకం j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        స్థిరరాశి దశాంశం amax = MAX(MAX(vec_వెలికితీయి(amaxv[0], 0),
                                   vec_వెలికితీయి(amaxv[0], 1)),
                               MAX(vec_వెలికితీయి(amaxv[0], 2),
                                   vec_వెలికితీయి(amaxv[0], 3)));

        స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
        స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_CPU_FP32_గమ్యం_FP16(d);

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            స్థిరరాశి float32x4_t v = vec_mul(srcv[j], vec_splats(id));
            /* Uses non-default rounding for vec_signed or vec_round */
            స్థిరరాశి int32x4_t vi = vec_signed(__builtin_s390_vfisb(v, 4, 1));

            y[i].qs[4*j + 0] = vec_వెలికితీయి(vi, 0);
            y[i].qs[4*j + 1] = vec_వెలికితీయి(vi, 1);
            y[i].qs[4*j + 2] = vec_వెలికితీయి(vi, 2);
            y[i].qs[4*j + 3] = vec_వెలికితీయి(vi, 3);
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

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) srcv[j] = vec_xl(0, x + i*32 + 4*j);
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);
        కొరకు (పూర్ణాంకం j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        కొరకు (పూర్ణాంకం j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        కొరకు (పూర్ణాంకం j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        స్థిరరాశి దశాంశం amax = MAX(MAX(vec_వెలికితీయి(amaxv[0], 0),
                                   vec_వెలికితీయి(amaxv[0], 1)),
                               MAX(vec_వెలికితీయి(amaxv[0], 2),
                                   vec_వెలికితీయి(amaxv[0], 3)));

        స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
        స్థిరరాశి దశాంశం id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_CPU_FP32_గమ్యం_FP16(d);

        int32x4_t acc = vec_splats(0);

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            స్థిరరాశి float32x4_t v = vec_mul(srcv[j], vec_splats(id));
            /* Uses non-default rounding for vec_signed or vec_round */
            స్థిరరాశి int32x4_t vi = vec_signed(__builtin_s390_vfisb(v, 4, 1));

            y[i].qs[4*j + 0] = vec_వెలికితీయి(vi, 0);
            y[i].qs[4*j + 1] = vec_వెలికితీయి(vi, 1);
            y[i].qs[4*j + 2] = vec_వెలికితీయి(vi, 2);
            y[i].qs[4*j + 3] = vec_వెలికితీయి(vi, 3);

            acc = vec_కూడు(acc, vi);
        }

        y[i].s = GGML_CPU_FP32_గమ్యం_FP16(d * (acc[0] + acc[1] + acc[2] + acc[3]));
    }
#లేకపోతే
    GGML_వాడని(nb);
    // scalar
    quantize_అడ్డువరుస_q8_1_ref(x, y, k);
#అయితే_ముగింపు
}


//===================================== Dot products =================================

శూన్య ggml_vec_dot_q1_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK1_0;  // 128
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q1_0 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y = vy;

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    float32x4_t v_sumf = vec_splats(0.0f);

    స్థిరరాశి uint8x16_t v_zero = vec_splats((uint8_t)0x00);  // zero
    స్థిరరాశి uint8x16_t v_bias = vec_splats((uint8_t)0x80);  // bias from signed to unsigned
                                                          // v ^ 0x80 == v + 128

    స్థిరరాశి uint8x16_t v_idx = (స్థిరరాశి uint8x16_t){ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
    స్థిరరాశి uint8x16_t v_బిట్టు = (స్థిరరాశి uint8x16_t){ 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి uint8x16_t  v_x  = vec_xl(0, (స్థిరరాశి uint8_t *)x[i].qs);
        స్థిరరాశి float32x4_t v_xd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));

        కొరకు (పూర్ణాంకం k = 0; k < 4; ++k) {
            // sub-block k holds elements 32k .. 32k+31
            స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT yb = &y[i*4 + k];
            స్థిరరాశి float32x4_t v_yd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(yb->d));

            స్థిరరాశి uint8x16_t v_xrl = vec_perm(v_x, v_x, vec_కూడు(v_idx, vec_splats((uint8_t)(k*4 + 0))));
            స్థిరరాశి uint8x16_t v_xrh = vec_perm(v_x, v_x, vec_కూడు(v_idx, vec_splats((uint8_t)(k*4 + 2))));

            // isolate each lane's bit, then set all ones where that bit is clear, the -d case
            స్థిరరాశి int8x16_t v_ml = (int8x16_t)vec_cmpeq(vec_and(v_xrl, v_బిట్టు), v_zero);
            స్థిరరాశి int8x16_t v_mh = (int8x16_t)vec_cmpeq(vec_and(v_xrh, v_బిట్టు), v_zero);

            స్థిరరాశి int8x16_t v_yl = vec_xl(0,       (స్థిరరాశి int8_t *)yb->qs);
            స్థిరరాశి int8x16_t v_yh = vec_xl(QK8_0/2, (స్థిరరాశి int8_t *)yb->qs);

            // weights are only +1 or -1, so negate y
            స్థిరరాశి int8x16_t v_ysl = vec_sub(vec_xor(v_yl, v_ml), v_ml);
            స్థిరరాశి int8x16_t v_ysh = vec_sub(vec_xor(v_yh, v_mh), v_mh);

            // bias to unsigned, then vec_sum4 adds each group of 4 bytes into one word
            స్థిరరాశి uint32x4_t v_p = vec_కూడు(vec_sum4(vec_xor((uint8x16_t)v_ysl, v_bias), v_zero),
                                           vec_sum4(vec_xor((uint8x16_t)v_ysh, v_bias), v_zero));

            // each word summed 8 biased bytes, so take back 8 * 128
            స్థిరరాశి int32x4_t v_xy = vec_sub((int32x4_t)v_p, vec_splats((int32_t)1024));

            // apply both block scales and add into the running total
            v_sumf = vec_madd(vec_float(v_xy), vec_mul(v_xd, v_yd), v_sumf);
        }
    }

    *s = vec_hsum_f32x4(v_sumf);
#లేకపోతే
    వాడని(nb);
    వాడని(x);
    వాడని(y);
    ggml_vec_dot_q1_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

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

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    float32x4_t acc = vec_splats(0.0f);

    స్థిరరాశి uint8x16_t v_m = vec_splats((స్థిరరాశి uint8_t)0x0F);
    స్థిరరాశి int8x16_t  v_s = vec_splats( (స్థిరరాశి int8_t)0x08);

    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి uint8x16_t v_x = vec_xl(0, x[ib].qs);
        స్థిరరాశి int8x16_t v_xl = (స్థిరరాశి int8x16_t)(v_x & v_m);
        స్థిరరాశి int8x16_t v_xh = (స్థిరరాశి int8x16_t)(v_x >> 4);

        స్థిరరాశి int8x16_t v_xls = vec_sub(v_xl, v_s);
        స్థిరరాశి int8x16_t v_xhs = vec_sub(v_xh, v_s);

        స్థిరరాశి int8x16_t v_yl = vec_xl(0      , y[ib].qs);
        స్థిరరాశి int8x16_t v_yh = vec_xl(QK8_0/2, y[ib].qs);

        స్థిరరాశి int16x8_t v_xylso = vec_mulo(v_xls, v_yl);
        స్థిరరాశి int16x8_t v_xyl = vec_meadd(v_xls, v_yl, v_xylso);
        స్థిరరాశి int16x8_t v_xyhso = vec_mulo(v_xhs, v_yh);
        స్థిరరాశి int16x8_t v_xyh = vec_meadd(v_xhs, v_yh, v_xyhso);

        int16x8_t v_xy_ = v_xyl + v_xyh; v_xy_ += vec_reve(v_xy_);

        స్థిరరాశి float32x4_t v_xy = vec_float(vec_unpackh(v_xy_));
        స్థిరరాశి float32x4_t v_d = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));

        acc = vec_madd(v_xy, v_d, acc);
    }

    sumf = vec_hsum_f32x4(acc);
    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(x);
    వాడని(y);
    వాడని(ib);
    వాడని(sumf);
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
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

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    దశాంశం summs = 0;
    float32x4_t acc = vec_splats(0.0f);

    స్థిరరాశి uint8x16_t v_m = vec_splat_u8(0x0F);

#సూచన GCC unroll 4
    కొరకు (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        summs += GGML_CPU_FP16_గమ్యం_FP32(x[ib].m) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].s);

        స్థిరరాశి uint8x16_t v_x = vec_xl(0, x[ib].qs);
        స్థిరరాశి int8x16_t v_xl = (స్థిరరాశి int8x16_t)(v_x & v_m);
        స్థిరరాశి int8x16_t v_xh = (స్థిరరాశి int8x16_t)(v_x >> 4);

        స్థిరరాశి int8x16_t v_yl = vec_xl(0      , y[ib].qs);
        స్థిరరాశి int8x16_t v_yh = vec_xl(QK8_1/2, y[ib].qs);

        స్థిరరాశి int32x4_t v_xy_ = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_xl, v_yl), v_xh, v_yh);
        స్థిరరాశి float32x4_t v_xy = vec_float(v_xy_);

        స్థిరరాశి float32x4_t v_d = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));

        acc = vec_madd(v_xy, v_d, acc);
    }

    sumf = vec_hsum_f32x4(acc) + summs;
    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(x);
    వాడని(y);
    వాడని(ib);
    వాడని(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_mxfp4_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);
    ప్రతిజ్ఞా(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    స్థిరరాశి పూర్ణాంకం qk = QK_MXFP4;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    స్థిరరాశి ఖండం_mxfp4 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0  * GGML_RESTRICT y = vy;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0.0f;

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    స్థిరరాశి int8x16_t  v_k = vec_xl(0, kvalues_mxfp4);
    స్థిరరాశి uint8x16_t v_m = vec_splats((స్థిరరాశి uint8_t)0x0F);

    float32x4_t v_acc = vec_splats(0.0f);

    #సూచన GCC unroll 8
    కొరకు (; ib + 1 < nb; ib += 2) {
        స్థిరరాశి ఖండం_mxfp4 * GGML_RESTRICT x0 = &x[ib + 0];
        స్థిరరాశి ఖండం_mxfp4 * GGML_RESTRICT x1 = &x[ib + 1];
        స్థిరరాశి ఖండం_q8_0  * GGML_RESTRICT y0 = &y[ib + 0];
        స్థిరరాశి ఖండం_q8_0  * GGML_RESTRICT y1 = &y[ib + 1];

        స్థిరరాశి uint8x16_t v_x0 = vec_xl(0, x0->qs);
        స్థిరరాశి uint8x16_t v_x1 = vec_xl(0, x1->qs);

        int8x16_t v_x0l = (int8x16_t)vec_and(v_x0, v_m);
        int8x16_t v_x0h = (int8x16_t)vec_sr(v_x0, 4);
        int8x16_t v_x1l = (int8x16_t)vec_and(v_x1, v_m);
        int8x16_t v_x1h = (int8x16_t)vec_sr(v_x1, 4);

        v_x0l = vec_perm(v_k, v_k, (uchar8x16_t)v_x0l);
        v_x0h = vec_perm(v_k, v_k, (uchar8x16_t)v_x0h);
        v_x1l = vec_perm(v_k, v_k, (uchar8x16_t)v_x1l);
        v_x1h = vec_perm(v_k, v_k, (uchar8x16_t)v_x1h);

        స్థిరరాశి int8x16_t v_y0l = vec_xl(0,       y0->qs);
        స్థిరరాశి int8x16_t v_y0h = vec_xl(QK8_0/2, y0->qs);
        స్థిరరాశి int8x16_t v_y1l = vec_xl(0,       y1->qs);
        స్థిరరాశి int8x16_t v_y1h = vec_xl(QK8_0/2, y1->qs);

        స్థిరరాశి int32x4_t v_xy0 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_x0l, v_y0l), v_x0h, v_y0h);
        స్థిరరాశి int32x4_t v_xy1 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_x1l, v_y1l), v_x1h, v_y1h);

        స్థిరరాశి float32x4_t v_xy0f = vec_float(v_xy0);
        స్థిరరాశి float32x4_t v_xy1f = vec_float(v_xy1);

        స్థిరరాశి float32x4_t v_d0 = vec_splats(GGML_E8_M0_గమ్యం_FP32_HALF(x0->e) * GGML_CPU_FP16_గమ్యం_FP32(y0->d));
        స్థిరరాశి float32x4_t v_d1 = vec_splats(GGML_E8_M0_గమ్యం_FP32_HALF(x1->e) * GGML_CPU_FP16_గమ్యం_FP32(y1->d));

        v_acc = vec_madd(v_xy0f, v_d0, v_acc);
        v_acc = vec_madd(v_xy1f, v_d1, v_acc);
    }

    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి ఖండం_mxfp4 * GGML_RESTRICT x0 = &x[ib + 0];
        స్థిరరాశి ఖండం_q8_0  * GGML_RESTRICT y0 = &y[ib + 0];

        స్థిరరాశి uint8x16_t v_x = vec_xl(0, x0->qs);

        int8x16_t v_xl = (int8x16_t)vec_and(v_x, v_m);
        int8x16_t v_xh = (int8x16_t)vec_sr(v_x, 4);

        v_xl = vec_perm(v_k, v_k, (uchar8x16_t)v_xl);
        v_xh = vec_perm(v_k, v_k, (uchar8x16_t)v_xh);

        స్థిరరాశి int8x16_t v_yl = vec_xl(0,       y0->qs);
        స్థిరరాశి int8x16_t v_yh = vec_xl(QK8_0/2, y0->qs);

        స్థిరరాశి int32x4_t v_xy = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_xl, v_yl), v_xh, v_yh);
        స్థిరరాశి float32x4_t v_xyf = vec_float(v_xy);

        స్థిరరాశి float32x4_t v_d = vec_splats(GGML_E8_M0_గమ్యం_FP32_HALF(x0->e) * GGML_CPU_FP16_గమ్యం_FP32(y0->d));
        v_acc = vec_madd(v_xyf, v_d, v_acc);
    }

    sumf = vec_hsum_f32x4(v_acc);
    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(x);
    వాడని(y);
    వాడని(ib);
    వాడని(sumf);
    ggml_vec_dot_mxfp4_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q5_0_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(qk == QK5_0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q5_0 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y = vy;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0.0f;

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    float32x4_t v_sum0 = vec_splats(0.0f);
    float32x4_t v_sum1 = vec_splats(0.0f);

    uint32_t qh0, qh1;
    uint64_t tmp0[4], tmp1[4];

    స్థిరరాశి uint8x16_t v_m = vec_splats((uint8_t)0x0F);

    #సూచన GCC unroll 4
    కొరకు (; ib + 1 < nb; ib += 2) {
        స్థిరరాశి ఖండం_q5_0 * GGML_RESTRICT x0 = &x[ib + 0];
        స్థిరరాశి ఖండం_q5_0 * GGML_RESTRICT x1 = &x[ib + 1];
        స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
        స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

        జ్ఞాపకం_నకలు(&qh0, x0->qh, పరిమాణం(qh0));
        జ్ఞాపకం_నకలు(&qh1, x1->qh, పరిమాణం(qh1));

        tmp0[0] = పట్టిక_b2b_1[(qh0 >>  0) & 0xFF];
        tmp0[1] = పట్టిక_b2b_1[(qh0 >>  8) & 0xFF];
        tmp0[2] = పట్టిక_b2b_1[(qh0 >> 16) & 0xFF];
        tmp0[3] = పట్టిక_b2b_1[(qh0 >> 24)       ];

        tmp1[0] = పట్టిక_b2b_1[(qh1 >>  0) & 0xFF];
        tmp1[1] = పట్టిక_b2b_1[(qh1 >>  8) & 0xFF];
        tmp1[2] = పట్టిక_b2b_1[(qh1 >> 16) & 0xFF];
        tmp1[3] = పట్టిక_b2b_1[(qh1 >> 24)       ];

        int8x16_t v_qh0l = vec_xl(0, (స్థిరరాశి int8_t *)(tmp0 + 0));
        int8x16_t v_qh0h = vec_xl(0, (స్థిరరాశి int8_t *)(tmp0 + 2));
        int8x16_t v_qh1l = vec_xl(0, (స్థిరరాశి int8_t *)(tmp1 + 0));
        int8x16_t v_qh1h = vec_xl(0, (స్థిరరాశి int8_t *)(tmp1 + 2));

        // required for fixing the byteorder
        v_qh0l = vec_perm(v_qh0l, v_qh0l, v_kperm);
        v_qh0h = vec_perm(v_qh0h, v_qh0h, v_kperm);
        v_qh1l = vec_perm(v_qh1l, v_qh1l, v_kperm);
        v_qh1h = vec_perm(v_qh1h, v_qh1h, v_kperm);

        స్థిరరాశి uint8x16_t v_x0 = vec_xl(0, (స్థిరరాశి uint8_t *)x0->qs);
        స్థిరరాశి uint8x16_t v_x1 = vec_xl(0, (స్థిరరాశి uint8_t *)x1->qs);

        int8x16_t v_x0l = (int8x16_t)vec_and(v_x0, v_m);
        int8x16_t v_x0h = (int8x16_t)vec_sr(v_x0, 4);
        int8x16_t v_x1l = (int8x16_t)vec_and(v_x1, v_m);
        int8x16_t v_x1h = (int8x16_t)vec_sr(v_x1, 4);

        స్థిరరాశి int8x16_t v_x0lf = vec_sub(v_x0l, v_qh0l);
        స్థిరరాశి int8x16_t v_x0hf = vec_sub(v_x0h, v_qh0h);
        స్థిరరాశి int8x16_t v_x1lf = vec_sub(v_x1l, v_qh1l);
        స్థిరరాశి int8x16_t v_x1hf = vec_sub(v_x1h, v_qh1h);

        స్థిరరాశి int8x16_t v_y0l = vec_xl(0,       (స్థిరరాశి int8_t *)y0->qs);
        స్థిరరాశి int8x16_t v_y0h = vec_xl(QK8_0/2, (స్థిరరాశి int8_t *)y0->qs);
        స్థిరరాశి int8x16_t v_y1l = vec_xl(0,       (స్థిరరాశి int8_t *)y1->qs);
        స్థిరరాశి int8x16_t v_y1h = vec_xl(QK8_0/2, (స్థిరరాశి int8_t *)y1->qs);

        స్థిరరాశి int32x4_t v_xy0 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_x0lf, v_y0l), v_x0hf, v_y0h);
        స్థిరరాశి int32x4_t v_xy1 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_x1lf, v_y1l), v_x1hf, v_y1h);

        స్థిరరాశి float32x4_t v_xy0f = vec_float(v_xy0);
        స్థిరరాశి float32x4_t v_xy1f = vec_float(v_xy1);

        స్థిరరాశి float32x4_t v_d0 = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d));
        స్థిరరాశి float32x4_t v_d1 = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x1->d) * GGML_CPU_FP16_గమ్యం_FP32(y1->d));

        v_sum0 = vec_madd(v_xy0f, v_d0, v_sum0);
        v_sum1 = vec_madd(v_xy1f, v_d1, v_sum1);
    }

    sumf += vec_hsum_f32x4(v_sum0) + vec_hsum_f32x4(v_sum1);

    #సూచన GCC unroll 4
    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి ఖండం_q5_0 * GGML_RESTRICT x0 = &x[ib];
        స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y0 = &y[ib];

        uint32_t qh;
        జ్ఞాపకం_నకలు(&qh, x0->qh, పరిమాణం(qh));

        uint64_t tmp[4];
        tmp[0] = పట్టిక_b2b_1[(qh >>  0) & 0xFF];
        tmp[1] = పట్టిక_b2b_1[(qh >>  8) & 0xFF];
        tmp[2] = పట్టిక_b2b_1[(qh >> 16) & 0xFF];
        tmp[3] = పట్టిక_b2b_1[(qh >> 24)       ];

        int8x16_t v_qhl = vec_xl(0, (స్థిరరాశి int8_t *)(tmp + 0));
        int8x16_t v_qhh = vec_xl(0, (స్థిరరాశి int8_t *)(tmp + 2));

        // required for fixing the byteorder
        v_qhl = vec_perm(v_qhl, v_qhl, v_kperm);
        v_qhh = vec_perm(v_qhh, v_qhh, v_kperm);

        స్థిరరాశి uint8x16_t v_x = vec_xl(0, (స్థిరరాశి uint8_t *)x0->qs);
        int8x16_t v_xl = (int8x16_t)vec_and(v_x, v_m);
        int8x16_t v_xh = (int8x16_t)vec_sr(v_x, 4);

        స్థిరరాశి int8x16_t v_xlf = vec_sub(v_xl, v_qhl);
        స్థిరరాశి int8x16_t v_xhf = vec_sub(v_xh, v_qhh);

        స్థిరరాశి int8x16_t v_yl = vec_xl(0,       (స్థిరరాశి int8_t *)y0->qs);
        స్థిరరాశి int8x16_t v_yh = vec_xl(QK8_0/2, (స్థిరరాశి int8_t *)y0->qs);

        స్థిరరాశి int32x4_t v_xy = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_xlf, v_yl), v_xhf, v_yh);
        స్థిరరాశి float32x4_t v_xyf = vec_float(v_xy);

        స్థిరరాశి float32x4_t v_d = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d));
        స్థిరరాశి float32x4_t v_acc = vec_madd(v_xyf, v_d, vec_splats(0.0f));

        sumf += vec_hsum_f32x4(v_acc);
    }

    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(x);
    వాడని(y);
    వాడని(ib);
    వాడని(sumf);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_q5_1_q8_1(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_1;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(qk == QK5_1);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_q5_1 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y = vy;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0.0f;

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    float32x4_t v_sum0 = vec_splats(0.0f);
    float32x4_t v_sum1 = vec_splats(0.0f);

    దశాంశం summs0 = 0.0f;
    దశాంశం summs1 = 0.0f;

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    స్థిరరాశి uint8x16_t v_m = vec_splats((uint8_t)0x0F);

    #సూచన GCC unroll 4
    కొరకు (; ib + 1 < nb; ib += 2) {
        స్థిరరాశి ఖండం_q5_1 * GGML_RESTRICT x0 = &x[ib + 0];
        స్థిరరాశి ఖండం_q5_1 * GGML_RESTRICT x1 = &x[ib + 1];
        స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y0 = &y[ib + 0];
        స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y1 = &y[ib + 1];

        summs0 += GGML_CPU_FP16_గమ్యం_FP32(x0->m) * GGML_CPU_FP16_గమ్యం_FP32(y0->s);
        summs1 += GGML_CPU_FP16_గమ్యం_FP32(x1->m) * GGML_CPU_FP16_గమ్యం_FP32(y1->s);

        జ్ఞాపకం_నకలు(&qh0, x0->qh, పరిమాణం(qh0));
        జ్ఞాపకం_నకలు(&qh1, x1->qh, పరిమాణం(qh1));

        tmp0[0] = పట్టిక_b2b_0[(qh0 >>  0) & 0xFF];
        tmp0[1] = పట్టిక_b2b_0[(qh0 >>  8) & 0xFF];
        tmp0[2] = పట్టిక_b2b_0[(qh0 >> 16) & 0xFF];
        tmp0[3] = పట్టిక_b2b_0[(qh0 >> 24)       ];

        tmp1[0] = పట్టిక_b2b_0[(qh1 >>  0) & 0xFF];
        tmp1[1] = పట్టిక_b2b_0[(qh1 >>  8) & 0xFF];
        tmp1[2] = పట్టిక_b2b_0[(qh1 >> 16) & 0xFF];
        tmp1[3] = పట్టిక_b2b_0[(qh1 >> 24)       ];

        int8x16_t v_qh0l = vec_xl(0, (స్థిరరాశి int8_t *)(tmp0 + 0));
        int8x16_t v_qh0h = vec_xl(0, (స్థిరరాశి int8_t *)(tmp0 + 2));
        int8x16_t v_qh1l = vec_xl(0, (స్థిరరాశి int8_t *)(tmp1 + 0));
        int8x16_t v_qh1h = vec_xl(0, (స్థిరరాశి int8_t *)(tmp1 + 2));

        // required for fixing the byteorder
        v_qh0l = vec_perm(v_qh0l, v_qh0l, v_kperm);
        v_qh0h = vec_perm(v_qh0h, v_qh0h, v_kperm);
        v_qh1l = vec_perm(v_qh1l, v_qh1l, v_kperm);
        v_qh1h = vec_perm(v_qh1h, v_qh1h, v_kperm);

        స్థిరరాశి uint8x16_t v_x0 = vec_xl(0, x0->qs);
        స్థిరరాశి uint8x16_t v_x1 = vec_xl(0, x1->qs);

        స్థిరరాశి int8x16_t v_x0l = (int8x16_t)vec_and(v_x0, v_m);
        స్థిరరాశి int8x16_t v_x0h = (int8x16_t)vec_sr(v_x0, 4);
        స్థిరరాశి int8x16_t v_x1l = (int8x16_t)vec_and(v_x1, v_m);
        స్థిరరాశి int8x16_t v_x1h = (int8x16_t)vec_sr(v_x1, 4);

        స్థిరరాశి int8x16_t v_x0lf = vec_or(v_x0l, v_qh0l);
        స్థిరరాశి int8x16_t v_x0hf = vec_or(v_x0h, v_qh0h);
        స్థిరరాశి int8x16_t v_x1lf = vec_or(v_x1l, v_qh1l);
        స్థిరరాశి int8x16_t v_x1hf = vec_or(v_x1h, v_qh1h);

        స్థిరరాశి int8x16_t v_y0l = vec_xl(0      , y0->qs);
        స్థిరరాశి int8x16_t v_y0h = vec_xl(QK8_1/2, y0->qs);
        స్థిరరాశి int8x16_t v_y1l = vec_xl(0      , y1->qs);
        స్థిరరాశి int8x16_t v_y1h = vec_xl(QK8_1/2, y1->qs);

        స్థిరరాశి int32x4_t v_xy0 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_x0lf, v_y0l), v_x0hf, v_y0h);
        స్థిరరాశి int32x4_t v_xy1 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_x1lf, v_y1l), v_x1hf, v_y1h);

        స్థిరరాశి float32x4_t v_xy0f = vec_float(v_xy0);
        స్థిరరాశి float32x4_t v_xy1f = vec_float(v_xy1);

        స్థిరరాశి float32x4_t v_d0 = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d));
        స్థిరరాశి float32x4_t v_d1 = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x1->d) * GGML_CPU_FP16_గమ్యం_FP32(y1->d));

        v_sum0 = vec_madd(v_xy0f, v_d0, v_sum0);
        v_sum1 = vec_madd(v_xy1f, v_d1, v_sum1);
    }

    sumf += vec_hsum_f32x4(v_sum0) + vec_hsum_f32x4(v_sum1) + summs0 + summs1;

    #సూచన GCC unroll 4
    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి ఖండం_q5_1 * GGML_RESTRICT x0 = &x[ib];
        స్థిరరాశి ఖండం_q8_1 * GGML_RESTRICT y0 = &y[ib];

        దశాంశం summs = GGML_CPU_FP16_గమ్యం_FP32(x0->m) * GGML_CPU_FP16_గమ్యం_FP32(y0->s);

        uint32_t qh;
        జ్ఞాపకం_నకలు(&qh, x0->qh, పరిమాణం(qh));

        uint64_t tmp[4];
        tmp[0] = పట్టిక_b2b_0[(qh >>  0) & 0xFF];
        tmp[1] = పట్టిక_b2b_0[(qh >>  8) & 0xFF];
        tmp[2] = పట్టిక_b2b_0[(qh >> 16) & 0xFF];
        tmp[3] = పట్టిక_b2b_0[(qh >> 24)       ];

        int8x16_t v_qhl = vec_xl(0, (స్థిరరాశి int8_t *)(tmp + 0));
        int8x16_t v_qhh = vec_xl(0, (స్థిరరాశి int8_t *)(tmp + 2));

        // required for fixing the byteorder
        v_qhl = vec_perm(v_qhl, v_qhl, v_kperm);
        v_qhh = vec_perm(v_qhh, v_qhh, v_kperm);

        స్థిరరాశి uint8x16_t v_x = vec_xl(0, x0->qs);
        స్థిరరాశి int8x16_t v_xl = (int8x16_t)vec_and(v_x, v_m);
        స్థిరరాశి int8x16_t v_xh = (int8x16_t)vec_sr(v_x, 4);

        స్థిరరాశి int8x16_t v_xlf = vec_or(v_xl, v_qhl);
        స్థిరరాశి int8x16_t v_xhf = vec_or(v_xh, v_qhh);

        స్థిరరాశి int8x16_t v_yl = vec_xl(0      , y0->qs);
        స్థిరరాశి int8x16_t v_yh = vec_xl(QK8_1/2, y0->qs);

        స్థిరరాశి int32x4_t v_xy = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_xlf, v_yl), v_xhf, v_yh);
        స్థిరరాశి float32x4_t v_xyf = vec_float(v_xy);

        స్థిరరాశి float32x4_t v_d = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d));
        స్థిరరాశి float32x4_t v_acc = vec_madd(v_xyf, v_d, vec_splats(0.0f));

        sumf += vec_hsum_f32x4(v_acc) + summs;
    }

    *s = sumf;
#లేకపోతే
    వాడని(nb);
    వాడని(x);
    వాడని(y);
    వాడని(ib);
    వాడని(sumf);
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

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    float32x4_t acc = vec_splats(0.0f);

#సూచన GCC unroll 8
    కొరకు (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        స్థిరరాశి int8x16_t v_xl = vec_xl(0      , x[ib].qs);
        స్థిరరాశి int8x16_t v_xh = vec_xl(QK8_0/2, x[ib].qs);
        స్థిరరాశి int8x16_t v_yl = vec_xl(0      , y[ib].qs);
        స్థిరరాశి int8x16_t v_yh = vec_xl(QK8_0/2, y[ib].qs);

        స్థిరరాశి int32x4_t v_xy_ = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_xl, v_yl), v_xh, v_yh);
        స్థిరరాశి float32x4_t v_xy = vec_float(v_xy_);
        స్థిరరాశి float32x4_t v_d = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d) * GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));

        acc = vec_madd(v_xy, v_d, acc);
    }

    sumf = vec_hsum_f32x4(acc);

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

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    uint32_t aux[3];
    uint32_t utmp[4];

    స్థిరరాశి int32x4_t v_z = vec_splat_s32(0);
    స్థిరరాశి uint8x16_t v_3m = vec_splat_u8(0x03);

    స్థిరరాశి uint8x16_t v_0c = vec_splat_u8(1);
    స్థిరరాశి uint8x16_t v_1c = vec_sl(v_0c, 1);
    స్థిరరాశి uint8x16_t v_2c = vec_sl(v_0c, 2);
    స్థిరరాశి uint8x16_t v_3c = vec_sl(v_0c, 3);

    uint8x16_t q3h[4];
    uint8x16_t q3b[2];
    int8x16_t q3bytes[4];
    int8x16_t q8bytes[8];
    uint8x16_t qhbits[2];

    దశాంశం కూడిక = 0;

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);

        స్థిరరాశి uint8_t * పరిమిత x0l = x[i].qs;
        స్థిరరాశి uint8_t * పరిమిత x0h = x[i].hmask;
        స్థిరరాశి int8_t  * పరిమిత y0  = y[i].qs;

        qhbits[0] = vec_xl(0 , x0h);
        qhbits[1] = vec_xl(16, x0h);

        int32_t isum = 0;

        జ్ఞాపకం_నకలు(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        కొరకు (పూర్ణాంకం j = 0; j < 16; ++j) scale[j] -= 32;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/128; ++j) {
            int32x4_t isum0, isum1, isum2, isum3;

            q3b[0] = vec_xl(0 , x0l);
            q3b[1] = vec_xl(16, x0l);
            x0l += 32;

            q8bytes[0] = vec_xl(0  , y0);
            q8bytes[1] = vec_xl(16 , y0);
            q8bytes[2] = vec_xl(32 , y0);
            q8bytes[3] = vec_xl(48 , y0);
            q8bytes[4] = vec_xl(64 , y0);
            q8bytes[5] = vec_xl(80 , y0);
            q8bytes[6] = vec_xl(96 , y0);
            q8bytes[7] = vec_xl(112, y0);
            y0 += 128;

            q3h[0] = vec_sl(vec_andc(v_0c, qhbits[0]), 2);
            q3h[1] = vec_sl(vec_andc(v_0c, qhbits[1]), 2);
            q3h[2] = vec_sl(vec_andc(v_1c, qhbits[0]), 1);
            q3h[3] = vec_sl(vec_andc(v_1c, qhbits[1]), 1);

            q3bytes[0] = vec_sub((int8x16_t)vec_and(q3b[0], v_3m), (int8x16_t)q3h[0]);
            q3bytes[1] = vec_sub((int8x16_t)vec_and(q3b[1], v_3m), (int8x16_t)q3h[1]);
            q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 2), v_3m), (int8x16_t)q3h[2]);
            q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 2), v_3m), (int8x16_t)q3h[3]);

            isum0 = ggml_vec_dot_s32(v_z, q3bytes[0], q8bytes[0]);
            isum1 = ggml_vec_dot_s32(v_z, q3bytes[1], q8bytes[1]);
            isum2 = ggml_vec_dot_s32(v_z, q3bytes[2], q8bytes[2]);
            isum3 = ggml_vec_dot_s32(v_z, q3bytes[3], q8bytes[3]);

            isum += (isum0[0] + isum0[1] + isum0[2] + isum0[3]) * scale[0];
            isum += (isum1[0] + isum1[1] + isum1[2] + isum1[3]) * scale[1];
            isum += (isum2[0] + isum2[1] + isum2[2] + isum2[3]) * scale[2];
            isum += (isum3[0] + isum3[1] + isum3[2] + isum3[3]) * scale[3];

            scale += 4;

            q3h[0] = vec_andc(v_2c, qhbits[0]);
            q3h[1] = vec_andc(v_2c, qhbits[1]);
            q3h[2] = vec_sr(vec_andc(v_3c, qhbits[0]), 1);
            q3h[3] = vec_sr(vec_andc(v_3c, qhbits[1]), 1);

            q3bytes[0] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 4), v_3m), (int8x16_t)q3h[0]);
            q3bytes[1] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 4), v_3m), (int8x16_t)q3h[1]);
            q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 6), v_3m), (int8x16_t)q3h[2]);
            q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 6), v_3m), (int8x16_t)q3h[3]);

            isum0 = ggml_vec_dot_s32(v_z, q3bytes[0], q8bytes[4]);
            isum1 = ggml_vec_dot_s32(v_z, q3bytes[1], q8bytes[5]);
            isum2 = ggml_vec_dot_s32(v_z, q3bytes[2], q8bytes[6]);
            isum3 = ggml_vec_dot_s32(v_z, q3bytes[3], q8bytes[7]);

            isum += vec_hsum_i32x4(isum0) * scale[0];
            isum += vec_hsum_i32x4(isum1) * scale[1];
            isum += vec_hsum_i32x4(isum2) * scale[2];
            isum += vec_hsum_i32x4(isum3) * scale[3];

            scale += 4;

            ఒకవేళ (j == 0) {
                qhbits[0] = vec_sr(qhbits[0], 4);
                qhbits[1] = vec_sr(qhbits[1], 4);
            }
        }

        కూడిక += d * isum;
    }

    *s = కూడిక;

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

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    స్థిరరాశి uint8x16_t v_lm = vec_splat_u8(0x0F);
    స్థిరరాశి int32x4_t v_z = vec_splat_s32(0);

    uint8x16_t v_x[2];
    int8x16_t  v_xl[2];
    int8x16_t  v_y[2];

    దశాంశం sumf = 0;

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);
        స్థిరరాశి దశాంశం dmin = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin);

        స్థిరరాశి int16x8_t v_ysumsl = vec_xl(0 , y[i].bsums);
        స్థిరరాశి int16x8_t v_ysumsh = vec_xl(16, y[i].bsums);
        స్థిరరాశి int16x8_t v_ysums = vec_padd_s16(v_ysumsl, v_ysumsh);

        జ్ఞాపకం_నకలు(utmp, x[i].scales, 12);

        uint32x4_t v_mins8 = { 0 };
        v_mins8 = vec_చొప్పించు(utmp[1] & kmask1, v_mins8, 0);
        v_mins8 = vec_చొప్పించు(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), v_mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        స్థిరరాశి int16x8_t v_minsh = (int16x8_t)vec_unpackh((uint8x16_t)v_mins8);

        స్థిరరాశి int32x4_t v_minso = vec_mulo(v_ysums, v_minsh);
        స్థిరరాశి int32x4_t v_mins = vec_meadd(v_ysums, v_minsh, v_minso);
        sumf -= dmin * (v_mins[0] + v_mins[1] + v_mins[2] + v_mins[3]);

        స్థిరరాశి uint8_t * scales = (స్థిరరాశి uint8_t *)utmp;
        స్థిరరాశి uint8_t * GGML_RESTRICT x0 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT y0 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/64; ++j) {
            v_x[0] = vec_xl(0 , x0);
            v_x[1] = vec_xl(16, x0);
            x0 += 32;

            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            y0 += 32;

            v_xl[0] = (int8x16_t)vec_and(v_x[0], v_lm);
            v_xl[1] = (int8x16_t)vec_and(v_x[1], v_lm);

            స్థిరరాశి int32x4_t p1 = ggml_vec_dot_s32(ggml_vec_dot_s32(v_z, v_xl[0], v_y[0]), v_xl[1], v_y[1]);
            sumi1 += vec_hsum_i32x4(p1) * scales[2*j+0];

            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            y0 += 32;

            v_xl[0] = (int8x16_t)vec_sr(v_x[0], 4);
            v_xl[1] = (int8x16_t)vec_sr(v_x[1], 4);

            స్థిరరాశి int32x4_t p2 = ggml_vec_dot_s32(ggml_vec_dot_s32(v_z, v_xl[0], v_y[0]), v_xl[1], v_y[1]);
            sumi2 += vec_hsum_i32x4(p2) * scales[2*j+1];
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

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    స్థిరరాశి uint8x16_t v_lm = vec_splat_u8(0x0F);
    స్థిరరాశి uint8x16_t v_1m = vec_splat_u8(0x01);
    స్థిరరాశి uint8x16_t v_2m = vec_splat_u8(0x02);

    స్థిరరాశి int32x4_t v_z = vec_splat_s32(0);

    స్థిరరాశి uchar8x16_t v_minsm = {
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
    };

    int8x16_t  q5b[4];
    uint8x16_t q5h[4];

    uint8x16_t v_xl[2];
    uint8x16_t v_xh[2];
    int8x16_t  v_y[4];

    దశాంశం sumf = 0;

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].d);
        స్థిరరాశి దశాంశం dmin = y[i].d * GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin);

        స్థిరరాశి int16x8_t v_ysumsl = vec_xl(0 , y[i].bsums);
        స్థిరరాశి int16x8_t v_ysumsh = vec_xl(16, y[i].bsums);
        స్థిరరాశి int16x8_t v_ysums = vec_padd_s16(v_ysumsl, v_ysumsh);

        జ్ఞాపకం_నకలు(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        స్థిరరాశి uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        స్థిరరాశి uint8x16_t v_mins16 = vec_xl(0, (స్థిరరాశి uint8_t *)utmp);
        స్థిరరాశి uint8x16_t v_mins8 = vec_perm(v_mins16, v_mins16, v_minsm);
        స్థిరరాశి int16x8_t v_minsh = (int16x8_t)vec_unpackh(v_mins8);

        స్థిరరాశి int32x4_t v_minsho = vec_mulo(v_ysums, v_minsh);
        స్థిరరాశి int32x4_t v_mins = vec_meadd(v_ysums, v_minsh, v_minsho);
        స్థిరరాశి int32_t mins = vec_hsum_i32x4(v_mins);

        స్థిరరాశి uint8_t * scales = (స్థిరరాశి uint8_t *)utmp;
        స్థిరరాశి uint8_t * GGML_RESTRICT x0l = x[i].qs;
        స్థిరరాశి uint8_t * GGML_RESTRICT x0h = x[i].qh;
        స్థిరరాశి int8_t  * GGML_RESTRICT y0 = y[i].qs;

        v_xh[0] = vec_xl(0 , x0h);
        v_xh[1] = vec_xl(16, x0h);

        int32_t sumi = 0;
        కొరకు (పూర్ణాంకం j = 0; j < QK_K/64; ++j) {
            v_xl[0] = vec_xl(0 , x0l);
            v_xl[1] = vec_xl(16, x0l);
            x0l += 32;

            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            v_y[2] = vec_xl(32, y0);
            v_y[3] = vec_xl(48, y0);
            y0 += 64;

            q5h[0] = vec_sl(vec_and(v_1m, v_xh[0]), 4);
            q5h[1] = vec_sl(vec_and(v_1m, v_xh[1]), 4);
            q5h[2] = vec_sl(vec_and(v_2m, v_xh[0]), 3);
            q5h[3] = vec_sl(vec_and(v_2m, v_xh[1]), 3);
            v_xh[0] = vec_sr(v_xh[0], 2);
            v_xh[1] = vec_sr(v_xh[1], 2);

            q5b[0] = (int8x16_t)vec_or(vec_and(v_xl[0], v_lm), q5h[0]);
            q5b[1] = (int8x16_t)vec_or(vec_and(v_xl[1], v_lm), q5h[1]);
            q5b[2] = (int8x16_t)vec_or(vec_sr(v_xl[0], 4), q5h[2]);
            q5b[3] = (int8x16_t)vec_or(vec_sr(v_xl[1], 4), q5h[3]);

            int32x4_t sumi0 = ggml_vec_dot_s32(ggml_vec_dot_s32(v_z, q5b[0], v_y[0]), q5b[1], v_y[1]);
            int32x4_t sumi1 = ggml_vec_dot_s32(ggml_vec_dot_s32(v_z, q5b[2], v_y[2]), q5b[3], v_y[3]);

            sumi += vec_hsum_i32x4(sumi0) * *scales++;
            sumi += vec_hsum_i32x4(sumi1) * *scales++;
        }

        sumf += d * sumi - dmin * mins;
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

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    దశాంశం కూడిక = 0;

    // Lower 4-bit and upper 2-bit masks
    స్థిరరాశి uint8x16_t v_lm = vec_splat_u8(0x0F);
    స్థిరరాశి uint8x16_t v_um = vec_splat_u8(0x03);

    స్థిరరాశి int32x4_t v_z = vec_splat_s32(0);

    int8x16_t  q6b[4];
    uint8x16_t q6h[4];

    uint8x16_t v_xl[4];
    uint8x16_t v_xh[2];
    int8x16_t  v_y[4];

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        స్థిరరాశి దశాంశం d_all = GGML_CPU_FP16_గమ్యం_FP32(x[i].d);

        స్థిరరాశి uint8_t * GGML_RESTRICT x0l = x[i].ql;
        స్థిరరాశి uint8_t * GGML_RESTRICT x0h = x[i].qh;
        స్థిరరాశి int8_t  * GGML_RESTRICT y0 = y[i].qs;

        స్థిరరాశి int8_t  * GGML_RESTRICT scale = x[i].scales;

        స్థిరరాశి int16x8_t v_ysumsl = vec_xl(0 , y[i].bsums);
        స్థిరరాశి int16x8_t v_ysumsh = vec_xl(16, y[i].bsums);

        స్థిరరాశి int8x16_t v_scale  = vec_xl(0, scale);
        స్థిరరాశి int16x8_t v_scalel = vec_unpackh(v_scale);
        స్థిరరాశి int16x8_t v_scaleh = vec_unpackl(v_scale);

        స్థిరరాశి int32x4_t v_minslo = vec_mulo(v_ysumsl, v_scalel);
        స్థిరరాశి int32x4_t v_minsl = vec_meadd(v_ysumsl, v_scalel, v_minslo);
        స్థిరరాశి int32x4_t v_minsho = vec_mulo(v_ysumsh, v_scaleh);
        స్థిరరాశి int32x4_t v_minsh = vec_meadd(v_ysumsh, v_scaleh, v_minsho);
        స్థిరరాశి int32x4_t v_mins = vec_కూడు(v_minsl, v_minsh);

        స్థిరరాశి int32_t mins = vec_hsum_i32x4(v_mins);

        int32_t isum = 0;
        కొరకు (పూర్ణాంకం j = 0; j < QK_K/128; ++j) {
            // Load model upper 2 bits
            v_xh[0] = vec_xl(0 , x0h);
            v_xh[1] = vec_xl(16, x0h);
            x0h += 32;

            // Load model lower 4 bits
            v_xl[0] = vec_xl(0 , x0l);
            v_xl[1] = vec_xl(16, x0l);
            v_xl[2] = vec_xl(32, x0l);
            v_xl[3] = vec_xl(48, x0l);
            x0l += 64;

            // Load activation quants
            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            v_y[2] = vec_xl(32, y0);
            v_y[3] = vec_xl(48, y0);
            y0 += 64;

            q6h[0] = vec_sl(vec_and(v_um, v_xh[0]), 4);
            q6h[1] = vec_sl(vec_and(v_um, v_xh[1]), 4);
            uint8x16_t shifted = vec_sr(v_xh[0], 2);
            q6h[2] = vec_sl(vec_and(v_um, shifted), 4);
            shifted = vec_sr(v_xh[1], 2);
            q6h[3] = vec_sl(vec_and(v_um, shifted), 4);

            q6b[0] = (int8x16_t)(vec_or(vec_and(v_xl[0], v_lm), q6h[0]));
            q6b[1] = (int8x16_t)(vec_or(vec_and(v_xl[1], v_lm), q6h[1]));
            q6b[2] = (int8x16_t)(vec_or(vec_and(v_xl[2], v_lm), q6h[2]));
            q6b[3] = (int8x16_t)(vec_or(vec_and(v_xl[3], v_lm), q6h[3]));

            int32x4_t summs0 = ggml_vec_dot_s32(v_z, q6b[0], v_y[0]);
            int32x4_t summs1 = ggml_vec_dot_s32(v_z, q6b[1], v_y[1]);
            int32x4_t summs2 = ggml_vec_dot_s32(v_z, q6b[2], v_y[2]);
            int32x4_t summs3 = ggml_vec_dot_s32(v_z, q6b[3], v_y[3]);

            isum += vec_hsum_i32x4(summs0) * scale[0] +
                    vec_hsum_i32x4(summs1) * scale[1] +
                    vec_hsum_i32x4(summs2) * scale[2] +
                    vec_hsum_i32x4(summs3) * scale[3];

            scale += 4;


            // Load activation quants
            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            v_y[2] = vec_xl(32, y0);
            v_y[3] = vec_xl(48, y0);
            y0 += 64;

            shifted = vec_sr(v_xh[0], 4);
            q6h[0] = vec_sl(vec_and(v_um, shifted), 4);
            shifted = vec_sr(v_xh[1], 4);
            q6h[1] = vec_sl(vec_and(v_um, shifted), 4);
            shifted = vec_sr(v_xh[0], 6);
            q6h[2] = vec_sl(vec_and(v_um, shifted), 4);
            shifted = vec_sr(v_xh[1], 6);
            q6h[3] = vec_sl(vec_and(v_um, shifted), 4);

            q6b[0] = (int8x16_t)(vec_or(vec_sr(v_xl[0], 4), q6h[0]));
            q6b[1] = (int8x16_t)(vec_or(vec_sr(v_xl[1], 4), q6h[1]));
            q6b[2] = (int8x16_t)(vec_or(vec_sr(v_xl[2], 4), q6h[2]));
            q6b[3] = (int8x16_t)(vec_or(vec_sr(v_xl[3], 4), q6h[3]));

            summs0 = ggml_vec_dot_s32(v_z, q6b[0], v_y[0]);
            summs1 = ggml_vec_dot_s32(v_z, q6b[1], v_y[1]);
            summs2 = ggml_vec_dot_s32(v_z, q6b[2], v_y[2]);
            summs3 = ggml_vec_dot_s32(v_z, q6b[3], v_y[3]);

            isum += vec_hsum_i32x4(summs0) * scale[0] +
                    vec_hsum_i32x4(summs1) * scale[1] +
                    vec_hsum_i32x4(summs2) * scale[2] +
                    vec_hsum_i32x4(summs3) * scale[3];

            scale += 4;
        }

        కూడిక += d_all * y[i].d * (isum - 32 * mins);
    }

    *s = కూడిక;

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

// #if defined(__VXE__) || defined(__VXE2__)
// static const int8_t keven_signs_q2xs[1024] = {
//      1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
//      1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
//      1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
//      1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
//      1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
//      1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
//      1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
//      1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
//      1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
//      1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
//      1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
//      1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
//      1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
//      1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
//      1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
//      1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
//      1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
//      1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
//      1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
//      1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
//      1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
//      1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
//      1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
//      1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
//      1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
//      1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
//      1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
//      1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
//      1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
//      1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
//      1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
//      1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
// };
// #endif

// void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
//     assert(n % QK_K == 0);
//     assert(nrc == 1);
//     UNUSED(nrc);
//     UNUSED(bx);
//     UNUSED(by);
//     UNUSED(bs);

//     const block_iq2_xxs * GGML_RESTRICT x = vx;
//     const block_q8_K    * GGML_RESTRICT y = vy;

//     const int nb = n / QK_K;

// #if defined(__VXE__) || defined(__VXE2__)
//    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

//    uint32_t aux32[4];
//    const uint8_t * aux8 = (const uint8_t *)aux32;

//    float sumf = 0;

//    for (int i = 0; i < nb; ++i) {
//        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
//        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
//        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

//        float sumf1 = 0, sumf2 = 0;

//        for (int ib32 = 0; ib32 < QK_K/32; ib += 2) {
//            int8x16_t q8b0 = vec_xl( 0, q8);
//            int8x16_t qb81 = vec_xl(16, q8);
//            int8x16_t q8b2 = vec_xl(32, q8);
//            int8x16_t q8b3 = vec_xl(48, q8);
//            q8 += 64;

//            memcpy(aux32, q2, 4 * sizeof(uint32_t));
//            q2 += 8;

//            int8x16_t q2u0 = { *(const int64_t *)(iq2xxs_grid + aux8[ 0]), *(const int64_t *)(iq2xxs_grid + aux8[ 1]) };
//            int8x16_t q2u1 = { *(const int64_t *)(iq2xxs_grid + aux8[ 2]), *(const int64_t *)(iq2xxs_grid + aux8[ 3]) };
//            int8x16_t q2u2 = { *(const int64_t *)(iq2xxs_grid + aux8[ 8]), *(const int64_t *)(iq2xxs_grid + aux8[ 9]) };
//            int8x16_t q2u3 = { *(const int64_t *)(iq2xxs_grid + aux8[10]), *(const int64_t *)(iq2xxs_grid + aux8[11]) };

//            int8x16_t q2s0 = { *(const int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >>  7) & 127)) };
//            int8x16_t q2s1 = { *(const int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >> 21) & 127)) };
//            int8x16_t q2s2 = { *(const int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >>  7) & 127)) };
//            int8x16_t q2s3 = { *(const int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >> 21) & 127)) };

//            q2u0 = vec_mul(q2u0, q2s0);
//            q2u1 = vec_mul(q2u1, q2s1);
//            q2u2 = vec_mul(q2u2, q2s2);
//            q2u3 = vec_mul(q2u3, q2s3);

//            const int32x4_t p1 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splat_s32(0), q2u0, q8b0), q2u1, q8b1);
//            const int32x4_t p2 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splat_s32(0), q2u2, q8b2), q2u3, q8b3);

//            sumf1 += (p1[0] + p1[1] + p1[2] + p1[3]) * (0.5f + (aux32[1] >> 28));
//            sumf2 += (p2[0] + p2[1] + p2[2] + p2[3]) * (0.5f + (aux32[3] >> 28));
//        }

//        sumf += d * (sumf1 + sumf2);
//    }

//    *s = 0.25f * sumf;

// #else

//     uint32_t aux32[2];
//     const uint8_t * aux8 = (const uint8_t *)aux32;

//     float sumf = 0.f;
//     for (int i = 0; i < nb; ++i) {
//         const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
//         const uint16_t * GGML_RESTRICT q2 = x[i].qs;
//         const int8_t   * GGML_RESTRICT q8 = y[i].qs;
//         int32_t bsum = 0;
//         for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
//             memcpy(aux32, q2, 2*sizeof(uint32_t));
//             q2 += 4;
//             const uint32_t ls = 2*(aux32[1] >> 28) + 1;
//             int32_t sumi = 0;
//             for (int l = 0; l < 4; ++l) {
//                 const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
//                 const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
//                 for (int j = 0; j < 8; ++j) {
//                     sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
//                 }
//                 q8 += 8;
//             }
//             bsum += sumi * ls;
//         }
//         sumf += d * bsum;
//     }
//     *s = 0.125f * sumf;
// #endif
// }

శూన్య ggml_vec_dot_iq4_nl_q8_0(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);
    ప్రతిజ్ఞా(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    స్థిరరాశి ఖండం_iq4_nl * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0   * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK4_NL;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    స్థిరరాశి int8x16_t v_k = vec_xl(0, kvalues_iq4nl);
    స్థిరరాశి uint8x16_t v_m = vec_splat_u8(0x0F);

    కొరకు (; ib < nb; ++ib) {
        స్థిరరాశి ఖండం_iq4_nl * GGML_RESTRICT x0 = &x[ib];
        స్థిరరాశి ఖండం_q8_0   * GGML_RESTRICT y0 = &y[ib];

        స్థిరరాశి uint8x16_t v_x = vec_xl(0, x0->qs);
        int8x16_t v_xl = (int8x16_t)vec_and(v_x, v_m);
        int8x16_t v_xh = (int8x16_t)vec_sr(v_x, 4);

        v_xl = vec_perm(v_k, v_k, (uchar8x16_t)v_xl);
        v_xh = vec_perm(v_k, v_k, (uchar8x16_t)v_xh);

        స్థిరరాశి int8x16_t v_yl = vec_xl(0      , y0->qs);
        స్థిరరాశి int8x16_t v_yh = vec_xl(QK8_0/2, y0->qs);
        స్థిరరాశి int32x4_t v_xy = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_xl, v_yl), v_xh, v_yh);

        sumf += GGML_CPU_FP16_గమ్యం_FP32(x0->d) * GGML_CPU_FP16_గమ్యం_FP32(y0->d) * vec_hsum_i32x4(v_xy);
    }

    *s = sumf;
#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    వాడని(ib);
    వాడని(sumf);
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq4_xs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);
    ప్రతిజ్ఞా(n % QK_K == 0);

    స్థిరరాశి ఖండం_iq4_xs * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K   * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__VXE__) || defined(__VXE2__)
    స్థిరరాశి int8x16_t v_k = vec_xl(0, kvalues_iq4nl);
    స్థిరరాశి uint8x16_t v_m = vec_splat_u8(0x0F);

    దశాంశం sumf = 0;

    కొరకు (పూర్ణాంకం ibl = 0; ibl < nb; ++ibl) {
        స్థిరరాశి uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        uint16_t h = x[ibl].scales_h;

        పూర్ణాంకం sumi1 = 0, sumi2 = 0;
        కొరకు (పూర్ణాంకం ib = 0; ib < QK_K/64; ++ib) {
            స్థిరరాశి uint8x16_t v_x0 = vec_xl(0       , q4);
            స్థిరరాశి uint8x16_t v_x1 = vec_xl(QK4_NL/2, q4);
            q4 += 32;

            int8x16_t v_x0l = (int8x16_t)vec_and(v_x0, v_m);
            int8x16_t v_x0h = (int8x16_t)vec_sr(v_x0, 4);
            int8x16_t v_x1l = (int8x16_t)vec_and(v_x1, v_m);
            int8x16_t v_x1h = (int8x16_t)vec_sr(v_x1, 4);

            v_x0l = vec_perm(v_k, v_k, (uchar8x16_t)v_x0l);
            v_x0h = vec_perm(v_k, v_k, (uchar8x16_t)v_x0h);
            v_x1l = vec_perm(v_k, v_k, (uchar8x16_t)v_x1l);
            v_x1h = vec_perm(v_k, v_k, (uchar8x16_t)v_x1h);

            స్థిరరాశి int8x16_t v_y0 = vec_xl( 0, q8);
            స్థిరరాశి int8x16_t v_y1 = vec_xl(16, q8);
            స్థిరరాశి int8x16_t v_y2 = vec_xl(32, q8);
            స్థిరరాశి int8x16_t v_y3 = vec_xl(48, q8);
            q8 += 64;

            int32x4_t vsumi0 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_x0l, v_y0), v_x0h, v_y1);
            int32x4_t vsumi1 = ggml_vec_dot_s32(ggml_vec_dot_s32(vec_splats(0), v_x1l, v_y2), v_x1h, v_y3);

            పూర్ణాంకం ls1 = ((x[ibl].scales_l[ib] & 0xF) | ((h << 4) & 0x30)) - 32;
            పూర్ణాంకం ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;

            h >>= 4;

            sumi1 += vec_hsum_i32x4(vsumi0) * ls1;
            sumi2 += vec_hsum_i32x4(vsumi1) * ls2;
        }

        sumf += GGML_CPU_FP16_గమ్యం_FP32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2);
    }

    *s = sumf;

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}
