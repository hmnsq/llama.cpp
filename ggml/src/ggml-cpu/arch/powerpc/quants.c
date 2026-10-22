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

#ఒకవేళ defined(__POWER9_VECTOR__)
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

#ఒకవేళ defined(__POWER9_VECTOR__)
    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        సదిశ దశాంశం srcv [8];
        సదిశ దశాంశం asrcv[8];
        సదిశ దశాంశం amaxv[8];
        సదిశ గుర్తుగల పూర్ణాంకం vi[8];

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        కొరకు (పూర్ణాంకం j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        కొరకు (పూర్ణాంకం j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        కొరకు (పూర్ణాంకం j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        స్థిరరాశి దశాంశం amax = MAX(MAX(vec_వెలికితీయి(amaxv[0], 0),
                                   vec_వెలికితీయి(amaxv[0], 1)),
                               MAX(vec_వెలికితీయి(amaxv[0], 2),
                                   vec_వెలికితీయి(amaxv[0], 3)));

        స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
        స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;
        స్థిరరాశి సదిశ దశాంశం vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_గమ్యం_FP16(d);

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            స్థిరరాశి సదిశ దశాంశం v  = vec_విడత(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
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

#ఒకవేళ defined(__POWER9_VECTOR__)
    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        సదిశ దశాంశం srcv [8];
        సదిశ దశాంశం asrcv[8];
        సదిశ దశాంశం amaxv[8];
        సదిశ గుర్తుగల పూర్ణాంకం vi[8];

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        కొరకు (పూర్ణాంకం j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        కొరకు (పూర్ణాంకం j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        కొరకు (పూర్ణాంకం j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        స్థిరరాశి దశాంశం amax = MAX(MAX(vec_వెలికితీయి(amaxv[0], 0),
                                   vec_వెలికితీయి(amaxv[0], 1)),
                               MAX(vec_వెలికితీయి(amaxv[0], 2),
                                   vec_వెలికితీయి(amaxv[0], 3)));

        స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
        స్థిరరాశి దశాంశం id = d ? 1.0f/d : 0.0f;
        స్థిరరాశి సదిశ దశాంశం vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_గమ్యం_FP16(d);

        సదిశ పూర్ణాంకం accv = vec_splats(0);

        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            స్థిరరాశి సదిశ దశాంశం v  = vec_విడత(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_కూడు(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_కూడు(accv, vec_sld(accv, accv, 4));
        accv = vec_కూడు(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_CPU_FP32_గమ్యం_FP16(d * vec_వెలికితీయి(accv, 0));
    }

#లేకపోతే
    GGML_వాడని(nb);
    // scalar
    quantize_అడ్డువరుస_q8_1_ref(x, y, k);
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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ గుర్తుగల పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం v8 = vec_splats((గుర్తుగల అక్షరం)0x8);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);

#సూచన GCC unroll 8
    కొరకు (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d));
        సదిశ దశాంశం vyd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల అక్షరం qxs = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, x[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, y[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, y[ib].qs);

        సదిశ గుర్తుగల అక్షరం q4x0 = vec_and(qxs, దిగువ_ముసుగు);
        సదిశ గుర్తుగల అక్షరం q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_వెలికితీయి(vsumf0, 0);

    *s = sumf;
#లేకపోతే
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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ గుర్తుగల పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);

#సూచన GCC unroll 4
    కొరకు (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d));
        సదిశ దశాంశం vyd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ దశాంశం vxmin = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].m));
        సదిశ దశాంశం vys = {GGML_CPU_FP16_గమ్యం_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        సదిశ గుర్తుగల అక్షరం qxs = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, x[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, y[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, y[ib].qs);

        సదిశ గుర్తులేని అక్షరం q4x0 = (సదిశ గుర్తులేని అక్షరం)vec_and(qxs, దిగువ_ముసుగు);
        సదిశ గుర్తులేని అక్షరం q4x1 = (సదిశ గుర్తులేని అక్షరం)vec_sr(qxs, v4);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_వెలికితీయి(vsumf0, 0);

    *s = sumf;
#లేకపోతే
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

    స్థిరరాశి ఖండం_mxfp4 * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_MXFP4;

    పూర్ణాంకం ib = 0;
    దశాంశం sumf = 0;

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం vshift4 = vec_splats((గుర్తులేని అక్షరం)4);
    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);

    సదిశ గుర్తుగల అక్షరం kv = vec_xl(0, (స్థిరరాశి గుర్తుగల అక్షరం *)kvalues_mxfp4);

#సూచన GCC unroll 8
    కొరకు (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        సదిశ దశాంశం vyd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(y[ib].d) *
                                      GGML_E8_M0_గమ్యం_FP32_HALF(x[ib].e));

        సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, y[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, y[ib].qs);

        సదిశ గుర్తుగల అక్షరం qxs = (సదిశ గుర్తుగల అక్షరం)vec_xl(0, x[ib].qs);

        సదిశ గుర్తులేని అక్షరం lo_nibbles = (సదిశ గుర్తులేని అక్షరం)vec_and(qxs, దిగువ_ముసుగు);
        సదిశ గుర్తులేని అక్షరం hi_nibbles = (సదిశ గుర్తులేని అక్షరం)vec_sr(qxs, vshift4);

        సదిశ గుర్తుగల అక్షరం q4x0 = vec_perm(kv, kv, lo_nibbles);
        సదిశ గుర్తుగల అక్షరం q4x1 = vec_perm(kv, kv, hi_nibbles);

        సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = vec_splats((int32_t)0);
        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vyd, vsumf0);
    }

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));
    sumf = vec_వెలికితీయి(vsumf0, 0);
    *s = sumf;
#లేకపోతే
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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)4);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);

#సూచన GCC unroll 4
    కొరకు (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d));
        సదిశ దశాంశం vyd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_0 = {(uint64_t)(పట్టిక_b2b_1[x[ib].qh[0]]), (uint64_t)(పట్టిక_b2b_1[x[ib].qh[1]])};
        సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_1 = {(uint64_t)(పట్టిక_b2b_1[x[ib].qh[2]]), (uint64_t)(పట్టిక_b2b_1[x[ib].qh[3]])};

        సదిశ గుర్తుగల అక్షరం qh0 = (సదిశ గుర్తుగల అక్షరం)aux64x2_0;
        సదిశ గుర్తుగల అక్షరం qh1 = (సదిశ గుర్తుగల అక్షరం)aux64x2_1;

        సదిశ గుర్తుగల అక్షరం qxs = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, x[ib].qs);

        సదిశ గుర్తుగల అక్షరం q5x0 = vec_sub(vec_and (qxs, దిగువ_ముసుగు), qh0);
        సదిశ గుర్తుగల అక్షరం q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

        సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl(  0, y[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl( 16, y[ib].qs);

        సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
        సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q5x1, q8y1), vec_mulo(q5x1, q8y1));

        qv0 = vec_కూడు(qv0, qv1);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = vec_కూడు(vec_unpackh(qv0), vec_unpackl(qv0));

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_వెలికితీయి(vsumf0, 0);

    *s = sumf;
#లేకపోతే
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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ గుర్తుగల పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);

#సూచన GCC unroll 4
    కొరకు (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d));
        సదిశ దశాంశం vyd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ దశాంశం vxmin = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].m));
        సదిశ దశాంశం vys = {GGML_CPU_FP16_గమ్యం_FP32(y[ib].s), 0.f, 0.f, 0.f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        సదిశ గుర్తులేని దీర్ఘ దీర్ఘ aux64x2_0 = {(uint64_t)(పట్టిక_b2b_0[x[ib].qh[0]]), (uint64_t)(పట్టిక_b2b_0[x[ib].qh[1]])};
        సదిశ గుర్తులేని దీర్ఘ దీర్ఘ aux64x2_1 = {(uint64_t)(పట్టిక_b2b_0[x[ib].qh[2]]), (uint64_t)(పట్టిక_b2b_0[x[ib].qh[3]])};

        సదిశ గుర్తుగల అక్షరం qh0 = (సదిశ గుర్తుగల అక్షరం)aux64x2_0;
        సదిశ గుర్తుగల అక్షరం qh1 = (సదిశ గుర్తుగల అక్షరం)aux64x2_1;

        సదిశ గుర్తుగల అక్షరం qxs = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, x[ib].qs);

        సదిశ గుర్తులేని అక్షరం q5x0 = (సదిశ గుర్తులేని అక్షరం)vec_or(vec_and(qxs, దిగువ_ముసుగు), qh0);
        సదిశ గుర్తులేని అక్షరం q5x1 = (సదిశ గుర్తులేని అక్షరం)vec_or(vec_sr(qxs, v4), qh1);

        సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl(  0, y[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl( 16, y[ib].qs);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q5x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q5x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_వెలికితీయి(vsumf0, 0);

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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల పూర్ణాంకం v0 = vec_splats((int32_t)0);
    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);

#సూచన GCC unroll 8
    కొరకు (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d));
        సదిశ దశాంశం vyd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల అక్షరం q8x0 = vec_xl( 0, x[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8x1 = vec_xl(16, x[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, y[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, y[ib].qs);

        సదిశ గుర్తుగల పొట్టి qv0 = vec_mule(q8x0, q8y0);
        సదిశ గుర్తుగల పొట్టి qv1 = vec_mulo(q8x0, q8y0);
        సదిశ గుర్తుగల పొట్టి qv2 = vec_mule(q8x1, q8y1);
        సదిశ గుర్తుగల పొట్టి qv3 = vec_mulo(q8x1, q8y1);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_కూడు(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_వెలికితీయి(vsumf0, 0);

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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0x3);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_Scale_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v2 = vec_splats((గుర్తులేని అక్షరం)0x2);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v6 = vec_splats((గుర్తులేని అక్షరం)0x6);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ దశాంశం vxmin = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin));
        సదిశ దశాంశం vdmin = vec_mul(vxmin, vyd);

        సదిశ గుర్తుగల పొట్టి q8ysums0 = vec_xl( 0, y[i].bsums);
        సదిశ గుర్తుగల పొట్టి q8ysums1 = vec_xl(16, y[i].bsums);

        సదిశ గుర్తుగల అక్షరం q2xmins = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, x[i].scales);
        సదిశ గుర్తుగల అక్షరం vscales = vec_and(q2xmins, దిగువ_Scale_ముసుగు);

        q2xmins = vec_sr(q2xmins, v4);
        సదిశ గుర్తుగల పొట్టి q2xmins0 = vec_unpackh(q2xmins);
        సదిశ గుర్తుగల పొట్టి q2xmins1 = vec_unpackl(q2xmins);

        సదిశ గుర్తుగల పూర్ణాంకం prod0 = vec_mule(q2xmins0, q8ysums0);
        సదిశ గుర్తుగల పూర్ణాంకం prod1 = vec_mulo(q2xmins0, q8ysums0);
        సదిశ గుర్తుగల పూర్ణాంకం prod2 = vec_mule(q2xmins1, q8ysums1);
        సదిశ గుర్తుగల పూర్ణాంకం prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi4 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi5 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi6 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi7 = v0;

        స్థిరరాశి uint8_t * GGML_RESTRICT q2 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తుగల అక్షరం qxs0 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, q2);
            సదిశ గుర్తుగల అక్షరం qxs1 = (సదిశ గుర్తుగల అక్షరం)vec_xl(16, q2);
            q2 += 32;

            సదిశ గుర్తులేని అక్షరం q2x00 = (సదిశ గుర్తులేని అక్షరం)vec_and(qxs0, దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q2x01 = (సదిశ గుర్తులేని అక్షరం)vec_and(vec_sr(qxs0, v2), దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q2x02 = (సదిశ గుర్తులేని అక్షరం)vec_and(vec_sr(qxs0, v4), దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q2x03 = (సదిశ గుర్తులేని అక్షరం)vec_and(vec_sr(qxs0, v6), దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q2x10 = (సదిశ గుర్తులేని అక్షరం)vec_and(qxs1, దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q2x11 = (సదిశ గుర్తులేని అక్షరం)vec_and(vec_sr(qxs1, v2), దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q2x12 = (సదిశ గుర్తులేని అక్షరం)vec_and(vec_sr(qxs1, v4), దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q2x13 = (సదిశ గుర్తులేని అక్షరం)vec_and(vec_sr(qxs1, v6), దిగువ_ముసుగు);

            సదిశ గుర్తుగల అక్షరం q8y00 = vec_xl(  0, q8);
            సదిశ గుర్తుగల అక్షరం q8y10 = vec_xl( 16, q8);
            సదిశ గుర్తుగల అక్షరం q8y01 = vec_xl( 32, q8);
            సదిశ గుర్తుగల అక్షరం q8y11 = vec_xl( 48, q8);
            సదిశ గుర్తుగల అక్షరం q8y02 = vec_xl( 64, q8);
            సదిశ గుర్తుగల అక్షరం q8y12 = vec_xl( 80, q8);
            సదిశ గుర్తుగల అక్షరం q8y03 = vec_xl( 96, q8);
            సదిశ గుర్తుగల అక్షరం q8y13 = vec_xl(112, q8);
            q8 += 128;

            సదిశ గుర్తుగల పూర్ణాంకం qv0 = vec_msum(q8y00, q2x00, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv1 = vec_msum(q8y01, q2x01, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv2 = vec_msum(q8y02, q2x02, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv3 = vec_msum(q8y03, q2x03, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv4 = vec_msum(q8y10, q2x10, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv5 = vec_msum(q8y11, q2x11, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv6 = vec_msum(q8y12, q2x12, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv7 = vec_msum(q8y13, q2x13, v0);

            సదిశ గుర్తుగల పొట్టి vscales_07 = vec_unpackh(vscales);
            సదిశ గుర్తుగల పూర్ణాంకం vscales_03 = vec_unpackh(vscales_07);
            సదిశ గుర్తుగల పూర్ణాంకం vscales_47 = vec_unpackl(vscales_07);
            సదిశ గుర్తుగల పూర్ణాంకం vs0 = vec_splat(vscales_03, 0);
            సదిశ గుర్తుగల పూర్ణాంకం vs1 = vec_splat(vscales_03, 1);
            సదిశ గుర్తుగల పూర్ణాంకం vs2 = vec_splat(vscales_03, 2);
            సదిశ గుర్తుగల పూర్ణాంకం vs3 = vec_splat(vscales_03, 3);
            సదిశ గుర్తుగల పూర్ణాంకం vs4 = vec_splat(vscales_47, 0);
            సదిశ గుర్తుగల పూర్ణాంకం vs5 = vec_splat(vscales_47, 1);
            సదిశ గుర్తుగల పూర్ణాంకం vs6 = vec_splat(vscales_47, 2);
            సదిశ గుర్తుగల పూర్ణాంకం vs7 = vec_splat(vscales_47, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_కూడు(vec_mul(qv0, vs0), vsumi0);
            vsumi1 = vec_కూడు(vec_mul(qv1, vs2), vsumi1);
            vsumi2 = vec_కూడు(vec_mul(qv2, vs4), vsumi2);
            vsumi3 = vec_కూడు(vec_mul(qv3, vs6), vsumi3);
            vsumi4 = vec_కూడు(vec_mul(qv4, vs1), vsumi4);
            vsumi5 = vec_కూడు(vec_mul(qv5, vs3), vsumi5);
            vsumi6 = vec_కూడు(vec_mul(qv6, vs5), vsumi6);
            vsumi7 = vec_కూడు(vec_mul(qv7, vs7), vsumi7);
        }

        vsumi0 = vec_కూడు(vsumi0, vsumi4);
        vsumi1 = vec_కూడు(vsumi1, vsumi5);
        vsumi2 = vec_కూడు(vsumi2, vsumi6);
        vsumi3 = vec_కూడు(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_వెలికితీయి(vsumf0, 0);

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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0x3);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_Mask1 = vec_splats((int8_t)0xf);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_Mask2 = vec_splats((int8_t)0x30);
    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం v1 = vec_splats((గుర్తుగల అక్షరం)0x1);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v2 = vec_splats((గుర్తులేని అక్షరం)0x2);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v3 = vec_splats((గుర్తులేని అక్షరం)0x3);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v6 = vec_splats((గుర్తులేని అక్షరం)0x6);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం off = vec_splats((గుర్తుగల అక్షరం)0x20);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        వాడని(kmask1);
        వాడని(kmask2);

        సదిశ గుర్తుగల అక్షరం u0 = (సదిశ గుర్తుగల అక్షరం)vec_xl_len(x[i].scales, 8);
        సదిశ గుర్తుగల అక్షరం u1 = vec_and(u0, దిగువ_Mask1);
        సదిశ గుర్తుగల అక్షరం u2 = (సదిశ గుర్తుగల అక్షరం)vec_xl_len(x[i].scales + 8, 4);
        సదిశ గుర్తుగల అక్షరం u3 = (సదిశ గుర్తుగల అక్షరం)vec_mergeh((సదిశ గుర్తుగల పూర్ణాంకం)u2, (సదిశ గుర్తుగల పూర్ణాంకం)vec_sr(u2, v2));
        సదిశ గుర్తుగల అక్షరం u30 = vec_sl(vec_and(u3, దిగువ_ముసుగు), v4);
        సదిశ గుర్తుగల అక్షరం u31 = vec_and(u3, దిగువ_Mask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        సదిశ గుర్తుగల అక్షరం vscales = (సదిశ గుర్తుగల అక్షరం)vec_mergeh((సదిశ గుర్తుగల దీర్ఘ దీర్ఘ)u1, (సదిశ గుర్తుగల దీర్ఘ దీర్ఘ)u2);
        సదిశ గుర్తుగల అక్షరం qxhs0 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, x[i].hmask);
        సదిశ గుర్తుగల అక్షరం qxhs1 = (సదిశ గుర్తుగల అక్షరం)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi4 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi5 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi6 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi7 = v0;

        స్థిరరాశి uint8_t * GGML_RESTRICT q3 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తుగల అక్షరం qxs0 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, q3);
            సదిశ గుర్తుగల అక్షరం qxs1 = (సదిశ గుర్తుగల అక్షరం)vec_xl(16, q3);
            q3 += 32;

            //the low 2 bits
            సదిశ గుర్తుగల అక్షరం qxs00 = vec_and(qxs0, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs01 = vec_and(vec_sr(qxs0, v2), దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs02 = vec_and(vec_sr(qxs0, v4), దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs03 = vec_and(vec_sr(qxs0, v6), దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs10 = vec_and(qxs1, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs11 = vec_and(vec_sr(qxs1, v2), దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs12 = vec_and(vec_sr(qxs1, v4), దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs13 = vec_and(vec_sr(qxs1, v6), దిగువ_ముసుగు);

            //the 3rd bit
            సదిశ గుర్తుగల అక్షరం qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            సదిశ గుర్తుగల అక్షరం qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (సదిశ గుర్తులేని అక్షరం)v1)), v2);
            సదిశ గుర్తుగల అక్షరం qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            సదిశ గుర్తుగల అక్షరం qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            సదిశ గుర్తుగల అక్షరం qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            సదిశ గుర్తుగల అక్షరం qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (సదిశ గుర్తులేని అక్షరం)v1)), v2);
            సదిశ గుర్తుగల అక్షరం qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            సదిశ గుర్తుగల అక్షరం qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            సదిశ గుర్తుగల అక్షరం q3x00 = vec_sub(qxs00, qxh00);
            సదిశ గుర్తుగల అక్షరం q3x01 = vec_sub(qxs01, qxh01);
            సదిశ గుర్తుగల అక్షరం q3x02 = vec_sub(qxs02, qxh02);
            సదిశ గుర్తుగల అక్షరం q3x03 = vec_sub(qxs03, qxh03);
            సదిశ గుర్తుగల అక్షరం q3x10 = vec_sub(qxs10, qxh10);
            సదిశ గుర్తుగల అక్షరం q3x11 = vec_sub(qxs11, qxh11);
            సదిశ గుర్తుగల అక్షరం q3x12 = vec_sub(qxs12, qxh12);
            సదిశ గుర్తుగల అక్షరం q3x13 = vec_sub(qxs13, qxh13);

            సదిశ గుర్తుగల అక్షరం q8y00 = vec_xl(  0, q8);
            సదిశ గుర్తుగల అక్షరం q8y10 = vec_xl( 16, q8);
            సదిశ గుర్తుగల అక్షరం q8y01 = vec_xl( 32, q8);
            సదిశ గుర్తుగల అక్షరం q8y11 = vec_xl( 48, q8);
            సదిశ గుర్తుగల అక్షరం q8y02 = vec_xl( 64, q8);
            సదిశ గుర్తుగల అక్షరం q8y12 = vec_xl( 80, q8);
            సదిశ గుర్తుగల అక్షరం q8y03 = vec_xl( 96, q8);
            సదిశ గుర్తుగల అక్షరం q8y13 = vec_xl(112, q8);
            q8 += 128;

            సదిశ గుర్తుగల పొట్టి vscales_h = vec_unpackh(vscales);
            సదిశ గుర్తుగల పొట్టి vs0 = vec_splat(vscales_h, 0);
            సదిశ గుర్తుగల పొట్టి vs1 = vec_splat(vscales_h, 1);
            సదిశ గుర్తుగల పొట్టి vs2 = vec_splat(vscales_h, 2);
            సదిశ గుర్తుగల పొట్టి vs3 = vec_splat(vscales_h, 3);
            సదిశ గుర్తుగల పొట్టి vs4 = vec_splat(vscales_h, 4);
            సదిశ గుర్తుగల పొట్టి vs5 = vec_splat(vscales_h, 5);
            సదిశ గుర్తుగల పొట్టి vs6 = vec_splat(vscales_h, 6);
            సదిశ గుర్తుగల పొట్టి vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            సదిశ గుర్తుగల పొట్టి qv00 = vec_కూడు(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            సదిశ గుర్తుగల పొట్టి qv01 = vec_కూడు(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            సదిశ గుర్తుగల పొట్టి qv02 = vec_కూడు(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            సదిశ గుర్తుగల పొట్టి qv03 = vec_కూడు(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            సదిశ గుర్తుగల పొట్టి qv10 = vec_కూడు(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            సదిశ గుర్తుగల పొట్టి qv11 = vec_కూడు(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            సదిశ గుర్తుగల పొట్టి qv12 = vec_కూడు(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            సదిశ గుర్తుగల పొట్టి qv13 = vec_కూడు(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs2, vsumi1);
            vsumi2 = vec_msum(qv02, vs4, vsumi2);
            vsumi3 = vec_msum(qv03, vs6, vsumi3);
            vsumi4 = vec_msum(qv10, vs1, vsumi4);
            vsumi5 = vec_msum(qv11, vs3, vsumi5);
            vsumi6 = vec_msum(qv12, vs5, vsumi6);
            vsumi7 = vec_msum(qv13, vs7, vsumi7);
        }

        vsumi0 = vec_కూడు(vsumi0, vsumi4);
        vsumi1 = vec_కూడు(vsumi1, vsumi5);
        vsumi2 = vec_కూడు(vsumi2, vsumi6);
        vsumi3 = vec_కూడు(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_వెలికితీయి(vsumf0, 0);

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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_Mask1 = vec_splats((int8_t)0x3f);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_Mask2 = vec_splats((int8_t)0x30);
    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v2 = vec_splats((uint8_t)2);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ దశాంశం vxmin = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin));
        సదిశ దశాంశం vdmin = vec_mul(vxmin, vyd);

        సదిశ గుర్తుగల పొట్టి q8ysums0 = vec_xl( 0, y[i].bsums);
        సదిశ గుర్తుగల పొట్టి q8ysums1 = vec_xl(16, y[i].bsums);

        వాడని(kmask1);
        వాడని(kmask2);
        వాడని(kmask3);
        వాడని(utmp);

        సదిశ గుర్తుగల అక్షరం u0 = (సదిశ గుర్తుగల అక్షరం)vec_xl_len(x[i].scales, 8);
        సదిశ గుర్తుగల అక్షరం u1 = vec_and(vec_sr(u0, v2), దిగువ_Mask2);
        సదిశ గుర్తుగల అక్షరం u2 = (సదిశ గుర్తుగల అక్షరం)vec_xl_len(x[i].scales + 8, 4);
        సదిశ గుర్తుగల అక్షరం u3 = vec_sr(u2, v4);

        సదిశ గుర్తుగల అక్షరం u30 = u1;
        సదిశ గుర్తుగల అక్షరం u31 = (సదిశ గుర్తుగల అక్షరం)vec_mergeh((సదిశ గుర్తుగల పూర్ణాంకం)vec_and(u2, దిగువ_ముసుగు), (సదిశ గుర్తుగల పూర్ణాంకం)u3);

        u1 = vec_and(u0, దిగువ_Mask1);
        u2 = vec_or(u30, u31);

        సదిశ గుర్తుగల అక్షరం utmps = (సదిశ గుర్తుగల అక్షరం)vec_mergeh((సదిశ గుర్తుగల పూర్ణాంకం)u1, (సదిశ గుర్తుగల పూర్ణాంకం)u2);

        సదిశ గుర్తుగల పొట్టి vscales = vec_unpackh(utmps);
        సదిశ గుర్తుగల పొట్టి q4xmins = vec_unpackl(utmps);
        సదిశ గుర్తుగల పొట్టి q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        సదిశ గుర్తుగల పొట్టి q4xmins1 = vec_mergel(q4xmins, q4xmins);

        సదిశ గుర్తుగల పూర్ణాంకం prod0 = vec_mule(q4xmins0, q8ysums0);
        సదిశ గుర్తుగల పూర్ణాంకం prod1 = vec_mule(q4xmins1, q8ysums1);
        సదిశ గుర్తుగల పూర్ణాంకం prod2 = vec_mulo(q4xmins0, q8ysums0);
        సదిశ గుర్తుగల పూర్ణాంకం prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;

        స్థిరరాశి uint8_t * GGML_RESTRICT q4 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తుగల అక్షరం qxs0 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, q4);
            సదిశ గుర్తుగల అక్షరం qxs1 = (సదిశ గుర్తుగల అక్షరం)vec_xl(16, q4);
            సదిశ గుర్తుగల అక్షరం qxs2 = (సదిశ గుర్తుగల అక్షరం)vec_xl(32, q4);
            సదిశ గుర్తుగల అక్షరం qxs3 = (సదిశ గుర్తుగల అక్షరం)vec_xl(48, q4);
            q4 += 64;

            సదిశ గుర్తులేని అక్షరం q4x00 = (సదిశ గుర్తులేని అక్షరం)vec_and(qxs0, దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q4x01 = (సదిశ గుర్తులేని అక్షరం)vec_sr(qxs0, v4);
            సదిశ గుర్తులేని అక్షరం q4x10 = (సదిశ గుర్తులేని అక్షరం)vec_and(qxs1, దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q4x11 = (సదిశ గుర్తులేని అక్షరం)vec_sr(qxs1, v4);
            సదిశ గుర్తులేని అక్షరం q4x20 = (సదిశ గుర్తులేని అక్షరం)vec_and(qxs2, దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q4x21 = (సదిశ గుర్తులేని అక్షరం)vec_sr(qxs2, v4);
            సదిశ గుర్తులేని అక్షరం q4x30 = (సదిశ గుర్తులేని అక్షరం)vec_and(qxs3, దిగువ_ముసుగు);
            సదిశ గుర్తులేని అక్షరం q4x31 = (సదిశ గుర్తులేని అక్షరం)vec_sr(qxs3, v4);

            సదిశ గుర్తుగల అక్షరం q8y00 = vec_xl(  0, q8);
            సదిశ గుర్తుగల అక్షరం q8y10 = vec_xl( 16, q8);
            సదిశ గుర్తుగల అక్షరం q8y01 = vec_xl( 32, q8);
            సదిశ గుర్తుగల అక్షరం q8y11 = vec_xl( 48, q8);
            సదిశ గుర్తుగల అక్షరం q8y20 = vec_xl( 64, q8);
            సదిశ గుర్తుగల అక్షరం q8y30 = vec_xl( 80, q8);
            సదిశ గుర్తుగల అక్షరం q8y21 = vec_xl( 96, q8);
            సదిశ గుర్తుగల అక్షరం q8y31 = vec_xl(112, q8);
            q8 += 128;

            సదిశ గుర్తుగల పూర్ణాంకం qv00 = vec_msum(q8y00, q4x00, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv01 = vec_msum(q8y01, q4x01, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv10 = vec_msum(q8y10, q4x10, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv11 = vec_msum(q8y11, q4x11, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv20 = vec_msum(q8y20, q4x20, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv21 = vec_msum(q8y21, q4x21, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv30 = vec_msum(q8y30, q4x30, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv31 = vec_msum(q8y31, q4x31, v0);

            సదిశ గుర్తుగల పూర్ణాంకం vscales_h = vec_unpackh(vscales);
            సదిశ గుర్తుగల పూర్ణాంకం vs0 = vec_splat(vscales_h, 0);
            సదిశ గుర్తుగల పూర్ణాంకం vs1 = vec_splat(vscales_h, 1);
            సదిశ గుర్తుగల పూర్ణాంకం vs2 = vec_splat(vscales_h, 2);
            సదిశ గుర్తుగల పూర్ణాంకం vs3 = vec_splat(vscales_h, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_కూడు(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_కూడు(vec_mul(qv01, vs1), vsumi1);
            vsumi2 = vec_కూడు(vec_mul(qv20, vs2), vsumi2);
            vsumi3 = vec_కూడు(vec_mul(qv21, vs3), vsumi3);

            vsumi0 = vec_కూడు(vec_mul(qv10, vs0), vsumi0);
            vsumi1 = vec_కూడు(vec_mul(qv11, vs1), vsumi1);
            vsumi2 = vec_కూడు(vec_mul(qv30, vs2), vsumi2);
            vsumi3 = vec_కూడు(vec_mul(qv31, vs3), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_వెలికితీయి(vsumf0, 0);

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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_Mask1 = vec_splats((int8_t)0x3f);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_Mask2 = vec_splats((int8_t)0x30);
    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v1 = vec_splats((గుర్తులేని అక్షరం)0x1);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v2 = vec_splats((గుర్తులేని అక్షరం)0x2);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v3 = vec_splats((గుర్తులేని అక్షరం)0x3);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ దశాంశం vxmin = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].dmin));
        సదిశ దశాంశం vdmin = vec_mul(vxmin, vyd);

        వాడని(kmask1);
        వాడని(kmask2);
        వాడని(kmask3);
        వాడని(utmp);

        సదిశ గుర్తుగల అక్షరం u0 = (సదిశ గుర్తుగల అక్షరం)vec_xl_len(x[i].scales, 8);
        సదిశ గుర్తుగల అక్షరం u1 = vec_and(vec_sr(u0, v2), దిగువ_Mask2);
        సదిశ గుర్తుగల అక్షరం u2 = (సదిశ గుర్తుగల అక్షరం)vec_xl_len(x[i].scales + 8, 4);
        సదిశ గుర్తుగల అక్షరం u3 = vec_sr(u2, v4);

        సదిశ గుర్తుగల అక్షరం u30 = u1;
        సదిశ గుర్తుగల అక్షరం u31 = (సదిశ గుర్తుగల అక్షరం)vec_mergeh((సదిశ గుర్తుగల పూర్ణాంకం)vec_and(u2, దిగువ_ముసుగు), (సదిశ గుర్తుగల పూర్ణాంకం)u3);

        u1 = vec_and(u0, దిగువ_Mask1);
        u2 = vec_or(u30, u31);

        సదిశ గుర్తుగల అక్షరం utmps = (సదిశ గుర్తుగల అక్షరం)vec_mergeh((సదిశ గుర్తుగల పూర్ణాంకం)u1, (సదిశ గుర్తుగల పూర్ణాంకం)u2);

        సదిశ గుర్తుగల పొట్టి q8ysums0 = vec_xl( 0, y[i].bsums);
        సదిశ గుర్తుగల పొట్టి q8ysums1 = vec_xl(16, y[i].bsums);

        సదిశ గుర్తుగల పొట్టి vscales = vec_unpackh(utmps);

        సదిశ గుర్తుగల పొట్టి q5xmins = vec_unpackl(utmps);
        సదిశ గుర్తుగల పొట్టి q5xmins0 = vec_mergeh(q5xmins, q5xmins);
        సదిశ గుర్తుగల పొట్టి q5xmins1 = vec_mergel(q5xmins, q5xmins);

        సదిశ గుర్తుగల పూర్ణాంకం prod0 = vec_mule(q5xmins0, q8ysums0);
        సదిశ గుర్తుగల పూర్ణాంకం prod1 = vec_mule(q5xmins1, q8ysums1);
        సదిశ గుర్తుగల పూర్ణాంకం prod2 = vec_mulo(q5xmins0, q8ysums0);
        సదిశ గుర్తుగల పూర్ణాంకం prod3 = vec_mulo(q5xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        సదిశ గుర్తుగల అక్షరం qxhs0 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, x[i].qh);
        సదిశ గుర్తుగల అక్షరం qxhs1 = (సదిశ గుర్తుగల అక్షరం)vec_xl(16, x[i].qh);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;

        స్థిరరాశి uint8_t * GGML_RESTRICT q5 = x[i].qs;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q5, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తుగల అక్షరం qxs0 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, q5);
            సదిశ గుర్తుగల అక్షరం qxs1 = (సదిశ గుర్తుగల అక్షరం)vec_xl(16, q5);
            q5 += 32;

            సదిశ గుర్తుగల అక్షరం qxs00 = vec_and(qxs0, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs01 = vec_sr(qxs0, v4);
            సదిశ గుర్తుగల అక్షరం qxs10 = vec_and(qxs1, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs11 = vec_sr(qxs1, v4);

            సదిశ గుర్తుగల అక్షరం q5h00 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v1, qxhs0), v4);
            సదిశ గుర్తుగల అక్షరం q5h01 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v2, qxhs0), v3);
            సదిశ గుర్తుగల అక్షరం q5h10 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v1, qxhs1), v4);
            సదిశ గుర్తుగల అక్షరం q5h11 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v2, qxhs1), v3);
            qxhs0 = vec_sr(qxhs0, v2);
            qxhs1 = vec_sr(qxhs1, v2);

            సదిశ గుర్తులేని అక్షరం q5x00 = (సదిశ గుర్తులేని అక్షరం)vec_or(q5h00, qxs00);
            సదిశ గుర్తులేని అక్షరం q5x01 = (సదిశ గుర్తులేని అక్షరం)vec_or(q5h01, qxs01);
            సదిశ గుర్తులేని అక్షరం q5x10 = (సదిశ గుర్తులేని అక్షరం)vec_or(q5h10, qxs10);
            సదిశ గుర్తులేని అక్షరం q5x11 = (సదిశ గుర్తులేని అక్షరం)vec_or(q5h11, qxs11);

            సదిశ గుర్తుగల అక్షరం q8y00 = vec_xl( 0, q8);
            సదిశ గుర్తుగల అక్షరం q8y10 = vec_xl(16, q8);
            సదిశ గుర్తుగల అక్షరం q8y01 = vec_xl(32, q8);
            సదిశ గుర్తుగల అక్షరం q8y11 = vec_xl(48, q8);
            q8 += 64;

            సదిశ గుర్తుగల పూర్ణాంకం qv00 = vec_msum(q8y00, q5x00, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv01 = vec_msum(q8y01, q5x01, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv10 = vec_msum(q8y10, q5x10, v0);
            సదిశ గుర్తుగల పూర్ణాంకం qv11 = vec_msum(q8y11, q5x11, v0);

            సదిశ గుర్తుగల పూర్ణాంకం vscales_h = vec_unpackh(vscales);
            సదిశ గుర్తుగల పూర్ణాంకం vs0 = vec_splat(vscales_h, 0);
            సదిశ గుర్తుగల పూర్ణాంకం vs1 = vec_splat(vscales_h, 1);
            vscales = vec_sld(vscales, vscales, 12);

            vsumi0 = vec_కూడు(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_కూడు(vec_mul(qv10, vs0), vsumi1);
            vsumi2 = vec_కూడు(vec_mul(qv01, vs1), vsumi2);
            vsumi3 = vec_కూడు(vec_mul(qv11, vs1), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_వెలికితీయి(vsumf0, 0);

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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v2 = vec_splats((గుర్తులేని అక్షరం)0x2);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v3 = vec_splats((గుర్తులేని అక్షరం)0x3);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v6 = vec_splats((గుర్తులేని అక్షరం)0x6);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం off = vec_splats((గుర్తుగల అక్షరం)0x20);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi4 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi5 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi6 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi7 = v0;

        స్థిరరాశి uint8_t * GGML_RESTRICT q6 = x[i].ql;
        స్థిరరాశి uint8_t * GGML_RESTRICT qh = x[i].qh;
        స్థిరరాశి int8_t  * GGML_RESTRICT qs = x[i].scales;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q6, 0, 0);
            __builtin_prefetch(qh, 0, 0);
            __builtin_prefetch(q8, 0, 0);

            సదిశ గుర్తుగల అక్షరం qxs0 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, q6);
            సదిశ గుర్తుగల అక్షరం qxs1 = (సదిశ గుర్తుగల అక్షరం)vec_xl(16, q6);
            సదిశ గుర్తుగల అక్షరం qxs2 = (సదిశ గుర్తుగల అక్షరం)vec_xl(32, q6);
            సదిశ గుర్తుగల అక్షరం qxs3 = (సదిశ గుర్తుగల అక్షరం)vec_xl(48, q6);
            q6 += 64;

            సదిశ గుర్తుగల అక్షరం qxs00 = vec_and(qxs0, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs01 = vec_sr(qxs0, v4);
            సదిశ గుర్తుగల అక్షరం qxs10 = vec_and(qxs1, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs11 = vec_sr(qxs1, v4);
            సదిశ గుర్తుగల అక్షరం qxs20 = vec_and(qxs2, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs21 = vec_sr(qxs2, v4);
            సదిశ గుర్తుగల అక్షరం qxs30 = vec_and(qxs3, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం qxs31 = vec_sr(qxs3, v4);

            సదిశ గుర్తుగల అక్షరం qxhs0 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, qh);
            సదిశ గుర్తుగల అక్షరం qxhs1 = (సదిశ గుర్తుగల అక్షరం)vec_xl(16, qh);
            qh += 32;

            సదిశ గుర్తుగల అక్షరం qxh00 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v3, qxhs0), v4);
            సదిశ గుర్తుగల అక్షరం qxh01 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v3, vec_sr(qxhs0, v4)), v4);
            సదిశ గుర్తుగల అక్షరం qxh10 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v3, qxhs1), v4);
            సదిశ గుర్తుగల అక్షరం qxh11 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v3, vec_sr(qxhs1, v4)), v4);
            సదిశ గుర్తుగల అక్షరం qxh20 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v3, vec_sr(qxhs0, v2)), v4);
            సదిశ గుర్తుగల అక్షరం qxh21 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v3, vec_sr(qxhs0, v6)), v4);
            సదిశ గుర్తుగల అక్షరం qxh30 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v3, vec_sr(qxhs1, v2)), v4);
            సదిశ గుర్తుగల అక్షరం qxh31 = vec_sl(vec_and((సదిశ గుర్తుగల అక్షరం)v3, vec_sr(qxhs1, v6)), v4);

            సదిశ గుర్తుగల అక్షరం q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
            సదిశ గుర్తుగల అక్షరం q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
            సదిశ గుర్తుగల అక్షరం q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
            సదిశ గుర్తుగల అక్షరం q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
            సదిశ గుర్తుగల అక్షరం q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
            సదిశ గుర్తుగల అక్షరం q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
            సదిశ గుర్తుగల అక్షరం q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
            సదిశ గుర్తుగల అక్షరం q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

            సదిశ గుర్తుగల అక్షరం q8y00 = vec_xl(  0, q8);
            సదిశ గుర్తుగల అక్షరం q8y10 = vec_xl( 16, q8);
            సదిశ గుర్తుగల అక్షరం q8y20 = vec_xl( 32, q8);
            సదిశ గుర్తుగల అక్షరం q8y30 = vec_xl( 48, q8);
            సదిశ గుర్తుగల అక్షరం q8y01 = vec_xl( 64, q8);
            సదిశ గుర్తుగల అక్షరం q8y11 = vec_xl( 80, q8);
            సదిశ గుర్తుగల అక్షరం q8y21 = vec_xl( 96, q8);
            సదిశ గుర్తుగల అక్షరం q8y31 = vec_xl(112, q8);
            q8 += 128;

            సదిశ గుర్తుగల పొట్టి qv00 = vec_కూడు(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
            సదిశ గుర్తుగల పొట్టి qv10 = vec_కూడు(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
            సదిశ గుర్తుగల పొట్టి qv20 = vec_కూడు(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
            సదిశ గుర్తుగల పొట్టి qv30 = vec_కూడు(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
            సదిశ గుర్తుగల పొట్టి qv01 = vec_కూడు(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
            సదిశ గుర్తుగల పొట్టి qv11 = vec_కూడు(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
            సదిశ గుర్తుగల పొట్టి qv21 = vec_కూడు(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
            సదిశ గుర్తుగల పొట్టి qv31 = vec_కూడు(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

            సదిశ గుర్తుగల పొట్టి vscales = vec_unpackh(vec_xl_len(qs, 8));
            qs += 8;

            సదిశ గుర్తుగల పొట్టి vs0 = vec_splat(vscales, 0);
            సదిశ గుర్తుగల పొట్టి vs1 = vec_splat(vscales, 1);
            సదిశ గుర్తుగల పొట్టి vs2 = vec_splat(vscales, 2);
            సదిశ గుర్తుగల పొట్టి vs3 = vec_splat(vscales, 3);
            సదిశ గుర్తుగల పొట్టి vs4 = vec_splat(vscales, 4);
            సదిశ గుర్తుగల పొట్టి vs5 = vec_splat(vscales, 5);
            సదిశ గుర్తుగల పొట్టి vs6 = vec_splat(vscales, 6);
            సదిశ గుర్తుగల పొట్టి vs7 = vec_splat(vscales, 7);

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs4, vsumi1);
            vsumi2 = vec_msum(qv10, vs1, vsumi2);
            vsumi3 = vec_msum(qv11, vs5, vsumi3);
            vsumi4 = vec_msum(qv20, vs2, vsumi4);
            vsumi5 = vec_msum(qv21, vs6, vsumi5);
            vsumi6 = vec_msum(qv30, vs3, vsumi6);
            vsumi7 = vec_msum(qv31, vs7, vsumi7);
        }

        vsumi0 = vec_కూడు(vsumi0, vsumi4);
        vsumi1 = vec_కూడు(vsumi1, vsumi5);
        vsumi2 = vec_కూడు(vsumi2, vsumi6);
        vsumi3 = vec_కూడు(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_వెలికితీయి(vsumf0, 0);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

#ఒకవేళ defined (__POWER9_VECTOR__)
స్థిర స్థిరరాశి int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
#అయితే_ముగింపు

శూన్య ggml_vec_dot_iq2_xxs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq2_xxs * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K    * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);
    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    స్థిరరాశి uint64_t * signs64 = (స్థిరరాశి uint64_t *)keven_signs_q2xs;

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;

        స్థిరరాశి uint16_t * GGML_RESTRICT q2 = x[i].qs;
        స్థిరరాశి int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            స్థిరరాశి uint8_t * aux8 = (స్థిరరాశి uint8_t *)aux32;

            జ్ఞాపకం_నకలు(aux32, q2, 4*పరిమాణం(uint32_t));
            q2 += 8;

            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_0 = {*(స్థిరరాశి int64_t *)(iq2xxs_grid + aux8[ 0]), *(స్థిరరాశి int64_t *)(iq2xxs_grid + aux8[ 1])};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_1 = {*(స్థిరరాశి int64_t *)(iq2xxs_grid + aux8[ 2]), *(స్థిరరాశి int64_t *)(iq2xxs_grid + aux8[ 3])};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_2 = {*(స్థిరరాశి int64_t *)(iq2xxs_grid + aux8[ 8]), *(స్థిరరాశి int64_t *)(iq2xxs_grid + aux8[ 9])};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_3 = {*(స్థిరరాశి int64_t *)(iq2xxs_grid + aux8[10]), *(స్థిరరాశి int64_t *)(iq2xxs_grid + aux8[11])};

            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ vsigns0 = {*(స్థిరరాశి int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(స్థిరరాశి int64_t *)(signs64 + ((aux32[1] >>  7) & 127))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ vsigns1 = {*(స్థిరరాశి int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(స్థిరరాశి int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ vsigns2 = {*(స్థిరరాశి int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(స్థిరరాశి int64_t *)(signs64 + ((aux32[3] >>  7) & 127))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ vsigns3 = {*(స్థిరరాశి int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(స్థిరరాశి int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            సదిశ గుర్తుగల అక్షరం q2x0 = (సదిశ గుర్తుగల అక్షరం)vec_mul((సదిశ గుర్తుగల అక్షరం)vsigns0, (సదిశ గుర్తుగల అక్షరం)aux64x2_0);
            సదిశ గుర్తుగల అక్షరం q2x1 = (సదిశ గుర్తుగల అక్షరం)vec_mul((సదిశ గుర్తుగల అక్షరం)vsigns1, (సదిశ గుర్తుగల అక్షరం)aux64x2_1);
            సదిశ గుర్తుగల అక్షరం q2x2 = (సదిశ గుర్తుగల అక్షరం)vec_mul((సదిశ గుర్తుగల అక్షరం)vsigns2, (సదిశ గుర్తుగల అక్షరం)aux64x2_2);
            సదిశ గుర్తుగల అక్షరం q2x3 = (సదిశ గుర్తుగల అక్షరం)vec_mul((సదిశ గుర్తుగల అక్షరం)vsigns3, (సదిశ గుర్తుగల అక్షరం)aux64x2_3);

            సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, q8);
            సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, q8);
            సదిశ గుర్తుగల అక్షరం q8y2 = vec_xl(32, q8);
            సదిశ గుర్తుగల అక్షరం q8y3 = vec_xl(48, q8);
            q8 += 64;

            సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            సదిశ గుర్తుగల పొట్టి qv2 = vec_కూడు(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            సదిశ గుర్తుగల పొట్టి qv3 = vec_కూడు(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            స్థిరరాశి uint16_t ls0 = aux32[1] >> 28;
            స్థిరరాశి uint16_t ls1 = aux32[3] >> 28;

            సదిశ గుర్తుగల పొట్టి vscales01 = vec_splats((int16_t)(2*ls0+1));
            సదిశ గుర్తుగల పొట్టి vscales23 = vec_splats((int16_t)(2*ls1+1));

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_వెలికితీయి(vsumf0, 0);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq2_xs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq2_xs * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K   * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);
    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    స్థిరరాశి uint64_t * signs64 = (స్థిరరాశి uint64_t *)keven_signs_q2xs;

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;

        స్థిరరాశి uint16_t * GGML_RESTRICT q2 = x[i].qs;
        స్థిరరాశి uint8_t  * GGML_RESTRICT sc = x[i].scales;
        స్థిరరాశి int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_0 = {*(స్థిరరాశి int64_t *)(iq2xs_grid + (q2[0] & 511)), *(స్థిరరాశి int64_t *)(iq2xs_grid + (q2[1] & 511))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_1 = {*(స్థిరరాశి int64_t *)(iq2xs_grid + (q2[2] & 511)), *(స్థిరరాశి int64_t *)(iq2xs_grid + (q2[3] & 511))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_2 = {*(స్థిరరాశి int64_t *)(iq2xs_grid + (q2[4] & 511)), *(స్థిరరాశి int64_t *)(iq2xs_grid + (q2[5] & 511))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_3 = {*(స్థిరరాశి int64_t *)(iq2xs_grid + (q2[6] & 511)), *(స్థిరరాశి int64_t *)(iq2xs_grid + (q2[7] & 511))};

            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ vsigns0 = {*(స్థిరరాశి int64_t *)(signs64 + ((q2[0] >> 9))), *(స్థిరరాశి int64_t *)(signs64 + ((q2[1] >> 9)))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ vsigns1 = {*(స్థిరరాశి int64_t *)(signs64 + ((q2[2] >> 9))), *(స్థిరరాశి int64_t *)(signs64 + ((q2[3] >> 9)))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ vsigns2 = {*(స్థిరరాశి int64_t *)(signs64 + ((q2[4] >> 9))), *(స్థిరరాశి int64_t *)(signs64 + ((q2[5] >> 9)))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ vsigns3 = {*(స్థిరరాశి int64_t *)(signs64 + ((q2[6] >> 9))), *(స్థిరరాశి int64_t *)(signs64 + ((q2[7] >> 9)))};
            q2 += 8;

            సదిశ గుర్తుగల అక్షరం q2x0 = (సదిశ గుర్తుగల అక్షరం)vec_mul((సదిశ గుర్తుగల అక్షరం)vsigns0, (సదిశ గుర్తుగల అక్షరం)aux64x2_0);
            సదిశ గుర్తుగల అక్షరం q2x1 = (సదిశ గుర్తుగల అక్షరం)vec_mul((సదిశ గుర్తుగల అక్షరం)vsigns1, (సదిశ గుర్తుగల అక్షరం)aux64x2_1);
            సదిశ గుర్తుగల అక్షరం q2x2 = (సదిశ గుర్తుగల అక్షరం)vec_mul((సదిశ గుర్తుగల అక్షరం)vsigns2, (సదిశ గుర్తుగల అక్షరం)aux64x2_2);
            సదిశ గుర్తుగల అక్షరం q2x3 = (సదిశ గుర్తుగల అక్షరం)vec_mul((సదిశ గుర్తుగల అక్షరం)vsigns3, (సదిశ గుర్తుగల అక్షరం)aux64x2_3);

            సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, q8);
            సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, q8);
            సదిశ గుర్తుగల అక్షరం q8y2 = vec_xl(32, q8);
            సదిశ గుర్తుగల అక్షరం q8y3 = vec_xl(48, q8);
            q8 += 64;

            సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            సదిశ గుర్తుగల పొట్టి qv2 = vec_కూడు(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            సదిశ గుర్తుగల పొట్టి qv3 = vec_కూడు(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            స్థిరరాశి uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            స్థిరరాశి uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            స్థిరరాశి uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            స్థిరరాశి uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            సదిశ గుర్తుగల పొట్టి vscales0 = vec_splats((int16_t)(2*ls0+1));
            సదిశ గుర్తుగల పొట్టి vscales1 = vec_splats((int16_t)(2*ls1+1));
            సదిశ గుర్తుగల పొట్టి vscales2 = vec_splats((int16_t)(2*ls2+1));
            సదిశ గుర్తుగల పొట్టి vscales3 = vec_splats((int16_t)(2*ls3+1));

            vsumi0 = vec_msum(qv0, vscales0, vsumi0);
            vsumi1 = vec_msum(qv1, vscales1, vsumi1);
            vsumi2 = vec_msum(qv2, vscales2, vsumi2);
            vsumi3 = vec_msum(qv3, vscales3, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_వెలికితీయి(vsumf0, 0);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq2_s_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq2_s * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K  * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిర స్థిరరాశి uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    స్థిర స్థిరరాశి uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    స్థిరరాశి సదిశ గుర్తులేని అక్షరం mask0 = vec_xl( 0, k_mask1);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం mask1 = vec_xl(16, k_mask1);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం mask2 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, k_mask2);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;

        స్థిరరాశి uint8_t *  GGML_RESTRICT q2 = x[i].qs;
        స్థిరరాశి uint8_t *  GGML_RESTRICT qh = x[i].qh;
        స్థిరరాశి uint16_t * GGML_RESTRICT signs = (స్థిరరాశి uint16_t *)(x[i].qs + QK_K/8);
        స్థిరరాశి uint8_t *  GGML_RESTRICT sc = x[i].scales;
        స్థిరరాశి int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_0 = {*(స్థిరరాశి int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(స్థిరరాశి int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_1 = {*(స్థిరరాశి int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(స్థిరరాశి int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_2 = {*(స్థిరరాశి int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(స్థిరరాశి int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_3 = {*(స్థిరరాశి int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(స్థిరరాశి int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            సదిశ గుర్తుగల అక్షరం vsigns01 = (సదిశ గుర్తుగల అక్షరం)vec_splats(*(స్థిరరాశి uint32_t *)&signs[0]);
            సదిశ గుర్తుగల అక్షరం vsigns23 = (సదిశ గుర్తుగల అక్షరం)vec_splats(*(స్థిరరాశి uint32_t *)&signs[2]);
            signs += 4;

            సదిశ గుర్తుగల అక్షరం vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            సదిశ గుర్తుగల అక్షరం vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            సదిశ గుర్తుగల అక్షరం vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            సదిశ గుర్తుగల అక్షరం vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (సదిశ గుర్తుగల అక్షరం)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (సదిశ గుర్తుగల అక్షరం)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (సదిశ గుర్తుగల అక్షరం)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (సదిశ గుర్తుగల అక్షరం)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            సదిశ గుర్తుగల అక్షరం q2x0 = vec_sub(vec_xor(vsigns0, (సదిశ గుర్తుగల అక్షరం)aux64x2_0), vsigns0);
            సదిశ గుర్తుగల అక్షరం q2x1 = vec_sub(vec_xor(vsigns1, (సదిశ గుర్తుగల అక్షరం)aux64x2_1), vsigns1);
            సదిశ గుర్తుగల అక్షరం q2x2 = vec_sub(vec_xor(vsigns2, (సదిశ గుర్తుగల అక్షరం)aux64x2_2), vsigns2);
            సదిశ గుర్తుగల అక్షరం q2x3 = vec_sub(vec_xor(vsigns3, (సదిశ గుర్తుగల అక్షరం)aux64x2_3), vsigns3);

            సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, q8);
            సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, q8);
            సదిశ గుర్తుగల అక్షరం q8y2 = vec_xl(32, q8);
            సదిశ గుర్తుగల అక్షరం q8y3 = vec_xl(48, q8);
            q8 += 64;

            సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            సదిశ గుర్తుగల పొట్టి qv2 = vec_కూడు(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            సదిశ గుర్తుగల పొట్టి qv3 = vec_కూడు(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            స్థిరరాశి uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            స్థిరరాశి uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            స్థిరరాశి uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            స్థిరరాశి uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            సదిశ గుర్తుగల పొట్టి vscales0 = vec_splats((int16_t)(2*ls0+1));
            సదిశ గుర్తుగల పొట్టి vscales1 = vec_splats((int16_t)(2*ls1+1));
            సదిశ గుర్తుగల పొట్టి vscales2 = vec_splats((int16_t)(2*ls2+1));
            సదిశ గుర్తుగల పొట్టి vscales3 = vec_splats((int16_t)(2*ls3+1));

            vsumi0 = vec_msum(qv0, vscales0, vsumi0);
            vsumi1 = vec_msum(qv1, vscales1, vsumi1);
            vsumi2 = vec_msum(qv2, vscales2, vsumi2);
            vsumi3 = vec_msum(qv3, vscales3, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_వెలికితీయి(vsumf0, 0);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq3_xxs_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq3_xxs * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K    * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి uint64_t * signs64 = (స్థిరరాశి uint64_t *)keven_signs_q2xs;

    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;

        స్థిరరాశి uint8_t * GGML_RESTRICT q3 = x[i].qs;
        స్థిరరాశి uint32_t * GGML_RESTRICT signs = (స్థిరరాశి uint32_t *)(x[i].qs + QK_K/4);
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[i].qs;

#సూచన GCC unroll 1
        కొరకు (పూర్ణాంకం j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తులేని పూర్ణాంకం aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            సదిశ గుర్తులేని పూర్ణాంకం aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            సదిశ గుర్తులేని పూర్ణాంకం aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            సదిశ గుర్తులేని పూర్ణాంకం aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            సదిశ గుర్తులేని దీర్ఘ దీర్ఘ aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            సదిశ గుర్తులేని దీర్ఘ దీర్ఘ aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            సదిశ గుర్తులేని దీర్ఘ దీర్ఘ aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            సదిశ గుర్తులేని దీర్ఘ దీర్ఘ aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            సదిశ గుర్తుగల అక్షరం q3x0 = vec_mul((సదిశ గుర్తుగల అక్షరం)aux64x2_0, (సదిశ గుర్తుగల అక్షరం)aux32x4_0);
            సదిశ గుర్తుగల అక్షరం q3x1 = vec_mul((సదిశ గుర్తుగల అక్షరం)aux64x2_1, (సదిశ గుర్తుగల అక్షరం)aux32x4_1);
            సదిశ గుర్తుగల అక్షరం q3x2 = vec_mul((సదిశ గుర్తుగల అక్షరం)aux64x2_2, (సదిశ గుర్తుగల అక్షరం)aux32x4_2);
            సదిశ గుర్తుగల అక్షరం q3x3 = vec_mul((సదిశ గుర్తుగల అక్షరం)aux64x2_3, (సదిశ గుర్తుగల అక్షరం)aux32x4_3);

            సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, q8);
            సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, q8);
            సదిశ గుర్తుగల అక్షరం q8y2 = vec_xl(32, q8);
            సదిశ గుర్తుగల అక్షరం q8y3 = vec_xl(48, q8);
            q8 += 64;

            సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            సదిశ గుర్తుగల పొట్టి qv2 = vec_కూడు(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            సదిశ గుర్తుగల పొట్టి qv3 = vec_కూడు(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            స్థిరరాశి uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            స్థిరరాశి uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            సదిశ గుర్తుగల పొట్టి vscales01 = (సదిశ గుర్తుగల పొట్టి)vec_splats((uint16_t)(2*ls0+1));
            సదిశ గుర్తుగల పొట్టి vscales23 = (సదిశ గుర్తుగల పొట్టి)vec_splats((uint16_t)(2*ls1+1));

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.25f * vec_వెలికితీయి(vsumf0, 0);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq3_s_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq3_s * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K  * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిర స్థిరరాశి uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    స్థిర స్థిరరాశి uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    స్థిరరాశి సదిశ గుర్తులేని అక్షరం mask0 = vec_xl( 0, k_mask1);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం mask1 = vec_xl(16, k_mask1);
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం mask2 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, k_mask2);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        స్థిరరాశి uint8_t *  GGML_RESTRICT q3 = x[i].qs;
        స్థిరరాశి uint8_t *  GGML_RESTRICT qh = x[i].qh;
        స్థిరరాశి uint16_t * GGML_RESTRICT signs = (స్థిరరాశి uint16_t *)(x[i].signs);
        స్థిరరాశి uint8_t *  GGML_RESTRICT sc = x[i].scales;
        స్థిరరాశి int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తులేని పూర్ణాంకం aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            సదిశ గుర్తులేని పూర్ణాంకం aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            సదిశ గుర్తులేని పూర్ణాంకం aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            సదిశ గుర్తులేని పూర్ణాంకం aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            సదిశ గుర్తుగల అక్షరం vsigns01 = (సదిశ గుర్తుగల అక్షరం)vec_splats(*(స్థిరరాశి uint32_t *)&signs[0]);
            సదిశ గుర్తుగల అక్షరం vsigns02 = (సదిశ గుర్తుగల అక్షరం)vec_splats(*(స్థిరరాశి uint32_t *)&signs[2]);
            signs += 4;

            సదిశ గుర్తుగల అక్షరం vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            సదిశ గుర్తుగల అక్షరం vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            సదిశ గుర్తుగల అక్షరం vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            సదిశ గుర్తుగల అక్షరం vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (సదిశ గుర్తుగల అక్షరం)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (సదిశ గుర్తుగల అక్షరం)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (సదిశ గుర్తుగల అక్షరం)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (సదిశ గుర్తుగల అక్షరం)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            సదిశ గుర్తుగల అక్షరం q3x0 = vec_sub(vec_xor(vsigns0, (సదిశ గుర్తుగల అక్షరం)aux32x4_0), vsigns0);
            సదిశ గుర్తుగల అక్షరం q3x1 = vec_sub(vec_xor(vsigns1, (సదిశ గుర్తుగల అక్షరం)aux32x4_1), vsigns1);
            సదిశ గుర్తుగల అక్షరం q3x2 = vec_sub(vec_xor(vsigns2, (సదిశ గుర్తుగల అక్షరం)aux32x4_2), vsigns2);
            సదిశ గుర్తుగల అక్షరం q3x3 = vec_sub(vec_xor(vsigns3, (సదిశ గుర్తుగల అక్షరం)aux32x4_3), vsigns3);

            సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, q8);
            సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, q8);
            సదిశ గుర్తుగల అక్షరం q8y2 = vec_xl(32, q8);
            సదిశ గుర్తుగల అక్షరం q8y3 = vec_xl(48, q8);
            q8 += 64;

            సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            సదిశ గుర్తుగల పొట్టి qv2 = vec_కూడు(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            సదిశ గుర్తుగల పొట్టి qv3 = vec_కూడు(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            స్థిరరాశి uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            స్థిరరాశి uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            సదిశ గుర్తుగల పొట్టి vscales01 = (సదిశ గుర్తుగల పొట్టి)vec_splats((uint16_t)(2*ls0+1));
            సదిశ గుర్తుగల పొట్టి vscales23 = (సదిశ గుర్తుగల పొట్టి)vec_splats((uint16_t)(2*ls1+1));

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_వెలికితీయి(vsumf0, 0);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

శూన్య ggml_vec_dot_iq1_s_q8_K(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, size_t bx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, size_t by, పూర్ణాంకం nrc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nrc == 1);
    వాడని(nrc);
    వాడని(bx);
    వాడని(by);
    వాడని(bs);

    స్థిరరాశి ఖండం_iq1_s * GGML_RESTRICT x = vx;
    స్థిరరాశి ఖండం_q8_K  * GGML_RESTRICT y = vy;

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v0 = vec_splats((గుర్తులేని అక్షరం)0x0);
    స్థిరరాశి సదిశ గుర్తులేని పొట్టి vsign = vec_splats((గుర్తులేని పొట్టి)0x8000);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    కొరకు (పూర్ణాంకం i = 0; i < nb; ++i) {
        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[i].d));
        సదిశ దశాంశం vyd = vec_splats(y[i].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = vec_splats((int32_t)0);
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = vec_splats((int32_t)0);
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = vec_splats((int32_t)0);
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = vec_splats((int32_t)0);
        సదిశ గుర్తుగల పూర్ణాంకం vsumi8 = vec_splats((int32_t)0);

        స్థిరరాశి uint8_t  * GGML_RESTRICT q1 = x[i].qs;
        స్థిరరాశి uint16_t * GGML_RESTRICT qh = x[i].qh;
        స్థిరరాశి int8_t   * GGML_RESTRICT q8 = y[i].qs;
        స్థిరరాశి int16_t  * GGML_RESTRICT qs = y[i].bsums;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_0 = {*(స్థిరరాశి int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(స్థిరరాశి int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_1 = {*(స్థిరరాశి int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(స్థిరరాశి int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_2 = {*(స్థిరరాశి int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(స్థిరరాశి int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            సదిశ గుర్తుగల దీర్ఘ దీర్ఘ aux64x2_3 = {*(స్థిరరాశి int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(స్థిరరాశి int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            సదిశ గుర్తుగల అక్షరం q1x0 = (సదిశ గుర్తుగల అక్షరం)aux64x2_0;
            సదిశ గుర్తుగల అక్షరం q1x1 = (సదిశ గుర్తుగల అక్షరం)aux64x2_1;
            సదిశ గుర్తుగల అక్షరం q1x2 = (సదిశ గుర్తుగల అక్షరం)aux64x2_2;
            సదిశ గుర్తుగల అక్షరం q1x3 = (సదిశ గుర్తుగల అక్షరం)aux64x2_3;

            సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, q8);
            సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, q8);
            సదిశ గుర్తుగల అక్షరం q8y2 = vec_xl(32, q8);
            సదిశ గుర్తుగల అక్షరం q8y3 = vec_xl(48, q8);
            q8 += 64;

            సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            సదిశ గుర్తుగల పొట్టి qv2 = vec_కూడు(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            సదిశ గుర్తుగల పొట్టి qv3 = vec_కూడు(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            స్థిరరాశి uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            స్థిరరాశి uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            సదిశ గుర్తుగల పొట్టి vscales01 = (సదిశ గుర్తుగల పొట్టి)vec_splats((uint16_t)(2*ls0+1));
            సదిశ గుర్తుగల పొట్టి vscales23 = (సదిశ గుర్తుగల పొట్టి)vec_splats((uint16_t)(2*ls1+1));
            సదిశ గుర్తుగల పొట్టి vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            సదిశ గుర్తుగల పొట్టి q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (సదిశ గుర్తుగల పొట్టి)v0);

            సదిశ గుర్తుగల పొట్టి qxh = (సదిశ గుర్తుగల పొట్టి)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            సదిశ __bool పొట్టి vsel = vec_cmpge(qxh, (సదిశ గుర్తుగల పొట్టి)v0);

            సదిశ గుర్తుగల పొట్టి q8ysum = vec_sel((సదిశ గుర్తుగల పొట్టి)vec_xor((సదిశ గుర్తులేని పొట్టి)q8ysums, vsign), q8ysums, vsel);

            vsumi8 = vec_కూడు(vec_mule(q8ysum, vscales), vsumi8);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);

        vsumf0 = vec_madd(vec_ctf(vsumi8, 0), vec_mul(vd, vec_splats(IQ1S_DELTA)), vsumf0);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_వెలికితీయి(vsumf0, 0);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}

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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ గుర్తుగల పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);

    స్థిరరాశి సదిశ గుర్తుగల అక్షరం విలువలు = vec_xl( 0, kvalues_iq4nl);

#సూచన GCC unroll 4
    కొరకు (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ib].d));
        సదిశ దశాంశం vyd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(y[ib].d));
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల అక్షరం qxs = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, x[ib].qs);
        సదిశ గుర్తుగల అక్షరం q4x0 = vec_and(qxs, దిగువ_ముసుగు);
        సదిశ గుర్తుగల అక్షరం q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(విలువలు, విలువలు, (సదిశ గుర్తులేని అక్షరం)q4x0);
        q4x1 = vec_perm(విలువలు, విలువలు, (సదిశ గుర్తులేని అక్షరం)q4x1);

        సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, y[ib].qs);
        సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, y[ib].qs);

        సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_వెలికితీయి(vsumf0, 0);

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

#ఒకవేళ defined(__POWER9_VECTOR__)
    స్థిరరాశి సదిశ గుర్తుగల అక్షరం దిగువ_ముసుగు = vec_splats((గుర్తుగల అక్షరం)0xF);
    స్థిరరాశి సదిశ పూర్ణాంకం v0 = vec_splats((int32_t)0);
    స్థిరరాశి సదిశ గుర్తులేని అక్షరం v4 = vec_splats((గుర్తులేని అక్షరం)0x4);

    సదిశ దశాంశం vsumf0 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf1 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf2 = vec_splats(0.0f);
    సదిశ దశాంశం vsumf3 = vec_splats(0.0f);

    స్థిరరాశి సదిశ గుర్తుగల అక్షరం విలువలు = vec_xl( 0, kvalues_iq4nl);

    కొరకు (పూర్ణాంకం ibl = 0; ibl < nb; ++ibl) {

        సదిశ దశాంశం vxd = vec_splats(GGML_CPU_FP16_గమ్యం_FP32(x[ibl].d));
        సదిశ దశాంశం vyd = vec_splats(y[ibl].d);
        సదిశ దశాంశం vd = vec_mul(vxd, vyd);

        సదిశ గుర్తుగల పూర్ణాంకం vsumi0 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi1 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi2 = v0;
        సదిశ గుర్తుగల పూర్ణాంకం vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        స్థిరరాశి uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        స్థిరరాశి uint8_t * GGML_RESTRICT sc = x[ibl].scales_l;
        స్థిరరాశి int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        కొరకు (పూర్ణాంకం ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            సదిశ గుర్తుగల అక్షరం qxs0 = (సదిశ గుర్తుగల అక్షరం)vec_xl( 0, q4);
            సదిశ గుర్తుగల అక్షరం qxs1 = (సదిశ గుర్తుగల అక్షరం)vec_xl(16, q4);
            q4 += 32;

            సదిశ గుర్తుగల అక్షరం q4x00 = (సదిశ గుర్తుగల అక్షరం)vec_and(qxs0, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం q4x01 = (సదిశ గుర్తుగల అక్షరం)vec_sr(qxs0, v4);
            సదిశ గుర్తుగల అక్షరం q4x10 = (సదిశ గుర్తుగల అక్షరం)vec_and(qxs1, దిగువ_ముసుగు);
            సదిశ గుర్తుగల అక్షరం q4x11 = (సదిశ గుర్తుగల అక్షరం)vec_sr(qxs1, v4);

            q4x00 = vec_perm(విలువలు, విలువలు, (సదిశ గుర్తులేని అక్షరం)q4x00);
            q4x01 = vec_perm(విలువలు, విలువలు, (సదిశ గుర్తులేని అక్షరం)q4x01);
            q4x10 = vec_perm(విలువలు, విలువలు, (సదిశ గుర్తులేని అక్షరం)q4x10);
            q4x11 = vec_perm(విలువలు, విలువలు, (సదిశ గుర్తులేని అక్షరం)q4x11);

            సదిశ గుర్తుగల అక్షరం q8y0 = vec_xl( 0, q8);
            సదిశ గుర్తుగల అక్షరం q8y1 = vec_xl(16, q8);
            సదిశ గుర్తుగల అక్షరం q8y2 = vec_xl(32, q8);
            సదిశ గుర్తుగల అక్షరం q8y3 = vec_xl(48, q8);
            q8 += 64;

            సదిశ గుర్తుగల పొట్టి qv0 = vec_కూడు(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            సదిశ గుర్తుగల పొట్టి qv1 = vec_కూడు(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            సదిశ గుర్తుగల పొట్టి qv2 = vec_కూడు(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            సదిశ గుర్తుగల పొట్టి qv3 = vec_కూడు(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            స్థిరరాశి uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            స్థిరరాశి uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            సదిశ గుర్తుగల పొట్టి vscales01 = vec_splats((int16_t)ls0);
            సదిశ గుర్తుగల పొట్టి vscales23 = vec_splats((int16_t)ls1);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_కూడు(vsumf0, vsumf2);
    vsumf1 = vec_కూడు(vsumf1, vsumf3);

    vsumf0 = vec_కూడు(vsumf0, vsumf1);

    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_కూడు(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_వెలికితీయి(vsumf0, 0);

#లేకపోతే
    వాడని(x);
    వాడని(y);
    వాడని(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#అయితే_ముగింపు
}
