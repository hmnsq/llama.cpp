#परिभाषय GGML_COMMON_IMPL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "ggml-quants.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "simd-mappings.h"

#अन्तर्भाव "../../quants.h"
#अन्तर्भाव "../../ggml-cpu-impl.h"

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <assert.h>
#अन्तर्भाव <float.h>
#अन्तर्भाव <stdlib.h> // for qsort
#अन्तर्भाव <stdio.h>  // for GGML_ASSERT

#परिभाषय गणः_MAX_EPS 1e-15f
#परिभाषय गणः_MAX_EPS_IQ3_XXS 1e-8f
#परिभाषय गणः_MAX_EPS_IQ2_S 1e-8f
#परिभाषय गणः_MAX_EPS_IQ1_M 1e-7f
#परिभाषय गणः_MAX_EPS_IQ1_S 1e-12f

#परिभाषय अप्रयुक्तम् GGML_अप्रयुक्तम्

#यदि defined(__POWER9_VECTOR__)
#परिभाषय B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#परिभाषय B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#परिभाषय B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#परिभाषय B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#परिभाषय B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#परिभाषय B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#परिभाषय B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#परिभाषय B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
स्थिर ध्रुव uint64_t सारणी_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
स्थिर ध्रुव uint64_t सारणी_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#यदि_अन्तः

शून्य quantize_row_q8_0(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k) {
    प्रतिज्ञा(QK8_0 == 32);
    प्रतिज्ञा(k % QK8_0 == 0);
    ध्रुव पूर्णाङ्क nb = k / QK8_0;

    खण्डम्_q8_0 * GGML_RESTRICT y = vy;

#यदि defined(__POWER9_VECTOR__)
    प्रति (पूर्णाङ्क i = 0; i < nb; i++) {
        सदिशम् भिन्न srcv [8];
        सदिशम् भिन्न asrcv[8];
        सदिशम् भिन्न amaxv[8];
        सदिशम् चिह्नित पूर्णाङ्क vi[8];

        प्रति (पूर्णाङ्क j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        प्रति (पूर्णाङ्क j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        प्रति (पूर्णाङ्क j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        प्रति (पूर्णाङ्क j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        प्रति (पूर्णाङ्क j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ध्रुव भिन्न amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        ध्रुव भिन्न d = amax / ((1 << 7) - 1);
        ध्रुव भिन्न id = d ? 1.0f/d : 0.0f;
        ध्रुव सदिशम् भिन्न vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_गन्तव्यम्_FP16(d);

        प्रति (पूर्णाङ्क j = 0; j < 8; j++) {
            ध्रुव सदिशम् भिन्न v  = vec_आवर्तनम्(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }
#अन्यथा
    GGML_अप्रयुक्तम्(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#यदि_अन्तः
}

शून्य quantize_row_q8_1(ध्रुव भिन्न * GGML_RESTRICT x, शून्य * GGML_RESTRICT vy, int64_t k) {
    प्रतिज्ञा(k % QK8_1 == 0);
    ध्रुव पूर्णाङ्क nb = k / QK8_1;

    खण्डम्_q8_1 * GGML_RESTRICT y = vy;

#यदि defined(__POWER9_VECTOR__)
    प्रति (पूर्णाङ्क i = 0; i < nb; i++) {
        सदिशम् भिन्न srcv [8];
        सदिशम् भिन्न asrcv[8];
        सदिशम् भिन्न amaxv[8];
        सदिशम् चिह्नित पूर्णाङ्क vi[8];

        प्रति (पूर्णाङ्क j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        प्रति (पूर्णाङ्क j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        प्रति (पूर्णाङ्क j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        प्रति (पूर्णाङ्क j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        प्रति (पूर्णाङ्क j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ध्रुव भिन्न amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        ध्रुव भिन्न d = amax / ((1 << 7) - 1);
        ध्रुव भिन्न id = d ? 1.0f/d : 0.0f;
        ध्रुव सदिशम् भिन्न vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_गन्तव्यम्_FP16(d);

        सदिशम् पूर्णाङ्क accv = vec_splats(0);

        प्रति (पूर्णाङ्क j = 0; j < 8; j++) {
            ध्रुव सदिशम् भिन्न v  = vec_आवर्तनम्(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_सङ्कलय(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_सङ्कलय(accv, vec_sld(accv, accv, 4));
        accv = vec_सङ्कलय(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_CPU_FP32_गन्तव्यम्_FP16(d * vec_extract(accv, 0));
    }

#अन्यथा
    GGML_अप्रयुक्तम्(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#यदि_अन्तः
}


//===================================== Dot products =================================

शून्य ggml_vec_dot_q4_0_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    ध्रुव पूर्णाङ्क qk = QK8_0;
    ध्रुव पूर्णाङ्क nb = n / qk;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_q4_0 * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_0 * GGML_RESTRICT y = vy;

    पूर्णाङ्क ib = 0;
    भिन्न sumf = 0;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् चिह्नित पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);
    ध्रुव सदिशम् चिह्नित वर्ण v8 = vec_splats((चिह्नित वर्ण)0x8);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);

#निर्देशः GCC unroll 8
    प्रति (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[ib].d));
        सदिशम् भिन्न vyd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(y[ib].d));
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित वर्ण qxs = (सदिशम् चिह्नित वर्ण)vec_xl( 0, x[ib].qs);
        सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, y[ib].qs);
        सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, y[ib].qs);

        सदिशम् चिह्नित वर्ण q4x0 = vec_and(qxs, नीचः_आवरणम्);
        सदिशम् चिह्नित वर्ण q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(ib);
    अप्रयुक्तम्(sumf);
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_q4_1_q8_1(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    ध्रुव पूर्णाङ्क qk = QK8_1;
    ध्रुव पूर्णाङ्क nb = n / qk;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_q4_1 * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_1 * GGML_RESTRICT y = vy;

    पूर्णाङ्क ib = 0;
    भिन्न sumf = 0;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् चिह्नित पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);

#निर्देशः GCC unroll 4
    प्रति (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[ib].d));
        सदिशम् भिन्न vyd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(y[ib].d));
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् भिन्न vxmin = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[ib].m));
        सदिशम् भिन्न vys = {GGML_CPU_FP16_गन्तव्यम्_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        सदिशम् चिह्नित वर्ण qxs = (सदिशम् चिह्नित वर्ण)vec_xl( 0, x[ib].qs);
        सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, y[ib].qs);
        सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, y[ib].qs);

        सदिशम् अचिह्नित वर्ण q4x0 = (सदिशम् अचिह्नित वर्ण)vec_and(qxs, नीचः_आवरणम्);
        सदिशम् अचिह्नित वर्ण q4x1 = (सदिशम् अचिह्नित वर्ण)vec_sr(qxs, v4);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(ib);
    अप्रयुक्तम्(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_mxfp4_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);
    प्रतिज्ञा(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ध्रुव खण्डम्_mxfp4 * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_0 * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_MXFP4;

    पूर्णाङ्क ib = 0;
    भिन्न sumf = 0;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् अचिह्नित वर्ण vshift4 = vec_splats((अचिह्नित वर्ण)4);
    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);

    सदिशम् चिह्नित वर्ण kv = vec_xl(0, (ध्रुव चिह्नित वर्ण *)kvalues_mxfp4);

#निर्देशः GCC unroll 8
    प्रति (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        सदिशम् भिन्न vyd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(y[ib].d) *
                                      GGML_E8_M0_गन्तव्यम्_FP32_HALF(x[ib].e));

        सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, y[ib].qs);
        सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, y[ib].qs);

        सदिशम् चिह्नित वर्ण qxs = (सदिशम् चिह्नित वर्ण)vec_xl(0, x[ib].qs);

        सदिशम् अचिह्नित वर्ण lo_nibbles = (सदिशम् अचिह्नित वर्ण)vec_and(qxs, नीचः_आवरणम्);
        सदिशम् अचिह्नित वर्ण hi_nibbles = (सदिशम् अचिह्नित वर्ण)vec_sr(qxs, vshift4);

        सदिशम् चिह्नित वर्ण q4x0 = vec_perm(kv, kv, lo_nibbles);
        सदिशम् चिह्नित वर्ण q4x1 = vec_perm(kv, kv, hi_nibbles);

        सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = vec_splats((int32_t)0);
        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vyd, vsumf0);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));
    sumf = vec_extract(vsumf0, 0);
    *s = sumf;
#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(ib);
    अप्रयुक्तम्(sumf);
    ggml_vec_dot_mxfp4_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_q5_0_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    ध्रुव पूर्णाङ्क qk = QK8_0;
    ध्रुव पूर्णाङ्क nb = n / qk;

    पूर्णाङ्क ib = 0;
    भिन्न sumf = 0;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(qk == QK5_0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_q5_0 * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_0 * GGML_RESTRICT y = vy;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)4);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);

#निर्देशः GCC unroll 4
    प्रति (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[ib].d));
        सदिशम् भिन्न vyd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(y[ib].d));
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_0 = {(uint64_t)(सारणी_b2b_1[x[ib].qh[0]]), (uint64_t)(सारणी_b2b_1[x[ib].qh[1]])};
        सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_1 = {(uint64_t)(सारणी_b2b_1[x[ib].qh[2]]), (uint64_t)(सारणी_b2b_1[x[ib].qh[3]])};

        सदिशम् चिह्नित वर्ण qh0 = (सदिशम् चिह्नित वर्ण)aux64x2_0;
        सदिशम् चिह्नित वर्ण qh1 = (सदिशम् चिह्नित वर्ण)aux64x2_1;

        सदिशम् चिह्नित वर्ण qxs = (सदिशम् चिह्नित वर्ण)vec_xl( 0, x[ib].qs);

        सदिशम् चिह्नित वर्ण q5x0 = vec_sub(vec_and (qxs, नीचः_आवरणम्), qh0);
        सदिशम् चिह्नित वर्ण q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

        सदिशम् चिह्नित वर्ण q8y0 = vec_xl(  0, y[ib].qs);
        सदिशम् चिह्नित वर्ण q8y1 = vec_xl( 16, y[ib].qs);

        सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
        सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q5x1, q8y1), vec_mulo(q5x1, q8y1));

        qv0 = vec_सङ्कलय(qv0, qv1);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = vec_सङ्कलय(vec_unpackh(qv0), vec_unpackl(qv0));

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    अप्रयुक्तम्(ib);
    अप्रयुक्तम्(sumf);
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_q5_1_q8_1(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    ध्रुव पूर्णाङ्क qk = QK8_1;
    ध्रुव पूर्णाङ्क nb = n / qk;

    पूर्णाङ्क ib = 0;
    भिन्न sumf = 0;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(qk == QK5_1);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_q5_1 * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_1 * GGML_RESTRICT y = vy;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् चिह्नित पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);

#निर्देशः GCC unroll 4
    प्रति (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[ib].d));
        सदिशम् भिन्न vyd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(y[ib].d));
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् भिन्न vxmin = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[ib].m));
        सदिशम् भिन्न vys = {GGML_CPU_FP16_गन्तव्यम्_FP32(y[ib].s), 0.f, 0.f, 0.f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        सदिशम् अचिह्नित दीर्घ दीर्घ aux64x2_0 = {(uint64_t)(सारणी_b2b_0[x[ib].qh[0]]), (uint64_t)(सारणी_b2b_0[x[ib].qh[1]])};
        सदिशम् अचिह्नित दीर्घ दीर्घ aux64x2_1 = {(uint64_t)(सारणी_b2b_0[x[ib].qh[2]]), (uint64_t)(सारणी_b2b_0[x[ib].qh[3]])};

        सदिशम् चिह्नित वर्ण qh0 = (सदिशम् चिह्नित वर्ण)aux64x2_0;
        सदिशम् चिह्नित वर्ण qh1 = (सदिशम् चिह्नित वर्ण)aux64x2_1;

        सदिशम् चिह्नित वर्ण qxs = (सदिशम् चिह्नित वर्ण)vec_xl( 0, x[ib].qs);

        सदिशम् अचिह्नित वर्ण q5x0 = (सदिशम् अचिह्नित वर्ण)vec_or(vec_and(qxs, नीचः_आवरणम्), qh0);
        सदिशम् अचिह्नित वर्ण q5x1 = (सदिशम् अचिह्नित वर्ण)vec_or(vec_sr(qxs, v4), qh1);

        सदिशम् चिह्नित वर्ण q8y0 = vec_xl(  0, y[ib].qs);
        सदिशम् चिह्नित वर्ण q8y1 = vec_xl( 16, y[ib].qs);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q5x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q5x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    अप्रयुक्तम्(nb);
    अप्रयुक्तम्(ib);
    अप्रयुक्तम्(sumf);
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_q8_0_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    ध्रुव पूर्णाङ्क qk = QK8_0;
    ध्रुव पूर्णाङ्क nb = n / qk;

    प्रतिज्ञा(n % qk == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_q8_0 * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_0 * GGML_RESTRICT y = vy;

    पूर्णाङ्क ib = 0;
    भिन्न sumf = 0;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित पूर्णाङ्क v0 = vec_splats((int32_t)0);
    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);

#निर्देशः GCC unroll 8
    प्रति (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[ib].d));
        सदिशम् भिन्न vyd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(y[ib].d));
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित वर्ण q8x0 = vec_xl( 0, x[ib].qs);
        सदिशम् चिह्नित वर्ण q8x1 = vec_xl(16, x[ib].qs);
        सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, y[ib].qs);
        सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, y[ib].qs);

        सदिशम् चिह्नित ह्रस्व qv0 = vec_mule(q8x0, q8y0);
        सदिशम् चिह्नित ह्रस्व qv1 = vec_mulo(q8x0, q8y0);
        सदिशम् चिह्नित ह्रस्व qv2 = vec_mule(q8x1, q8y1);
        सदिशम् चिह्नित ह्रस्व qv3 = vec_mulo(q8x1, q8y1);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_सङ्कलय(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    अप्रयुक्तम्(nb);
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(ib);
    अप्रयुक्तम्(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_q2_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_q2_K * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0x3);
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_Scale_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् अचिह्नित वर्ण v2 = vec_splats((अचिह्नित वर्ण)0x2);
    ध्रुव सदिशम् अचिह्नित वर्ण v6 = vec_splats((अचिह्नित वर्ण)0x6);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् भिन्न vxmin = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].dmin));
        सदिशम् भिन्न vdmin = vec_mul(vxmin, vyd);

        सदिशम् चिह्नित ह्रस्व q8ysums0 = vec_xl( 0, y[i].bsums);
        सदिशम् चिह्नित ह्रस्व q8ysums1 = vec_xl(16, y[i].bsums);

        सदिशम् चिह्नित वर्ण q2xmins = (सदिशम् चिह्नित वर्ण)vec_xl( 0, x[i].scales);
        सदिशम् चिह्नित वर्ण vscales = vec_and(q2xmins, नीचः_Scale_आवरणम्);

        q2xmins = vec_sr(q2xmins, v4);
        सदिशम् चिह्नित ह्रस्व q2xmins0 = vec_unpackh(q2xmins);
        सदिशम् चिह्नित ह्रस्व q2xmins1 = vec_unpackl(q2xmins);

        सदिशम् चिह्नित पूर्णाङ्क prod0 = vec_mule(q2xmins0, q8ysums0);
        सदिशम् चिह्नित पूर्णाङ्क prod1 = vec_mulo(q2xmins0, q8ysums0);
        सदिशम् चिह्नित पूर्णाङ्क prod2 = vec_mule(q2xmins1, q8ysums1);
        सदिशम् चिह्नित पूर्णाङ्क prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi4 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi5 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi6 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi7 = v0;

        ध्रुव uint8_t * GGML_RESTRICT q2 = x[i].qs;
        ध्रुव int8_t  * GGML_RESTRICT q8 = y[i].qs;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् चिह्नित वर्ण qxs0 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, q2);
            सदिशम् चिह्नित वर्ण qxs1 = (सदिशम् चिह्नित वर्ण)vec_xl(16, q2);
            q2 += 32;

            सदिशम् अचिह्नित वर्ण q2x00 = (सदिशम् अचिह्नित वर्ण)vec_and(qxs0, नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q2x01 = (सदिशम् अचिह्नित वर्ण)vec_and(vec_sr(qxs0, v2), नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q2x02 = (सदिशम् अचिह्नित वर्ण)vec_and(vec_sr(qxs0, v4), नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q2x03 = (सदिशम् अचिह्नित वर्ण)vec_and(vec_sr(qxs0, v6), नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q2x10 = (सदिशम् अचिह्नित वर्ण)vec_and(qxs1, नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q2x11 = (सदिशम् अचिह्नित वर्ण)vec_and(vec_sr(qxs1, v2), नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q2x12 = (सदिशम् अचिह्नित वर्ण)vec_and(vec_sr(qxs1, v4), नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q2x13 = (सदिशम् अचिह्नित वर्ण)vec_and(vec_sr(qxs1, v6), नीचः_आवरणम्);

            सदिशम् चिह्नित वर्ण q8y00 = vec_xl(  0, q8);
            सदिशम् चिह्नित वर्ण q8y10 = vec_xl( 16, q8);
            सदिशम् चिह्नित वर्ण q8y01 = vec_xl( 32, q8);
            सदिशम् चिह्नित वर्ण q8y11 = vec_xl( 48, q8);
            सदिशम् चिह्नित वर्ण q8y02 = vec_xl( 64, q8);
            सदिशम् चिह्नित वर्ण q8y12 = vec_xl( 80, q8);
            सदिशम् चिह्नित वर्ण q8y03 = vec_xl( 96, q8);
            सदिशम् चिह्नित वर्ण q8y13 = vec_xl(112, q8);
            q8 += 128;

            सदिशम् चिह्नित पूर्णाङ्क qv0 = vec_msum(q8y00, q2x00, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv1 = vec_msum(q8y01, q2x01, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv2 = vec_msum(q8y02, q2x02, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv3 = vec_msum(q8y03, q2x03, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv4 = vec_msum(q8y10, q2x10, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv5 = vec_msum(q8y11, q2x11, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv6 = vec_msum(q8y12, q2x12, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv7 = vec_msum(q8y13, q2x13, v0);

            सदिशम् चिह्नित ह्रस्व vscales_07 = vec_unpackh(vscales);
            सदिशम् चिह्नित पूर्णाङ्क vscales_03 = vec_unpackh(vscales_07);
            सदिशम् चिह्नित पूर्णाङ्क vscales_47 = vec_unpackl(vscales_07);
            सदिशम् चिह्नित पूर्णाङ्क vs0 = vec_splat(vscales_03, 0);
            सदिशम् चिह्नित पूर्णाङ्क vs1 = vec_splat(vscales_03, 1);
            सदिशम् चिह्नित पूर्णाङ्क vs2 = vec_splat(vscales_03, 2);
            सदिशम् चिह्नित पूर्णाङ्क vs3 = vec_splat(vscales_03, 3);
            सदिशम् चिह्नित पूर्णाङ्क vs4 = vec_splat(vscales_47, 0);
            सदिशम् चिह्नित पूर्णाङ्क vs5 = vec_splat(vscales_47, 1);
            सदिशम् चिह्नित पूर्णाङ्क vs6 = vec_splat(vscales_47, 2);
            सदिशम् चिह्नित पूर्णाङ्क vs7 = vec_splat(vscales_47, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_सङ्कलय(vec_mul(qv0, vs0), vsumi0);
            vsumi1 = vec_सङ्कलय(vec_mul(qv1, vs2), vsumi1);
            vsumi2 = vec_सङ्कलय(vec_mul(qv2, vs4), vsumi2);
            vsumi3 = vec_सङ्कलय(vec_mul(qv3, vs6), vsumi3);
            vsumi4 = vec_सङ्कलय(vec_mul(qv4, vs1), vsumi4);
            vsumi5 = vec_सङ्कलय(vec_mul(qv5, vs3), vsumi5);
            vsumi6 = vec_सङ्कलय(vec_mul(qv6, vs5), vsumi6);
            vsumi7 = vec_सङ्कलय(vec_mul(qv7, vs7), vsumi7);
        }

        vsumi0 = vec_सङ्कलय(vsumi0, vsumi4);
        vsumi1 = vec_सङ्कलय(vsumi1, vsumi5);
        vsumi2 = vec_सङ्कलय(vsumi2, vsumi6);
        vsumi3 = vec_सङ्कलय(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_q3_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव uint32_t kmask1 = 0x03030303;
    ध्रुव uint32_t kmask2 = 0x0f0f0f0f;

    ध्रुव खण्डम्_q3_K * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0x3);
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_Mask1 = vec_splats((int8_t)0xf);
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_Mask2 = vec_splats((int8_t)0x30);
    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् चिह्नित वर्ण v1 = vec_splats((चिह्नित वर्ण)0x1);
    ध्रुव सदिशम् अचिह्नित वर्ण v2 = vec_splats((अचिह्नित वर्ण)0x2);
    ध्रुव सदिशम् अचिह्नित वर्ण v3 = vec_splats((अचिह्नित वर्ण)0x3);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);
    ध्रुव सदिशम् अचिह्नित वर्ण v6 = vec_splats((अचिह्नित वर्ण)0x6);
    ध्रुव सदिशम् चिह्नित वर्ण off = vec_splats((चिह्नित वर्ण)0x20);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        अप्रयुक्तम्(kmask1);
        अप्रयुक्तम्(kmask2);

        सदिशम् चिह्नित वर्ण u0 = (सदिशम् चिह्नित वर्ण)vec_xl_len(x[i].scales, 8);
        सदिशम् चिह्नित वर्ण u1 = vec_and(u0, नीचः_Mask1);
        सदिशम् चिह्नित वर्ण u2 = (सदिशम् चिह्नित वर्ण)vec_xl_len(x[i].scales + 8, 4);
        सदिशम् चिह्नित वर्ण u3 = (सदिशम् चिह्नित वर्ण)vec_mergeh((सदिशम् चिह्नित पूर्णाङ्क)u2, (सदिशम् चिह्नित पूर्णाङ्क)vec_sr(u2, v2));
        सदिशम् चिह्नित वर्ण u30 = vec_sl(vec_and(u3, नीचः_आवरणम्), v4);
        सदिशम् चिह्नित वर्ण u31 = vec_and(u3, नीचः_Mask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        सदिशम् चिह्नित वर्ण vscales = (सदिशम् चिह्नित वर्ण)vec_mergeh((सदिशम् चिह्नित दीर्घ दीर्घ)u1, (सदिशम् चिह्नित दीर्घ दीर्घ)u2);
        सदिशम् चिह्नित वर्ण qxhs0 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, x[i].hmask);
        सदिशम् चिह्नित वर्ण qxhs1 = (सदिशम् चिह्नित वर्ण)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi4 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi5 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi6 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi7 = v0;

        ध्रुव uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ध्रुव int8_t  * GGML_RESTRICT q8 = y[i].qs;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् चिह्नित वर्ण qxs0 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, q3);
            सदिशम् चिह्नित वर्ण qxs1 = (सदिशम् चिह्नित वर्ण)vec_xl(16, q3);
            q3 += 32;

            //the low 2 bits
            सदिशम् चिह्नित वर्ण qxs00 = vec_and(qxs0, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs01 = vec_and(vec_sr(qxs0, v2), नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs02 = vec_and(vec_sr(qxs0, v4), नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs03 = vec_and(vec_sr(qxs0, v6), नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs10 = vec_and(qxs1, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs11 = vec_and(vec_sr(qxs1, v2), नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs12 = vec_and(vec_sr(qxs1, v4), नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs13 = vec_and(vec_sr(qxs1, v6), नीचः_आवरणम्);

            //the 3rd bit
            सदिशम् चिह्नित वर्ण qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            सदिशम् चिह्नित वर्ण qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (सदिशम् अचिह्नित वर्ण)v1)), v2);
            सदिशम् चिह्नित वर्ण qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            सदिशम् चिह्नित वर्ण qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            सदिशम् चिह्नित वर्ण qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            सदिशम् चिह्नित वर्ण qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (सदिशम् अचिह्नित वर्ण)v1)), v2);
            सदिशम् चिह्नित वर्ण qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            सदिशम् चिह्नित वर्ण qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            सदिशम् चिह्नित वर्ण q3x00 = vec_sub(qxs00, qxh00);
            सदिशम् चिह्नित वर्ण q3x01 = vec_sub(qxs01, qxh01);
            सदिशम् चिह्नित वर्ण q3x02 = vec_sub(qxs02, qxh02);
            सदिशम् चिह्नित वर्ण q3x03 = vec_sub(qxs03, qxh03);
            सदिशम् चिह्नित वर्ण q3x10 = vec_sub(qxs10, qxh10);
            सदिशम् चिह्नित वर्ण q3x11 = vec_sub(qxs11, qxh11);
            सदिशम् चिह्नित वर्ण q3x12 = vec_sub(qxs12, qxh12);
            सदिशम् चिह्नित वर्ण q3x13 = vec_sub(qxs13, qxh13);

            सदिशम् चिह्नित वर्ण q8y00 = vec_xl(  0, q8);
            सदिशम् चिह्नित वर्ण q8y10 = vec_xl( 16, q8);
            सदिशम् चिह्नित वर्ण q8y01 = vec_xl( 32, q8);
            सदिशम् चिह्नित वर्ण q8y11 = vec_xl( 48, q8);
            सदिशम् चिह्नित वर्ण q8y02 = vec_xl( 64, q8);
            सदिशम् चिह्नित वर्ण q8y12 = vec_xl( 80, q8);
            सदिशम् चिह्नित वर्ण q8y03 = vec_xl( 96, q8);
            सदिशम् चिह्नित वर्ण q8y13 = vec_xl(112, q8);
            q8 += 128;

            सदिशम् चिह्नित ह्रस्व vscales_h = vec_unpackh(vscales);
            सदिशम् चिह्नित ह्रस्व vs0 = vec_splat(vscales_h, 0);
            सदिशम् चिह्नित ह्रस्व vs1 = vec_splat(vscales_h, 1);
            सदिशम् चिह्नित ह्रस्व vs2 = vec_splat(vscales_h, 2);
            सदिशम् चिह्नित ह्रस्व vs3 = vec_splat(vscales_h, 3);
            सदिशम् चिह्नित ह्रस्व vs4 = vec_splat(vscales_h, 4);
            सदिशम् चिह्नित ह्रस्व vs5 = vec_splat(vscales_h, 5);
            सदिशम् चिह्नित ह्रस्व vs6 = vec_splat(vscales_h, 6);
            सदिशम् चिह्नित ह्रस्व vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            सदिशम् चिह्नित ह्रस्व qv00 = vec_सङ्कलय(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            सदिशम् चिह्नित ह्रस्व qv01 = vec_सङ्कलय(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            सदिशम् चिह्नित ह्रस्व qv02 = vec_सङ्कलय(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            सदिशम् चिह्नित ह्रस्व qv03 = vec_सङ्कलय(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            सदिशम् चिह्नित ह्रस्व qv10 = vec_सङ्कलय(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            सदिशम् चिह्नित ह्रस्व qv11 = vec_सङ्कलय(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            सदिशम् चिह्नित ह्रस्व qv12 = vec_सङ्कलय(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            सदिशम् चिह्नित ह्रस्व qv13 = vec_सङ्कलय(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs2, vsumi1);
            vsumi2 = vec_msum(qv02, vs4, vsumi2);
            vsumi3 = vec_msum(qv03, vs6, vsumi3);
            vsumi4 = vec_msum(qv10, vs1, vsumi4);
            vsumi5 = vec_msum(qv11, vs3, vsumi5);
            vsumi6 = vec_msum(qv12, vs5, vsumi6);
            vsumi7 = vec_msum(qv13, vs7, vsumi7);
        }

        vsumi0 = vec_सङ्कलय(vsumi0, vsumi4);
        vsumi1 = vec_सङ्कलय(vsumi1, vsumi5);
        vsumi2 = vec_सङ्कलय(vsumi2, vsumi6);
        vsumi3 = vec_सङ्कलय(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(kmask1);
    अप्रयुक्तम्(kmask2);
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_q4_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_q4_K * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

    स्थिर ध्रुव uint32_t kmask1 = 0x3f3f3f3f;
    स्थिर ध्रुव uint32_t kmask2 = 0x0f0f0f0f;
    स्थिर ध्रुव uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_Mask1 = vec_splats((int8_t)0x3f);
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_Mask2 = vec_splats((int8_t)0x30);
    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् अचिह्नित वर्ण v2 = vec_splats((uint8_t)2);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् भिन्न vxmin = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].dmin));
        सदिशम् भिन्न vdmin = vec_mul(vxmin, vyd);

        सदिशम् चिह्नित ह्रस्व q8ysums0 = vec_xl( 0, y[i].bsums);
        सदिशम् चिह्नित ह्रस्व q8ysums1 = vec_xl(16, y[i].bsums);

        अप्रयुक्तम्(kmask1);
        अप्रयुक्तम्(kmask2);
        अप्रयुक्तम्(kmask3);
        अप्रयुक्तम्(utmp);

        सदिशम् चिह्नित वर्ण u0 = (सदिशम् चिह्नित वर्ण)vec_xl_len(x[i].scales, 8);
        सदिशम् चिह्नित वर्ण u1 = vec_and(vec_sr(u0, v2), नीचः_Mask2);
        सदिशम् चिह्नित वर्ण u2 = (सदिशम् चिह्नित वर्ण)vec_xl_len(x[i].scales + 8, 4);
        सदिशम् चिह्नित वर्ण u3 = vec_sr(u2, v4);

        सदिशम् चिह्नित वर्ण u30 = u1;
        सदिशम् चिह्नित वर्ण u31 = (सदिशम् चिह्नित वर्ण)vec_mergeh((सदिशम् चिह्नित पूर्णाङ्क)vec_and(u2, नीचः_आवरणम्), (सदिशम् चिह्नित पूर्णाङ्क)u3);

        u1 = vec_and(u0, नीचः_Mask1);
        u2 = vec_or(u30, u31);

        सदिशम् चिह्नित वर्ण utmps = (सदिशम् चिह्नित वर्ण)vec_mergeh((सदिशम् चिह्नित पूर्णाङ्क)u1, (सदिशम् चिह्नित पूर्णाङ्क)u2);

        सदिशम् चिह्नित ह्रस्व vscales = vec_unpackh(utmps);
        सदिशम् चिह्नित ह्रस्व q4xmins = vec_unpackl(utmps);
        सदिशम् चिह्नित ह्रस्व q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        सदिशम् चिह्नित ह्रस्व q4xmins1 = vec_mergel(q4xmins, q4xmins);

        सदिशम् चिह्नित पूर्णाङ्क prod0 = vec_mule(q4xmins0, q8ysums0);
        सदिशम् चिह्नित पूर्णाङ्क prod1 = vec_mule(q4xmins1, q8ysums1);
        सदिशम् चिह्नित पूर्णाङ्क prod2 = vec_mulo(q4xmins0, q8ysums0);
        सदिशम् चिह्नित पूर्णाङ्क prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;

        ध्रुव uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ध्रुव int8_t  * GGML_RESTRICT q8 = y[i].qs;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् चिह्नित वर्ण qxs0 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, q4);
            सदिशम् चिह्नित वर्ण qxs1 = (सदिशम् चिह्नित वर्ण)vec_xl(16, q4);
            सदिशम् चिह्नित वर्ण qxs2 = (सदिशम् चिह्नित वर्ण)vec_xl(32, q4);
            सदिशम् चिह्नित वर्ण qxs3 = (सदिशम् चिह्नित वर्ण)vec_xl(48, q4);
            q4 += 64;

            सदिशम् अचिह्नित वर्ण q4x00 = (सदिशम् अचिह्नित वर्ण)vec_and(qxs0, नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q4x01 = (सदिशम् अचिह्नित वर्ण)vec_sr(qxs0, v4);
            सदिशम् अचिह्नित वर्ण q4x10 = (सदिशम् अचिह्नित वर्ण)vec_and(qxs1, नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q4x11 = (सदिशम् अचिह्नित वर्ण)vec_sr(qxs1, v4);
            सदिशम् अचिह्नित वर्ण q4x20 = (सदिशम् अचिह्नित वर्ण)vec_and(qxs2, नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q4x21 = (सदिशम् अचिह्नित वर्ण)vec_sr(qxs2, v4);
            सदिशम् अचिह्नित वर्ण q4x30 = (सदिशम् अचिह्नित वर्ण)vec_and(qxs3, नीचः_आवरणम्);
            सदिशम् अचिह्नित वर्ण q4x31 = (सदिशम् अचिह्नित वर्ण)vec_sr(qxs3, v4);

            सदिशम् चिह्नित वर्ण q8y00 = vec_xl(  0, q8);
            सदिशम् चिह्नित वर्ण q8y10 = vec_xl( 16, q8);
            सदिशम् चिह्नित वर्ण q8y01 = vec_xl( 32, q8);
            सदिशम् चिह्नित वर्ण q8y11 = vec_xl( 48, q8);
            सदिशम् चिह्नित वर्ण q8y20 = vec_xl( 64, q8);
            सदिशम् चिह्नित वर्ण q8y30 = vec_xl( 80, q8);
            सदिशम् चिह्नित वर्ण q8y21 = vec_xl( 96, q8);
            सदिशम् चिह्नित वर्ण q8y31 = vec_xl(112, q8);
            q8 += 128;

            सदिशम् चिह्नित पूर्णाङ्क qv00 = vec_msum(q8y00, q4x00, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv01 = vec_msum(q8y01, q4x01, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv10 = vec_msum(q8y10, q4x10, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv11 = vec_msum(q8y11, q4x11, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv20 = vec_msum(q8y20, q4x20, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv21 = vec_msum(q8y21, q4x21, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv30 = vec_msum(q8y30, q4x30, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv31 = vec_msum(q8y31, q4x31, v0);

            सदिशम् चिह्नित पूर्णाङ्क vscales_h = vec_unpackh(vscales);
            सदिशम् चिह्नित पूर्णाङ्क vs0 = vec_splat(vscales_h, 0);
            सदिशम् चिह्नित पूर्णाङ्क vs1 = vec_splat(vscales_h, 1);
            सदिशम् चिह्नित पूर्णाङ्क vs2 = vec_splat(vscales_h, 2);
            सदिशम् चिह्नित पूर्णाङ्क vs3 = vec_splat(vscales_h, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_सङ्कलय(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_सङ्कलय(vec_mul(qv01, vs1), vsumi1);
            vsumi2 = vec_सङ्कलय(vec_mul(qv20, vs2), vsumi2);
            vsumi3 = vec_सङ्कलय(vec_mul(qv21, vs3), vsumi3);

            vsumi0 = vec_सङ्कलय(vec_mul(qv10, vs0), vsumi0);
            vsumi1 = vec_सङ्कलय(vec_mul(qv11, vs1), vsumi1);
            vsumi2 = vec_सङ्कलय(vec_mul(qv30, vs2), vsumi2);
            vsumi3 = vec_सङ्कलय(vec_mul(qv31, vs3), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    अप्रयुक्तम्(kmask1);
    अप्रयुक्तम्(kmask2);
    अप्रयुक्तम्(kmask3);
    अप्रयुक्तम्(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_q5_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy,  size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_q5_K * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

    स्थिर ध्रुव uint32_t kmask1 = 0x3f3f3f3f;
    स्थिर ध्रुव uint32_t kmask2 = 0x0f0f0f0f;
    स्थिर ध्रुव uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_Mask1 = vec_splats((int8_t)0x3f);
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_Mask2 = vec_splats((int8_t)0x30);
    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् अचिह्नित वर्ण v1 = vec_splats((अचिह्नित वर्ण)0x1);
    ध्रुव सदिशम् अचिह्नित वर्ण v2 = vec_splats((अचिह्नित वर्ण)0x2);
    ध्रुव सदिशम् अचिह्नित वर्ण v3 = vec_splats((अचिह्नित वर्ण)0x3);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् भिन्न vxmin = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].dmin));
        सदिशम् भिन्न vdmin = vec_mul(vxmin, vyd);

        अप्रयुक्तम्(kmask1);
        अप्रयुक्तम्(kmask2);
        अप्रयुक्तम्(kmask3);
        अप्रयुक्तम्(utmp);

        सदिशम् चिह्नित वर्ण u0 = (सदिशम् चिह्नित वर्ण)vec_xl_len(x[i].scales, 8);
        सदिशम् चिह्नित वर्ण u1 = vec_and(vec_sr(u0, v2), नीचः_Mask2);
        सदिशम् चिह्नित वर्ण u2 = (सदिशम् चिह्नित वर्ण)vec_xl_len(x[i].scales + 8, 4);
        सदिशम् चिह्नित वर्ण u3 = vec_sr(u2, v4);

        सदिशम् चिह्नित वर्ण u30 = u1;
        सदिशम् चिह्नित वर्ण u31 = (सदिशम् चिह्नित वर्ण)vec_mergeh((सदिशम् चिह्नित पूर्णाङ्क)vec_and(u2, नीचः_आवरणम्), (सदिशम् चिह्नित पूर्णाङ्क)u3);

        u1 = vec_and(u0, नीचः_Mask1);
        u2 = vec_or(u30, u31);

        सदिशम् चिह्नित वर्ण utmps = (सदिशम् चिह्नित वर्ण)vec_mergeh((सदिशम् चिह्नित पूर्णाङ्क)u1, (सदिशम् चिह्नित पूर्णाङ्क)u2);

        सदिशम् चिह्नित ह्रस्व q8ysums0 = vec_xl( 0, y[i].bsums);
        सदिशम् चिह्नित ह्रस्व q8ysums1 = vec_xl(16, y[i].bsums);

        सदिशम् चिह्नित ह्रस्व vscales = vec_unpackh(utmps);

        सदिशम् चिह्नित ह्रस्व q5xmins = vec_unpackl(utmps);
        सदिशम् चिह्नित ह्रस्व q5xmins0 = vec_mergeh(q5xmins, q5xmins);
        सदिशम् चिह्नित ह्रस्व q5xmins1 = vec_mergel(q5xmins, q5xmins);

        सदिशम् चिह्नित पूर्णाङ्क prod0 = vec_mule(q5xmins0, q8ysums0);
        सदिशम् चिह्नित पूर्णाङ्क prod1 = vec_mule(q5xmins1, q8ysums1);
        सदिशम् चिह्नित पूर्णाङ्क prod2 = vec_mulo(q5xmins0, q8ysums0);
        सदिशम् चिह्नित पूर्णाङ्क prod3 = vec_mulo(q5xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        सदिशम् चिह्नित वर्ण qxhs0 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, x[i].qh);
        सदिशम् चिह्नित वर्ण qxhs1 = (सदिशम् चिह्नित वर्ण)vec_xl(16, x[i].qh);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;

        ध्रुव uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ध्रुव int8_t  * GGML_RESTRICT q8 = y[i].qs;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q5, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् चिह्नित वर्ण qxs0 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, q5);
            सदिशम् चिह्नित वर्ण qxs1 = (सदिशम् चिह्नित वर्ण)vec_xl(16, q5);
            q5 += 32;

            सदिशम् चिह्नित वर्ण qxs00 = vec_and(qxs0, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs01 = vec_sr(qxs0, v4);
            सदिशम् चिह्नित वर्ण qxs10 = vec_and(qxs1, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs11 = vec_sr(qxs1, v4);

            सदिशम् चिह्नित वर्ण q5h00 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v1, qxhs0), v4);
            सदिशम् चिह्नित वर्ण q5h01 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v2, qxhs0), v3);
            सदिशम् चिह्नित वर्ण q5h10 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v1, qxhs1), v4);
            सदिशम् चिह्नित वर्ण q5h11 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v2, qxhs1), v3);
            qxhs0 = vec_sr(qxhs0, v2);
            qxhs1 = vec_sr(qxhs1, v2);

            सदिशम् अचिह्नित वर्ण q5x00 = (सदिशम् अचिह्नित वर्ण)vec_or(q5h00, qxs00);
            सदिशम् अचिह्नित वर्ण q5x01 = (सदिशम् अचिह्नित वर्ण)vec_or(q5h01, qxs01);
            सदिशम् अचिह्नित वर्ण q5x10 = (सदिशम् अचिह्नित वर्ण)vec_or(q5h10, qxs10);
            सदिशम् अचिह्नित वर्ण q5x11 = (सदिशम् अचिह्नित वर्ण)vec_or(q5h11, qxs11);

            सदिशम् चिह्नित वर्ण q8y00 = vec_xl( 0, q8);
            सदिशम् चिह्नित वर्ण q8y10 = vec_xl(16, q8);
            सदिशम् चिह्नित वर्ण q8y01 = vec_xl(32, q8);
            सदिशम् चिह्नित वर्ण q8y11 = vec_xl(48, q8);
            q8 += 64;

            सदिशम् चिह्नित पूर्णाङ्क qv00 = vec_msum(q8y00, q5x00, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv01 = vec_msum(q8y01, q5x01, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv10 = vec_msum(q8y10, q5x10, v0);
            सदिशम् चिह्नित पूर्णाङ्क qv11 = vec_msum(q8y11, q5x11, v0);

            सदिशम् चिह्नित पूर्णाङ्क vscales_h = vec_unpackh(vscales);
            सदिशम् चिह्नित पूर्णाङ्क vs0 = vec_splat(vscales_h, 0);
            सदिशम् चिह्नित पूर्णाङ्क vs1 = vec_splat(vscales_h, 1);
            vscales = vec_sld(vscales, vscales, 12);

            vsumi0 = vec_सङ्कलय(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_सङ्कलय(vec_mul(qv10, vs0), vsumi1);
            vsumi2 = vec_सङ्कलय(vec_mul(qv01, vs1), vsumi2);
            vsumi3 = vec_सङ्कलय(vec_mul(qv11, vs1), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    अप्रयुक्तम्(kmask1);
    अप्रयुक्तम्(kmask2);
    अप्रयुक्तम्(kmask3);
    अप्रयुक्तम्(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_q6_K_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_q6_K * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् अचिह्नित वर्ण v2 = vec_splats((अचिह्नित वर्ण)0x2);
    ध्रुव सदिशम् अचिह्नित वर्ण v3 = vec_splats((अचिह्नित वर्ण)0x3);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);
    ध्रुव सदिशम् अचिह्नित वर्ण v6 = vec_splats((अचिह्नित वर्ण)0x6);
    ध्रुव सदिशम् चिह्नित वर्ण off = vec_splats((चिह्नित वर्ण)0x20);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi4 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi5 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi6 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi7 = v0;

        ध्रुव uint8_t * GGML_RESTRICT q6 = x[i].ql;
        ध्रुव uint8_t * GGML_RESTRICT qh = x[i].qh;
        ध्रुव int8_t  * GGML_RESTRICT qs = x[i].scales;
        ध्रुव int8_t  * GGML_RESTRICT q8 = y[i].qs;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q6, 0, 0);
            __builtin_prefetch(qh, 0, 0);
            __builtin_prefetch(q8, 0, 0);

            सदिशम् चिह्नित वर्ण qxs0 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, q6);
            सदिशम् चिह्नित वर्ण qxs1 = (सदिशम् चिह्नित वर्ण)vec_xl(16, q6);
            सदिशम् चिह्नित वर्ण qxs2 = (सदिशम् चिह्नित वर्ण)vec_xl(32, q6);
            सदिशम् चिह्नित वर्ण qxs3 = (सदिशम् चिह्नित वर्ण)vec_xl(48, q6);
            q6 += 64;

            सदिशम् चिह्नित वर्ण qxs00 = vec_and(qxs0, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs01 = vec_sr(qxs0, v4);
            सदिशम् चिह्नित वर्ण qxs10 = vec_and(qxs1, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs11 = vec_sr(qxs1, v4);
            सदिशम् चिह्नित वर्ण qxs20 = vec_and(qxs2, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs21 = vec_sr(qxs2, v4);
            सदिशम् चिह्नित वर्ण qxs30 = vec_and(qxs3, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण qxs31 = vec_sr(qxs3, v4);

            सदिशम् चिह्नित वर्ण qxhs0 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, qh);
            सदिशम् चिह्नित वर्ण qxhs1 = (सदिशम् चिह्नित वर्ण)vec_xl(16, qh);
            qh += 32;

            सदिशम् चिह्नित वर्ण qxh00 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v3, qxhs0), v4);
            सदिशम् चिह्नित वर्ण qxh01 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v3, vec_sr(qxhs0, v4)), v4);
            सदिशम् चिह्नित वर्ण qxh10 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v3, qxhs1), v4);
            सदिशम् चिह्नित वर्ण qxh11 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v3, vec_sr(qxhs1, v4)), v4);
            सदिशम् चिह्नित वर्ण qxh20 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v3, vec_sr(qxhs0, v2)), v4);
            सदिशम् चिह्नित वर्ण qxh21 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v3, vec_sr(qxhs0, v6)), v4);
            सदिशम् चिह्नित वर्ण qxh30 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v3, vec_sr(qxhs1, v2)), v4);
            सदिशम् चिह्नित वर्ण qxh31 = vec_sl(vec_and((सदिशम् चिह्नित वर्ण)v3, vec_sr(qxhs1, v6)), v4);

            सदिशम् चिह्नित वर्ण q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
            सदिशम् चिह्नित वर्ण q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
            सदिशम् चिह्नित वर्ण q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
            सदिशम् चिह्नित वर्ण q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
            सदिशम् चिह्नित वर्ण q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
            सदिशम् चिह्नित वर्ण q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
            सदिशम् चिह्नित वर्ण q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
            सदिशम् चिह्नित वर्ण q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

            सदिशम् चिह्नित वर्ण q8y00 = vec_xl(  0, q8);
            सदिशम् चिह्नित वर्ण q8y10 = vec_xl( 16, q8);
            सदिशम् चिह्नित वर्ण q8y20 = vec_xl( 32, q8);
            सदिशम् चिह्नित वर्ण q8y30 = vec_xl( 48, q8);
            सदिशम् चिह्नित वर्ण q8y01 = vec_xl( 64, q8);
            सदिशम् चिह्नित वर्ण q8y11 = vec_xl( 80, q8);
            सदिशम् चिह्नित वर्ण q8y21 = vec_xl( 96, q8);
            सदिशम् चिह्नित वर्ण q8y31 = vec_xl(112, q8);
            q8 += 128;

            सदिशम् चिह्नित ह्रस्व qv00 = vec_सङ्कलय(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
            सदिशम् चिह्नित ह्रस्व qv10 = vec_सङ्कलय(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
            सदिशम् चिह्नित ह्रस्व qv20 = vec_सङ्कलय(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
            सदिशम् चिह्नित ह्रस्व qv30 = vec_सङ्कलय(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
            सदिशम् चिह्नित ह्रस्व qv01 = vec_सङ्कलय(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
            सदिशम् चिह्नित ह्रस्व qv11 = vec_सङ्कलय(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
            सदिशम् चिह्नित ह्रस्व qv21 = vec_सङ्कलय(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
            सदिशम् चिह्नित ह्रस्व qv31 = vec_सङ्कलय(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

            सदिशम् चिह्नित ह्रस्व vscales = vec_unpackh(vec_xl_len(qs, 8));
            qs += 8;

            सदिशम् चिह्नित ह्रस्व vs0 = vec_splat(vscales, 0);
            सदिशम् चिह्नित ह्रस्व vs1 = vec_splat(vscales, 1);
            सदिशम् चिह्नित ह्रस्व vs2 = vec_splat(vscales, 2);
            सदिशम् चिह्नित ह्रस्व vs3 = vec_splat(vscales, 3);
            सदिशम् चिह्नित ह्रस्व vs4 = vec_splat(vscales, 4);
            सदिशम् चिह्नित ह्रस्व vs5 = vec_splat(vscales, 5);
            सदिशम् चिह्नित ह्रस्व vs6 = vec_splat(vscales, 6);
            सदिशम् चिह्नित ह्रस्व vs7 = vec_splat(vscales, 7);

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs4, vsumi1);
            vsumi2 = vec_msum(qv10, vs1, vsumi2);
            vsumi3 = vec_msum(qv11, vs5, vsumi3);
            vsumi4 = vec_msum(qv20, vs2, vsumi4);
            vsumi5 = vec_msum(qv21, vs6, vsumi5);
            vsumi6 = vec_msum(qv30, vs3, vsumi6);
            vsumi7 = vec_msum(qv31, vs7, vsumi7);
        }

        vsumi0 = vec_सङ्कलय(vsumi0, vsumi4);
        vsumi1 = vec_सङ्कलय(vsumi1, vsumi5);
        vsumi2 = vec_सङ्कलय(vsumi2, vsumi6);
        vsumi3 = vec_सङ्कलय(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

#यदि defined (__POWER9_VECTOR__)
स्थिर ध्रुव int8_t keven_signs_q2xs[1024] = {
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
#यदि_अन्तः

शून्य ggml_vec_dot_iq2_xxs_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_iq2_xxs * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K    * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);
    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    ध्रुव uint64_t * signs64 = (ध्रुव uint64_t *)keven_signs_q2xs;

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;

        ध्रुव uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ध्रुव int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            ध्रुव uint8_t * aux8 = (ध्रुव uint8_t *)aux32;

            स्मृतिं_प्रतिलिख(aux32, q2, 4*परिमाण(uint32_t));
            q2 += 8;

            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_0 = {*(ध्रुव int64_t *)(iq2xxs_grid + aux8[ 0]), *(ध्रुव int64_t *)(iq2xxs_grid + aux8[ 1])};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_1 = {*(ध्रुव int64_t *)(iq2xxs_grid + aux8[ 2]), *(ध्रुव int64_t *)(iq2xxs_grid + aux8[ 3])};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_2 = {*(ध्रुव int64_t *)(iq2xxs_grid + aux8[ 8]), *(ध्रुव int64_t *)(iq2xxs_grid + aux8[ 9])};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_3 = {*(ध्रुव int64_t *)(iq2xxs_grid + aux8[10]), *(ध्रुव int64_t *)(iq2xxs_grid + aux8[11])};

            सदिशम् चिह्नित दीर्घ दीर्घ vsigns0 = {*(ध्रुव int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(ध्रुव int64_t *)(signs64 + ((aux32[1] >>  7) & 127))};
            सदिशम् चिह्नित दीर्घ दीर्घ vsigns1 = {*(ध्रुव int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(ध्रुव int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            सदिशम् चिह्नित दीर्घ दीर्घ vsigns2 = {*(ध्रुव int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(ध्रुव int64_t *)(signs64 + ((aux32[3] >>  7) & 127))};
            सदिशम् चिह्नित दीर्घ दीर्घ vsigns3 = {*(ध्रुव int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(ध्रुव int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            सदिशम् चिह्नित वर्ण q2x0 = (सदिशम् चिह्नित वर्ण)vec_mul((सदिशम् चिह्नित वर्ण)vsigns0, (सदिशम् चिह्नित वर्ण)aux64x2_0);
            सदिशम् चिह्नित वर्ण q2x1 = (सदिशम् चिह्नित वर्ण)vec_mul((सदिशम् चिह्नित वर्ण)vsigns1, (सदिशम् चिह्नित वर्ण)aux64x2_1);
            सदिशम् चिह्नित वर्ण q2x2 = (सदिशम् चिह्नित वर्ण)vec_mul((सदिशम् चिह्नित वर्ण)vsigns2, (सदिशम् चिह्नित वर्ण)aux64x2_2);
            सदिशम् चिह्नित वर्ण q2x3 = (सदिशम् चिह्नित वर्ण)vec_mul((सदिशम् चिह्नित वर्ण)vsigns3, (सदिशम् चिह्नित वर्ण)aux64x2_3);

            सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, q8);
            सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, q8);
            सदिशम् चिह्नित वर्ण q8y2 = vec_xl(32, q8);
            सदिशम् चिह्नित वर्ण q8y3 = vec_xl(48, q8);
            q8 += 64;

            सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            सदिशम् चिह्नित ह्रस्व qv2 = vec_सङ्कलय(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            सदिशम् चिह्नित ह्रस्व qv3 = vec_सङ्कलय(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ध्रुव uint16_t ls0 = aux32[1] >> 28;
            ध्रुव uint16_t ls1 = aux32[3] >> 28;

            सदिशम् चिह्नित ह्रस्व vscales01 = vec_splats((int16_t)(2*ls0+1));
            सदिशम् चिह्नित ह्रस्व vscales23 = vec_splats((int16_t)(2*ls1+1));

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

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_iq2_xs_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_iq2_xs * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K   * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);
    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    ध्रुव uint64_t * signs64 = (ध्रुव uint64_t *)keven_signs_q2xs;

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;

        ध्रुव uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ध्रुव uint8_t  * GGML_RESTRICT sc = x[i].scales;
        ध्रुव int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_0 = {*(ध्रुव int64_t *)(iq2xs_grid + (q2[0] & 511)), *(ध्रुव int64_t *)(iq2xs_grid + (q2[1] & 511))};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_1 = {*(ध्रुव int64_t *)(iq2xs_grid + (q2[2] & 511)), *(ध्रुव int64_t *)(iq2xs_grid + (q2[3] & 511))};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_2 = {*(ध्रुव int64_t *)(iq2xs_grid + (q2[4] & 511)), *(ध्रुव int64_t *)(iq2xs_grid + (q2[5] & 511))};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_3 = {*(ध्रुव int64_t *)(iq2xs_grid + (q2[6] & 511)), *(ध्रुव int64_t *)(iq2xs_grid + (q2[7] & 511))};

            सदिशम् चिह्नित दीर्घ दीर्घ vsigns0 = {*(ध्रुव int64_t *)(signs64 + ((q2[0] >> 9))), *(ध्रुव int64_t *)(signs64 + ((q2[1] >> 9)))};
            सदिशम् चिह्नित दीर्घ दीर्घ vsigns1 = {*(ध्रुव int64_t *)(signs64 + ((q2[2] >> 9))), *(ध्रुव int64_t *)(signs64 + ((q2[3] >> 9)))};
            सदिशम् चिह्नित दीर्घ दीर्घ vsigns2 = {*(ध्रुव int64_t *)(signs64 + ((q2[4] >> 9))), *(ध्रुव int64_t *)(signs64 + ((q2[5] >> 9)))};
            सदिशम् चिह्नित दीर्घ दीर्घ vsigns3 = {*(ध्रुव int64_t *)(signs64 + ((q2[6] >> 9))), *(ध्रुव int64_t *)(signs64 + ((q2[7] >> 9)))};
            q2 += 8;

            सदिशम् चिह्नित वर्ण q2x0 = (सदिशम् चिह्नित वर्ण)vec_mul((सदिशम् चिह्नित वर्ण)vsigns0, (सदिशम् चिह्नित वर्ण)aux64x2_0);
            सदिशम् चिह्नित वर्ण q2x1 = (सदिशम् चिह्नित वर्ण)vec_mul((सदिशम् चिह्नित वर्ण)vsigns1, (सदिशम् चिह्नित वर्ण)aux64x2_1);
            सदिशम् चिह्नित वर्ण q2x2 = (सदिशम् चिह्नित वर्ण)vec_mul((सदिशम् चिह्नित वर्ण)vsigns2, (सदिशम् चिह्नित वर्ण)aux64x2_2);
            सदिशम् चिह्नित वर्ण q2x3 = (सदिशम् चिह्नित वर्ण)vec_mul((सदिशम् चिह्नित वर्ण)vsigns3, (सदिशम् चिह्नित वर्ण)aux64x2_3);

            सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, q8);
            सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, q8);
            सदिशम् चिह्नित वर्ण q8y2 = vec_xl(32, q8);
            सदिशम् चिह्नित वर्ण q8y3 = vec_xl(48, q8);
            q8 += 64;

            सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            सदिशम् चिह्नित ह्रस्व qv2 = vec_सङ्कलय(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            सदिशम् चिह्नित ह्रस्व qv3 = vec_सङ्कलय(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ध्रुव uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ध्रुव uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ध्रुव uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ध्रुव uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            सदिशम् चिह्नित ह्रस्व vscales0 = vec_splats((int16_t)(2*ls0+1));
            सदिशम् चिह्नित ह्रस्व vscales1 = vec_splats((int16_t)(2*ls1+1));
            सदिशम् चिह्नित ह्रस्व vscales2 = vec_splats((int16_t)(2*ls2+1));
            सदिशम् चिह्नित ह्रस्व vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_iq2_s_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_iq2_s * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K  * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    स्थिर ध्रुव uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    स्थिर ध्रुव uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    ध्रुव सदिशम् अचिह्नित वर्ण mask0 = vec_xl( 0, k_mask1);
    ध्रुव सदिशम् अचिह्नित वर्ण mask1 = vec_xl(16, k_mask1);
    ध्रुव सदिशम् चिह्नित वर्ण mask2 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, k_mask2);

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;

        ध्रुव uint8_t *  GGML_RESTRICT q2 = x[i].qs;
        ध्रुव uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ध्रुव uint16_t * GGML_RESTRICT signs = (ध्रुव uint16_t *)(x[i].qs + QK_K/8);
        ध्रुव uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ध्रुव int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_0 = {*(ध्रुव int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(ध्रुव int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_1 = {*(ध्रुव int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(ध्रुव int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_2 = {*(ध्रुव int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(ध्रुव int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_3 = {*(ध्रुव int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(ध्रुव int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            सदिशम् चिह्नित वर्ण vsigns01 = (सदिशम् चिह्नित वर्ण)vec_splats(*(ध्रुव uint32_t *)&signs[0]);
            सदिशम् चिह्नित वर्ण vsigns23 = (सदिशम् चिह्नित वर्ण)vec_splats(*(ध्रुव uint32_t *)&signs[2]);
            signs += 4;

            सदिशम् चिह्नित वर्ण vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            सदिशम् चिह्नित वर्ण vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            सदिशम् चिह्नित वर्ण vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            सदिशम् चिह्नित वर्ण vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (सदिशम् चिह्नित वर्ण)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (सदिशम् चिह्नित वर्ण)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (सदिशम् चिह्नित वर्ण)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (सदिशम् चिह्नित वर्ण)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            सदिशम् चिह्नित वर्ण q2x0 = vec_sub(vec_xor(vsigns0, (सदिशम् चिह्नित वर्ण)aux64x2_0), vsigns0);
            सदिशम् चिह्नित वर्ण q2x1 = vec_sub(vec_xor(vsigns1, (सदिशम् चिह्नित वर्ण)aux64x2_1), vsigns1);
            सदिशम् चिह्नित वर्ण q2x2 = vec_sub(vec_xor(vsigns2, (सदिशम् चिह्नित वर्ण)aux64x2_2), vsigns2);
            सदिशम् चिह्नित वर्ण q2x3 = vec_sub(vec_xor(vsigns3, (सदिशम् चिह्नित वर्ण)aux64x2_3), vsigns3);

            सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, q8);
            सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, q8);
            सदिशम् चिह्नित वर्ण q8y2 = vec_xl(32, q8);
            सदिशम् चिह्नित वर्ण q8y3 = vec_xl(48, q8);
            q8 += 64;

            सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            सदिशम् चिह्नित ह्रस्व qv2 = vec_सङ्कलय(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            सदिशम् चिह्नित ह्रस्व qv3 = vec_सङ्कलय(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ध्रुव uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ध्रुव uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ध्रुव uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ध्रुव uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            सदिशम् चिह्नित ह्रस्व vscales0 = vec_splats((int16_t)(2*ls0+1));
            सदिशम् चिह्नित ह्रस्व vscales1 = vec_splats((int16_t)(2*ls1+1));
            सदिशम् चिह्नित ह्रस्व vscales2 = vec_splats((int16_t)(2*ls2+1));
            सदिशम् चिह्नित ह्रस्व vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_iq3_xxs_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_iq3_xxs * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K    * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव uint64_t * signs64 = (ध्रुव uint64_t *)keven_signs_q2xs;

    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;

        ध्रुव uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ध्रुव uint32_t * GGML_RESTRICT signs = (ध्रुव uint32_t *)(x[i].qs + QK_K/4);
        ध्रुव int8_t  * GGML_RESTRICT q8 = y[i].qs;

#निर्देशः GCC unroll 1
        प्रति (पूर्णाङ्क j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् अचिह्नित पूर्णाङ्क aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            सदिशम् अचिह्नित पूर्णाङ्क aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            सदिशम् अचिह्नित पूर्णाङ्क aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            सदिशम् अचिह्नित पूर्णाङ्क aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            सदिशम् अचिह्नित दीर्घ दीर्घ aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            सदिशम् अचिह्नित दीर्घ दीर्घ aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            सदिशम् अचिह्नित दीर्घ दीर्घ aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            सदिशम् अचिह्नित दीर्घ दीर्घ aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            सदिशम् चिह्नित वर्ण q3x0 = vec_mul((सदिशम् चिह्नित वर्ण)aux64x2_0, (सदिशम् चिह्नित वर्ण)aux32x4_0);
            सदिशम् चिह्नित वर्ण q3x1 = vec_mul((सदिशम् चिह्नित वर्ण)aux64x2_1, (सदिशम् चिह्नित वर्ण)aux32x4_1);
            सदिशम् चिह्नित वर्ण q3x2 = vec_mul((सदिशम् चिह्नित वर्ण)aux64x2_2, (सदिशम् चिह्नित वर्ण)aux32x4_2);
            सदिशम् चिह्नित वर्ण q3x3 = vec_mul((सदिशम् चिह्नित वर्ण)aux64x2_3, (सदिशम् चिह्नित वर्ण)aux32x4_3);

            सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, q8);
            सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, q8);
            सदिशम् चिह्नित वर्ण q8y2 = vec_xl(32, q8);
            सदिशम् चिह्नित वर्ण q8y3 = vec_xl(48, q8);
            q8 += 64;

            सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            सदिशम् चिह्नित ह्रस्व qv2 = vec_सङ्कलय(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            सदिशम् चिह्नित ह्रस्व qv3 = vec_सङ्कलय(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ध्रुव uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            ध्रुव uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            सदिशम् चिह्नित ह्रस्व vscales01 = (सदिशम् चिह्नित ह्रस्व)vec_splats((uint16_t)(2*ls0+1));
            सदिशम् चिह्नित ह्रस्व vscales23 = (सदिशम् चिह्नित ह्रस्व)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.25f * vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_iq3_s_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_iq3_s * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K  * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    स्थिर ध्रुव uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    स्थिर ध्रुव uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    ध्रुव सदिशम् अचिह्नित वर्ण mask0 = vec_xl( 0, k_mask1);
    ध्रुव सदिशम् अचिह्नित वर्ण mask1 = vec_xl(16, k_mask1);
    ध्रुव सदिशम् चिह्नित वर्ण mask2 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, k_mask2);

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        ध्रुव uint8_t *  GGML_RESTRICT q3 = x[i].qs;
        ध्रुव uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ध्रुव uint16_t * GGML_RESTRICT signs = (ध्रुव uint16_t *)(x[i].signs);
        ध्रुव uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ध्रुव int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् अचिह्नित पूर्णाङ्क aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            सदिशम् अचिह्नित पूर्णाङ्क aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            सदिशम् अचिह्नित पूर्णाङ्क aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            सदिशम् अचिह्नित पूर्णाङ्क aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            सदिशम् चिह्नित वर्ण vsigns01 = (सदिशम् चिह्नित वर्ण)vec_splats(*(ध्रुव uint32_t *)&signs[0]);
            सदिशम् चिह्नित वर्ण vsigns02 = (सदिशम् चिह्नित वर्ण)vec_splats(*(ध्रुव uint32_t *)&signs[2]);
            signs += 4;

            सदिशम् चिह्नित वर्ण vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            सदिशम् चिह्नित वर्ण vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            सदिशम् चिह्नित वर्ण vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            सदिशम् चिह्नित वर्ण vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (सदिशम् चिह्नित वर्ण)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (सदिशम् चिह्नित वर्ण)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (सदिशम् चिह्नित वर्ण)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (सदिशम् चिह्नित वर्ण)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            सदिशम् चिह्नित वर्ण q3x0 = vec_sub(vec_xor(vsigns0, (सदिशम् चिह्नित वर्ण)aux32x4_0), vsigns0);
            सदिशम् चिह्नित वर्ण q3x1 = vec_sub(vec_xor(vsigns1, (सदिशम् चिह्नित वर्ण)aux32x4_1), vsigns1);
            सदिशम् चिह्नित वर्ण q3x2 = vec_sub(vec_xor(vsigns2, (सदिशम् चिह्नित वर्ण)aux32x4_2), vsigns2);
            सदिशम् चिह्नित वर्ण q3x3 = vec_sub(vec_xor(vsigns3, (सदिशम् चिह्नित वर्ण)aux32x4_3), vsigns3);

            सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, q8);
            सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, q8);
            सदिशम् चिह्नित वर्ण q8y2 = vec_xl(32, q8);
            सदिशम् चिह्नित वर्ण q8y3 = vec_xl(48, q8);
            q8 += 64;

            सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            सदिशम् चिह्नित ह्रस्व qv2 = vec_सङ्कलय(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            सदिशम् चिह्नित ह्रस्व qv3 = vec_सङ्कलय(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ध्रुव uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ध्रुव uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            सदिशम् चिह्नित ह्रस्व vscales01 = (सदिशम् चिह्नित ह्रस्व)vec_splats((uint16_t)(2*ls0+1));
            सदिशम् चिह्नित ह्रस्व vscales23 = (सदिशम् चिह्नित ह्रस्व)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_iq1_s_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(n % QK_K == 0);
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);

    ध्रुव खण्डम्_iq1_s * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K  * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् अचिह्नित वर्ण v0 = vec_splats((अचिह्नित वर्ण)0x0);
    ध्रुव सदिशम् अचिह्नित ह्रस्व vsign = vec_splats((अचिह्नित ह्रस्व)0x8000);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    प्रति (पूर्णाङ्क i = 0; i < nb; ++i) {
        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[i].d));
        सदिशम् भिन्न vyd = vec_splats(y[i].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = vec_splats((int32_t)0);
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = vec_splats((int32_t)0);
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = vec_splats((int32_t)0);
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = vec_splats((int32_t)0);
        सदिशम् चिह्नित पूर्णाङ्क vsumi8 = vec_splats((int32_t)0);

        ध्रुव uint8_t  * GGML_RESTRICT q1 = x[i].qs;
        ध्रुव uint16_t * GGML_RESTRICT qh = x[i].qh;
        ध्रुव int8_t   * GGML_RESTRICT q8 = y[i].qs;
        ध्रुव int16_t  * GGML_RESTRICT qs = y[i].bsums;

        प्रति (पूर्णाङ्क j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_0 = {*(ध्रुव int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(ध्रुव int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_1 = {*(ध्रुव int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(ध्रुव int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_2 = {*(ध्रुव int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(ध्रुव int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            सदिशम् चिह्नित दीर्घ दीर्घ aux64x2_3 = {*(ध्रुव int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(ध्रुव int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            सदिशम् चिह्नित वर्ण q1x0 = (सदिशम् चिह्नित वर्ण)aux64x2_0;
            सदिशम् चिह्नित वर्ण q1x1 = (सदिशम् चिह्नित वर्ण)aux64x2_1;
            सदिशम् चिह्नित वर्ण q1x2 = (सदिशम् चिह्नित वर्ण)aux64x2_2;
            सदिशम् चिह्नित वर्ण q1x3 = (सदिशम् चिह्नित वर्ण)aux64x2_3;

            सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, q8);
            सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, q8);
            सदिशम् चिह्नित वर्ण q8y2 = vec_xl(32, q8);
            सदिशम् चिह्नित वर्ण q8y3 = vec_xl(48, q8);
            q8 += 64;

            सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            सदिशम् चिह्नित ह्रस्व qv2 = vec_सङ्कलय(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            सदिशम् चिह्नित ह्रस्व qv3 = vec_सङ्कलय(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            ध्रुव uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            ध्रुव uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            सदिशम् चिह्नित ह्रस्व vscales01 = (सदिशम् चिह्नित ह्रस्व)vec_splats((uint16_t)(2*ls0+1));
            सदिशम् चिह्नित ह्रस्व vscales23 = (सदिशम् चिह्नित ह्रस्व)vec_splats((uint16_t)(2*ls1+1));
            सदिशम् चिह्नित ह्रस्व vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            सदिशम् चिह्नित ह्रस्व q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (सदिशम् चिह्नित ह्रस्व)v0);

            सदिशम् चिह्नित ह्रस्व qxh = (सदिशम् चिह्नित ह्रस्व)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            सदिशम् __bool ह्रस्व vsel = vec_cmpge(qxh, (सदिशम् चिह्नित ह्रस्व)v0);

            सदिशम् चिह्नित ह्रस्व q8ysum = vec_sel((सदिशम् चिह्नित ह्रस्व)vec_xor((सदिशम् अचिह्नित ह्रस्व)q8ysums, vsign), q8ysums, vsel);

            vsumi8 = vec_सङ्कलय(vec_mule(q8ysum, vscales), vsumi8);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);

        vsumf0 = vec_madd(vec_ctf(vsumi8, 0), vec_mul(vd, vec_splats(IQ1S_DELTA)), vsumf0);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_iq4_nl_q8_0(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);
    प्रतिज्ञा(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ध्रुव खण्डम्_iq4_nl * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_0   * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK4_NL;

    पूर्णाङ्क ib = 0;
    भिन्न sumf = 0;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् चिह्नित पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);

    ध्रुव सदिशम् चिह्नित वर्ण मानानि = vec_xl( 0, kvalues_iq4nl);

#निर्देशः GCC unroll 4
    प्रति (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[ib].d));
        सदिशम् भिन्न vyd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(y[ib].d));
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित वर्ण qxs = (सदिशम् चिह्नित वर्ण)vec_xl( 0, x[ib].qs);
        सदिशम् चिह्नित वर्ण q4x0 = vec_and(qxs, नीचः_आवरणम्);
        सदिशम् चिह्नित वर्ण q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(मानानि, मानानि, (सदिशम् अचिह्नित वर्ण)q4x0);
        q4x1 = vec_perm(मानानि, मानानि, (सदिशम् अचिह्नित वर्ण)q4x1);

        सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, y[ib].qs);
        सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, y[ib].qs);

        सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    }

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    अप्रयुक्तम्(ib);
    अप्रयुक्तम्(sumf);
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}

शून्य ggml_vec_dot_iq4_xs_q8_K(पूर्णाङ्क n, भिन्न * GGML_RESTRICT s, size_t bs, ध्रुव शून्य * GGML_RESTRICT vx, size_t bx, ध्रुव शून्य * GGML_RESTRICT vy, size_t by, पूर्णाङ्क nrc) {
    प्रतिज्ञा(nrc == 1);
    अप्रयुक्तम्(nrc);
    अप्रयुक्तम्(bx);
    अप्रयुक्तम्(by);
    अप्रयुक्तम्(bs);
    प्रतिज्ञा(n % QK_K == 0);

    ध्रुव खण्डम्_iq4_xs * GGML_RESTRICT x = vx;
    ध्रुव खण्डम्_q8_K   * GGML_RESTRICT y = vy;

    ध्रुव पूर्णाङ्क nb = n / QK_K;

#यदि defined(__POWER9_VECTOR__)
    ध्रुव सदिशम् चिह्नित वर्ण नीचः_आवरणम् = vec_splats((चिह्नित वर्ण)0xF);
    ध्रुव सदिशम् पूर्णाङ्क v0 = vec_splats((int32_t)0);
    ध्रुव सदिशम् अचिह्नित वर्ण v4 = vec_splats((अचिह्नित वर्ण)0x4);

    सदिशम् भिन्न vsumf0 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf1 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf2 = vec_splats(0.0f);
    सदिशम् भिन्न vsumf3 = vec_splats(0.0f);

    ध्रुव सदिशम् चिह्नित वर्ण मानानि = vec_xl( 0, kvalues_iq4nl);

    प्रति (पूर्णाङ्क ibl = 0; ibl < nb; ++ibl) {

        सदिशम् भिन्न vxd = vec_splats(GGML_CPU_FP16_गन्तव्यम्_FP32(x[ibl].d));
        सदिशम् भिन्न vyd = vec_splats(y[ibl].d);
        सदिशम् भिन्न vd = vec_mul(vxd, vyd);

        सदिशम् चिह्नित पूर्णाङ्क vsumi0 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi1 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi2 = v0;
        सदिशम् चिह्नित पूर्णाङ्क vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        ध्रुव uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        ध्रुव uint8_t * GGML_RESTRICT sc = x[ibl].scales_l;
        ध्रुव int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        प्रति (पूर्णाङ्क ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            सदिशम् चिह्नित वर्ण qxs0 = (सदिशम् चिह्नित वर्ण)vec_xl( 0, q4);
            सदिशम् चिह्नित वर्ण qxs1 = (सदिशम् चिह्नित वर्ण)vec_xl(16, q4);
            q4 += 32;

            सदिशम् चिह्नित वर्ण q4x00 = (सदिशम् चिह्नित वर्ण)vec_and(qxs0, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण q4x01 = (सदिशम् चिह्नित वर्ण)vec_sr(qxs0, v4);
            सदिशम् चिह्नित वर्ण q4x10 = (सदिशम् चिह्नित वर्ण)vec_and(qxs1, नीचः_आवरणम्);
            सदिशम् चिह्नित वर्ण q4x11 = (सदिशम् चिह्नित वर्ण)vec_sr(qxs1, v4);

            q4x00 = vec_perm(मानानि, मानानि, (सदिशम् अचिह्नित वर्ण)q4x00);
            q4x01 = vec_perm(मानानि, मानानि, (सदिशम् अचिह्नित वर्ण)q4x01);
            q4x10 = vec_perm(मानानि, मानानि, (सदिशम् अचिह्नित वर्ण)q4x10);
            q4x11 = vec_perm(मानानि, मानानि, (सदिशम् अचिह्नित वर्ण)q4x11);

            सदिशम् चिह्नित वर्ण q8y0 = vec_xl( 0, q8);
            सदिशम् चिह्नित वर्ण q8y1 = vec_xl(16, q8);
            सदिशम् चिह्नित वर्ण q8y2 = vec_xl(32, q8);
            सदिशम् चिह्नित वर्ण q8y3 = vec_xl(48, q8);
            q8 += 64;

            सदिशम् चिह्नित ह्रस्व qv0 = vec_सङ्कलय(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            सदिशम् चिह्नित ह्रस्व qv1 = vec_सङ्कलय(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            सदिशम् चिह्नित ह्रस्व qv2 = vec_सङ्कलय(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            सदिशम् चिह्नित ह्रस्व qv3 = vec_सङ्कलय(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            ध्रुव uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            ध्रुव uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            सदिशम् चिह्नित ह्रस्व vscales01 = vec_splats((int16_t)ls0);
            सदिशम् चिह्नित ह्रस्व vscales23 = vec_splats((int16_t)ls1);

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

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf2);
    vsumf1 = vec_सङ्कलय(vsumf1, vsumf3);

    vsumf0 = vec_सङ्कलय(vsumf0, vsumf1);

    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_सङ्कलय(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    अप्रयुक्तम्(x);
    अप्रयुक्तम्(y);
    अप्रयुक्तम्(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#यदि_अन्तः
}
