#घोषणा GGML_COMMON_IMPL_C
#समावेश "ggml-common.h"
#समावेश "ggml-quants.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-cpu.h"
#समावेश "simd-mappings.h"

#समावेश "../../quants.h"
#समावेश "../../ggml-cpu-impl.h"

#समावेश <math.h>
#समावेश <string.h>
#समावेश <assert.h>
#समावेश <float.h>
#समावेश <stdlib.h> // for qsort
#समावेश <stdio.h>  // for GGML_ASSERT

#घोषणा GROUP_MAX_EPS 1e-15f
#घोषणा GROUP_MAX_EPS_IQ3_XXS 1e-8f
#घोषणा GROUP_MAX_EPS_IQ2_S 1e-8f
#घोषणा GROUP_MAX_EPS_IQ1_M 1e-7f
#घोषणा GROUP_MAX_EPS_IQ1_S 1e-12f

#घोषणा UNUSED GGML_UNUSED

#अगर defined(__POWER9_VECTOR__)
#घोषणा B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#घोषणा B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#घोषणा B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#घोषणा B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#घोषणा B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#घोषणा B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#घोषणा B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#घोषणा B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
अटल स्थिर uint64_t table_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
अटल स्थिर uint64_t table_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#पूर्ण_अगर

व्योम quantize_row_q8_0(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t k) {
    निश्चित(QK8_0 == 32);
    निश्चित(k % QK8_0 == 0);
    स्थिर पूर्णांक nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#अगर defined(__POWER9_VECTOR__)
    क्रम (पूर्णांक i = 0; i < nb; i++) {
        vector भग्न srcv [8];
        vector भग्न asrcv[8];
        vector भग्न amaxv[8];
        vector चिन्हित पूर्णांक vi[8];

        क्रम (पूर्णांक j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        क्रम (पूर्णांक j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        क्रम (पूर्णांक j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        क्रम (पूर्णांक j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        क्रम (पूर्णांक j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        स्थिर भग्न amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        स्थिर भग्न d = amax / ((1 << 7) - 1);
        स्थिर भग्न id = d ? 1.0f/d : 0.0f;
        स्थिर vector भग्न vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        क्रम (पूर्णांक j = 0; j < 8; j++) {
            स्थिर vector भग्न v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }
#अन्यथा
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#पूर्ण_अगर
}

व्योम quantize_row_q8_1(स्थिर भग्न * GGML_RESTRICT x, व्योम * GGML_RESTRICT vy, int64_t k) {
    निश्चित(k % QK8_1 == 0);
    स्थिर पूर्णांक nb = k / QK8_1;

    block_q8_1 * GGML_RESTRICT y = vy;

#अगर defined(__POWER9_VECTOR__)
    क्रम (पूर्णांक i = 0; i < nb; i++) {
        vector भग्न srcv [8];
        vector भग्न asrcv[8];
        vector भग्न amaxv[8];
        vector चिन्हित पूर्णांक vi[8];

        क्रम (पूर्णांक j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        क्रम (पूर्णांक j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        क्रम (पूर्णांक j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        क्रम (पूर्णांक j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        क्रम (पूर्णांक j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        स्थिर भग्न amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        स्थिर भग्न d = amax / ((1 << 7) - 1);
        स्थिर भग्न id = d ? 1.0f/d : 0.0f;
        स्थिर vector भग्न vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        vector पूर्णांक accv = vec_splats(0);

        क्रम (पूर्णांक j = 0; j < 8; j++) {
            स्थिर vector भग्न v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_add(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_add(accv, vec_sld(accv, accv, 4));
        accv = vec_add(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_CPU_FP32_TO_FP16(d * vec_extract(accv, 0));
    }

#अन्यथा
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#पूर्ण_अगर
}


//===================================== Dot products =================================

व्योम ggml_vec_dot_q4_0_q8_0(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;

    निश्चित(n % qk == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_q4_0 * GGML_RESTRICT x = vx;
    स्थिर block_q8_0 * GGML_RESTRICT y = vy;

    पूर्णांक ib = 0;
    भग्न sumf = 0;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector चिन्हित पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);
    स्थिर vector चिन्हित अक्षर v8 = vec_splats((चिन्हित अक्षर)0x8);

    vector भग्न vsumf0 = vec_splats(0.0f);

#आशय GCC unroll 8
    क्रम (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector भग्न vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित अक्षर qxs = (vector चिन्हित अक्षर)vec_xl( 0, x[ib].qs);
        vector चिन्हित अक्षर q8y0 = vec_xl( 0, y[ib].qs);
        vector चिन्हित अक्षर q8y1 = vec_xl(16, y[ib].qs);

        vector चिन्हित अक्षर q4x0 = vec_and(qxs, निचला_Mask);
        vector चिन्हित अक्षर q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        vector चिन्हित लघु qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector चिन्हित लघु qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector चिन्हित पूर्णांक vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_q4_1_q8_1(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिर पूर्णांक qk = QK8_1;
    स्थिर पूर्णांक nb = n / qk;

    निश्चित(n % qk == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_q4_1 * GGML_RESTRICT x = vx;
    स्थिर block_q8_1 * GGML_RESTRICT y = vy;

    पूर्णांक ib = 0;
    भग्न sumf = 0;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector चिन्हित पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);

    vector भग्न vsumf0 = vec_splats(0.0f);

#आशय GCC unroll 4
    क्रम (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector भग्न vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector भग्न vd = vec_mul(vxd, vyd);

        vector भग्न vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].m));
        vector भग्न vys = {GGML_CPU_FP16_TO_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        vector चिन्हित अक्षर qxs = (vector चिन्हित अक्षर)vec_xl( 0, x[ib].qs);
        vector चिन्हित अक्षर q8y0 = vec_xl( 0, y[ib].qs);
        vector चिन्हित अक्षर q8y1 = vec_xl(16, y[ib].qs);

        vector अचिन्हित अक्षर q4x0 = (vector अचिन्हित अक्षर)vec_and(qxs, निचला_Mask);
        vector अचिन्हित अक्षर q4x1 = (vector अचिन्हित अक्षर)vec_sr(qxs, v4);

        vector चिन्हित पूर्णांक vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_mxfp4_q8_0(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    निश्चित(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    स्थिर block_mxfp4 * GGML_RESTRICT x = vx;
    स्थिर block_q8_0 * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_MXFP4;

    पूर्णांक ib = 0;
    भग्न sumf = 0;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector अचिन्हित अक्षर vshift4 = vec_splats((अचिन्हित अक्षर)4);
    vector भग्न vsumf0 = vec_splats(0.0f);

    vector चिन्हित अक्षर kv = vec_xl(0, (स्थिर चिन्हित अक्षर *)kvalues_mxfp4);

#आशय GCC unroll 8
    क्रम (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector भग्न vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d) *
                                      GGML_E8M0_TO_FP32_HALF(x[ib].e));

        vector चिन्हित अक्षर q8y0 = vec_xl( 0, y[ib].qs);
        vector चिन्हित अक्षर q8y1 = vec_xl(16, y[ib].qs);

        vector चिन्हित अक्षर qxs = (vector चिन्हित अक्षर)vec_xl(0, x[ib].qs);

        vector अचिन्हित अक्षर lo_nibbles = (vector अचिन्हित अक्षर)vec_and(qxs, निचला_Mask);
        vector अचिन्हित अक्षर hi_nibbles = (vector अचिन्हित अक्षर)vec_sr(qxs, vshift4);

        vector चिन्हित अक्षर q4x0 = vec_perm(kv, kv, lo_nibbles);
        vector चिन्हित अक्षर q4x1 = vec_perm(kv, kv, hi_nibbles);

        vector चिन्हित लघु qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector चिन्हित लघु qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector चिन्हित पूर्णांक vsumi0 = vec_splats((int32_t)0);
        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vyd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));
    sumf = vec_extract(vsumf0, 0);
    *s = sumf;
#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_mxfp4_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_q5_0_q8_0(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;

    पूर्णांक ib = 0;
    भग्न sumf = 0;

    निश्चित(n % qk == 0);
    निश्चित(qk == QK5_0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_q5_0 * GGML_RESTRICT x = vx;
    स्थिर block_q8_0 * GGML_RESTRICT y = vy;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)4);

    vector भग्न vsumf0 = vec_splats(0.0f);

#आशय GCC unroll 4
    क्रम (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector भग्न vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित दीर्घ दीर्घ aux64x2_0 = {(uint64_t)(table_b2b_1[x[ib].qh[0]]), (uint64_t)(table_b2b_1[x[ib].qh[1]])};
        vector चिन्हित दीर्घ दीर्घ aux64x2_1 = {(uint64_t)(table_b2b_1[x[ib].qh[2]]), (uint64_t)(table_b2b_1[x[ib].qh[3]])};

        vector चिन्हित अक्षर qh0 = (vector चिन्हित अक्षर)aux64x2_0;
        vector चिन्हित अक्षर qh1 = (vector चिन्हित अक्षर)aux64x2_1;

        vector चिन्हित अक्षर qxs = (vector चिन्हित अक्षर)vec_xl( 0, x[ib].qs);

        vector चिन्हित अक्षर q5x0 = vec_sub(vec_and (qxs, निचला_Mask), qh0);
        vector चिन्हित अक्षर q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

        vector चिन्हित अक्षर q8y0 = vec_xl(  0, y[ib].qs);
        vector चिन्हित अक्षर q8y1 = vec_xl( 16, y[ib].qs);

        vector चिन्हित लघु qv0 = vec_add(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
        vector चिन्हित लघु qv1 = vec_add(vec_mule(q5x1, q8y1), vec_mulo(q5x1, q8y1));

        qv0 = vec_add(qv0, qv1);

        vector चिन्हित पूर्णांक vsumi0 = vec_add(vec_unpackh(qv0), vec_unpackl(qv0));

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    UNUSED(ib);
    UNUSED(sumf);
    UNUSED(x);
    UNUSED(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_q5_1_q8_1(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिर पूर्णांक qk = QK8_1;
    स्थिर पूर्णांक nb = n / qk;

    पूर्णांक ib = 0;
    भग्न sumf = 0;

    निश्चित(n % qk == 0);
    निश्चित(qk == QK5_1);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_q5_1 * GGML_RESTRICT x = vx;
    स्थिर block_q8_1 * GGML_RESTRICT y = vy;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector चिन्हित पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);

    vector भग्न vsumf0 = vec_splats(0.0f);

#आशय GCC unroll 4
    क्रम (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector भग्न vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector भग्न vd = vec_mul(vxd, vyd);

        vector भग्न vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].m));
        vector भग्न vys = {GGML_CPU_FP16_TO_FP32(y[ib].s), 0.f, 0.f, 0.f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        vector अचिन्हित दीर्घ दीर्घ aux64x2_0 = {(uint64_t)(table_b2b_0[x[ib].qh[0]]), (uint64_t)(table_b2b_0[x[ib].qh[1]])};
        vector अचिन्हित दीर्घ दीर्घ aux64x2_1 = {(uint64_t)(table_b2b_0[x[ib].qh[2]]), (uint64_t)(table_b2b_0[x[ib].qh[3]])};

        vector चिन्हित अक्षर qh0 = (vector चिन्हित अक्षर)aux64x2_0;
        vector चिन्हित अक्षर qh1 = (vector चिन्हित अक्षर)aux64x2_1;

        vector चिन्हित अक्षर qxs = (vector चिन्हित अक्षर)vec_xl( 0, x[ib].qs);

        vector अचिन्हित अक्षर q5x0 = (vector अचिन्हित अक्षर)vec_or(vec_and(qxs, निचला_Mask), qh0);
        vector अचिन्हित अक्षर q5x1 = (vector अचिन्हित अक्षर)vec_or(vec_sr(qxs, v4), qh1);

        vector चिन्हित अक्षर q8y0 = vec_xl(  0, y[ib].qs);
        vector चिन्हित अक्षर q8y1 = vec_xl( 16, y[ib].qs);

        vector चिन्हित पूर्णांक vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q5x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q5x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    UNUSED(nb);
    UNUSED(ib);
    UNUSED(sumf);
    UNUSED(x);
    UNUSED(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_q8_0_q8_0(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    स्थिर पूर्णांक qk = QK8_0;
    स्थिर पूर्णांक nb = n / qk;

    निश्चित(n % qk == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_q8_0 * GGML_RESTRICT x = vx;
    स्थिर block_q8_0 * GGML_RESTRICT y = vy;

    पूर्णांक ib = 0;
    भग्न sumf = 0;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित पूर्णांक v0 = vec_splats((int32_t)0);
    vector भग्न vsumf0 = vec_splats(0.0f);

#आशय GCC unroll 8
    क्रम (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector भग्न vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित अक्षर q8x0 = vec_xl( 0, x[ib].qs);
        vector चिन्हित अक्षर q8x1 = vec_xl(16, x[ib].qs);
        vector चिन्हित अक्षर q8y0 = vec_xl( 0, y[ib].qs);
        vector चिन्हित अक्षर q8y1 = vec_xl(16, y[ib].qs);

        vector चिन्हित लघु qv0 = vec_mule(q8x0, q8y0);
        vector चिन्हित लघु qv1 = vec_mulo(q8x0, q8y0);
        vector चिन्हित लघु qv2 = vec_mule(q8x1, q8y1);
        vector चिन्हित लघु qv3 = vec_mulo(q8x1, q8y1);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_add(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_q2_K_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_q2_K * GGML_RESTRICT x = vx;
    स्थिर block_q8_K * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0x3);
    स्थिर vector चिन्हित अक्षर निचला_Scale_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector अचिन्हित अक्षर v2 = vec_splats((अचिन्हित अक्षर)0x2);
    स्थिर vector अचिन्हित अक्षर v6 = vec_splats((अचिन्हित अक्षर)0x6);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector भग्न vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].dmin));
        vector भग्न vdmin = vec_mul(vxmin, vyd);

        vector चिन्हित लघु q8ysums0 = vec_xl( 0, y[i].bsums);
        vector चिन्हित लघु q8ysums1 = vec_xl(16, y[i].bsums);

        vector चिन्हित अक्षर q2xmins = (vector चिन्हित अक्षर)vec_xl( 0, x[i].scales);
        vector चिन्हित अक्षर vscales = vec_and(q2xmins, निचला_Scale_Mask);

        q2xmins = vec_sr(q2xmins, v4);
        vector चिन्हित लघु q2xmins0 = vec_unpackh(q2xmins);
        vector चिन्हित लघु q2xmins1 = vec_unpackl(q2xmins);

        vector चिन्हित पूर्णांक prod0 = vec_mule(q2xmins0, q8ysums0);
        vector चिन्हित पूर्णांक prod1 = vec_mulo(q2xmins0, q8ysums0);
        vector चिन्हित पूर्णांक prod2 = vec_mule(q2xmins1, q8ysums1);
        vector चिन्हित पूर्णांक prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;
        vector चिन्हित पूर्णांक vsumi4 = v0;
        vector चिन्हित पूर्णांक vsumi5 = v0;
        vector चिन्हित पूर्णांक vsumi6 = v0;
        vector चिन्हित पूर्णांक vsumi7 = v0;

        स्थिर uint8_t * GGML_RESTRICT q2 = x[i].qs;
        स्थिर int8_t  * GGML_RESTRICT q8 = y[i].qs;

        क्रम (पूर्णांक j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector चिन्हित अक्षर qxs0 = (vector चिन्हित अक्षर)vec_xl( 0, q2);
            vector चिन्हित अक्षर qxs1 = (vector चिन्हित अक्षर)vec_xl(16, q2);
            q2 += 32;

            vector अचिन्हित अक्षर q2x00 = (vector अचिन्हित अक्षर)vec_and(qxs0, निचला_Mask);
            vector अचिन्हित अक्षर q2x01 = (vector अचिन्हित अक्षर)vec_and(vec_sr(qxs0, v2), निचला_Mask);
            vector अचिन्हित अक्षर q2x02 = (vector अचिन्हित अक्षर)vec_and(vec_sr(qxs0, v4), निचला_Mask);
            vector अचिन्हित अक्षर q2x03 = (vector अचिन्हित अक्षर)vec_and(vec_sr(qxs0, v6), निचला_Mask);
            vector अचिन्हित अक्षर q2x10 = (vector अचिन्हित अक्षर)vec_and(qxs1, निचला_Mask);
            vector अचिन्हित अक्षर q2x11 = (vector अचिन्हित अक्षर)vec_and(vec_sr(qxs1, v2), निचला_Mask);
            vector अचिन्हित अक्षर q2x12 = (vector अचिन्हित अक्षर)vec_and(vec_sr(qxs1, v4), निचला_Mask);
            vector अचिन्हित अक्षर q2x13 = (vector अचिन्हित अक्षर)vec_and(vec_sr(qxs1, v6), निचला_Mask);

            vector चिन्हित अक्षर q8y00 = vec_xl(  0, q8);
            vector चिन्हित अक्षर q8y10 = vec_xl( 16, q8);
            vector चिन्हित अक्षर q8y01 = vec_xl( 32, q8);
            vector चिन्हित अक्षर q8y11 = vec_xl( 48, q8);
            vector चिन्हित अक्षर q8y02 = vec_xl( 64, q8);
            vector चिन्हित अक्षर q8y12 = vec_xl( 80, q8);
            vector चिन्हित अक्षर q8y03 = vec_xl( 96, q8);
            vector चिन्हित अक्षर q8y13 = vec_xl(112, q8);
            q8 += 128;

            vector चिन्हित पूर्णांक qv0 = vec_msum(q8y00, q2x00, v0);
            vector चिन्हित पूर्णांक qv1 = vec_msum(q8y01, q2x01, v0);
            vector चिन्हित पूर्णांक qv2 = vec_msum(q8y02, q2x02, v0);
            vector चिन्हित पूर्णांक qv3 = vec_msum(q8y03, q2x03, v0);
            vector चिन्हित पूर्णांक qv4 = vec_msum(q8y10, q2x10, v0);
            vector चिन्हित पूर्णांक qv5 = vec_msum(q8y11, q2x11, v0);
            vector चिन्हित पूर्णांक qv6 = vec_msum(q8y12, q2x12, v0);
            vector चिन्हित पूर्णांक qv7 = vec_msum(q8y13, q2x13, v0);

            vector चिन्हित लघु vscales_07 = vec_unpackh(vscales);
            vector चिन्हित पूर्णांक vscales_03 = vec_unpackh(vscales_07);
            vector चिन्हित पूर्णांक vscales_47 = vec_unpackl(vscales_07);
            vector चिन्हित पूर्णांक vs0 = vec_splat(vscales_03, 0);
            vector चिन्हित पूर्णांक vs1 = vec_splat(vscales_03, 1);
            vector चिन्हित पूर्णांक vs2 = vec_splat(vscales_03, 2);
            vector चिन्हित पूर्णांक vs3 = vec_splat(vscales_03, 3);
            vector चिन्हित पूर्णांक vs4 = vec_splat(vscales_47, 0);
            vector चिन्हित पूर्णांक vs5 = vec_splat(vscales_47, 1);
            vector चिन्हित पूर्णांक vs6 = vec_splat(vscales_47, 2);
            vector चिन्हित पूर्णांक vs7 = vec_splat(vscales_47, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_add(vec_mul(qv0, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv1, vs2), vsumi1);
            vsumi2 = vec_add(vec_mul(qv2, vs4), vsumi2);
            vsumi3 = vec_add(vec_mul(qv3, vs6), vsumi3);
            vsumi4 = vec_add(vec_mul(qv4, vs1), vsumi4);
            vsumi5 = vec_add(vec_mul(qv5, vs3), vsumi5);
            vsumi6 = vec_add(vec_mul(qv6, vs5), vsumi6);
            vsumi7 = vec_add(vec_mul(qv7, vs7), vsumi7);
        }

        vsumi0 = vec_add(vsumi0, vsumi4);
        vsumi1 = vec_add(vsumi1, vsumi5);
        vsumi2 = vec_add(vsumi2, vsumi6);
        vsumi3 = vec_add(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_q3_K_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर uint32_t kmask1 = 0x03030303;
    स्थिर uint32_t kmask2 = 0x0f0f0f0f;

    स्थिर block_q3_K * GGML_RESTRICT x = vx;
    स्थिर block_q8_K * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0x3);
    स्थिर vector चिन्हित अक्षर निचला_Mask1 = vec_splats((int8_t)0xf);
    स्थिर vector चिन्हित अक्षर निचला_Mask2 = vec_splats((int8_t)0x30);
    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector चिन्हित अक्षर v1 = vec_splats((चिन्हित अक्षर)0x1);
    स्थिर vector अचिन्हित अक्षर v2 = vec_splats((अचिन्हित अक्षर)0x2);
    स्थिर vector अचिन्हित अक्षर v3 = vec_splats((अचिन्हित अक्षर)0x3);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);
    स्थिर vector अचिन्हित अक्षर v6 = vec_splats((अचिन्हित अक्षर)0x6);
    स्थिर vector चिन्हित अक्षर off = vec_splats((चिन्हित अक्षर)0x20);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        UNUSED(kmask1);
        UNUSED(kmask2);

        vector चिन्हित अक्षर u0 = (vector चिन्हित अक्षर)vec_xl_len(x[i].scales, 8);
        vector चिन्हित अक्षर u1 = vec_and(u0, निचला_Mask1);
        vector चिन्हित अक्षर u2 = (vector चिन्हित अक्षर)vec_xl_len(x[i].scales + 8, 4);
        vector चिन्हित अक्षर u3 = (vector चिन्हित अक्षर)vec_mergeh((vector चिन्हित पूर्णांक)u2, (vector चिन्हित पूर्णांक)vec_sr(u2, v2));
        vector चिन्हित अक्षर u30 = vec_sl(vec_and(u3, निचला_Mask), v4);
        vector चिन्हित अक्षर u31 = vec_and(u3, निचला_Mask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        vector चिन्हित अक्षर vscales = (vector चिन्हित अक्षर)vec_mergeh((vector चिन्हित दीर्घ दीर्घ)u1, (vector चिन्हित दीर्घ दीर्घ)u2);
        vector चिन्हित अक्षर qxhs0 = (vector चिन्हित अक्षर)vec_xl( 0, x[i].hmask);
        vector चिन्हित अक्षर qxhs1 = (vector चिन्हित अक्षर)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;
        vector चिन्हित पूर्णांक vsumi4 = v0;
        vector चिन्हित पूर्णांक vsumi5 = v0;
        vector चिन्हित पूर्णांक vsumi6 = v0;
        vector चिन्हित पूर्णांक vsumi7 = v0;

        स्थिर uint8_t * GGML_RESTRICT q3 = x[i].qs;
        स्थिर int8_t  * GGML_RESTRICT q8 = y[i].qs;

        क्रम (पूर्णांक j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector चिन्हित अक्षर qxs0 = (vector चिन्हित अक्षर)vec_xl( 0, q3);
            vector चिन्हित अक्षर qxs1 = (vector चिन्हित अक्षर)vec_xl(16, q3);
            q3 += 32;

            //the low 2 bits
            vector चिन्हित अक्षर qxs00 = vec_and(qxs0, निचला_Mask);
            vector चिन्हित अक्षर qxs01 = vec_and(vec_sr(qxs0, v2), निचला_Mask);
            vector चिन्हित अक्षर qxs02 = vec_and(vec_sr(qxs0, v4), निचला_Mask);
            vector चिन्हित अक्षर qxs03 = vec_and(vec_sr(qxs0, v6), निचला_Mask);
            vector चिन्हित अक्षर qxs10 = vec_and(qxs1, निचला_Mask);
            vector चिन्हित अक्षर qxs11 = vec_and(vec_sr(qxs1, v2), निचला_Mask);
            vector चिन्हित अक्षर qxs12 = vec_and(vec_sr(qxs1, v4), निचला_Mask);
            vector चिन्हित अक्षर qxs13 = vec_and(vec_sr(qxs1, v6), निचला_Mask);

            //the 3rd bit
            vector चिन्हित अक्षर qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            vector चिन्हित अक्षर qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (vector अचिन्हित अक्षर)v1)), v2);
            vector चिन्हित अक्षर qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            vector चिन्हित अक्षर qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            vector चिन्हित अक्षर qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            vector चिन्हित अक्षर qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (vector अचिन्हित अक्षर)v1)), v2);
            vector चिन्हित अक्षर qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            vector चिन्हित अक्षर qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            vector चिन्हित अक्षर q3x00 = vec_sub(qxs00, qxh00);
            vector चिन्हित अक्षर q3x01 = vec_sub(qxs01, qxh01);
            vector चिन्हित अक्षर q3x02 = vec_sub(qxs02, qxh02);
            vector चिन्हित अक्षर q3x03 = vec_sub(qxs03, qxh03);
            vector चिन्हित अक्षर q3x10 = vec_sub(qxs10, qxh10);
            vector चिन्हित अक्षर q3x11 = vec_sub(qxs11, qxh11);
            vector चिन्हित अक्षर q3x12 = vec_sub(qxs12, qxh12);
            vector चिन्हित अक्षर q3x13 = vec_sub(qxs13, qxh13);

            vector चिन्हित अक्षर q8y00 = vec_xl(  0, q8);
            vector चिन्हित अक्षर q8y10 = vec_xl( 16, q8);
            vector चिन्हित अक्षर q8y01 = vec_xl( 32, q8);
            vector चिन्हित अक्षर q8y11 = vec_xl( 48, q8);
            vector चिन्हित अक्षर q8y02 = vec_xl( 64, q8);
            vector चिन्हित अक्षर q8y12 = vec_xl( 80, q8);
            vector चिन्हित अक्षर q8y03 = vec_xl( 96, q8);
            vector चिन्हित अक्षर q8y13 = vec_xl(112, q8);
            q8 += 128;

            vector चिन्हित लघु vscales_h = vec_unpackh(vscales);
            vector चिन्हित लघु vs0 = vec_splat(vscales_h, 0);
            vector चिन्हित लघु vs1 = vec_splat(vscales_h, 1);
            vector चिन्हित लघु vs2 = vec_splat(vscales_h, 2);
            vector चिन्हित लघु vs3 = vec_splat(vscales_h, 3);
            vector चिन्हित लघु vs4 = vec_splat(vscales_h, 4);
            vector चिन्हित लघु vs5 = vec_splat(vscales_h, 5);
            vector चिन्हित लघु vs6 = vec_splat(vscales_h, 6);
            vector चिन्हित लघु vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            vector चिन्हित लघु qv00 = vec_add(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            vector चिन्हित लघु qv01 = vec_add(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            vector चिन्हित लघु qv02 = vec_add(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            vector चिन्हित लघु qv03 = vec_add(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            vector चिन्हित लघु qv10 = vec_add(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            vector चिन्हित लघु qv11 = vec_add(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            vector चिन्हित लघु qv12 = vec_add(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            vector चिन्हित लघु qv13 = vec_add(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs2, vsumi1);
            vsumi2 = vec_msum(qv02, vs4, vsumi2);
            vsumi3 = vec_msum(qv03, vs6, vsumi3);
            vsumi4 = vec_msum(qv10, vs1, vsumi4);
            vsumi5 = vec_msum(qv11, vs3, vsumi5);
            vsumi6 = vec_msum(qv12, vs5, vsumi6);
            vsumi7 = vec_msum(qv13, vs7, vsumi7);
        }

        vsumi0 = vec_add(vsumi0, vsumi4);
        vsumi1 = vec_add(vsumi1, vsumi5);
        vsumi2 = vec_add(vsumi2, vsumi6);
        vsumi3 = vec_add(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_q4_K_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_q4_K * GGML_RESTRICT x = vx;
    स्थिर block_q8_K * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

    अटल स्थिर uint32_t kmask1 = 0x3f3f3f3f;
    अटल स्थिर uint32_t kmask2 = 0x0f0f0f0f;
    अटल स्थिर uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector चिन्हित अक्षर निचला_Mask1 = vec_splats((int8_t)0x3f);
    स्थिर vector चिन्हित अक्षर निचला_Mask2 = vec_splats((int8_t)0x30);
    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector अचिन्हित अक्षर v2 = vec_splats((uint8_t)2);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector भग्न vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].dmin));
        vector भग्न vdmin = vec_mul(vxmin, vyd);

        vector चिन्हित लघु q8ysums0 = vec_xl( 0, y[i].bsums);
        vector चिन्हित लघु q8ysums1 = vec_xl(16, y[i].bsums);

        UNUSED(kmask1);
        UNUSED(kmask2);
        UNUSED(kmask3);
        UNUSED(utmp);

        vector चिन्हित अक्षर u0 = (vector चिन्हित अक्षर)vec_xl_len(x[i].scales, 8);
        vector चिन्हित अक्षर u1 = vec_and(vec_sr(u0, v2), निचला_Mask2);
        vector चिन्हित अक्षर u2 = (vector चिन्हित अक्षर)vec_xl_len(x[i].scales + 8, 4);
        vector चिन्हित अक्षर u3 = vec_sr(u2, v4);

        vector चिन्हित अक्षर u30 = u1;
        vector चिन्हित अक्षर u31 = (vector चिन्हित अक्षर)vec_mergeh((vector चिन्हित पूर्णांक)vec_and(u2, निचला_Mask), (vector चिन्हित पूर्णांक)u3);

        u1 = vec_and(u0, निचला_Mask1);
        u2 = vec_or(u30, u31);

        vector चिन्हित अक्षर utmps = (vector चिन्हित अक्षर)vec_mergeh((vector चिन्हित पूर्णांक)u1, (vector चिन्हित पूर्णांक)u2);

        vector चिन्हित लघु vscales = vec_unpackh(utmps);
        vector चिन्हित लघु q4xmins = vec_unpackl(utmps);
        vector चिन्हित लघु q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        vector चिन्हित लघु q4xmins1 = vec_mergel(q4xmins, q4xmins);

        vector चिन्हित पूर्णांक prod0 = vec_mule(q4xmins0, q8ysums0);
        vector चिन्हित पूर्णांक prod1 = vec_mule(q4xmins1, q8ysums1);
        vector चिन्हित पूर्णांक prod2 = vec_mulo(q4xmins0, q8ysums0);
        vector चिन्हित पूर्णांक prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;

        स्थिर uint8_t * GGML_RESTRICT q4 = x[i].qs;
        स्थिर int8_t  * GGML_RESTRICT q8 = y[i].qs;

        क्रम (पूर्णांक j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector चिन्हित अक्षर qxs0 = (vector चिन्हित अक्षर)vec_xl( 0, q4);
            vector चिन्हित अक्षर qxs1 = (vector चिन्हित अक्षर)vec_xl(16, q4);
            vector चिन्हित अक्षर qxs2 = (vector चिन्हित अक्षर)vec_xl(32, q4);
            vector चिन्हित अक्षर qxs3 = (vector चिन्हित अक्षर)vec_xl(48, q4);
            q4 += 64;

            vector अचिन्हित अक्षर q4x00 = (vector अचिन्हित अक्षर)vec_and(qxs0, निचला_Mask);
            vector अचिन्हित अक्षर q4x01 = (vector अचिन्हित अक्षर)vec_sr(qxs0, v4);
            vector अचिन्हित अक्षर q4x10 = (vector अचिन्हित अक्षर)vec_and(qxs1, निचला_Mask);
            vector अचिन्हित अक्षर q4x11 = (vector अचिन्हित अक्षर)vec_sr(qxs1, v4);
            vector अचिन्हित अक्षर q4x20 = (vector अचिन्हित अक्षर)vec_and(qxs2, निचला_Mask);
            vector अचिन्हित अक्षर q4x21 = (vector अचिन्हित अक्षर)vec_sr(qxs2, v4);
            vector अचिन्हित अक्षर q4x30 = (vector अचिन्हित अक्षर)vec_and(qxs3, निचला_Mask);
            vector अचिन्हित अक्षर q4x31 = (vector अचिन्हित अक्षर)vec_sr(qxs3, v4);

            vector चिन्हित अक्षर q8y00 = vec_xl(  0, q8);
            vector चिन्हित अक्षर q8y10 = vec_xl( 16, q8);
            vector चिन्हित अक्षर q8y01 = vec_xl( 32, q8);
            vector चिन्हित अक्षर q8y11 = vec_xl( 48, q8);
            vector चिन्हित अक्षर q8y20 = vec_xl( 64, q8);
            vector चिन्हित अक्षर q8y30 = vec_xl( 80, q8);
            vector चिन्हित अक्षर q8y21 = vec_xl( 96, q8);
            vector चिन्हित अक्षर q8y31 = vec_xl(112, q8);
            q8 += 128;

            vector चिन्हित पूर्णांक qv00 = vec_msum(q8y00, q4x00, v0);
            vector चिन्हित पूर्णांक qv01 = vec_msum(q8y01, q4x01, v0);
            vector चिन्हित पूर्णांक qv10 = vec_msum(q8y10, q4x10, v0);
            vector चिन्हित पूर्णांक qv11 = vec_msum(q8y11, q4x11, v0);
            vector चिन्हित पूर्णांक qv20 = vec_msum(q8y20, q4x20, v0);
            vector चिन्हित पूर्णांक qv21 = vec_msum(q8y21, q4x21, v0);
            vector चिन्हित पूर्णांक qv30 = vec_msum(q8y30, q4x30, v0);
            vector चिन्हित पूर्णांक qv31 = vec_msum(q8y31, q4x31, v0);

            vector चिन्हित पूर्णांक vscales_h = vec_unpackh(vscales);
            vector चिन्हित पूर्णांक vs0 = vec_splat(vscales_h, 0);
            vector चिन्हित पूर्णांक vs1 = vec_splat(vscales_h, 1);
            vector चिन्हित पूर्णांक vs2 = vec_splat(vscales_h, 2);
            vector चिन्हित पूर्णांक vs3 = vec_splat(vscales_h, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_add(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv01, vs1), vsumi1);
            vsumi2 = vec_add(vec_mul(qv20, vs2), vsumi2);
            vsumi3 = vec_add(vec_mul(qv21, vs3), vsumi3);

            vsumi0 = vec_add(vec_mul(qv10, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv11, vs1), vsumi1);
            vsumi2 = vec_add(vec_mul(qv30, vs2), vsumi2);
            vsumi3 = vec_add(vec_mul(qv31, vs3), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_q5_K_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy,  माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_q5_K * GGML_RESTRICT x = vx;
    स्थिर block_q8_K * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

    अटल स्थिर uint32_t kmask1 = 0x3f3f3f3f;
    अटल स्थिर uint32_t kmask2 = 0x0f0f0f0f;
    अटल स्थिर uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector चिन्हित अक्षर निचला_Mask1 = vec_splats((int8_t)0x3f);
    स्थिर vector चिन्हित अक्षर निचला_Mask2 = vec_splats((int8_t)0x30);
    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector अचिन्हित अक्षर v1 = vec_splats((अचिन्हित अक्षर)0x1);
    स्थिर vector अचिन्हित अक्षर v2 = vec_splats((अचिन्हित अक्षर)0x2);
    स्थिर vector अचिन्हित अक्षर v3 = vec_splats((अचिन्हित अक्षर)0x3);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector भग्न vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].dmin));
        vector भग्न vdmin = vec_mul(vxmin, vyd);

        UNUSED(kmask1);
        UNUSED(kmask2);
        UNUSED(kmask3);
        UNUSED(utmp);

        vector चिन्हित अक्षर u0 = (vector चिन्हित अक्षर)vec_xl_len(x[i].scales, 8);
        vector चिन्हित अक्षर u1 = vec_and(vec_sr(u0, v2), निचला_Mask2);
        vector चिन्हित अक्षर u2 = (vector चिन्हित अक्षर)vec_xl_len(x[i].scales + 8, 4);
        vector चिन्हित अक्षर u3 = vec_sr(u2, v4);

        vector चिन्हित अक्षर u30 = u1;
        vector चिन्हित अक्षर u31 = (vector चिन्हित अक्षर)vec_mergeh((vector चिन्हित पूर्णांक)vec_and(u2, निचला_Mask), (vector चिन्हित पूर्णांक)u3);

        u1 = vec_and(u0, निचला_Mask1);
        u2 = vec_or(u30, u31);

        vector चिन्हित अक्षर utmps = (vector चिन्हित अक्षर)vec_mergeh((vector चिन्हित पूर्णांक)u1, (vector चिन्हित पूर्णांक)u2);

        vector चिन्हित लघु q8ysums0 = vec_xl( 0, y[i].bsums);
        vector चिन्हित लघु q8ysums1 = vec_xl(16, y[i].bsums);

        vector चिन्हित लघु vscales = vec_unpackh(utmps);

        vector चिन्हित लघु q5xmins = vec_unpackl(utmps);
        vector चिन्हित लघु q5xmins0 = vec_mergeh(q5xmins, q5xmins);
        vector चिन्हित लघु q5xmins1 = vec_mergel(q5xmins, q5xmins);

        vector चिन्हित पूर्णांक prod0 = vec_mule(q5xmins0, q8ysums0);
        vector चिन्हित पूर्णांक prod1 = vec_mule(q5xmins1, q8ysums1);
        vector चिन्हित पूर्णांक prod2 = vec_mulo(q5xmins0, q8ysums0);
        vector चिन्हित पूर्णांक prod3 = vec_mulo(q5xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector चिन्हित अक्षर qxhs0 = (vector चिन्हित अक्षर)vec_xl( 0, x[i].qh);
        vector चिन्हित अक्षर qxhs1 = (vector चिन्हित अक्षर)vec_xl(16, x[i].qh);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;

        स्थिर uint8_t * GGML_RESTRICT q5 = x[i].qs;
        स्थिर int8_t  * GGML_RESTRICT q8 = y[i].qs;

        क्रम (पूर्णांक j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q5, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector चिन्हित अक्षर qxs0 = (vector चिन्हित अक्षर)vec_xl( 0, q5);
            vector चिन्हित अक्षर qxs1 = (vector चिन्हित अक्षर)vec_xl(16, q5);
            q5 += 32;

            vector चिन्हित अक्षर qxs00 = vec_and(qxs0, निचला_Mask);
            vector चिन्हित अक्षर qxs01 = vec_sr(qxs0, v4);
            vector चिन्हित अक्षर qxs10 = vec_and(qxs1, निचला_Mask);
            vector चिन्हित अक्षर qxs11 = vec_sr(qxs1, v4);

            vector चिन्हित अक्षर q5h00 = vec_sl(vec_and((vector चिन्हित अक्षर)v1, qxhs0), v4);
            vector चिन्हित अक्षर q5h01 = vec_sl(vec_and((vector चिन्हित अक्षर)v2, qxhs0), v3);
            vector चिन्हित अक्षर q5h10 = vec_sl(vec_and((vector चिन्हित अक्षर)v1, qxhs1), v4);
            vector चिन्हित अक्षर q5h11 = vec_sl(vec_and((vector चिन्हित अक्षर)v2, qxhs1), v3);
            qxhs0 = vec_sr(qxhs0, v2);
            qxhs1 = vec_sr(qxhs1, v2);

            vector अचिन्हित अक्षर q5x00 = (vector अचिन्हित अक्षर)vec_or(q5h00, qxs00);
            vector अचिन्हित अक्षर q5x01 = (vector अचिन्हित अक्षर)vec_or(q5h01, qxs01);
            vector अचिन्हित अक्षर q5x10 = (vector अचिन्हित अक्षर)vec_or(q5h10, qxs10);
            vector अचिन्हित अक्षर q5x11 = (vector अचिन्हित अक्षर)vec_or(q5h11, qxs11);

            vector चिन्हित अक्षर q8y00 = vec_xl( 0, q8);
            vector चिन्हित अक्षर q8y10 = vec_xl(16, q8);
            vector चिन्हित अक्षर q8y01 = vec_xl(32, q8);
            vector चिन्हित अक्षर q8y11 = vec_xl(48, q8);
            q8 += 64;

            vector चिन्हित पूर्णांक qv00 = vec_msum(q8y00, q5x00, v0);
            vector चिन्हित पूर्णांक qv01 = vec_msum(q8y01, q5x01, v0);
            vector चिन्हित पूर्णांक qv10 = vec_msum(q8y10, q5x10, v0);
            vector चिन्हित पूर्णांक qv11 = vec_msum(q8y11, q5x11, v0);

            vector चिन्हित पूर्णांक vscales_h = vec_unpackh(vscales);
            vector चिन्हित पूर्णांक vs0 = vec_splat(vscales_h, 0);
            vector चिन्हित पूर्णांक vs1 = vec_splat(vscales_h, 1);
            vscales = vec_sld(vscales, vscales, 12);

            vsumi0 = vec_add(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv10, vs0), vsumi1);
            vsumi2 = vec_add(vec_mul(qv01, vs1), vsumi2);
            vsumi3 = vec_add(vec_mul(qv11, vs1), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_q6_K_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_q6_K * GGML_RESTRICT x = vx;
    स्थिर block_q8_K * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector अचिन्हित अक्षर v2 = vec_splats((अचिन्हित अक्षर)0x2);
    स्थिर vector अचिन्हित अक्षर v3 = vec_splats((अचिन्हित अक्षर)0x3);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);
    स्थिर vector अचिन्हित अक्षर v6 = vec_splats((अचिन्हित अक्षर)0x6);
    स्थिर vector चिन्हित अक्षर off = vec_splats((चिन्हित अक्षर)0x20);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;
        vector चिन्हित पूर्णांक vsumi4 = v0;
        vector चिन्हित पूर्णांक vsumi5 = v0;
        vector चिन्हित पूर्णांक vsumi6 = v0;
        vector चिन्हित पूर्णांक vsumi7 = v0;

        स्थिर uint8_t * GGML_RESTRICT q6 = x[i].ql;
        स्थिर uint8_t * GGML_RESTRICT qh = x[i].qh;
        स्थिर int8_t  * GGML_RESTRICT qs = x[i].scales;
        स्थिर int8_t  * GGML_RESTRICT q8 = y[i].qs;

        क्रम (पूर्णांक j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q6, 0, 0);
            __builtin_prefetch(qh, 0, 0);
            __builtin_prefetch(q8, 0, 0);

            vector चिन्हित अक्षर qxs0 = (vector चिन्हित अक्षर)vec_xl( 0, q6);
            vector चिन्हित अक्षर qxs1 = (vector चिन्हित अक्षर)vec_xl(16, q6);
            vector चिन्हित अक्षर qxs2 = (vector चिन्हित अक्षर)vec_xl(32, q6);
            vector चिन्हित अक्षर qxs3 = (vector चिन्हित अक्षर)vec_xl(48, q6);
            q6 += 64;

            vector चिन्हित अक्षर qxs00 = vec_and(qxs0, निचला_Mask);
            vector चिन्हित अक्षर qxs01 = vec_sr(qxs0, v4);
            vector चिन्हित अक्षर qxs10 = vec_and(qxs1, निचला_Mask);
            vector चिन्हित अक्षर qxs11 = vec_sr(qxs1, v4);
            vector चिन्हित अक्षर qxs20 = vec_and(qxs2, निचला_Mask);
            vector चिन्हित अक्षर qxs21 = vec_sr(qxs2, v4);
            vector चिन्हित अक्षर qxs30 = vec_and(qxs3, निचला_Mask);
            vector चिन्हित अक्षर qxs31 = vec_sr(qxs3, v4);

            vector चिन्हित अक्षर qxhs0 = (vector चिन्हित अक्षर)vec_xl( 0, qh);
            vector चिन्हित अक्षर qxhs1 = (vector चिन्हित अक्षर)vec_xl(16, qh);
            qh += 32;

            vector चिन्हित अक्षर qxh00 = vec_sl(vec_and((vector चिन्हित अक्षर)v3, qxhs0), v4);
            vector चिन्हित अक्षर qxh01 = vec_sl(vec_and((vector चिन्हित अक्षर)v3, vec_sr(qxhs0, v4)), v4);
            vector चिन्हित अक्षर qxh10 = vec_sl(vec_and((vector चिन्हित अक्षर)v3, qxhs1), v4);
            vector चिन्हित अक्षर qxh11 = vec_sl(vec_and((vector चिन्हित अक्षर)v3, vec_sr(qxhs1, v4)), v4);
            vector चिन्हित अक्षर qxh20 = vec_sl(vec_and((vector चिन्हित अक्षर)v3, vec_sr(qxhs0, v2)), v4);
            vector चिन्हित अक्षर qxh21 = vec_sl(vec_and((vector चिन्हित अक्षर)v3, vec_sr(qxhs0, v6)), v4);
            vector चिन्हित अक्षर qxh30 = vec_sl(vec_and((vector चिन्हित अक्षर)v3, vec_sr(qxhs1, v2)), v4);
            vector चिन्हित अक्षर qxh31 = vec_sl(vec_and((vector चिन्हित अक्षर)v3, vec_sr(qxhs1, v6)), v4);

            vector चिन्हित अक्षर q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
            vector चिन्हित अक्षर q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
            vector चिन्हित अक्षर q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
            vector चिन्हित अक्षर q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
            vector चिन्हित अक्षर q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
            vector चिन्हित अक्षर q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
            vector चिन्हित अक्षर q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
            vector चिन्हित अक्षर q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

            vector चिन्हित अक्षर q8y00 = vec_xl(  0, q8);
            vector चिन्हित अक्षर q8y10 = vec_xl( 16, q8);
            vector चिन्हित अक्षर q8y20 = vec_xl( 32, q8);
            vector चिन्हित अक्षर q8y30 = vec_xl( 48, q8);
            vector चिन्हित अक्षर q8y01 = vec_xl( 64, q8);
            vector चिन्हित अक्षर q8y11 = vec_xl( 80, q8);
            vector चिन्हित अक्षर q8y21 = vec_xl( 96, q8);
            vector चिन्हित अक्षर q8y31 = vec_xl(112, q8);
            q8 += 128;

            vector चिन्हित लघु qv00 = vec_add(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
            vector चिन्हित लघु qv10 = vec_add(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
            vector चिन्हित लघु qv20 = vec_add(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
            vector चिन्हित लघु qv30 = vec_add(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
            vector चिन्हित लघु qv01 = vec_add(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
            vector चिन्हित लघु qv11 = vec_add(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
            vector चिन्हित लघु qv21 = vec_add(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
            vector चिन्हित लघु qv31 = vec_add(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

            vector चिन्हित लघु vscales = vec_unpackh(vec_xl_len(qs, 8));
            qs += 8;

            vector चिन्हित लघु vs0 = vec_splat(vscales, 0);
            vector चिन्हित लघु vs1 = vec_splat(vscales, 1);
            vector चिन्हित लघु vs2 = vec_splat(vscales, 2);
            vector चिन्हित लघु vs3 = vec_splat(vscales, 3);
            vector चिन्हित लघु vs4 = vec_splat(vscales, 4);
            vector चिन्हित लघु vs5 = vec_splat(vscales, 5);
            vector चिन्हित लघु vs6 = vec_splat(vscales, 6);
            vector चिन्हित लघु vs7 = vec_splat(vscales, 7);

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs4, vsumi1);
            vsumi2 = vec_msum(qv10, vs1, vsumi2);
            vsumi3 = vec_msum(qv11, vs5, vsumi3);
            vsumi4 = vec_msum(qv20, vs2, vsumi4);
            vsumi5 = vec_msum(qv21, vs6, vsumi5);
            vsumi6 = vec_msum(qv30, vs3, vsumi6);
            vsumi7 = vec_msum(qv31, vs7, vsumi7);
        }

        vsumi0 = vec_add(vsumi0, vsumi4);
        vsumi1 = vec_add(vsumi1, vsumi5);
        vsumi2 = vec_add(vsumi2, vsumi6);
        vsumi3 = vec_add(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

#अगर defined (__POWER9_VECTOR__)
अटल स्थिर int8_t keven_signs_q2xs[1024] = {
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
#पूर्ण_अगर

व्योम ggml_vec_dot_iq2_xxs_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_iq2_xxs * GGML_RESTRICT x = vx;
    स्थिर block_q8_K    * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);
    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    स्थिर uint64_t * signs64 = (स्थिर uint64_t *)keven_signs_q2xs;

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;

        स्थिर uint16_t * GGML_RESTRICT q2 = x[i].qs;
        स्थिर int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        क्रम (पूर्णांक j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            स्थिर uint8_t * aux8 = (स्थिर uint8_t *)aux32;

            स_नकल(aux32, q2, 4*माप(uint32_t));
            q2 += 8;

            vector चिन्हित दीर्घ दीर्घ aux64x2_0 = {*(स्थिर int64_t *)(iq2xxs_grid + aux8[ 0]), *(स्थिर int64_t *)(iq2xxs_grid + aux8[ 1])};
            vector चिन्हित दीर्घ दीर्घ aux64x2_1 = {*(स्थिर int64_t *)(iq2xxs_grid + aux8[ 2]), *(स्थिर int64_t *)(iq2xxs_grid + aux8[ 3])};
            vector चिन्हित दीर्घ दीर्घ aux64x2_2 = {*(स्थिर int64_t *)(iq2xxs_grid + aux8[ 8]), *(स्थिर int64_t *)(iq2xxs_grid + aux8[ 9])};
            vector चिन्हित दीर्घ दीर्घ aux64x2_3 = {*(स्थिर int64_t *)(iq2xxs_grid + aux8[10]), *(स्थिर int64_t *)(iq2xxs_grid + aux8[11])};

            vector चिन्हित दीर्घ दीर्घ vsigns0 = {*(स्थिर int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(स्थिर int64_t *)(signs64 + ((aux32[1] >>  7) & 127))};
            vector चिन्हित दीर्घ दीर्घ vsigns1 = {*(स्थिर int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(स्थिर int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            vector चिन्हित दीर्घ दीर्घ vsigns2 = {*(स्थिर int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(स्थिर int64_t *)(signs64 + ((aux32[3] >>  7) & 127))};
            vector चिन्हित दीर्घ दीर्घ vsigns3 = {*(स्थिर int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(स्थिर int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            vector चिन्हित अक्षर q2x0 = (vector चिन्हित अक्षर)vec_mul((vector चिन्हित अक्षर)vsigns0, (vector चिन्हित अक्षर)aux64x2_0);
            vector चिन्हित अक्षर q2x1 = (vector चिन्हित अक्षर)vec_mul((vector चिन्हित अक्षर)vsigns1, (vector चिन्हित अक्षर)aux64x2_1);
            vector चिन्हित अक्षर q2x2 = (vector चिन्हित अक्षर)vec_mul((vector चिन्हित अक्षर)vsigns2, (vector चिन्हित अक्षर)aux64x2_2);
            vector चिन्हित अक्षर q2x3 = (vector चिन्हित अक्षर)vec_mul((vector चिन्हित अक्षर)vsigns3, (vector चिन्हित अक्षर)aux64x2_3);

            vector चिन्हित अक्षर q8y0 = vec_xl( 0, q8);
            vector चिन्हित अक्षर q8y1 = vec_xl(16, q8);
            vector चिन्हित अक्षर q8y2 = vec_xl(32, q8);
            vector चिन्हित अक्षर q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector चिन्हित लघु qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector चिन्हित लघु qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector चिन्हित लघु qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector चिन्हित लघु qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            स्थिर uint16_t ls0 = aux32[1] >> 28;
            स्थिर uint16_t ls1 = aux32[3] >> 28;

            vector चिन्हित लघु vscales01 = vec_splats((int16_t)(2*ls0+1));
            vector चिन्हित लघु vscales23 = vec_splats((int16_t)(2*ls1+1));

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

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_iq2_xs_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_iq2_xs * GGML_RESTRICT x = vx;
    स्थिर block_q8_K   * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);
    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    स्थिर uint64_t * signs64 = (स्थिर uint64_t *)keven_signs_q2xs;

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;

        स्थिर uint16_t * GGML_RESTRICT q2 = x[i].qs;
        स्थिर uint8_t  * GGML_RESTRICT sc = x[i].scales;
        स्थिर int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        क्रम (पूर्णांक j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector चिन्हित दीर्घ दीर्घ aux64x2_0 = {*(स्थिर int64_t *)(iq2xs_grid + (q2[0] & 511)), *(स्थिर int64_t *)(iq2xs_grid + (q2[1] & 511))};
            vector चिन्हित दीर्घ दीर्घ aux64x2_1 = {*(स्थिर int64_t *)(iq2xs_grid + (q2[2] & 511)), *(स्थिर int64_t *)(iq2xs_grid + (q2[3] & 511))};
            vector चिन्हित दीर्घ दीर्घ aux64x2_2 = {*(स्थिर int64_t *)(iq2xs_grid + (q2[4] & 511)), *(स्थिर int64_t *)(iq2xs_grid + (q2[5] & 511))};
            vector चिन्हित दीर्घ दीर्घ aux64x2_3 = {*(स्थिर int64_t *)(iq2xs_grid + (q2[6] & 511)), *(स्थिर int64_t *)(iq2xs_grid + (q2[7] & 511))};

            vector चिन्हित दीर्घ दीर्घ vsigns0 = {*(स्थिर int64_t *)(signs64 + ((q2[0] >> 9))), *(स्थिर int64_t *)(signs64 + ((q2[1] >> 9)))};
            vector चिन्हित दीर्घ दीर्घ vsigns1 = {*(स्थिर int64_t *)(signs64 + ((q2[2] >> 9))), *(स्थिर int64_t *)(signs64 + ((q2[3] >> 9)))};
            vector चिन्हित दीर्घ दीर्घ vsigns2 = {*(स्थिर int64_t *)(signs64 + ((q2[4] >> 9))), *(स्थिर int64_t *)(signs64 + ((q2[5] >> 9)))};
            vector चिन्हित दीर्घ दीर्घ vsigns3 = {*(स्थिर int64_t *)(signs64 + ((q2[6] >> 9))), *(स्थिर int64_t *)(signs64 + ((q2[7] >> 9)))};
            q2 += 8;

            vector चिन्हित अक्षर q2x0 = (vector चिन्हित अक्षर)vec_mul((vector चिन्हित अक्षर)vsigns0, (vector चिन्हित अक्षर)aux64x2_0);
            vector चिन्हित अक्षर q2x1 = (vector चिन्हित अक्षर)vec_mul((vector चिन्हित अक्षर)vsigns1, (vector चिन्हित अक्षर)aux64x2_1);
            vector चिन्हित अक्षर q2x2 = (vector चिन्हित अक्षर)vec_mul((vector चिन्हित अक्षर)vsigns2, (vector चिन्हित अक्षर)aux64x2_2);
            vector चिन्हित अक्षर q2x3 = (vector चिन्हित अक्षर)vec_mul((vector चिन्हित अक्षर)vsigns3, (vector चिन्हित अक्षर)aux64x2_3);

            vector चिन्हित अक्षर q8y0 = vec_xl( 0, q8);
            vector चिन्हित अक्षर q8y1 = vec_xl(16, q8);
            vector चिन्हित अक्षर q8y2 = vec_xl(32, q8);
            vector चिन्हित अक्षर q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector चिन्हित लघु qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector चिन्हित लघु qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector चिन्हित लघु qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector चिन्हित लघु qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            स्थिर uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            स्थिर uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            स्थिर uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            स्थिर uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            vector चिन्हित लघु vscales0 = vec_splats((int16_t)(2*ls0+1));
            vector चिन्हित लघु vscales1 = vec_splats((int16_t)(2*ls1+1));
            vector चिन्हित लघु vscales2 = vec_splats((int16_t)(2*ls2+1));
            vector चिन्हित लघु vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_iq2_s_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_iq2_s * GGML_RESTRICT x = vx;
    स्थिर block_q8_K  * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    अटल स्थिर uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    अटल स्थिर uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    स्थिर vector अचिन्हित अक्षर mask0 = vec_xl( 0, k_mask1);
    स्थिर vector अचिन्हित अक्षर mask1 = vec_xl(16, k_mask1);
    स्थिर vector चिन्हित अक्षर mask2 = (vector चिन्हित अक्षर)vec_xl( 0, k_mask2);

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;

        स्थिर uint8_t *  GGML_RESTRICT q2 = x[i].qs;
        स्थिर uint8_t *  GGML_RESTRICT qh = x[i].qh;
        स्थिर uint16_t * GGML_RESTRICT signs = (स्थिर uint16_t *)(x[i].qs + QK_K/8);
        स्थिर uint8_t *  GGML_RESTRICT sc = x[i].scales;
        स्थिर int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        क्रम (पूर्णांक j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector चिन्हित दीर्घ दीर्घ aux64x2_0 = {*(स्थिर int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(स्थिर int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            vector चिन्हित दीर्घ दीर्घ aux64x2_1 = {*(स्थिर int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(स्थिर int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            vector चिन्हित दीर्घ दीर्घ aux64x2_2 = {*(स्थिर int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(स्थिर int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            vector चिन्हित दीर्घ दीर्घ aux64x2_3 = {*(स्थिर int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(स्थिर int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            vector चिन्हित अक्षर vsigns01 = (vector चिन्हित अक्षर)vec_splats(*(स्थिर uint32_t *)&signs[0]);
            vector चिन्हित अक्षर vsigns23 = (vector चिन्हित अक्षर)vec_splats(*(स्थिर uint32_t *)&signs[2]);
            signs += 4;

            vector चिन्हित अक्षर vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            vector चिन्हित अक्षर vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            vector चिन्हित अक्षर vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            vector चिन्हित अक्षर vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (vector चिन्हित अक्षर)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (vector चिन्हित अक्षर)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (vector चिन्हित अक्षर)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (vector चिन्हित अक्षर)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            vector चिन्हित अक्षर q2x0 = vec_sub(vec_xor(vsigns0, (vector चिन्हित अक्षर)aux64x2_0), vsigns0);
            vector चिन्हित अक्षर q2x1 = vec_sub(vec_xor(vsigns1, (vector चिन्हित अक्षर)aux64x2_1), vsigns1);
            vector चिन्हित अक्षर q2x2 = vec_sub(vec_xor(vsigns2, (vector चिन्हित अक्षर)aux64x2_2), vsigns2);
            vector चिन्हित अक्षर q2x3 = vec_sub(vec_xor(vsigns3, (vector चिन्हित अक्षर)aux64x2_3), vsigns3);

            vector चिन्हित अक्षर q8y0 = vec_xl( 0, q8);
            vector चिन्हित अक्षर q8y1 = vec_xl(16, q8);
            vector चिन्हित अक्षर q8y2 = vec_xl(32, q8);
            vector चिन्हित अक्षर q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector चिन्हित लघु qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector चिन्हित लघु qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector चिन्हित लघु qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector चिन्हित लघु qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            स्थिर uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            स्थिर uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            स्थिर uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            स्थिर uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            vector चिन्हित लघु vscales0 = vec_splats((int16_t)(2*ls0+1));
            vector चिन्हित लघु vscales1 = vec_splats((int16_t)(2*ls1+1));
            vector चिन्हित लघु vscales2 = vec_splats((int16_t)(2*ls2+1));
            vector चिन्हित लघु vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_iq3_xxs_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_iq3_xxs * GGML_RESTRICT x = vx;
    स्थिर block_q8_K    * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    स्थिर uint64_t * signs64 = (स्थिर uint64_t *)keven_signs_q2xs;

    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;

        स्थिर uint8_t * GGML_RESTRICT q3 = x[i].qs;
        स्थिर uint32_t * GGML_RESTRICT signs = (स्थिर uint32_t *)(x[i].qs + QK_K/4);
        स्थिर int8_t  * GGML_RESTRICT q8 = y[i].qs;

#आशय GCC unroll 1
        क्रम (पूर्णांक j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector अचिन्हित पूर्णांक aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            vector अचिन्हित पूर्णांक aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            vector अचिन्हित पूर्णांक aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            vector अचिन्हित पूर्णांक aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            vector अचिन्हित दीर्घ दीर्घ aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            vector अचिन्हित दीर्घ दीर्घ aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            vector अचिन्हित दीर्घ दीर्घ aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            vector अचिन्हित दीर्घ दीर्घ aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            vector चिन्हित अक्षर q3x0 = vec_mul((vector चिन्हित अक्षर)aux64x2_0, (vector चिन्हित अक्षर)aux32x4_0);
            vector चिन्हित अक्षर q3x1 = vec_mul((vector चिन्हित अक्षर)aux64x2_1, (vector चिन्हित अक्षर)aux32x4_1);
            vector चिन्हित अक्षर q3x2 = vec_mul((vector चिन्हित अक्षर)aux64x2_2, (vector चिन्हित अक्षर)aux32x4_2);
            vector चिन्हित अक्षर q3x3 = vec_mul((vector चिन्हित अक्षर)aux64x2_3, (vector चिन्हित अक्षर)aux32x4_3);

            vector चिन्हित अक्षर q8y0 = vec_xl( 0, q8);
            vector चिन्हित अक्षर q8y1 = vec_xl(16, q8);
            vector चिन्हित अक्षर q8y2 = vec_xl(32, q8);
            vector चिन्हित अक्षर q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector चिन्हित लघु qv0 = vec_add(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            vector चिन्हित लघु qv1 = vec_add(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            vector चिन्हित लघु qv2 = vec_add(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            vector चिन्हित लघु qv3 = vec_add(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            स्थिर uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            स्थिर uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            vector चिन्हित लघु vscales01 = (vector चिन्हित लघु)vec_splats((uint16_t)(2*ls0+1));
            vector चिन्हित लघु vscales23 = (vector चिन्हित लघु)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.25f * vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_iq3_s_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_iq3_s * GGML_RESTRICT x = vx;
    स्थिर block_q8_K  * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    अटल स्थिर uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    अटल स्थिर uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    स्थिर vector अचिन्हित अक्षर mask0 = vec_xl( 0, k_mask1);
    स्थिर vector अचिन्हित अक्षर mask1 = vec_xl(16, k_mask1);
    स्थिर vector चिन्हित अक्षर mask2 = (vector चिन्हित अक्षर)vec_xl( 0, k_mask2);

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        स्थिर uint8_t *  GGML_RESTRICT q3 = x[i].qs;
        स्थिर uint8_t *  GGML_RESTRICT qh = x[i].qh;
        स्थिर uint16_t * GGML_RESTRICT signs = (स्थिर uint16_t *)(x[i].signs);
        स्थिर uint8_t *  GGML_RESTRICT sc = x[i].scales;
        स्थिर int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;

        क्रम (पूर्णांक j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector अचिन्हित पूर्णांक aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            vector अचिन्हित पूर्णांक aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            vector अचिन्हित पूर्णांक aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            vector अचिन्हित पूर्णांक aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            vector चिन्हित अक्षर vsigns01 = (vector चिन्हित अक्षर)vec_splats(*(स्थिर uint32_t *)&signs[0]);
            vector चिन्हित अक्षर vsigns02 = (vector चिन्हित अक्षर)vec_splats(*(स्थिर uint32_t *)&signs[2]);
            signs += 4;

            vector चिन्हित अक्षर vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            vector चिन्हित अक्षर vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            vector चिन्हित अक्षर vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            vector चिन्हित अक्षर vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (vector चिन्हित अक्षर)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (vector चिन्हित अक्षर)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (vector चिन्हित अक्षर)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (vector चिन्हित अक्षर)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            vector चिन्हित अक्षर q3x0 = vec_sub(vec_xor(vsigns0, (vector चिन्हित अक्षर)aux32x4_0), vsigns0);
            vector चिन्हित अक्षर q3x1 = vec_sub(vec_xor(vsigns1, (vector चिन्हित अक्षर)aux32x4_1), vsigns1);
            vector चिन्हित अक्षर q3x2 = vec_sub(vec_xor(vsigns2, (vector चिन्हित अक्षर)aux32x4_2), vsigns2);
            vector चिन्हित अक्षर q3x3 = vec_sub(vec_xor(vsigns3, (vector चिन्हित अक्षर)aux32x4_3), vsigns3);

            vector चिन्हित अक्षर q8y0 = vec_xl( 0, q8);
            vector चिन्हित अक्षर q8y1 = vec_xl(16, q8);
            vector चिन्हित अक्षर q8y2 = vec_xl(32, q8);
            vector चिन्हित अक्षर q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector चिन्हित लघु qv0 = vec_add(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            vector चिन्हित लघु qv1 = vec_add(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            vector चिन्हित लघु qv2 = vec_add(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            vector चिन्हित लघु qv3 = vec_add(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            स्थिर uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            स्थिर uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            vector चिन्हित लघु vscales01 = (vector चिन्हित लघु)vec_splats((uint16_t)(2*ls0+1));
            vector चिन्हित लघु vscales23 = (vector चिन्हित लघु)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_iq1_s_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(n % QK_K == 0);
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    स्थिर block_iq1_s * GGML_RESTRICT x = vx;
    स्थिर block_q8_K  * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector अचिन्हित अक्षर v0 = vec_splats((अचिन्हित अक्षर)0x0);
    स्थिर vector अचिन्हित लघु vsign = vec_splats((अचिन्हित लघु)0x8000);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    क्रम (पूर्णांक i = 0; i < nb; ++i) {
        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector भग्न vyd = vec_splats(y[i].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित पूर्णांक vsumi0 = vec_splats((int32_t)0);
        vector चिन्हित पूर्णांक vsumi1 = vec_splats((int32_t)0);
        vector चिन्हित पूर्णांक vsumi2 = vec_splats((int32_t)0);
        vector चिन्हित पूर्णांक vsumi3 = vec_splats((int32_t)0);
        vector चिन्हित पूर्णांक vsumi8 = vec_splats((int32_t)0);

        स्थिर uint8_t  * GGML_RESTRICT q1 = x[i].qs;
        स्थिर uint16_t * GGML_RESTRICT qh = x[i].qh;
        स्थिर int8_t   * GGML_RESTRICT q8 = y[i].qs;
        स्थिर int16_t  * GGML_RESTRICT qs = y[i].bsums;

        क्रम (पूर्णांक j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector चिन्हित दीर्घ दीर्घ aux64x2_0 = {*(स्थिर int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(स्थिर int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            vector चिन्हित दीर्घ दीर्घ aux64x2_1 = {*(स्थिर int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(स्थिर int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            vector चिन्हित दीर्घ दीर्घ aux64x2_2 = {*(स्थिर int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(स्थिर int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            vector चिन्हित दीर्घ दीर्घ aux64x2_3 = {*(स्थिर int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(स्थिर int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            vector चिन्हित अक्षर q1x0 = (vector चिन्हित अक्षर)aux64x2_0;
            vector चिन्हित अक्षर q1x1 = (vector चिन्हित अक्षर)aux64x2_1;
            vector चिन्हित अक्षर q1x2 = (vector चिन्हित अक्षर)aux64x2_2;
            vector चिन्हित अक्षर q1x3 = (vector चिन्हित अक्षर)aux64x2_3;

            vector चिन्हित अक्षर q8y0 = vec_xl( 0, q8);
            vector चिन्हित अक्षर q8y1 = vec_xl(16, q8);
            vector चिन्हित अक्षर q8y2 = vec_xl(32, q8);
            vector चिन्हित अक्षर q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector चिन्हित लघु qv0 = vec_add(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            vector चिन्हित लघु qv1 = vec_add(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            vector चिन्हित लघु qv2 = vec_add(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            vector चिन्हित लघु qv3 = vec_add(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            स्थिर uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            स्थिर uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            vector चिन्हित लघु vscales01 = (vector चिन्हित लघु)vec_splats((uint16_t)(2*ls0+1));
            vector चिन्हित लघु vscales23 = (vector चिन्हित लघु)vec_splats((uint16_t)(2*ls1+1));
            vector चिन्हित लघु vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            vector चिन्हित लघु q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (vector चिन्हित लघु)v0);

            vector चिन्हित लघु qxh = (vector चिन्हित लघु)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            vector __bool लघु vsel = vec_cmpge(qxh, (vector चिन्हित लघु)v0);

            vector चिन्हित लघु q8ysum = vec_sel((vector चिन्हित लघु)vec_xor((vector अचिन्हित लघु)q8ysums, vsign), q8ysums, vsel);

            vsumi8 = vec_add(vec_mule(q8ysum, vscales), vsumi8);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);

        vsumf0 = vec_madd(vec_ctf(vsumi8, 0), vec_mul(vd, vec_splats(IQ1S_DELTA)), vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_iq4_nl_q8_0(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    निश्चित(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    स्थिर block_iq4_nl * GGML_RESTRICT x = vx;
    स्थिर block_q8_0   * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK4_NL;

    पूर्णांक ib = 0;
    भग्न sumf = 0;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector चिन्हित पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);

    स्थिर vector चिन्हित अक्षर मूल्यावली = vec_xl( 0, kvalues_iq4nl);

#आशय GCC unroll 4
    क्रम (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector भग्न vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित अक्षर qxs = (vector चिन्हित अक्षर)vec_xl( 0, x[ib].qs);
        vector चिन्हित अक्षर q4x0 = vec_and(qxs, निचला_Mask);
        vector चिन्हित अक्षर q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(मूल्यावली, मूल्यावली, (vector अचिन्हित अक्षर)q4x0);
        q4x1 = vec_perm(मूल्यावली, मूल्यावली, (vector अचिन्हित अक्षर)q4x1);

        vector चिन्हित अक्षर q8y0 = vec_xl( 0, y[ib].qs);
        vector चिन्हित अक्षर q8y1 = vec_xl(16, y[ib].qs);

        vector चिन्हित लघु qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector चिन्हित लघु qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    }

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}

व्योम ggml_vec_dot_iq4_xs_q8_K(पूर्णांक n, भग्न * GGML_RESTRICT s, माप_प्रकार bs, स्थिर व्योम * GGML_RESTRICT vx, माप_प्रकार bx, स्थिर व्योम * GGML_RESTRICT vy, माप_प्रकार by, पूर्णांक nrc) {
    निश्चित(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    निश्चित(n % QK_K == 0);

    स्थिर block_iq4_xs * GGML_RESTRICT x = vx;
    स्थिर block_q8_K   * GGML_RESTRICT y = vy;

    स्थिर पूर्णांक nb = n / QK_K;

#अगर defined(__POWER9_VECTOR__)
    स्थिर vector चिन्हित अक्षर निचला_Mask = vec_splats((चिन्हित अक्षर)0xF);
    स्थिर vector पूर्णांक v0 = vec_splats((int32_t)0);
    स्थिर vector अचिन्हित अक्षर v4 = vec_splats((अचिन्हित अक्षर)0x4);

    vector भग्न vsumf0 = vec_splats(0.0f);
    vector भग्न vsumf1 = vec_splats(0.0f);
    vector भग्न vsumf2 = vec_splats(0.0f);
    vector भग्न vsumf3 = vec_splats(0.0f);

    स्थिर vector चिन्हित अक्षर मूल्यावली = vec_xl( 0, kvalues_iq4nl);

    क्रम (पूर्णांक ibl = 0; ibl < nb; ++ibl) {

        vector भग्न vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ibl].d));
        vector भग्न vyd = vec_splats(y[ibl].d);
        vector भग्न vd = vec_mul(vxd, vyd);

        vector चिन्हित पूर्णांक vsumi0 = v0;
        vector चिन्हित पूर्णांक vsumi1 = v0;
        vector चिन्हित पूर्णांक vsumi2 = v0;
        vector चिन्हित पूर्णांक vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        स्थिर uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        स्थिर uint8_t * GGML_RESTRICT sc = x[ibl].scales_l;
        स्थिर int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        क्रम (पूर्णांक ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector चिन्हित अक्षर qxs0 = (vector चिन्हित अक्षर)vec_xl( 0, q4);
            vector चिन्हित अक्षर qxs1 = (vector चिन्हित अक्षर)vec_xl(16, q4);
            q4 += 32;

            vector चिन्हित अक्षर q4x00 = (vector चिन्हित अक्षर)vec_and(qxs0, निचला_Mask);
            vector चिन्हित अक्षर q4x01 = (vector चिन्हित अक्षर)vec_sr(qxs0, v4);
            vector चिन्हित अक्षर q4x10 = (vector चिन्हित अक्षर)vec_and(qxs1, निचला_Mask);
            vector चिन्हित अक्षर q4x11 = (vector चिन्हित अक्षर)vec_sr(qxs1, v4);

            q4x00 = vec_perm(मूल्यावली, मूल्यावली, (vector अचिन्हित अक्षर)q4x00);
            q4x01 = vec_perm(मूल्यावली, मूल्यावली, (vector अचिन्हित अक्षर)q4x01);
            q4x10 = vec_perm(मूल्यावली, मूल्यावली, (vector अचिन्हित अक्षर)q4x10);
            q4x11 = vec_perm(मूल्यावली, मूल्यावली, (vector अचिन्हित अक्षर)q4x11);

            vector चिन्हित अक्षर q8y0 = vec_xl( 0, q8);
            vector चिन्हित अक्षर q8y1 = vec_xl(16, q8);
            vector चिन्हित अक्षर q8y2 = vec_xl(32, q8);
            vector चिन्हित अक्षर q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector चिन्हित लघु qv0 = vec_add(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            vector चिन्हित लघु qv1 = vec_add(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            vector चिन्हित लघु qv2 = vec_add(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            vector चिन्हित लघु qv3 = vec_add(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            स्थिर uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            स्थिर uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            vector चिन्हित लघु vscales01 = vec_splats((int16_t)ls0);
            vector चिन्हित लघु vscales23 = vec_splats((int16_t)ls1);

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

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#अन्यथा
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#पूर्ण_अगर
}
