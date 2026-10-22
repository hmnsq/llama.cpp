#تعریف_کن GGML_COMMON_IMPL_C
#ضمن "ggml-common.h"
#ضمن "ggml-quants.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-cpu.h"
#ضمن "simd-mappings.h"

#ضمن "../../quants.h"
#ضمن "../../ggml-cpu-impl.h"

#ضمن <math.h>
#ضمن <string.h>
#ضمن <assert.h>
#ضمن <float.h>
#ضمن <stdlib.h> // for qsort
#ضمن <stdio.h>  // for GGML_ASSERT

#تعریف_کن گروه_MAX_EPS 1e-15f
#تعریف_کن گروه_MAX_EPS_IQ3_XXS 1e-8f
#تعریف_کن گروه_MAX_EPS_IQ2_S 1e-8f
#تعریف_کن گروه_MAX_EPS_IQ1_M 1e-7f
#تعریف_کن گروه_MAX_EPS_IQ1_S 1e-12f

#تعریف_کن استفادهنشده GGML_استفادهنشده

#إذا defined(__POWER9_VECTOR__)
#تعریف_کن B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#تعریف_کن B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#تعریف_کن B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#تعریف_کن B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#تعریف_کن B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#تعریف_کن B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#تعریف_کن B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#تعریف_کن B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
ساكن ثابت uint64_t جدول_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
ساكن ثابت uint64_t جدول_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#پایان_اگر

فراغ quantize_ردیف_q8_0(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k) {
    أكد(QK8_0 == 32);
    أكد(k % QK8_0 == 0);
    ثابت صحيح nb = k / QK8_0;

    بلوک_q8_0 * GGML_RESTRICT y = vy;

#إذا defined(__POWER9_VECTOR__)
    لكل (صحيح i = 0; i < nb; i++) {
        وکتور عائم srcv [8];
        وکتور عائم asrcv[8];
        وکتور عائم amaxv[8];
        وکتور علامتدار صحيح vi[8];

        لكل (صحيح j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        لكل (صحيح j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        لكل (صحيح j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        لكل (صحيح j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        لكل (صحيح j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ثابت عائم amax = MAX(MAX(vec_استخراج(amaxv[0], 0),
                                   vec_استخراج(amaxv[0], 1)),
                               MAX(vec_استخراج(amaxv[0], 2),
                                   vec_استخراج(amaxv[0], 3)));

        ثابت عائم d = amax / ((1 << 7) - 1);
        ثابت عائم id = d ? 1.0f/d : 0.0f;
        ثابت وکتور عائم vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_مقصد_FP16(d);

        لكل (صحيح j = 0; j < 8; j++) {
            ثابت وکتور عائم v  = vec_دور(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }
#وإلا
    GGML_استفادهنشده(nb);
    // scalar
    quantize_ردیف_q8_0_ref(x, y, k);
#پایان_اگر
}

فراغ quantize_ردیف_q8_1(ثابت عائم * GGML_RESTRICT x, فراغ * GGML_RESTRICT vy, int64_t k) {
    أكد(k % QK8_1 == 0);
    ثابت صحيح nb = k / QK8_1;

    بلوک_q8_1 * GGML_RESTRICT y = vy;

#إذا defined(__POWER9_VECTOR__)
    لكل (صحيح i = 0; i < nb; i++) {
        وکتور عائم srcv [8];
        وکتور عائم asrcv[8];
        وکتور عائم amaxv[8];
        وکتور علامتدار صحيح vi[8];

        لكل (صحيح j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        لكل (صحيح j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        لكل (صحيح j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        لكل (صحيح j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        لكل (صحيح j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ثابت عائم amax = MAX(MAX(vec_استخراج(amaxv[0], 0),
                                   vec_استخراج(amaxv[0], 1)),
                               MAX(vec_استخراج(amaxv[0], 2),
                                   vec_استخراج(amaxv[0], 3)));

        ثابت عائم d = amax / ((1 << 7) - 1);
        ثابت عائم id = d ? 1.0f/d : 0.0f;
        ثابت وکتور عائم vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_مقصد_FP16(d);

        وکتور صحيح accv = vec_splats(0);

        لكل (صحيح j = 0; j < 8; j++) {
            ثابت وکتور عائم v  = vec_دور(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_جمع_کن(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_جمع_کن(accv, vec_sld(accv, accv, 4));
        accv = vec_جمع_کن(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_CPU_FP32_مقصد_FP16(d * vec_استخراج(accv, 0));
    }

#وإلا
    GGML_استفادهنشده(nb);
    // scalar
    quantize_ردیف_q8_1_ref(x, y, k);
#پایان_اگر
}


//===================================== Dot products =================================

فراغ ggml_vec_dot_q4_0_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_q4_0 * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_0 * GGML_RESTRICT y = vy;

    صحيح ib = 0;
    عائم sumf = 0;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور علامتدار صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);
    ثابت وکتور علامتدار حرف v8 = vec_splats((علامتدار حرف)0x8);

    وکتور عائم vsumf0 = vec_splats(0.0f);

#رهنمود GCC unroll 8
    لكل (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        وکتور عائم vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار حرف qxs = (وکتور علامتدار حرف)vec_xl( 0, x[ib].qs);
        وکتور علامتدار حرف q8y0 = vec_xl( 0, y[ib].qs);
        وکتور علامتدار حرف q8y1 = vec_xl(16, y[ib].qs);

        وکتور علامتدار حرف q4x0 = vec_and(qxs, کم_ماسک);
        وکتور علامتدار حرف q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        وکتور علامتدار صحيح vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_استخراج(vsumf0, 0);

    *s = sumf;
#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(ib);
    استفادهنشده(sumf);
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_q4_1_q8_1(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    ثابت صحيح qk = QK8_1;
    ثابت صحيح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_q4_1 * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_1 * GGML_RESTRICT y = vy;

    صحيح ib = 0;
    عائم sumf = 0;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور علامتدار صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);

    وکتور عائم vsumf0 = vec_splats(0.0f);

#رهنمود GCC unroll 4
    لكل (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        وکتور عائم vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور عائم vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].m));
        وکتور عائم vys = {GGML_CPU_FP16_مقصد_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        وکتور علامتدار حرف qxs = (وکتور علامتدار حرف)vec_xl( 0, x[ib].qs);
        وکتور علامتدار حرف q8y0 = vec_xl( 0, y[ib].qs);
        وکتور علامتدار حرف q8y1 = vec_xl(16, y[ib].qs);

        وکتور بیعلامت حرف q4x0 = (وکتور بیعلامت حرف)vec_and(qxs, کم_ماسک);
        وکتور بیعلامت حرف q4x1 = (وکتور بیعلامت حرف)vec_sr(qxs, v4);

        وکتور علامتدار صحيح vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_استخراج(vsumf0, 0);

    *s = sumf;
#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(ib);
    استفادهنشده(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_mxfp4_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);
    أكد(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ثابت بلوک_mxfp4 * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_0 * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_MXFP4;

    صحيح ib = 0;
    عائم sumf = 0;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور بیعلامت حرف vshift4 = vec_splats((بیعلامت حرف)4);
    وکتور عائم vsumf0 = vec_splats(0.0f);

    وکتور علامتدار حرف kv = vec_xl(0, (ثابت علامتدار حرف *)kvalues_mxfp4);

#رهنمود GCC unroll 8
    لكل (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        وکتور عائم vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d) *
                                      GGML_E8_M0_مقصد_FP32_HALF(x[ib].e));

        وکتور علامتدار حرف q8y0 = vec_xl( 0, y[ib].qs);
        وکتور علامتدار حرف q8y1 = vec_xl(16, y[ib].qs);

        وکتور علامتدار حرف qxs = (وکتور علامتدار حرف)vec_xl(0, x[ib].qs);

        وکتور بیعلامت حرف lo_nibbles = (وکتور بیعلامت حرف)vec_and(qxs, کم_ماسک);
        وکتور بیعلامت حرف hi_nibbles = (وکتور بیعلامت حرف)vec_sr(qxs, vshift4);

        وکتور علامتدار حرف q4x0 = vec_perm(kv, kv, lo_nibbles);
        وکتور علامتدار حرف q4x1 = vec_perm(kv, kv, hi_nibbles);

        وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        وکتور علامتدار صحيح vsumi0 = vec_splats((int32_t)0);
        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vyd, vsumf0);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));
    sumf = vec_استخراج(vsumf0, 0);
    *s = sumf;
#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(ib);
    استفادهنشده(sumf);
    ggml_vec_dot_mxfp4_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_q5_0_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;

    صحيح ib = 0;
    عائم sumf = 0;

    أكد(n % qk == 0);
    أكد(qk == QK5_0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_q5_0 * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_0 * GGML_RESTRICT y = vy;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)4);

    وکتور عائم vsumf0 = vec_splats(0.0f);

#رهنمود GCC unroll 4
    لكل (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        وکتور عائم vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار دراز دراز aux64x2_0 = {(uint64_t)(جدول_b2b_1[x[ib].qh[0]]), (uint64_t)(جدول_b2b_1[x[ib].qh[1]])};
        وکتور علامتدار دراز دراز aux64x2_1 = {(uint64_t)(جدول_b2b_1[x[ib].qh[2]]), (uint64_t)(جدول_b2b_1[x[ib].qh[3]])};

        وکتور علامتدار حرف qh0 = (وکتور علامتدار حرف)aux64x2_0;
        وکتور علامتدار حرف qh1 = (وکتور علامتدار حرف)aux64x2_1;

        وکتور علامتدار حرف qxs = (وکتور علامتدار حرف)vec_xl( 0, x[ib].qs);

        وکتور علامتدار حرف q5x0 = vec_sub(vec_and (qxs, کم_ماسک), qh0);
        وکتور علامتدار حرف q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

        وکتور علامتدار حرف q8y0 = vec_xl(  0, y[ib].qs);
        وکتور علامتدار حرف q8y1 = vec_xl( 16, y[ib].qs);

        وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
        وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q5x1, q8y1), vec_mulo(q5x1, q8y1));

        qv0 = vec_جمع_کن(qv0, qv1);

        وکتور علامتدار صحيح vsumi0 = vec_جمع_کن(vec_unpackh(qv0), vec_unpackl(qv0));

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_استخراج(vsumf0, 0);

    *s = sumf;
#وإلا
    استفادهنشده(ib);
    استفادهنشده(sumf);
    استفادهنشده(x);
    استفادهنشده(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_q5_1_q8_1(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    ثابت صحيح qk = QK8_1;
    ثابت صحيح nb = n / qk;

    صحيح ib = 0;
    عائم sumf = 0;

    أكد(n % qk == 0);
    أكد(qk == QK5_1);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_q5_1 * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_1 * GGML_RESTRICT y = vy;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور علامتدار صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);

    وکتور عائم vsumf0 = vec_splats(0.0f);

#رهنمود GCC unroll 4
    لكل (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        وکتور عائم vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور عائم vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].m));
        وکتور عائم vys = {GGML_CPU_FP16_مقصد_FP32(y[ib].s), 0.f, 0.f, 0.f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        وکتور بیعلامت دراز دراز aux64x2_0 = {(uint64_t)(جدول_b2b_0[x[ib].qh[0]]), (uint64_t)(جدول_b2b_0[x[ib].qh[1]])};
        وکتور بیعلامت دراز دراز aux64x2_1 = {(uint64_t)(جدول_b2b_0[x[ib].qh[2]]), (uint64_t)(جدول_b2b_0[x[ib].qh[3]])};

        وکتور علامتدار حرف qh0 = (وکتور علامتدار حرف)aux64x2_0;
        وکتور علامتدار حرف qh1 = (وکتور علامتدار حرف)aux64x2_1;

        وکتور علامتدار حرف qxs = (وکتور علامتدار حرف)vec_xl( 0, x[ib].qs);

        وکتور بیعلامت حرف q5x0 = (وکتور بیعلامت حرف)vec_or(vec_and(qxs, کم_ماسک), qh0);
        وکتور بیعلامت حرف q5x1 = (وکتور بیعلامت حرف)vec_or(vec_sr(qxs, v4), qh1);

        وکتور علامتدار حرف q8y0 = vec_xl(  0, y[ib].qs);
        وکتور علامتدار حرف q8y1 = vec_xl( 16, y[ib].qs);

        وکتور علامتدار صحيح vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q5x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q5x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_استخراج(vsumf0, 0);

    *s = sumf;
#وإلا
    استفادهنشده(nb);
    استفادهنشده(ib);
    استفادهنشده(sumf);
    استفادهنشده(x);
    استفادهنشده(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_q8_0_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    ثابت صحيح qk = QK8_0;
    ثابت صحيح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_q8_0 * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_0 * GGML_RESTRICT y = vy;

    صحيح ib = 0;
    عائم sumf = 0;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار صحيح v0 = vec_splats((int32_t)0);
    وکتور عائم vsumf0 = vec_splats(0.0f);

#رهنمود GCC unroll 8
    لكل (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        وکتور عائم vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار حرف q8x0 = vec_xl( 0, x[ib].qs);
        وکتور علامتدار حرف q8x1 = vec_xl(16, x[ib].qs);
        وکتور علامتدار حرف q8y0 = vec_xl( 0, y[ib].qs);
        وکتور علامتدار حرف q8y1 = vec_xl(16, y[ib].qs);

        وکتور علامتدار کوتاه qv0 = vec_mule(q8x0, q8y0);
        وکتور علامتدار کوتاه qv1 = vec_mulo(q8x0, q8y0);
        وکتور علامتدار کوتاه qv2 = vec_mule(q8x1, q8y1);
        وکتور علامتدار کوتاه qv3 = vec_mulo(q8x1, q8y1);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_جمع_کن(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_استخراج(vsumf0, 0);

    *s = sumf;
#وإلا
    استفادهنشده(nb);
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(ib);
    استفادهنشده(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_q2_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_q2_K * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0x3);
    ثابت وکتور علامتدار حرف کم_Scale_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور بیعلامت حرف v2 = vec_splats((بیعلامت حرف)0x2);
    ثابت وکتور بیعلامت حرف v6 = vec_splats((بیعلامت حرف)0x6);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور عائم vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].dmin));
        وکتور عائم vdmin = vec_mul(vxmin, vyd);

        وکتور علامتدار کوتاه q8ysums0 = vec_xl( 0, y[i].bsums);
        وکتور علامتدار کوتاه q8ysums1 = vec_xl(16, y[i].bsums);

        وکتور علامتدار حرف q2xmins = (وکتور علامتدار حرف)vec_xl( 0, x[i].scales);
        وکتور علامتدار حرف vscales = vec_and(q2xmins, کم_Scale_ماسک);

        q2xmins = vec_sr(q2xmins, v4);
        وکتور علامتدار کوتاه q2xmins0 = vec_unpackh(q2xmins);
        وکتور علامتدار کوتاه q2xmins1 = vec_unpackl(q2xmins);

        وکتور علامتدار صحيح prod0 = vec_mule(q2xmins0, q8ysums0);
        وکتور علامتدار صحيح prod1 = vec_mulo(q2xmins0, q8ysums0);
        وکتور علامتدار صحيح prod2 = vec_mule(q2xmins1, q8ysums1);
        وکتور علامتدار صحيح prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;
        وکتور علامتدار صحيح vsumi4 = v0;
        وکتور علامتدار صحيح vsumi5 = v0;
        وکتور علامتدار صحيح vsumi6 = v0;
        وکتور علامتدار صحيح vsumi7 = v0;

        ثابت uint8_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        لكل (صحيح j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور علامتدار حرف qxs0 = (وکتور علامتدار حرف)vec_xl( 0, q2);
            وکتور علامتدار حرف qxs1 = (وکتور علامتدار حرف)vec_xl(16, q2);
            q2 += 32;

            وکتور بیعلامت حرف q2x00 = (وکتور بیعلامت حرف)vec_and(qxs0, کم_ماسک);
            وکتور بیعلامت حرف q2x01 = (وکتور بیعلامت حرف)vec_and(vec_sr(qxs0, v2), کم_ماسک);
            وکتور بیعلامت حرف q2x02 = (وکتور بیعلامت حرف)vec_and(vec_sr(qxs0, v4), کم_ماسک);
            وکتور بیعلامت حرف q2x03 = (وکتور بیعلامت حرف)vec_and(vec_sr(qxs0, v6), کم_ماسک);
            وکتور بیعلامت حرف q2x10 = (وکتور بیعلامت حرف)vec_and(qxs1, کم_ماسک);
            وکتور بیعلامت حرف q2x11 = (وکتور بیعلامت حرف)vec_and(vec_sr(qxs1, v2), کم_ماسک);
            وکتور بیعلامت حرف q2x12 = (وکتور بیعلامت حرف)vec_and(vec_sr(qxs1, v4), کم_ماسک);
            وکتور بیعلامت حرف q2x13 = (وکتور بیعلامت حرف)vec_and(vec_sr(qxs1, v6), کم_ماسک);

            وکتور علامتدار حرف q8y00 = vec_xl(  0, q8);
            وکتور علامتدار حرف q8y10 = vec_xl( 16, q8);
            وکتور علامتدار حرف q8y01 = vec_xl( 32, q8);
            وکتور علامتدار حرف q8y11 = vec_xl( 48, q8);
            وکتور علامتدار حرف q8y02 = vec_xl( 64, q8);
            وکتور علامتدار حرف q8y12 = vec_xl( 80, q8);
            وکتور علامتدار حرف q8y03 = vec_xl( 96, q8);
            وکتور علامتدار حرف q8y13 = vec_xl(112, q8);
            q8 += 128;

            وکتور علامتدار صحيح qv0 = vec_msum(q8y00, q2x00, v0);
            وکتور علامتدار صحيح qv1 = vec_msum(q8y01, q2x01, v0);
            وکتور علامتدار صحيح qv2 = vec_msum(q8y02, q2x02, v0);
            وکتور علامتدار صحيح qv3 = vec_msum(q8y03, q2x03, v0);
            وکتور علامتدار صحيح qv4 = vec_msum(q8y10, q2x10, v0);
            وکتور علامتدار صحيح qv5 = vec_msum(q8y11, q2x11, v0);
            وکتور علامتدار صحيح qv6 = vec_msum(q8y12, q2x12, v0);
            وکتور علامتدار صحيح qv7 = vec_msum(q8y13, q2x13, v0);

            وکتور علامتدار کوتاه vscales_07 = vec_unpackh(vscales);
            وکتور علامتدار صحيح vscales_03 = vec_unpackh(vscales_07);
            وکتور علامتدار صحيح vscales_47 = vec_unpackl(vscales_07);
            وکتور علامتدار صحيح vs0 = vec_splat(vscales_03, 0);
            وکتور علامتدار صحيح vs1 = vec_splat(vscales_03, 1);
            وکتور علامتدار صحيح vs2 = vec_splat(vscales_03, 2);
            وکتور علامتدار صحيح vs3 = vec_splat(vscales_03, 3);
            وکتور علامتدار صحيح vs4 = vec_splat(vscales_47, 0);
            وکتور علامتدار صحيح vs5 = vec_splat(vscales_47, 1);
            وکتور علامتدار صحيح vs6 = vec_splat(vscales_47, 2);
            وکتور علامتدار صحيح vs7 = vec_splat(vscales_47, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_جمع_کن(vec_mul(qv0, vs0), vsumi0);
            vsumi1 = vec_جمع_کن(vec_mul(qv1, vs2), vsumi1);
            vsumi2 = vec_جمع_کن(vec_mul(qv2, vs4), vsumi2);
            vsumi3 = vec_جمع_کن(vec_mul(qv3, vs6), vsumi3);
            vsumi4 = vec_جمع_کن(vec_mul(qv4, vs1), vsumi4);
            vsumi5 = vec_جمع_کن(vec_mul(qv5, vs3), vsumi5);
            vsumi6 = vec_جمع_کن(vec_mul(qv6, vs5), vsumi6);
            vsumi7 = vec_جمع_کن(vec_mul(qv7, vs7), vsumi7);
        }

        vsumi0 = vec_جمع_کن(vsumi0, vsumi4);
        vsumi1 = vec_جمع_کن(vsumi1, vsumi5);
        vsumi2 = vec_جمع_کن(vsumi2, vsumi6);
        vsumi3 = vec_جمع_کن(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_q3_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت uint32_t kmask1 = 0x03030303;
    ثابت uint32_t kmask2 = 0x0f0f0f0f;

    ثابت بلوک_q3_K * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0x3);
    ثابت وکتور علامتدار حرف کم_Mask1 = vec_splats((int8_t)0xf);
    ثابت وکتور علامتدار حرف کم_Mask2 = vec_splats((int8_t)0x30);
    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور علامتدار حرف v1 = vec_splats((علامتدار حرف)0x1);
    ثابت وکتور بیعلامت حرف v2 = vec_splats((بیعلامت حرف)0x2);
    ثابت وکتور بیعلامت حرف v3 = vec_splats((بیعلامت حرف)0x3);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);
    ثابت وکتور بیعلامت حرف v6 = vec_splats((بیعلامت حرف)0x6);
    ثابت وکتور علامتدار حرف off = vec_splats((علامتدار حرف)0x20);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        استفادهنشده(kmask1);
        استفادهنشده(kmask2);

        وکتور علامتدار حرف u0 = (وکتور علامتدار حرف)vec_xl_len(x[i].scales, 8);
        وکتور علامتدار حرف u1 = vec_and(u0, کم_Mask1);
        وکتور علامتدار حرف u2 = (وکتور علامتدار حرف)vec_xl_len(x[i].scales + 8, 4);
        وکتور علامتدار حرف u3 = (وکتور علامتدار حرف)vec_mergeh((وکتور علامتدار صحيح)u2, (وکتور علامتدار صحيح)vec_sr(u2, v2));
        وکتور علامتدار حرف u30 = vec_sl(vec_and(u3, کم_ماسک), v4);
        وکتور علامتدار حرف u31 = vec_and(u3, کم_Mask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        وکتور علامتدار حرف vscales = (وکتور علامتدار حرف)vec_mergeh((وکتور علامتدار دراز دراز)u1, (وکتور علامتدار دراز دراز)u2);
        وکتور علامتدار حرف qxhs0 = (وکتور علامتدار حرف)vec_xl( 0, x[i].hmask);
        وکتور علامتدار حرف qxhs1 = (وکتور علامتدار حرف)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;
        وکتور علامتدار صحيح vsumi4 = v0;
        وکتور علامتدار صحيح vsumi5 = v0;
        وکتور علامتدار صحيح vsumi6 = v0;
        وکتور علامتدار صحيح vsumi7 = v0;

        ثابت uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        لكل (صحيح j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور علامتدار حرف qxs0 = (وکتور علامتدار حرف)vec_xl( 0, q3);
            وکتور علامتدار حرف qxs1 = (وکتور علامتدار حرف)vec_xl(16, q3);
            q3 += 32;

            //the low 2 bits
            وکتور علامتدار حرف qxs00 = vec_and(qxs0, کم_ماسک);
            وکتور علامتدار حرف qxs01 = vec_and(vec_sr(qxs0, v2), کم_ماسک);
            وکتور علامتدار حرف qxs02 = vec_and(vec_sr(qxs0, v4), کم_ماسک);
            وکتور علامتدار حرف qxs03 = vec_and(vec_sr(qxs0, v6), کم_ماسک);
            وکتور علامتدار حرف qxs10 = vec_and(qxs1, کم_ماسک);
            وکتور علامتدار حرف qxs11 = vec_and(vec_sr(qxs1, v2), کم_ماسک);
            وکتور علامتدار حرف qxs12 = vec_and(vec_sr(qxs1, v4), کم_ماسک);
            وکتور علامتدار حرف qxs13 = vec_and(vec_sr(qxs1, v6), کم_ماسک);

            //the 3rd bit
            وکتور علامتدار حرف qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            وکتور علامتدار حرف qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (وکتور بیعلامت حرف)v1)), v2);
            وکتور علامتدار حرف qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            وکتور علامتدار حرف qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            وکتور علامتدار حرف qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            وکتور علامتدار حرف qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (وکتور بیعلامت حرف)v1)), v2);
            وکتور علامتدار حرف qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            وکتور علامتدار حرف qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            وکتور علامتدار حرف q3x00 = vec_sub(qxs00, qxh00);
            وکتور علامتدار حرف q3x01 = vec_sub(qxs01, qxh01);
            وکتور علامتدار حرف q3x02 = vec_sub(qxs02, qxh02);
            وکتور علامتدار حرف q3x03 = vec_sub(qxs03, qxh03);
            وکتور علامتدار حرف q3x10 = vec_sub(qxs10, qxh10);
            وکتور علامتدار حرف q3x11 = vec_sub(qxs11, qxh11);
            وکتور علامتدار حرف q3x12 = vec_sub(qxs12, qxh12);
            وکتور علامتدار حرف q3x13 = vec_sub(qxs13, qxh13);

            وکتور علامتدار حرف q8y00 = vec_xl(  0, q8);
            وکتور علامتدار حرف q8y10 = vec_xl( 16, q8);
            وکتور علامتدار حرف q8y01 = vec_xl( 32, q8);
            وکتور علامتدار حرف q8y11 = vec_xl( 48, q8);
            وکتور علامتدار حرف q8y02 = vec_xl( 64, q8);
            وکتور علامتدار حرف q8y12 = vec_xl( 80, q8);
            وکتور علامتدار حرف q8y03 = vec_xl( 96, q8);
            وکتور علامتدار حرف q8y13 = vec_xl(112, q8);
            q8 += 128;

            وکتور علامتدار کوتاه vscales_h = vec_unpackh(vscales);
            وکتور علامتدار کوتاه vs0 = vec_splat(vscales_h, 0);
            وکتور علامتدار کوتاه vs1 = vec_splat(vscales_h, 1);
            وکتور علامتدار کوتاه vs2 = vec_splat(vscales_h, 2);
            وکتور علامتدار کوتاه vs3 = vec_splat(vscales_h, 3);
            وکتور علامتدار کوتاه vs4 = vec_splat(vscales_h, 4);
            وکتور علامتدار کوتاه vs5 = vec_splat(vscales_h, 5);
            وکتور علامتدار کوتاه vs6 = vec_splat(vscales_h, 6);
            وکتور علامتدار کوتاه vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            وکتور علامتدار کوتاه qv00 = vec_جمع_کن(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            وکتور علامتدار کوتاه qv01 = vec_جمع_کن(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            وکتور علامتدار کوتاه qv02 = vec_جمع_کن(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            وکتور علامتدار کوتاه qv03 = vec_جمع_کن(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            وکتور علامتدار کوتاه qv10 = vec_جمع_کن(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            وکتور علامتدار کوتاه qv11 = vec_جمع_کن(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            وکتور علامتدار کوتاه qv12 = vec_جمع_کن(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            وکتور علامتدار کوتاه qv13 = vec_جمع_کن(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs2, vsumi1);
            vsumi2 = vec_msum(qv02, vs4, vsumi2);
            vsumi3 = vec_msum(qv03, vs6, vsumi3);
            vsumi4 = vec_msum(qv10, vs1, vsumi4);
            vsumi5 = vec_msum(qv11, vs3, vsumi5);
            vsumi6 = vec_msum(qv12, vs5, vsumi6);
            vsumi7 = vec_msum(qv13, vs7, vsumi7);
        }

        vsumi0 = vec_جمع_کن(vsumi0, vsumi4);
        vsumi1 = vec_جمع_کن(vsumi1, vsumi5);
        vsumi2 = vec_جمع_کن(vsumi2, vsumi6);
        vsumi3 = vec_جمع_کن(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(kmask1);
    استفادهنشده(kmask2);
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_q4_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_q4_K * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

    ساكن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساكن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساكن ثابت uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور علامتدار حرف کم_Mask1 = vec_splats((int8_t)0x3f);
    ثابت وکتور علامتدار حرف کم_Mask2 = vec_splats((int8_t)0x30);
    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور بیعلامت حرف v2 = vec_splats((uint8_t)2);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور عائم vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].dmin));
        وکتور عائم vdmin = vec_mul(vxmin, vyd);

        وکتور علامتدار کوتاه q8ysums0 = vec_xl( 0, y[i].bsums);
        وکتور علامتدار کوتاه q8ysums1 = vec_xl(16, y[i].bsums);

        استفادهنشده(kmask1);
        استفادهنشده(kmask2);
        استفادهنشده(kmask3);
        استفادهنشده(utmp);

        وکتور علامتدار حرف u0 = (وکتور علامتدار حرف)vec_xl_len(x[i].scales, 8);
        وکتور علامتدار حرف u1 = vec_and(vec_sr(u0, v2), کم_Mask2);
        وکتور علامتدار حرف u2 = (وکتور علامتدار حرف)vec_xl_len(x[i].scales + 8, 4);
        وکتور علامتدار حرف u3 = vec_sr(u2, v4);

        وکتور علامتدار حرف u30 = u1;
        وکتور علامتدار حرف u31 = (وکتور علامتدار حرف)vec_mergeh((وکتور علامتدار صحيح)vec_and(u2, کم_ماسک), (وکتور علامتدار صحيح)u3);

        u1 = vec_and(u0, کم_Mask1);
        u2 = vec_or(u30, u31);

        وکتور علامتدار حرف utmps = (وکتور علامتدار حرف)vec_mergeh((وکتور علامتدار صحيح)u1, (وکتور علامتدار صحيح)u2);

        وکتور علامتدار کوتاه vscales = vec_unpackh(utmps);
        وکتور علامتدار کوتاه q4xmins = vec_unpackl(utmps);
        وکتور علامتدار کوتاه q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        وکتور علامتدار کوتاه q4xmins1 = vec_mergel(q4xmins, q4xmins);

        وکتور علامتدار صحيح prod0 = vec_mule(q4xmins0, q8ysums0);
        وکتور علامتدار صحيح prod1 = vec_mule(q4xmins1, q8ysums1);
        وکتور علامتدار صحيح prod2 = vec_mulo(q4xmins0, q8ysums0);
        وکتور علامتدار صحيح prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;

        ثابت uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        لكل (صحيح j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور علامتدار حرف qxs0 = (وکتور علامتدار حرف)vec_xl( 0, q4);
            وکتور علامتدار حرف qxs1 = (وکتور علامتدار حرف)vec_xl(16, q4);
            وکتور علامتدار حرف qxs2 = (وکتور علامتدار حرف)vec_xl(32, q4);
            وکتور علامتدار حرف qxs3 = (وکتور علامتدار حرف)vec_xl(48, q4);
            q4 += 64;

            وکتور بیعلامت حرف q4x00 = (وکتور بیعلامت حرف)vec_and(qxs0, کم_ماسک);
            وکتور بیعلامت حرف q4x01 = (وکتور بیعلامت حرف)vec_sr(qxs0, v4);
            وکتور بیعلامت حرف q4x10 = (وکتور بیعلامت حرف)vec_and(qxs1, کم_ماسک);
            وکتور بیعلامت حرف q4x11 = (وکتور بیعلامت حرف)vec_sr(qxs1, v4);
            وکتور بیعلامت حرف q4x20 = (وکتور بیعلامت حرف)vec_and(qxs2, کم_ماسک);
            وکتور بیعلامت حرف q4x21 = (وکتور بیعلامت حرف)vec_sr(qxs2, v4);
            وکتور بیعلامت حرف q4x30 = (وکتور بیعلامت حرف)vec_and(qxs3, کم_ماسک);
            وکتور بیعلامت حرف q4x31 = (وکتور بیعلامت حرف)vec_sr(qxs3, v4);

            وکتور علامتدار حرف q8y00 = vec_xl(  0, q8);
            وکتور علامتدار حرف q8y10 = vec_xl( 16, q8);
            وکتور علامتدار حرف q8y01 = vec_xl( 32, q8);
            وکتور علامتدار حرف q8y11 = vec_xl( 48, q8);
            وکتور علامتدار حرف q8y20 = vec_xl( 64, q8);
            وکتور علامتدار حرف q8y30 = vec_xl( 80, q8);
            وکتور علامتدار حرف q8y21 = vec_xl( 96, q8);
            وکتور علامتدار حرف q8y31 = vec_xl(112, q8);
            q8 += 128;

            وکتور علامتدار صحيح qv00 = vec_msum(q8y00, q4x00, v0);
            وکتور علامتدار صحيح qv01 = vec_msum(q8y01, q4x01, v0);
            وکتور علامتدار صحيح qv10 = vec_msum(q8y10, q4x10, v0);
            وکتور علامتدار صحيح qv11 = vec_msum(q8y11, q4x11, v0);
            وکتور علامتدار صحيح qv20 = vec_msum(q8y20, q4x20, v0);
            وکتور علامتدار صحيح qv21 = vec_msum(q8y21, q4x21, v0);
            وکتور علامتدار صحيح qv30 = vec_msum(q8y30, q4x30, v0);
            وکتور علامتدار صحيح qv31 = vec_msum(q8y31, q4x31, v0);

            وکتور علامتدار صحيح vscales_h = vec_unpackh(vscales);
            وکتور علامتدار صحيح vs0 = vec_splat(vscales_h, 0);
            وکتور علامتدار صحيح vs1 = vec_splat(vscales_h, 1);
            وکتور علامتدار صحيح vs2 = vec_splat(vscales_h, 2);
            وکتور علامتدار صحيح vs3 = vec_splat(vscales_h, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_جمع_کن(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_جمع_کن(vec_mul(qv01, vs1), vsumi1);
            vsumi2 = vec_جمع_کن(vec_mul(qv20, vs2), vsumi2);
            vsumi3 = vec_جمع_کن(vec_mul(qv21, vs3), vsumi3);

            vsumi0 = vec_جمع_کن(vec_mul(qv10, vs0), vsumi0);
            vsumi1 = vec_جمع_کن(vec_mul(qv11, vs1), vsumi1);
            vsumi2 = vec_جمع_کن(vec_mul(qv30, vs2), vsumi2);
            vsumi3 = vec_جمع_کن(vec_mul(qv31, vs3), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    استفادهنشده(kmask1);
    استفادهنشده(kmask2);
    استفادهنشده(kmask3);
    استفادهنشده(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_q5_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy,  size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_q5_K * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

    ساكن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساكن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساكن ثابت uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور علامتدار حرف کم_Mask1 = vec_splats((int8_t)0x3f);
    ثابت وکتور علامتدار حرف کم_Mask2 = vec_splats((int8_t)0x30);
    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور بیعلامت حرف v1 = vec_splats((بیعلامت حرف)0x1);
    ثابت وکتور بیعلامت حرف v2 = vec_splats((بیعلامت حرف)0x2);
    ثابت وکتور بیعلامت حرف v3 = vec_splats((بیعلامت حرف)0x3);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور عائم vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].dmin));
        وکتور عائم vdmin = vec_mul(vxmin, vyd);

        استفادهنشده(kmask1);
        استفادهنشده(kmask2);
        استفادهنشده(kmask3);
        استفادهنشده(utmp);

        وکتور علامتدار حرف u0 = (وکتور علامتدار حرف)vec_xl_len(x[i].scales, 8);
        وکتور علامتدار حرف u1 = vec_and(vec_sr(u0, v2), کم_Mask2);
        وکتور علامتدار حرف u2 = (وکتور علامتدار حرف)vec_xl_len(x[i].scales + 8, 4);
        وکتور علامتدار حرف u3 = vec_sr(u2, v4);

        وکتور علامتدار حرف u30 = u1;
        وکتور علامتدار حرف u31 = (وکتور علامتدار حرف)vec_mergeh((وکتور علامتدار صحيح)vec_and(u2, کم_ماسک), (وکتور علامتدار صحيح)u3);

        u1 = vec_and(u0, کم_Mask1);
        u2 = vec_or(u30, u31);

        وکتور علامتدار حرف utmps = (وکتور علامتدار حرف)vec_mergeh((وکتور علامتدار صحيح)u1, (وکتور علامتدار صحيح)u2);

        وکتور علامتدار کوتاه q8ysums0 = vec_xl( 0, y[i].bsums);
        وکتور علامتدار کوتاه q8ysums1 = vec_xl(16, y[i].bsums);

        وکتور علامتدار کوتاه vscales = vec_unpackh(utmps);

        وکتور علامتدار کوتاه q5xmins = vec_unpackl(utmps);
        وکتور علامتدار کوتاه q5xmins0 = vec_mergeh(q5xmins, q5xmins);
        وکتور علامتدار کوتاه q5xmins1 = vec_mergel(q5xmins, q5xmins);

        وکتور علامتدار صحيح prod0 = vec_mule(q5xmins0, q8ysums0);
        وکتور علامتدار صحيح prod1 = vec_mule(q5xmins1, q8ysums1);
        وکتور علامتدار صحيح prod2 = vec_mulo(q5xmins0, q8ysums0);
        وکتور علامتدار صحيح prod3 = vec_mulo(q5xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        وکتور علامتدار حرف qxhs0 = (وکتور علامتدار حرف)vec_xl( 0, x[i].qh);
        وکتور علامتدار حرف qxhs1 = (وکتور علامتدار حرف)vec_xl(16, x[i].qh);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;

        ثابت uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        لكل (صحيح j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q5, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور علامتدار حرف qxs0 = (وکتور علامتدار حرف)vec_xl( 0, q5);
            وکتور علامتدار حرف qxs1 = (وکتور علامتدار حرف)vec_xl(16, q5);
            q5 += 32;

            وکتور علامتدار حرف qxs00 = vec_and(qxs0, کم_ماسک);
            وکتور علامتدار حرف qxs01 = vec_sr(qxs0, v4);
            وکتور علامتدار حرف qxs10 = vec_and(qxs1, کم_ماسک);
            وکتور علامتدار حرف qxs11 = vec_sr(qxs1, v4);

            وکتور علامتدار حرف q5h00 = vec_sl(vec_and((وکتور علامتدار حرف)v1, qxhs0), v4);
            وکتور علامتدار حرف q5h01 = vec_sl(vec_and((وکتور علامتدار حرف)v2, qxhs0), v3);
            وکتور علامتدار حرف q5h10 = vec_sl(vec_and((وکتور علامتدار حرف)v1, qxhs1), v4);
            وکتور علامتدار حرف q5h11 = vec_sl(vec_and((وکتور علامتدار حرف)v2, qxhs1), v3);
            qxhs0 = vec_sr(qxhs0, v2);
            qxhs1 = vec_sr(qxhs1, v2);

            وکتور بیعلامت حرف q5x00 = (وکتور بیعلامت حرف)vec_or(q5h00, qxs00);
            وکتور بیعلامت حرف q5x01 = (وکتور بیعلامت حرف)vec_or(q5h01, qxs01);
            وکتور بیعلامت حرف q5x10 = (وکتور بیعلامت حرف)vec_or(q5h10, qxs10);
            وکتور بیعلامت حرف q5x11 = (وکتور بیعلامت حرف)vec_or(q5h11, qxs11);

            وکتور علامتدار حرف q8y00 = vec_xl( 0, q8);
            وکتور علامتدار حرف q8y10 = vec_xl(16, q8);
            وکتور علامتدار حرف q8y01 = vec_xl(32, q8);
            وکتور علامتدار حرف q8y11 = vec_xl(48, q8);
            q8 += 64;

            وکتور علامتدار صحيح qv00 = vec_msum(q8y00, q5x00, v0);
            وکتور علامتدار صحيح qv01 = vec_msum(q8y01, q5x01, v0);
            وکتور علامتدار صحيح qv10 = vec_msum(q8y10, q5x10, v0);
            وکتور علامتدار صحيح qv11 = vec_msum(q8y11, q5x11, v0);

            وکتور علامتدار صحيح vscales_h = vec_unpackh(vscales);
            وکتور علامتدار صحيح vs0 = vec_splat(vscales_h, 0);
            وکتور علامتدار صحيح vs1 = vec_splat(vscales_h, 1);
            vscales = vec_sld(vscales, vscales, 12);

            vsumi0 = vec_جمع_کن(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_جمع_کن(vec_mul(qv10, vs0), vsumi1);
            vsumi2 = vec_جمع_کن(vec_mul(qv01, vs1), vsumi2);
            vsumi3 = vec_جمع_کن(vec_mul(qv11, vs1), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    استفادهنشده(kmask1);
    استفادهنشده(kmask2);
    استفادهنشده(kmask3);
    استفادهنشده(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_q6_K_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_q6_K * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور بیعلامت حرف v2 = vec_splats((بیعلامت حرف)0x2);
    ثابت وکتور بیعلامت حرف v3 = vec_splats((بیعلامت حرف)0x3);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);
    ثابت وکتور بیعلامت حرف v6 = vec_splats((بیعلامت حرف)0x6);
    ثابت وکتور علامتدار حرف off = vec_splats((علامتدار حرف)0x20);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;
        وکتور علامتدار صحيح vsumi4 = v0;
        وکتور علامتدار صحيح vsumi5 = v0;
        وکتور علامتدار صحيح vsumi6 = v0;
        وکتور علامتدار صحيح vsumi7 = v0;

        ثابت uint8_t * GGML_RESTRICT q6 = x[i].ql;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        ثابت int8_t  * GGML_RESTRICT qs = x[i].scales;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        لكل (صحيح j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q6, 0, 0);
            __builtin_prefetch(qh, 0, 0);
            __builtin_prefetch(q8, 0, 0);

            وکتور علامتدار حرف qxs0 = (وکتور علامتدار حرف)vec_xl( 0, q6);
            وکتور علامتدار حرف qxs1 = (وکتور علامتدار حرف)vec_xl(16, q6);
            وکتور علامتدار حرف qxs2 = (وکتور علامتدار حرف)vec_xl(32, q6);
            وکتور علامتدار حرف qxs3 = (وکتور علامتدار حرف)vec_xl(48, q6);
            q6 += 64;

            وکتور علامتدار حرف qxs00 = vec_and(qxs0, کم_ماسک);
            وکتور علامتدار حرف qxs01 = vec_sr(qxs0, v4);
            وکتور علامتدار حرف qxs10 = vec_and(qxs1, کم_ماسک);
            وکتور علامتدار حرف qxs11 = vec_sr(qxs1, v4);
            وکتور علامتدار حرف qxs20 = vec_and(qxs2, کم_ماسک);
            وکتور علامتدار حرف qxs21 = vec_sr(qxs2, v4);
            وکتور علامتدار حرف qxs30 = vec_and(qxs3, کم_ماسک);
            وکتور علامتدار حرف qxs31 = vec_sr(qxs3, v4);

            وکتور علامتدار حرف qxhs0 = (وکتور علامتدار حرف)vec_xl( 0, qh);
            وکتور علامتدار حرف qxhs1 = (وکتور علامتدار حرف)vec_xl(16, qh);
            qh += 32;

            وکتور علامتدار حرف qxh00 = vec_sl(vec_and((وکتور علامتدار حرف)v3, qxhs0), v4);
            وکتور علامتدار حرف qxh01 = vec_sl(vec_and((وکتور علامتدار حرف)v3, vec_sr(qxhs0, v4)), v4);
            وکتور علامتدار حرف qxh10 = vec_sl(vec_and((وکتور علامتدار حرف)v3, qxhs1), v4);
            وکتور علامتدار حرف qxh11 = vec_sl(vec_and((وکتور علامتدار حرف)v3, vec_sr(qxhs1, v4)), v4);
            وکتور علامتدار حرف qxh20 = vec_sl(vec_and((وکتور علامتدار حرف)v3, vec_sr(qxhs0, v2)), v4);
            وکتور علامتدار حرف qxh21 = vec_sl(vec_and((وکتور علامتدار حرف)v3, vec_sr(qxhs0, v6)), v4);
            وکتور علامتدار حرف qxh30 = vec_sl(vec_and((وکتور علامتدار حرف)v3, vec_sr(qxhs1, v2)), v4);
            وکتور علامتدار حرف qxh31 = vec_sl(vec_and((وکتور علامتدار حرف)v3, vec_sr(qxhs1, v6)), v4);

            وکتور علامتدار حرف q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
            وکتور علامتدار حرف q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
            وکتور علامتدار حرف q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
            وکتور علامتدار حرف q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
            وکتور علامتدار حرف q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
            وکتور علامتدار حرف q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
            وکتور علامتدار حرف q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
            وکتور علامتدار حرف q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

            وکتور علامتدار حرف q8y00 = vec_xl(  0, q8);
            وکتور علامتدار حرف q8y10 = vec_xl( 16, q8);
            وکتور علامتدار حرف q8y20 = vec_xl( 32, q8);
            وکتور علامتدار حرف q8y30 = vec_xl( 48, q8);
            وکتور علامتدار حرف q8y01 = vec_xl( 64, q8);
            وکتور علامتدار حرف q8y11 = vec_xl( 80, q8);
            وکتور علامتدار حرف q8y21 = vec_xl( 96, q8);
            وکتور علامتدار حرف q8y31 = vec_xl(112, q8);
            q8 += 128;

            وکتور علامتدار کوتاه qv00 = vec_جمع_کن(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
            وکتور علامتدار کوتاه qv10 = vec_جمع_کن(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
            وکتور علامتدار کوتاه qv20 = vec_جمع_کن(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
            وکتور علامتدار کوتاه qv30 = vec_جمع_کن(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
            وکتور علامتدار کوتاه qv01 = vec_جمع_کن(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
            وکتور علامتدار کوتاه qv11 = vec_جمع_کن(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
            وکتور علامتدار کوتاه qv21 = vec_جمع_کن(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
            وکتور علامتدار کوتاه qv31 = vec_جمع_کن(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

            وکتور علامتدار کوتاه vscales = vec_unpackh(vec_xl_len(qs, 8));
            qs += 8;

            وکتور علامتدار کوتاه vs0 = vec_splat(vscales, 0);
            وکتور علامتدار کوتاه vs1 = vec_splat(vscales, 1);
            وکتور علامتدار کوتاه vs2 = vec_splat(vscales, 2);
            وکتور علامتدار کوتاه vs3 = vec_splat(vscales, 3);
            وکتور علامتدار کوتاه vs4 = vec_splat(vscales, 4);
            وکتور علامتدار کوتاه vs5 = vec_splat(vscales, 5);
            وکتور علامتدار کوتاه vs6 = vec_splat(vscales, 6);
            وکتور علامتدار کوتاه vs7 = vec_splat(vscales, 7);

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs4, vsumi1);
            vsumi2 = vec_msum(qv10, vs1, vsumi2);
            vsumi3 = vec_msum(qv11, vs5, vsumi3);
            vsumi4 = vec_msum(qv20, vs2, vsumi4);
            vsumi5 = vec_msum(qv21, vs6, vsumi5);
            vsumi6 = vec_msum(qv30, vs3, vsumi6);
            vsumi7 = vec_msum(qv31, vs7, vsumi7);
        }

        vsumi0 = vec_جمع_کن(vsumi0, vsumi4);
        vsumi1 = vec_جمع_کن(vsumi1, vsumi5);
        vsumi2 = vec_جمع_کن(vsumi2, vsumi6);
        vsumi3 = vec_جمع_کن(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

#إذا defined (__POWER9_VECTOR__)
ساكن ثابت int8_t keven_signs_q2xs[1024] = {
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
#پایان_اگر

فراغ ggml_vec_dot_iq2_xxs_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_iq2_xxs * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K    * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);
    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;

        ثابت uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        لكل (صحيح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            ثابت uint8_t * aux8 = (ثابت uint8_t *)aux32;

            کپی_حافظه(aux32, q2, 4*حجم(uint32_t));
            q2 += 8;

            وکتور علامتدار دراز دراز aux64x2_0 = {*(ثابت int64_t *)(iq2xxs_grid + aux8[ 0]), *(ثابت int64_t *)(iq2xxs_grid + aux8[ 1])};
            وکتور علامتدار دراز دراز aux64x2_1 = {*(ثابت int64_t *)(iq2xxs_grid + aux8[ 2]), *(ثابت int64_t *)(iq2xxs_grid + aux8[ 3])};
            وکتور علامتدار دراز دراز aux64x2_2 = {*(ثابت int64_t *)(iq2xxs_grid + aux8[ 8]), *(ثابت int64_t *)(iq2xxs_grid + aux8[ 9])};
            وکتور علامتدار دراز دراز aux64x2_3 = {*(ثابت int64_t *)(iq2xxs_grid + aux8[10]), *(ثابت int64_t *)(iq2xxs_grid + aux8[11])};

            وکتور علامتدار دراز دراز vsigns0 = {*(ثابت int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(ثابت int64_t *)(signs64 + ((aux32[1] >>  7) & 127))};
            وکتور علامتدار دراز دراز vsigns1 = {*(ثابت int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(ثابت int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            وکتور علامتدار دراز دراز vsigns2 = {*(ثابت int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(ثابت int64_t *)(signs64 + ((aux32[3] >>  7) & 127))};
            وکتور علامتدار دراز دراز vsigns3 = {*(ثابت int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(ثابت int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            وکتور علامتدار حرف q2x0 = (وکتور علامتدار حرف)vec_mul((وکتور علامتدار حرف)vsigns0, (وکتور علامتدار حرف)aux64x2_0);
            وکتور علامتدار حرف q2x1 = (وکتور علامتدار حرف)vec_mul((وکتور علامتدار حرف)vsigns1, (وکتور علامتدار حرف)aux64x2_1);
            وکتور علامتدار حرف q2x2 = (وکتور علامتدار حرف)vec_mul((وکتور علامتدار حرف)vsigns2, (وکتور علامتدار حرف)aux64x2_2);
            وکتور علامتدار حرف q2x3 = (وکتور علامتدار حرف)vec_mul((وکتور علامتدار حرف)vsigns3, (وکتور علامتدار حرف)aux64x2_3);

            وکتور علامتدار حرف q8y0 = vec_xl( 0, q8);
            وکتور علامتدار حرف q8y1 = vec_xl(16, q8);
            وکتور علامتدار حرف q8y2 = vec_xl(32, q8);
            وکتور علامتدار حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            وکتور علامتدار کوتاه qv2 = vec_جمع_کن(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            وکتور علامتدار کوتاه qv3 = vec_جمع_کن(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ثابت uint16_t ls0 = aux32[1] >> 28;
            ثابت uint16_t ls1 = aux32[3] >> 28;

            وکتور علامتدار کوتاه vscales01 = vec_splats((int16_t)(2*ls0+1));
            وکتور علامتدار کوتاه vscales23 = vec_splats((int16_t)(2*ls1+1));

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

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_iq2_xs_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_iq2_xs * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K   * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);
    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;

        ثابت uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت uint8_t  * GGML_RESTRICT sc = x[i].scales;
        ثابت int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        لكل (صحيح j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور علامتدار دراز دراز aux64x2_0 = {*(ثابت int64_t *)(iq2xs_grid + (q2[0] & 511)), *(ثابت int64_t *)(iq2xs_grid + (q2[1] & 511))};
            وکتور علامتدار دراز دراز aux64x2_1 = {*(ثابت int64_t *)(iq2xs_grid + (q2[2] & 511)), *(ثابت int64_t *)(iq2xs_grid + (q2[3] & 511))};
            وکتور علامتدار دراز دراز aux64x2_2 = {*(ثابت int64_t *)(iq2xs_grid + (q2[4] & 511)), *(ثابت int64_t *)(iq2xs_grid + (q2[5] & 511))};
            وکتور علامتدار دراز دراز aux64x2_3 = {*(ثابت int64_t *)(iq2xs_grid + (q2[6] & 511)), *(ثابت int64_t *)(iq2xs_grid + (q2[7] & 511))};

            وکتور علامتدار دراز دراز vsigns0 = {*(ثابت int64_t *)(signs64 + ((q2[0] >> 9))), *(ثابت int64_t *)(signs64 + ((q2[1] >> 9)))};
            وکتور علامتدار دراز دراز vsigns1 = {*(ثابت int64_t *)(signs64 + ((q2[2] >> 9))), *(ثابت int64_t *)(signs64 + ((q2[3] >> 9)))};
            وکتور علامتدار دراز دراز vsigns2 = {*(ثابت int64_t *)(signs64 + ((q2[4] >> 9))), *(ثابت int64_t *)(signs64 + ((q2[5] >> 9)))};
            وکتور علامتدار دراز دراز vsigns3 = {*(ثابت int64_t *)(signs64 + ((q2[6] >> 9))), *(ثابت int64_t *)(signs64 + ((q2[7] >> 9)))};
            q2 += 8;

            وکتور علامتدار حرف q2x0 = (وکتور علامتدار حرف)vec_mul((وکتور علامتدار حرف)vsigns0, (وکتور علامتدار حرف)aux64x2_0);
            وکتور علامتدار حرف q2x1 = (وکتور علامتدار حرف)vec_mul((وکتور علامتدار حرف)vsigns1, (وکتور علامتدار حرف)aux64x2_1);
            وکتور علامتدار حرف q2x2 = (وکتور علامتدار حرف)vec_mul((وکتور علامتدار حرف)vsigns2, (وکتور علامتدار حرف)aux64x2_2);
            وکتور علامتدار حرف q2x3 = (وکتور علامتدار حرف)vec_mul((وکتور علامتدار حرف)vsigns3, (وکتور علامتدار حرف)aux64x2_3);

            وکتور علامتدار حرف q8y0 = vec_xl( 0, q8);
            وکتور علامتدار حرف q8y1 = vec_xl(16, q8);
            وکتور علامتدار حرف q8y2 = vec_xl(32, q8);
            وکتور علامتدار حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            وکتور علامتدار کوتاه qv2 = vec_جمع_کن(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            وکتور علامتدار کوتاه qv3 = vec_جمع_کن(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ثابت uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ثابت uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ثابت uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            وکتور علامتدار کوتاه vscales0 = vec_splats((int16_t)(2*ls0+1));
            وکتور علامتدار کوتاه vscales1 = vec_splats((int16_t)(2*ls1+1));
            وکتور علامتدار کوتاه vscales2 = vec_splats((int16_t)(2*ls2+1));
            وکتور علامتدار کوتاه vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_iq2_s_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_iq2_s * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ساكن ثابت uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ساكن ثابت uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    ثابت وکتور بیعلامت حرف mask0 = vec_xl( 0, k_mask1);
    ثابت وکتور بیعلامت حرف mask1 = vec_xl(16, k_mask1);
    ثابت وکتور علامتدار حرف mask2 = (وکتور علامتدار حرف)vec_xl( 0, k_mask2);

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;

        ثابت uint8_t *  GGML_RESTRICT q2 = x[i].qs;
        ثابت uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ثابت uint16_t * GGML_RESTRICT signs = (ثابت uint16_t *)(x[i].qs + QK_K/8);
        ثابت uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ثابت int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        لكل (صحيح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور علامتدار دراز دراز aux64x2_0 = {*(ثابت int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(ثابت int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            وکتور علامتدار دراز دراز aux64x2_1 = {*(ثابت int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(ثابت int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            وکتور علامتدار دراز دراز aux64x2_2 = {*(ثابت int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(ثابت int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            وکتور علامتدار دراز دراز aux64x2_3 = {*(ثابت int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(ثابت int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            وکتور علامتدار حرف vsigns01 = (وکتور علامتدار حرف)vec_splats(*(ثابت uint32_t *)&signs[0]);
            وکتور علامتدار حرف vsigns23 = (وکتور علامتدار حرف)vec_splats(*(ثابت uint32_t *)&signs[2]);
            signs += 4;

            وکتور علامتدار حرف vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            وکتور علامتدار حرف vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            وکتور علامتدار حرف vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            وکتور علامتدار حرف vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (وکتور علامتدار حرف)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (وکتور علامتدار حرف)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (وکتور علامتدار حرف)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (وکتور علامتدار حرف)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            وکتور علامتدار حرف q2x0 = vec_sub(vec_xor(vsigns0, (وکتور علامتدار حرف)aux64x2_0), vsigns0);
            وکتور علامتدار حرف q2x1 = vec_sub(vec_xor(vsigns1, (وکتور علامتدار حرف)aux64x2_1), vsigns1);
            وکتور علامتدار حرف q2x2 = vec_sub(vec_xor(vsigns2, (وکتور علامتدار حرف)aux64x2_2), vsigns2);
            وکتور علامتدار حرف q2x3 = vec_sub(vec_xor(vsigns3, (وکتور علامتدار حرف)aux64x2_3), vsigns3);

            وکتور علامتدار حرف q8y0 = vec_xl( 0, q8);
            وکتور علامتدار حرف q8y1 = vec_xl(16, q8);
            وکتور علامتدار حرف q8y2 = vec_xl(32, q8);
            وکتور علامتدار حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            وکتور علامتدار کوتاه qv2 = vec_جمع_کن(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            وکتور علامتدار کوتاه qv3 = vec_جمع_کن(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ثابت uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ثابت uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ثابت uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            وکتور علامتدار کوتاه vscales0 = vec_splats((int16_t)(2*ls0+1));
            وکتور علامتدار کوتاه vscales1 = vec_splats((int16_t)(2*ls1+1));
            وکتور علامتدار کوتاه vscales2 = vec_splats((int16_t)(2*ls2+1));
            وکتور علامتدار کوتاه vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_iq3_xxs_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_iq3_xxs * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K    * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;

        ثابت uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ثابت uint32_t * GGML_RESTRICT signs = (ثابت uint32_t *)(x[i].qs + QK_K/4);
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

#رهنمود GCC unroll 1
        لكل (صحيح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور بیعلامت صحيح aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            وکتور بیعلامت صحيح aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            وکتور بیعلامت صحيح aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            وکتور بیعلامت صحيح aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            وکتور بیعلامت دراز دراز aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            وکتور بیعلامت دراز دراز aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            وکتور بیعلامت دراز دراز aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            وکتور بیعلامت دراز دراز aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            وکتور علامتدار حرف q3x0 = vec_mul((وکتور علامتدار حرف)aux64x2_0, (وکتور علامتدار حرف)aux32x4_0);
            وکتور علامتدار حرف q3x1 = vec_mul((وکتور علامتدار حرف)aux64x2_1, (وکتور علامتدار حرف)aux32x4_1);
            وکتور علامتدار حرف q3x2 = vec_mul((وکتور علامتدار حرف)aux64x2_2, (وکتور علامتدار حرف)aux32x4_2);
            وکتور علامتدار حرف q3x3 = vec_mul((وکتور علامتدار حرف)aux64x2_3, (وکتور علامتدار حرف)aux32x4_3);

            وکتور علامتدار حرف q8y0 = vec_xl( 0, q8);
            وکتور علامتدار حرف q8y1 = vec_xl(16, q8);
            وکتور علامتدار حرف q8y2 = vec_xl(32, q8);
            وکتور علامتدار حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            وکتور علامتدار کوتاه qv2 = vec_جمع_کن(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            وکتور علامتدار کوتاه qv3 = vec_جمع_کن(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            ثابت uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            وکتور علامتدار کوتاه vscales01 = (وکتور علامتدار کوتاه)vec_splats((uint16_t)(2*ls0+1));
            وکتور علامتدار کوتاه vscales23 = (وکتور علامتدار کوتاه)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.25f * vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_iq3_s_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_iq3_s * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ساكن ثابت uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ساكن ثابت uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    ثابت وکتور بیعلامت حرف mask0 = vec_xl( 0, k_mask1);
    ثابت وکتور بیعلامت حرف mask1 = vec_xl(16, k_mask1);
    ثابت وکتور علامتدار حرف mask2 = (وکتور علامتدار حرف)vec_xl( 0, k_mask2);

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        ثابت uint8_t *  GGML_RESTRICT q3 = x[i].qs;
        ثابت uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ثابت uint16_t * GGML_RESTRICT signs = (ثابت uint16_t *)(x[i].signs);
        ثابت uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ثابت int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;

        لكل (صحيح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور بیعلامت صحيح aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            وکتور بیعلامت صحيح aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            وکتور بیعلامت صحيح aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            وکتور بیعلامت صحيح aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            وکتور علامتدار حرف vsigns01 = (وکتور علامتدار حرف)vec_splats(*(ثابت uint32_t *)&signs[0]);
            وکتور علامتدار حرف vsigns02 = (وکتور علامتدار حرف)vec_splats(*(ثابت uint32_t *)&signs[2]);
            signs += 4;

            وکتور علامتدار حرف vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            وکتور علامتدار حرف vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            وکتور علامتدار حرف vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            وکتور علامتدار حرف vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (وکتور علامتدار حرف)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (وکتور علامتدار حرف)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (وکتور علامتدار حرف)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (وکتور علامتدار حرف)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            وکتور علامتدار حرف q3x0 = vec_sub(vec_xor(vsigns0, (وکتور علامتدار حرف)aux32x4_0), vsigns0);
            وکتور علامتدار حرف q3x1 = vec_sub(vec_xor(vsigns1, (وکتور علامتدار حرف)aux32x4_1), vsigns1);
            وکتور علامتدار حرف q3x2 = vec_sub(vec_xor(vsigns2, (وکتور علامتدار حرف)aux32x4_2), vsigns2);
            وکتور علامتدار حرف q3x3 = vec_sub(vec_xor(vsigns3, (وکتور علامتدار حرف)aux32x4_3), vsigns3);

            وکتور علامتدار حرف q8y0 = vec_xl( 0, q8);
            وکتور علامتدار حرف q8y1 = vec_xl(16, q8);
            وکتور علامتدار حرف q8y2 = vec_xl(32, q8);
            وکتور علامتدار حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            وکتور علامتدار کوتاه qv2 = vec_جمع_کن(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            وکتور علامتدار کوتاه qv3 = vec_جمع_کن(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ثابت uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            وکتور علامتدار کوتاه vscales01 = (وکتور علامتدار کوتاه)vec_splats((uint16_t)(2*ls0+1));
            وکتور علامتدار کوتاه vscales23 = (وکتور علامتدار کوتاه)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_iq1_s_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);

    ثابت بلوک_iq1_s * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور بیعلامت حرف v0 = vec_splats((بیعلامت حرف)0x0);
    ثابت وکتور بیعلامت کوتاه vsign = vec_splats((بیعلامت کوتاه)0x8000);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    لكل (صحيح i = 0; i < nb; ++i) {
        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        وکتور عائم vyd = vec_splats(y[i].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار صحيح vsumi0 = vec_splats((int32_t)0);
        وکتور علامتدار صحيح vsumi1 = vec_splats((int32_t)0);
        وکتور علامتدار صحيح vsumi2 = vec_splats((int32_t)0);
        وکتور علامتدار صحيح vsumi3 = vec_splats((int32_t)0);
        وکتور علامتدار صحيح vsumi8 = vec_splats((int32_t)0);

        ثابت uint8_t  * GGML_RESTRICT q1 = x[i].qs;
        ثابت uint16_t * GGML_RESTRICT qh = x[i].qh;
        ثابت int8_t   * GGML_RESTRICT q8 = y[i].qs;
        ثابت int16_t  * GGML_RESTRICT qs = y[i].bsums;

        لكل (صحيح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور علامتدار دراز دراز aux64x2_0 = {*(ثابت int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(ثابت int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            وکتور علامتدار دراز دراز aux64x2_1 = {*(ثابت int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(ثابت int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            وکتور علامتدار دراز دراز aux64x2_2 = {*(ثابت int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(ثابت int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            وکتور علامتدار دراز دراز aux64x2_3 = {*(ثابت int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(ثابت int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            وکتور علامتدار حرف q1x0 = (وکتور علامتدار حرف)aux64x2_0;
            وکتور علامتدار حرف q1x1 = (وکتور علامتدار حرف)aux64x2_1;
            وکتور علامتدار حرف q1x2 = (وکتور علامتدار حرف)aux64x2_2;
            وکتور علامتدار حرف q1x3 = (وکتور علامتدار حرف)aux64x2_3;

            وکتور علامتدار حرف q8y0 = vec_xl( 0, q8);
            وکتور علامتدار حرف q8y1 = vec_xl(16, q8);
            وکتور علامتدار حرف q8y2 = vec_xl(32, q8);
            وکتور علامتدار حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            وکتور علامتدار کوتاه qv2 = vec_جمع_کن(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            وکتور علامتدار کوتاه qv3 = vec_جمع_کن(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            ثابت uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            وکتور علامتدار کوتاه vscales01 = (وکتور علامتدار کوتاه)vec_splats((uint16_t)(2*ls0+1));
            وکتور علامتدار کوتاه vscales23 = (وکتور علامتدار کوتاه)vec_splats((uint16_t)(2*ls1+1));
            وکتور علامتدار کوتاه vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            وکتور علامتدار کوتاه q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (وکتور علامتدار کوتاه)v0);

            وکتور علامتدار کوتاه qxh = (وکتور علامتدار کوتاه)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            وکتور __bool کوتاه vsel = vec_cmpge(qxh, (وکتور علامتدار کوتاه)v0);

            وکتور علامتدار کوتاه q8ysum = vec_sel((وکتور علامتدار کوتاه)vec_xor((وکتور بیعلامت کوتاه)q8ysums, vsign), q8ysums, vsel);

            vsumi8 = vec_جمع_کن(vec_mule(q8ysum, vscales), vsumi8);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);

        vsumf0 = vec_madd(vec_ctf(vsumi8, 0), vec_mul(vd, vec_splats(IQ1S_DELTA)), vsumf0);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_iq4_nl_q8_0(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);
    أكد(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ثابت بلوک_iq4_nl * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_0   * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK4_NL;

    صحيح ib = 0;
    عائم sumf = 0;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور علامتدار صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);

    ثابت وکتور علامتدار حرف مقادیر = vec_xl( 0, kvalues_iq4nl);

#رهنمود GCC unroll 4
    لكل (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        وکتور عائم vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار حرف qxs = (وکتور علامتدار حرف)vec_xl( 0, x[ib].qs);
        وکتور علامتدار حرف q4x0 = vec_and(qxs, کم_ماسک);
        وکتور علامتدار حرف q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(مقادیر, مقادیر, (وکتور بیعلامت حرف)q4x0);
        q4x1 = vec_perm(مقادیر, مقادیر, (وکتور بیعلامت حرف)q4x1);

        وکتور علامتدار حرف q8y0 = vec_xl( 0, y[ib].qs);
        وکتور علامتدار حرف q8y1 = vec_xl(16, y[ib].qs);

        وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    }

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_استخراج(vsumf0, 0);

    *s = sumf;
#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    استفادهنشده(ib);
    استفادهنشده(sumf);
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}

فراغ ggml_vec_dot_iq4_xs_q8_K(صحيح n, عائم * GGML_RESTRICT s, size_t bs, ثابت فراغ * GGML_RESTRICT vx, size_t bx, ثابت فراغ * GGML_RESTRICT vy, size_t by, صحيح nrc) {
    أكد(nrc == 1);
    استفادهنشده(nrc);
    استفادهنشده(bx);
    استفادهنشده(by);
    استفادهنشده(bs);
    أكد(n % QK_K == 0);

    ثابت بلوک_iq4_xs * GGML_RESTRICT x = vx;
    ثابت بلوک_q8_K   * GGML_RESTRICT y = vy;

    ثابت صحيح nb = n / QK_K;

#إذا defined(__POWER9_VECTOR__)
    ثابت وکتور علامتدار حرف کم_ماسک = vec_splats((علامتدار حرف)0xF);
    ثابت وکتور صحيح v0 = vec_splats((int32_t)0);
    ثابت وکتور بیعلامت حرف v4 = vec_splats((بیعلامت حرف)0x4);

    وکتور عائم vsumf0 = vec_splats(0.0f);
    وکتور عائم vsumf1 = vec_splats(0.0f);
    وکتور عائم vsumf2 = vec_splats(0.0f);
    وکتور عائم vsumf3 = vec_splats(0.0f);

    ثابت وکتور علامتدار حرف مقادیر = vec_xl( 0, kvalues_iq4nl);

    لكل (صحيح ibl = 0; ibl < nb; ++ibl) {

        وکتور عائم vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ibl].d));
        وکتور عائم vyd = vec_splats(y[ibl].d);
        وکتور عائم vd = vec_mul(vxd, vyd);

        وکتور علامتدار صحيح vsumi0 = v0;
        وکتور علامتدار صحيح vsumi1 = v0;
        وکتور علامتدار صحيح vsumi2 = v0;
        وکتور علامتدار صحيح vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        ثابت uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        ثابت uint8_t * GGML_RESTRICT sc = x[ibl].scales_l;
        ثابت int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        لكل (صحيح ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            وکتور علامتدار حرف qxs0 = (وکتور علامتدار حرف)vec_xl( 0, q4);
            وکتور علامتدار حرف qxs1 = (وکتور علامتدار حرف)vec_xl(16, q4);
            q4 += 32;

            وکتور علامتدار حرف q4x00 = (وکتور علامتدار حرف)vec_and(qxs0, کم_ماسک);
            وکتور علامتدار حرف q4x01 = (وکتور علامتدار حرف)vec_sr(qxs0, v4);
            وکتور علامتدار حرف q4x10 = (وکتور علامتدار حرف)vec_and(qxs1, کم_ماسک);
            وکتور علامتدار حرف q4x11 = (وکتور علامتدار حرف)vec_sr(qxs1, v4);

            q4x00 = vec_perm(مقادیر, مقادیر, (وکتور بیعلامت حرف)q4x00);
            q4x01 = vec_perm(مقادیر, مقادیر, (وکتور بیعلامت حرف)q4x01);
            q4x10 = vec_perm(مقادیر, مقادیر, (وکتور بیعلامت حرف)q4x10);
            q4x11 = vec_perm(مقادیر, مقادیر, (وکتور بیعلامت حرف)q4x11);

            وکتور علامتدار حرف q8y0 = vec_xl( 0, q8);
            وکتور علامتدار حرف q8y1 = vec_xl(16, q8);
            وکتور علامتدار حرف q8y2 = vec_xl(32, q8);
            وکتور علامتدار حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            وکتور علامتدار کوتاه qv0 = vec_جمع_کن(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            وکتور علامتدار کوتاه qv1 = vec_جمع_کن(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            وکتور علامتدار کوتاه qv2 = vec_جمع_کن(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            وکتور علامتدار کوتاه qv3 = vec_جمع_کن(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            ثابت uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            وکتور علامتدار کوتاه vscales01 = vec_splats((int16_t)ls0);
            وکتور علامتدار کوتاه vscales23 = vec_splats((int16_t)ls1);

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

    vsumf0 = vec_جمع_کن(vsumf0, vsumf2);
    vsumf1 = vec_جمع_کن(vsumf1, vsumf3);

    vsumf0 = vec_جمع_کن(vsumf0, vsumf1);

    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جمع_کن(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_استخراج(vsumf0, 0);

#وإلا
    استفادهنشده(x);
    استفادهنشده(y);
    استفادهنشده(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#پایان_اگر
}
