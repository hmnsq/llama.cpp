#تعریف_کرو GGML_COMMON_IMPL_C
#شامل "ggml-common.h"
#شامل "ggml-quants.h"
#شامل "ggml-impl.h"
#شامل "ggml-cpu.h"
#شامل "simd-mappings.h"

#شامل "../../quants.h"
#شامل "../../ggml-cpu-impl.h"

#شامل <math.h>
#شامل <string.h>
#شامل <assert.h>
#شامل <float.h>
#شامل <stdlib.h> // for qsort
#شامل <stdio.h>  // for GGML_ASSERT

#تعریف_کرو گروہ_MAX_EPS 1e-15f
#تعریف_کرو گروہ_MAX_EPS_IQ3_XXS 1e-8f
#تعریف_کرو گروہ_MAX_EPS_IQ2_S 1e-8f
#تعریف_کرو گروہ_MAX_EPS_IQ1_M 1e-7f
#تعریف_کرو گروہ_MAX_EPS_IQ1_S 1e-12f

#تعریف_کرو غیر_مستعمل GGML_غیر_مستعمل

#اگر defined(__POWER9_VECTOR__)
#تعریف_کرو B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#تعریف_کرو B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#تعریف_کرو B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#تعریف_کرو B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#تعریف_کرو B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#تعریف_کرو B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#تعریف_کرو B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#تعریف_کرو B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
ساکن ثابت uint64_t جدول_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
ساکن ثابت uint64_t جدول_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#اگر_ختم

خالی quantize_صف_q8_0(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT vy, int64_t k) {
    أكد(QK8_0 == 32);
    أكد(k % QK8_0 == 0);
    ثابت صحیح nb = k / QK8_0;

    بلاک_q8_0 * GGML_RESTRICT y = vy;

#اگر defined(__POWER9_VECTOR__)
    برائے (صحیح i = 0; i < nb; i++) {
        سمتیہ اعشاری srcv [8];
        سمتیہ اعشاری asrcv[8];
        سمتیہ اعشاری amaxv[8];
        سمتیہ علامتی صحیح vi[8];

        برائے (صحیح j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        برائے (صحیح j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        برائے (صحیح j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        برائے (صحیح j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        برائے (صحیح j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ثابت اعشاری amax = MAX(MAX(vec_نکالو(amaxv[0], 0),
                                   vec_نکالو(amaxv[0], 1)),
                               MAX(vec_نکالو(amaxv[0], 2),
                                   vec_نکالو(amaxv[0], 3)));

        ثابت اعشاری d = amax / ((1 << 7) - 1);
        ثابت اعشاری id = d ? 1.0f/d : 0.0f;
        ثابت سمتیہ اعشاری vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_مقصد_FP16(d);

        برائے (صحیح j = 0; j < 8; j++) {
            ثابت سمتیہ اعشاری v  = vec_باری(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }
#ورنہ
    GGML_غیر_مستعمل(nb);
    // scalar
    quantize_صف_q8_0_ref(x, y, k);
#اگر_ختم
}

خالی quantize_صف_q8_1(ثابت اعشاری * GGML_RESTRICT x, خالی * GGML_RESTRICT vy, int64_t k) {
    أكد(k % QK8_1 == 0);
    ثابت صحیح nb = k / QK8_1;

    بلاک_q8_1 * GGML_RESTRICT y = vy;

#اگر defined(__POWER9_VECTOR__)
    برائے (صحیح i = 0; i < nb; i++) {
        سمتیہ اعشاری srcv [8];
        سمتیہ اعشاری asrcv[8];
        سمتیہ اعشاری amaxv[8];
        سمتیہ علامتی صحیح vi[8];

        برائے (صحیح j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        برائے (صحیح j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        برائے (صحیح j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        برائے (صحیح j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        برائے (صحیح j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ثابت اعشاری amax = MAX(MAX(vec_نکالو(amaxv[0], 0),
                                   vec_نکالو(amaxv[0], 1)),
                               MAX(vec_نکالو(amaxv[0], 2),
                                   vec_نکالو(amaxv[0], 3)));

        ثابت اعشاری d = amax / ((1 << 7) - 1);
        ثابت اعشاری id = d ? 1.0f/d : 0.0f;
        ثابت سمتیہ اعشاری vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_مقصد_FP16(d);

        سمتیہ صحیح accv = vec_splats(0);

        برائے (صحیح j = 0; j < 8; j++) {
            ثابت سمتیہ اعشاری v  = vec_باری(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_جوڑو(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_جوڑو(accv, vec_sld(accv, accv, 4));
        accv = vec_جوڑو(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_CPU_FP32_مقصد_FP16(d * vec_نکالو(accv, 0));
    }

#ورنہ
    GGML_غیر_مستعمل(nb);
    // scalar
    quantize_صف_q8_1_ref(x, y, k);
#اگر_ختم
}


//===================================== Dot products =================================

خالی ggml_vec_dot_q4_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_0;
    ثابت صحیح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q4_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ علامتی صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);
    ثابت سمتیہ علامتی حرف v8 = vec_splats((علامتی حرف)0x8);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);

#ہدایت GCC unroll 8
    برائے (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        سمتیہ اعشاری vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی حرف qxs = (سمتیہ علامتی حرف)vec_xl( 0, x[ib].qs);
        سمتیہ علامتی حرف q8y0 = vec_xl( 0, y[ib].qs);
        سمتیہ علامتی حرف q8y1 = vec_xl(16, y[ib].qs);

        سمتیہ علامتی حرف q4x0 = vec_and(qxs, کم_نقاب);
        سمتیہ علامتی حرف q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        سمتیہ علامتی صحیح vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_نکالو(vsumf0, 0);

    *s = sumf;
#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q4_1_q8_1(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_1;
    ثابت صحیح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q4_1 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_1 * GGML_RESTRICT y = vy;

    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ علامتی صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);

#ہدایت GCC unroll 4
    برائے (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        سمتیہ اعشاری vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ اعشاری vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].m));
        سمتیہ اعشاری vys = {GGML_CPU_FP16_مقصد_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        سمتیہ علامتی حرف qxs = (سمتیہ علامتی حرف)vec_xl( 0, x[ib].qs);
        سمتیہ علامتی حرف q8y0 = vec_xl( 0, y[ib].qs);
        سمتیہ علامتی حرف q8y1 = vec_xl(16, y[ib].qs);

        سمتیہ غیرعلامتی حرف q4x0 = (سمتیہ غیرعلامتی حرف)vec_and(qxs, کم_نقاب);
        سمتیہ غیرعلامتی حرف q4x1 = (سمتیہ غیرعلامتی حرف)vec_sr(qxs, v4);

        سمتیہ علامتی صحیح vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_نکالو(vsumf0, 0);

    *s = sumf;
#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_mxfp4_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);
    أكد(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ثابت بلاک_mxfp4 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_MXFP4;

    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ غیرعلامتی حرف vshift4 = vec_splats((غیرعلامتی حرف)4);
    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);

    سمتیہ علامتی حرف kv = vec_xl(0, (ثابت علامتی حرف *)kvalues_mxfp4);

#ہدایت GCC unroll 8
    برائے (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        سمتیہ اعشاری vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d) *
                                      GGML_E8_M0_مقصد_FP32_HALF(x[ib].e));

        سمتیہ علامتی حرف q8y0 = vec_xl( 0, y[ib].qs);
        سمتیہ علامتی حرف q8y1 = vec_xl(16, y[ib].qs);

        سمتیہ علامتی حرف qxs = (سمتیہ علامتی حرف)vec_xl(0, x[ib].qs);

        سمتیہ غیرعلامتی حرف lo_nibbles = (سمتیہ غیرعلامتی حرف)vec_and(qxs, کم_نقاب);
        سمتیہ غیرعلامتی حرف hi_nibbles = (سمتیہ غیرعلامتی حرف)vec_sr(qxs, vshift4);

        سمتیہ علامتی حرف q4x0 = vec_perm(kv, kv, lo_nibbles);
        سمتیہ علامتی حرف q4x1 = vec_perm(kv, kv, hi_nibbles);

        سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        سمتیہ علامتی صحیح vsumi0 = vec_splats((int32_t)0);
        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vyd, vsumf0);
    }

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));
    sumf = vec_نکالو(vsumf0, 0);
    *s = sumf;
#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
    ggml_vec_dot_mxfp4_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q5_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_0;
    ثابت صحیح nb = n / qk;

    صحیح ib = 0;
    اعشاری sumf = 0;

    أكد(n % qk == 0);
    أكد(qk == QK5_0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q5_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)4);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);

#ہدایت GCC unroll 4
    برائے (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        سمتیہ اعشاری vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی طویل طویل aux64x2_0 = {(uint64_t)(جدول_b2b_1[x[ib].qh[0]]), (uint64_t)(جدول_b2b_1[x[ib].qh[1]])};
        سمتیہ علامتی طویل طویل aux64x2_1 = {(uint64_t)(جدول_b2b_1[x[ib].qh[2]]), (uint64_t)(جدول_b2b_1[x[ib].qh[3]])};

        سمتیہ علامتی حرف qh0 = (سمتیہ علامتی حرف)aux64x2_0;
        سمتیہ علامتی حرف qh1 = (سمتیہ علامتی حرف)aux64x2_1;

        سمتیہ علامتی حرف qxs = (سمتیہ علامتی حرف)vec_xl( 0, x[ib].qs);

        سمتیہ علامتی حرف q5x0 = vec_sub(vec_and (qxs, کم_نقاب), qh0);
        سمتیہ علامتی حرف q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

        سمتیہ علامتی حرف q8y0 = vec_xl(  0, y[ib].qs);
        سمتیہ علامتی حرف q8y1 = vec_xl( 16, y[ib].qs);

        سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
        سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q5x1, q8y1), vec_mulo(q5x1, q8y1));

        qv0 = vec_جوڑو(qv0, qv1);

        سمتیہ علامتی صحیح vsumi0 = vec_جوڑو(vec_unpackh(qv0), vec_unpackl(qv0));

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_نکالو(vsumf0, 0);

    *s = sumf;
#ورنہ
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q5_1_q8_1(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_1;
    ثابت صحیح nb = n / qk;

    صحیح ib = 0;
    اعشاری sumf = 0;

    أكد(n % qk == 0);
    أكد(qk == QK5_1);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q5_1 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_1 * GGML_RESTRICT y = vy;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ علامتی صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);

#ہدایت GCC unroll 4
    برائے (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        سمتیہ اعشاری vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ اعشاری vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].m));
        سمتیہ اعشاری vys = {GGML_CPU_FP16_مقصد_FP32(y[ib].s), 0.f, 0.f, 0.f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        سمتیہ غیرعلامتی طویل طویل aux64x2_0 = {(uint64_t)(جدول_b2b_0[x[ib].qh[0]]), (uint64_t)(جدول_b2b_0[x[ib].qh[1]])};
        سمتیہ غیرعلامتی طویل طویل aux64x2_1 = {(uint64_t)(جدول_b2b_0[x[ib].qh[2]]), (uint64_t)(جدول_b2b_0[x[ib].qh[3]])};

        سمتیہ علامتی حرف qh0 = (سمتیہ علامتی حرف)aux64x2_0;
        سمتیہ علامتی حرف qh1 = (سمتیہ علامتی حرف)aux64x2_1;

        سمتیہ علامتی حرف qxs = (سمتیہ علامتی حرف)vec_xl( 0, x[ib].qs);

        سمتیہ غیرعلامتی حرف q5x0 = (سمتیہ غیرعلامتی حرف)vec_or(vec_and(qxs, کم_نقاب), qh0);
        سمتیہ غیرعلامتی حرف q5x1 = (سمتیہ غیرعلامتی حرف)vec_or(vec_sr(qxs, v4), qh1);

        سمتیہ علامتی حرف q8y0 = vec_xl(  0, y[ib].qs);
        سمتیہ علامتی حرف q8y1 = vec_xl( 16, y[ib].qs);

        سمتیہ علامتی صحیح vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q5x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q5x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_نکالو(vsumf0, 0);

    *s = sumf;
#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q8_0_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    ثابت صحیح qk = QK8_0;
    ثابت صحیح nb = n / qk;

    أكد(n % qk == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q8_0 * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0 * GGML_RESTRICT y = vy;

    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی صحیح v0 = vec_splats((int32_t)0);
    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);

#ہدایت GCC unroll 8
    برائے (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        سمتیہ اعشاری vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی حرف q8x0 = vec_xl( 0, x[ib].qs);
        سمتیہ علامتی حرف q8x1 = vec_xl(16, x[ib].qs);
        سمتیہ علامتی حرف q8y0 = vec_xl( 0, y[ib].qs);
        سمتیہ علامتی حرف q8y1 = vec_xl(16, y[ib].qs);

        سمتیہ علامتی مختصر qv0 = vec_mule(q8x0, q8y0);
        سمتیہ علامتی مختصر qv1 = vec_mulo(q8x0, q8y0);
        سمتیہ علامتی مختصر qv2 = vec_mule(q8x1, q8y1);
        سمتیہ علامتی مختصر qv3 = vec_mulo(q8x1, q8y1);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_جوڑو(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_نکالو(vsumf0, 0);

    *s = sumf;
#ورنہ
    غیر_مستعمل(nb);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q2_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q2_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0x3);
    ثابت سمتیہ علامتی حرف کم_Scale_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ غیرعلامتی حرف v2 = vec_splats((غیرعلامتی حرف)0x2);
    ثابت سمتیہ غیرعلامتی حرف v6 = vec_splats((غیرعلامتی حرف)0x6);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ اعشاری vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].dmin));
        سمتیہ اعشاری vdmin = vec_mul(vxmin, vyd);

        سمتیہ علامتی مختصر q8ysums0 = vec_xl( 0, y[i].bsums);
        سمتیہ علامتی مختصر q8ysums1 = vec_xl(16, y[i].bsums);

        سمتیہ علامتی حرف q2xmins = (سمتیہ علامتی حرف)vec_xl( 0, x[i].scales);
        سمتیہ علامتی حرف vscales = vec_and(q2xmins, کم_Scale_نقاب);

        q2xmins = vec_sr(q2xmins, v4);
        سمتیہ علامتی مختصر q2xmins0 = vec_unpackh(q2xmins);
        سمتیہ علامتی مختصر q2xmins1 = vec_unpackl(q2xmins);

        سمتیہ علامتی صحیح prod0 = vec_mule(q2xmins0, q8ysums0);
        سمتیہ علامتی صحیح prod1 = vec_mulo(q2xmins0, q8ysums0);
        سمتیہ علامتی صحیح prod2 = vec_mule(q2xmins1, q8ysums1);
        سمتیہ علامتی صحیح prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;
        سمتیہ علامتی صحیح vsumi4 = v0;
        سمتیہ علامتی صحیح vsumi5 = v0;
        سمتیہ علامتی صحیح vsumi6 = v0;
        سمتیہ علامتی صحیح vsumi7 = v0;

        ثابت uint8_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        برائے (صحیح j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ علامتی حرف qxs0 = (سمتیہ علامتی حرف)vec_xl( 0, q2);
            سمتیہ علامتی حرف qxs1 = (سمتیہ علامتی حرف)vec_xl(16, q2);
            q2 += 32;

            سمتیہ غیرعلامتی حرف q2x00 = (سمتیہ غیرعلامتی حرف)vec_and(qxs0, کم_نقاب);
            سمتیہ غیرعلامتی حرف q2x01 = (سمتیہ غیرعلامتی حرف)vec_and(vec_sr(qxs0, v2), کم_نقاب);
            سمتیہ غیرعلامتی حرف q2x02 = (سمتیہ غیرعلامتی حرف)vec_and(vec_sr(qxs0, v4), کم_نقاب);
            سمتیہ غیرعلامتی حرف q2x03 = (سمتیہ غیرعلامتی حرف)vec_and(vec_sr(qxs0, v6), کم_نقاب);
            سمتیہ غیرعلامتی حرف q2x10 = (سمتیہ غیرعلامتی حرف)vec_and(qxs1, کم_نقاب);
            سمتیہ غیرعلامتی حرف q2x11 = (سمتیہ غیرعلامتی حرف)vec_and(vec_sr(qxs1, v2), کم_نقاب);
            سمتیہ غیرعلامتی حرف q2x12 = (سمتیہ غیرعلامتی حرف)vec_and(vec_sr(qxs1, v4), کم_نقاب);
            سمتیہ غیرعلامتی حرف q2x13 = (سمتیہ غیرعلامتی حرف)vec_and(vec_sr(qxs1, v6), کم_نقاب);

            سمتیہ علامتی حرف q8y00 = vec_xl(  0, q8);
            سمتیہ علامتی حرف q8y10 = vec_xl( 16, q8);
            سمتیہ علامتی حرف q8y01 = vec_xl( 32, q8);
            سمتیہ علامتی حرف q8y11 = vec_xl( 48, q8);
            سمتیہ علامتی حرف q8y02 = vec_xl( 64, q8);
            سمتیہ علامتی حرف q8y12 = vec_xl( 80, q8);
            سمتیہ علامتی حرف q8y03 = vec_xl( 96, q8);
            سمتیہ علامتی حرف q8y13 = vec_xl(112, q8);
            q8 += 128;

            سمتیہ علامتی صحیح qv0 = vec_msum(q8y00, q2x00, v0);
            سمتیہ علامتی صحیح qv1 = vec_msum(q8y01, q2x01, v0);
            سمتیہ علامتی صحیح qv2 = vec_msum(q8y02, q2x02, v0);
            سمتیہ علامتی صحیح qv3 = vec_msum(q8y03, q2x03, v0);
            سمتیہ علامتی صحیح qv4 = vec_msum(q8y10, q2x10, v0);
            سمتیہ علامتی صحیح qv5 = vec_msum(q8y11, q2x11, v0);
            سمتیہ علامتی صحیح qv6 = vec_msum(q8y12, q2x12, v0);
            سمتیہ علامتی صحیح qv7 = vec_msum(q8y13, q2x13, v0);

            سمتیہ علامتی مختصر vscales_07 = vec_unpackh(vscales);
            سمتیہ علامتی صحیح vscales_03 = vec_unpackh(vscales_07);
            سمتیہ علامتی صحیح vscales_47 = vec_unpackl(vscales_07);
            سمتیہ علامتی صحیح vs0 = vec_splat(vscales_03, 0);
            سمتیہ علامتی صحیح vs1 = vec_splat(vscales_03, 1);
            سمتیہ علامتی صحیح vs2 = vec_splat(vscales_03, 2);
            سمتیہ علامتی صحیح vs3 = vec_splat(vscales_03, 3);
            سمتیہ علامتی صحیح vs4 = vec_splat(vscales_47, 0);
            سمتیہ علامتی صحیح vs5 = vec_splat(vscales_47, 1);
            سمتیہ علامتی صحیح vs6 = vec_splat(vscales_47, 2);
            سمتیہ علامتی صحیح vs7 = vec_splat(vscales_47, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_جوڑو(vec_mul(qv0, vs0), vsumi0);
            vsumi1 = vec_جوڑو(vec_mul(qv1, vs2), vsumi1);
            vsumi2 = vec_جوڑو(vec_mul(qv2, vs4), vsumi2);
            vsumi3 = vec_جوڑو(vec_mul(qv3, vs6), vsumi3);
            vsumi4 = vec_جوڑو(vec_mul(qv4, vs1), vsumi4);
            vsumi5 = vec_جوڑو(vec_mul(qv5, vs3), vsumi5);
            vsumi6 = vec_جوڑو(vec_mul(qv6, vs5), vsumi6);
            vsumi7 = vec_جوڑو(vec_mul(qv7, vs7), vsumi7);
        }

        vsumi0 = vec_جوڑو(vsumi0, vsumi4);
        vsumi1 = vec_جوڑو(vsumi1, vsumi5);
        vsumi2 = vec_جوڑو(vsumi2, vsumi6);
        vsumi3 = vec_جوڑو(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q3_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت uint32_t kmask1 = 0x03030303;
    ثابت uint32_t kmask2 = 0x0f0f0f0f;

    ثابت بلاک_q3_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0x3);
    ثابت سمتیہ علامتی حرف کم_Mask1 = vec_splats((int8_t)0xf);
    ثابت سمتیہ علامتی حرف کم_Mask2 = vec_splats((int8_t)0x30);
    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ علامتی حرف v1 = vec_splats((علامتی حرف)0x1);
    ثابت سمتیہ غیرعلامتی حرف v2 = vec_splats((غیرعلامتی حرف)0x2);
    ثابت سمتیہ غیرعلامتی حرف v3 = vec_splats((غیرعلامتی حرف)0x3);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);
    ثابت سمتیہ غیرعلامتی حرف v6 = vec_splats((غیرعلامتی حرف)0x6);
    ثابت سمتیہ علامتی حرف off = vec_splats((علامتی حرف)0x20);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        غیر_مستعمل(kmask1);
        غیر_مستعمل(kmask2);

        سمتیہ علامتی حرف u0 = (سمتیہ علامتی حرف)vec_xl_len(x[i].scales, 8);
        سمتیہ علامتی حرف u1 = vec_and(u0, کم_Mask1);
        سمتیہ علامتی حرف u2 = (سمتیہ علامتی حرف)vec_xl_len(x[i].scales + 8, 4);
        سمتیہ علامتی حرف u3 = (سمتیہ علامتی حرف)vec_mergeh((سمتیہ علامتی صحیح)u2, (سمتیہ علامتی صحیح)vec_sr(u2, v2));
        سمتیہ علامتی حرف u30 = vec_sl(vec_and(u3, کم_نقاب), v4);
        سمتیہ علامتی حرف u31 = vec_and(u3, کم_Mask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        سمتیہ علامتی حرف vscales = (سمتیہ علامتی حرف)vec_mergeh((سمتیہ علامتی طویل طویل)u1, (سمتیہ علامتی طویل طویل)u2);
        سمتیہ علامتی حرف qxhs0 = (سمتیہ علامتی حرف)vec_xl( 0, x[i].hmask);
        سمتیہ علامتی حرف qxhs1 = (سمتیہ علامتی حرف)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;
        سمتیہ علامتی صحیح vsumi4 = v0;
        سمتیہ علامتی صحیح vsumi5 = v0;
        سمتیہ علامتی صحیح vsumi6 = v0;
        سمتیہ علامتی صحیح vsumi7 = v0;

        ثابت uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        برائے (صحیح j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ علامتی حرف qxs0 = (سمتیہ علامتی حرف)vec_xl( 0, q3);
            سمتیہ علامتی حرف qxs1 = (سمتیہ علامتی حرف)vec_xl(16, q3);
            q3 += 32;

            //the low 2 bits
            سمتیہ علامتی حرف qxs00 = vec_and(qxs0, کم_نقاب);
            سمتیہ علامتی حرف qxs01 = vec_and(vec_sr(qxs0, v2), کم_نقاب);
            سمتیہ علامتی حرف qxs02 = vec_and(vec_sr(qxs0, v4), کم_نقاب);
            سمتیہ علامتی حرف qxs03 = vec_and(vec_sr(qxs0, v6), کم_نقاب);
            سمتیہ علامتی حرف qxs10 = vec_and(qxs1, کم_نقاب);
            سمتیہ علامتی حرف qxs11 = vec_and(vec_sr(qxs1, v2), کم_نقاب);
            سمتیہ علامتی حرف qxs12 = vec_and(vec_sr(qxs1, v4), کم_نقاب);
            سمتیہ علامتی حرف qxs13 = vec_and(vec_sr(qxs1, v6), کم_نقاب);

            //the 3rd bit
            سمتیہ علامتی حرف qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            سمتیہ علامتی حرف qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (سمتیہ غیرعلامتی حرف)v1)), v2);
            سمتیہ علامتی حرف qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            سمتیہ علامتی حرف qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            سمتیہ علامتی حرف qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            سمتیہ علامتی حرف qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (سمتیہ غیرعلامتی حرف)v1)), v2);
            سمتیہ علامتی حرف qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            سمتیہ علامتی حرف qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            سمتیہ علامتی حرف q3x00 = vec_sub(qxs00, qxh00);
            سمتیہ علامتی حرف q3x01 = vec_sub(qxs01, qxh01);
            سمتیہ علامتی حرف q3x02 = vec_sub(qxs02, qxh02);
            سمتیہ علامتی حرف q3x03 = vec_sub(qxs03, qxh03);
            سمتیہ علامتی حرف q3x10 = vec_sub(qxs10, qxh10);
            سمتیہ علامتی حرف q3x11 = vec_sub(qxs11, qxh11);
            سمتیہ علامتی حرف q3x12 = vec_sub(qxs12, qxh12);
            سمتیہ علامتی حرف q3x13 = vec_sub(qxs13, qxh13);

            سمتیہ علامتی حرف q8y00 = vec_xl(  0, q8);
            سمتیہ علامتی حرف q8y10 = vec_xl( 16, q8);
            سمتیہ علامتی حرف q8y01 = vec_xl( 32, q8);
            سمتیہ علامتی حرف q8y11 = vec_xl( 48, q8);
            سمتیہ علامتی حرف q8y02 = vec_xl( 64, q8);
            سمتیہ علامتی حرف q8y12 = vec_xl( 80, q8);
            سمتیہ علامتی حرف q8y03 = vec_xl( 96, q8);
            سمتیہ علامتی حرف q8y13 = vec_xl(112, q8);
            q8 += 128;

            سمتیہ علامتی مختصر vscales_h = vec_unpackh(vscales);
            سمتیہ علامتی مختصر vs0 = vec_splat(vscales_h, 0);
            سمتیہ علامتی مختصر vs1 = vec_splat(vscales_h, 1);
            سمتیہ علامتی مختصر vs2 = vec_splat(vscales_h, 2);
            سمتیہ علامتی مختصر vs3 = vec_splat(vscales_h, 3);
            سمتیہ علامتی مختصر vs4 = vec_splat(vscales_h, 4);
            سمتیہ علامتی مختصر vs5 = vec_splat(vscales_h, 5);
            سمتیہ علامتی مختصر vs6 = vec_splat(vscales_h, 6);
            سمتیہ علامتی مختصر vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            سمتیہ علامتی مختصر qv00 = vec_جوڑو(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            سمتیہ علامتی مختصر qv01 = vec_جوڑو(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            سمتیہ علامتی مختصر qv02 = vec_جوڑو(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            سمتیہ علامتی مختصر qv03 = vec_جوڑو(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            سمتیہ علامتی مختصر qv10 = vec_جوڑو(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            سمتیہ علامتی مختصر qv11 = vec_جوڑو(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            سمتیہ علامتی مختصر qv12 = vec_جوڑو(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            سمتیہ علامتی مختصر qv13 = vec_جوڑو(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs2, vsumi1);
            vsumi2 = vec_msum(qv02, vs4, vsumi2);
            vsumi3 = vec_msum(qv03, vs6, vsumi3);
            vsumi4 = vec_msum(qv10, vs1, vsumi4);
            vsumi5 = vec_msum(qv11, vs3, vsumi5);
            vsumi6 = vec_msum(qv12, vs5, vsumi6);
            vsumi7 = vec_msum(qv13, vs7, vsumi7);
        }

        vsumi0 = vec_جوڑو(vsumi0, vsumi4);
        vsumi1 = vec_جوڑو(vsumi1, vsumi5);
        vsumi2 = vec_جوڑو(vsumi2, vsumi6);
        vsumi3 = vec_جوڑو(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(kmask1);
    غیر_مستعمل(kmask2);
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q4_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q4_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

    ساکن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساکن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساکن ثابت uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ علامتی حرف کم_Mask1 = vec_splats((int8_t)0x3f);
    ثابت سمتیہ علامتی حرف کم_Mask2 = vec_splats((int8_t)0x30);
    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ غیرعلامتی حرف v2 = vec_splats((uint8_t)2);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ اعشاری vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].dmin));
        سمتیہ اعشاری vdmin = vec_mul(vxmin, vyd);

        سمتیہ علامتی مختصر q8ysums0 = vec_xl( 0, y[i].bsums);
        سمتیہ علامتی مختصر q8ysums1 = vec_xl(16, y[i].bsums);

        غیر_مستعمل(kmask1);
        غیر_مستعمل(kmask2);
        غیر_مستعمل(kmask3);
        غیر_مستعمل(utmp);

        سمتیہ علامتی حرف u0 = (سمتیہ علامتی حرف)vec_xl_len(x[i].scales, 8);
        سمتیہ علامتی حرف u1 = vec_and(vec_sr(u0, v2), کم_Mask2);
        سمتیہ علامتی حرف u2 = (سمتیہ علامتی حرف)vec_xl_len(x[i].scales + 8, 4);
        سمتیہ علامتی حرف u3 = vec_sr(u2, v4);

        سمتیہ علامتی حرف u30 = u1;
        سمتیہ علامتی حرف u31 = (سمتیہ علامتی حرف)vec_mergeh((سمتیہ علامتی صحیح)vec_and(u2, کم_نقاب), (سمتیہ علامتی صحیح)u3);

        u1 = vec_and(u0, کم_Mask1);
        u2 = vec_or(u30, u31);

        سمتیہ علامتی حرف utmps = (سمتیہ علامتی حرف)vec_mergeh((سمتیہ علامتی صحیح)u1, (سمتیہ علامتی صحیح)u2);

        سمتیہ علامتی مختصر vscales = vec_unpackh(utmps);
        سمتیہ علامتی مختصر q4xmins = vec_unpackl(utmps);
        سمتیہ علامتی مختصر q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        سمتیہ علامتی مختصر q4xmins1 = vec_mergel(q4xmins, q4xmins);

        سمتیہ علامتی صحیح prod0 = vec_mule(q4xmins0, q8ysums0);
        سمتیہ علامتی صحیح prod1 = vec_mule(q4xmins1, q8ysums1);
        سمتیہ علامتی صحیح prod2 = vec_mulo(q4xmins0, q8ysums0);
        سمتیہ علامتی صحیح prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;

        ثابت uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        برائے (صحیح j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ علامتی حرف qxs0 = (سمتیہ علامتی حرف)vec_xl( 0, q4);
            سمتیہ علامتی حرف qxs1 = (سمتیہ علامتی حرف)vec_xl(16, q4);
            سمتیہ علامتی حرف qxs2 = (سمتیہ علامتی حرف)vec_xl(32, q4);
            سمتیہ علامتی حرف qxs3 = (سمتیہ علامتی حرف)vec_xl(48, q4);
            q4 += 64;

            سمتیہ غیرعلامتی حرف q4x00 = (سمتیہ غیرعلامتی حرف)vec_and(qxs0, کم_نقاب);
            سمتیہ غیرعلامتی حرف q4x01 = (سمتیہ غیرعلامتی حرف)vec_sr(qxs0, v4);
            سمتیہ غیرعلامتی حرف q4x10 = (سمتیہ غیرعلامتی حرف)vec_and(qxs1, کم_نقاب);
            سمتیہ غیرعلامتی حرف q4x11 = (سمتیہ غیرعلامتی حرف)vec_sr(qxs1, v4);
            سمتیہ غیرعلامتی حرف q4x20 = (سمتیہ غیرعلامتی حرف)vec_and(qxs2, کم_نقاب);
            سمتیہ غیرعلامتی حرف q4x21 = (سمتیہ غیرعلامتی حرف)vec_sr(qxs2, v4);
            سمتیہ غیرعلامتی حرف q4x30 = (سمتیہ غیرعلامتی حرف)vec_and(qxs3, کم_نقاب);
            سمتیہ غیرعلامتی حرف q4x31 = (سمتیہ غیرعلامتی حرف)vec_sr(qxs3, v4);

            سمتیہ علامتی حرف q8y00 = vec_xl(  0, q8);
            سمتیہ علامتی حرف q8y10 = vec_xl( 16, q8);
            سمتیہ علامتی حرف q8y01 = vec_xl( 32, q8);
            سمتیہ علامتی حرف q8y11 = vec_xl( 48, q8);
            سمتیہ علامتی حرف q8y20 = vec_xl( 64, q8);
            سمتیہ علامتی حرف q8y30 = vec_xl( 80, q8);
            سمتیہ علامتی حرف q8y21 = vec_xl( 96, q8);
            سمتیہ علامتی حرف q8y31 = vec_xl(112, q8);
            q8 += 128;

            سمتیہ علامتی صحیح qv00 = vec_msum(q8y00, q4x00, v0);
            سمتیہ علامتی صحیح qv01 = vec_msum(q8y01, q4x01, v0);
            سمتیہ علامتی صحیح qv10 = vec_msum(q8y10, q4x10, v0);
            سمتیہ علامتی صحیح qv11 = vec_msum(q8y11, q4x11, v0);
            سمتیہ علامتی صحیح qv20 = vec_msum(q8y20, q4x20, v0);
            سمتیہ علامتی صحیح qv21 = vec_msum(q8y21, q4x21, v0);
            سمتیہ علامتی صحیح qv30 = vec_msum(q8y30, q4x30, v0);
            سمتیہ علامتی صحیح qv31 = vec_msum(q8y31, q4x31, v0);

            سمتیہ علامتی صحیح vscales_h = vec_unpackh(vscales);
            سمتیہ علامتی صحیح vs0 = vec_splat(vscales_h, 0);
            سمتیہ علامتی صحیح vs1 = vec_splat(vscales_h, 1);
            سمتیہ علامتی صحیح vs2 = vec_splat(vscales_h, 2);
            سمتیہ علامتی صحیح vs3 = vec_splat(vscales_h, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_جوڑو(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_جوڑو(vec_mul(qv01, vs1), vsumi1);
            vsumi2 = vec_جوڑو(vec_mul(qv20, vs2), vsumi2);
            vsumi3 = vec_جوڑو(vec_mul(qv21, vs3), vsumi3);

            vsumi0 = vec_جوڑو(vec_mul(qv10, vs0), vsumi0);
            vsumi1 = vec_جوڑو(vec_mul(qv11, vs1), vsumi1);
            vsumi2 = vec_جوڑو(vec_mul(qv30, vs2), vsumi2);
            vsumi3 = vec_جوڑو(vec_mul(qv31, vs3), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    غیر_مستعمل(kmask1);
    غیر_مستعمل(kmask2);
    غیر_مستعمل(kmask3);
    غیر_مستعمل(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q5_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy,  size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q5_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

    ساکن ثابت uint32_t kmask1 = 0x3f3f3f3f;
    ساکن ثابت uint32_t kmask2 = 0x0f0f0f0f;
    ساکن ثابت uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ علامتی حرف کم_Mask1 = vec_splats((int8_t)0x3f);
    ثابت سمتیہ علامتی حرف کم_Mask2 = vec_splats((int8_t)0x30);
    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ غیرعلامتی حرف v1 = vec_splats((غیرعلامتی حرف)0x1);
    ثابت سمتیہ غیرعلامتی حرف v2 = vec_splats((غیرعلامتی حرف)0x2);
    ثابت سمتیہ غیرعلامتی حرف v3 = vec_splats((غیرعلامتی حرف)0x3);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ اعشاری vxmin = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].dmin));
        سمتیہ اعشاری vdmin = vec_mul(vxmin, vyd);

        غیر_مستعمل(kmask1);
        غیر_مستعمل(kmask2);
        غیر_مستعمل(kmask3);
        غیر_مستعمل(utmp);

        سمتیہ علامتی حرف u0 = (سمتیہ علامتی حرف)vec_xl_len(x[i].scales, 8);
        سمتیہ علامتی حرف u1 = vec_and(vec_sr(u0, v2), کم_Mask2);
        سمتیہ علامتی حرف u2 = (سمتیہ علامتی حرف)vec_xl_len(x[i].scales + 8, 4);
        سمتیہ علامتی حرف u3 = vec_sr(u2, v4);

        سمتیہ علامتی حرف u30 = u1;
        سمتیہ علامتی حرف u31 = (سمتیہ علامتی حرف)vec_mergeh((سمتیہ علامتی صحیح)vec_and(u2, کم_نقاب), (سمتیہ علامتی صحیح)u3);

        u1 = vec_and(u0, کم_Mask1);
        u2 = vec_or(u30, u31);

        سمتیہ علامتی حرف utmps = (سمتیہ علامتی حرف)vec_mergeh((سمتیہ علامتی صحیح)u1, (سمتیہ علامتی صحیح)u2);

        سمتیہ علامتی مختصر q8ysums0 = vec_xl( 0, y[i].bsums);
        سمتیہ علامتی مختصر q8ysums1 = vec_xl(16, y[i].bsums);

        سمتیہ علامتی مختصر vscales = vec_unpackh(utmps);

        سمتیہ علامتی مختصر q5xmins = vec_unpackl(utmps);
        سمتیہ علامتی مختصر q5xmins0 = vec_mergeh(q5xmins, q5xmins);
        سمتیہ علامتی مختصر q5xmins1 = vec_mergel(q5xmins, q5xmins);

        سمتیہ علامتی صحیح prod0 = vec_mule(q5xmins0, q8ysums0);
        سمتیہ علامتی صحیح prod1 = vec_mule(q5xmins1, q8ysums1);
        سمتیہ علامتی صحیح prod2 = vec_mulo(q5xmins0, q8ysums0);
        سمتیہ علامتی صحیح prod3 = vec_mulo(q5xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        سمتیہ علامتی حرف qxhs0 = (سمتیہ علامتی حرف)vec_xl( 0, x[i].qh);
        سمتیہ علامتی حرف qxhs1 = (سمتیہ علامتی حرف)vec_xl(16, x[i].qh);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;

        ثابت uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        برائے (صحیح j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q5, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ علامتی حرف qxs0 = (سمتیہ علامتی حرف)vec_xl( 0, q5);
            سمتیہ علامتی حرف qxs1 = (سمتیہ علامتی حرف)vec_xl(16, q5);
            q5 += 32;

            سمتیہ علامتی حرف qxs00 = vec_and(qxs0, کم_نقاب);
            سمتیہ علامتی حرف qxs01 = vec_sr(qxs0, v4);
            سمتیہ علامتی حرف qxs10 = vec_and(qxs1, کم_نقاب);
            سمتیہ علامتی حرف qxs11 = vec_sr(qxs1, v4);

            سمتیہ علامتی حرف q5h00 = vec_sl(vec_and((سمتیہ علامتی حرف)v1, qxhs0), v4);
            سمتیہ علامتی حرف q5h01 = vec_sl(vec_and((سمتیہ علامتی حرف)v2, qxhs0), v3);
            سمتیہ علامتی حرف q5h10 = vec_sl(vec_and((سمتیہ علامتی حرف)v1, qxhs1), v4);
            سمتیہ علامتی حرف q5h11 = vec_sl(vec_and((سمتیہ علامتی حرف)v2, qxhs1), v3);
            qxhs0 = vec_sr(qxhs0, v2);
            qxhs1 = vec_sr(qxhs1, v2);

            سمتیہ غیرعلامتی حرف q5x00 = (سمتیہ غیرعلامتی حرف)vec_or(q5h00, qxs00);
            سمتیہ غیرعلامتی حرف q5x01 = (سمتیہ غیرعلامتی حرف)vec_or(q5h01, qxs01);
            سمتیہ غیرعلامتی حرف q5x10 = (سمتیہ غیرعلامتی حرف)vec_or(q5h10, qxs10);
            سمتیہ غیرعلامتی حرف q5x11 = (سمتیہ غیرعلامتی حرف)vec_or(q5h11, qxs11);

            سمتیہ علامتی حرف q8y00 = vec_xl( 0, q8);
            سمتیہ علامتی حرف q8y10 = vec_xl(16, q8);
            سمتیہ علامتی حرف q8y01 = vec_xl(32, q8);
            سمتیہ علامتی حرف q8y11 = vec_xl(48, q8);
            q8 += 64;

            سمتیہ علامتی صحیح qv00 = vec_msum(q8y00, q5x00, v0);
            سمتیہ علامتی صحیح qv01 = vec_msum(q8y01, q5x01, v0);
            سمتیہ علامتی صحیح qv10 = vec_msum(q8y10, q5x10, v0);
            سمتیہ علامتی صحیح qv11 = vec_msum(q8y11, q5x11, v0);

            سمتیہ علامتی صحیح vscales_h = vec_unpackh(vscales);
            سمتیہ علامتی صحیح vs0 = vec_splat(vscales_h, 0);
            سمتیہ علامتی صحیح vs1 = vec_splat(vscales_h, 1);
            vscales = vec_sld(vscales, vscales, 12);

            vsumi0 = vec_جوڑو(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_جوڑو(vec_mul(qv10, vs0), vsumi1);
            vsumi2 = vec_جوڑو(vec_mul(qv01, vs1), vsumi2);
            vsumi3 = vec_جوڑو(vec_mul(qv11, vs1), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    غیر_مستعمل(kmask1);
    غیر_مستعمل(kmask2);
    غیر_مستعمل(kmask3);
    غیر_مستعمل(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_q6_K_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_q6_K * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ غیرعلامتی حرف v2 = vec_splats((غیرعلامتی حرف)0x2);
    ثابت سمتیہ غیرعلامتی حرف v3 = vec_splats((غیرعلامتی حرف)0x3);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);
    ثابت سمتیہ غیرعلامتی حرف v6 = vec_splats((غیرعلامتی حرف)0x6);
    ثابت سمتیہ علامتی حرف off = vec_splats((علامتی حرف)0x20);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;
        سمتیہ علامتی صحیح vsumi4 = v0;
        سمتیہ علامتی صحیح vsumi5 = v0;
        سمتیہ علامتی صحیح vsumi6 = v0;
        سمتیہ علامتی صحیح vsumi7 = v0;

        ثابت uint8_t * GGML_RESTRICT q6 = x[i].ql;
        ثابت uint8_t * GGML_RESTRICT qh = x[i].qh;
        ثابت int8_t  * GGML_RESTRICT qs = x[i].scales;
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

        برائے (صحیح j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q6, 0, 0);
            __builtin_prefetch(qh, 0, 0);
            __builtin_prefetch(q8, 0, 0);

            سمتیہ علامتی حرف qxs0 = (سمتیہ علامتی حرف)vec_xl( 0, q6);
            سمتیہ علامتی حرف qxs1 = (سمتیہ علامتی حرف)vec_xl(16, q6);
            سمتیہ علامتی حرف qxs2 = (سمتیہ علامتی حرف)vec_xl(32, q6);
            سمتیہ علامتی حرف qxs3 = (سمتیہ علامتی حرف)vec_xl(48, q6);
            q6 += 64;

            سمتیہ علامتی حرف qxs00 = vec_and(qxs0, کم_نقاب);
            سمتیہ علامتی حرف qxs01 = vec_sr(qxs0, v4);
            سمتیہ علامتی حرف qxs10 = vec_and(qxs1, کم_نقاب);
            سمتیہ علامتی حرف qxs11 = vec_sr(qxs1, v4);
            سمتیہ علامتی حرف qxs20 = vec_and(qxs2, کم_نقاب);
            سمتیہ علامتی حرف qxs21 = vec_sr(qxs2, v4);
            سمتیہ علامتی حرف qxs30 = vec_and(qxs3, کم_نقاب);
            سمتیہ علامتی حرف qxs31 = vec_sr(qxs3, v4);

            سمتیہ علامتی حرف qxhs0 = (سمتیہ علامتی حرف)vec_xl( 0, qh);
            سمتیہ علامتی حرف qxhs1 = (سمتیہ علامتی حرف)vec_xl(16, qh);
            qh += 32;

            سمتیہ علامتی حرف qxh00 = vec_sl(vec_and((سمتیہ علامتی حرف)v3, qxhs0), v4);
            سمتیہ علامتی حرف qxh01 = vec_sl(vec_and((سمتیہ علامتی حرف)v3, vec_sr(qxhs0, v4)), v4);
            سمتیہ علامتی حرف qxh10 = vec_sl(vec_and((سمتیہ علامتی حرف)v3, qxhs1), v4);
            سمتیہ علامتی حرف qxh11 = vec_sl(vec_and((سمتیہ علامتی حرف)v3, vec_sr(qxhs1, v4)), v4);
            سمتیہ علامتی حرف qxh20 = vec_sl(vec_and((سمتیہ علامتی حرف)v3, vec_sr(qxhs0, v2)), v4);
            سمتیہ علامتی حرف qxh21 = vec_sl(vec_and((سمتیہ علامتی حرف)v3, vec_sr(qxhs0, v6)), v4);
            سمتیہ علامتی حرف qxh30 = vec_sl(vec_and((سمتیہ علامتی حرف)v3, vec_sr(qxhs1, v2)), v4);
            سمتیہ علامتی حرف qxh31 = vec_sl(vec_and((سمتیہ علامتی حرف)v3, vec_sr(qxhs1, v6)), v4);

            سمتیہ علامتی حرف q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
            سمتیہ علامتی حرف q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
            سمتیہ علامتی حرف q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
            سمتیہ علامتی حرف q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
            سمتیہ علامتی حرف q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
            سمتیہ علامتی حرف q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
            سمتیہ علامتی حرف q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
            سمتیہ علامتی حرف q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

            سمتیہ علامتی حرف q8y00 = vec_xl(  0, q8);
            سمتیہ علامتی حرف q8y10 = vec_xl( 16, q8);
            سمتیہ علامتی حرف q8y20 = vec_xl( 32, q8);
            سمتیہ علامتی حرف q8y30 = vec_xl( 48, q8);
            سمتیہ علامتی حرف q8y01 = vec_xl( 64, q8);
            سمتیہ علامتی حرف q8y11 = vec_xl( 80, q8);
            سمتیہ علامتی حرف q8y21 = vec_xl( 96, q8);
            سمتیہ علامتی حرف q8y31 = vec_xl(112, q8);
            q8 += 128;

            سمتیہ علامتی مختصر qv00 = vec_جوڑو(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
            سمتیہ علامتی مختصر qv10 = vec_جوڑو(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
            سمتیہ علامتی مختصر qv20 = vec_جوڑو(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
            سمتیہ علامتی مختصر qv30 = vec_جوڑو(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
            سمتیہ علامتی مختصر qv01 = vec_جوڑو(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
            سمتیہ علامتی مختصر qv11 = vec_جوڑو(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
            سمتیہ علامتی مختصر qv21 = vec_جوڑو(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
            سمتیہ علامتی مختصر qv31 = vec_جوڑو(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

            سمتیہ علامتی مختصر vscales = vec_unpackh(vec_xl_len(qs, 8));
            qs += 8;

            سمتیہ علامتی مختصر vs0 = vec_splat(vscales, 0);
            سمتیہ علامتی مختصر vs1 = vec_splat(vscales, 1);
            سمتیہ علامتی مختصر vs2 = vec_splat(vscales, 2);
            سمتیہ علامتی مختصر vs3 = vec_splat(vscales, 3);
            سمتیہ علامتی مختصر vs4 = vec_splat(vscales, 4);
            سمتیہ علامتی مختصر vs5 = vec_splat(vscales, 5);
            سمتیہ علامتی مختصر vs6 = vec_splat(vscales, 6);
            سمتیہ علامتی مختصر vs7 = vec_splat(vscales, 7);

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs4, vsumi1);
            vsumi2 = vec_msum(qv10, vs1, vsumi2);
            vsumi3 = vec_msum(qv11, vs5, vsumi3);
            vsumi4 = vec_msum(qv20, vs2, vsumi4);
            vsumi5 = vec_msum(qv21, vs6, vsumi5);
            vsumi6 = vec_msum(qv30, vs3, vsumi6);
            vsumi7 = vec_msum(qv31, vs7, vsumi7);
        }

        vsumi0 = vec_جوڑو(vsumi0, vsumi4);
        vsumi1 = vec_جوڑو(vsumi1, vsumi5);
        vsumi2 = vec_جوڑو(vsumi2, vsumi6);
        vsumi3 = vec_جوڑو(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

#اگر defined (__POWER9_VECTOR__)
ساکن ثابت int8_t keven_signs_q2xs[1024] = {
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
#اگر_ختم

خالی ggml_vec_dot_iq2_xxs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq2_xxs * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K    * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);
    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;

        ثابت uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        برائے (صحیح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            ثابت uint8_t * aux8 = (ثابت uint8_t *)aux32;

            حافظہ_نقل(aux32, q2, 4*حجم(uint32_t));
            q2 += 8;

            سمتیہ علامتی طویل طویل aux64x2_0 = {*(ثابت int64_t *)(iq2xxs_grid + aux8[ 0]), *(ثابت int64_t *)(iq2xxs_grid + aux8[ 1])};
            سمتیہ علامتی طویل طویل aux64x2_1 = {*(ثابت int64_t *)(iq2xxs_grid + aux8[ 2]), *(ثابت int64_t *)(iq2xxs_grid + aux8[ 3])};
            سمتیہ علامتی طویل طویل aux64x2_2 = {*(ثابت int64_t *)(iq2xxs_grid + aux8[ 8]), *(ثابت int64_t *)(iq2xxs_grid + aux8[ 9])};
            سمتیہ علامتی طویل طویل aux64x2_3 = {*(ثابت int64_t *)(iq2xxs_grid + aux8[10]), *(ثابت int64_t *)(iq2xxs_grid + aux8[11])};

            سمتیہ علامتی طویل طویل vsigns0 = {*(ثابت int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(ثابت int64_t *)(signs64 + ((aux32[1] >>  7) & 127))};
            سمتیہ علامتی طویل طویل vsigns1 = {*(ثابت int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(ثابت int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            سمتیہ علامتی طویل طویل vsigns2 = {*(ثابت int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(ثابت int64_t *)(signs64 + ((aux32[3] >>  7) & 127))};
            سمتیہ علامتی طویل طویل vsigns3 = {*(ثابت int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(ثابت int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            سمتیہ علامتی حرف q2x0 = (سمتیہ علامتی حرف)vec_mul((سمتیہ علامتی حرف)vsigns0, (سمتیہ علامتی حرف)aux64x2_0);
            سمتیہ علامتی حرف q2x1 = (سمتیہ علامتی حرف)vec_mul((سمتیہ علامتی حرف)vsigns1, (سمتیہ علامتی حرف)aux64x2_1);
            سمتیہ علامتی حرف q2x2 = (سمتیہ علامتی حرف)vec_mul((سمتیہ علامتی حرف)vsigns2, (سمتیہ علامتی حرف)aux64x2_2);
            سمتیہ علامتی حرف q2x3 = (سمتیہ علامتی حرف)vec_mul((سمتیہ علامتی حرف)vsigns3, (سمتیہ علامتی حرف)aux64x2_3);

            سمتیہ علامتی حرف q8y0 = vec_xl( 0, q8);
            سمتیہ علامتی حرف q8y1 = vec_xl(16, q8);
            سمتیہ علامتی حرف q8y2 = vec_xl(32, q8);
            سمتیہ علامتی حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            سمتیہ علامتی مختصر qv2 = vec_جوڑو(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            سمتیہ علامتی مختصر qv3 = vec_جوڑو(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ثابت uint16_t ls0 = aux32[1] >> 28;
            ثابت uint16_t ls1 = aux32[3] >> 28;

            سمتیہ علامتی مختصر vscales01 = vec_splats((int16_t)(2*ls0+1));
            سمتیہ علامتی مختصر vscales23 = vec_splats((int16_t)(2*ls1+1));

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

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq2_xs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq2_xs * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K   * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);
    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;

        ثابت uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ثابت uint8_t  * GGML_RESTRICT sc = x[i].scales;
        ثابت int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        برائے (صحیح j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ علامتی طویل طویل aux64x2_0 = {*(ثابت int64_t *)(iq2xs_grid + (q2[0] & 511)), *(ثابت int64_t *)(iq2xs_grid + (q2[1] & 511))};
            سمتیہ علامتی طویل طویل aux64x2_1 = {*(ثابت int64_t *)(iq2xs_grid + (q2[2] & 511)), *(ثابت int64_t *)(iq2xs_grid + (q2[3] & 511))};
            سمتیہ علامتی طویل طویل aux64x2_2 = {*(ثابت int64_t *)(iq2xs_grid + (q2[4] & 511)), *(ثابت int64_t *)(iq2xs_grid + (q2[5] & 511))};
            سمتیہ علامتی طویل طویل aux64x2_3 = {*(ثابت int64_t *)(iq2xs_grid + (q2[6] & 511)), *(ثابت int64_t *)(iq2xs_grid + (q2[7] & 511))};

            سمتیہ علامتی طویل طویل vsigns0 = {*(ثابت int64_t *)(signs64 + ((q2[0] >> 9))), *(ثابت int64_t *)(signs64 + ((q2[1] >> 9)))};
            سمتیہ علامتی طویل طویل vsigns1 = {*(ثابت int64_t *)(signs64 + ((q2[2] >> 9))), *(ثابت int64_t *)(signs64 + ((q2[3] >> 9)))};
            سمتیہ علامتی طویل طویل vsigns2 = {*(ثابت int64_t *)(signs64 + ((q2[4] >> 9))), *(ثابت int64_t *)(signs64 + ((q2[5] >> 9)))};
            سمتیہ علامتی طویل طویل vsigns3 = {*(ثابت int64_t *)(signs64 + ((q2[6] >> 9))), *(ثابت int64_t *)(signs64 + ((q2[7] >> 9)))};
            q2 += 8;

            سمتیہ علامتی حرف q2x0 = (سمتیہ علامتی حرف)vec_mul((سمتیہ علامتی حرف)vsigns0, (سمتیہ علامتی حرف)aux64x2_0);
            سمتیہ علامتی حرف q2x1 = (سمتیہ علامتی حرف)vec_mul((سمتیہ علامتی حرف)vsigns1, (سمتیہ علامتی حرف)aux64x2_1);
            سمتیہ علامتی حرف q2x2 = (سمتیہ علامتی حرف)vec_mul((سمتیہ علامتی حرف)vsigns2, (سمتیہ علامتی حرف)aux64x2_2);
            سمتیہ علامتی حرف q2x3 = (سمتیہ علامتی حرف)vec_mul((سمتیہ علامتی حرف)vsigns3, (سمتیہ علامتی حرف)aux64x2_3);

            سمتیہ علامتی حرف q8y0 = vec_xl( 0, q8);
            سمتیہ علامتی حرف q8y1 = vec_xl(16, q8);
            سمتیہ علامتی حرف q8y2 = vec_xl(32, q8);
            سمتیہ علامتی حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            سمتیہ علامتی مختصر qv2 = vec_جوڑو(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            سمتیہ علامتی مختصر qv3 = vec_جوڑو(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ثابت uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ثابت uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ثابت uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            سمتیہ علامتی مختصر vscales0 = vec_splats((int16_t)(2*ls0+1));
            سمتیہ علامتی مختصر vscales1 = vec_splats((int16_t)(2*ls1+1));
            سمتیہ علامتی مختصر vscales2 = vec_splats((int16_t)(2*ls2+1));
            سمتیہ علامتی مختصر vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq2_s_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq2_s * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ساکن ثابت uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ساکن ثابت uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    ثابت سمتیہ غیرعلامتی حرف mask0 = vec_xl( 0, k_mask1);
    ثابت سمتیہ غیرعلامتی حرف mask1 = vec_xl(16, k_mask1);
    ثابت سمتیہ علامتی حرف mask2 = (سمتیہ علامتی حرف)vec_xl( 0, k_mask2);

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;

        ثابت uint8_t *  GGML_RESTRICT q2 = x[i].qs;
        ثابت uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ثابت uint16_t * GGML_RESTRICT signs = (ثابت uint16_t *)(x[i].qs + QK_K/8);
        ثابت uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ثابت int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        برائے (صحیح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ علامتی طویل طویل aux64x2_0 = {*(ثابت int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(ثابت int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            سمتیہ علامتی طویل طویل aux64x2_1 = {*(ثابت int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(ثابت int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            سمتیہ علامتی طویل طویل aux64x2_2 = {*(ثابت int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(ثابت int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            سمتیہ علامتی طویل طویل aux64x2_3 = {*(ثابت int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(ثابت int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            سمتیہ علامتی حرف vsigns01 = (سمتیہ علامتی حرف)vec_splats(*(ثابت uint32_t *)&signs[0]);
            سمتیہ علامتی حرف vsigns23 = (سمتیہ علامتی حرف)vec_splats(*(ثابت uint32_t *)&signs[2]);
            signs += 4;

            سمتیہ علامتی حرف vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            سمتیہ علامتی حرف vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            سمتیہ علامتی حرف vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            سمتیہ علامتی حرف vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (سمتیہ علامتی حرف)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (سمتیہ علامتی حرف)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (سمتیہ علامتی حرف)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (سمتیہ علامتی حرف)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            سمتیہ علامتی حرف q2x0 = vec_sub(vec_xor(vsigns0, (سمتیہ علامتی حرف)aux64x2_0), vsigns0);
            سمتیہ علامتی حرف q2x1 = vec_sub(vec_xor(vsigns1, (سمتیہ علامتی حرف)aux64x2_1), vsigns1);
            سمتیہ علامتی حرف q2x2 = vec_sub(vec_xor(vsigns2, (سمتیہ علامتی حرف)aux64x2_2), vsigns2);
            سمتیہ علامتی حرف q2x3 = vec_sub(vec_xor(vsigns3, (سمتیہ علامتی حرف)aux64x2_3), vsigns3);

            سمتیہ علامتی حرف q8y0 = vec_xl( 0, q8);
            سمتیہ علامتی حرف q8y1 = vec_xl(16, q8);
            سمتیہ علامتی حرف q8y2 = vec_xl(32, q8);
            سمتیہ علامتی حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            سمتیہ علامتی مختصر qv2 = vec_جوڑو(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            سمتیہ علامتی مختصر qv3 = vec_جوڑو(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ثابت uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ثابت uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ثابت uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            سمتیہ علامتی مختصر vscales0 = vec_splats((int16_t)(2*ls0+1));
            سمتیہ علامتی مختصر vscales1 = vec_splats((int16_t)(2*ls1+1));
            سمتیہ علامتی مختصر vscales2 = vec_splats((int16_t)(2*ls2+1));
            سمتیہ علامتی مختصر vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq3_xxs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq3_xxs * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K    * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ثابت uint64_t * signs64 = (ثابت uint64_t *)keven_signs_q2xs;

    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;

        ثابت uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ثابت uint32_t * GGML_RESTRICT signs = (ثابت uint32_t *)(x[i].qs + QK_K/4);
        ثابت int8_t  * GGML_RESTRICT q8 = y[i].qs;

#ہدایت GCC unroll 1
        برائے (صحیح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ غیرعلامتی صحیح aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            سمتیہ غیرعلامتی صحیح aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            سمتیہ غیرعلامتی صحیح aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            سمتیہ غیرعلامتی صحیح aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            سمتیہ غیرعلامتی طویل طویل aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            سمتیہ غیرعلامتی طویل طویل aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            سمتیہ غیرعلامتی طویل طویل aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            سمتیہ غیرعلامتی طویل طویل aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            سمتیہ علامتی حرف q3x0 = vec_mul((سمتیہ علامتی حرف)aux64x2_0, (سمتیہ علامتی حرف)aux32x4_0);
            سمتیہ علامتی حرف q3x1 = vec_mul((سمتیہ علامتی حرف)aux64x2_1, (سمتیہ علامتی حرف)aux32x4_1);
            سمتیہ علامتی حرف q3x2 = vec_mul((سمتیہ علامتی حرف)aux64x2_2, (سمتیہ علامتی حرف)aux32x4_2);
            سمتیہ علامتی حرف q3x3 = vec_mul((سمتیہ علامتی حرف)aux64x2_3, (سمتیہ علامتی حرف)aux32x4_3);

            سمتیہ علامتی حرف q8y0 = vec_xl( 0, q8);
            سمتیہ علامتی حرف q8y1 = vec_xl(16, q8);
            سمتیہ علامتی حرف q8y2 = vec_xl(32, q8);
            سمتیہ علامتی حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            سمتیہ علامتی مختصر qv2 = vec_جوڑو(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            سمتیہ علامتی مختصر qv3 = vec_جوڑو(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            ثابت uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            سمتیہ علامتی مختصر vscales01 = (سمتیہ علامتی مختصر)vec_splats((uint16_t)(2*ls0+1));
            سمتیہ علامتی مختصر vscales23 = (سمتیہ علامتی مختصر)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.25f * vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq3_s_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq3_s * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ساکن ثابت uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ساکن ثابت uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    ثابت سمتیہ غیرعلامتی حرف mask0 = vec_xl( 0, k_mask1);
    ثابت سمتیہ غیرعلامتی حرف mask1 = vec_xl(16, k_mask1);
    ثابت سمتیہ علامتی حرف mask2 = (سمتیہ علامتی حرف)vec_xl( 0, k_mask2);

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        ثابت uint8_t *  GGML_RESTRICT q3 = x[i].qs;
        ثابت uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ثابت uint16_t * GGML_RESTRICT signs = (ثابت uint16_t *)(x[i].signs);
        ثابت uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ثابت int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;

        برائے (صحیح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ غیرعلامتی صحیح aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            سمتیہ غیرعلامتی صحیح aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            سمتیہ غیرعلامتی صحیح aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            سمتیہ غیرعلامتی صحیح aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            سمتیہ علامتی حرف vsigns01 = (سمتیہ علامتی حرف)vec_splats(*(ثابت uint32_t *)&signs[0]);
            سمتیہ علامتی حرف vsigns02 = (سمتیہ علامتی حرف)vec_splats(*(ثابت uint32_t *)&signs[2]);
            signs += 4;

            سمتیہ علامتی حرف vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            سمتیہ علامتی حرف vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            سمتیہ علامتی حرف vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            سمتیہ علامتی حرف vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (سمتیہ علامتی حرف)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (سمتیہ علامتی حرف)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (سمتیہ علامتی حرف)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (سمتیہ علامتی حرف)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            سمتیہ علامتی حرف q3x0 = vec_sub(vec_xor(vsigns0, (سمتیہ علامتی حرف)aux32x4_0), vsigns0);
            سمتیہ علامتی حرف q3x1 = vec_sub(vec_xor(vsigns1, (سمتیہ علامتی حرف)aux32x4_1), vsigns1);
            سمتیہ علامتی حرف q3x2 = vec_sub(vec_xor(vsigns2, (سمتیہ علامتی حرف)aux32x4_2), vsigns2);
            سمتیہ علامتی حرف q3x3 = vec_sub(vec_xor(vsigns3, (سمتیہ علامتی حرف)aux32x4_3), vsigns3);

            سمتیہ علامتی حرف q8y0 = vec_xl( 0, q8);
            سمتیہ علامتی حرف q8y1 = vec_xl(16, q8);
            سمتیہ علامتی حرف q8y2 = vec_xl(32, q8);
            سمتیہ علامتی حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            سمتیہ علامتی مختصر qv2 = vec_جوڑو(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            سمتیہ علامتی مختصر qv3 = vec_جوڑو(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ثابت uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            سمتیہ علامتی مختصر vscales01 = (سمتیہ علامتی مختصر)vec_splats((uint16_t)(2*ls0+1));
            سمتیہ علامتی مختصر vscales23 = (سمتیہ علامتی مختصر)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq1_s_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(n % QK_K == 0);
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);

    ثابت بلاک_iq1_s * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K  * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ غیرعلامتی حرف v0 = vec_splats((غیرعلامتی حرف)0x0);
    ثابت سمتیہ غیرعلامتی مختصر vsign = vec_splats((غیرعلامتی مختصر)0x8000);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    برائے (صحیح i = 0; i < nb; ++i) {
        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[i].d));
        سمتیہ اعشاری vyd = vec_splats(y[i].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی صحیح vsumi0 = vec_splats((int32_t)0);
        سمتیہ علامتی صحیح vsumi1 = vec_splats((int32_t)0);
        سمتیہ علامتی صحیح vsumi2 = vec_splats((int32_t)0);
        سمتیہ علامتی صحیح vsumi3 = vec_splats((int32_t)0);
        سمتیہ علامتی صحیح vsumi8 = vec_splats((int32_t)0);

        ثابت uint8_t  * GGML_RESTRICT q1 = x[i].qs;
        ثابت uint16_t * GGML_RESTRICT qh = x[i].qh;
        ثابت int8_t   * GGML_RESTRICT q8 = y[i].qs;
        ثابت int16_t  * GGML_RESTRICT qs = y[i].bsums;

        برائے (صحیح j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ علامتی طویل طویل aux64x2_0 = {*(ثابت int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(ثابت int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            سمتیہ علامتی طویل طویل aux64x2_1 = {*(ثابت int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(ثابت int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            سمتیہ علامتی طویل طویل aux64x2_2 = {*(ثابت int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(ثابت int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            سمتیہ علامتی طویل طویل aux64x2_3 = {*(ثابت int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(ثابت int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            سمتیہ علامتی حرف q1x0 = (سمتیہ علامتی حرف)aux64x2_0;
            سمتیہ علامتی حرف q1x1 = (سمتیہ علامتی حرف)aux64x2_1;
            سمتیہ علامتی حرف q1x2 = (سمتیہ علامتی حرف)aux64x2_2;
            سمتیہ علامتی حرف q1x3 = (سمتیہ علامتی حرف)aux64x2_3;

            سمتیہ علامتی حرف q8y0 = vec_xl( 0, q8);
            سمتیہ علامتی حرف q8y1 = vec_xl(16, q8);
            سمتیہ علامتی حرف q8y2 = vec_xl(32, q8);
            سمتیہ علامتی حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            سمتیہ علامتی مختصر qv2 = vec_جوڑو(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            سمتیہ علامتی مختصر qv3 = vec_جوڑو(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            ثابت uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            ثابت uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            سمتیہ علامتی مختصر vscales01 = (سمتیہ علامتی مختصر)vec_splats((uint16_t)(2*ls0+1));
            سمتیہ علامتی مختصر vscales23 = (سمتیہ علامتی مختصر)vec_splats((uint16_t)(2*ls1+1));
            سمتیہ علامتی مختصر vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            سمتیہ علامتی مختصر q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (سمتیہ علامتی مختصر)v0);

            سمتیہ علامتی مختصر qxh = (سمتیہ علامتی مختصر)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            سمتیہ __bool مختصر vsel = vec_cmpge(qxh, (سمتیہ علامتی مختصر)v0);

            سمتیہ علامتی مختصر q8ysum = vec_sel((سمتیہ علامتی مختصر)vec_xor((سمتیہ غیرعلامتی مختصر)q8ysums, vsign), q8ysums, vsel);

            vsumi8 = vec_جوڑو(vec_mule(q8ysum, vscales), vsumi8);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);

        vsumf0 = vec_madd(vec_ctf(vsumi8, 0), vec_mul(vd, vec_splats(IQ1S_DELTA)), vsumf0);
    }

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq4_nl_q8_0(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);
    أكد(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ثابت بلاک_iq4_nl * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_0   * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK4_NL;

    صحیح ib = 0;
    اعشاری sumf = 0;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ علامتی صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);

    ثابت سمتیہ علامتی حرف اقدار = vec_xl( 0, kvalues_iq4nl);

#ہدایت GCC unroll 4
    برائے (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ib].d));
        سمتیہ اعشاری vyd = vec_splats(GGML_CPU_FP16_مقصد_FP32(y[ib].d));
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی حرف qxs = (سمتیہ علامتی حرف)vec_xl( 0, x[ib].qs);
        سمتیہ علامتی حرف q4x0 = vec_and(qxs, کم_نقاب);
        سمتیہ علامتی حرف q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(اقدار, اقدار, (سمتیہ غیرعلامتی حرف)q4x0);
        q4x1 = vec_perm(اقدار, اقدار, (سمتیہ غیرعلامتی حرف)q4x1);

        سمتیہ علامتی حرف q8y0 = vec_xl( 0, y[ib].qs);
        سمتیہ علامتی حرف q8y1 = vec_xl(16, y[ib].qs);

        سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    }

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_نکالو(vsumf0, 0);

    *s = sumf;
#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    غیر_مستعمل(ib);
    غیر_مستعمل(sumf);
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}

خالی ggml_vec_dot_iq4_xs_q8_K(صحیح n, اعشاری * GGML_RESTRICT s, size_t bs, ثابت خالی * GGML_RESTRICT vx, size_t bx, ثابت خالی * GGML_RESTRICT vy, size_t by, صحیح nrc) {
    أكد(nrc == 1);
    غیر_مستعمل(nrc);
    غیر_مستعمل(bx);
    غیر_مستعمل(by);
    غیر_مستعمل(bs);
    أكد(n % QK_K == 0);

    ثابت بلاک_iq4_xs * GGML_RESTRICT x = vx;
    ثابت بلاک_q8_K   * GGML_RESTRICT y = vy;

    ثابت صحیح nb = n / QK_K;

#اگر defined(__POWER9_VECTOR__)
    ثابت سمتیہ علامتی حرف کم_نقاب = vec_splats((علامتی حرف)0xF);
    ثابت سمتیہ صحیح v0 = vec_splats((int32_t)0);
    ثابت سمتیہ غیرعلامتی حرف v4 = vec_splats((غیرعلامتی حرف)0x4);

    سمتیہ اعشاری vsumf0 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf1 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf2 = vec_splats(0.0f);
    سمتیہ اعشاری vsumf3 = vec_splats(0.0f);

    ثابت سمتیہ علامتی حرف اقدار = vec_xl( 0, kvalues_iq4nl);

    برائے (صحیح ibl = 0; ibl < nb; ++ibl) {

        سمتیہ اعشاری vxd = vec_splats(GGML_CPU_FP16_مقصد_FP32(x[ibl].d));
        سمتیہ اعشاری vyd = vec_splats(y[ibl].d);
        سمتیہ اعشاری vd = vec_mul(vxd, vyd);

        سمتیہ علامتی صحیح vsumi0 = v0;
        سمتیہ علامتی صحیح vsumi1 = v0;
        سمتیہ علامتی صحیح vsumi2 = v0;
        سمتیہ علامتی صحیح vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        ثابت uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        ثابت uint8_t * GGML_RESTRICT sc = x[ibl].scales_l;
        ثابت int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        برائے (صحیح ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            سمتیہ علامتی حرف qxs0 = (سمتیہ علامتی حرف)vec_xl( 0, q4);
            سمتیہ علامتی حرف qxs1 = (سمتیہ علامتی حرف)vec_xl(16, q4);
            q4 += 32;

            سمتیہ علامتی حرف q4x00 = (سمتیہ علامتی حرف)vec_and(qxs0, کم_نقاب);
            سمتیہ علامتی حرف q4x01 = (سمتیہ علامتی حرف)vec_sr(qxs0, v4);
            سمتیہ علامتی حرف q4x10 = (سمتیہ علامتی حرف)vec_and(qxs1, کم_نقاب);
            سمتیہ علامتی حرف q4x11 = (سمتیہ علامتی حرف)vec_sr(qxs1, v4);

            q4x00 = vec_perm(اقدار, اقدار, (سمتیہ غیرعلامتی حرف)q4x00);
            q4x01 = vec_perm(اقدار, اقدار, (سمتیہ غیرعلامتی حرف)q4x01);
            q4x10 = vec_perm(اقدار, اقدار, (سمتیہ غیرعلامتی حرف)q4x10);
            q4x11 = vec_perm(اقدار, اقدار, (سمتیہ غیرعلامتی حرف)q4x11);

            سمتیہ علامتی حرف q8y0 = vec_xl( 0, q8);
            سمتیہ علامتی حرف q8y1 = vec_xl(16, q8);
            سمتیہ علامتی حرف q8y2 = vec_xl(32, q8);
            سمتیہ علامتی حرف q8y3 = vec_xl(48, q8);
            q8 += 64;

            سمتیہ علامتی مختصر qv0 = vec_جوڑو(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            سمتیہ علامتی مختصر qv1 = vec_جوڑو(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            سمتیہ علامتی مختصر qv2 = vec_جوڑو(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            سمتیہ علامتی مختصر qv3 = vec_جوڑو(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            ثابت uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            ثابت uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            سمتیہ علامتی مختصر vscales01 = vec_splats((int16_t)ls0);
            سمتیہ علامتی مختصر vscales23 = vec_splats((int16_t)ls1);

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

    vsumf0 = vec_جوڑو(vsumf0, vsumf2);
    vsumf1 = vec_جوڑو(vsumf1, vsumf3);

    vsumf0 = vec_جوڑو(vsumf0, vsumf1);

    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_جوڑو(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_نکالو(vsumf0, 0);

#ورنہ
    غیر_مستعمل(x);
    غیر_مستعمل(y);
    غیر_مستعمل(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#اگر_ختم
}
