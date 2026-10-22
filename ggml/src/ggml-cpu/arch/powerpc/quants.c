#define GGML_COMMON_IMPL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "simd-mappings.h"

#ಅನ್ತರ್ಭಾವ "../../quants.h"
#ಅನ್ತರ್ಭಾವ "../../ggml-cpu-impl.h"

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <float.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h> // for qsort
#ಅನ್ತರ್ಭಾವ <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

#ಯದಿ defined(__POWER9_VECTOR__)
#define B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#define B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#define B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#define B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#define B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#define B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#define B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#define B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
ಸ್ಥಿರ ಧ್ರುವ uint64_t table_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
ಸ್ಥಿರ ಧ್ರುವ uint64_t table_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#endif

ಶೂನ್ಯ quantize_row_q8_0(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k) {
    ಪ್ರತಿಜ್ಞಾ(QK8_0 == 32);
    ಪ್ರತಿಜ್ಞಾ(k % QK8_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; i++) {
        vector ಭಿನ್ನ srcv [8];
        vector ಭಿನ್ನ asrcv[8];
        vector ಭಿನ್ನ amaxv[8];
        vector signed ಪೂರ್ಣಾಙ್ಕ vi[8];

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ಧ್ರುವ ಭಿನ್ನ amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        ಧ್ರುವ ಭಿನ್ನ d = amax / ((1 << 7) - 1);
        ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f/d : 0.0f;
        ಧ್ರುವ vector ಭಿನ್ನ vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; j++) {
            ಧ್ರುವ vector ಭಿನ್ನ v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }
#ಅನ್ಯಥಾ
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}

ಶೂನ್ಯ quantize_row_q8_1(ಧ್ರುವ ಭಿನ್ನ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k) {
    ಪ್ರತಿಜ್ಞಾ(k % QK8_1 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = k / QK8_1;

    block_q8_1 * GGML_RESTRICT y = vy;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; i++) {
        vector ಭಿನ್ನ srcv [8];
        vector ಭಿನ್ನ asrcv[8];
        vector ಭಿನ್ನ amaxv[8];
        vector signed ಪೂರ್ಣಾಙ್ಕ vi[8];

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ಧ್ರುವ ಭಿನ್ನ amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        ಧ್ರುವ ಭಿನ್ನ d = amax / ((1 << 7) - 1);
        ಧ್ರುವ ಭಿನ್ನ id = d ? 1.0f/d : 0.0f;
        ಧ್ರುವ vector ಭಿನ್ನ vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        vector ಪೂರ್ಣಾಙ್ಕ accv = vec_splats(0);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; j++) {
            ಧ್ರುವ vector ಭಿನ್ನ v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_add(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_add(accv, vec_sld(accv, accv, 4));
        accv = vec_add(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_CPU_FP32_TO_FP16(d * vec_extract(accv, 0));
    }

#ಅನ್ಯಥಾ
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}


//===================================== Dot products =================================

ಶೂನ್ಯ ggml_vec_dot_q4_0_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ qk = QK8_0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / qk;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_q4_0 * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_0 * GGML_RESTRICT y = vy;

    ಪೂರ್ಣಾಙ್ಕ ib = 0;
    ಭಿನ್ನ sumf = 0;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector signed ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);
    ಧ್ರುವ vector signed ವರ್ಣ v8 = vec_splats((signed ವರ್ಣ)0x8);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
    ಪ್ರತಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector ಭಿನ್ನ vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ವರ್ಣ qxs = (vector signed ವರ್ಣ)vec_xl( 0, x[ib].qs);
        vector signed ವರ್ಣ q8y0 = vec_xl( 0, y[ib].qs);
        vector signed ವರ್ಣ q8y1 = vec_xl(16, y[ib].qs);

        vector signed ವರ್ಣ q4x0 = vec_and(qxs, ಕೆಳಗಿನ_Mask);
        vector signed ವರ್ಣ q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        vector signed short qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_q4_1_q8_1(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ qk = QK8_1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / qk;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_q4_1 * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_1 * GGML_RESTRICT y = vy;

    ಪೂರ್ಣಾಙ್ಕ ib = 0;
    ಭಿನ್ನ sumf = 0;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector signed ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
    ಪ್ರತಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector ಭಿನ್ನ vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector ಭಿನ್ನ vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].m));
        vector ಭಿನ್ನ vys = {GGML_CPU_FP16_TO_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        vector signed ವರ್ಣ qxs = (vector signed ವರ್ಣ)vec_xl( 0, x[ib].qs);
        vector signed ವರ್ಣ q8y0 = vec_xl( 0, y[ib].qs);
        vector signed ವರ್ಣ q8y1 = vec_xl(16, y[ib].qs);

        vector unsigned ವರ್ಣ q4x0 = (vector unsigned ವರ್ಣ)vec_and(qxs, ಕೆಳಗಿನ_Mask);
        vector unsigned ವರ್ಣ q4x1 = (vector unsigned ವರ್ಣ)vec_sr(qxs, v4);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_mxfp4_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ಧ್ರುವ block_mxfp4 * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_0 * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_MXFP4;

    ಪೂರ್ಣಾಙ್ಕ ib = 0;
    ಭಿನ್ನ sumf = 0;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector unsigned ವರ್ಣ vshift4 = vec_splats((unsigned ವರ್ಣ)4);
    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);

    vector signed ವರ್ಣ kv = vec_xl(0, (ಧ್ರುವ signed ವರ್ಣ *)kvalues_mxfp4);

#pragma GCC unroll 8
    ಪ್ರತಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector ಭಿನ್ನ vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d) *
                                      GGML_E8M0_TO_FP32_HALF(x[ib].e));

        vector signed ವರ್ಣ q8y0 = vec_xl( 0, y[ib].qs);
        vector signed ವರ್ಣ q8y1 = vec_xl(16, y[ib].qs);

        vector signed ವರ್ಣ qxs = (vector signed ವರ್ಣ)vec_xl(0, x[ib].qs);

        vector unsigned ವರ್ಣ lo_nibbles = (vector unsigned ವರ್ಣ)vec_and(qxs, ಕೆಳಗಿನ_Mask);
        vector unsigned ವರ್ಣ hi_nibbles = (vector unsigned ವರ್ಣ)vec_sr(qxs, vshift4);

        vector signed ವರ್ಣ q4x0 = vec_perm(kv, kv, lo_nibbles);
        vector signed ವರ್ಣ q4x1 = vec_perm(kv, kv, hi_nibbles);

        vector signed short qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = vec_splats((int32_t)0);
        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vyd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));
    sumf = vec_extract(vsumf0, 0);
    *s = sumf;
#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_mxfp4_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_q5_0_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ qk = QK8_0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / qk;

    ಪೂರ್ಣಾಙ್ಕ ib = 0;
    ಭಿನ್ನ sumf = 0;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(qk == QK5_0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_q5_0 * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_0 * GGML_RESTRICT y = vy;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)4);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
    ಪ್ರತಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector ಭಿನ್ನ vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed long long aux64x2_0 = {(uint64_t)(table_b2b_1[x[ib].qh[0]]), (uint64_t)(table_b2b_1[x[ib].qh[1]])};
        vector signed long long aux64x2_1 = {(uint64_t)(table_b2b_1[x[ib].qh[2]]), (uint64_t)(table_b2b_1[x[ib].qh[3]])};

        vector signed ವರ್ಣ qh0 = (vector signed ವರ್ಣ)aux64x2_0;
        vector signed ವರ್ಣ qh1 = (vector signed ವರ್ಣ)aux64x2_1;

        vector signed ವರ್ಣ qxs = (vector signed ವರ್ಣ)vec_xl( 0, x[ib].qs);

        vector signed ವರ್ಣ q5x0 = vec_sub(vec_and (qxs, ಕೆಳಗಿನ_Mask), qh0);
        vector signed ವರ್ಣ q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

        vector signed ವರ್ಣ q8y0 = vec_xl(  0, y[ib].qs);
        vector signed ವರ್ಣ q8y1 = vec_xl( 16, y[ib].qs);

        vector signed short qv0 = vec_add(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q5x1, q8y1), vec_mulo(q5x1, q8y1));

        qv0 = vec_add(qv0, qv1);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = vec_add(vec_unpackh(qv0), vec_unpackl(qv0));

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#ಅನ್ಯಥಾ
    UNUSED(ib);
    UNUSED(sumf);
    UNUSED(x);
    UNUSED(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_q5_1_q8_1(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ qk = QK8_1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / qk;

    ಪೂರ್ಣಾಙ್ಕ ib = 0;
    ಭಿನ್ನ sumf = 0;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(qk == QK5_1);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_q5_1 * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_1 * GGML_RESTRICT y = vy;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector signed ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
    ಪ್ರತಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector ಭಿನ್ನ vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector ಭಿನ್ನ vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].m));
        vector ಭಿನ್ನ vys = {GGML_CPU_FP16_TO_FP32(y[ib].s), 0.f, 0.f, 0.f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        vector unsigned long long aux64x2_0 = {(uint64_t)(table_b2b_0[x[ib].qh[0]]), (uint64_t)(table_b2b_0[x[ib].qh[1]])};
        vector unsigned long long aux64x2_1 = {(uint64_t)(table_b2b_0[x[ib].qh[2]]), (uint64_t)(table_b2b_0[x[ib].qh[3]])};

        vector signed ವರ್ಣ qh0 = (vector signed ವರ್ಣ)aux64x2_0;
        vector signed ವರ್ಣ qh1 = (vector signed ವರ್ಣ)aux64x2_1;

        vector signed ವರ್ಣ qxs = (vector signed ವರ್ಣ)vec_xl( 0, x[ib].qs);

        vector unsigned ವರ್ಣ q5x0 = (vector unsigned ವರ್ಣ)vec_or(vec_and(qxs, ಕೆಳಗಿನ_Mask), qh0);
        vector unsigned ವರ್ಣ q5x1 = (vector unsigned ವರ್ಣ)vec_or(vec_sr(qxs, v4), qh1);

        vector signed ವರ್ಣ q8y0 = vec_xl(  0, y[ib].qs);
        vector signed ವರ್ಣ q8y1 = vec_xl( 16, y[ib].qs);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q5x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q5x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

    *s = sumf;
#ಅನ್ಯಥಾ
    UNUSED(nb);
    UNUSED(ib);
    UNUSED(sumf);
    UNUSED(x);
    UNUSED(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_q8_0_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ qk = QK8_0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / qk;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_q8_0 * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_0 * GGML_RESTRICT y = vy;

    ಪೂರ್ಣಾಙ್ಕ ib = 0;
    ಭಿನ್ನ sumf = 0;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
    ಪ್ರತಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector ಭಿನ್ನ vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ವರ್ಣ q8x0 = vec_xl( 0, x[ib].qs);
        vector signed ವರ್ಣ q8x1 = vec_xl(16, x[ib].qs);
        vector signed ವರ್ಣ q8y0 = vec_xl( 0, y[ib].qs);
        vector signed ವರ್ಣ q8y1 = vec_xl(16, y[ib].qs);

        vector signed short qv0 = vec_mule(q8x0, q8y0);
        vector signed short qv1 = vec_mulo(q8x0, q8y0);
        vector signed short qv2 = vec_mule(q8x1, q8y1);
        vector signed short qv3 = vec_mulo(q8x1, q8y1);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;

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
#ಅನ್ಯಥಾ
    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_q2_K * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0x3);
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Scale_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector unsigned ವರ್ಣ v2 = vec_splats((unsigned ವರ್ಣ)0x2);
    ಧ್ರುವ vector unsigned ವರ್ಣ v6 = vec_splats((unsigned ವರ್ಣ)0x6);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector ಭಿನ್ನ vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].dmin));
        vector ಭಿನ್ನ vdmin = vec_mul(vxmin, vyd);

        vector signed short q8ysums0 = vec_xl( 0, y[i].bsums);
        vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

        vector signed ವರ್ಣ q2xmins = (vector signed ವರ್ಣ)vec_xl( 0, x[i].scales);
        vector signed ವರ್ಣ vscales = vec_and(q2xmins, ಕೆಳಗಿನ_Scale_Mask);

        q2xmins = vec_sr(q2xmins, v4);
        vector signed short q2xmins0 = vec_unpackh(q2xmins);
        vector signed short q2xmins1 = vec_unpackl(q2xmins);

        vector signed ಪೂರ್ಣಾಙ್ಕ prod0 = vec_mule(q2xmins0, q8ysums0);
        vector signed ಪೂರ್ಣಾಙ್ಕ prod1 = vec_mulo(q2xmins0, q8ysums0);
        vector signed ಪೂರ್ಣಾಙ್ಕ prod2 = vec_mule(q2xmins1, q8ysums1);
        vector signed ಪೂರ್ಣಾಙ್ಕ prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi4 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi5 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi6 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi7 = v0;

        ಧ್ರುವ uint8_t * GGML_RESTRICT q2 = x[i].qs;
        ಧ್ರುವ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed ವರ್ಣ qxs0 = (vector signed ವರ್ಣ)vec_xl( 0, q2);
            vector signed ವರ್ಣ qxs1 = (vector signed ವರ್ಣ)vec_xl(16, q2);
            q2 += 32;

            vector unsigned ವರ್ಣ q2x00 = (vector unsigned ವರ್ಣ)vec_and(qxs0, ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q2x01 = (vector unsigned ವರ್ಣ)vec_and(vec_sr(qxs0, v2), ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q2x02 = (vector unsigned ವರ್ಣ)vec_and(vec_sr(qxs0, v4), ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q2x03 = (vector unsigned ವರ್ಣ)vec_and(vec_sr(qxs0, v6), ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q2x10 = (vector unsigned ವರ್ಣ)vec_and(qxs1, ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q2x11 = (vector unsigned ವರ್ಣ)vec_and(vec_sr(qxs1, v2), ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q2x12 = (vector unsigned ವರ್ಣ)vec_and(vec_sr(qxs1, v4), ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q2x13 = (vector unsigned ವರ್ಣ)vec_and(vec_sr(qxs1, v6), ಕೆಳಗಿನ_Mask);

            vector signed ವರ್ಣ q8y00 = vec_xl(  0, q8);
            vector signed ವರ್ಣ q8y10 = vec_xl( 16, q8);
            vector signed ವರ್ಣ q8y01 = vec_xl( 32, q8);
            vector signed ವರ್ಣ q8y11 = vec_xl( 48, q8);
            vector signed ವರ್ಣ q8y02 = vec_xl( 64, q8);
            vector signed ವರ್ಣ q8y12 = vec_xl( 80, q8);
            vector signed ವರ್ಣ q8y03 = vec_xl( 96, q8);
            vector signed ವರ್ಣ q8y13 = vec_xl(112, q8);
            q8 += 128;

            vector signed ಪೂರ್ಣಾಙ್ಕ qv0 = vec_msum(q8y00, q2x00, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv1 = vec_msum(q8y01, q2x01, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv2 = vec_msum(q8y02, q2x02, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv3 = vec_msum(q8y03, q2x03, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv4 = vec_msum(q8y10, q2x10, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv5 = vec_msum(q8y11, q2x11, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv6 = vec_msum(q8y12, q2x12, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv7 = vec_msum(q8y13, q2x13, v0);

            vector signed short vscales_07 = vec_unpackh(vscales);
            vector signed ಪೂರ್ಣಾಙ್ಕ vscales_03 = vec_unpackh(vscales_07);
            vector signed ಪೂರ್ಣಾಙ್ಕ vscales_47 = vec_unpackl(vscales_07);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs0 = vec_splat(vscales_03, 0);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs1 = vec_splat(vscales_03, 1);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs2 = vec_splat(vscales_03, 2);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs3 = vec_splat(vscales_03, 3);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs4 = vec_splat(vscales_47, 0);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs5 = vec_splat(vscales_47, 1);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs6 = vec_splat(vscales_47, 2);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs7 = vec_splat(vscales_47, 3);
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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ uint32_t kmask1 = 0x03030303;
    ಧ್ರುವ uint32_t kmask2 = 0x0f0f0f0f;

    ಧ್ರುವ block_q3_K * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0x3);
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask1 = vec_splats((int8_t)0xf);
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask2 = vec_splats((int8_t)0x30);
    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector signed ವರ್ಣ v1 = vec_splats((signed ವರ್ಣ)0x1);
    ಧ್ರುವ vector unsigned ವರ್ಣ v2 = vec_splats((unsigned ವರ್ಣ)0x2);
    ಧ್ರುವ vector unsigned ವರ್ಣ v3 = vec_splats((unsigned ವರ್ಣ)0x3);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);
    ಧ್ರುವ vector unsigned ವರ್ಣ v6 = vec_splats((unsigned ವರ್ಣ)0x6);
    ಧ್ರುವ vector signed ವರ್ಣ off = vec_splats((signed ವರ್ಣ)0x20);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        UNUSED(kmask1);
        UNUSED(kmask2);

        vector signed ವರ್ಣ u0 = (vector signed ವರ್ಣ)vec_xl_len(x[i].scales, 8);
        vector signed ವರ್ಣ u1 = vec_and(u0, ಕೆಳಗಿನ_Mask1);
        vector signed ವರ್ಣ u2 = (vector signed ವರ್ಣ)vec_xl_len(x[i].scales + 8, 4);
        vector signed ವರ್ಣ u3 = (vector signed ವರ್ಣ)vec_mergeh((vector signed ಪೂರ್ಣಾಙ್ಕ)u2, (vector signed ಪೂರ್ಣಾಙ್ಕ)vec_sr(u2, v2));
        vector signed ವರ್ಣ u30 = vec_sl(vec_and(u3, ಕೆಳಗಿನ_Mask), v4);
        vector signed ವರ್ಣ u31 = vec_and(u3, ಕೆಳಗಿನ_Mask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        vector signed ವರ್ಣ vscales = (vector signed ವರ್ಣ)vec_mergeh((vector signed long long)u1, (vector signed long long)u2);
        vector signed ವರ್ಣ qxhs0 = (vector signed ವರ್ಣ)vec_xl( 0, x[i].hmask);
        vector signed ವರ್ಣ qxhs1 = (vector signed ವರ್ಣ)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi4 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi5 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi6 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi7 = v0;

        ಧ್ರುವ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ಧ್ರುವ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed ವರ್ಣ qxs0 = (vector signed ವರ್ಣ)vec_xl( 0, q3);
            vector signed ವರ್ಣ qxs1 = (vector signed ವರ್ಣ)vec_xl(16, q3);
            q3 += 32;

            //the low 2 bits
            vector signed ವರ್ಣ qxs00 = vec_and(qxs0, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs01 = vec_and(vec_sr(qxs0, v2), ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs02 = vec_and(vec_sr(qxs0, v4), ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs03 = vec_and(vec_sr(qxs0, v6), ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs10 = vec_and(qxs1, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs11 = vec_and(vec_sr(qxs1, v2), ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs12 = vec_and(vec_sr(qxs1, v4), ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs13 = vec_and(vec_sr(qxs1, v6), ಕೆಳಗಿನ_Mask);

            //the 3rd bit
            vector signed ವರ್ಣ qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            vector signed ವರ್ಣ qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (vector unsigned ವರ್ಣ)v1)), v2);
            vector signed ವರ್ಣ qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            vector signed ವರ್ಣ qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            vector signed ವರ್ಣ qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            vector signed ವರ್ಣ qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (vector unsigned ವರ್ಣ)v1)), v2);
            vector signed ವರ್ಣ qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            vector signed ವರ್ಣ qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            vector signed ವರ್ಣ q3x00 = vec_sub(qxs00, qxh00);
            vector signed ವರ್ಣ q3x01 = vec_sub(qxs01, qxh01);
            vector signed ವರ್ಣ q3x02 = vec_sub(qxs02, qxh02);
            vector signed ವರ್ಣ q3x03 = vec_sub(qxs03, qxh03);
            vector signed ವರ್ಣ q3x10 = vec_sub(qxs10, qxh10);
            vector signed ವರ್ಣ q3x11 = vec_sub(qxs11, qxh11);
            vector signed ವರ್ಣ q3x12 = vec_sub(qxs12, qxh12);
            vector signed ವರ್ಣ q3x13 = vec_sub(qxs13, qxh13);

            vector signed ವರ್ಣ q8y00 = vec_xl(  0, q8);
            vector signed ವರ್ಣ q8y10 = vec_xl( 16, q8);
            vector signed ವರ್ಣ q8y01 = vec_xl( 32, q8);
            vector signed ವರ್ಣ q8y11 = vec_xl( 48, q8);
            vector signed ವರ್ಣ q8y02 = vec_xl( 64, q8);
            vector signed ವರ್ಣ q8y12 = vec_xl( 80, q8);
            vector signed ವರ್ಣ q8y03 = vec_xl( 96, q8);
            vector signed ವರ್ಣ q8y13 = vec_xl(112, q8);
            q8 += 128;

            vector signed short vscales_h = vec_unpackh(vscales);
            vector signed short vs0 = vec_splat(vscales_h, 0);
            vector signed short vs1 = vec_splat(vscales_h, 1);
            vector signed short vs2 = vec_splat(vscales_h, 2);
            vector signed short vs3 = vec_splat(vscales_h, 3);
            vector signed short vs4 = vec_splat(vscales_h, 4);
            vector signed short vs5 = vec_splat(vscales_h, 5);
            vector signed short vs6 = vec_splat(vscales_h, 6);
            vector signed short vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            vector signed short qv00 = vec_add(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            vector signed short qv01 = vec_add(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            vector signed short qv02 = vec_add(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            vector signed short qv03 = vec_add(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            vector signed short qv10 = vec_add(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            vector signed short qv11 = vec_add(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            vector signed short qv12 = vec_add(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            vector signed short qv13 = vec_add(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

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

#ಅನ್ಯಥಾ
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_q4_K * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

    ಸ್ಥಿರ ಧ್ರುವ uint32_t kmask1 = 0x3f3f3f3f;
    ಸ್ಥಿರ ಧ್ರುವ uint32_t kmask2 = 0x0f0f0f0f;
    ಸ್ಥಿರ ಧ್ರುವ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask1 = vec_splats((int8_t)0x3f);
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask2 = vec_splats((int8_t)0x30);
    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector unsigned ವರ್ಣ v2 = vec_splats((uint8_t)2);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector ಭಿನ್ನ vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].dmin));
        vector ಭಿನ್ನ vdmin = vec_mul(vxmin, vyd);

        vector signed short q8ysums0 = vec_xl( 0, y[i].bsums);
        vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

        UNUSED(kmask1);
        UNUSED(kmask2);
        UNUSED(kmask3);
        UNUSED(utmp);

        vector signed ವರ್ಣ u0 = (vector signed ವರ್ಣ)vec_xl_len(x[i].scales, 8);
        vector signed ವರ್ಣ u1 = vec_and(vec_sr(u0, v2), ಕೆಳಗಿನ_Mask2);
        vector signed ವರ್ಣ u2 = (vector signed ವರ್ಣ)vec_xl_len(x[i].scales + 8, 4);
        vector signed ವರ್ಣ u3 = vec_sr(u2, v4);

        vector signed ವರ್ಣ u30 = u1;
        vector signed ವರ್ಣ u31 = (vector signed ವರ್ಣ)vec_mergeh((vector signed ಪೂರ್ಣಾಙ್ಕ)vec_and(u2, ಕೆಳಗಿನ_Mask), (vector signed ಪೂರ್ಣಾಙ್ಕ)u3);

        u1 = vec_and(u0, ಕೆಳಗಿನ_Mask1);
        u2 = vec_or(u30, u31);

        vector signed ವರ್ಣ utmps = (vector signed ವರ್ಣ)vec_mergeh((vector signed ಪೂರ್ಣಾಙ್ಕ)u1, (vector signed ಪೂರ್ಣಾಙ್ಕ)u2);

        vector signed short vscales = vec_unpackh(utmps);
        vector signed short q4xmins = vec_unpackl(utmps);
        vector signed short q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        vector signed short q4xmins1 = vec_mergel(q4xmins, q4xmins);

        vector signed ಪೂರ್ಣಾಙ್ಕ prod0 = vec_mule(q4xmins0, q8ysums0);
        vector signed ಪೂರ್ಣಾಙ್ಕ prod1 = vec_mule(q4xmins1, q8ysums1);
        vector signed ಪೂರ್ಣಾಙ್ಕ prod2 = vec_mulo(q4xmins0, q8ysums0);
        vector signed ಪೂರ್ಣಾಙ್ಕ prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;

        ಧ್ರುವ uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ಧ್ರುವ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed ವರ್ಣ qxs0 = (vector signed ವರ್ಣ)vec_xl( 0, q4);
            vector signed ವರ್ಣ qxs1 = (vector signed ವರ್ಣ)vec_xl(16, q4);
            vector signed ವರ್ಣ qxs2 = (vector signed ವರ್ಣ)vec_xl(32, q4);
            vector signed ವರ್ಣ qxs3 = (vector signed ವರ್ಣ)vec_xl(48, q4);
            q4 += 64;

            vector unsigned ವರ್ಣ q4x00 = (vector unsigned ವರ್ಣ)vec_and(qxs0, ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q4x01 = (vector unsigned ವರ್ಣ)vec_sr(qxs0, v4);
            vector unsigned ವರ್ಣ q4x10 = (vector unsigned ವರ್ಣ)vec_and(qxs1, ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q4x11 = (vector unsigned ವರ್ಣ)vec_sr(qxs1, v4);
            vector unsigned ವರ್ಣ q4x20 = (vector unsigned ವರ್ಣ)vec_and(qxs2, ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q4x21 = (vector unsigned ವರ್ಣ)vec_sr(qxs2, v4);
            vector unsigned ವರ್ಣ q4x30 = (vector unsigned ವರ್ಣ)vec_and(qxs3, ಕೆಳಗಿನ_Mask);
            vector unsigned ವರ್ಣ q4x31 = (vector unsigned ವರ್ಣ)vec_sr(qxs3, v4);

            vector signed ವರ್ಣ q8y00 = vec_xl(  0, q8);
            vector signed ವರ್ಣ q8y10 = vec_xl( 16, q8);
            vector signed ವರ್ಣ q8y01 = vec_xl( 32, q8);
            vector signed ವರ್ಣ q8y11 = vec_xl( 48, q8);
            vector signed ವರ್ಣ q8y20 = vec_xl( 64, q8);
            vector signed ವರ್ಣ q8y30 = vec_xl( 80, q8);
            vector signed ವರ್ಣ q8y21 = vec_xl( 96, q8);
            vector signed ವರ್ಣ q8y31 = vec_xl(112, q8);
            q8 += 128;

            vector signed ಪೂರ್ಣಾಙ್ಕ qv00 = vec_msum(q8y00, q4x00, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv01 = vec_msum(q8y01, q4x01, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv10 = vec_msum(q8y10, q4x10, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv11 = vec_msum(q8y11, q4x11, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv20 = vec_msum(q8y20, q4x20, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv21 = vec_msum(q8y21, q4x21, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv30 = vec_msum(q8y30, q4x30, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv31 = vec_msum(q8y31, q4x31, v0);

            vector signed ಪೂರ್ಣಾಙ್ಕ vscales_h = vec_unpackh(vscales);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs0 = vec_splat(vscales_h, 0);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs1 = vec_splat(vscales_h, 1);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs2 = vec_splat(vscales_h, 2);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs3 = vec_splat(vscales_h, 3);
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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_q5_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy,  size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_q5_K * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

    ಸ್ಥಿರ ಧ್ರುವ uint32_t kmask1 = 0x3f3f3f3f;
    ಸ್ಥಿರ ಧ್ರುವ uint32_t kmask2 = 0x0f0f0f0f;
    ಸ್ಥಿರ ಧ್ರುವ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask1 = vec_splats((int8_t)0x3f);
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask2 = vec_splats((int8_t)0x30);
    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector unsigned ವರ್ಣ v1 = vec_splats((unsigned ವರ್ಣ)0x1);
    ಧ್ರುವ vector unsigned ವರ್ಣ v2 = vec_splats((unsigned ವರ್ಣ)0x2);
    ಧ್ರುವ vector unsigned ವರ್ಣ v3 = vec_splats((unsigned ವರ್ಣ)0x3);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector ಭಿನ್ನ vxmin = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].dmin));
        vector ಭಿನ್ನ vdmin = vec_mul(vxmin, vyd);

        UNUSED(kmask1);
        UNUSED(kmask2);
        UNUSED(kmask3);
        UNUSED(utmp);

        vector signed ವರ್ಣ u0 = (vector signed ವರ್ಣ)vec_xl_len(x[i].scales, 8);
        vector signed ವರ್ಣ u1 = vec_and(vec_sr(u0, v2), ಕೆಳಗಿನ_Mask2);
        vector signed ವರ್ಣ u2 = (vector signed ವರ್ಣ)vec_xl_len(x[i].scales + 8, 4);
        vector signed ವರ್ಣ u3 = vec_sr(u2, v4);

        vector signed ವರ್ಣ u30 = u1;
        vector signed ವರ್ಣ u31 = (vector signed ವರ್ಣ)vec_mergeh((vector signed ಪೂರ್ಣಾಙ್ಕ)vec_and(u2, ಕೆಳಗಿನ_Mask), (vector signed ಪೂರ್ಣಾಙ್ಕ)u3);

        u1 = vec_and(u0, ಕೆಳಗಿನ_Mask1);
        u2 = vec_or(u30, u31);

        vector signed ವರ್ಣ utmps = (vector signed ವರ್ಣ)vec_mergeh((vector signed ಪೂರ್ಣಾಙ್ಕ)u1, (vector signed ಪೂರ್ಣಾಙ್ಕ)u2);

        vector signed short q8ysums0 = vec_xl( 0, y[i].bsums);
        vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

        vector signed short vscales = vec_unpackh(utmps);

        vector signed short q5xmins = vec_unpackl(utmps);
        vector signed short q5xmins0 = vec_mergeh(q5xmins, q5xmins);
        vector signed short q5xmins1 = vec_mergel(q5xmins, q5xmins);

        vector signed ಪೂರ್ಣಾಙ್ಕ prod0 = vec_mule(q5xmins0, q8ysums0);
        vector signed ಪೂರ್ಣಾಙ್ಕ prod1 = vec_mule(q5xmins1, q8ysums1);
        vector signed ಪೂರ್ಣಾಙ್ಕ prod2 = vec_mulo(q5xmins0, q8ysums0);
        vector signed ಪೂರ್ಣಾಙ್ಕ prod3 = vec_mulo(q5xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector signed ವರ್ಣ qxhs0 = (vector signed ವರ್ಣ)vec_xl( 0, x[i].qh);
        vector signed ವರ್ಣ qxhs1 = (vector signed ವರ್ಣ)vec_xl(16, x[i].qh);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;

        ಧ್ರುವ uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ಧ್ರುವ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q5, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed ವರ್ಣ qxs0 = (vector signed ವರ್ಣ)vec_xl( 0, q5);
            vector signed ವರ್ಣ qxs1 = (vector signed ವರ್ಣ)vec_xl(16, q5);
            q5 += 32;

            vector signed ವರ್ಣ qxs00 = vec_and(qxs0, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs01 = vec_sr(qxs0, v4);
            vector signed ವರ್ಣ qxs10 = vec_and(qxs1, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs11 = vec_sr(qxs1, v4);

            vector signed ವರ್ಣ q5h00 = vec_sl(vec_and((vector signed ವರ್ಣ)v1, qxhs0), v4);
            vector signed ವರ್ಣ q5h01 = vec_sl(vec_and((vector signed ವರ್ಣ)v2, qxhs0), v3);
            vector signed ವರ್ಣ q5h10 = vec_sl(vec_and((vector signed ವರ್ಣ)v1, qxhs1), v4);
            vector signed ವರ್ಣ q5h11 = vec_sl(vec_and((vector signed ವರ್ಣ)v2, qxhs1), v3);
            qxhs0 = vec_sr(qxhs0, v2);
            qxhs1 = vec_sr(qxhs1, v2);

            vector unsigned ವರ್ಣ q5x00 = (vector unsigned ವರ್ಣ)vec_or(q5h00, qxs00);
            vector unsigned ವರ್ಣ q5x01 = (vector unsigned ವರ್ಣ)vec_or(q5h01, qxs01);
            vector unsigned ವರ್ಣ q5x10 = (vector unsigned ವರ್ಣ)vec_or(q5h10, qxs10);
            vector unsigned ವರ್ಣ q5x11 = (vector unsigned ವರ್ಣ)vec_or(q5h11, qxs11);

            vector signed ವರ್ಣ q8y00 = vec_xl( 0, q8);
            vector signed ವರ್ಣ q8y10 = vec_xl(16, q8);
            vector signed ವರ್ಣ q8y01 = vec_xl(32, q8);
            vector signed ವರ್ಣ q8y11 = vec_xl(48, q8);
            q8 += 64;

            vector signed ಪೂರ್ಣಾಙ್ಕ qv00 = vec_msum(q8y00, q5x00, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv01 = vec_msum(q8y01, q5x01, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv10 = vec_msum(q8y10, q5x10, v0);
            vector signed ಪೂರ್ಣಾಙ್ಕ qv11 = vec_msum(q8y11, q5x11, v0);

            vector signed ಪೂರ್ಣಾಙ್ಕ vscales_h = vec_unpackh(vscales);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs0 = vec_splat(vscales_h, 0);
            vector signed ಪೂರ್ಣಾಙ್ಕ vs1 = vec_splat(vscales_h, 1);
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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_q6_K * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector unsigned ವರ್ಣ v2 = vec_splats((unsigned ವರ್ಣ)0x2);
    ಧ್ರುವ vector unsigned ವರ್ಣ v3 = vec_splats((unsigned ವರ್ಣ)0x3);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);
    ಧ್ರುವ vector unsigned ವರ್ಣ v6 = vec_splats((unsigned ವರ್ಣ)0x6);
    ಧ್ರುವ vector signed ವರ್ಣ off = vec_splats((signed ವರ್ಣ)0x20);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi4 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi5 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi6 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi7 = v0;

        ಧ್ರುವ uint8_t * GGML_RESTRICT q6 = x[i].ql;
        ಧ್ರುವ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಧ್ರುವ int8_t  * GGML_RESTRICT qs = x[i].scales;
        ಧ್ರುವ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q6, 0, 0);
            __builtin_prefetch(qh, 0, 0);
            __builtin_prefetch(q8, 0, 0);

            vector signed ವರ್ಣ qxs0 = (vector signed ವರ್ಣ)vec_xl( 0, q6);
            vector signed ವರ್ಣ qxs1 = (vector signed ವರ್ಣ)vec_xl(16, q6);
            vector signed ವರ್ಣ qxs2 = (vector signed ವರ್ಣ)vec_xl(32, q6);
            vector signed ವರ್ಣ qxs3 = (vector signed ವರ್ಣ)vec_xl(48, q6);
            q6 += 64;

            vector signed ವರ್ಣ qxs00 = vec_and(qxs0, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs01 = vec_sr(qxs0, v4);
            vector signed ವರ್ಣ qxs10 = vec_and(qxs1, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs11 = vec_sr(qxs1, v4);
            vector signed ವರ್ಣ qxs20 = vec_and(qxs2, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs21 = vec_sr(qxs2, v4);
            vector signed ವರ್ಣ qxs30 = vec_and(qxs3, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ qxs31 = vec_sr(qxs3, v4);

            vector signed ವರ್ಣ qxhs0 = (vector signed ವರ್ಣ)vec_xl( 0, qh);
            vector signed ವರ್ಣ qxhs1 = (vector signed ವರ್ಣ)vec_xl(16, qh);
            qh += 32;

            vector signed ವರ್ಣ qxh00 = vec_sl(vec_and((vector signed ವರ್ಣ)v3, qxhs0), v4);
            vector signed ವರ್ಣ qxh01 = vec_sl(vec_and((vector signed ವರ್ಣ)v3, vec_sr(qxhs0, v4)), v4);
            vector signed ವರ್ಣ qxh10 = vec_sl(vec_and((vector signed ವರ್ಣ)v3, qxhs1), v4);
            vector signed ವರ್ಣ qxh11 = vec_sl(vec_and((vector signed ವರ್ಣ)v3, vec_sr(qxhs1, v4)), v4);
            vector signed ವರ್ಣ qxh20 = vec_sl(vec_and((vector signed ವರ್ಣ)v3, vec_sr(qxhs0, v2)), v4);
            vector signed ವರ್ಣ qxh21 = vec_sl(vec_and((vector signed ವರ್ಣ)v3, vec_sr(qxhs0, v6)), v4);
            vector signed ವರ್ಣ qxh30 = vec_sl(vec_and((vector signed ವರ್ಣ)v3, vec_sr(qxhs1, v2)), v4);
            vector signed ವರ್ಣ qxh31 = vec_sl(vec_and((vector signed ವರ್ಣ)v3, vec_sr(qxhs1, v6)), v4);

            vector signed ವರ್ಣ q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
            vector signed ವರ್ಣ q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
            vector signed ವರ್ಣ q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
            vector signed ವರ್ಣ q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
            vector signed ವರ್ಣ q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
            vector signed ವರ್ಣ q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
            vector signed ವರ್ಣ q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
            vector signed ವರ್ಣ q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

            vector signed ವರ್ಣ q8y00 = vec_xl(  0, q8);
            vector signed ವರ್ಣ q8y10 = vec_xl( 16, q8);
            vector signed ವರ್ಣ q8y20 = vec_xl( 32, q8);
            vector signed ವರ್ಣ q8y30 = vec_xl( 48, q8);
            vector signed ವರ್ಣ q8y01 = vec_xl( 64, q8);
            vector signed ವರ್ಣ q8y11 = vec_xl( 80, q8);
            vector signed ವರ್ಣ q8y21 = vec_xl( 96, q8);
            vector signed ವರ್ಣ q8y31 = vec_xl(112, q8);
            q8 += 128;

            vector signed short qv00 = vec_add(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
            vector signed short qv10 = vec_add(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
            vector signed short qv20 = vec_add(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
            vector signed short qv30 = vec_add(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
            vector signed short qv01 = vec_add(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
            vector signed short qv11 = vec_add(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
            vector signed short qv21 = vec_add(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
            vector signed short qv31 = vec_add(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

            vector signed short vscales = vec_unpackh(vec_xl_len(qs, 8));
            qs += 8;

            vector signed short vs0 = vec_splat(vscales, 0);
            vector signed short vs1 = vec_splat(vscales, 1);
            vector signed short vs2 = vec_splat(vscales, 2);
            vector signed short vs3 = vec_splat(vscales, 3);
            vector signed short vs4 = vec_splat(vscales, 4);
            vector signed short vs5 = vec_splat(vscales, 5);
            vector signed short vs6 = vec_splat(vscales, 6);
            vector signed short vs7 = vec_splat(vscales, 7);

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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#ಯದಿ defined (__POWER9_VECTOR__)
ಸ್ಥಿರ ಧ್ರುವ int8_t keven_signs_q2xs[1024] = {
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
#endif

ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_iq2_xxs * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K    * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಧ್ರುವ uint64_t * signs64 = (ಧ್ರುವ uint64_t *)keven_signs_q2xs;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;

        ಧ್ರುವ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ಧ್ರುವ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            ಧ್ರುವ uint8_t * aux8 = (ಧ್ರುವ uint8_t *)aux32;

            memcpy(aux32, q2, 4*ಪರಿಮಾಣ(uint32_t));
            q2 += 8;

            vector signed long long aux64x2_0 = {*(ಧ್ರುವ int64_t *)(iq2xxs_grid + aux8[ 0]), *(ಧ್ರುವ int64_t *)(iq2xxs_grid + aux8[ 1])};
            vector signed long long aux64x2_1 = {*(ಧ್ರುವ int64_t *)(iq2xxs_grid + aux8[ 2]), *(ಧ್ರುವ int64_t *)(iq2xxs_grid + aux8[ 3])};
            vector signed long long aux64x2_2 = {*(ಧ್ರುವ int64_t *)(iq2xxs_grid + aux8[ 8]), *(ಧ್ರುವ int64_t *)(iq2xxs_grid + aux8[ 9])};
            vector signed long long aux64x2_3 = {*(ಧ್ರುವ int64_t *)(iq2xxs_grid + aux8[10]), *(ಧ್ರುವ int64_t *)(iq2xxs_grid + aux8[11])};

            vector signed long long vsigns0 = {*(ಧ್ರುವ int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(ಧ್ರುವ int64_t *)(signs64 + ((aux32[1] >>  7) & 127))};
            vector signed long long vsigns1 = {*(ಧ್ರುವ int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(ಧ್ರುವ int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            vector signed long long vsigns2 = {*(ಧ್ರುವ int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(ಧ್ರುವ int64_t *)(signs64 + ((aux32[3] >>  7) & 127))};
            vector signed long long vsigns3 = {*(ಧ್ರುವ int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(ಧ್ರುವ int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            vector signed ವರ್ಣ q2x0 = (vector signed ವರ್ಣ)vec_mul((vector signed ವರ್ಣ)vsigns0, (vector signed ವರ್ಣ)aux64x2_0);
            vector signed ವರ್ಣ q2x1 = (vector signed ವರ್ಣ)vec_mul((vector signed ವರ್ಣ)vsigns1, (vector signed ವರ್ಣ)aux64x2_1);
            vector signed ವರ್ಣ q2x2 = (vector signed ವರ್ಣ)vec_mul((vector signed ವರ್ಣ)vsigns2, (vector signed ವರ್ಣ)aux64x2_2);
            vector signed ವರ್ಣ q2x3 = (vector signed ವರ್ಣ)vec_mul((vector signed ವರ್ಣ)vsigns3, (vector signed ವರ್ಣ)aux64x2_3);

            vector signed ವರ್ಣ q8y0 = vec_xl( 0, q8);
            vector signed ವರ್ಣ q8y1 = vec_xl(16, q8);
            vector signed ವರ್ಣ q8y2 = vec_xl(32, q8);
            vector signed ವರ್ಣ q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ಧ್ರುವ uint16_t ls0 = aux32[1] >> 28;
            ಧ್ರುವ uint16_t ls1 = aux32[3] >> 28;

            vector signed short vscales01 = vec_splats((int16_t)(2*ls0+1));
            vector signed short vscales23 = vec_splats((int16_t)(2*ls1+1));

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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_iq2_xs * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K   * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಧ್ರುವ uint64_t * signs64 = (ಧ್ರುವ uint64_t *)keven_signs_q2xs;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;

        ಧ್ರುವ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ಧ್ರುವ uint8_t  * GGML_RESTRICT sc = x[i].scales;
        ಧ್ರುವ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed long long aux64x2_0 = {*(ಧ್ರುವ int64_t *)(iq2xs_grid + (q2[0] & 511)), *(ಧ್ರುವ int64_t *)(iq2xs_grid + (q2[1] & 511))};
            vector signed long long aux64x2_1 = {*(ಧ್ರುವ int64_t *)(iq2xs_grid + (q2[2] & 511)), *(ಧ್ರುವ int64_t *)(iq2xs_grid + (q2[3] & 511))};
            vector signed long long aux64x2_2 = {*(ಧ್ರುವ int64_t *)(iq2xs_grid + (q2[4] & 511)), *(ಧ್ರುವ int64_t *)(iq2xs_grid + (q2[5] & 511))};
            vector signed long long aux64x2_3 = {*(ಧ್ರುವ int64_t *)(iq2xs_grid + (q2[6] & 511)), *(ಧ್ರುವ int64_t *)(iq2xs_grid + (q2[7] & 511))};

            vector signed long long vsigns0 = {*(ಧ್ರುವ int64_t *)(signs64 + ((q2[0] >> 9))), *(ಧ್ರುವ int64_t *)(signs64 + ((q2[1] >> 9)))};
            vector signed long long vsigns1 = {*(ಧ್ರುವ int64_t *)(signs64 + ((q2[2] >> 9))), *(ಧ್ರುವ int64_t *)(signs64 + ((q2[3] >> 9)))};
            vector signed long long vsigns2 = {*(ಧ್ರುವ int64_t *)(signs64 + ((q2[4] >> 9))), *(ಧ್ರುವ int64_t *)(signs64 + ((q2[5] >> 9)))};
            vector signed long long vsigns3 = {*(ಧ್ರುವ int64_t *)(signs64 + ((q2[6] >> 9))), *(ಧ್ರುವ int64_t *)(signs64 + ((q2[7] >> 9)))};
            q2 += 8;

            vector signed ವರ್ಣ q2x0 = (vector signed ವರ್ಣ)vec_mul((vector signed ವರ್ಣ)vsigns0, (vector signed ವರ್ಣ)aux64x2_0);
            vector signed ವರ್ಣ q2x1 = (vector signed ವರ್ಣ)vec_mul((vector signed ವರ್ಣ)vsigns1, (vector signed ವರ್ಣ)aux64x2_1);
            vector signed ವರ್ಣ q2x2 = (vector signed ವರ್ಣ)vec_mul((vector signed ವರ್ಣ)vsigns2, (vector signed ವರ್ಣ)aux64x2_2);
            vector signed ವರ್ಣ q2x3 = (vector signed ವರ್ಣ)vec_mul((vector signed ವರ್ಣ)vsigns3, (vector signed ವರ್ಣ)aux64x2_3);

            vector signed ವರ್ಣ q8y0 = vec_xl( 0, q8);
            vector signed ವರ್ಣ q8y1 = vec_xl(16, q8);
            vector signed ವರ್ಣ q8y2 = vec_xl(32, q8);
            vector signed ವರ್ಣ q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ಧ್ರುವ uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ಧ್ರುವ uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ಧ್ರುವ uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ಧ್ರುವ uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            vector signed short vscales0 = vec_splats((int16_t)(2*ls0+1));
            vector signed short vscales1 = vec_splats((int16_t)(2*ls1+1));
            vector signed short vscales2 = vec_splats((int16_t)(2*ls2+1));
            vector signed short vscales3 = vec_splats((int16_t)(2*ls3+1));

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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_iq2_s * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K  * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರ ಧ್ರುವ uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ಸ್ಥಿರ ಧ್ರುವ uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಧ್ರುವ vector unsigned ವರ್ಣ mask0 = vec_xl( 0, k_mask1);
    ಧ್ರುವ vector unsigned ವರ್ಣ mask1 = vec_xl(16, k_mask1);
    ಧ್ರುವ vector signed ವರ್ಣ mask2 = (vector signed ವರ್ಣ)vec_xl( 0, k_mask2);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;

        ಧ್ರುವ uint8_t *  GGML_RESTRICT q2 = x[i].qs;
        ಧ್ರುವ uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ಧ್ರುವ uint16_t * GGML_RESTRICT signs = (ಧ್ರುವ uint16_t *)(x[i].qs + QK_K/8);
        ಧ್ರುವ uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ಧ್ರುವ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed long long aux64x2_0 = {*(ಧ್ರುವ int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(ಧ್ರುವ int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            vector signed long long aux64x2_1 = {*(ಧ್ರುವ int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(ಧ್ರುವ int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            vector signed long long aux64x2_2 = {*(ಧ್ರುವ int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(ಧ್ರುವ int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            vector signed long long aux64x2_3 = {*(ಧ್ರುವ int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(ಧ್ರುವ int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            vector signed ವರ್ಣ vsigns01 = (vector signed ವರ್ಣ)vec_splats(*(ಧ್ರುವ uint32_t *)&signs[0]);
            vector signed ವರ್ಣ vsigns23 = (vector signed ವರ್ಣ)vec_splats(*(ಧ್ರುವ uint32_t *)&signs[2]);
            signs += 4;

            vector signed ವರ್ಣ vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            vector signed ವರ್ಣ vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            vector signed ವರ್ಣ vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            vector signed ವರ್ಣ vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (vector signed ವರ್ಣ)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (vector signed ವರ್ಣ)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (vector signed ವರ್ಣ)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (vector signed ವರ್ಣ)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            vector signed ವರ್ಣ q2x0 = vec_sub(vec_xor(vsigns0, (vector signed ವರ್ಣ)aux64x2_0), vsigns0);
            vector signed ವರ್ಣ q2x1 = vec_sub(vec_xor(vsigns1, (vector signed ವರ್ಣ)aux64x2_1), vsigns1);
            vector signed ವರ್ಣ q2x2 = vec_sub(vec_xor(vsigns2, (vector signed ವರ್ಣ)aux64x2_2), vsigns2);
            vector signed ವರ್ಣ q2x3 = vec_sub(vec_xor(vsigns3, (vector signed ವರ್ಣ)aux64x2_3), vsigns3);

            vector signed ವರ್ಣ q8y0 = vec_xl( 0, q8);
            vector signed ವರ್ಣ q8y1 = vec_xl(16, q8);
            vector signed ವರ್ಣ q8y2 = vec_xl(32, q8);
            vector signed ವರ್ಣ q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ಧ್ರುವ uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ಧ್ರುವ uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ಧ್ರುವ uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ಧ್ರುವ uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            vector signed short vscales0 = vec_splats((int16_t)(2*ls0+1));
            vector signed short vscales1 = vec_splats((int16_t)(2*ls1+1));
            vector signed short vscales2 = vec_splats((int16_t)(2*ls2+1));
            vector signed short vscales3 = vec_splats((int16_t)(2*ls3+1));

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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_iq3_xxs * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K    * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ uint64_t * signs64 = (ಧ್ರುವ uint64_t *)keven_signs_q2xs;

    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;

        ಧ್ರುವ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ಧ್ರುವ uint32_t * GGML_RESTRICT signs = (ಧ್ರುವ uint32_t *)(x[i].qs + QK_K/4);
        ಧ್ರುವ int8_t  * GGML_RESTRICT q8 = y[i].qs;

#pragma GCC unroll 1
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector unsigned ಪೂರ್ಣಾಙ್ಕ aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            vector unsigned ಪೂರ್ಣಾಙ್ಕ aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            vector unsigned ಪೂರ್ಣಾಙ್ಕ aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            vector unsigned ಪೂರ್ಣಾಙ್ಕ aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            vector unsigned long long aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            vector unsigned long long aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            vector unsigned long long aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            vector unsigned long long aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            vector signed ವರ್ಣ q3x0 = vec_mul((vector signed ವರ್ಣ)aux64x2_0, (vector signed ವರ್ಣ)aux32x4_0);
            vector signed ವರ್ಣ q3x1 = vec_mul((vector signed ವರ್ಣ)aux64x2_1, (vector signed ವರ್ಣ)aux32x4_1);
            vector signed ವರ್ಣ q3x2 = vec_mul((vector signed ವರ್ಣ)aux64x2_2, (vector signed ವರ್ಣ)aux32x4_2);
            vector signed ವರ್ಣ q3x3 = vec_mul((vector signed ವರ್ಣ)aux64x2_3, (vector signed ವರ್ಣ)aux32x4_3);

            vector signed ವರ್ಣ q8y0 = vec_xl( 0, q8);
            vector signed ವರ್ಣ q8y1 = vec_xl(16, q8);
            vector signed ವರ್ಣ q8y2 = vec_xl(32, q8);
            vector signed ವರ್ಣ q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ಧ್ರುವ uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            ಧ್ರುವ uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            vector signed short vscales01 = (vector signed short)vec_splats((uint16_t)(2*ls0+1));
            vector signed short vscales23 = (vector signed short)vec_splats((uint16_t)(2*ls1+1));

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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_iq3_s * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K  * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರ ಧ್ರುವ uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ಸ್ಥಿರ ಧ್ರುವ uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಧ್ರುವ vector unsigned ವರ್ಣ mask0 = vec_xl( 0, k_mask1);
    ಧ್ರುವ vector unsigned ವರ್ಣ mask1 = vec_xl(16, k_mask1);
    ಧ್ರುವ vector signed ವರ್ಣ mask2 = (vector signed ವರ್ಣ)vec_xl( 0, k_mask2);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        ಧ್ರುವ uint8_t *  GGML_RESTRICT q3 = x[i].qs;
        ಧ್ರುವ uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ಧ್ರುವ uint16_t * GGML_RESTRICT signs = (ಧ್ರುವ uint16_t *)(x[i].signs);
        ಧ್ರುವ uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ಧ್ರುವ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector unsigned ಪೂರ್ಣಾಙ್ಕ aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            vector unsigned ಪೂರ್ಣಾಙ್ಕ aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            vector unsigned ಪೂರ್ಣಾಙ್ಕ aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            vector unsigned ಪೂರ್ಣಾಙ್ಕ aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            vector signed ವರ್ಣ vsigns01 = (vector signed ವರ್ಣ)vec_splats(*(ಧ್ರುವ uint32_t *)&signs[0]);
            vector signed ವರ್ಣ vsigns02 = (vector signed ವರ್ಣ)vec_splats(*(ಧ್ರುವ uint32_t *)&signs[2]);
            signs += 4;

            vector signed ವರ್ಣ vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            vector signed ವರ್ಣ vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            vector signed ವರ್ಣ vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            vector signed ವರ್ಣ vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (vector signed ವರ್ಣ)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (vector signed ವರ್ಣ)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (vector signed ವರ್ಣ)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (vector signed ವರ್ಣ)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            vector signed ವರ್ಣ q3x0 = vec_sub(vec_xor(vsigns0, (vector signed ವರ್ಣ)aux32x4_0), vsigns0);
            vector signed ವರ್ಣ q3x1 = vec_sub(vec_xor(vsigns1, (vector signed ವರ್ಣ)aux32x4_1), vsigns1);
            vector signed ವರ್ಣ q3x2 = vec_sub(vec_xor(vsigns2, (vector signed ವರ್ಣ)aux32x4_2), vsigns2);
            vector signed ವರ್ಣ q3x3 = vec_sub(vec_xor(vsigns3, (vector signed ವರ್ಣ)aux32x4_3), vsigns3);

            vector signed ವರ್ಣ q8y0 = vec_xl( 0, q8);
            vector signed ವರ್ಣ q8y1 = vec_xl(16, q8);
            vector signed ವರ್ಣ q8y2 = vec_xl(32, q8);
            vector signed ವರ್ಣ q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ಧ್ರುವ uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ಧ್ರುವ uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            vector signed short vscales01 = (vector signed short)vec_splats((uint16_t)(2*ls0+1));
            vector signed short vscales23 = (vector signed short)vec_splats((uint16_t)(2*ls1+1));

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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ಧ್ರುವ block_iq1_s * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K  * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector unsigned ವರ್ಣ v0 = vec_splats((unsigned ವರ್ಣ)0x0);
    ಧ್ರುವ vector unsigned short vsign = vec_splats((unsigned short)0x8000);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nb; ++i) {
        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[i].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[i].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = vec_splats((int32_t)0);
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = vec_splats((int32_t)0);
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = vec_splats((int32_t)0);
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = vec_splats((int32_t)0);
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi8 = vec_splats((int32_t)0);

        ಧ್ರುವ uint8_t  * GGML_RESTRICT q1 = x[i].qs;
        ಧ್ರುವ uint16_t * GGML_RESTRICT qh = x[i].qh;
        ಧ್ರುವ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        ಧ್ರುವ int16_t  * GGML_RESTRICT qs = y[i].bsums;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed long long aux64x2_0 = {*(ಧ್ರುವ int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(ಧ್ರುವ int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            vector signed long long aux64x2_1 = {*(ಧ್ರುವ int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(ಧ್ರುವ int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            vector signed long long aux64x2_2 = {*(ಧ್ರುವ int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(ಧ್ರುವ int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            vector signed long long aux64x2_3 = {*(ಧ್ರುವ int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(ಧ್ರುವ int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            vector signed ವರ್ಣ q1x0 = (vector signed ವರ್ಣ)aux64x2_0;
            vector signed ವರ್ಣ q1x1 = (vector signed ವರ್ಣ)aux64x2_1;
            vector signed ವರ್ಣ q1x2 = (vector signed ವರ್ಣ)aux64x2_2;
            vector signed ವರ್ಣ q1x3 = (vector signed ವರ್ಣ)aux64x2_3;

            vector signed ವರ್ಣ q8y0 = vec_xl( 0, q8);
            vector signed ವರ್ಣ q8y1 = vec_xl(16, q8);
            vector signed ವರ್ಣ q8y2 = vec_xl(32, q8);
            vector signed ವರ್ಣ q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            ಧ್ರುವ uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            ಧ್ರುವ uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            vector signed short vscales01 = (vector signed short)vec_splats((uint16_t)(2*ls0+1));
            vector signed short vscales23 = (vector signed short)vec_splats((uint16_t)(2*ls1+1));
            vector signed short vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            vector signed short q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (vector signed short)v0);

            vector signed short qxh = (vector signed short)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            vector __bool short vsel = vec_cmpge(qxh, (vector signed short)v0);

            vector signed short q8ysum = vec_sel((vector signed short)vec_xor((vector unsigned short)q8ysums, vsign), q8ysums, vsel);

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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_iq4_nl_q8_0(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ಧ್ರುವ block_iq4_nl * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_0   * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK4_NL;

    ಪೂರ್ಣಾಙ್ಕ ib = 0;
    ಭಿನ್ನ sumf = 0;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector signed ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);

    ಧ್ರುವ vector signed ವರ್ಣ ಮೌಲ್ಯಗಳು = vec_xl( 0, kvalues_iq4nl);

#pragma GCC unroll 4
    ಪ್ರತಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d));
        vector ಭಿನ್ನ vyd = vec_splats(GGML_CPU_FP16_TO_FP32(y[ib].d));
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ವರ್ಣ qxs = (vector signed ವರ್ಣ)vec_xl( 0, x[ib].qs);
        vector signed ವರ್ಣ q4x0 = vec_and(qxs, ಕೆಳಗಿನ_Mask);
        vector signed ವರ್ಣ q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (vector unsigned ವರ್ಣ)q4x0);
        q4x1 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (vector unsigned ವರ್ಣ)q4x1);

        vector signed ವರ್ಣ q8y0 = vec_xl( 0, y[ib].qs);
        vector signed ವರ್ಣ q8y1 = vec_xl(16, y[ib].qs);

        vector signed short qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;

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
#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K(ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ * GGML_RESTRICT s, size_t bs, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಧ್ರುವ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಙ್ಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);

    ಧ್ರುವ block_iq4_xs * GGML_RESTRICT x = vx;
    ಧ್ರುವ block_q8_K   * GGML_RESTRICT y = vy;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb = n / QK_K;

#ಯದಿ defined(__POWER9_VECTOR__)
    ಧ್ರುವ vector signed ವರ್ಣ ಕೆಳಗಿನ_Mask = vec_splats((signed ವರ್ಣ)0xF);
    ಧ್ರುವ vector ಪೂರ್ಣಾಙ್ಕ v0 = vec_splats((int32_t)0);
    ಧ್ರುವ vector unsigned ವರ್ಣ v4 = vec_splats((unsigned ವರ್ಣ)0x4);

    vector ಭಿನ್ನ vsumf0 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf1 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf2 = vec_splats(0.0f);
    vector ಭಿನ್ನ vsumf3 = vec_splats(0.0f);

    ಧ್ರುವ vector signed ವರ್ಣ ಮೌಲ್ಯಗಳು = vec_xl( 0, kvalues_iq4nl);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ibl = 0; ibl < nb; ++ibl) {

        vector ಭಿನ್ನ vxd = vec_splats(GGML_CPU_FP16_TO_FP32(x[ibl].d));
        vector ಭಿನ್ನ vyd = vec_splats(y[ibl].d);
        vector ಭಿನ್ನ vd = vec_mul(vxd, vyd);

        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi0 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi1 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi2 = v0;
        vector signed ಪೂರ್ಣಾಙ್ಕ vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        ಧ್ರುವ uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        ಧ್ರುವ uint8_t * GGML_RESTRICT sc = x[ibl].scales_l;
        ಧ್ರುವ int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed ವರ್ಣ qxs0 = (vector signed ವರ್ಣ)vec_xl( 0, q4);
            vector signed ವರ್ಣ qxs1 = (vector signed ವರ್ಣ)vec_xl(16, q4);
            q4 += 32;

            vector signed ವರ್ಣ q4x00 = (vector signed ವರ್ಣ)vec_and(qxs0, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ q4x01 = (vector signed ವರ್ಣ)vec_sr(qxs0, v4);
            vector signed ವರ್ಣ q4x10 = (vector signed ವರ್ಣ)vec_and(qxs1, ಕೆಳಗಿನ_Mask);
            vector signed ವರ್ಣ q4x11 = (vector signed ವರ್ಣ)vec_sr(qxs1, v4);

            q4x00 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (vector unsigned ವರ್ಣ)q4x00);
            q4x01 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (vector unsigned ವರ್ಣ)q4x01);
            q4x10 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (vector unsigned ವರ್ಣ)q4x10);
            q4x11 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (vector unsigned ವರ್ಣ)q4x11);

            vector signed ವರ್ಣ q8y0 = vec_xl( 0, q8);
            vector signed ವರ್ಣ q8y1 = vec_xl(16, q8);
            vector signed ವರ್ಣ q8y2 = vec_xl(32, q8);
            vector signed ವರ್ಣ q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            ಧ್ರುವ uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            ಧ್ರುವ uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            vector signed short vscales01 = vec_splats((int16_t)ls0);
            vector signed short vscales23 = vec_splats((int16_t)ls1);

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

#ಅನ್ಯಥಾ
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
