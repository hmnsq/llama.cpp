#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_IMPL_C
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

#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS 1e-15f
#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS_IQ3_XXS 1e-8f
#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS_IQ2_S 1e-8f
#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS_IQ1_M 1e-7f
#ವ್ಯಾಖ್ಯಾನಿಸು ಗುಂಪು_MAX_EPS_IQ1_S 1e-12f

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಳಸದ GGML_ಬಳಸದ

#ಆದರೆ defined(__POWER9_VECTOR__)
#ವ್ಯಾಖ್ಯಾನಿಸು B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#ವ್ಯಾಖ್ಯಾನಿಸು B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#ವ್ಯಾಖ್ಯಾನಿಸು B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#ವ್ಯಾಖ್ಯಾನಿಸು B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#ವ್ಯಾಖ್ಯಾನಿಸು B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#ವ್ಯಾಖ್ಯಾನಿಸು B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#ವ್ಯಾಖ್ಯಾನಿಸು B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#ವ್ಯಾಖ್ಯಾನಿಸು B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint64_t ಕೋಷ್ಟಕ_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint64_t ಕೋಷ್ಟಕ_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_0(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k) {
    ಪ್ರತಿಜ್ಞಾ(QK8_0 == 32);
    ಪ್ರತಿಜ್ಞಾ(k % QK8_0 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / QK8_0;

    ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        ಸದಿಶ ದಶಮಾಂಶ srcv [8];
        ಸದಿಶ ದಶಮಾಂಶ asrcv[8];
        ಸದಿಶ ದಶಮಾಂಶ amaxv[8];
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vi[8];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ amax = MAX(MAX(vec_ಹೊರತೆಗೆ(amaxv[0], 0),
                                   vec_ಹೊರತೆಗೆ(amaxv[0], 1)),
                               MAX(vec_ಹೊರತೆಗೆ(amaxv[0], 2),
                                   vec_ಹೊರತೆಗೆ(amaxv[0], 3)));

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax / ((1 << 7) - 1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;
        ಸ್ಥಿರಾಂಕ ಸದಿಶ ದಶಮಾಂಶ vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_ಗಮ್ಯ_FP16(d);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
            ಸ್ಥಿರಾಂಕ ಸದಿಶ ದಶಮಾಂಶ v  = vec_ಸುತ್ತು(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(nb);
    // scalar
    quantize_ಅಡ್ಡಸಾಲು_q8_0_ref(x, y, k);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * GGML_RESTRICT x, ಶೂನ್ಯ * GGML_RESTRICT vy, int64_t k) {
    ಪ್ರತಿಜ್ಞಾ(k % QK8_1 == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = k / QK8_1;

    ಖಂಡ_q8_1 * GGML_RESTRICT y = vy;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; i++) {
        ಸದಿಶ ದಶಮಾಂಶ srcv [8];
        ಸದಿಶ ದಶಮಾಂಶ asrcv[8];
        ಸದಿಶ ದಶಮಾಂಶ amaxv[8];
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vi[8];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ amax = MAX(MAX(vec_ಹೊರತೆಗೆ(amaxv[0], 0),
                                   vec_ಹೊರತೆಗೆ(amaxv[0], 1)),
                               MAX(vec_ಹೊರತೆಗೆ(amaxv[0], 2),
                                   vec_ಹೊರತೆಗೆ(amaxv[0], 3)));

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = amax / ((1 << 7) - 1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id = d ? 1.0f/d : 0.0f;
        ಸ್ಥಿರಾಂಕ ಸದಿಶ ದಶಮಾಂಶ vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_ಗಮ್ಯ_FP16(d);

        ಸದಿಶ ಪೂರ್ಣಾಂಕ accv = vec_splats(0);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
            ಸ್ಥಿರಾಂಕ ಸದಿಶ ದಶಮಾಂಶ v  = vec_ಸುತ್ತು(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_ಕೂಡಿಸು(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_ಕೂಡಿಸು(accv, vec_sld(accv, accv, 4));
        accv = vec_ಕೂಡಿಸು(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_CPU_FP32_ಗಮ್ಯ_FP16(d * vec_ಹೊರತೆಗೆ(accv, 0));
    }

#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(nb);
    // scalar
    quantize_ಅಡ್ಡಸಾಲು_q8_1_ref(x, y, k);
#ಆದರೆ_ಅಂತ್ಯ
}


//===================================== Dot products =================================

ಶೂನ್ಯ ggml_vec_dot_q4_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ v8 = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0x8);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);

#ಸೂಚನೆ GCC unroll 8
    ಗಾಗಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, x[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, y[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, y[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x0 = vec_and(qxs, ಕೆಳಗಿನ_ಮುಖವಾಡ);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ಹೊರತೆಗೆ(vsumf0, 0);

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(ib);
    ಬಳಸದ(sumf);
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q4_1_q8_1(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * GGML_RESTRICT y = vy;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);

#ಸೂಚನೆ GCC unroll 4
    ಗಾಗಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ದಶಮಾಂಶ vxmin = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].m));
        ಸದಿಶ ದಶಮಾಂಶ vys = {GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, x[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, y[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, y[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x0 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(qxs, ಕೆಳಗಿನ_ಮುಖವಾಡ);
        ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x1 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_sr(qxs, v4);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ಹೊರತೆಗೆ(vsumf0, 0);

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(ib);
    ಬಳಸದ(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_mxfp4_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_MXFP4;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ vshift4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)4);
    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);

    ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ kv = vec_xl(0, (ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ *)kvalues_mxfp4);

#ಸೂಚನೆ GCC unroll 8
    ಗಾಗಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d) *
                                      GGML_E8_M0_ಗಮ್ಯ_FP32_HALF(x[ib].e));

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, y[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, y[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(0, x[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ lo_nibbles = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(qxs, ಕೆಳಗಿನ_ಮುಖವಾಡ);
        ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ hi_nibbles = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_sr(qxs, vshift4);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x0 = vec_perm(kv, kv, lo_nibbles);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x1 = vec_perm(kv, kv, hi_nibbles);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = vec_splats((int32_t)0);
        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vyd, vsumf0);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));
    sumf = vec_ಹೊರತೆಗೆ(vsumf0, 0);
    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(ib);
    ಬಳಸದ(sumf);
    ggml_vec_dot_mxfp4_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q5_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(qk == QK5_0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)4);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);

#ಸೂಚನೆ GCC unroll 4
    ಗಾಗಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_0 = {(uint64_t)(ಕೋಷ್ಟಕ_b2b_1[x[ib].qh[0]]), (uint64_t)(ಕೋಷ್ಟಕ_b2b_1[x[ib].qh[1]])};
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_1 = {(uint64_t)(ಕೋಷ್ಟಕ_b2b_1[x[ib].qh[2]]), (uint64_t)(ಕೋಷ್ಟಕ_b2b_1[x[ib].qh[3]])};

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qh0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qh1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_1;

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, x[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q5x0 = vec_sub(vec_and (qxs, ಕೆಳಗಿನ_ಮುಖವಾಡ), qh0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl(  0, y[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl( 16, y[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q5x1, q8y1), vec_mulo(q5x1, q8y1));

        qv0 = vec_ಕೂಡಿಸು(qv0, qv1);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = vec_ಕೂಡಿಸು(vec_unpackh(qv0), vec_unpackl(qv0));

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ಹೊರತೆಗೆ(vsumf0, 0);

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(ib);
    ಬಳಸದ(sumf);
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q5_1_q8_1(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(qk == QK5_1);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * GGML_RESTRICT y = vy;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);

#ಸೂಚನೆ GCC unroll 4
    ಗಾಗಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ದಶಮಾಂಶ vxmin = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].m));
        ಸದಿಶ ದಶಮಾಂಶ vys = {GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].s), 0.f, 0.f, 0.f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ aux64x2_0 = {(uint64_t)(ಕೋಷ್ಟಕ_b2b_0[x[ib].qh[0]]), (uint64_t)(ಕೋಷ್ಟಕ_b2b_0[x[ib].qh[1]])};
        ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ aux64x2_1 = {(uint64_t)(ಕೋಷ್ಟಕ_b2b_0[x[ib].qh[2]]), (uint64_t)(ಕೋಷ್ಟಕ_b2b_0[x[ib].qh[3]])};

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qh0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qh1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_1;

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, x[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q5x0 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_or(vec_and(qxs, ಕೆಳಗಿನ_ಮುಖವಾಡ), qh0);
        ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q5x1 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_or(vec_sr(qxs, v4), qh1);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl(  0, y[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl( 16, y[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q5x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q5x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ಹೊರತೆಗೆ(vsumf0, 0);

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(nb);
    ಬಳಸದ(ib);
    ಬಳಸದ(sumf);
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q8_0_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qk = QK8_0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / qk;

    ಪ್ರತಿಜ್ಞಾ(n % qk == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * GGML_RESTRICT y = vy;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);

#ಸೂಚನೆ GCC unroll 8
    ಗಾಗಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8x0 = vec_xl( 0, x[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8x1 = vec_xl(16, x[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, y[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, y[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_mule(q8x0, q8y0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_mulo(q8x0, q8y0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv2 = vec_mule(q8x1, q8y1);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv3 = vec_mulo(q8x1, q8y1);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_ಕೂಡಿಸು(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ಹೊರತೆಗೆ(vsumf0, 0);

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(nb);
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(ib);
    ಬಳಸದ(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q2_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0x3);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_Scale_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v2 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x2);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v6 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x6);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ದಶಮಾಂಶ vxmin = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin));
        ಸದಿಶ ದಶಮಾಂಶ vdmin = vec_mul(vxmin, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q8ysums0 = vec_xl( 0, y[i].bsums);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q8ysums1 = vec_xl(16, y[i].bsums);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2xmins = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, x[i].scales);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vscales = vec_and(q2xmins, ಕೆಳಗಿನ_Scale_ಮುಖವಾಡ);

        q2xmins = vec_sr(q2xmins, v4);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q2xmins0 = vec_unpackh(q2xmins);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q2xmins1 = vec_unpackl(q2xmins);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod0 = vec_mule(q2xmins0, q8ysums0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod1 = vec_mulo(q2xmins0, q8ysums0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod2 = vec_mule(q2xmins1, q8ysums1);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi4 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi5 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi6 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi7 = v0;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q2 = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, q2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(16, q2);
            q2 += 32;

            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q2x00 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(qxs0, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q2x01 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(vec_sr(qxs0, v2), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q2x02 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(vec_sr(qxs0, v4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q2x03 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(vec_sr(qxs0, v6), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q2x10 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(qxs1, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q2x11 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(vec_sr(qxs1, v2), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q2x12 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(vec_sr(qxs1, v4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q2x13 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(vec_sr(qxs1, v6), ಕೆಳಗಿನ_ಮುಖವಾಡ);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y00 = vec_xl(  0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y10 = vec_xl( 16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y01 = vec_xl( 32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y11 = vec_xl( 48, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y02 = vec_xl( 64, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y12 = vec_xl( 80, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y03 = vec_xl( 96, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y13 = vec_xl(112, q8);
            q8 += 128;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv0 = vec_msum(q8y00, q2x00, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv1 = vec_msum(q8y01, q2x01, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv2 = vec_msum(q8y02, q2x02, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv3 = vec_msum(q8y03, q2x03, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv4 = vec_msum(q8y10, q2x10, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv5 = vec_msum(q8y11, q2x11, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv6 = vec_msum(q8y12, q2x12, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv7 = vec_msum(q8y13, q2x13, v0);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales_07 = vec_unpackh(vscales);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vscales_03 = vec_unpackh(vscales_07);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vscales_47 = vec_unpackl(vscales_07);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs0 = vec_splat(vscales_03, 0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs1 = vec_splat(vscales_03, 1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs2 = vec_splat(vscales_03, 2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs3 = vec_splat(vscales_03, 3);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs4 = vec_splat(vscales_47, 0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs5 = vec_splat(vscales_47, 1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs6 = vec_splat(vscales_47, 2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs7 = vec_splat(vscales_47, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_ಕೂಡಿಸು(vec_mul(qv0, vs0), vsumi0);
            vsumi1 = vec_ಕೂಡಿಸು(vec_mul(qv1, vs2), vsumi1);
            vsumi2 = vec_ಕೂಡಿಸು(vec_mul(qv2, vs4), vsumi2);
            vsumi3 = vec_ಕೂಡಿಸು(vec_mul(qv3, vs6), vsumi3);
            vsumi4 = vec_ಕೂಡಿಸು(vec_mul(qv4, vs1), vsumi4);
            vsumi5 = vec_ಕೂಡಿಸು(vec_mul(qv5, vs3), vsumi5);
            vsumi6 = vec_ಕೂಡಿಸು(vec_mul(qv6, vs5), vsumi6);
            vsumi7 = vec_ಕೂಡಿಸು(vec_mul(qv7, vs7), vsumi7);
        }

        vsumi0 = vec_ಕೂಡಿಸು(vsumi0, vsumi4);
        vsumi1 = vec_ಕೂಡಿಸು(vsumi1, vsumi5);
        vsumi2 = vec_ಕೂಡಿಸು(vsumi2, vsumi6);
        vsumi3 = vec_ಕೂಡಿಸು(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q3_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x03030303;
    ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0x3);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_Mask1 = vec_splats((int8_t)0xf);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_Mask2 = vec_splats((int8_t)0x30);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ v1 = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0x1);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v2 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x2);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v3 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x3);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v6 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x6);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ off = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0x20);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಬಳಸದ(kmask1);
        ಬಳಸದ(kmask2);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl_len(x[i].scales, 8);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u1 = vec_and(u0, ಕೆಳಗಿನ_Mask1);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl_len(x[i].scales + 8, 4);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u3 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mergeh((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)u2, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)vec_sr(u2, v2));
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u30 = vec_sl(vec_and(u3, ಕೆಳಗಿನ_ಮುಖವಾಡ), v4);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u31 = vec_and(u3, ಕೆಳಗಿನ_Mask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vscales = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mergeh((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ)u1, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ)u2);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxhs0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, x[i].hmask);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxhs1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi4 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi5 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi6 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi7 = v0;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, q3);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(16, q3);
            q3 += 32;

            //the low 2 bits
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs00 = vec_and(qxs0, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs01 = vec_and(vec_sr(qxs0, v2), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs02 = vec_and(vec_sr(qxs0, v4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs03 = vec_and(vec_sr(qxs0, v6), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs10 = vec_and(qxs1, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs11 = vec_and(vec_sr(qxs1, v2), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs12 = vec_and(vec_sr(qxs1, v4), ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs13 = vec_and(vec_sr(qxs1, v6), ಕೆಳಗಿನ_ಮುಖವಾಡ);

            //the 3rd bit
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)v1)), v2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)v1)), v2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x00 = vec_sub(qxs00, qxh00);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x01 = vec_sub(qxs01, qxh01);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x02 = vec_sub(qxs02, qxh02);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x03 = vec_sub(qxs03, qxh03);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x10 = vec_sub(qxs10, qxh10);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x11 = vec_sub(qxs11, qxh11);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x12 = vec_sub(qxs12, qxh12);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x13 = vec_sub(qxs13, qxh13);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y00 = vec_xl(  0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y10 = vec_xl( 16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y01 = vec_xl( 32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y11 = vec_xl( 48, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y02 = vec_xl( 64, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y12 = vec_xl( 80, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y03 = vec_xl( 96, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y13 = vec_xl(112, q8);
            q8 += 128;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales_h = vec_unpackh(vscales);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs0 = vec_splat(vscales_h, 0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs1 = vec_splat(vscales_h, 1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs2 = vec_splat(vscales_h, 2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs3 = vec_splat(vscales_h, 3);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs4 = vec_splat(vscales_h, 4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs5 = vec_splat(vscales_h, 5);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs6 = vec_splat(vscales_h, 6);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv00 = vec_ಕೂಡಿಸು(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv01 = vec_ಕೂಡಿಸು(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv02 = vec_ಕೂಡಿಸು(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv03 = vec_ಕೂಡಿಸು(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv10 = vec_ಕೂಡಿಸು(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv11 = vec_ಕೂಡಿಸು(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv12 = vec_ಕೂಡಿಸು(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv13 = vec_ಕೂಡಿಸು(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs2, vsumi1);
            vsumi2 = vec_msum(qv02, vs4, vsumi2);
            vsumi3 = vec_msum(qv03, vs6, vsumi3);
            vsumi4 = vec_msum(qv10, vs1, vsumi4);
            vsumi5 = vec_msum(qv11, vs3, vsumi5);
            vsumi6 = vec_msum(qv12, vs5, vsumi6);
            vsumi7 = vec_msum(qv13, vs7, vsumi7);
        }

        vsumi0 = vec_ಕೂಡಿಸು(vsumi0, vsumi4);
        vsumi1 = vec_ಕೂಡಿಸು(vsumi1, vsumi5);
        vsumi2 = vec_ಕೂಡಿಸು(vsumi2, vsumi6);
        vsumi3 = vec_ಕೂಡಿಸು(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(kmask1);
    ಬಳಸದ(kmask2);
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q4_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x3f3f3f3f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_Mask1 = vec_splats((int8_t)0x3f);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_Mask2 = vec_splats((int8_t)0x30);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v2 = vec_splats((uint8_t)2);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ದಶಮಾಂಶ vxmin = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin));
        ಸದಿಶ ದಶಮಾಂಶ vdmin = vec_mul(vxmin, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q8ysums0 = vec_xl( 0, y[i].bsums);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q8ysums1 = vec_xl(16, y[i].bsums);

        ಬಳಸದ(kmask1);
        ಬಳಸದ(kmask2);
        ಬಳಸದ(kmask3);
        ಬಳಸದ(utmp);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl_len(x[i].scales, 8);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u1 = vec_and(vec_sr(u0, v2), ಕೆಳಗಿನ_Mask2);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl_len(x[i].scales + 8, 4);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u3 = vec_sr(u2, v4);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u30 = u1;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u31 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mergeh((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)vec_and(u2, ಕೆಳಗಿನ_ಮುಖವಾಡ), (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)u3);

        u1 = vec_and(u0, ಕೆಳಗಿನ_Mask1);
        u2 = vec_or(u30, u31);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ utmps = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mergeh((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)u1, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)u2);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales = vec_unpackh(utmps);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q4xmins = vec_unpackl(utmps);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q4xmins1 = vec_mergel(q4xmins, q4xmins);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod0 = vec_mule(q4xmins0, q8ysums0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod1 = vec_mule(q4xmins1, q8ysums1);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod2 = vec_mulo(q4xmins0, q8ysums0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, q4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(16, q4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(32, q4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs3 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(48, q4);
            q4 += 64;

            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x00 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(qxs0, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x01 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_sr(qxs0, v4);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x10 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(qxs1, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x11 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_sr(qxs1, v4);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x20 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(qxs2, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x21 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_sr(qxs2, v4);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x30 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_and(qxs3, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q4x31 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_sr(qxs3, v4);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y00 = vec_xl(  0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y10 = vec_xl( 16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y01 = vec_xl( 32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y11 = vec_xl( 48, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y20 = vec_xl( 64, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y30 = vec_xl( 80, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y21 = vec_xl( 96, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y31 = vec_xl(112, q8);
            q8 += 128;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv00 = vec_msum(q8y00, q4x00, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv01 = vec_msum(q8y01, q4x01, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv10 = vec_msum(q8y10, q4x10, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv11 = vec_msum(q8y11, q4x11, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv20 = vec_msum(q8y20, q4x20, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv21 = vec_msum(q8y21, q4x21, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv30 = vec_msum(q8y30, q4x30, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv31 = vec_msum(q8y31, q4x31, v0);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vscales_h = vec_unpackh(vscales);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs0 = vec_splat(vscales_h, 0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs1 = vec_splat(vscales_h, 1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs2 = vec_splat(vscales_h, 2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs3 = vec_splat(vscales_h, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_ಕೂಡಿಸು(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_ಕೂಡಿಸು(vec_mul(qv01, vs1), vsumi1);
            vsumi2 = vec_ಕೂಡಿಸು(vec_mul(qv20, vs2), vsumi2);
            vsumi3 = vec_ಕೂಡಿಸು(vec_mul(qv21, vs3), vsumi3);

            vsumi0 = vec_ಕೂಡಿಸು(vec_mul(qv10, vs0), vsumi0);
            vsumi1 = vec_ಕೂಡಿಸು(vec_mul(qv11, vs1), vsumi1);
            vsumi2 = vec_ಕೂಡಿಸು(vec_mul(qv30, vs2), vsumi2);
            vsumi3 = vec_ಕೂಡಿಸು(vec_mul(qv31, vs3), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ಬಳಸದ(kmask1);
    ಬಳಸದ(kmask2);
    ಬಳಸದ(kmask3);
    ಬಳಸದ(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q5_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy,  size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x3f3f3f3f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_Mask1 = vec_splats((int8_t)0x3f);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_Mask2 = vec_splats((int8_t)0x30);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v1 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x1);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v2 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x2);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v3 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x3);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ದಶಮಾಂಶ vxmin = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].dmin));
        ಸದಿಶ ದಶಮಾಂಶ vdmin = vec_mul(vxmin, vyd);

        ಬಳಸದ(kmask1);
        ಬಳಸದ(kmask2);
        ಬಳಸದ(kmask3);
        ಬಳಸದ(utmp);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl_len(x[i].scales, 8);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u1 = vec_and(vec_sr(u0, v2), ಕೆಳಗಿನ_Mask2);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl_len(x[i].scales + 8, 4);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u3 = vec_sr(u2, v4);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u30 = u1;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ u31 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mergeh((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)vec_and(u2, ಕೆಳಗಿನ_ಮುಖವಾಡ), (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)u3);

        u1 = vec_and(u0, ಕೆಳಗಿನ_Mask1);
        u2 = vec_or(u30, u31);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ utmps = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mergeh((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)u1, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ)u2);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q8ysums0 = vec_xl( 0, y[i].bsums);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q8ysums1 = vec_xl(16, y[i].bsums);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales = vec_unpackh(utmps);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q5xmins = vec_unpackl(utmps);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q5xmins0 = vec_mergeh(q5xmins, q5xmins);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q5xmins1 = vec_mergel(q5xmins, q5xmins);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod0 = vec_mule(q5xmins0, q8ysums0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod1 = vec_mule(q5xmins1, q8ysums1);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod2 = vec_mulo(q5xmins0, q8ysums0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ prod3 = vec_mulo(q5xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxhs0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, x[i].qh);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxhs1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(16, x[i].qh);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q5, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, q5);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(16, q5);
            q5 += 32;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs00 = vec_and(qxs0, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs01 = vec_sr(qxs0, v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs10 = vec_and(qxs1, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs11 = vec_sr(qxs1, v4);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q5h00 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v1, qxhs0), v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q5h01 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v2, qxhs0), v3);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q5h10 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v1, qxhs1), v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q5h11 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v2, qxhs1), v3);
            qxhs0 = vec_sr(qxhs0, v2);
            qxhs1 = vec_sr(qxhs1, v2);

            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q5x00 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_or(q5h00, qxs00);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q5x01 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_or(q5h01, qxs01);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q5x10 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_or(q5h10, qxs10);
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ q5x11 = (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)vec_or(q5h11, qxs11);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y00 = vec_xl( 0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y10 = vec_xl(16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y01 = vec_xl(32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y11 = vec_xl(48, q8);
            q8 += 64;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv00 = vec_msum(q8y00, q5x00, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv01 = vec_msum(q8y01, q5x01, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv10 = vec_msum(q8y10, q5x10, v0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ qv11 = vec_msum(q8y11, q5x11, v0);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vscales_h = vec_unpackh(vscales);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs0 = vec_splat(vscales_h, 0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vs1 = vec_splat(vscales_h, 1);
            vscales = vec_sld(vscales, vscales, 12);

            vsumi0 = vec_ಕೂಡಿಸು(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_ಕೂಡಿಸು(vec_mul(qv10, vs0), vsumi1);
            vsumi2 = vec_ಕೂಡಿಸು(vec_mul(qv01, vs1), vsumi2);
            vsumi3 = vec_ಕೂಡಿಸು(vec_mul(qv11, vs1), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ಬಳಸದ(kmask1);
    ಬಳಸದ(kmask2);
    ಬಳಸದ(kmask3);
    ಬಳಸದ(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_q6_K_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v2 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x2);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v3 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x3);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v6 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x6);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ off = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0x20);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi4 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi5 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi6 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi7 = v0;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q6 = x[i].ql;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT qs = x[i].scales;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q6, 0, 0);
            __builtin_prefetch(qh, 0, 0);
            __builtin_prefetch(q8, 0, 0);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, q6);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(16, q6);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(32, q6);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs3 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(48, q6);
            q6 += 64;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs00 = vec_and(qxs0, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs01 = vec_sr(qxs0, v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs10 = vec_and(qxs1, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs11 = vec_sr(qxs1, v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs20 = vec_and(qxs2, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs21 = vec_sr(qxs2, v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs30 = vec_and(qxs3, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs31 = vec_sr(qxs3, v4);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxhs0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, qh);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxhs1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(16, qh);
            qh += 32;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh00 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v3, qxhs0), v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh01 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v3, vec_sr(qxhs0, v4)), v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh10 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v3, qxhs1), v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh11 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v3, vec_sr(qxhs1, v4)), v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh20 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v3, vec_sr(qxhs0, v2)), v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh21 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v3, vec_sr(qxhs0, v6)), v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh30 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v3, vec_sr(qxhs1, v2)), v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxh31 = vec_sl(vec_and((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)v3, vec_sr(qxhs1, v6)), v4);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y00 = vec_xl(  0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y10 = vec_xl( 16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y20 = vec_xl( 32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y30 = vec_xl( 48, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y01 = vec_xl( 64, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y11 = vec_xl( 80, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y21 = vec_xl( 96, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y31 = vec_xl(112, q8);
            q8 += 128;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv00 = vec_ಕೂಡಿಸು(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv10 = vec_ಕೂಡಿಸು(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv20 = vec_ಕೂಡಿಸು(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv30 = vec_ಕೂಡಿಸು(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv01 = vec_ಕೂಡಿಸು(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv11 = vec_ಕೂಡಿಸು(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv21 = vec_ಕೂಡಿಸು(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv31 = vec_ಕೂಡಿಸು(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales = vec_unpackh(vec_xl_len(qs, 8));
            qs += 8;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs0 = vec_splat(vscales, 0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs1 = vec_splat(vscales, 1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs2 = vec_splat(vscales, 2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs3 = vec_splat(vscales, 3);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs4 = vec_splat(vscales, 4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs5 = vec_splat(vscales, 5);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs6 = vec_splat(vscales, 6);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vs7 = vec_splat(vscales, 7);

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs4, vsumi1);
            vsumi2 = vec_msum(qv10, vs1, vsumi2);
            vsumi3 = vec_msum(qv11, vs5, vsumi3);
            vsumi4 = vec_msum(qv20, vs2, vsumi4);
            vsumi5 = vec_msum(qv21, vs6, vsumi5);
            vsumi6 = vec_msum(qv30, vs3, vsumi6);
            vsumi7 = vec_msum(qv31, vs7, vsumi7);
        }

        vsumi0 = vec_ಕೂಡಿಸು(vsumi0, vsumi4);
        vsumi1 = vec_ಕೂಡಿಸು(vsumi1, vsumi5);
        vsumi2 = vec_ಕೂಡಿಸು(vsumi2, vsumi6);
        vsumi3 = vec_ಕೂಡಿಸು(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ defined (__POWER9_VECTOR__)
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int8_t keven_signs_q2xs[1024] = {
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
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_vec_dot_iq2_xxs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K    * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;

        ಸ್ಥಿರಾಂಕ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ಸ್ಥಿರಾಂಕ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            ಸ್ಥಿರಾಂಕ uint8_t * aux8 = (ಸ್ಥಿರಾಂಕ uint8_t *)aux32;

            ಸ್ಮರಣೆ_ನಕಲು(aux32, q2, 4*ಗಾತ್ರ(uint32_t));
            q2 += 8;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_0 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xxs_grid + aux8[ 0]), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xxs_grid + aux8[ 1])};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_1 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xxs_grid + aux8[ 2]), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xxs_grid + aux8[ 3])};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_2 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xxs_grid + aux8[ 8]), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xxs_grid + aux8[ 9])};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_3 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xxs_grid + aux8[10]), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xxs_grid + aux8[11])};

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ vsigns0 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((aux32[1] >>  7) & 127))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ vsigns1 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ vsigns2 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((aux32[3] >>  7) & 127))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ vsigns3 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vsigns0, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vsigns1, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vsigns2, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x3 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vsigns3, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_3);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y2 = vec_xl(32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv2 = vec_ಕೂಡಿಸು(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv3 = vec_ಕೂಡಿಸು(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ಸ್ಥಿರಾಂಕ uint16_t ls0 = aux32[1] >> 28;
            ಸ್ಥಿರಾಂಕ uint16_t ls1 = aux32[3] >> 28;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales01 = vec_splats((int16_t)(2*ls0+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales23 = vec_splats((int16_t)(2*ls1+1));

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

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_iq2_xs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;

        ಸ್ಥಿರಾಂಕ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t  * GGML_RESTRICT sc = x[i].scales;
        ಸ್ಥಿರಾಂಕ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_0 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xs_grid + (q2[0] & 511)), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xs_grid + (q2[1] & 511))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_1 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xs_grid + (q2[2] & 511)), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xs_grid + (q2[3] & 511))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_2 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xs_grid + (q2[4] & 511)), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xs_grid + (q2[5] & 511))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_3 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xs_grid + (q2[6] & 511)), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2xs_grid + (q2[7] & 511))};

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ vsigns0 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((q2[0] >> 9))), *(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((q2[1] >> 9)))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ vsigns1 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((q2[2] >> 9))), *(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((q2[3] >> 9)))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ vsigns2 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((q2[4] >> 9))), *(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((q2[5] >> 9)))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ vsigns3 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((q2[6] >> 9))), *(ಸ್ಥಿರಾಂಕ int64_t *)(signs64 + ((q2[7] >> 9)))};
            q2 += 8;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vsigns0, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vsigns1, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vsigns2, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x3 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vsigns3, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_3);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y2 = vec_xl(32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv2 = vec_ಕೂಡಿಸು(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv3 = vec_ಕೂಡಿಸು(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ಸ್ಥಿರಾಂಕ uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ಸ್ಥಿರಾಂಕ uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ಸ್ಥಿರಾಂಕ uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ಸ್ಥಿರಾಂಕ uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales0 = vec_splats((int16_t)(2*ls0+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales1 = vec_splats((int16_t)(2*ls1+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales2 = vec_splats((int16_t)(2*ls2+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_iq2_s_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ mask0 = vec_xl( 0, k_mask1);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ mask1 = vec_xl(16, k_mask1);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ mask2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, k_mask2);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;

        ಸ್ಥಿರಾಂಕ uint8_t *  GGML_RESTRICT q2 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint16_t * GGML_RESTRICT signs = (ಸ್ಥಿರಾಂಕ uint16_t *)(x[i].qs + QK_K/8);
        ಸ್ಥಿರಾಂಕ uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ಸ್ಥಿರಾಂಕ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_0 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_1 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_2 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_3 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns01 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_splats(*(ಸ್ಥಿರಾಂಕ uint32_t *)&signs[0]);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns23 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_splats(*(ಸ್ಥಿರಾಂಕ uint32_t *)&signs[2]);
            signs += 4;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x0 = vec_sub(vec_xor(vsigns0, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_0), vsigns0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x1 = vec_sub(vec_xor(vsigns1, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_1), vsigns1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x2 = vec_sub(vec_xor(vsigns2, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_2), vsigns2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q2x3 = vec_sub(vec_xor(vsigns3, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_3), vsigns3);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y2 = vec_xl(32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv2 = vec_ಕೂಡಿಸು(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv3 = vec_ಕೂಡಿಸು(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ಸ್ಥಿರಾಂಕ uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ಸ್ಥಿರಾಂಕ uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ಸ್ಥಿರಾಂಕ uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ಸ್ಥಿರಾಂಕ uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales0 = vec_splats((int16_t)(2*ls0+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales1 = vec_splats((int16_t)(2*ls1+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales2 = vec_splats((int16_t)(2*ls2+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_iq3_xxs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K    * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ uint64_t * signs64 = (ಸ್ಥಿರಾಂಕ uint64_t *)keven_signs_q2xs;

    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint32_t * GGML_RESTRICT signs = (ಸ್ಥಿರಾಂಕ uint32_t *)(x[i].qs + QK_K/4);
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[i].qs;

#ಸೂಚನೆ GCC unroll 1
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x0 = vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_0, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux32x4_0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x1 = vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_1, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux32x4_1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x2 = vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_2, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux32x4_2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x3 = vec_mul((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_3, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux32x4_3);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y2 = vec_xl(32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv2 = vec_ಕೂಡಿಸು(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv3 = vec_ಕೂಡಿಸು(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ಸ್ಥಿರಾಂಕ uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            ಸ್ಥಿರಾಂಕ uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales01 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)vec_splats((uint16_t)(2*ls0+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales23 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.25f * vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_iq3_s_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ mask0 = vec_xl( 0, k_mask1);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ mask1 = vec_xl(16, k_mask1);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ mask2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, k_mask2);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸ್ಥಿರಾಂಕ uint8_t *  GGML_RESTRICT q3 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ uint16_t * GGML_RESTRICT signs = (ಸ್ಥಿರಾಂಕ uint16_t *)(x[i].signs);
        ಸ್ಥಿರಾಂಕ uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ಸ್ಥಿರಾಂಕ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns01 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_splats(*(ಸ್ಥಿರಾಂಕ uint32_t *)&signs[0]);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns02 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_splats(*(ಸ್ಥಿರಾಂಕ uint32_t *)&signs[2]);
            signs += 4;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x0 = vec_sub(vec_xor(vsigns0, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux32x4_0), vsigns0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x1 = vec_sub(vec_xor(vsigns1, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux32x4_1), vsigns1);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x2 = vec_sub(vec_xor(vsigns2, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux32x4_2), vsigns2);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q3x3 = vec_sub(vec_xor(vsigns3, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux32x4_3), vsigns3);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y2 = vec_xl(32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv2 = vec_ಕೂಡಿಸು(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv3 = vec_ಕೂಡಿಸು(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ಸ್ಥಿರಾಂಕ uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ಸ್ಥಿರಾಂಕ uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales01 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)vec_splats((uint16_t)(2*ls0+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales23 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_iq1_s_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K  * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v0 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ vsign = vec_splats((ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ)0x8000);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nb; ++i) {
        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[i].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[i].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = vec_splats((int32_t)0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = vec_splats((int32_t)0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = vec_splats((int32_t)0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = vec_splats((int32_t)0);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi8 = vec_splats((int32_t)0);

        ಸ್ಥಿರಾಂಕ uint8_t  * GGML_RESTRICT q1 = x[i].qs;
        ಸ್ಥಿರಾಂಕ uint16_t * GGML_RESTRICT qh = x[i].qh;
        ಸ್ಥಿರಾಂಕ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        ಸ್ಥಿರಾಂಕ int16_t  * GGML_RESTRICT qs = y[i].bsums;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_0 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_1 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_2 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ದೀರ್ಘ ದೀರ್ಘ aux64x2_3 = {*(ಸ್ಥಿರಾಂಕ int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(ಸ್ಥಿರಾಂಕ int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q1x0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_0;
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q1x1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_1;
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q1x2 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_2;
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q1x3 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)aux64x2_3;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y2 = vec_xl(32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv2 = vec_ಕೂಡಿಸು(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv3 = vec_ಕೂಡಿಸು(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            ಸ್ಥಿರಾಂಕ uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            ಸ್ಥಿರಾಂಕ uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales01 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)vec_splats((uint16_t)(2*ls0+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales23 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)vec_splats((uint16_t)(2*ls1+1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)v0);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qxh = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            ಸದಿಶ __bool ಗಿಡ್ಡ vsel = vec_cmpge(qxh, (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)v0);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ q8ysum = vec_sel((ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ)vec_xor((ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ)q8ysums, vsign), q8ysums, vsel);

            vsumi8 = vec_ಕೂಡಿಸು(vec_mule(q8ysum, vscales), vsumi8);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);

        vsumf0 = vec_madd(vec_ctf(vsumi8, 0), vec_mul(vd, vec_splats(IQ1S_DELTA)), vsumf0);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_iq4_nl_q8_0(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK4_NL;

    ಪೂರ್ಣಾಂಕ ib = 0;
    ದಶಮಾಂಶ sumf = 0;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);

    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಮೌಲ್ಯಗಳು = vec_xl( 0, kvalues_iq4nl);

#ಸೂಚನೆ GCC unroll 4
    ಗಾಗಿ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(y[ib].d));
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, x[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x0 = vec_and(qxs, ಕೆಳಗಿನ_ಮುಖವಾಡ);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)q4x0);
        q4x1 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)q4x1);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, y[ib].qs);
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, y[ib].qs);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    }

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ಹೊರತೆಗೆ(vsumf0, 0);

    *s = sumf;
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ಬಳಸದ(ib);
    ಬಳಸದ(sumf);
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vec_dot_iq4_xs_q8_K(ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ * GGML_RESTRICT s, size_t bs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vx, size_t bx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT vy, size_t by, ಪೂರ್ಣಾಂಕ nrc) {
    ಪ್ರತಿಜ್ಞಾ(nrc == 1);
    ಬಳಸದ(nrc);
    ಬಳಸದ(bx);
    ಬಳಸದ(by);
    ಬಳಸದ(bs);
    ಪ್ರತಿಜ್ಞಾ(n % QK_K == 0);

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * GGML_RESTRICT x = vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_K   * GGML_RESTRICT y = vy;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = n / QK_K;

#ಆದರೆ defined(__POWER9_VECTOR__)
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಕೆಳಗಿನ_ಮುಖವಾಡ = vec_splats((ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)0xF);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಪೂರ್ಣಾಂಕ v0 = vec_splats((int32_t)0);
    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ v4 = vec_splats((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)0x4);

    ಸದಿಶ ದಶಮಾಂಶ vsumf0 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf1 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf2 = vec_splats(0.0f);
    ಸದಿಶ ದಶಮಾಂಶ vsumf3 = vec_splats(0.0f);

    ಸ್ಥಿರಾಂಕ ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ ಮೌಲ್ಯಗಳು = vec_xl( 0, kvalues_iq4nl);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibl = 0; ibl < nb; ++ibl) {

        ಸದಿಶ ದಶಮಾಂಶ vxd = vec_splats(GGML_CPU_FP16_ಗಮ್ಯ_FP32(x[ibl].d));
        ಸದಿಶ ದಶಮಾಂಶ vyd = vec_splats(y[ibl].d);
        ಸದಿಶ ದಶಮಾಂಶ vd = vec_mul(vxd, vyd);

        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi0 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi1 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi2 = v0;
        ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಪೂರ್ಣಾಂಕ vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT sc = x[ibl].scales_l;
        ಸ್ಥಿರಾಂಕ int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs0 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl( 0, q4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ qxs1 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_xl(16, q4);
            q4 += 32;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x00 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_and(qxs0, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x01 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_sr(qxs0, v4);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x10 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_and(qxs1, ಕೆಳಗಿನ_ಮುಖವಾಡ);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q4x11 = (ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ)vec_sr(qxs1, v4);

            q4x00 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)q4x00);
            q4x01 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)q4x01);
            q4x10 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)q4x10);
            q4x11 = vec_perm(ಮೌಲ್ಯಗಳು, ಮೌಲ್ಯಗಳು, (ಸದಿಶ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)q4x11);

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y0 = vec_xl( 0, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y1 = vec_xl(16, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y2 = vec_xl(32, q8);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಅಕ್ಷರ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv0 = vec_ಕೂಡಿಸು(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv1 = vec_ಕೂಡಿಸು(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv2 = vec_ಕೂಡಿಸು(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ qv3 = vec_ಕೂಡಿಸು(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            ಸ್ಥಿರಾಂಕ uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            ಸ್ಥಿರಾಂಕ uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales01 = vec_splats((int16_t)ls0);
            ಸದಿಶ ಚಿಹ್ನೆಯುಳ್ಳ ಗಿಡ್ಡ vscales23 = vec_splats((int16_t)ls1);

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

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf2);
    vsumf1 = vec_ಕೂಡಿಸು(vsumf1, vsumf3);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vsumf1);

    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ಕೂಡಿಸು(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ಹೊರತೆಗೆ(vsumf0, 0);

#ಇಲ್ಲದಿದ್ದರೆ
    ಬಳಸದ(x);
    ಬಳಸದ(y);
    ಬಳಸದ(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ಆದರೆ_ಅಂತ್ಯ
}
