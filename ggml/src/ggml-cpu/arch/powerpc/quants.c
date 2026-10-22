#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "simd-mappings.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "../../quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../../ggml-cpu-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h> // for qsort
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>  // for GGML_ASSERT

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS 1e-15f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS_IQ3_XXS 1e-8f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS_IQ2_S 1e-8f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS_IQ1_M 1e-7f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਮੂਹ_MAX_EPS_IQ1_S 1e-12f

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਣਵਰਤਿਆ GGML_ਅਣਵਰਤਿਆ

#ਜੇ defined(__POWER9_VECTOR__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
ਸਥਿਰ ਅਚਲ uint64_t ਸਾਰਣੀ_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
ਸਥਿਰ ਅਚਲ uint64_t ਸਾਰਣੀ_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#ਜੇ_ਅੰਤ

ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_0(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(QK8_0 == 32);
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK8_0;

    ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

#ਜੇ defined(__POWER9_VECTOR__)
    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ srcv [8];
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ asrcv[8];
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ amaxv[8];
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vi[8];

        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        ਲਈ (ਪੂਰਨ j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        ਲਈ (ਪੂਰਨ j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ਅਚਲ ਦਸ਼ਮਲਵ amax = MAX(MAX(vec_ਕੱਢੋ(amaxv[0], 0),
                                   vec_ਕੱਢੋ(amaxv[0], 1)),
                               MAX(vec_ਕੱਢੋ(amaxv[0], 2),
                                   vec_ਕੱਢੋ(amaxv[0], 3)));

        ਅਚਲ ਦਸ਼ਮਲਵ d = amax / ((1 << 7) - 1);
        ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f/d : 0.0f;
        ਅਚਲ ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(d);

        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            ਅਚਲ ਸਦਿਸ਼ ਦਸ਼ਮਲਵ v  = vec_ਗੇੜਾ(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(nb);
    // scalar
    quantize_ਪੰਕਤੀ_q8_0_ref(x, y, k);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_1(ਅਚਲ ਦਸ਼ਮਲਵ * GGML_RESTRICT x, ਖਾਲੀ * GGML_RESTRICT vy, int64_t k) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % QK8_1 == 0);
    ਅਚਲ ਪੂਰਨ nb = k / QK8_1;

    ਖੰਡ_q8_1 * GGML_RESTRICT y = vy;

#ਜੇ defined(__POWER9_VECTOR__)
    ਲਈ (ਪੂਰਨ i = 0; i < nb; i++) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ srcv [8];
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ asrcv[8];
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ amaxv[8];
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vi[8];

        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        ਲਈ (ਪੂਰਨ j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        ਲਈ (ਪੂਰਨ j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        ਅਚਲ ਦਸ਼ਮਲਵ amax = MAX(MAX(vec_ਕੱਢੋ(amaxv[0], 0),
                                   vec_ਕੱਢੋ(amaxv[0], 1)),
                               MAX(vec_ਕੱਢੋ(amaxv[0], 2),
                                   vec_ਕੱਢੋ(amaxv[0], 3)));

        ਅਚਲ ਦਸ਼ਮਲਵ d = amax / ((1 << 7) - 1);
        ਅਚਲ ਦਸ਼ਮਲਵ id = d ? 1.0f/d : 0.0f;
        ਅਚਲ ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vid = vec_splats(id);

        y[i].d = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(d);

        ਸਦਿਸ਼ ਪੂਰਨ accv = vec_splats(0);

        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            ਅਚਲ ਸਦਿਸ਼ ਦਸ਼ਮਲਵ v  = vec_ਗੇੜਾ(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_ਜੋੜੋ(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_ਜੋੜੋ(accv, vec_sld(accv, accv, 4));
        accv = vec_ਜੋੜੋ(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_CPU_FP32_ਮੰਜ਼ਿਲ_FP16(d * vec_ਕੱਢੋ(accv, 0));
    }

#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(nb);
    // scalar
    quantize_ਪੰਕਤੀ_q8_1_ref(x, y, k);
#ਜੇ_ਅੰਤ
}


//===================================== Dot products =================================

ਖਾਲੀ ggml_vec_dot_q4_0_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q4_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ v8 = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0x8);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);

#ਹਦਾਇਤ GCC unroll 8
    ਲਈ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, x[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, y[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, y[ib].qs);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x0 = vec_and(qxs, ਨੀਵਾਂ_ਨਕਾਬ);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ਕੱਢੋ(vsumf0, 0);

    *s = sumf;
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(ib);
    ਅਣਵਰਤਿਆ(sumf);
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_q4_1_q8_1(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_1;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q4_1 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_1 * GGML_RESTRICT y = vy;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);

#ਹਦਾਇਤ GCC unroll 4
    ਲਈ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxmin = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].m));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vys = {GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, x[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, y[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, y[ib].qs);

        ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x0 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(qxs, ਨੀਵਾਂ_ਨਕਾਬ);
        ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x1 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_sr(qxs, v4);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ਕੱਢੋ(vsumf0, 0);

    *s = sumf;
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(ib);
    ਅਣਵਰਤਿਆ(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_mxfp4_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    ਅਚਲ ਖੰਡ_mxfp4 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_MXFP4;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ vshift4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)4);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);

    ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ kv = vec_xl(0, (ਅਚਲ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ *)kvalues_mxfp4);

#ਹਦਾਇਤ GCC unroll 8
    ਲਈ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d) *
                                      GGML_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(x[ib].e));

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, y[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, y[ib].qs);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(0, x[ib].qs);

        ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ lo_nibbles = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(qxs, ਨੀਵਾਂ_ਨਕਾਬ);
        ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ hi_nibbles = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_sr(qxs, vshift4);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x0 = vec_perm(kv, kv, lo_nibbles);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x1 = vec_perm(kv, kv, hi_nibbles);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = vec_splats((int32_t)0);
        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vyd, vsumf0);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));
    sumf = vec_ਕੱਢੋ(vsumf0, 0);
    *s = sumf;
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(ib);
    ਅਣਵਰਤਿਆ(sumf);
    ggml_vec_dot_mxfp4_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_q5_0_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(qk == QK5_0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q5_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)4);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);

#ਹਦਾਇਤ GCC unroll 4
    ਲਈ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_0 = {(uint64_t)(ਸਾਰਣੀ_b2b_1[x[ib].qh[0]]), (uint64_t)(ਸਾਰਣੀ_b2b_1[x[ib].qh[1]])};
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_1 = {(uint64_t)(ਸਾਰਣੀ_b2b_1[x[ib].qh[2]]), (uint64_t)(ਸਾਰਣੀ_b2b_1[x[ib].qh[3]])};

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qh0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qh1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_1;

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, x[ib].qs);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q5x0 = vec_sub(vec_and (qxs, ਨੀਵਾਂ_ਨਕਾਬ), qh0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl(  0, y[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl( 16, y[ib].qs);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q5x1, q8y1), vec_mulo(q5x1, q8y1));

        qv0 = vec_ਜੋੜੋ(qv0, qv1);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = vec_ਜੋੜੋ(vec_unpackh(qv0), vec_unpackl(qv0));

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ਕੱਢੋ(vsumf0, 0);

    *s = sumf;
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(ib);
    ਅਣਵਰਤਿਆ(sumf);
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_q5_1_q8_1(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_1;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(qk == QK5_1);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q5_1 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_1 * GGML_RESTRICT y = vy;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);

#ਹਦਾਇਤ GCC unroll 4
    ਲਈ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxmin = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].m));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vys = {GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].s), 0.f, 0.f, 0.f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ aux64x2_0 = {(uint64_t)(ਸਾਰਣੀ_b2b_0[x[ib].qh[0]]), (uint64_t)(ਸਾਰਣੀ_b2b_0[x[ib].qh[1]])};
        ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ aux64x2_1 = {(uint64_t)(ਸਾਰਣੀ_b2b_0[x[ib].qh[2]]), (uint64_t)(ਸਾਰਣੀ_b2b_0[x[ib].qh[3]])};

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qh0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qh1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_1;

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, x[ib].qs);

        ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q5x0 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_or(vec_and(qxs, ਨੀਵਾਂ_ਨਕਾਬ), qh0);
        ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q5x1 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_or(vec_sr(qxs, v4), qh1);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl(  0, y[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl( 16, y[ib].qs);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q5x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q5x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ਕੱਢੋ(vsumf0, 0);

    *s = sumf;
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ib);
    ਅਣਵਰਤਿਆ(sumf);
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_q8_0_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਅਚਲ ਪੂਰਨ qk = QK8_0;
    ਅਚਲ ਪੂਰਨ nb = n / qk;

    ਪ੍ਰਤਿਜ੍ਞਾ(n % qk == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0 * GGML_RESTRICT y = vy;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);

#ਹਦਾਇਤ GCC unroll 8
    ਲਈ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8x0 = vec_xl( 0, x[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8x1 = vec_xl(16, x[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, y[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, y[ib].qs);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_mule(q8x0, q8y0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_mulo(q8x0, q8y0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv2 = vec_mule(q8x1, q8y1);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv3 = vec_mulo(q8x1, q8y1);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_ਜੋੜੋ(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ਕੱਢੋ(vsumf0, 0);

    *s = sumf;
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(ib);
    ਅਣਵਰਤਿਆ(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_q2_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q2_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0x3);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_Scale_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v2 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x2);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v6 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x6);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxmin = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].dmin));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vdmin = vec_mul(vxmin, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q8ysums0 = vec_xl( 0, y[i].bsums);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q8ysums1 = vec_xl(16, y[i].bsums);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2xmins = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, x[i].scales);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vscales = vec_and(q2xmins, ਨੀਵਾਂ_Scale_ਨਕਾਬ);

        q2xmins = vec_sr(q2xmins, v4);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q2xmins0 = vec_unpackh(q2xmins);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q2xmins1 = vec_unpackl(q2xmins);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod0 = vec_mule(q2xmins0, q8ysums0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod1 = vec_mulo(q2xmins0, q8ysums0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod2 = vec_mule(q2xmins1, q8ysums1);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi4 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi5 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi6 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi7 = v0;

        ਅਚਲ uint8_t * GGML_RESTRICT q2 = x[i].qs;
        ਅਚਲ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, q2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(16, q2);
            q2 += 32;

            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q2x00 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(qxs0, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q2x01 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(vec_sr(qxs0, v2), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q2x02 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(vec_sr(qxs0, v4), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q2x03 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(vec_sr(qxs0, v6), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q2x10 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(qxs1, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q2x11 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(vec_sr(qxs1, v2), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q2x12 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(vec_sr(qxs1, v4), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q2x13 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(vec_sr(qxs1, v6), ਨੀਵਾਂ_ਨਕਾਬ);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y00 = vec_xl(  0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y10 = vec_xl( 16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y01 = vec_xl( 32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y11 = vec_xl( 48, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y02 = vec_xl( 64, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y12 = vec_xl( 80, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y03 = vec_xl( 96, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y13 = vec_xl(112, q8);
            q8 += 128;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv0 = vec_msum(q8y00, q2x00, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv1 = vec_msum(q8y01, q2x01, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv2 = vec_msum(q8y02, q2x02, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv3 = vec_msum(q8y03, q2x03, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv4 = vec_msum(q8y10, q2x10, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv5 = vec_msum(q8y11, q2x11, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv6 = vec_msum(q8y12, q2x12, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv7 = vec_msum(q8y13, q2x13, v0);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales_07 = vec_unpackh(vscales);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vscales_03 = vec_unpackh(vscales_07);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vscales_47 = vec_unpackl(vscales_07);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs0 = vec_splat(vscales_03, 0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs1 = vec_splat(vscales_03, 1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs2 = vec_splat(vscales_03, 2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs3 = vec_splat(vscales_03, 3);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs4 = vec_splat(vscales_47, 0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs5 = vec_splat(vscales_47, 1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs6 = vec_splat(vscales_47, 2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs7 = vec_splat(vscales_47, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_ਜੋੜੋ(vec_mul(qv0, vs0), vsumi0);
            vsumi1 = vec_ਜੋੜੋ(vec_mul(qv1, vs2), vsumi1);
            vsumi2 = vec_ਜੋੜੋ(vec_mul(qv2, vs4), vsumi2);
            vsumi3 = vec_ਜੋੜੋ(vec_mul(qv3, vs6), vsumi3);
            vsumi4 = vec_ਜੋੜੋ(vec_mul(qv4, vs1), vsumi4);
            vsumi5 = vec_ਜੋੜੋ(vec_mul(qv5, vs3), vsumi5);
            vsumi6 = vec_ਜੋੜੋ(vec_mul(qv6, vs5), vsumi6);
            vsumi7 = vec_ਜੋੜੋ(vec_mul(qv7, vs7), vsumi7);
        }

        vsumi0 = vec_ਜੋੜੋ(vsumi0, vsumi4);
        vsumi1 = vec_ਜੋੜੋ(vsumi1, vsumi5);
        vsumi2 = vec_ਜੋੜੋ(vsumi2, vsumi6);
        vsumi3 = vec_ਜੋੜੋ(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_q3_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ uint32_t kmask1 = 0x03030303;
    ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;

    ਅਚਲ ਖੰਡ_q3_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0x3);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_Mask1 = vec_splats((int8_t)0xf);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_Mask2 = vec_splats((int8_t)0x30);
    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ v1 = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0x1);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v2 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x2);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v3 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x3);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v6 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x6);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ off = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0x20);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਅਣਵਰਤਿਆ(kmask1);
        ਅਣਵਰਤਿਆ(kmask2);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl_len(x[i].scales, 8);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u1 = vec_and(u0, ਨੀਵਾਂ_Mask1);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl_len(x[i].scales + 8, 4);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u3 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mergeh((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)u2, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)vec_sr(u2, v2));
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u30 = vec_sl(vec_and(u3, ਨੀਵਾਂ_ਨਕਾਬ), v4);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u31 = vec_and(u3, ਨੀਵਾਂ_Mask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vscales = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mergeh((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ)u1, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ)u2);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxhs0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, x[i].hmask);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxhs1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi4 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi5 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi6 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi7 = v0;

        ਅਚਲ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ਅਚਲ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, q3);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(16, q3);
            q3 += 32;

            //the low 2 bits
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs00 = vec_and(qxs0, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs01 = vec_and(vec_sr(qxs0, v2), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs02 = vec_and(vec_sr(qxs0, v4), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs03 = vec_and(vec_sr(qxs0, v6), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs10 = vec_and(qxs1, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs11 = vec_and(vec_sr(qxs1, v2), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs12 = vec_and(vec_sr(qxs1, v4), ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs13 = vec_and(vec_sr(qxs1, v6), ਨੀਵਾਂ_ਨਕਾਬ);

            //the 3rd bit
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)v1)), v2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)v1)), v2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x00 = vec_sub(qxs00, qxh00);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x01 = vec_sub(qxs01, qxh01);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x02 = vec_sub(qxs02, qxh02);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x03 = vec_sub(qxs03, qxh03);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x10 = vec_sub(qxs10, qxh10);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x11 = vec_sub(qxs11, qxh11);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x12 = vec_sub(qxs12, qxh12);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x13 = vec_sub(qxs13, qxh13);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y00 = vec_xl(  0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y10 = vec_xl( 16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y01 = vec_xl( 32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y11 = vec_xl( 48, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y02 = vec_xl( 64, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y12 = vec_xl( 80, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y03 = vec_xl( 96, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y13 = vec_xl(112, q8);
            q8 += 128;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales_h = vec_unpackh(vscales);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs0 = vec_splat(vscales_h, 0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs1 = vec_splat(vscales_h, 1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs2 = vec_splat(vscales_h, 2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs3 = vec_splat(vscales_h, 3);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs4 = vec_splat(vscales_h, 4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs5 = vec_splat(vscales_h, 5);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs6 = vec_splat(vscales_h, 6);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv00 = vec_ਜੋੜੋ(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv01 = vec_ਜੋੜੋ(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv02 = vec_ਜੋੜੋ(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv03 = vec_ਜੋੜੋ(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv10 = vec_ਜੋੜੋ(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv11 = vec_ਜੋੜੋ(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv12 = vec_ਜੋੜੋ(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv13 = vec_ਜੋੜੋ(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs2, vsumi1);
            vsumi2 = vec_msum(qv02, vs4, vsumi2);
            vsumi3 = vec_msum(qv03, vs6, vsumi3);
            vsumi4 = vec_msum(qv10, vs1, vsumi4);
            vsumi5 = vec_msum(qv11, vs3, vsumi5);
            vsumi6 = vec_msum(qv12, vs5, vsumi6);
            vsumi7 = vec_msum(qv13, vs7, vsumi7);
        }

        vsumi0 = vec_ਜੋੜੋ(vsumi0, vsumi4);
        vsumi1 = vec_ਜੋੜੋ(vsumi1, vsumi5);
        vsumi2 = vec_ਜੋੜੋ(vsumi2, vsumi6);
        vsumi3 = vec_ਜੋੜੋ(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(kmask1);
    ਅਣਵਰਤਿਆ(kmask2);
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_q4_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q4_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਸਥਿਰ ਅਚਲ uint32_t kmask1 = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_Mask1 = vec_splats((int8_t)0x3f);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_Mask2 = vec_splats((int8_t)0x30);
    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v2 = vec_splats((uint8_t)2);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxmin = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].dmin));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vdmin = vec_mul(vxmin, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q8ysums0 = vec_xl( 0, y[i].bsums);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q8ysums1 = vec_xl(16, y[i].bsums);

        ਅਣਵਰਤਿਆ(kmask1);
        ਅਣਵਰਤਿਆ(kmask2);
        ਅਣਵਰਤਿਆ(kmask3);
        ਅਣਵਰਤਿਆ(utmp);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl_len(x[i].scales, 8);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u1 = vec_and(vec_sr(u0, v2), ਨੀਵਾਂ_Mask2);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl_len(x[i].scales + 8, 4);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u3 = vec_sr(u2, v4);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u30 = u1;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u31 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mergeh((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)vec_and(u2, ਨੀਵਾਂ_ਨਕਾਬ), (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)u3);

        u1 = vec_and(u0, ਨੀਵਾਂ_Mask1);
        u2 = vec_or(u30, u31);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ utmps = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mergeh((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)u1, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)u2);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales = vec_unpackh(utmps);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q4xmins = vec_unpackl(utmps);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q4xmins1 = vec_mergel(q4xmins, q4xmins);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod0 = vec_mule(q4xmins0, q8ysums0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod1 = vec_mule(q4xmins1, q8ysums1);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod2 = vec_mulo(q4xmins0, q8ysums0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;

        ਅਚਲ uint8_t * GGML_RESTRICT q4 = x[i].qs;
        ਅਚਲ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, q4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(16, q4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(32, q4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs3 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(48, q4);
            q4 += 64;

            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x00 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(qxs0, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x01 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_sr(qxs0, v4);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x10 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(qxs1, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x11 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_sr(qxs1, v4);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x20 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(qxs2, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x21 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_sr(qxs2, v4);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x30 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_and(qxs3, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q4x31 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_sr(qxs3, v4);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y00 = vec_xl(  0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y10 = vec_xl( 16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y01 = vec_xl( 32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y11 = vec_xl( 48, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y20 = vec_xl( 64, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y30 = vec_xl( 80, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y21 = vec_xl( 96, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y31 = vec_xl(112, q8);
            q8 += 128;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv00 = vec_msum(q8y00, q4x00, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv01 = vec_msum(q8y01, q4x01, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv10 = vec_msum(q8y10, q4x10, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv11 = vec_msum(q8y11, q4x11, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv20 = vec_msum(q8y20, q4x20, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv21 = vec_msum(q8y21, q4x21, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv30 = vec_msum(q8y30, q4x30, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv31 = vec_msum(q8y31, q4x31, v0);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vscales_h = vec_unpackh(vscales);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs0 = vec_splat(vscales_h, 0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs1 = vec_splat(vscales_h, 1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs2 = vec_splat(vscales_h, 2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs3 = vec_splat(vscales_h, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_ਜੋੜੋ(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_ਜੋੜੋ(vec_mul(qv01, vs1), vsumi1);
            vsumi2 = vec_ਜੋੜੋ(vec_mul(qv20, vs2), vsumi2);
            vsumi3 = vec_ਜੋੜੋ(vec_mul(qv21, vs3), vsumi3);

            vsumi0 = vec_ਜੋੜੋ(vec_mul(qv10, vs0), vsumi0);
            vsumi1 = vec_ਜੋੜੋ(vec_mul(qv11, vs1), vsumi1);
            vsumi2 = vec_ਜੋੜੋ(vec_mul(qv30, vs2), vsumi2);
            vsumi3 = vec_ਜੋੜੋ(vec_mul(qv31, vs3), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(kmask1);
    ਅਣਵਰਤਿਆ(kmask2);
    ਅਣਵਰਤਿਆ(kmask3);
    ਅਣਵਰਤਿਆ(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_q5_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy,  size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q5_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

    ਸਥਿਰ ਅਚਲ uint32_t kmask1 = 0x3f3f3f3f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;
    ਸਥਿਰ ਅਚਲ uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_Mask1 = vec_splats((int8_t)0x3f);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_Mask2 = vec_splats((int8_t)0x30);
    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v1 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x1);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v2 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x2);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v3 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x3);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxmin = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].dmin));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vdmin = vec_mul(vxmin, vyd);

        ਅਣਵਰਤਿਆ(kmask1);
        ਅਣਵਰਤਿਆ(kmask2);
        ਅਣਵਰਤਿਆ(kmask3);
        ਅਣਵਰਤਿਆ(utmp);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl_len(x[i].scales, 8);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u1 = vec_and(vec_sr(u0, v2), ਨੀਵਾਂ_Mask2);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl_len(x[i].scales + 8, 4);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u3 = vec_sr(u2, v4);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u30 = u1;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ u31 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mergeh((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)vec_and(u2, ਨੀਵਾਂ_ਨਕਾਬ), (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)u3);

        u1 = vec_and(u0, ਨੀਵਾਂ_Mask1);
        u2 = vec_or(u30, u31);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ utmps = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mergeh((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)u1, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ)u2);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q8ysums0 = vec_xl( 0, y[i].bsums);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q8ysums1 = vec_xl(16, y[i].bsums);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales = vec_unpackh(utmps);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q5xmins = vec_unpackl(utmps);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q5xmins0 = vec_mergeh(q5xmins, q5xmins);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q5xmins1 = vec_mergel(q5xmins, q5xmins);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod0 = vec_mule(q5xmins0, q8ysums0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod1 = vec_mule(q5xmins1, q8ysums1);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod2 = vec_mulo(q5xmins0, q8ysums0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ prod3 = vec_mulo(q5xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxhs0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, x[i].qh);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxhs1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(16, x[i].qh);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;

        ਅਚਲ uint8_t * GGML_RESTRICT q5 = x[i].qs;
        ਅਚਲ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q5, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, q5);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(16, q5);
            q5 += 32;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs00 = vec_and(qxs0, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs01 = vec_sr(qxs0, v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs10 = vec_and(qxs1, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs11 = vec_sr(qxs1, v4);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q5h00 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v1, qxhs0), v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q5h01 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v2, qxhs0), v3);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q5h10 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v1, qxhs1), v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q5h11 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v2, qxhs1), v3);
            qxhs0 = vec_sr(qxhs0, v2);
            qxhs1 = vec_sr(qxhs1, v2);

            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q5x00 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_or(q5h00, qxs00);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q5x01 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_or(q5h01, qxs01);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q5x10 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_or(q5h10, qxs10);
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ q5x11 = (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)vec_or(q5h11, qxs11);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y00 = vec_xl( 0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y10 = vec_xl(16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y01 = vec_xl(32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y11 = vec_xl(48, q8);
            q8 += 64;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv00 = vec_msum(q8y00, q5x00, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv01 = vec_msum(q8y01, q5x01, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv10 = vec_msum(q8y10, q5x10, v0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ qv11 = vec_msum(q8y11, q5x11, v0);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vscales_h = vec_unpackh(vscales);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs0 = vec_splat(vscales_h, 0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vs1 = vec_splat(vscales_h, 1);
            vscales = vec_sld(vscales, vscales, 12);

            vsumi0 = vec_ਜੋੜੋ(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_ਜੋੜੋ(vec_mul(qv10, vs0), vsumi1);
            vsumi2 = vec_ਜੋੜੋ(vec_mul(qv01, vs1), vsumi2);
            vsumi3 = vec_ਜੋੜੋ(vec_mul(qv11, vs1), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(kmask1);
    ਅਣਵਰਤਿਆ(kmask2);
    ਅਣਵਰਤਿਆ(kmask3);
    ਅਣਵਰਤਿਆ(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_q6_K_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_q6_K * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v2 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x2);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v3 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x3);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v6 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x6);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ off = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0x20);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi4 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi5 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi6 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi7 = v0;

        ਅਚਲ uint8_t * GGML_RESTRICT q6 = x[i].ql;
        ਅਚਲ uint8_t * GGML_RESTRICT qh = x[i].qh;
        ਅਚਲ int8_t  * GGML_RESTRICT qs = x[i].scales;
        ਅਚਲ int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q6, 0, 0);
            __builtin_prefetch(qh, 0, 0);
            __builtin_prefetch(q8, 0, 0);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, q6);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(16, q6);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(32, q6);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs3 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(48, q6);
            q6 += 64;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs00 = vec_and(qxs0, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs01 = vec_sr(qxs0, v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs10 = vec_and(qxs1, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs11 = vec_sr(qxs1, v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs20 = vec_and(qxs2, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs21 = vec_sr(qxs2, v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs30 = vec_and(qxs3, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs31 = vec_sr(qxs3, v4);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxhs0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, qh);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxhs1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(16, qh);
            qh += 32;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh00 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v3, qxhs0), v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh01 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v3, vec_sr(qxhs0, v4)), v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh10 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v3, qxhs1), v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh11 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v3, vec_sr(qxhs1, v4)), v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh20 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v3, vec_sr(qxhs0, v2)), v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh21 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v3, vec_sr(qxhs0, v6)), v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh30 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v3, vec_sr(qxhs1, v2)), v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxh31 = vec_sl(vec_and((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)v3, vec_sr(qxhs1, v6)), v4);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y00 = vec_xl(  0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y10 = vec_xl( 16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y20 = vec_xl( 32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y30 = vec_xl( 48, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y01 = vec_xl( 64, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y11 = vec_xl( 80, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y21 = vec_xl( 96, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y31 = vec_xl(112, q8);
            q8 += 128;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv00 = vec_ਜੋੜੋ(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv10 = vec_ਜੋੜੋ(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv20 = vec_ਜੋੜੋ(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv30 = vec_ਜੋੜੋ(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv01 = vec_ਜੋੜੋ(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv11 = vec_ਜੋੜੋ(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv21 = vec_ਜੋੜੋ(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv31 = vec_ਜੋੜੋ(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales = vec_unpackh(vec_xl_len(qs, 8));
            qs += 8;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs0 = vec_splat(vscales, 0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs1 = vec_splat(vscales, 1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs2 = vec_splat(vscales, 2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs3 = vec_splat(vscales, 3);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs4 = vec_splat(vscales, 4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs5 = vec_splat(vscales, 5);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs6 = vec_splat(vscales, 6);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vs7 = vec_splat(vscales, 7);

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs4, vsumi1);
            vsumi2 = vec_msum(qv10, vs1, vsumi2);
            vsumi3 = vec_msum(qv11, vs5, vsumi3);
            vsumi4 = vec_msum(qv20, vs2, vsumi4);
            vsumi5 = vec_msum(qv21, vs6, vsumi5);
            vsumi6 = vec_msum(qv30, vs3, vsumi6);
            vsumi7 = vec_msum(qv31, vs7, vsumi7);
        }

        vsumi0 = vec_ਜੋੜੋ(vsumi0, vsumi4);
        vsumi1 = vec_ਜੋੜੋ(vsumi1, vsumi5);
        vsumi2 = vec_ਜੋੜੋ(vsumi2, vsumi6);
        vsumi3 = vec_ਜੋੜੋ(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

#ਜੇ defined (__POWER9_VECTOR__)
ਸਥਿਰ ਅਚਲ int8_t keven_signs_q2xs[1024] = {
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
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_vec_dot_iq2_xxs_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq2_xxs * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K    * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਅਚਲ uint64_t * signs64 = (ਅਚਲ uint64_t *)keven_signs_q2xs;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;

        ਅਚਲ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ਅਚਲ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            ਅਚਲ uint8_t * aux8 = (ਅਚਲ uint8_t *)aux32;

            ਸਮ੍ਰਿਤੀ_ਨਕਲ(aux32, q2, 4*ਆਕਾਰ(uint32_t));
            q2 += 8;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_0 = {*(ਅਚਲ int64_t *)(iq2xxs_grid + aux8[ 0]), *(ਅਚਲ int64_t *)(iq2xxs_grid + aux8[ 1])};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_1 = {*(ਅਚਲ int64_t *)(iq2xxs_grid + aux8[ 2]), *(ਅਚਲ int64_t *)(iq2xxs_grid + aux8[ 3])};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_2 = {*(ਅਚਲ int64_t *)(iq2xxs_grid + aux8[ 8]), *(ਅਚਲ int64_t *)(iq2xxs_grid + aux8[ 9])};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_3 = {*(ਅਚਲ int64_t *)(iq2xxs_grid + aux8[10]), *(ਅਚਲ int64_t *)(iq2xxs_grid + aux8[11])};

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ vsigns0 = {*(ਅਚਲ int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(ਅਚਲ int64_t *)(signs64 + ((aux32[1] >>  7) & 127))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ vsigns1 = {*(ਅਚਲ int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(ਅਚਲ int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ vsigns2 = {*(ਅਚਲ int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(ਅਚਲ int64_t *)(signs64 + ((aux32[3] >>  7) & 127))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ vsigns3 = {*(ਅਚਲ int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(ਅਚਲ int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vsigns0, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vsigns1, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vsigns2, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x3 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vsigns3, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_3);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y2 = vec_xl(32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv2 = vec_ਜੋੜੋ(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv3 = vec_ਜੋੜੋ(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ਅਚਲ uint16_t ls0 = aux32[1] >> 28;
            ਅਚਲ uint16_t ls1 = aux32[3] >> 28;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales01 = vec_splats((int16_t)(2*ls0+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales23 = vec_splats((int16_t)(2*ls1+1));

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

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_iq2_xs_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq2_xs * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K   * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਅਚਲ uint64_t * signs64 = (ਅਚਲ uint64_t *)keven_signs_q2xs;

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;

        ਅਚਲ uint16_t * GGML_RESTRICT q2 = x[i].qs;
        ਅਚਲ uint8_t  * GGML_RESTRICT sc = x[i].scales;
        ਅਚਲ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_0 = {*(ਅਚਲ int64_t *)(iq2xs_grid + (q2[0] & 511)), *(ਅਚਲ int64_t *)(iq2xs_grid + (q2[1] & 511))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_1 = {*(ਅਚਲ int64_t *)(iq2xs_grid + (q2[2] & 511)), *(ਅਚਲ int64_t *)(iq2xs_grid + (q2[3] & 511))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_2 = {*(ਅਚਲ int64_t *)(iq2xs_grid + (q2[4] & 511)), *(ਅਚਲ int64_t *)(iq2xs_grid + (q2[5] & 511))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_3 = {*(ਅਚਲ int64_t *)(iq2xs_grid + (q2[6] & 511)), *(ਅਚਲ int64_t *)(iq2xs_grid + (q2[7] & 511))};

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ vsigns0 = {*(ਅਚਲ int64_t *)(signs64 + ((q2[0] >> 9))), *(ਅਚਲ int64_t *)(signs64 + ((q2[1] >> 9)))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ vsigns1 = {*(ਅਚਲ int64_t *)(signs64 + ((q2[2] >> 9))), *(ਅਚਲ int64_t *)(signs64 + ((q2[3] >> 9)))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ vsigns2 = {*(ਅਚਲ int64_t *)(signs64 + ((q2[4] >> 9))), *(ਅਚਲ int64_t *)(signs64 + ((q2[5] >> 9)))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ vsigns3 = {*(ਅਚਲ int64_t *)(signs64 + ((q2[6] >> 9))), *(ਅਚਲ int64_t *)(signs64 + ((q2[7] >> 9)))};
            q2 += 8;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vsigns0, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vsigns1, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vsigns2, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x3 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vsigns3, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_3);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y2 = vec_xl(32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv2 = vec_ਜੋੜੋ(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv3 = vec_ਜੋੜੋ(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ਅਚਲ uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ਅਚਲ uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ਅਚਲ uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ਅਚਲ uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales0 = vec_splats((int16_t)(2*ls0+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales1 = vec_splats((int16_t)(2*ls1+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales2 = vec_splats((int16_t)(2*ls2+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_iq2_s_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq2_s * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K  * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਸਥਿਰ ਅਚਲ uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ਸਥਿਰ ਅਚਲ uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ mask0 = vec_xl( 0, k_mask1);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ mask1 = vec_xl(16, k_mask1);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ mask2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, k_mask2);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;

        ਅਚਲ uint8_t *  GGML_RESTRICT q2 = x[i].qs;
        ਅਚਲ uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ਅਚਲ uint16_t * GGML_RESTRICT signs = (ਅਚਲ uint16_t *)(x[i].qs + QK_K/8);
        ਅਚਲ uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ਅਚਲ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_0 = {*(ਅਚਲ int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(ਅਚਲ int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_1 = {*(ਅਚਲ int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(ਅਚਲ int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_2 = {*(ਅਚਲ int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(ਅਚਲ int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_3 = {*(ਅਚਲ int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(ਅਚਲ int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns01 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_splats(*(ਅਚਲ uint32_t *)&signs[0]);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns23 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_splats(*(ਅਚਲ uint32_t *)&signs[2]);
            signs += 4;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x0 = vec_sub(vec_xor(vsigns0, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_0), vsigns0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x1 = vec_sub(vec_xor(vsigns1, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_1), vsigns1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x2 = vec_sub(vec_xor(vsigns2, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_2), vsigns2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q2x3 = vec_sub(vec_xor(vsigns3, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_3), vsigns3);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y2 = vec_xl(32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv2 = vec_ਜੋੜੋ(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv3 = vec_ਜੋੜੋ(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            ਅਚਲ uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ਅਚਲ uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            ਅਚਲ uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            ਅਚਲ uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales0 = vec_splats((int16_t)(2*ls0+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales1 = vec_splats((int16_t)(2*ls1+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales2 = vec_splats((int16_t)(2*ls2+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales3 = vec_splats((int16_t)(2*ls3+1));

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

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_iq3_xxs_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq3_xxs * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K    * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ uint64_t * signs64 = (ਅਚਲ uint64_t *)keven_signs_q2xs;

    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;

        ਅਚਲ uint8_t * GGML_RESTRICT q3 = x[i].qs;
        ਅਚਲ uint32_t * GGML_RESTRICT signs = (ਅਚਲ uint32_t *)(x[i].qs + QK_K/4);
        ਅਚਲ int8_t  * GGML_RESTRICT q8 = y[i].qs;

#ਹਦਾਇਤ GCC unroll 1
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x0 = vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_0, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux32x4_0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x1 = vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_1, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux32x4_1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x2 = vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_2, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux32x4_2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x3 = vec_mul((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_3, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux32x4_3);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y2 = vec_xl(32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv2 = vec_ਜੋੜੋ(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv3 = vec_ਜੋੜੋ(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ਅਚਲ uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            ਅਚਲ uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales01 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)vec_splats((uint16_t)(2*ls0+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales23 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.25f * vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_iq3_s_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq3_s * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K  * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਸਥਿਰ ਅਚਲ uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    ਸਥਿਰ ਅਚਲ uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ mask0 = vec_xl( 0, k_mask1);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ mask1 = vec_xl(16, k_mask1);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ mask2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, k_mask2);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਅਚਲ uint8_t *  GGML_RESTRICT q3 = x[i].qs;
        ਅਚਲ uint8_t *  GGML_RESTRICT qh = x[i].qh;
        ਅਚਲ uint16_t * GGML_RESTRICT signs = (ਅਚਲ uint16_t *)(x[i].signs);
        ਅਚਲ uint8_t *  GGML_RESTRICT sc = x[i].scales;
        ਅਚਲ int8_t  *  GGML_RESTRICT q8 = y[i].qs;

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns01 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_splats(*(ਅਚਲ uint32_t *)&signs[0]);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns02 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_splats(*(ਅਚਲ uint32_t *)&signs[2]);
            signs += 4;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x0 = vec_sub(vec_xor(vsigns0, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux32x4_0), vsigns0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x1 = vec_sub(vec_xor(vsigns1, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux32x4_1), vsigns1);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x2 = vec_sub(vec_xor(vsigns2, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux32x4_2), vsigns2);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q3x3 = vec_sub(vec_xor(vsigns3, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux32x4_3), vsigns3);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y2 = vec_xl(32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv2 = vec_ਜੋੜੋ(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv3 = vec_ਜੋੜੋ(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            ਅਚਲ uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            ਅਚਲ uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales01 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)vec_splats((uint16_t)(2*ls0+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales23 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)vec_splats((uint16_t)(2*ls1+1));

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

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_iq1_s_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);

    ਅਚਲ ਖੰਡ_iq1_s * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K  * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v0 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ vsign = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ)0x8000);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਲਈ (ਪੂਰਨ i = 0; i < nb; ++i) {
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[i].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[i].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = vec_splats((int32_t)0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = vec_splats((int32_t)0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = vec_splats((int32_t)0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = vec_splats((int32_t)0);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi8 = vec_splats((int32_t)0);

        ਅਚਲ uint8_t  * GGML_RESTRICT q1 = x[i].qs;
        ਅਚਲ uint16_t * GGML_RESTRICT qh = x[i].qh;
        ਅਚਲ int8_t   * GGML_RESTRICT q8 = y[i].qs;
        ਅਚਲ int16_t  * GGML_RESTRICT qs = y[i].bsums;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_0 = {*(ਅਚਲ int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(ਅਚਲ int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_1 = {*(ਅਚਲ int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(ਅਚਲ int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_2 = {*(ਅਚਲ int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(ਅਚਲ int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਲੰਮਾ ਲੰਮਾ aux64x2_3 = {*(ਅਚਲ int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(ਅਚਲ int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q1x0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_0;
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q1x1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_1;
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q1x2 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_2;
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q1x3 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)aux64x2_3;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y2 = vec_xl(32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv2 = vec_ਜੋੜੋ(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv3 = vec_ਜੋੜੋ(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            ਅਚਲ uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            ਅਚਲ uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales01 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)vec_splats((uint16_t)(2*ls0+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales23 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)vec_splats((uint16_t)(2*ls1+1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)v0);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qxh = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            ਸਦਿਸ਼ __bool ਛੋਟਾ vsel = vec_cmpge(qxh, (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)v0);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ q8ysum = vec_sel((ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ)vec_xor((ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ)q8ysums, vsign), q8ysums, vsel);

            vsumi8 = vec_ਜੋੜੋ(vec_mule(q8ysum, vscales), vsumi8);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);

        vsumf0 = vec_madd(vec_ctf(vsumi8, 0), vec_mul(vd, vec_splats(IQ1S_DELTA)), vsumf0);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_iq4_nl_q8_0(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    ਅਚਲ ਖੰਡ_iq4_nl * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_0   * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK4_NL;

    ਪੂਰਨ ib = 0;
    ਦਸ਼ਮਲਵ sumf = 0;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);

    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਮੁੱਲਾਂ = vec_xl( 0, kvalues_iq4nl);

#ਹਦਾਇਤ GCC unroll 4
    ਲਈ (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(y[ib].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, x[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x0 = vec_and(qxs, ਨੀਵਾਂ_ਨਕਾਬ);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(ਮੁੱਲਾਂ, ਮੁੱਲਾਂ, (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)q4x0);
        q4x1 = vec_perm(ਮੁੱਲਾਂ, ਮੁੱਲਾਂ, (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)q4x1);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, y[ib].qs);
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, y[ib].qs);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    }

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_ਕੱਢੋ(vsumf0, 0);

    *s = sumf;
#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ਅਣਵਰਤਿਆ(ib);
    ਅਣਵਰਤਿਆ(sumf);
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ ggml_vec_dot_iq4_xs_q8_K(ਪੂਰਨ n, ਦਸ਼ਮਲਵ * GGML_RESTRICT s, size_t bs, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vx, size_t bx, ਅਚਲ ਖਾਲੀ * GGML_RESTRICT vy, size_t by, ਪੂਰਨ nrc) {
    ਪ੍ਰਤਿਜ੍ਞਾ(nrc == 1);
    ਅਣਵਰਤਿਆ(nrc);
    ਅਣਵਰਤਿਆ(bx);
    ਅਣਵਰਤਿਆ(by);
    ਅਣਵਰਤਿਆ(bs);
    ਪ੍ਰਤਿਜ੍ਞਾ(n % QK_K == 0);

    ਅਚਲ ਖੰਡ_iq4_xs * GGML_RESTRICT x = vx;
    ਅਚਲ ਖੰਡ_q8_K   * GGML_RESTRICT y = vy;

    ਅਚਲ ਪੂਰਨ nb = n / QK_K;

#ਜੇ defined(__POWER9_VECTOR__)
    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਨੀਵਾਂ_ਨਕਾਬ = vec_splats((ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)0xF);
    ਅਚਲ ਸਦਿਸ਼ ਪੂਰਨ v0 = vec_splats((int32_t)0);
    ਅਚਲ ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ v4 = vec_splats((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)0x4);

    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf0 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf1 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf2 = vec_splats(0.0f);
    ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vsumf3 = vec_splats(0.0f);

    ਅਚਲ ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ ਮੁੱਲਾਂ = vec_xl( 0, kvalues_iq4nl);

    ਲਈ (ਪੂਰਨ ibl = 0; ibl < nb; ++ibl) {

        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vxd = vec_splats(GGML_CPU_FP16_ਮੰਜ਼ਿਲ_FP32(x[ibl].d));
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vyd = vec_splats(y[ibl].d);
        ਸਦਿਸ਼ ਦਸ਼ਮਲਵ vd = vec_mul(vxd, vyd);

        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi0 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi1 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi2 = v0;
        ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਪੂਰਨ vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        ਅਚਲ uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        ਅਚਲ uint8_t * GGML_RESTRICT sc = x[ibl].scales_l;
        ਅਚਲ int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        ਲਈ (ਪੂਰਨ ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs0 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl( 0, q4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ qxs1 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_xl(16, q4);
            q4 += 32;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x00 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_and(qxs0, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x01 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_sr(qxs0, v4);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x10 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_and(qxs1, ਨੀਵਾਂ_ਨਕਾਬ);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q4x11 = (ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ)vec_sr(qxs1, v4);

            q4x00 = vec_perm(ਮੁੱਲਾਂ, ਮੁੱਲਾਂ, (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)q4x00);
            q4x01 = vec_perm(ਮੁੱਲਾਂ, ਮੁੱਲਾਂ, (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)q4x01);
            q4x10 = vec_perm(ਮੁੱਲਾਂ, ਮੁੱਲਾਂ, (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)q4x10);
            q4x11 = vec_perm(ਮੁੱਲਾਂ, ਮੁੱਲਾਂ, (ਸਦਿਸ਼ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)q4x11);

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y0 = vec_xl( 0, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y1 = vec_xl(16, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y2 = vec_xl(32, q8);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਅੱਖਰ q8y3 = vec_xl(48, q8);
            q8 += 64;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv0 = vec_ਜੋੜੋ(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv1 = vec_ਜੋੜੋ(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv2 = vec_ਜੋੜੋ(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ qv3 = vec_ਜੋੜੋ(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            ਅਚਲ uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            ਅਚਲ uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales01 = vec_splats((int16_t)ls0);
            ਸਦਿਸ਼ ਨਿਸ਼ਾਨੀਵਾਲਾ ਛੋਟਾ vscales23 = vec_splats((int16_t)ls1);

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

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf2);
    vsumf1 = vec_ਜੋੜੋ(vsumf1, vsumf3);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vsumf1);

    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_ਜੋੜੋ(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_ਕੱਢੋ(vsumf0, 0);

#ਨਹੀਂਤਾਂ
    ਅਣਵਰਤਿਆ(x);
    ਅਣਵਰਤਿਆ(y);
    ਅਣਵਰਤਿਆ(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#ਜੇ_ਅੰਤ
}
