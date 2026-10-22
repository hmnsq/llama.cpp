#నిర్వచితం_కాకపోతే HVX_LOG_H
#నిర్వచించు HVX_LOG_H

#అన్తర్భావ "hvx-base.h"

// Approximates ln(x) element-wise for float vectors.
// x must contain positive float elements.
// Uses Abramowitz & Stegun polynomial approximation 4.1.44 for ln(1+y) over [0, 1].
స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_log_f32(HVX_సదిశ x) {
    // x = m * 2^e, where m in [1, 2)
    HVX_సదిశ biased_e = Q6_Vuw_vlsr_VuwR(x, 23);
    HVX_సదిశ e_int = Q6_Vw_vsub_VwVw(biased_e, Q6_V_vsplat_R(127));
    HVX_సదిశ e_float = Q6_Vsf_equals_Vw(e_int);

    // Extract mantissa and set exponent to 127 (which represents float value in [1.0, 2.0))
    HVX_సదిశ mant_ముసుగు = Q6_V_vsplat_R(0x007FFFFF);
    HVX_సదిశ exp_127 = Q6_V_vsplat_R(0x3F800000);
    HVX_సదిశ m = Q6_V_vor_VV(Q6_V_vand_VV(x, mant_ముసుగు), exp_127);

    // y = m - 1.0f, y in [0, 1)
    HVX_సదిశ y = hvx_vec_sub_f32_f32(m, hvx_vec_splat_f32(1.0f));

    // Abramowitz & Stegun 4.1.44 polynomial approximation of ln(1+y)
    HVX_సదిశ c;
    HVX_సదిశ res;

    c   = hvx_vec_splat_f32(-0.0064535442f);
    res = hvx_vec_mul_f32_f32(y, c);

    c   = hvx_vec_splat_f32(0.0360884937f);
    res = hvx_vec_కూడు_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.0953293897f);
    res = hvx_vec_కూడు_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.1676540711f);
    res = hvx_vec_కూడు_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.2407338084f);
    res = hvx_vec_కూడు_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.3317990258f);
    res = hvx_vec_కూడు_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.4998741238f);
    res = hvx_vec_కూడు_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.9999964239f);
    res = hvx_vec_కూడు_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    // ln(x) = e * ln(2) + ln(1+y)
    HVX_సదిశ ln2 = hvx_vec_splat_f32(0.69314718056f);
    HVX_సదిశ పదం_e = hvx_vec_mul_f32_f32(e_float, ln2);

    తిరిగి hvx_vec_కూడు_f32_f32(పదం_e, res);
}

స్థిర అంతఃవరుస శూన్య hvx_log_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);

    HVX_సదిశ * పరిమిత vdst = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc = (HVX_సదిశ *) src;

    స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t epv       = 128 / elem_కొలత;
    స్థిరరాశి uint32_t nvec      = n / epv;
    స్థిరరాశి uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        vdst[i] = hvx_vec_log_f32(vsrc[i]);
    }
    ఒకవేళ (nloe) {
        HVX_సదిశ v = hvx_vec_log_f32(vsrc[i]);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * elem_కొలత, v);
    }
}

// Compute log(x) for f16 by promoting to f32, applying hvx_vec_log_f32, and narrowing back.
స్థిర అంతఃవరుస శూన్య hvx_log_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);

    HVX_సదిశ * పరిమిత vdst = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc = (HVX_సదిశ *) src;

    స్థిరరాశి uint32_t nvec = n / VLEN_FP16;
    స్థిరరాశి uint32_t nloe = n % VLEN_FP16;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(vsrc[i]);
        HVX_సదిశ r0 = hvx_vec_log_f32(Q6_V_lo_W(p));
        HVX_సదిశ r1 = hvx_vec_log_f32(Q6_V_hi_W(p));
        vdst[i] = hvx_vec_f32_గమ్యం_f16(r0, r1);
    }
    ఒకవేళ (nloe) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(vsrc[i]);
        HVX_సదిశ r0 = hvx_vec_log_f32(Q6_V_lo_W(p));
        HVX_సదిశ r1 = hvx_vec_log_f32(Q6_V_hi_W(p));
        HVX_సదిశ v = hvx_vec_f32_గమ్యం_f16(r0, r1);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * SIZEOF_FP16, v);
    }
}

#అయితే_ముగింపు /* HVX_LOG_H */
