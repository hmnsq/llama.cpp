#که_نه_تعریف_شوی HVX_LOG_H
#تعریف_کړه HVX_LOG_H

#ضمن "hvx-base.h"

// Approximates ln(x) element-wise for float vectors.
// x must contain positive float elements.
// Uses Abramowitz & Stegun polynomial approximation 4.1.44 for ln(1+y) over [0, 1].
ساكن مضمن HVX_ویکټور hvx_vec_log_f32(HVX_ویکټور x) {
    // x = m * 2^e, where m in [1, 2)
    HVX_ویکټور biased_e = Q6_Vuw_vlsr_VuwR(x, 23);
    HVX_ویکټور e_int = Q6_Vw_vsub_VwVw(biased_e, Q6_V_vsplat_R(127));
    HVX_ویکټور e_float = Q6_Vsf_equals_Vw(e_int);

    // Extract mantissa and set exponent to 127 (which represents float value in [1.0, 2.0))
    HVX_ویکټور mant_ماسک = Q6_V_vsplat_R(0x007FFFFF);
    HVX_ویکټور exp_127 = Q6_V_vsplat_R(0x3F800000);
    HVX_ویکټور m = Q6_V_vor_VV(Q6_V_vand_VV(x, mant_ماسک), exp_127);

    // y = m - 1.0f, y in [0, 1)
    HVX_ویکټور y = hvx_vec_sub_f32_f32(m, hvx_vec_splat_f32(1.0f));

    // Abramowitz & Stegun 4.1.44 polynomial approximation of ln(1+y)
    HVX_ویکټور c;
    HVX_ویکټور res;

    c   = hvx_vec_splat_f32(-0.0064535442f);
    res = hvx_vec_mul_f32_f32(y, c);

    c   = hvx_vec_splat_f32(0.0360884937f);
    res = hvx_vec_جمع_کړه_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.0953293897f);
    res = hvx_vec_جمع_کړه_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.1676540711f);
    res = hvx_vec_جمع_کړه_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.2407338084f);
    res = hvx_vec_جمع_کړه_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.3317990258f);
    res = hvx_vec_جمع_کړه_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(-0.4998741238f);
    res = hvx_vec_جمع_کړه_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    c   = hvx_vec_splat_f32(0.9999964239f);
    res = hvx_vec_جمع_کړه_f32_f32(res, c);
    res = hvx_vec_mul_f32_f32(y, res);

    // ln(x) = e * ln(2) + ln(1+y)
    HVX_ویکټور ln2 = hvx_vec_splat_f32(0.69314718056f);
    HVX_ویکټور غړی_e = hvx_vec_mul_f32_f32(e_float, ln2);

    أرجع hvx_vec_جمع_کړه_f32_f32(غړی_e, res);
}

ساكن مضمن فراغ hvx_log_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);

    HVX_ویکټور * محدود vdst = (HVX_ویکټور *) dst;
    HVX_ویکټور * محدود vsrc = (HVX_ویکټور *) src;

    ثابت uint32_t elem_اندازه = حجم(عائم);
    ثابت uint32_t epv       = 128 / elem_اندازه;
    ثابت uint32_t nvec      = n / epv;
    ثابت uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_log_f32(vsrc[i]);
    }
    إذا (nloe) {
        HVX_ویکټور v = hvx_vec_log_f32(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v);
    }
}

// Compute log(x) for f16 by promoting to f32, applying hvx_vec_log_f32, and narrowing back.
ساكن مضمن فراغ hvx_log_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);

    HVX_ویکټور * محدود vdst = (HVX_ویکټور *) dst;
    HVX_ویکټور * محدود vsrc = (HVX_ویکټور *) src;

    ثابت uint32_t nvec = n / VLEN_FP16;
    ثابت uint32_t nloe = n % VLEN_FP16;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(vsrc[i]);
        HVX_ویکټور r0 = hvx_vec_log_f32(Q6_V_lo_W(p));
        HVX_ویکټور r1 = hvx_vec_log_f32(Q6_V_hi_W(p));
        vdst[i] = hvx_vec_f32_منزل_f16(r0, r1);
    }
    إذا (nloe) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(vsrc[i]);
        HVX_ویکټور r0 = hvx_vec_log_f32(Q6_V_lo_W(p));
        HVX_ویکټور r1 = hvx_vec_log_f32(Q6_V_hi_W(p));
        HVX_ویکټور v = hvx_vec_f32_منزل_f16(r0, r1);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * SIZEOF_FP16, v);
    }
}

#د_که_پای /* HVX_LOG_H */
