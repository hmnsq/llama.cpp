#جيڪڏهن_وصف_نه_ٿيل HVX_ERF_H
#وصف_ڪريو HVX_ERF_H

#ضمن "hvx-base.h"
#ضمن "hvx-exp.h"
#ضمن "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
ساكن __attribute__((noinline)) HVX_ويڪٽر hvx_vec_erf_f32(HVX_ويڪٽر x) {
    ثابت HVX_ويڪٽر zero = hvx_vec_splat_f32(0.0f);
    ثابت HVX_ويڪٽر ax = hvx_vec_abs_f32(x);
    HVX_ويڪٽر t = hvx_vec_inverse_f32(hvx_vec_جوڙيو_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_ويڪٽر poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_جوڙيو_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جوڙيو_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جوڙيو_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جوڙيو_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    ثابت HVX_ويڪٽر exp_رڪن = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_ويڪٽر نتيجو = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_رڪن));

    ثابت HVX_ويڪٽر_Pred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    نتيجو = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(نتيجو), نتيجو);
    أرجع نتيجو;
}

// GELU_ERF uses the erf definition.
ساكن مضمن HVX_ويڪٽر hvx_vec_gelu_erf_f32(HVX_ويڪٽر x) {
    ثابت HVX_ويڪٽر scale = hvx_vec_splat_f32(0.7071067811865475f);
    ثابت HVX_ويڪٽر half  = hvx_vec_splat_f32(0.5f);
    ثابت HVX_ويڪٽر one   = hvx_vec_splat_f32(1.0f);
    ثابت HVX_ويڪٽر max_x = hvx_vec_splat_f32(10.0f);
    ثابت HVX_ويڪٽر min_x = hvx_vec_splat_f32(-10.0f);
    ثابت HVX_ويڪٽر_Pred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    ثابت HVX_ويڪٽر_Pred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_ويڪٽر x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    ثابت HVX_ويڪٽر erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_ويڪٽر نتيجو = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_جوڙيو_f32_f32(one, erf));

    نتيجو = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), نتيجو);
    نتيجو = Q6_V_vmux_QVV(pos_large, x, نتيجو);
    أرجع نتيجو;
}

ساكن مضمن فراغ hvx_gelu_erf_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشان ڊگهو) dst % 128 == 0);
    أكد((بنانشان ڊگهو) src % 128 == 0);

    HVX_ويڪٽر * محدود vdst = (HVX_ويڪٽر *) dst;
    HVX_ويڪٽر * محدود vsrc = (HVX_ويڪٽر *) src;

    ثابت uint32_t elem_سائيز = حجم(عائم);
    ثابت uint32_t epv       = 128 / elem_سائيز;
    ثابت uint32_t nvec      = n / epv;
    ثابت uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    إذا (nloe) {
        HVX_ويڪٽر v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_سائيز, v);
    }
}

#جيڪڏهن_پورو /* HVX_ERF_H */
