#که_نه_تعریف_شوی HVX_ERF_H
#تعریف_کړه HVX_ERF_H

#ضمن "hvx-base.h"
#ضمن "hvx-exp.h"
#ضمن "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
ساكن __attribute__((noinline)) HVX_ویکټور hvx_vec_erf_f32(HVX_ویکټور x) {
    ثابت HVX_ویکټور zero = hvx_vec_splat_f32(0.0f);
    ثابت HVX_ویکټور ax = hvx_vec_abs_f32(x);
    HVX_ویکټور t = hvx_vec_inverse_f32(hvx_vec_جمع_کړه_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_ویکټور poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_جمع_کړه_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جمع_کړه_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جمع_کړه_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جمع_کړه_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    ثابت HVX_ویکټور exp_غړی = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_ویکټور پایله = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_غړی));

    ثابت HVX_ویکټور_Pred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    پایله = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(پایله), پایله);
    أرجع پایله;
}

// GELU_ERF uses the erf definition.
ساكن مضمن HVX_ویکټور hvx_vec_gelu_erf_f32(HVX_ویکټور x) {
    ثابت HVX_ویکټور scale = hvx_vec_splat_f32(0.7071067811865475f);
    ثابت HVX_ویکټور half  = hvx_vec_splat_f32(0.5f);
    ثابت HVX_ویکټور one   = hvx_vec_splat_f32(1.0f);
    ثابت HVX_ویکټور max_x = hvx_vec_splat_f32(10.0f);
    ثابت HVX_ویکټور min_x = hvx_vec_splat_f32(-10.0f);
    ثابت HVX_ویکټور_Pred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    ثابت HVX_ویکټور_Pred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_ویکټور x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    ثابت HVX_ویکټور erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_ویکټور پایله = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_جمع_کړه_f32_f32(one, erf));

    پایله = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), پایله);
    پایله = Q6_V_vmux_QVV(pos_large, x, پایله);
    أرجع پایله;
}

ساكن مضمن فراغ hvx_gelu_erf_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
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
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    إذا (nloe) {
        HVX_ویکټور v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v);
    }
}

#د_که_پای /* HVX_ERF_H */
