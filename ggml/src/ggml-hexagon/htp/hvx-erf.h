#اگر_تعریف_نشده HVX_ERF_H
#تعریف_کن HVX_ERF_H

#ضمن "hvx-base.h"
#ضمن "hvx-exp.h"
#ضمن "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
ساكن __attribute__((noinline)) HVX_وکتور hvx_vec_erf_f32(HVX_وکتور x) {
    ثابت HVX_وکتور zero = hvx_vec_splat_f32(0.0f);
    ثابت HVX_وکتور ax = hvx_vec_abs_f32(x);
    HVX_وکتور t = hvx_vec_inverse_f32(hvx_vec_جمع_کن_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_وکتور poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_جمع_کن_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جمع_کن_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جمع_کن_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جمع_کن_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    ثابت HVX_وکتور exp_جمله = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_وکتور نتیجه = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_جمله));

    ثابت HVX_وکتور_Pred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    نتیجه = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(نتیجه), نتیجه);
    أرجع نتیجه;
}

// GELU_ERF uses the erf definition.
ساكن مضمن HVX_وکتور hvx_vec_gelu_erf_f32(HVX_وکتور x) {
    ثابت HVX_وکتور scale = hvx_vec_splat_f32(0.7071067811865475f);
    ثابت HVX_وکتور half  = hvx_vec_splat_f32(0.5f);
    ثابت HVX_وکتور one   = hvx_vec_splat_f32(1.0f);
    ثابت HVX_وکتور max_x = hvx_vec_splat_f32(10.0f);
    ثابت HVX_وکتور min_x = hvx_vec_splat_f32(-10.0f);
    ثابت HVX_وکتور_Pred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    ثابت HVX_وکتور_Pred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_وکتور x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    ثابت HVX_وکتور erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_وکتور نتیجه = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_جمع_کن_f32_f32(one, erf));

    نتیجه = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), نتیجه);
    نتیجه = Q6_V_vmux_QVV(pos_large, x, نتیجه);
    أرجع نتیجه;
}

ساكن مضمن فراغ hvx_gelu_erf_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بیعلامت دراز) dst % 128 == 0);
    أكد((بیعلامت دراز) src % 128 == 0);

    HVX_وکتور * محدود vdst = (HVX_وکتور *) dst;
    HVX_وکتور * محدود vsrc = (HVX_وکتور *) src;

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
        HVX_وکتور v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_اندازه, v);
    }
}

#پایان_اگر /* HVX_ERF_H */
