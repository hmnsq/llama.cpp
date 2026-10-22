#جے_بنا_تعریف HVX_ERF_H
#تعریف_کرو HVX_ERF_H

#ضمن "hvx-base.h"
#ضمن "hvx-exp.h"
#ضمن "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
ساكن __attribute__((noinline)) HVX_سمتیہ hvx_vec_erf_f32(HVX_سمتیہ x) {
    ثابت HVX_سمتیہ zero = hvx_vec_splat_f32(0.0f);
    ثابت HVX_سمتیہ ax = hvx_vec_abs_f32(x);
    HVX_سمتیہ t = hvx_vec_inverse_f32(hvx_vec_جوڑو_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_سمتیہ poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_جوڑو_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جوڑو_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جوڑو_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_جوڑو_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    ثابت HVX_سمتیہ exp_رکن = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_سمتیہ نتیجہ = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_رکن));

    ثابت HVX_سمتیہ_Pred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    نتیجہ = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(نتیجہ), نتیجہ);
    أرجع نتیجہ;
}

// GELU_ERF uses the erf definition.
ساكن مضمن HVX_سمتیہ hvx_vec_gelu_erf_f32(HVX_سمتیہ x) {
    ثابت HVX_سمتیہ scale = hvx_vec_splat_f32(0.7071067811865475f);
    ثابت HVX_سمتیہ half  = hvx_vec_splat_f32(0.5f);
    ثابت HVX_سمتیہ one   = hvx_vec_splat_f32(1.0f);
    ثابت HVX_سمتیہ max_x = hvx_vec_splat_f32(10.0f);
    ثابت HVX_سمتیہ min_x = hvx_vec_splat_f32(-10.0f);
    ثابت HVX_سمتیہ_Pred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    ثابت HVX_سمتیہ_Pred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_سمتیہ x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    ثابت HVX_سمتیہ erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_سمتیہ نتیجہ = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_جوڑو_f32_f32(one, erf));

    نتیجہ = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), نتیجہ);
    نتیجہ = Q6_V_vmux_QVV(pos_large, x, نتیجہ);
    أرجع نتیجہ;
}

ساكن مضمن فراغ hvx_gelu_erf_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشانی لما) dst % 128 == 0);
    أكد((بنانشانی لما) src % 128 == 0);

    HVX_سمتیہ * محدود vdst = (HVX_سمتیہ *) dst;
    HVX_سمتیہ * محدود vsrc = (HVX_سمتیہ *) src;

    ثابت uint32_t elem_ناپ = حجم(عائم);
    ثابت uint32_t epv       = 128 / elem_ناپ;
    ثابت uint32_t nvec      = n / epv;
    ثابت uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    إذا (nloe) {
        HVX_سمتیہ v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * elem_ناپ, v);
    }
}

#جے_انت /* HVX_ERF_H */
