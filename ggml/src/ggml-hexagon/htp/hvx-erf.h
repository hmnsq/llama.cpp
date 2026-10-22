#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_ERF_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_ERF_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-exp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
ਸਥਿਰ __attribute__((noinline)) HVX_ਸਦਿਸ਼ hvx_vec_erf_f32(HVX_ਸਦਿਸ਼ x) {
    ਅਚਲ HVX_ਸਦਿਸ਼ zero = hvx_vec_splat_f32(0.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ ax = hvx_vec_abs_f32(x);
    HVX_ਸਦਿਸ਼ t = hvx_vec_inverse_f32(hvx_vec_ਜੋੜੋ_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_ਸਦਿਸ਼ poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    ਅਚਲ HVX_ਸਦਿਸ਼ exp_ਪਦ = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_ਪਦ));

    ਅਚਲ HVX_ਸਦਿਸ਼_Pred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    ਨਤੀਜਾ = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(ਨਤੀਜਾ), ਨਤੀਜਾ);
    ਵਾਪਸ ਨਤੀਜਾ;
}

// GELU_ERF uses the erf definition.
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_gelu_erf_f32(HVX_ਸਦਿਸ਼ x) {
    ਅਚਲ HVX_ਸਦਿਸ਼ scale = hvx_vec_splat_f32(0.7071067811865475f);
    ਅਚਲ HVX_ਸਦਿਸ਼ half  = hvx_vec_splat_f32(0.5f);
    ਅਚਲ HVX_ਸਦਿਸ਼ one   = hvx_vec_splat_f32(1.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_x = hvx_vec_splat_f32(10.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ min_x = hvx_vec_splat_f32(-10.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_ਸਦਿਸ਼ x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    ਅਚਲ HVX_ਸਦਿਸ਼ erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_ਜੋੜੋ_f32_f32(one, erf));

    ਨਤੀਜਾ = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), ਨਤੀਜਾ);
    ਨਤੀਜਾ = Q6_V_vmux_QVV(pos_large, x, ਨਤੀਜਾ);
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_gelu_erf_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst = (HVX_ਸਦਿਸ਼ *) dst;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc = (HVX_ਸਦਿਸ਼ *) src;

    ਅਚਲ uint32_t elem_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t epv       = 128 / elem_ਮਾਪ;
    ਅਚਲ uint32_t nvec      = n / epv;
    ਅਚਲ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * elem_ਮਾਪ, v);
    }
}

#ਜੇ_ਅੰਤ /* HVX_ERF_H */
