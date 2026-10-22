#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_ERF_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_ERF_H

#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hvx-exp.h"
#ಅನ್ತರ್ಭಾವ "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
ಸ್ಥಿರ __attribute__((noinline)) HVX_ಸದಿಶ hvx_vec_erf_f32(HVX_ಸದಿಶ x) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero = hvx_vec_splat_f32(0.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ax = hvx_vec_abs_f32(x);
    HVX_ಸದಿಶ t = hvx_vec_inverse_f32(hvx_vec_ಕೂಡಿಸು_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_ಸದಿಶ poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ exp_ಪದ = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_ಸದಿಶ ಫಲಿತಾಂಶ = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_ಪದ));

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    ಫಲಿತಾಂಶ = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(ಫಲಿತಾಂಶ), ಫಲಿತಾಂಶ);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// GELU_ERF uses the erf definition.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_gelu_erf_f32(HVX_ಸದಿಶ x) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scale = hvx_vec_splat_f32(0.7071067811865475f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ half  = hvx_vec_splat_f32(0.5f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one   = hvx_vec_splat_f32(1.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_x = hvx_vec_splat_f32(10.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_x = hvx_vec_splat_f32(-10.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_ಸದಿಶ x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_ಸದಿಶ ಫಲಿತಾಂಶ = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_ಕೂಡಿಸು_f32_f32(one, erf));

    ಫಲಿತಾಂಶ = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), ಫಲಿತಾಂಶ);
    ಫಲಿತಾಂಶ = Q6_V_vmux_QVV(pos_large, x, ಫಲಿತಾಂಶ);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_gelu_erf_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc = (HVX_ಸದಿಶ *) src;

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t epv       = 128 / elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t nvec      = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * elem_ಗಾತ್ರ, v);
    }
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_ERF_H */
