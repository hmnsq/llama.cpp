#यदि_अपरिभाषितम् HVX_ERF_H
#परिभाषय HVX_ERF_H

#अन्तर्भाव "hvx-base.h"
#अन्तर्भाव "hvx-exp.h"
#अन्तर्भाव "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
स्थिर __attribute__((noinline)) HVX_सदिशम् hvx_vec_erf_f32(HVX_सदिशम् x) {
    ध्रुव HVX_सदिशम् zero = hvx_vec_splat_f32(0.0f);
    ध्रुव HVX_सदिशम् ax = hvx_vec_abs_f32(x);
    HVX_सदिशम् t = hvx_vec_inverse_f32(hvx_vec_सङ्कलय_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_सदिशम् poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_सङ्कलय_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_सङ्कलय_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_सङ्कलय_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_सङ्कलय_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    ध्रुव HVX_सदिशम् exp_पदांशः = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_सदिशम् परिणाम = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_पदांशः));

    ध्रुव HVX_सदिशम्_Pred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    परिणाम = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(परिणाम), परिणाम);
    प्रत्यागम परिणाम;
}

// GELU_ERF uses the erf definition.
स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_gelu_erf_f32(HVX_सदिशम् x) {
    ध्रुव HVX_सदिशम् scale = hvx_vec_splat_f32(0.7071067811865475f);
    ध्रुव HVX_सदिशम् half  = hvx_vec_splat_f32(0.5f);
    ध्रुव HVX_सदिशम् one   = hvx_vec_splat_f32(1.0f);
    ध्रुव HVX_सदिशम् max_x = hvx_vec_splat_f32(10.0f);
    ध्रुव HVX_सदिशम् min_x = hvx_vec_splat_f32(-10.0f);
    ध्रुव HVX_सदिशम्_Pred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    ध्रुव HVX_सदिशम्_Pred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_सदिशम् x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    ध्रुव HVX_सदिशम् erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_सदिशम् परिणाम = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_सङ्कलय_f32_f32(one, erf));

    परिणाम = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), परिणाम);
    परिणाम = Q6_V_vmux_QVV(pos_large, x, परिणाम);
    प्रत्यागम परिणाम;
}

स्थिर अन्तःस्थ शून्य hvx_gelu_erf_f32_aa(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, uint32_t n) {
    प्रतिज्ञा((अचिह्नित दीर्घ) dst % 128 == 0);
    प्रतिज्ञा((अचिह्नित दीर्घ) src % 128 == 0);

    HVX_सदिशम् * सीमित vdst = (HVX_सदिशम् *) dst;
    HVX_सदिशम् * सीमित vsrc = (HVX_सदिशम् *) src;

    ध्रुव uint32_t elem_आकार = परिमाण(भिन्न);
    ध्रुव uint32_t epv       = 128 / elem_आकार;
    ध्रुव uint32_t nvec      = n / epv;
    ध्रुव uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    प्रति (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    यदि (nloe) {
        HVX_सदिशम् v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((शून्य *) &vdst[i], nloe * elem_आकार, v);
    }
}

#यदि_अन्तः /* HVX_ERF_H */
