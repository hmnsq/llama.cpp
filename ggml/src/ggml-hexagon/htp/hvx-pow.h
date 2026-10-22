#यदि_अपरिभाषितम् HVX_POW_H
#परिभाषय HVX_POW_H

#अन्तर्भाव <math.h>
#अन्तर्भाव "hvx-base.h"
#अन्तर्भाव "hvx-exp.h"
#अन्तर्भाव "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_pow_const_आधारः_f32(भिन्न आधारः, HVX_सदिशम् exponent) {
    भिन्न ln_आधारः = logf(आधारः);
    HVX_सदिशम् ln_आधारः_v = hvx_vec_splat_f32(ln_आधारः);
    HVX_सदिशम् x = hvx_vec_mul_f32_f32(exponent, ln_आधारः_v);

    स्थिर ध्रुव भिन्न kInf    = INFINITY;
    स्थिर ध्रुव भिन्न kMaxExp = 88.7228f;

    ध्रुव HVX_सदिशम् max_exp = hvx_vec_splat_f32(kMaxExp);
    ध्रुव HVX_सदिशम् inf     = hvx_vec_splat_f32(kInf);

    प्रत्यागम hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_pow_f32(HVX_सदिशम् आधारः, HVX_सदिशम् exponent) {
    HVX_सदिशम् ln_आधारः = hvx_vec_log_f32(आधारः);
    HVX_सदिशम् x = hvx_vec_mul_f32_f32(exponent, ln_आधारः);

    स्थिर ध्रुव भिन्न kInf    = INFINITY;
    स्थिर ध्रुव भिन्न kMaxExp = 88.7228f;

    ध्रुव HVX_सदिशम् max_exp = hvx_vec_splat_f32(kMaxExp);
    ध्रुव HVX_सदिशम् inf     = hvx_vec_splat_f32(kInf);

    प्रत्यागम hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#यदि_अन्तः /* HVX_POW_H */
