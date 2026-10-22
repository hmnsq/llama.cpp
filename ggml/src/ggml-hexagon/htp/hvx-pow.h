#ifndef HVX_POW_H
#define HVX_POW_H

#अन्तर्भाव <math.h>
#अन्तर्भाव "hvx-base.h"
#अन्तर्भाव "hvx-exp.h"
#अन्तर्भाव "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
स्थिर अन्तःस्थ HVX_Vector hvx_vec_pow_const_base_f32(भिन्न base, HVX_Vector exponent) {
    भिन्न ln_base = logf(base);
    HVX_Vector ln_base_v = hvx_vec_splat_f32(ln_base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base_v);

    स्थिर ध्रुव भिन्न kInf    = अनन्त;
    स्थिर ध्रुव भिन्न kMaxExp = 88.7228f;

    ध्रुव HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    ध्रुव HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    प्रत्यागम hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
स्थिर अन्तःस्थ HVX_Vector hvx_vec_pow_f32(HVX_Vector base, HVX_Vector exponent) {
    HVX_Vector ln_base = hvx_vec_log_f32(base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base);

    स्थिर ध्रुव भिन्न kInf    = अनन्त;
    स्थिर ध्रुव भिन्न kMaxExp = 88.7228f;

    ध्रुव HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    ध्रुव HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    प्रत्यागम hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#endif /* HVX_POW_H */
