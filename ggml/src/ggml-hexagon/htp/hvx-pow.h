#अगर_अघोषित HVX_POW_H
#घोषणा HVX_POW_H

#समावेश <math.h>
#समावेश "hvx-base.h"
#समावेश "hvx-exp.h"
#समावेश "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
अटल अंतरभूत HVX_Vector hvx_vec_pow_const_base_f32(भग्न base, HVX_Vector exponent) {
    भग्न ln_base = logf(base);
    HVX_Vector ln_base_v = hvx_vec_splat_f32(ln_base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base_v);

    अटल स्थिर भग्न kInf    = अनन्त;
    अटल स्थिर भग्न kMaxExp = 88.7228f;

    स्थिर HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    स्थिर HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    वापस hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
अटल अंतरभूत HVX_Vector hvx_vec_pow_f32(HVX_Vector base, HVX_Vector exponent) {
    HVX_Vector ln_base = hvx_vec_log_f32(base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base);

    अटल स्थिर भग्न kInf    = अनन्त;
    अटल स्थिर भग्न kMaxExp = 88.7228f;

    स्थिर HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    स्थिर HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    वापस hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#पूर्ण_अगर /* HVX_POW_H */
