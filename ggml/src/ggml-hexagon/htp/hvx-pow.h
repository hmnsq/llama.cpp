#ifndef HVX_POW_H
#define HVX_POW_H

#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ "hvx-base.h"
#ଅନ୍ତର୍ଭାଵ "hvx-exp.h"
#ଅନ୍ତର୍ଭାଵ "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_vec_pow_const_base_f32(ଦଶମିକ base, HVX_Vector exponent) {
    ଦଶମିକ ln_base = logf(base);
    HVX_Vector ln_base_v = hvx_vec_splat_f32(ln_base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base_v);

    ସ୍ଥିର ଅଚଳ ଦଶମିକ kInf    = INFINITY;
    ସ୍ଥିର ଅଚଳ ଦଶମିକ kMaxExp = 88.7228f;

    ଅଚଳ HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    ଅଚଳ HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    ଫେରନ୍ତୁ hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_vec_pow_f32(HVX_Vector base, HVX_Vector exponent) {
    HVX_Vector ln_base = hvx_vec_log_f32(base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base);

    ସ୍ଥିର ଅଚଳ ଦଶମିକ kInf    = INFINITY;
    ସ୍ଥିର ଅଚଳ ଦଶମିକ kMaxExp = 88.7228f;

    ଅଚଳ HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    ଅଚଳ HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    ଫେରନ୍ତୁ hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#endif /* HVX_POW_H */
