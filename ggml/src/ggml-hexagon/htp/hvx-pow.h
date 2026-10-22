#ifndef HVX_POW_H
#define HVX_POW_H

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hvx-exp.h"
#ಅನ್ತರ್ಭಾವ "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ HVX_Vector hvx_vec_pow_const_base_f32(ಭಿನ್ನ base, HVX_Vector exponent) {
    ಭಿನ್ನ ln_base = logf(base);
    HVX_Vector ln_base_v = hvx_vec_splat_f32(ln_base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base_v);

    ಸ್ಥಿರ ಧ್ರುವ ಭಿನ್ನ kInf    = INFINITY;
    ಸ್ಥಿರ ಧ್ರುವ ಭಿನ್ನ kMaxExp = 88.7228f;

    ಧ್ರುವ HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    ಧ್ರುವ HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    ಪ್ರತ್ಯಾಗಮ hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ HVX_Vector hvx_vec_pow_f32(HVX_Vector base, HVX_Vector exponent) {
    HVX_Vector ln_base = hvx_vec_log_f32(base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base);

    ಸ್ಥಿರ ಧ್ರುವ ಭಿನ್ನ kInf    = INFINITY;
    ಸ್ಥಿರ ಧ್ರುವ ಭಿನ್ನ kMaxExp = 88.7228f;

    ಧ್ರುವ HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    ಧ್ರುವ HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    ಪ್ರತ್ಯಾಗಮ hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#endif /* HVX_POW_H */
