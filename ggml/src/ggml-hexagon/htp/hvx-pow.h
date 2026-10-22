#ifndef HVX_POW_H
#define HVX_POW_H

#அந்தர்பாவ <math.h>
#அந்தர்பாவ "hvx-base.h"
#அந்தர்பாவ "hvx-exp.h"
#அந்தர்பாவ "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
நிலையான உள்வரி HVX_Vector hvx_vec_pow_const_base_f32(தசம base, HVX_Vector exponent) {
    தசம ln_base = logf(base);
    HVX_Vector ln_base_v = hvx_vec_splat_f32(ln_base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base_v);

    நிலையான மாறிலி தசம kInf    = INFINITY;
    நிலையான மாறிலி தசம kMaxExp = 88.7228f;

    மாறிலி HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    மாறிலி HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    திரும்பு hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
நிலையான உள்வரி HVX_Vector hvx_vec_pow_f32(HVX_Vector base, HVX_Vector exponent) {
    HVX_Vector ln_base = hvx_vec_log_f32(base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base);

    நிலையான மாறிலி தசம kInf    = INFINITY;
    நிலையான மாறிலி தசம kMaxExp = 88.7228f;

    மாறிலி HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    மாறிலி HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    திரும்பு hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#endif /* HVX_POW_H */
