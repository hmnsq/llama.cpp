#ifndef HVX_POW_H
#define HVX_POW_H

#شامل <math.h>
#شامل "hvx-base.h"
#شامل "hvx-exp.h"
#شامل "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ایستا درونخطی HVX_Vector hvx_vec_pow_const_base_f32(اعشاری base, HVX_Vector exponent) {
    اعشاری ln_base = logf(base);
    HVX_Vector ln_base_v = hvx_vec_splat_f32(ln_base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base_v);

    ایستا ثابت اعشاری kInf    = INFINITY;
    ایستا ثابت اعشاری kMaxExp = 88.7228f;

    ثابت HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    بازگشت hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ایستا درونخطی HVX_Vector hvx_vec_pow_f32(HVX_Vector base, HVX_Vector exponent) {
    HVX_Vector ln_base = hvx_vec_log_f32(base);
    HVX_Vector x = hvx_vec_mul_f32_f32(exponent, ln_base);

    ایستا ثابت اعشاری kInf    = INFINITY;
    ایستا ثابت اعشاری kMaxExp = 88.7228f;

    ثابت HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    بازگشت hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#endif /* HVX_POW_H */
