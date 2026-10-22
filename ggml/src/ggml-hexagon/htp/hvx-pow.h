#جيڪڏهن_وصف_نه_ٿيل HVX_POW_H
#وصف_ڪريو HVX_POW_H

#ضمن <math.h>
#ضمن "hvx-base.h"
#ضمن "hvx-exp.h"
#ضمن "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ساكن مضمن HVX_ويڪٽر hvx_vec_pow_const_بنياد_f32(عائم بنياد, HVX_ويڪٽر exponent) {
    عائم ln_بنياد = logf(بنياد);
    HVX_ويڪٽر ln_بنياد_v = hvx_vec_splat_f32(ln_بنياد);
    HVX_ويڪٽر x = hvx_vec_mul_f32_f32(exponent, ln_بنياد_v);

    ساكن ثابت عائم kInf    = INFINITY;
    ساكن ثابت عائم kMaxExp = 88.7228f;

    ثابت HVX_ويڪٽر max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_ويڪٽر inf     = hvx_vec_splat_f32(kInf);

    أرجع hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ساكن مضمن HVX_ويڪٽر hvx_vec_pow_f32(HVX_ويڪٽر بنياد, HVX_ويڪٽر exponent) {
    HVX_ويڪٽر ln_بنياد = hvx_vec_log_f32(بنياد);
    HVX_ويڪٽر x = hvx_vec_mul_f32_f32(exponent, ln_بنياد);

    ساكن ثابت عائم kInf    = INFINITY;
    ساكن ثابت عائم kMaxExp = 88.7228f;

    ثابت HVX_ويڪٽر max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_ويڪٽر inf     = hvx_vec_splat_f32(kInf);

    أرجع hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#جيڪڏهن_پورو /* HVX_POW_H */
