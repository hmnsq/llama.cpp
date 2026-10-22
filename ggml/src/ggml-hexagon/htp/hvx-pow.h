#اگر_تعریف_نشده HVX_POW_H
#تعریف_کن HVX_POW_H

#ضمن <math.h>
#ضمن "hvx-base.h"
#ضمن "hvx-exp.h"
#ضمن "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ساكن مضمن HVX_وکتور hvx_vec_pow_const_پایه_f32(عائم پایه, HVX_وکتور exponent) {
    عائم ln_پایه = logf(پایه);
    HVX_وکتور ln_پایه_v = hvx_vec_splat_f32(ln_پایه);
    HVX_وکتور x = hvx_vec_mul_f32_f32(exponent, ln_پایه_v);

    ساكن ثابت عائم kInf    = INFINITY;
    ساكن ثابت عائم kMaxExp = 88.7228f;

    ثابت HVX_وکتور max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_وکتور inf     = hvx_vec_splat_f32(kInf);

    أرجع hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ساكن مضمن HVX_وکتور hvx_vec_pow_f32(HVX_وکتور پایه, HVX_وکتور exponent) {
    HVX_وکتور ln_پایه = hvx_vec_log_f32(پایه);
    HVX_وکتور x = hvx_vec_mul_f32_f32(exponent, ln_پایه);

    ساكن ثابت عائم kInf    = INFINITY;
    ساكن ثابت عائم kMaxExp = 88.7228f;

    ثابت HVX_وکتور max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_وکتور inf     = hvx_vec_splat_f32(kInf);

    أرجع hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#پایان_اگر /* HVX_POW_H */
