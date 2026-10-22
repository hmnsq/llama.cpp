#جے_بنا_تعریف HVX_POW_H
#تعریف_کرو HVX_POW_H

#ضمن <math.h>
#ضمن "hvx-base.h"
#ضمن "hvx-exp.h"
#ضمن "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ساكن مضمن HVX_سمتیہ hvx_vec_pow_const_ادھار_f32(عائم ادھار, HVX_سمتیہ exponent) {
    عائم ln_ادھار = logf(ادھار);
    HVX_سمتیہ ln_ادھار_v = hvx_vec_splat_f32(ln_ادھار);
    HVX_سمتیہ x = hvx_vec_mul_f32_f32(exponent, ln_ادھار_v);

    ساكن ثابت عائم kInf    = INFINITY;
    ساكن ثابت عائم kMaxExp = 88.7228f;

    ثابت HVX_سمتیہ max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_سمتیہ inf     = hvx_vec_splat_f32(kInf);

    أرجع hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ساكن مضمن HVX_سمتیہ hvx_vec_pow_f32(HVX_سمتیہ ادھار, HVX_سمتیہ exponent) {
    HVX_سمتیہ ln_ادھار = hvx_vec_log_f32(ادھار);
    HVX_سمتیہ x = hvx_vec_mul_f32_f32(exponent, ln_ادھار);

    ساكن ثابت عائم kInf    = INFINITY;
    ساكن ثابت عائم kMaxExp = 88.7228f;

    ثابت HVX_سمتیہ max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_سمتیہ inf     = hvx_vec_splat_f32(kInf);

    أرجع hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#جے_انت /* HVX_POW_H */
