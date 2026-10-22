#اگر_غیر_تعریف_شدہ HVX_POW_H
#تعریف_کرو HVX_POW_H

#شامل <math.h>
#شامل "hvx-base.h"
#شامل "hvx-exp.h"
#شامل "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ساکن اندرونی HVX_سمتیہ hvx_vec_pow_const_بنیاد_f32(اعشاری بنیاد, HVX_سمتیہ exponent) {
    اعشاری ln_بنیاد = logf(بنیاد);
    HVX_سمتیہ ln_بنیاد_v = hvx_vec_splat_f32(ln_بنیاد);
    HVX_سمتیہ x = hvx_vec_mul_f32_f32(exponent, ln_بنیاد_v);

    ساکن ثابت اعشاری kInf    = INFINITY;
    ساکن ثابت اعشاری kMaxExp = 88.7228f;

    ثابت HVX_سمتیہ max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_سمتیہ inf     = hvx_vec_splat_f32(kInf);

    واپس hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ساکن اندرونی HVX_سمتیہ hvx_vec_pow_f32(HVX_سمتیہ بنیاد, HVX_سمتیہ exponent) {
    HVX_سمتیہ ln_بنیاد = hvx_vec_log_f32(بنیاد);
    HVX_سمتیہ x = hvx_vec_mul_f32_f32(exponent, ln_بنیاد);

    ساکن ثابت اعشاری kInf    = INFINITY;
    ساکن ثابت اعشاری kMaxExp = 88.7228f;

    ثابت HVX_سمتیہ max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_سمتیہ inf     = hvx_vec_splat_f32(kInf);

    واپس hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#اگر_ختم /* HVX_POW_H */
