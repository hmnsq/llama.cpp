#که_نه_تعریف_شوی HVX_POW_H
#تعریف_کړه HVX_POW_H

#ضمن <math.h>
#ضمن "hvx-base.h"
#ضمن "hvx-exp.h"
#ضمن "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ساكن مضمن HVX_ویکټور hvx_vec_pow_const_بنسټ_f32(عائم بنسټ, HVX_ویکټور exponent) {
    عائم ln_بنسټ = logf(بنسټ);
    HVX_ویکټور ln_بنسټ_v = hvx_vec_splat_f32(ln_بنسټ);
    HVX_ویکټور x = hvx_vec_mul_f32_f32(exponent, ln_بنسټ_v);

    ساكن ثابت عائم kInf    = INFINITY;
    ساكن ثابت عائم kMaxExp = 88.7228f;

    ثابت HVX_ویکټور max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_ویکټور inf     = hvx_vec_splat_f32(kInf);

    أرجع hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ساكن مضمن HVX_ویکټور hvx_vec_pow_f32(HVX_ویکټور بنسټ, HVX_ویکټور exponent) {
    HVX_ویکټور ln_بنسټ = hvx_vec_log_f32(بنسټ);
    HVX_ویکټور x = hvx_vec_mul_f32_f32(exponent, ln_بنسټ);

    ساكن ثابت عائم kInf    = INFINITY;
    ساكن ثابت عائم kMaxExp = 88.7228f;

    ثابت HVX_ویکټور max_exp = hvx_vec_splat_f32(kMaxExp);
    ثابت HVX_ویکټور inf     = hvx_vec_splat_f32(kInf);

    أرجع hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#د_که_پای /* HVX_POW_H */
