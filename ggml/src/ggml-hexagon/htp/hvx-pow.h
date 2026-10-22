#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_POW_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_POW_H

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-exp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_pow_const_ਅਧਾਰ_f32(ਦਸ਼ਮਲਵ ਅਧਾਰ, HVX_ਸਦਿਸ਼ exponent) {
    ਦਸ਼ਮਲਵ ln_ਅਧਾਰ = logf(ਅਧਾਰ);
    HVX_ਸਦਿਸ਼ ln_ਅਧਾਰ_v = hvx_vec_splat_f32(ln_ਅਧਾਰ);
    HVX_ਸਦਿਸ਼ x = hvx_vec_mul_f32_f32(exponent, ln_ਅਧਾਰ_v);

    ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ kInf    = INFINITY;
    ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ kMaxExp = 88.7228f;

    ਅਚਲ HVX_ਸਦਿਸ਼ max_exp = hvx_vec_splat_f32(kMaxExp);
    ਅਚਲ HVX_ਸਦਿਸ਼ inf     = hvx_vec_splat_f32(kInf);

    ਵਾਪਸ hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_pow_f32(HVX_ਸਦਿਸ਼ ਅਧਾਰ, HVX_ਸਦਿਸ਼ exponent) {
    HVX_ਸਦਿਸ਼ ln_ਅਧਾਰ = hvx_vec_log_f32(ਅਧਾਰ);
    HVX_ਸਦਿਸ਼ x = hvx_vec_mul_f32_f32(exponent, ln_ਅਧਾਰ);

    ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ kInf    = INFINITY;
    ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ kMaxExp = 88.7228f;

    ਅਚਲ HVX_ਸਦਿਸ਼ max_exp = hvx_vec_splat_f32(kMaxExp);
    ਅਚਲ HVX_ਸਦਿਸ਼ inf     = hvx_vec_splat_f32(kInf);

    ਵਾਪਸ hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#ਜੇ_ਅੰਤ /* HVX_POW_H */
