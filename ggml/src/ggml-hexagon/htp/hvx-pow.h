#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_POW_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_POW_H

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hvx-exp.h"
#ಅನ್ತರ್ಭಾವ "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_pow_const_ತಳ_f32(ದಶಮಾಂಶ ತಳ, HVX_ಸದಿಶ exponent) {
    ದಶಮಾಂಶ ln_ತಳ = logf(ತಳ);
    HVX_ಸದಿಶ ln_ತಳ_v = hvx_vec_splat_f32(ln_ತಳ);
    HVX_ಸದಿಶ x = hvx_vec_mul_f32_f32(exponent, ln_ತಳ_v);

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ kInf    = INFINITY;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ kMaxExp = 88.7228f;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_exp = hvx_vec_splat_f32(kMaxExp);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ inf     = hvx_vec_splat_f32(kInf);

    ಹಿಂತಿರುಗಿ hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_pow_f32(HVX_ಸದಿಶ ತಳ, HVX_ಸದಿಶ exponent) {
    HVX_ಸದಿಶ ln_ತಳ = hvx_vec_log_f32(ತಳ);
    HVX_ಸದಿಶ x = hvx_vec_mul_f32_f32(exponent, ln_ತಳ);

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ kInf    = INFINITY;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ kMaxExp = 88.7228f;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_exp = hvx_vec_splat_f32(kMaxExp);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ inf     = hvx_vec_splat_f32(kInf);

    ಹಿಂತಿರುಗಿ hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_POW_H */
