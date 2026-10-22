#నిర్వచితం_కాకపోతే HVX_POW_H
#నిర్వచించు HVX_POW_H

#అన్తర్భావ <math.h>
#అన్తర్భావ "hvx-base.h"
#అన్తర్భావ "hvx-exp.h"
#అన్తర్భావ "hvx-log.h"

// Approximates base^exponent element-wise for float vectors.
// base must be a positive constant. exponent is an HVX f32 vector.
// Uses base^x = exp(x * ln(base)).
స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_pow_const_పునాది_f32(దశాంశం పునాది, HVX_సదిశ exponent) {
    దశాంశం ln_పునాది = logf(పునాది);
    HVX_సదిశ ln_పునాది_v = hvx_vec_splat_f32(ln_పునాది);
    HVX_సదిశ x = hvx_vec_mul_f32_f32(exponent, ln_పునాది_v);

    స్థిర స్థిరరాశి దశాంశం kInf    = INFINITY;
    స్థిర స్థిరరాశి దశాంశం kMaxExp = 88.7228f;

    స్థిరరాశి HVX_సదిశ max_exp = hvx_vec_splat_f32(kMaxExp);
    స్థిరరాశి HVX_సదిశ inf     = hvx_vec_splat_f32(kInf);

    తిరిగి hvx_vec_exp_f32_guard(x, max_exp, inf);
}

// Approximates base^exponent element-wise for float vectors.
// base and exponent are HVX f32 vectors. base elements must be positive.
// Uses base^exponent = exp(exponent * ln(base)).
స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_pow_f32(HVX_సదిశ పునాది, HVX_సదిశ exponent) {
    HVX_సదిశ ln_పునాది = hvx_vec_log_f32(పునాది);
    HVX_సదిశ x = hvx_vec_mul_f32_f32(exponent, ln_పునాది);

    స్థిర స్థిరరాశి దశాంశం kInf    = INFINITY;
    స్థిర స్థిరరాశి దశాంశం kMaxExp = 88.7228f;

    స్థిరరాశి HVX_సదిశ max_exp = hvx_vec_splat_f32(kMaxExp);
    స్థిరరాశి HVX_సదిశ inf     = hvx_vec_splat_f32(kInf);

    తిరిగి hvx_vec_exp_f32_guard(x, max_exp, inf);
}

#అయితే_ముగింపు /* HVX_POW_H */
