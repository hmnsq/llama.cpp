#నిర్వచితం_కాకపోతే HVX_ERF_H
#నిర్వచించు HVX_ERF_H

#అన్తర్భావ "hvx-base.h"
#అన్తర్భావ "hvx-exp.h"
#అన్తర్భావ "hvx-inverse.h"

// Maximum error is about 1.5e-7 for the Abramowitz-Stegun approximation.
స్థిర __attribute__((noinline)) HVX_సదిశ hvx_vec_erf_f32(HVX_సదిశ x) {
    స్థిరరాశి HVX_సదిశ zero = hvx_vec_splat_f32(0.0f);
    స్థిరరాశి HVX_సదిశ ax = hvx_vec_abs_f32(x);
    HVX_సదిశ t = hvx_vec_inverse_f32(hvx_vec_కూడు_f32_f32(
        hvx_vec_splat_f32(1.0f), hvx_vec_mul_f32_f32(hvx_vec_splat_f32(0.3275911f), ax)));

    HVX_సదిశ poly = hvx_vec_splat_f32(1.061405429f);
    poly = hvx_vec_కూడు_f32_f32(hvx_vec_splat_f32(-1.453152027f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_కూడు_f32_f32(hvx_vec_splat_f32(1.421413741f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_కూడు_f32_f32(hvx_vec_splat_f32(-0.284496736f), hvx_vec_mul_f32_f32(poly, t));
    poly = hvx_vec_కూడు_f32_f32(hvx_vec_splat_f32(0.254829592f), hvx_vec_mul_f32_f32(poly, t));

    స్థిరరాశి HVX_సదిశ exp_పదం = hvx_vec_exp_f32(hvx_vec_neg_f32(hvx_vec_mul_f32_f32(ax, ax)));
    HVX_సదిశ ఫలితం = hvx_vec_sub_f32_f32(hvx_vec_splat_f32(1.0f),
        hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(poly, t), exp_పదం));

    స్థిరరాశి HVX_సదిశ_Pred neg = Q6_Q_vcmp_gt_VsfVsf(zero, x);
    ఫలితం = Q6_V_vmux_QVV(neg, hvx_vec_neg_f32(ఫలితం), ఫలితం);
    తిరిగి ఫలితం;
}

// GELU_ERF uses the erf definition.
స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_gelu_erf_f32(HVX_సదిశ x) {
    స్థిరరాశి HVX_సదిశ scale = hvx_vec_splat_f32(0.7071067811865475f);
    స్థిరరాశి HVX_సదిశ half  = hvx_vec_splat_f32(0.5f);
    స్థిరరాశి HVX_సదిశ one   = hvx_vec_splat_f32(1.0f);
    స్థిరరాశి HVX_సదిశ max_x = hvx_vec_splat_f32(10.0f);
    స్థిరరాశి HVX_సదిశ min_x = hvx_vec_splat_f32(-10.0f);
    స్థిరరాశి HVX_సదిశ_Pred neg_large = Q6_Q_vcmp_gt_VsfVsf(min_x, x);
    స్థిరరాశి HVX_సదిశ_Pred pos_large = Q6_Q_vcmp_gt_VsfVsf(x, max_x);
    HVX_సదిశ x_calc = Q6_V_vmux_QVV(neg_large, min_x, x);
    x_calc = Q6_V_vmux_QVV(pos_large, max_x, x_calc);
    స్థిరరాశి HVX_సదిశ erf = hvx_vec_erf_f32(hvx_vec_mul_f32_f32(x_calc, scale));

    HVX_సదిశ ఫలితం = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x_calc), hvx_vec_కూడు_f32_f32(one, erf));

    ఫలితం = Q6_V_vmux_QVV(neg_large, hvx_vec_splat_f32(0.0f), ఫలితం);
    ఫలితం = Q6_V_vmux_QVV(pos_large, x, ఫలితం);
    తిరిగి ఫలితం;
}

స్థిర అంతఃవరుస శూన్య hvx_gelu_erf_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);

    HVX_సదిశ * పరిమిత vdst = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc = (HVX_సదిశ *) src;

    స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t epv       = 128 / elem_కొలత;
    స్థిరరాశి uint32_t nvec      = n / epv;
    స్థిరరాశి uint32_t nloe      = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_erf_f32(vsrc[i]);
    }
    ఒకవేళ (nloe) {
        HVX_సదిశ v = hvx_vec_gelu_erf_f32(vsrc[i]);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * elem_కొలత, v);
    }
}

#అయితే_ముగింపు /* HVX_ERF_H */
