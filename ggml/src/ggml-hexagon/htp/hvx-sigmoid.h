#నిర్వచితం_కాకపోతే HVX_SIGMOID_H
#నిర్వచించు HVX_SIGMOID_H

#అన్తర్భావ "hvx-base.h"
#అన్తర్భావ "hvx-inverse.h"
#అన్తర్భావ "hvx-exp.h"

#నిర్వచించు FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#నిర్వచించు FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#నిర్వచించు FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#నిర్వచించు FAST_SIGMOID_C3    (0x3f000000)  // 0.5

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_fast_sigmoid_f32(HVX_సదిశ v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_సదిశ లోపల_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_సదిశ x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(లోపల_int));
    HVX_సదిశ xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_సదిశ v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_సదిశ v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_సదిశ v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(లోపల_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, లోపల_int, 24);

    HVX_సదిశ v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_సదిశ v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_సదిశ res = hvx_vec_inverse_f32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    తిరిగి Q6_Vsf_equals_Vqf32(res);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_fast_sigmoid_f32_guard(HVX_సదిశ v,
                                                         HVX_సదిశ one,
                                                         HVX_సదిశ max_exp,
                                                         HVX_సదిశ min_exp) {
    స్థిరరాశి HVX_సదిశ_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    స్థిరరాశి HVX_సదిశ_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_సదిశ బహిర్గతం = hvx_vec_fast_sigmoid_f32(v);
    బహిర్గతం            = Q6_V_vmux_QVV(pred_max, బహిర్గతం, one);
    తిరిగి Q6_V_vmux_QVV(pred_min, బహిర్గతం, Q6_V_vzero());
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_tanh_f32(HVX_సదిశ x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_సదిశ two = hvx_vec_splat_f32(2.0f);
    HVX_సదిశ one = hvx_vec_splat_f32(1.0f);
    HVX_సదిశ x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    HVX_సదిశ max_exp = hvx_vec_splat_f32(87.f);
    HVX_సదిశ min_exp = hvx_vec_splat_f32(-87.f);

    HVX_సదిశ sig2x = hvx_vec_fast_sigmoid_f32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_సదిశ res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    తిరిగి Q6_Vsf_equals_Vqf32(res);
}

#నిర్వచించు hvx_sigmoid_loop_body(dst_type, src_type, vec_store)                                  \
    చేయి {                                                                                      \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                          \
        src_type * పరిమిత vsrc = (src_type *) src;                                          \
                                                                                              \
        స్థిరరాశి HVX_సదిశ one     = hvx_vec_splat_f32(1.f);                                    \
        స్థిరరాశి HVX_సదిశ max_exp = hvx_vec_splat_f32(87.f);                                   \
        స్థిరరాశి HVX_సదిశ min_exp = hvx_vec_splat_f32(-87.f);                                  \
                                                                                              \
        స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(దశాంశం);                                            \
        స్థిరరాశి uint32_t nvec = n / epv;                                                        \
        స్థిరరాశి uint32_t nloe = n % epv;                                                        \
                                                                                              \
        uint32_t i = 0;                                                                       \
                                                                                              \
        _Pragma("unroll(4)")                                                                  \
        కొరకు (; i < nvec; i++) {                                                               \
             vdst[i] = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp);        \
        }                                                                                     \
        ఒకవేళ (nloe) {                                                                           \
             HVX_సదిశ tmp = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp); \
             vec_store((శూన్య *) &vdst[i], nloe * పరిమాణం(దశాంశం), tmp);                         \
        }                                                                                     \
    } అయితే(0)

#నిర్వచించు hvx_tanh_loop_body(dst_type, src_type, vec_store)             \
    చేయి {                                                              \
        dst_type * పరిమిత vdst = (dst_type *) dst;                  \
        src_type * పరిమిత vsrc = (src_type *) src;                  \
                                                                      \
        స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(దశాంశం);                    \
        స్థిరరాశి uint32_t nvec = n / epv;                                \
        స్థిరరాశి uint32_t nloe = n % epv;                                \
                                                                      \
        uint32_t i = 0;                                               \
                                                                      \
        _Pragma("unroll(4)")                                          \
        కొరకు (; i < nvec; i++) {                                       \
             vdst[i] = hvx_vec_tanh_f32(vsrc[i]);                     \
        }                                                             \
        ఒకవేళ (nloe) {                                                   \
             HVX_సదిశ tmp = hvx_vec_tanh_f32(vsrc[i]);              \
             vec_store((శూన్య *) &vdst[i], nloe * పరిమాణం(దశాంశం), tmp); \
        }                                                             \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_sigmoid_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sigmoid_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_sigmoid_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_sigmoid_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_sigmoid_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_sigmoid_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_tanh_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_tanh_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_fast_sigmoid_f16(HVX_సదిశ x_v) {
    స్థిరరాశి HVX_సదిశ v_one       = hvx_vec_splat_f16(1.0f);
    స్థిరరాశి HVX_సదిశ v_neg_log2e = hvx_vec_splat_f16(-EXP_LOG2E_F);
    స్థిరరాశి HVX_సదిశ em_ముసుగు     = Q6_Vh_vsplat_R(0x7FFF);

    // Compute absolute value of x_v
    HVX_సదిశ abs_x = Q6_V_vand_VV(x_v, em_ముసుగు);

    // Compute u = -abs_x * log2(e) <= 0.
    HVX_సదిశ u = hvx_vec_mul_f16_f16(abs_x, v_neg_log2e);

    // Clamp input to prevent underflow in exp2
    స్థిరరాశి HVX_సదిశ v_clamp_min = hvx_vec_splat_f16(-24.0f);
    u = Q6_Vhf_vmax_VhfVhf(v_clamp_min, u);

    HVX_సదిశ exp_val = hvx_vec_exp2_f16(u);
    HVX_సదిశ denom   = hvx_vec_కూడు_f16_f16(v_one, exp_val);
    HVX_సదిశ sig_abs = hvx_vec_inverse_f16(denom);

    // check if x_v < 0 (using integer comparison on absolute value)
    HVX_సదిశ_Pred is_neg = Q6_Q_vcmp_gt_VhVh(abs_x, x_v);

    // If x_v < 0, return 1.0f - sig_abs
    HVX_సదిశ sig_neg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_one, sig_abs));
    తిరిగి Q6_V_vmux_QVV(is_neg, sig_neg, sig_abs);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_tanh_f16(HVX_సదిశ x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    స్థిరరాశి HVX_సదిశ v_two = hvx_vec_splat_f16(2.0f);

    HVX_సదిశ x2 = hvx_vec_mul_f16_f16(x, v_two);
    HVX_సదిశ sig2x = hvx_vec_fast_sigmoid_f16(x2);

    స్థిరరాశి HVX_సదిశ v_neg_one = hvx_vec_splat_f16(-1.0f);
    తిరిగి hvx_vec_కూడు_f16_f16(hvx_vec_mul_f16_f16(sig2x, v_two), v_neg_one);
}

#నిర్వచించు hvx_sigmoid_f16_loop_body(dst_type, src_type, vec_store)         \
    చేయి {                                                                 \
        dst_type * పరిమిత vdst = (dst_type *) dst;                     \
        src_type * పరిమిత vsrc = (src_type *) src;                     \
                                                                         \
        స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(_Float16);                    \
        స్థిరరాశి uint32_t nvec = n / epv;                                   \
        స్థిరరాశి uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        కొరకు (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_fast_sigmoid_f16(vsrc[i]);                \
        }                                                                \
        ఒకవేళ (nloe) {                                                      \
             HVX_సదిశ tmp = hvx_vec_fast_sigmoid_f16(vsrc[i]);         \
             vec_store((శూన్య *) &vdst[i], nloe * పరిమాణం(_Float16), tmp); \
        }                                                                \
    } అయితే(0)

#నిర్వచించు hvx_tanh_f16_loop_body(dst_type, src_type, vec_store)            \
    చేయి {                                                                 \
        dst_type * పరిమిత vdst = (dst_type *) dst;                     \
        src_type * పరిమిత vsrc = (src_type *) src;                     \
                                                                         \
        స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(_Float16);                    \
        స్థిరరాశి uint32_t nvec = n / epv;                                   \
        స్థిరరాశి uint32_t nloe = n % epv;                                   \
                                                                         \
        uint32_t i = 0;                                                  \
                                                                         \
        _Pragma("unroll(4)")                                             \
        కొరకు (; i < nvec; i++) {                                          \
             vdst[i] = hvx_vec_tanh_f16(vsrc[i]);                        \
        }                                                                \
        ఒకవేళ (nloe) {                                                      \
             HVX_సదిశ tmp = hvx_vec_tanh_f16(vsrc[i]);                 \
             vec_store((శూన్య *) &vdst[i], nloe * పరిమాణం(_Float16), tmp); \
        }                                                                \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_sigmoid_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_sigmoid_f16_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_tanh_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_tanh_f16_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

#అయితే_ముగింపు /* HVX_SIGMOID_H */
