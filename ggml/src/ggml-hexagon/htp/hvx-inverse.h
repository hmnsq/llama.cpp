#నిర్వచితం_కాకపోతే HVX_INVERSE_H
#నిర్వచించు HVX_INVERSE_H

#అన్తర్భావ <HAP_farf.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <assert.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ "hvx-base.h"

// ====================================================
// FUNCTION: 1/(x+1)     y(0) = 1,  y(0.5) = 0.6667, y(1) = 0.5
// Order:3; continuity: True; Ends forced: True
// Mode: unsigned;   Result fractional bits: 14
// Peak Error: 1.1295e-04  Rms Error: 2.8410e-05   Mean Error: 1.1370e-05
//      32769  -32706   31252  -10589
//      32590  -30635   22793   -4493
//      32066  -27505   16481   -2348
//      31205  -24054   11849   -1306

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_recip_xp1_O3_unsigned(HVX_సదిశ vx) {
    // input is 0..0xffff representing 0.0  .. 1.0
    HVX_సదిశ p;
    p = Q6_Vh_vlut4_VuhPh(vx, 0xFAE6F6D4EE73D6A3ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x2E49406159097A14ull);
    p = Q6_Vh_vmps_VhVhVuhPuh_sat(p, vx, 0x5DF66B7177AB7FC2ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x79E57D427F4E8001ull);
    తిరిగి p;  // signed result, 14 fractional bits
}

// Find reciprocal of fp16.
// (1) first, convert to fp32, multiplying by 1.0; this is done to
//    handle denormals. Ignoring sign and zero, result should be at
//    least 5.9604645e-08 (32-bit code 0x33800000) and at most 131008 (0x47ffe000)
//    (exponent in range [103,143])
// (2) extract the mantissa into 16-bit unsigned; find reciprocal using a fitted poly
// (3) put this, along with '253-exp' (exp from (1)) together to make an qf32
// (4) convert that to fp16
// (5) put sign back in. Also, if the original value (w/o sign) was <0x81, replace
//     the result with the max value.
స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_inverse_f16(HVX_సదిశ vals) {
    HVX_సదిశ     em_ముసుగు  = Q6_Vh_vsplat_R(0x7FFF);
    HVX_సదిశ     avals    = Q6_V_vand_VV(vals, em_ముసుగు);
    HVX_సదిశ_Pred is_neg   = Q6_Q_vcmp_gt_VhVh(avals, vals);
    // is too small to 1/x ? for 'standard' fp16, this would be 0x101
    HVX_సదిశ_Pred is_small = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0x101), avals);

    HVX_సదిశ_Pair గమ్యం_qf32  = Q6_Wqf32_vmpy_VhfVhf(avals, Q6_Vh_vsplat_R(0x3C00));  // *1.0
    HVX_సదిశ     గమ్యం_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(గమ్యం_qf32));
    HVX_సదిశ     గమ్యం_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(గమ్యం_qf32));

    // bits 22..13 contain the mantissa now (w/o hidden bit); move to bit 14..5 of a 16-bit vector
    HVX_సదిశ mant_u16 = Q6_Vh_vshuffo_VhVh(Q6_Vw_vasl_VwR(గమ్యం_f32_1, 9), Q6_Vw_vasl_VwR(గమ్యం_f32_0, 9));
    // likewise extract the upper 16 from each, containing the exponents in range 103..142
    HVX_సదిశ exp_u16  = Q6_Vh_vshuffo_VhVh(గమ్యం_f32_1, గమ్యం_f32_0);
    //Get exponent in IEEE 32-bit representation
    exp_u16             = Q6_Vuh_vlsr_VuhR(exp_u16, 7);

    // so, mant_u16 contains an unbiased mantissa in upper 10 bits of each u16 lane
    // We can consider it to be x-1.0, with 16 fractional bits, where 'x' is in range [1.0,2.0)
    // Use poly to transform to 1/x, with 14 fractional bits
    //
    HVX_సదిశ rm = hvx_vec_recip_xp1_O3_unsigned(mant_u16);

    HVX_సదిశ vcl0 = Q6_Vuh_vcl0_Vuh(rm);  //count leading zeros

    // Get mantissa for 16-bit representation
    HVX_సదిశ mant_recip = Q6_V_vand_VV(Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhVh(rm, vcl0), 5), Q6_Vh_vsplat_R(0x03FF));

    //Compute Reciprocal Exponent
    HVX_సదిశ exp_recip =
        Q6_Vh_vsub_VhVh(Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(254), exp_u16), Q6_Vh_vsub_VhVh(vcl0, Q6_Vh_vsplat_R(1)));
    //Convert it for 16-bit representation
    exp_recip = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vsub_VhVh(exp_recip, Q6_Vh_vsplat_R(127)), Q6_Vh_vsplat_R(15));
    exp_recip = Q6_Vh_vasl_VhR(exp_recip, 10);

    //Merge exponent and mantissa for reciprocal
    HVX_సదిశ recip = Q6_V_vor_VV(exp_recip, mant_recip);
    // map 'small' inputs to standard largest value 0x7bff
    recip            = Q6_V_vmux_QVV(is_small, Q6_Vh_vsplat_R(0x7bff), recip);
    // add sign back
    recip            = Q6_V_vandor_VQR(recip, is_neg, 0x80008000);
    తిరిగి recip;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_inverse_f32(HVX_సదిశ v_sf) {
    HVX_సదిశ inv_aprox_sf = Q6_V_vsplat_R(0x7EEEEBB3);
    HVX_సదిశ two_sf       = hvx_vec_splat_f32(2.0);

    // First approximation
    HVX_సదిశ i_sf = Q6_Vw_vsub_VwVw(inv_aprox_sf, v_sf);

    HVX_సదిశ r_qf;

    // Refine
    r_qf = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v_sf)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));

    తిరిగి Q6_Vsf_equals_Vqf32(r_qf);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_inverse_f32_guard(HVX_సదిశ v_sf, HVX_సదిశ nan_inf_ముసుగు) {
    HVX_సదిశ బహిర్గతం = hvx_vec_inverse_f32(v_sf);

    HVX_సదిశ     masked_బహిర్గతం = Q6_V_vand_VV(బహిర్గతం, nan_inf_ముసుగు);
    స్థిరరాశి HVX_సదిశ_Pred pred = Q6_Q_vcmp_eq_VwVw(nan_inf_ముసుగు, masked_బహిర్గతం);

    తిరిగి Q6_V_vmux_QVV(pred, Q6_V_vzero(), బహిర్గతం);
}

#నిర్వచించు hvx_inverse_f32_loop_body(dst_type, src_type, vec_store)             \
    చేయి {                                                                     \
        dst_type * పరిమిత vdst = (dst_type *) dst;                         \
        src_type * పరిమిత vsrc = (src_type *) src;                         \
                                                                             \
        స్థిరరాశి HVX_సదిశ nan_inf_ముసుగు = Q6_V_vsplat_R(0x7f800000);           \
                                                                             \
        స్థిరరాశి uint32_t nvec = n / VLEN_FP32;                                 \
        స్థిరరాశి uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        కొరకు (; i < nvec; i++) {                                              \
             vdst[i] = hvx_vec_inverse_f32_guard(vsrc[i], nan_inf_ముసుగు);     \
        }                                                                    \
        ఒకవేళ (nloe) {                                                          \
            HVX_సదిశ v = hvx_vec_inverse_f32_guard(vsrc[i], nan_inf_ముసుగు); \
            vec_store((శూన్య *) &vdst[i], nloe * SIZEOF_FP32, v);             \
        }                                                                    \
    } అయితే(0)

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_inverse_f16_guard(HVX_సదిశ v_sf, HVX_సదిశ nan_inf_ముసుగు) {
    HVX_సదిశ బహిర్గతం = hvx_vec_inverse_f16(v_sf);

    HVX_సదిశ     masked_బహిర్గతం = Q6_V_vand_VV(బహిర్గతం, nan_inf_ముసుగు);
    స్థిరరాశి HVX_సదిశ_Pred pred = Q6_Q_vcmp_eq_VhVh(nan_inf_ముసుగు, masked_బహిర్గతం);

    తిరిగి Q6_V_vmux_QVV(pred, Q6_V_vzero(), బహిర్గతం);
}

#నిర్వచించు hvx_inverse_f16_loop_body(dst_type, src_type, vec_store)             \
    చేయి {                                                                     \
        dst_type * పరిమిత vdst = (dst_type *) dst;                         \
        src_type * పరిమిత vsrc = (src_type *) src;                         \
                                                                             \
        స్థిరరాశి HVX_సదిశ nan_inf_ముసుగు = Q6_Vh_vsplat_R(0x7c00);              \
                                                                             \
        స్థిరరాశి uint32_t nvec = n / VLEN_FP16;                                 \
        స్థిరరాశి uint32_t nloe = n % VLEN_FP16;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        కొరకు (; i < nvec; i++) {                                              \
             vdst[i] = hvx_vec_inverse_f16_guard(vsrc[i], nan_inf_ముసుగు);     \
        }                                                                    \
        ఒకవేళ (nloe) {                                                          \
            HVX_సదిశ v = hvx_vec_inverse_f16_guard(vsrc[i], nan_inf_ముసుగు); \
            vec_store((శూన్య *) &vdst[i], nloe * SIZEOF_FP16, v);             \
        }                                                                    \
    } అయితే(0)

// Generic macro to define alignment permutations for an op
#నిర్వచించు DEFINE_HVX_INV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY)                                           \
స్థిర అంతఃవరుస శూన్య OP_NAME##_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                             \
    ప్రతిజ్ఞా((uintptr_t) src % 128 == 0);                                                             \
    OP_LOOP_BODY(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);                                          \
}                                                                                                   \
స్థిర అంతఃవరుస శూన్య OP_NAME##_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) dst % 128 == 0);                                                             \
    OP_LOOP_BODY(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);                                         \
}                                                                                                   \
స్థిర అంతఃవరుస శూన్య OP_NAME##_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) { \
    ప్రతిజ్ఞా((uintptr_t) src % 128 == 0);                                                             \
    OP_LOOP_BODY(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);                                         \
}                                                                                                   \
స్థిర అంతఃవరుస శూన్య OP_NAME##_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, hvx_vec_store_u);                                        \
}                                                                                                   \

// Dispatcher logic
#నిర్వచించు HVX_INV_DISPATCHER(OP_NAME)                                                                          \
స్థిర అంతఃవరుస శూన్య OP_NAME(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి uint32_t అంకె_elems) { \
    ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128) && hex_is_aligned((శూన్య *) src, 128)) {                            \
        OP_NAME##_aa(dst, src, అంకె_elems);                                                                   \
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) dst, 128)) {                                                          \
        OP_NAME##_au(dst, src, అంకె_elems);                                                                   \
    } లేకపోతే ఒకవేళ (hex_is_aligned((శూన్య *) src, 128)) {                                                          \
        OP_NAME##_ua(dst, src, అంకె_elems);                                                                   \
    } లేకపోతే {                                                                                                 \
        OP_NAME##_uu(dst, src, అంకె_elems);                                                                   \
    }                                                                                                        \
}

DEFINE_HVX_INV_OP_VARIANTS(hvx_inverse_f32, hvx_inverse_f32_loop_body)
DEFINE_HVX_INV_OP_VARIANTS(hvx_inverse_f16, hvx_inverse_f16_loop_body)

HVX_INV_DISPATCHER(hvx_inverse_f32)
HVX_INV_DISPATCHER(hvx_inverse_f16)

#అయితే_ముగింపు // HVX_INVERSE_H
