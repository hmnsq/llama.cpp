// Dynamic quantizers that produce tiled activations

// q8_1: vector 9 holds d * sum(q) of the 32 k, replicated.
// q8_1_s16 (Q2_K, sums16): one fp16 d (vector 8) for the 256 k super-block starting at x_sb, and vector 9 holds the
// integer sums of k 0..15 (low halfword) and k 16..31 (high halfword) in every word
__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య quantize_ఖండం_f32_q8_1_tiled_impl(దశాంశం * పరిమిత x, uint8_t * పరిమిత y_ఖండం, స్థిరరాశి bool sums16,
                                                      స్థిరరాశి దశాంశం * పరిమిత x_sb) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) x % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) y_ఖండం % 128 == 0);

    HVX_సదిశ * vx = (HVX_సదిశ *) x;
    HVX_సదిశ zero = Q6_V_vzero();

    HVX_సదిశ vmax0_sf, vmax1_sf, vmax2_sf, vmax3_sf;
    ఒకవేళ (sums16) {
        స్థిరరాశి HVX_సదిశ * vsb = (స్థిరరాశి HVX_సదిశ *) x_sb;
        HVX_సదిశ vm = hvx_vec_abs_f32(vsb[0]);
        కొరకు (పూర్ణాంకం i = 1; i < 8; i++) {
            vm = Q6_Vsf_vmax_VsfVsf(vm, hvx_vec_abs_f32(vsb[i]));
        }
        vmax0_sf = vmax1_sf = vmax2_sf = vmax3_sf = hvx_vec_reduce_max_f32(vm);
    } లేకపోతే {
        vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
        vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
        vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
        vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));
    }

    HVX_సదిశ vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_సదిశ vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_సదిశ vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_సదిశ vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);

    HVX_సదిశ vmax01_hf   = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_సదిశ vx01_hf     = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_సదిశ vd01_qf16   = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_సదిశ vd01_hf     = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_సదిశ vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));

    HVX_సదిశ vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_సదిశ vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);
    HVX_సదిశ vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_సదిశ vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_సదిశ vmax23_hf   = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));
    HVX_సదిశ vx23_hf     = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));
    HVX_సదిశ vd23_qf16   = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_సదిశ vd23_hf     = Q6_Vhf_equals_Vqf16(vd23_qf16);
    HVX_సదిశ vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_సదిశ vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_సదిశ vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_సదిశ vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    స్థిరరాశి HVX_సదిశ ones = Q6_Vb_vsplat_R(1);
    HVX_సదిశ v_sums = Q6_Vw_vrmpy_VbVb(vx_i8, ones);
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 4));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 8));
    ఒకవేళ (!sums16) {
        v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 16));
    }
    // word 8b: sum of block b (sums16: k 0..15, and word 8b + 4: k 16..31)

    స్థిరరాశి HVX_సదిశ v_inv127 = hvx_vec_splat_f32(1.0f / 127.0f);
    HVX_సదిశ vd0_sf = hvx_vec_mul_f32_f32(vmax0_sf, v_inv127);
    HVX_సదిశ vd1_sf = hvx_vec_mul_f32_f32(vmax1_sf, v_inv127);
    HVX_సదిశ vd2_sf = hvx_vec_mul_f32_f32(vmax2_sf, v_inv127);
    HVX_సదిశ vd3_sf = hvx_vec_mul_f32_f32(vmax3_sf, v_inv127);

    HVX_సదిశ v_sums_sf = Q6_Vsf_equals_Vw(v_sums);
    HVX_సదిశ voff0_sf = hvx_vec_mul_f32_f32(vd0_sf, v_sums_sf);
    HVX_సదిశ voff1_sf = hvx_vec_mul_f32_f32(vd1_sf, Q6_V_vror_VR(v_sums_sf, 32));
    HVX_సదిశ voff2_sf = hvx_vec_mul_f32_f32(vd2_sf, Q6_V_vror_VR(v_sums_sf, 64));
    HVX_సదిశ voff3_sf = hvx_vec_mul_f32_f32(vd3_sf, Q6_V_vror_VR(v_sums_sf, 96));

    HVX_సదిశ voff01_hf = hvx_vec_f32_గమ్యం_f16(voff0_sf, voff1_sf);
    HVX_సదిశ voff23_hf = hvx_vec_f32_గమ్యం_f16(voff2_sf, voff3_sf);

    int32_t __attribute__((aligned(128))) ws[32];
    ఒకవేళ (sums16) {
        // |sum of 16 int8| <= 2032 fits a halfword
        *(HVX_సదిశ *) ws = v_sums;
    }

    స్థిర స్థిరరాశి uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_సదిశ v_repl_ctrl = * (స్థిరరాశి HVX_సదిశ *) repl;

    #సూచన unroll
    కొరకు (పూర్ణాంకం b = 0; b < 4; b++) {
        HVX_సదిశ v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_సదిశ vd_b   = (b < 2) ? vd01_hf   : vd23_hf;
        HVX_సదిశ voff_b = (b < 2) ? voff01_hf : voff23_hf;
        HVX_సదిశ r_scale  = hvx_vec_repl_f16((b & 1) ? Q6_V_vror_VR(vd_b, 64)   : vd_b);
        HVX_సదిశ r_స్థానభ్రంశం = sums16
            ? Q6_V_vsplat_R(((uint32_t) ws[8 * b + 4] << 16) | ((uint32_t) ws[8 * b] & 0xFFFF))
            : hvx_vec_repl_f16((b & 1) ? Q6_V_vror_VR(voff_b, 64) : voff_b);

        HVX_సదిశ r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_సదిశ r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_సదిశ r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_సదిశ r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_సదిశ r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_సదిశ r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_సదిశ r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_సదిశ r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_సదిశ * పరిమిత dst = (HVX_సదిశ *) (y_ఖండం + b * 1280);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
        dst[9] = r_స్థానభ్రంశం;
    }
}

స్థిర అంతఃవరుస శూన్య quantize_ఖండం_f32_q8_1_tiled(దశాంశం * పరిమిత x, uint8_t * పరిమిత y_ఖండం) {
    quantize_ఖండం_f32_q8_1_tiled_impl(x, y_ఖండం, false, ఖాళీ);
}

స్థిర అంతఃవరుస శూన్య quantize_ఖండం_f32_q8_1_s16_tiled(దశాంశం * పరిమిత x, uint8_t * పరిమిత y_ఖండం, స్థిరరాశి దశాంశం * పరిమిత x_sb) {
    quantize_ఖండం_f32_q8_1_tiled_impl(x, y_ఖండం, true, x_sb);
}

స్థిర అంతఃవరుస శూన్య quantize_ఖండం_f32_q8_0_tiled(దశాంశం * పరిమిత x, uint8_t * పరిమిత y_ఖండం) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) x % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) y_ఖండం % 128 == 0);

    HVX_సదిశ * vx = (HVX_సదిశ *) x;
    HVX_సదిశ zero   = Q6_V_vzero();

    HVX_సదిశ vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_సదిశ vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_సదిశ vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_సదిశ vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_సదిశ vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_సదిశ vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_సదిశ vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_సదిశ vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_సదిశ vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_సదిశ vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_సదిశ vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_సదిశ vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_సదిశ vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_సదిశ vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_సదిశ vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_సదిశ vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_సదిశ vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_సదిశ vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_సదిశ vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_సదిశ vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_సదిశ vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_సదిశ vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_సదిశ vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_సదిశ vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_సదిశ vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    HVX_సదిశ_Pair vp01 = Q6_W_vshuff_VVR(vd01_hf, vd01_hf, -64);
    HVX_సదిశ_Pair vp23 = Q6_W_vshuff_VVR(vd23_hf, vd23_hf, -64);

    స్థిర స్థిరరాశి uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_సదిశ v_repl_ctrl = * (స్థిరరాశి HVX_సదిశ *) repl;

    #సూచన unroll
    కొరకు (పూర్ణాంకం b = 0; b < 4; b++) {
        HVX_సదిశ v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_సదిశ r_scale;
        ఒకవేళ (b == 0) {
            r_scale = Q6_V_lo_W(vp01);
        } లేకపోతే ఒకవేళ (b == 1) {
            r_scale = Q6_V_hi_W(vp01);
        } లేకపోతే ఒకవేళ (b == 2) {
            r_scale = Q6_V_lo_W(vp23);
        } లేకపోతే {
            r_scale = Q6_V_hi_W(vp23);
        }

        HVX_సదిశ r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_సదిశ r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_సదిశ r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_సదిశ r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_సదిశ r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_సదిశ r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_సదిశ r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_సదిశ r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_సదిశ * పరిమిత dst = (HVX_సదిశ *) (y_ఖండం + b * 1152);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
    }
}

స్థిర శూన్య quantize_అడ్డువరుస_f32_q8_0_tiled(దశాంశం * పరిమిత x, uint8_t * పరిమిత y, uint32_t k) {
    ప్రతిజ్ఞా(k % 32 == 0);
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (k + qk - 1) / qk;

    కొరకు (uint32_t i = 0; i < nb; i++) {
        uint8_t * పరిమిత y_ఖండం = y + i * 4 * 1152;
        quantize_ఖండం_f32_q8_0_tiled(x + i * qk, y_ఖండం);
    }
}

స్థిర శూన్య quantize_అడ్డువరుస_f32_q8_1_tiled(దశాంశం * పరిమిత x, uint8_t * పరిమిత y, uint32_t k) {
    ప్రతిజ్ఞా(k % 32 == 0);
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (k + qk - 1) / qk;

    కొరకు (uint32_t i = 0; i < nb; i++) {
        uint8_t * పరిమిత y_ఖండం = y + i * 4 * 1280;
        quantize_ఖండం_f32_q8_1_tiled(x + i * qk, y_ఖండం);
    }
}

స్థిర శూన్య quantize_అడ్డువరుస_f32_q8_1_s16_tiled(దశాంశం * పరిమిత x, uint8_t * పరిమిత y, uint32_t k) {
    ప్రతిజ్ఞా(k % 32 == 0);
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (k + qk - 1) / qk;

    కొరకు (uint32_t i = 0; i < nb; i++) {
        uint8_t * పరిమిత y_ఖండం = y + i * 4 * 1280;
        quantize_ఖండం_f32_q8_1_s16_tiled(x + i * qk, y_ఖండం, x + (i & ~1u) * qk);  // k % 256 == 0 for Q2_K
    }
}

స్థిర అంతఃవరుస శూన్య quantize_half_f16_tiled_splat(HVX_సదిశ v_src, HVX_సదిశ * పరిమిత dst) {
    స్థిర స్థిరరాశి uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_సదిశ v_repl_ctrl = *(స్థిరరాశి HVX_సదిశ *) repl;

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం cp = 0; cp < 32; cp++) {
        dst[cp] = Q6_V_vdelta_VV(cp == 0 ? v_src : Q6_V_vror_VR(v_src, cp * 4), v_repl_ctrl);
    }
}

స్థిర అంతఃవరుస శూన్య quantize_ఖండం_f32_f16_tiled(స్థిరరాశి దశాంశం * పరిమిత x, uint8_t * పరిమిత y_ఖండం) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) x % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) y_ఖండం % 128 == 0);

    స్థిరరాశి HVX_సదిశ * vx = (స్థిరరాశి HVX_సదిశ *) x;
    HVX_సదిశ * పరిమిత dst = (HVX_సదిశ *) y_ఖండం;

    HVX_సదిశ vx01_hf = hvx_vec_f32_గమ్యం_f16(vx[0], vx[1]);
    quantize_half_f16_tiled_splat(vx01_hf, dst);

    HVX_సదిశ vx23_hf = hvx_vec_f32_గమ్యం_f16(vx[2], vx[3]);
    quantize_half_f16_tiled_splat(vx23_hf, dst + 32);
}

స్థిర అంతఃవరుస శూన్య quantize_ఖండం_f16_f16_tiled(స్థిరరాశి __fp16 * పరిమిత x, uint8_t * పరిమిత y_ఖండం) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) x % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) y_ఖండం % 128 == 0);

    స్థిరరాశి HVX_సదిశ * vx = (స్థిరరాశి HVX_సదిశ *) x;
    HVX_సదిశ * పరిమిత dst = (HVX_సదిశ *) y_ఖండం;

    quantize_half_f16_tiled_splat(vx[0], dst);
    quantize_half_f16_tiled_splat(vx[1], dst + 32);
}

స్థిర శూన్య quantize_అడ్డువరుస_f32_f16_tiled(స్థిరరాశి దశాంశం * పరిమిత x, uint8_t * పరిమిత y, uint32_t k) {
    ప్రతిజ్ఞా(k % 32 == 0);
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (k + qk - 1) / qk;

    కొరకు (uint32_t i = 0; i < nb; i++) {
        uint8_t * పరిమిత y_ఖండం = y + i * 4 * 2048;
        quantize_ఖండం_f32_f16_tiled(x + i * qk, y_ఖండం);
    }
}

స్థిర శూన్య quantize_అడ్డువరుస_f16_f16_tiled(స్థిరరాశి __fp16 * పరిమిత x, uint8_t * పరిమిత y, uint32_t k) {
    ప్రతిజ్ఞా(k % 32 == 0);
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (k + qk - 1) / qk;

    కొరకు (uint32_t i = 0; i < nb; i++) {
        uint8_t * పరిమిత y_ఖండం = y + i * 4 * 2048;
        quantize_ఖండం_f16_f16_tiled(x + i * qk, y_ఖండం);
    }
}

// Dot kernels & helpers that consume tiled activations

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(HVX_సదిశ v1, HVX_సదిశ v2) {
#ఒకవేళ __HVX_ARCH__ >= 79
    HVX_సదిశ_Pair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    తిరిగి Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4));
#లేకపోతే
    HVX_సదిశ_Pair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_సదిశ hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_సదిశ lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    తిరిగి Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -4));
#అయితే_ముగింపు
}

// both halves of hvx_vec_mul_f16_f16_to_f32_lower32: lo = products of lanes 0..31, hi = lanes 32..63
స్థిర అంతఃవరుస HVX_సదిశ_Pair hvx_vec_mul_f16_f16_గమ్యం_f32_pair(HVX_సదిశ v1, HVX_సదిశ v2) {
#ఒకవేళ __HVX_ARCH__ >= 79
    HVX_సదిశ_Pair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    తిరిగి Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4);
#లేకపోతే
    HVX_సదిశ_Pair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_సదిశ hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_సదిశ lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    తిరిగి Q6_W_vshuff_VVR(hi, lo, -4);
#అయితే_ముగింపు
}

స్థిర అంతఃవరుస HVX_సదిశ unpack_and_interleave_4bit(HVX_సదిశ v_a, HVX_సదిశ v_b, HVX_సదిశ ముసుగు_h4) {
    HVX_సదిశ v_W0 = Q6_V_vand_VV(v_a, ముసుగు_h4);
    HVX_సదిశ v_W1 = Q6_Vub_vlsr_VubR(v_a, 4);
    HVX_సదిశ v_W2 = Q6_V_vand_VV(v_b, ముసుగు_h4);
    HVX_సదిశ v_W3 = Q6_Vub_vlsr_VubR(v_b, 4);

    HVX_సదిశ_Pair v01_pair = Q6_W_vshuff_VVR(v_W1, v_W0, -1);
    HVX_సదిశ_Pair v23_pair = Q6_W_vshuff_VVR(v_W3, v_W2, -1);
    HVX_సదిశ_Pair v0123_pair = Q6_W_vshuff_VVR(Q6_V_lo_W(v23_pair), Q6_V_lo_W(v01_pair), -2);
    తిరిగి Q6_V_lo_W(v0123_pair);
}

స్థిర అంతఃవరుస HVX_సదిశ_Pair unpack_and_interleave_4bit_x2(HVX_సదిశ v_src, HVX_సదిశ ముసుగు_h4) {
    HVX_సదిశ v_lo = Q6_V_vand_VV(v_src, ముసుగు_h4);
    HVX_సదిశ v_hi = Q6_Vub_vlsr_VubR(v_src, 4);
    HVX_సదిశ_Pair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_సదిశ v01_lo = Q6_V_lo_W(v01_pair);
    HVX_సదిశ v01_hi = Q6_V_hi_W(v01_pair);

    HVX_సదిశ v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_సదిశ v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_సదిశ v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_సదిశ v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    తిరిగి Q6_W_vcombine_VV(v_W1, v_W0);
}

స్థిర అంతఃవరుస HVX_సదిశ unpack_q1_0_సమూహం(
    HVX_సదిశ v_plane,
    పూర్ణాంకం g,
    HVX_సదిశ v_neg_one,
    HVX_సదిశ v_pos_one
) {
    HVX_సదిశ_Pred q = Q6_Q_vand_VR(v_plane, 0x01010101u << g);
    తిరిగి Q6_V_vmux_QVV(q, v_pos_one, v_neg_one);
}

స్థిర అంతఃవరుస HVX_సదిశ accum_q1_0_32x1(
    HVX_సదిశ v_plane,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act,
    HVX_సదిశ v_neg_one,
    HVX_సదిశ v_pos_one
) {
    HVX_సదిశ v_కూడిక_lo = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_hi = Q6_V_vzero();

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q1_0_సమూహం(v_plane, g,     v_neg_one, v_pos_one);
        HVX_సదిశ v_W_hi = unpack_q1_0_సమూహం(v_plane, g + 4, v_neg_one, v_pos_one);
        v_కూడిక_lo = Q6_Vw_vrmpyacc_VwVbVb(v_కూడిక_lo, v_W_lo, v_act[g]);
        v_కూడిక_hi = Q6_Vw_vrmpyacc_VwVbVb(v_కూడిక_hi, v_W_hi, v_act[g + 4]);
    }

    తిరిగి Q6_Vw_vadd_VwVw(v_కూడిక_lo, v_కూడిక_hi);
}

స్థిర అంతఃవరుస HVX_సదిశ accum_4bit_32x1(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act,
    HVX_సదిశ i8
) {
    HVX_సదిశ v_sum0 = Q6_V_vzero();
    HVX_సదిశ v_sum1 = Q6_V_vzero();
    HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);

    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        HVX_సదిశ_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], ముసుగు_h4);
        HVX_సదిశ v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_సదిశ v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    తిరిగి Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

స్థిర అంతఃవరుస HVX_సదిశ accum_4bit_32x1_lut(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act,
    HVX_సదిశ ముసుగు_h4,
    HVX_సదిశ lut
) {
    HVX_సదిశ v_sum0 = Q6_V_vzero();
    HVX_సదిశ v_sum1 = Q6_V_vzero();

    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        HVX_సదిశ_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], ముసుగు_h4);
        HVX_సదిశ v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_సదిశ v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    తిరిగి Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

స్థిర అంతఃవరుస HVX_సదిశ_Pair accum_q1_0_32x2(
    HVX_సదిశ v_plane,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1,
    HVX_సదిశ v_neg_one,
    HVX_సదిశ v_pos_one
) {
    HVX_సదిశ v_sum0_lo = Q6_V_vzero();
    HVX_సదిశ v_sum0_hi = Q6_V_vzero();
    HVX_సదిశ v_sum1_lo = Q6_V_vzero();
    HVX_సదిశ v_sum1_hi = Q6_V_vzero();

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q1_0_సమూహం(v_plane, g,     v_neg_one, v_pos_one);
        HVX_సదిశ v_W_hi = unpack_q1_0_సమూహం(v_plane, g + 4, v_neg_one, v_pos_one);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    తిరిగి Q6_W_vcombine_VV(
        Q6_Vw_vadd_VwVw(v_sum1_lo, v_sum1_hi),
        Q6_Vw_vadd_VwVw(v_sum0_lo, v_sum0_hi));
}

స్థిర అంతఃవరుస HVX_సదిశ_Pair accum_4bit_32x2(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1,
    HVX_సదిశ i8
) {
    HVX_సదిశ v_sum0 = Q6_V_vzero();
    HVX_సదిశ v_sum1 = Q6_V_vzero();
    HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);

    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        HVX_సదిశ_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], ముసుగు_h4);
        HVX_సదిశ v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_సదిశ v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    తిరిగి Q6_W_vcombine_VV(v_sum1, v_sum0);
}

స్థిర అంతఃవరుస HVX_సదిశ_Pair accum_4bit_32x2_lut(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1,
    HVX_సదిశ ముసుగు_h4,
    HVX_సదిశ lut
) {
    HVX_సదిశ v_sum0 = Q6_V_vzero();
    HVX_సదిశ v_sum1 = Q6_V_vzero();

    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        HVX_సదిశ_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], ముసుగు_h4);
        HVX_సదిశ v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_సదిశ v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    తిరిగి Q6_W_vcombine_VV(v_sum1, v_sum0);
}

స్థిర అంతఃవరుస HVX_సదిశ accum_q8_0_32x1(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act
) {
    HVX_సదిశ v_కూడిక = Q6_V_vzero();
    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 8; g++) {
        HVX_సదిశ v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_సదిశ v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_కూడిక = Q6_Vw_vrmpyacc_VwVbVb(v_కూడిక, v_W, v_act[g]);
    }
    తిరిగి v_కూడిక;
}

స్థిర అంతఃవరుస HVX_సదిశ_Pair accum_q8_0_32x2(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1
) {
    HVX_సదిశ v_sum0 = Q6_V_vzero();
    HVX_సదిశ v_sum1 = Q6_V_vzero();
    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 8; g++) {
        HVX_సదిశ v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_సదిశ v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W, v_act0[g]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W, v_act1[g]);
    }
    తిరిగి Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q5_K: OR 0x10 into every lane of v whose flag j is set in the plane (see HTP_TENSOR_TILE_SIZE_Q5_K)
స్థిర అంతఃవరుస HVX_సదిశ hvx_q5k_or_hibit(HVX_సదిశ v, HVX_సదిశ v_plane, పూర్ణాంకం j) {
    HVX_సదిశ_Pred q = Q6_Q_vand_VR(v_plane, 0x01010101u << j);
    తిరిగి Q6_V_vandor_VQR(v, q, 0x10101010);
}

// 5-bit variant: the high bit comes from the plane, see hvx_q5k_or_hibit
స్థిర అంతఃవరుస HVX_సదిశ_Pair unpack_and_interleave_5bit_x2(HVX_సదిశ v_src, HVX_సదిశ v_plane, పూర్ణాంకం i, HVX_సదిశ ముసుగు_h4) {
    HVX_సదిశ v_lo = hvx_q5k_or_hibit(Q6_V_vand_VV(v_src, ముసుగు_h4), v_plane, 2 * i);
    HVX_సదిశ v_hi = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(v_src, 4), v_plane, 2 * i + 1);
    HVX_సదిశ_Pair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_సదిశ v01_lo = Q6_V_lo_W(v01_pair);
    HVX_సదిశ v01_hi = Q6_V_hi_W(v01_pair);

    HVX_సదిశ v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_సదిశ v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_సదిశ v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_సదిశ v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    తిరిగి Q6_W_vcombine_VV(v_W1, v_W0);
}

స్థిర అంతఃవరుస HVX_సదిశ accum_5bit_32x1(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act,
    HVX_సదిశ i8
) {
    HVX_సదిశ v_sum0 = Q6_V_vzero();
    HVX_సదిశ v_sum1 = Q6_V_vzero();
    HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_సదిశ v_plane = vptr[5];

    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        HVX_సదిశ_Pair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, ముసుగు_h4);
        HVX_సదిశ v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_సదిశ v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    తిరిగి Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

స్థిర అంతఃవరుస HVX_సదిశ_Pair accum_5bit_32x2(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1,
    HVX_సదిశ i8
) {
    HVX_సదిశ v_sum0 = Q6_V_vzero();
    HVX_సదిశ v_sum1 = Q6_V_vzero();
    HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_సదిశ v_plane = vptr[5];

    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        HVX_సదిశ_Pair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, ముసుగు_h4);
        HVX_సదిశ v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_సదిశ v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    తిరిగి Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q6_K weights are stored unsigned (0..63), see HTP_TENSOR_TILE_SIZE_Q6_K. Unpack k-group g of a tile to signed bytes (q - 32)
స్థిర అంతఃవరుస HVX_సదిశ unpack_q6_k_సమూహం(స్థిరరాశి HVX_సదిశ * పరిమిత vptr, పూర్ణాంకం g, HVX_సదిశ ముసుగు_0f, HVX_సదిశ ముసుగు_03, HVX_సదిశ i32) {
    HVX_సదిశ v_lo = (g & 1) ? Q6_Vub_vlsr_VubR(vptr[g >> 1], 4) : Q6_V_vand_VV(vptr[g >> 1], ముసుగు_0f);
    HVX_సదిశ v_hi = (g & 3) ? Q6_Vub_vlsr_VubR(vptr[4 + (g >> 2)], 2 * (g & 3)) : vptr[4 + (g >> 2)];
    HVX_సదిశ v_q  = Q6_V_vor_VV(v_lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_hi, ముసుగు_03), 4));
    తిరిగి Q6_Vb_vsub_VbVb(v_q, i32);
}

// k 0..15 and k 16..31 of a Q6_K tile have different scales: lo half of the pair sums k 0..15, hi half sums k 16..31
స్థిర అంతఃవరుస HVX_సదిశ_Pair accum_q6_k_32x1(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act,
    HVX_సదిశ i32
) {
    HVX_సదిశ v_కూడిక_lo = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_hi = Q6_V_vzero();
    HVX_సదిశ ముసుగు_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_సదిశ ముసుగు_03 = Q6_Vb_vsplat_R(0x03);

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q6_k_సమూహం(vptr, g,     ముసుగు_0f, ముసుగు_03, i32);
        HVX_సదిశ v_W_hi = unpack_q6_k_సమూహం(vptr, g + 4, ముసుగు_0f, ముసుగు_03, i32);
        v_కూడిక_lo = Q6_Vw_vrmpyacc_VwVbVb(v_కూడిక_lo, v_W_lo, v_act[g]);
        v_కూడిక_hi = Q6_Vw_vrmpyacc_VwVbVb(v_కూడిక_hi, v_W_hi, v_act[g + 4]);
    }

    తిరిగి Q6_W_vcombine_VV(v_కూడిక_hi, v_కూడిక_lo);
}

స్థిర అంతఃవరుస శూన్య accum_q6_k_32x2(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1,
    HVX_సదిశ i32,
    HVX_సదిశ_Pair * v_sums0,
    HVX_సదిశ_Pair * v_sums1
) {
    HVX_సదిశ v_sum0_lo = Q6_V_vzero();
    HVX_సదిశ v_sum0_hi = Q6_V_vzero();
    HVX_సదిశ v_sum1_lo = Q6_V_vzero();
    HVX_సదిశ v_sum1_hi = Q6_V_vzero();
    HVX_సదిశ ముసుగు_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_సదిశ ముసుగు_03 = Q6_Vb_vsplat_R(0x03);

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q6_k_సమూహం(vptr, g,     ముసుగు_0f, ముసుగు_03, i32);
        HVX_సదిశ v_W_hi = unpack_q6_k_సమూహం(vptr, g + 4, ముసుగు_0f, ముసుగు_03, i32);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// scale the two half sums with the per-row tile scales (v_scale_w = vptr[6]) and the activation scale
స్థిర అంతఃవరుస HVX_సదిశ scale_q6_k_32x1(HVX_సదిశ_Pair v_sums, HVX_సదిశ v_scale_w, HVX_సదిశ v_scale_a) {
    HVX_సదిశ_Pair v_scale = hvx_vec_mul_f16_f16_గమ్యం_f32_pair(v_scale_w, v_scale_a);
    HVX_సదిశ v_lo = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums)), Q6_V_lo_W(v_scale));
    HVX_సదిశ v_hi = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums)), Q6_V_hi_W(v_scale));
    తిరిగి hvx_vec_కూడు_f32_f32(v_lo, v_hi);
}

// Q3_K / Q2_K: low 2 bits of k-group g, see HTP_TENSOR_TILE_SIZE_Q3_K
స్థిర అంతఃవరుస HVX_సదిశ unpack_q3_k_low2(స్థిరరాశి HVX_సదిశ * పరిమిత vptr, పూర్ణాంకం g, HVX_సదిశ ముసుగు_03) {
    HVX_సదిశ v = vptr[g >> 2];
    ఒకవేళ ((g & 3) == 3) {
        తిరిగి Q6_Vub_vlsr_VubR(v, 6);
    }
    ఒకవేళ (g & 3) {
        v = Q6_Vub_vlsr_VubR(v, 2 * (g & 3));
    }
    తిరిగి Q6_V_vand_VV(v, ముసుగు_03);
}

// Q3_K k-group g as signed bytes: low2 | 0xFC (= low2 - 4) where bit g of vector 2 is set
స్థిర అంతఃవరుస HVX_సదిశ unpack_q3_k_సమూహం(స్థిరరాశి HVX_సదిశ * పరిమిత vptr, పూర్ణాంకం g, HVX_సదిశ ముసుగు_03) {
    HVX_సదిశ_Pred q_neg = Q6_Q_vand_VR(vptr[2], 0x01010101u << g);
    తిరిగి Q6_V_vandor_VQR(unpack_q3_k_low2(vptr, g, ముసుగు_03), q_neg, 0xFCFCFCFC);
}

// same half split as accum_q6_k_32x1
స్థిర అంతఃవరుస HVX_సదిశ_Pair accum_q3_k_32x1(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act
) {
    HVX_సదిశ v_కూడిక_lo = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_hi = Q6_V_vzero();
    HVX_సదిశ ముసుగు_03  = Q6_Vb_vsplat_R(0x03);

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q3_k_సమూహం(vptr, g,     ముసుగు_03);
        HVX_సదిశ v_W_hi = unpack_q3_k_సమూహం(vptr, g + 4, ముసుగు_03);
        v_కూడిక_lo = Q6_Vw_vrmpyacc_VwVbVb(v_కూడిక_lo, v_W_lo, v_act[g]);
        v_కూడిక_hi = Q6_Vw_vrmpyacc_VwVbVb(v_కూడిక_hi, v_W_hi, v_act[g + 4]);
    }

    తిరిగి Q6_W_vcombine_VV(v_కూడిక_hi, v_కూడిక_lo);
}

స్థిర అంతఃవరుస శూన్య accum_q3_k_32x2(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1,
    HVX_సదిశ_Pair * v_sums0,
    HVX_సదిశ_Pair * v_sums1
) {
    HVX_సదిశ v_sum0_lo = Q6_V_vzero();
    HVX_సదిశ v_sum0_hi = Q6_V_vzero();
    HVX_సదిశ v_sum1_lo = Q6_V_vzero();
    HVX_సదిశ v_sum1_hi = Q6_V_vzero();
    HVX_సదిశ ముసుగు_03   = Q6_Vb_vsplat_R(0x03);

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q3_k_సమూహం(vptr, g,     ముసుగు_03);
        HVX_సదిశ v_W_hi = unpack_q3_k_సమూహం(vptr, g + 4, ముసుగు_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// Q2_K: the Q3_K dot products without the -4 flags, scales and mins are applied in q2_k_int_accum
స్థిర అంతఃవరుస HVX_సదిశ_Pair accum_q2_k_32x1(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act
) {
    HVX_సదిశ v_కూడిక_lo = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_hi = Q6_V_vzero();
    HVX_సదిశ ముసుగు_03  = Q6_Vb_vsplat_R(0x03);

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q3_k_low2(vptr, g,     ముసుగు_03);
        HVX_సదిశ v_W_hi = unpack_q3_k_low2(vptr, g + 4, ముసుగు_03);
        v_కూడిక_lo = Q6_Vw_vrmpyacc_VwVbVb(v_కూడిక_lo, v_W_lo, v_act[g]);
        v_కూడిక_hi = Q6_Vw_vrmpyacc_VwVbVb(v_కూడిక_hi, v_W_hi, v_act[g + 4]);
    }

    తిరిగి Q6_W_vcombine_VV(v_కూడిక_hi, v_కూడిక_lo);
}

స్థిర అంతఃవరుస శూన్య accum_q2_k_32x2(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1,
    HVX_సదిశ_Pair * v_sums0,
    HVX_సదిశ_Pair * v_sums1
) {
    HVX_సదిశ v_sum0_lo = Q6_V_vzero();
    HVX_సదిశ v_sum0_hi = Q6_V_vzero();
    HVX_సదిశ v_sum1_lo = Q6_V_vzero();
    HVX_సదిశ v_sum1_hi = Q6_V_vzero();
    HVX_సదిశ ముసుగు_03   = Q6_Vb_vsplat_R(0x03);

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q3_k_low2(vptr, g,     ముసుగు_03);
        HVX_సదిశ v_W_hi = unpack_q3_k_low2(vptr, g + 4, ముసుగు_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// Q2_0: unsigned codes 0..3; the -1 offset is applied via the Q8_1 block sum (v_act[9])
స్థిర అంతఃవరుస HVX_సదిశ accum_q2_0_32x1(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act
) {
    HVX_సదిశ v_కూడిక_lo = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_hi = Q6_V_vzero();
    HVX_సదిశ ముసుగు_03  = Q6_Vb_vsplat_R(0x03);

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q3_k_low2(vptr, g,     ముసుగు_03);
        HVX_సదిశ v_W_hi = unpack_q3_k_low2(vptr, g + 4, ముసుగు_03);
        v_కూడిక_lo = Q6_Vw_vrmpyacc_VwVubVb(v_కూడిక_lo, v_W_lo, v_act[g]);
        v_కూడిక_hi = Q6_Vw_vrmpyacc_VwVubVb(v_కూడిక_hi, v_W_hi, v_act[g + 4]);
    }

    తిరిగి Q6_Vw_vadd_VwVw(v_కూడిక_lo, v_కూడిక_hi);
}

స్థిర అంతఃవరుస శూన్య accum_q2_0_32x2(
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0,
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1,
    HVX_సదిశ * v_sum0,
    HVX_సదిశ * v_sum1
) {
    HVX_సదిశ v_sum0_lo = Q6_V_vzero();
    HVX_సదిశ v_sum0_hi = Q6_V_vzero();
    HVX_సదిశ v_sum1_lo = Q6_V_vzero();
    HVX_సదిశ v_sum1_hi = Q6_V_vzero();
    HVX_సదిశ ముసుగు_03   = Q6_Vb_vsplat_R(0x03);

    #సూచన unroll
    కొరకు (పూర్ణాంకం g = 0; g < 4; g++) {
        HVX_సదిశ v_W_lo = unpack_q3_k_low2(vptr, g,     ముసుగు_03);
        HVX_సదిశ v_W_hi = unpack_q3_k_low2(vptr, g + 4, ముసుగు_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVubVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVubVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVubVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVubVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sum0 = Q6_Vw_vadd_VwVw(v_sum0_lo, v_sum0_hi);
    *v_sum1 = Q6_Vw_vadd_VwVw(v_sum1_lo, v_sum1_hi);
}

// Q2_K: fp16 d (*v_d) and -dmin (*v_m) of the 32 rows of a super-block (8 tiles, VTCM stride 384), each in both vector halves
స్థిర అంతఃవరుస శూన్య q2_k_ఎక్కించు_dmin(స్థిరరాశి uint8_t * పరిమిత sb, HVX_సదిశ * పరిమిత v_d, HVX_సదిశ * పరిమిత v_m) {
    HVX_సదిశ v_dd = Q6_V_vzero();  // lanes 0..31: d, lanes 32..63: -dmin

    #సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
        HVX_సదిశ v = hvx_vmem(sb + j * 384 + 256);  // the slice is at bytes 64..79
        v_dd = Q6_V_vmux_QVV(Q6_Q_vsetq_R(16 * j), v_dd, Q6_V_vror_VR(v, (64 - 16 * j) & 127));
    }

    HVX_సదిశ_Pred q_lo = Q6_Q_vsetq_R(64);
    HVX_సదిశ     v_r  = Q6_V_vror_VR(v_dd, 64);
    *v_d = Q6_V_vmux_QVV(q_lo, v_dd, v_r);
    *v_m = Q6_V_vmux_QVV(q_lo, v_r, v_dd);
}

// Q2_K: fp16 D = d * scale and M = -dmin * min of a tile from its scale bytes, in the lanes of the Q6_K scales
స్థిర అంతఃవరుస శూన్య q2_k_unpack_scales(స్థిరరాశి uint8_t * పరిమిత tile, HVX_సదిశ v_d, HVX_సదిశ v_m,
                                      HVX_సదిశ * పరిమిత v_D, HVX_సదిశ * పరిమిత v_M) {
    HVX_సదిశ v_sc = hvx_vmem(tile + 256);
    HVX_సదిశ v_l  = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(Q6_V_vand_VV(v_sc, Q6_Vb_vsplat_R(0x0F))));
    HVX_సదిశ v_mn = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(Q6_Vub_vlsr_VubR(v_sc, 4)));
    // qf32 product: rounds like GGML_FP32_TO_FP16(d * scale) (qf16 is off by 1 ulp for many values on v73)
    *v_D = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(v_l), v_d));
    *v_M = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(v_mn), v_m));
}

// Q2_K scale and min nibbles of a tile as halfword pairs: word r = sub-block 1 << 16 | sub-block 0 of row r
స్థిర అంతఃవరుస శూన్య q2_k_scale_జతలు(స్థిరరాశి uint8_t * పరిమిత tile, HVX_సదిశ * పరిమిత v_sc2, HVX_సదిశ * పరిమిత v_mn2) {
    HVX_సదిశ v_h = Q6_Vh_vshuff_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile + 256))));
    *v_sc2 = Q6_V_vand_VV(v_h, Q6_Vh_vsplat_R(0x000F));
    *v_mn2 = Q6_Vuh_vlsr_VuhR(v_h, 4);
}

// Q2_K with q8_1_s16 activations, one tile: isum += scale * (q . a), msum += min * sum(a), per row, in int32.
// |sum of 16 q * a| <= 3 * 128 * 16 fits a halfword, so both sub-blocks go through one vdmpy (no saturation)
స్థిర అంతఃవరుస శూన్య q2_k_int_accum(HVX_సదిశ_Pair v_sums, HVX_సదిశ v_sc2, HVX_సదిశ v_mn2, HVX_సదిశ v_bsum,
                                  HVX_సదిశ * పరిమిత v_isum, HVX_సదిశ * పరిమిత v_msum) {
    HVX_సదిశ v_s2 = Q6_Vh_vshuffe_VhVh(Q6_V_hi_W(v_sums), Q6_V_lo_W(v_sums));
    *v_isum = Q6_Vw_vadd_VwVw(*v_isum, Q6_Vw_vdmpy_VhVh_sat(v_s2, v_sc2));
    *v_msum = Q6_Vw_vadd_VwVw(*v_msum, Q6_Vw_vdmpy_VhVh_sat(v_mn2, v_bsum));
}

// Q2_K super-block result: d_a * (d * isum - dmin * msum), with v_d / v_m the fp16 d / -dmin of q2_k_load_dmin
స్థిర అంతఃవరుస HVX_సదిశ q2_k_int_scale(HVX_సదిశ v_isum, HVX_సదిశ v_msum, HVX_సదిశ v_d, HVX_సదిశ v_m, HVX_సదిశ v_act8) {
    HVX_సదిశ v_d32  = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(v_d));
    HVX_సదిశ v_m32  = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(v_m));
    HVX_సదిశ v_da32 = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(v_act8));
    HVX_సదిశ v_t = hvx_vec_కూడు_f32_f32(hvx_vec_mul_f32_f32(v_d32, Q6_Vsf_equals_Vw(v_isum)),
                                         hvx_vec_mul_f32_f32(v_m32, Q6_Vsf_equals_Vw(v_msum)));
    తిరిగి hvx_vec_mul_f32_f32(v_t, v_da32);
}

స్థిర శూన్య tiled_vec_dot_q1_0_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_qf  = Q6_V_vzero();
    HVX_సదిశ v_neg_one = Q6_Vb_vsplat_R(-1);
    HVX_సదిశ v_pos_one = Q6_Vb_vsplat_R(1);

    స్థిరరాశి uint32_t n_k_tiles = n / HTP_TENSOR_TILE_K_Q1_0;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * htp_tensor_get_tile_padded_కొలత(HTP_TYPE_Q1_0));
        HVX_సదిశ v_tile_qf = Q6_V_vzero();

        #సూచన unroll
        కొరకు (పూర్ణాంకం quarter = 0; quarter < 4; quarter++) {
            స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + (4 * kt + quarter) * HTP_MM_ACT_TILE_కొలత_Q8_0);
            HVX_సదిశ v_కూడిక_sf = Q6_Vsf_equals_Vw(accum_q1_0_32x1(vptr[quarter], v_act, v_neg_one, v_pos_one));
            HVX_సదిశ v_da_sf  = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32_shuff(v_act[8]));
            HVX_సదిశ v_q_qf   = Q6_Vqf32_vmpy_VsfVsf(v_కూడిక_sf, v_da_sf);
            v_tile_qf = (quarter == 0) ? v_q_qf : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf, v_q_qf);
        }

        HVX_సదిశ v_dw_sf = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(vptr[4]));
        v_కూడిక_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_కూడిక_qf, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf), v_dw_sf));
    }

    HVX_సదిశ v_కూడిక_float = Q6_Vsf_equals_Vqf32(v_కూడిక_qf);
    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_q1_0_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_qf_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_qf_c1 = Q6_V_vzero();
    HVX_సదిశ v_neg_one   = Q6_Vb_vsplat_R(-1);
    HVX_సదిశ v_pos_one   = Q6_Vb_vsplat_R(1);

    స్థిరరాశి uint32_t n_k_tiles = n / HTP_TENSOR_TILE_K_Q1_0;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * htp_tensor_get_tile_padded_కొలత(HTP_TYPE_Q1_0));
        HVX_సదిశ v_tile_qf_c0 = Q6_V_vzero();
        HVX_సదిశ v_tile_qf_c1 = Q6_V_vzero();

        కొరకు (పూర్ణాంకం quarter = 0; quarter < 4; quarter++) {
            స్థిరరాశి uint32_t act_tile = 4 * kt + quarter;
            స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + act_tile * HTP_MM_ACT_TILE_కొలత_Q8_0);
            స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + act_tile * HTP_MM_ACT_TILE_కొలత_Q8_0);
            HVX_సదిశ_Pair v_sums = accum_q1_0_32x2(vptr[quarter], v_act0, v_act1, v_neg_one, v_pos_one);
            HVX_సదిశ v_కూడిక_sf_c0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums));
            HVX_సదిశ v_కూడిక_sf_c1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums));
            HVX_సదిశ v_da_sf_c0  = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32_shuff(v_act0[8]));
            HVX_సదిశ v_da_sf_c1  = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32_shuff(v_act1[8]));

            v_tile_qf_c0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c0, Q6_Vqf32_vmpy_VsfVsf(v_కూడిక_sf_c0, v_da_sf_c0));
            v_tile_qf_c1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c1, Q6_Vqf32_vmpy_VsfVsf(v_కూడిక_sf_c1, v_da_sf_c1));
        }

        HVX_సదిశ v_dw_sf = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(vptr[4]));
        v_కూడిక_qf_c0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_కూడిక_qf_c0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf_c0), v_dw_sf));
        v_కూడిక_qf_c1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_కూడిక_qf_c1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf_c1), v_dw_sf));
    }

    HVX_సదిశ v_కూడిక_float_c0 = Q6_Vsf_equals_Vqf32(v_కూడిక_qf_c0);
    HVX_సదిశ v_కూడిక_float_c1 = Q6_Vsf_equals_Vqf32(v_కూడిక_qf_c1);
    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_q4_0_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();
    HVX_సదిశ i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 640);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + kt * 1152);

        HVX_సదిశ v_కూడిక = accum_4bit_32x1(vptr, v_act, i8);
        HVX_సదిశ v_కూడిక_sf = Q6_Vsf_equals_Vw(v_కూడిక);

        HVX_సదిశ v_scale_w = vptr[4];
        HVX_సదిశ v_scale_a = v_act[8];
        HVX_సదిశ v_scale_comb = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale_w, v_scale_a);
        HVX_సదిశ v_కూడిక_scaled = hvx_vec_mul_f32_f32(v_కూడిక_sf, v_scale_comb);

        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, v_కూడిక_scaled);
    }

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_q4_0_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();
    HVX_సదిశ i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 640);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + kt * 1152);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + kt * 1152);

        HVX_సదిశ_Pair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, i8);
        HVX_సదిశ v_కూడిక_c0 = Q6_V_lo_W(v_sums);
        HVX_సదిశ v_కూడిక_c1 = Q6_V_hi_W(v_sums);

        HVX_సదిశ v_కూడిక_sf_c0 = Q6_Vsf_equals_Vw(v_కూడిక_c0);
        HVX_సదిశ v_కూడిక_sf_c1 = Q6_Vsf_equals_Vw(v_కూడిక_c1);

        HVX_సదిశ v_scale_w = vptr[4];
        HVX_సదిశ v_scale_a_c0 = v_act0[8];
        HVX_సదిశ v_scale_a_c1 = v_act1[8];

        HVX_సదిశ v_scale_comb_c0 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_సదిశ v_scale_comb_c1 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_సదిశ v_కూడిక_scaled_c0 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c0, v_scale_comb_c0);
        HVX_సదిశ v_కూడిక_scaled_c1 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c1, v_scale_comb_c1);

        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, v_కూడిక_scaled_c0);
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, v_కూడిక_scaled_c1);
    }

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_q4_1_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 640);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + kt * 1280);

        HVX_సదిశ v_కూడిక = accum_4bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_సదిశ v_కూడిక_sf = Q6_Vsf_equals_Vw(v_కూడిక);

        HVX_సదిశ v_scale_స్థానభ్రంశం = vptr[4];
        HVX_సదిశ_Pair p_deal = Q6_W_vdeal_VVR(v_scale_స్థానభ్రంశం, v_scale_స్థానభ్రంశం, -2);
        HVX_సదిశ v_scale = Q6_V_lo_W(p_deal);
        HVX_సదిశ v_స్థానభ్రంశం = Q6_V_hi_W(p_deal);

        HVX_సదిశ v_scale_a = v_act[8];
        HVX_సదిశ v_కూడిక_a   = v_act[9];

        HVX_సదిశ v_scale_comb = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale, v_scale_a);
        HVX_సదిశ v_స్థానభ్రంశం_comb = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_స్థానభ్రంశం, v_కూడిక_a);

        HVX_సదిశ v_scaled_dot = hvx_vec_mul_f32_f32(v_కూడిక_sf, v_scale_comb);
        HVX_సదిశ v_కూడిక_scaled = hvx_vec_కూడు_f32_f32(v_scaled_dot, v_స్థానభ్రంశం_comb);

        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, v_కూడిక_scaled);
    }

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_q4_1_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 640);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + kt * 1280);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + kt * 1280);

        HVX_సదిశ_Pair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_సదిశ v_కూడిక_c0 = Q6_V_lo_W(v_sums);
        HVX_సదిశ v_కూడిక_c1 = Q6_V_hi_W(v_sums);

        HVX_సదిశ v_కూడిక_sf_c0 = Q6_Vsf_equals_Vw(v_కూడిక_c0);
        HVX_సదిశ v_కూడిక_sf_c1 = Q6_Vsf_equals_Vw(v_కూడిక_c1);

        HVX_సదిశ v_scale_స్థానభ్రంశం = vptr[4];
        HVX_సదిశ_Pair p_deal = Q6_W_vdeal_VVR(v_scale_స్థానభ్రంశం, v_scale_స్థానభ్రంశం, -2);
        HVX_సదిశ v_scale = Q6_V_lo_W(p_deal);
        HVX_సదిశ v_స్థానభ్రంశం = Q6_V_hi_W(p_deal);

        HVX_సదిశ v_scale_a_c0 = v_act0[8];
        HVX_సదిశ v_కూడిక_a_c0   = v_act0[9];
        HVX_సదిశ v_scale_a_c1 = v_act1[8];
        HVX_సదిశ v_కూడిక_a_c1   = v_act1[9];

        HVX_సదిశ v_scale_comb_c0 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale, v_scale_a_c0);
        HVX_సదిశ v_స్థానభ్రంశం_comb_c0 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_స్థానభ్రంశం, v_కూడిక_a_c0);
        HVX_సదిశ v_scale_comb_c1 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale, v_scale_a_c1);
        HVX_సదిశ v_స్థానభ్రంశం_comb_c1 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_స్థానభ్రంశం, v_కూడిక_a_c1);

        HVX_సదిశ v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c0, v_scale_comb_c0);
        HVX_సదిశ v_కూడిక_scaled_c0 = hvx_vec_కూడు_f32_f32(v_scaled_dot_c0, v_స్థానభ్రంశం_comb_c0);

        HVX_సదిశ v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c1, v_scale_comb_c1);
        HVX_సదిశ v_కూడిక_scaled_c1 = hvx_vec_కూడు_f32_f32(v_scaled_dot_c1, v_స్థానభ్రంశం_comb_c1);

        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, v_కూడిక_scaled_c0);
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, v_కూడిక_scaled_c1);
    }

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_q8_0_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 1152);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + kt * 1152);

        HVX_సదిశ v_కూడిక = accum_q8_0_32x1(vptr, v_act);
        HVX_సదిశ v_కూడిక_sf = Q6_Vsf_equals_Vw(v_కూడిక);

        HVX_సదిశ v_scale_w = vptr[8];
        HVX_సదిశ v_scale_a = v_act[8];
        HVX_సదిశ v_scale_comb = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale_w, v_scale_a);
        HVX_సదిశ v_కూడిక_scaled = hvx_vec_mul_f32_f32(v_కూడిక_sf, v_scale_comb);

        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, v_కూడిక_scaled);
    }

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_q8_0_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 1152);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + kt * 1152);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + kt * 1152);

        HVX_సదిశ_Pair v_sums = accum_q8_0_32x2(vptr, v_act0, v_act1);
        HVX_సదిశ v_కూడిక_c0 = Q6_V_lo_W(v_sums);
        HVX_సదిశ v_కూడిక_c1 = Q6_V_hi_W(v_sums);

        HVX_సదిశ v_కూడిక_sf_c0 = Q6_Vsf_equals_Vw(v_కూడిక_c0);
        HVX_సదిశ v_కూడిక_sf_c1 = Q6_Vsf_equals_Vw(v_కూడిక_c1);

        HVX_సదిశ v_scale_w = vptr[8];
        HVX_సదిశ v_scale_a_c0 = v_act0[8];
        HVX_సదిశ v_scale_a_c1 = v_act1[8];

        HVX_సదిశ v_scale_comb_c0 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_సదిశ v_scale_comb_c1 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_సదిశ v_కూడిక_scaled_c0 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c0, v_scale_comb_c0);
        HVX_సదిశ v_కూడిక_scaled_c1 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c1, v_scale_comb_c1);

        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, v_కూడిక_scaled_c0);
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, v_కూడిక_scaled_c1);
    }

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_q5_k_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 768);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + kt * 1280);

        HVX_సదిశ v_కూడిక = accum_5bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_సదిశ v_కూడిక_sf = Q6_Vsf_equals_Vw(v_కూడిక);

        HVX_సదిశ v_scale_స్థానభ్రంశం = vptr[4];
        HVX_సదిశ_Pair p_deal = Q6_W_vdeal_VVR(v_scale_స్థానభ్రంశం, v_scale_స్థానభ్రంశం, -2);
        HVX_సదిశ v_scale = Q6_V_lo_W(p_deal);
        HVX_సదిశ v_స్థానభ్రంశం = Q6_V_hi_W(p_deal);

        HVX_సదిశ v_scale_a = v_act[8];
        HVX_సదిశ v_కూడిక_a   = v_act[9];

        HVX_సదిశ v_scale_comb = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale, v_scale_a);
        HVX_సదిశ v_స్థానభ్రంశం_comb = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_స్థానభ్రంశం, v_కూడిక_a);

        HVX_సదిశ v_scaled_dot = hvx_vec_mul_f32_f32(v_కూడిక_sf, v_scale_comb);
        HVX_సదిశ v_కూడిక_scaled = hvx_vec_కూడు_f32_f32(v_scaled_dot, v_స్థానభ్రంశం_comb);

        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, v_కూడిక_scaled);
    }

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_q5_k_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 768);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + kt * 1280);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + kt * 1280);

        HVX_సదిశ_Pair v_sums = accum_5bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_సదిశ v_కూడిక_c0 = Q6_V_lo_W(v_sums);
        HVX_సదిశ v_కూడిక_c1 = Q6_V_hi_W(v_sums);

        HVX_సదిశ v_కూడిక_sf_c0 = Q6_Vsf_equals_Vw(v_కూడిక_c0);
        HVX_సదిశ v_కూడిక_sf_c1 = Q6_Vsf_equals_Vw(v_కూడిక_c1);

        HVX_సదిశ v_scale_స్థానభ్రంశం = vptr[4];
        HVX_సదిశ_Pair p_deal = Q6_W_vdeal_VVR(v_scale_స్థానభ్రంశం, v_scale_స్థానభ్రంశం, -2);
        HVX_సదిశ v_scale = Q6_V_lo_W(p_deal);
        HVX_సదిశ v_స్థానభ్రంశం = Q6_V_hi_W(p_deal);

        HVX_సదిశ v_scale_a_c0 = v_act0[8];
        HVX_సదిశ v_కూడిక_a_c0   = v_act0[9];
        HVX_సదిశ v_scale_a_c1 = v_act1[8];
        HVX_సదిశ v_కూడిక_a_c1   = v_act1[9];

        HVX_సదిశ v_scale_comb_c0 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale, v_scale_a_c0);
        HVX_సదిశ v_స్థానభ్రంశం_comb_c0 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_స్థానభ్రంశం, v_కూడిక_a_c0);
        HVX_సదిశ v_scale_comb_c1 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale, v_scale_a_c1);
        HVX_సదిశ v_స్థానభ్రంశం_comb_c1 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_స్థానభ్రంశం, v_కూడిక_a_c1);

        HVX_సదిశ v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c0, v_scale_comb_c0);
        HVX_సదిశ v_కూడిక_scaled_c0 = hvx_vec_కూడు_f32_f32(v_scaled_dot_c0, v_స్థానభ్రంశం_comb_c0);

        HVX_సదిశ v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c1, v_scale_comb_c1);
        HVX_సదిశ v_కూడిక_scaled_c1 = hvx_vec_కూడు_f32_f32(v_scaled_dot_c1, v_స్థానభ్రంశం_comb_c1);

        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, v_కూడిక_scaled_c0);
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, v_కూడిక_scaled_c1);
    }

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_q6_k_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();
    HVX_సదిశ i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 896);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + kt * 1152);

        HVX_సదిశ_Pair v_sums = accum_q6_k_32x1(vptr, v_act, i32);
        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, scale_q6_k_32x1(v_sums, vptr[6], v_act[8]));
    }

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_q6_k_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();
    HVX_సదిశ i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 896);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + kt * 1152);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + kt * 1152);

        HVX_సదిశ_Pair v_sums0, v_sums1;
        accum_q6_k_32x2(vptr, v_act0, v_act1, i32, &v_sums0, &v_sums1);

        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, scale_q6_k_32x1(v_sums0, vptr[6], v_act0[8]));
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, scale_q6_k_32x1(v_sums1, vptr[6], v_act1[8]));
    }

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_q3_k_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 512);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + kt * 1152);

        HVX_సదిశ_Pair v_sums = accum_q3_k_32x1(vptr, v_act);
        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, scale_q6_k_32x1(v_sums, vptr[3], v_act[8]));
    }

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_q3_k_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 512);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + kt * 1152);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + kt * 1152);

        HVX_సదిశ_Pair v_sums0, v_sums1;
        accum_q3_k_32x2(vptr, v_act0, v_act1, &v_sums0, &v_sums1);

        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, scale_q6_k_32x1(v_sums0, vptr[3], v_act0[8]));
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, scale_q6_k_32x1(v_sums1, vptr[3], v_act1[8]));
    }

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_q2_k_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kb = 0; kb < n_k_tiles; kb += 8) {
        HVX_సదిశ v_d, v_m;
        q2_k_ఎక్కించు_dmin(tile_ptr + kb * 384, &v_d, &v_m);

        HVX_సదిశ v_isum = Q6_V_vzero();
        HVX_సదిశ v_msum = Q6_V_vzero();
        కొరకు (uint32_t kt = kb; kt < kb + 8; kt++) {
            స్థిరరాశి uint8_t * పరిమిత tile = tile_ptr + kt * 384;
            స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + kt * 1280);

            HVX_సదిశ v_sc2, v_mn2;
            q2_k_scale_జతలు(tile, &v_sc2, &v_mn2);
            q2_k_int_accum(accum_q2_k_32x1((స్థిరరాశి HVX_సదిశ *) tile, v_act), v_sc2, v_mn2, v_act[9], &v_isum, &v_msum);
        }
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act_sb = (స్థిరరాశి HVX_సదిశ *) (y_q + kb * 1280);
        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, q2_k_int_scale(v_isum, v_msum, v_d, v_m, v_act_sb[8]));
    }

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_q2_k_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kb = 0; kb < n_k_tiles; kb += 8) {
        HVX_సదిశ v_d, v_m;
        q2_k_ఎక్కించు_dmin(tile_ptr + kb * 384, &v_d, &v_m);

        HVX_సదిశ v_isum0 = Q6_V_vzero();
        HVX_సదిశ v_msum0 = Q6_V_vzero();
        HVX_సదిశ v_isum1 = Q6_V_vzero();
        HVX_సదిశ v_msum1 = Q6_V_vzero();
        కొరకు (uint32_t kt = kb; kt < kb + 8; kt++) {
            స్థిరరాశి uint8_t * పరిమిత tile = tile_ptr + kt * 384;
            స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + kt * 1280);
            స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + kt * 1280);

            HVX_సదిశ v_sc2, v_mn2;
            q2_k_scale_జతలు(tile, &v_sc2, &v_mn2);

            HVX_సదిశ_Pair v_sums0, v_sums1;
            accum_q2_k_32x2((స్థిరరాశి HVX_సదిశ *) tile, v_act0, v_act1, &v_sums0, &v_sums1);
            q2_k_int_accum(v_sums0, v_sc2, v_mn2, v_act0[9], &v_isum0, &v_msum0);
            q2_k_int_accum(v_sums1, v_sc2, v_mn2, v_act1[9], &v_isum1, &v_msum1);
        }
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act0_sb = (స్థిరరాశి HVX_సదిశ *) (y0_q + kb * 1280);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act1_sb = (స్థిరరాశి HVX_సదిశ *) (y1_q + kb * 1280);
        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, q2_k_int_scale(v_isum0, v_msum0, v_d, v_m, v_act0_sb[8]));
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, q2_k_int_scale(v_isum1, v_msum1, v_d, v_m, v_act1_sb[8]));
    }

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

// One Q2_0 weight tile covers 64 k: act tiles 2 * kt and 2 * kt + 1, each 32 k wide (Q8_1)
స్థిర శూన్య tiled_vec_dot_q2_0_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();
    HVX_సదిశ v_one_f16   = hvx_vec_splat_f16(1.0);

    uint32_t n_బరువు_tiles = n / HTP_MM_బరువు_TILE_K_Q2_0;
    కొరకు (uint32_t kt = 0; kt < n_బరువు_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * HTP_MM_బరువు_ALIGNED_TILE_కొలత_Q2_0);
        HVX_సదిశ v_tile_qf = Q6_V_vzero();

        #సూచన unroll
        కొరకు (పూర్ణాంకం half = 0; half < 2; half++) {
            స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + (2 * kt + half) * HTP_MM_ACT_TILE_కొలత_Q8_1);

            HVX_సదిశ v_కూడిక_sf = Q6_Vsf_equals_Vw(accum_q2_0_32x1(vptr + 2 * half, v_act));
            HVX_సదిశ v_da_sf  = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32_shuff(v_act[8]));
            HVX_సదిశ v_sa_qf  = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act[9], v_one_f16));
            HVX_సదిశ v_h_qf   = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_కూడిక_sf, v_da_sf), v_sa_qf);
            v_tile_qf = (half == 0) ? v_h_qf : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf, v_h_qf);
        }

        HVX_సదిశ v_dw_sf = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(vptr[4]));
        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf), v_dw_sf));
    }

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_q2_0_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();
    HVX_సదిశ v_one_f16      = hvx_vec_splat_f16(1.0);

    uint32_t n_బరువు_tiles = n / HTP_MM_బరువు_TILE_K_Q2_0;
    కొరకు (uint32_t kt = 0; kt < n_బరువు_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * HTP_MM_బరువు_ALIGNED_TILE_కొలత_Q2_0);
        HVX_సదిశ v_tile_qf_c0 = Q6_V_vzero();
        HVX_సదిశ v_tile_qf_c1 = Q6_V_vzero();

        #సూచన unroll
        కొరకు (పూర్ణాంకం half = 0; half < 2; half++) {
            స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + (2 * kt + half) * HTP_MM_ACT_TILE_కొలత_Q8_1);
            స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + (2 * kt + half) * HTP_MM_ACT_TILE_కొలత_Q8_1);

            HVX_సదిశ v_కూడిక_c0, v_కూడిక_c1;
            accum_q2_0_32x2(vptr + 2 * half, v_act0, v_act1, &v_కూడిక_c0, &v_కూడిక_c1);

            HVX_సదిశ v_da_sf_c0 = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32_shuff(v_act0[8]));
            HVX_సదిశ v_da_sf_c1 = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32_shuff(v_act1[8]));
            HVX_సదిశ v_sa_qf_c0 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act0[9], v_one_f16));
            HVX_సదిశ v_sa_qf_c1 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act1[9], v_one_f16));

            HVX_సదిశ v_h_qf_c0 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(v_కూడిక_c0), v_da_sf_c0), v_sa_qf_c0);
            HVX_సదిశ v_h_qf_c1 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(v_కూడిక_c1), v_da_sf_c1), v_sa_qf_c1);

            v_tile_qf_c0 = (half == 0) ? v_h_qf_c0 : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c0, v_h_qf_c0);
            v_tile_qf_c1 = (half == 0) ? v_h_qf_c1 : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c1, v_h_qf_c1);
        }

        HVX_సదిశ v_dw_sf = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(vptr[4]));
        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf_c0), v_dw_sf));
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf_c1), v_dw_sf));
    }

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_iq4nl_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();
    HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_సదిశ lut = *(స్థిరరాశి HVX_సదిశ *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 640);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + kt * 1152);

        HVX_సదిశ v_కూడిక = accum_4bit_32x1_lut(vptr, v_act, ముసుగు_h4, lut);
        HVX_సదిశ v_కూడిక_sf = Q6_Vsf_equals_Vw(v_కూడిక);

        HVX_సదిశ v_scale_w = vptr[4];
        HVX_సదిశ v_scale_a = v_act[8];
        HVX_సదిశ v_scale_comb = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale_w, v_scale_a);
        HVX_సదిశ v_కూడిక_scaled = hvx_vec_mul_f32_f32(v_కూడిక_sf, v_scale_comb);

        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, v_కూడిక_scaled);
    }

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_iq4nl_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();
    HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_సదిశ lut = *(స్థిరరాశి HVX_సదిశ *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 640);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + kt * 1152);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + kt * 1152);

        HVX_సదిశ_Pair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, ముసుగు_h4, lut);
        HVX_సదిశ v_కూడిక_c0 = Q6_V_lo_W(v_sums);
        HVX_సదిశ v_కూడిక_c1 = Q6_V_hi_W(v_sums);

        HVX_సదిశ v_కూడిక_sf_c0 = Q6_Vsf_equals_Vw(v_కూడిక_c0);
        HVX_సదిశ v_కూడిక_sf_c1 = Q6_Vsf_equals_Vw(v_కూడిక_c1);

        HVX_సదిశ v_scale_w = vptr[4];
        HVX_సదిశ v_scale_a_c0 = v_act0[8];
        HVX_సదిశ v_scale_a_c1 = v_act1[8];

        HVX_సదిశ v_scale_comb_c0 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_సదిశ v_scale_comb_c1 = hvx_vec_mul_f16_f16_గమ్యం_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_సదిశ v_కూడిక_scaled_c0 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c0, v_scale_comb_c0);
        HVX_సదిశ v_కూడిక_scaled_c1 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c1, v_scale_comb_c1);

        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, v_కూడిక_scaled_c0);
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, v_కూడిక_scaled_c1);
    }

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_mxfp4_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y_q = vy;

    HVX_సదిశ v_కూడిక_float = Q6_V_vzero();
    HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_సదిశ lut = *(స్థిరరాశి HVX_సదిశ *) kvalues_mxfp4_lut;
    HVX_సదిశ expand = *(స్థిరరాశి HVX_సదిశ *) expand_x32_e8m0;
    HVX_సదిశ e8m0_ముసుగు = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 640);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) (y_q + kt * 1152);

        HVX_సదిశ v_కూడిక = accum_4bit_32x1_lut(vptr, v_act, ముసుగు_h4, lut);
        HVX_సదిశ v_కూడిక_sf = Q6_Vsf_equals_Vw(v_కూడిక);

        HVX_సదిశ v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_సదిశ r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_ముసుగు);
        HVX_సదిశ v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_సదిశ v_scale_a_f16 = v_act[8];
        HVX_సదిశ_Pair p_scale_a_f32 = hvx_vec_f16_గమ్యం_f32_shuff(v_scale_a_f16);
        HVX_సదిశ v_scale_a = Q6_V_lo_W(p_scale_a_f32);

        HVX_సదిశ v_scale_comb = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a);
        HVX_సదిశ v_కూడిక_scaled = hvx_vec_mul_f32_f32(v_కూడిక_sf, v_scale_comb);

        v_కూడిక_float = hvx_vec_కూడు_f32_f32(v_కూడిక_float, v_కూడిక_scaled);
    }

    v_కూడిక_float = hvx_vec_mul_f32_f32(v_కూడిక_float, hvx_vec_splat_f32(0.5f));

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float);
    }
}

స్థిర శూన్య tiled_vec_dot_mxfp4_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి uint8_t * పరిమిత tile_ptr = vx;
    స్థిరరాశి uint8_t * పరిమిత y0_q = vy0;
    స్థిరరాశి uint8_t * పరిమిత y1_q = vy1;

    HVX_సదిశ v_కూడిక_float_c0 = Q6_V_vzero();
    HVX_సదిశ v_కూడిక_float_c1 = Q6_V_vzero();
    HVX_సదిశ ముసుగు_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_సదిశ lut = *(స్థిరరాశి HVX_సదిశ *) kvalues_mxfp4_lut;
    HVX_సదిశ expand = *(స్థిరరాశి HVX_సదిశ *) expand_x32_e8m0;
    HVX_సదిశ e8m0_ముసుగు = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    కొరకు (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        స్థిరరాశి HVX_సదిశ * పరిమిత vptr = (స్థిరరాశి HVX_సదిశ *) (tile_ptr + kt * 640);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) (y0_q + kt * 1152);
        స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) (y1_q + kt * 1152);

        HVX_సదిశ_Pair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, ముసుగు_h4, lut);
        HVX_సదిశ v_కూడిక_c0 = Q6_V_lo_W(v_sums);
        HVX_సదిశ v_కూడిక_c1 = Q6_V_hi_W(v_sums);

        HVX_సదిశ v_కూడిక_sf_c0 = Q6_Vsf_equals_Vw(v_కూడిక_c0);
        HVX_సదిశ v_కూడిక_sf_c1 = Q6_Vsf_equals_Vw(v_కూడిక_c1);

        HVX_సదిశ v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_సదిశ r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_ముసుగు);
        HVX_సదిశ v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_సదిశ v_scale_a_c0_f16 = v_act0[8];
        HVX_సదిశ v_scale_a_c1_f16 = v_act1[8];

        HVX_సదిశ_Pair p_scale_a_c0_f32 = hvx_vec_f16_గమ్యం_f32_shuff(v_scale_a_c0_f16);
        HVX_సదిశ_Pair p_scale_a_c1_f32 = hvx_vec_f16_గమ్యం_f32_shuff(v_scale_a_c1_f16);

        HVX_సదిశ v_scale_a_c0 = Q6_V_lo_W(p_scale_a_c0_f32);
        HVX_సదిశ v_scale_a_c1 = Q6_V_lo_W(p_scale_a_c1_f32);

        HVX_సదిశ v_scale_comb_c0 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c0);
        HVX_సదిశ v_scale_comb_c1 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c1);

        HVX_సదిశ v_కూడిక_scaled_c0 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c0, v_scale_comb_c0);
        HVX_సదిశ v_కూడిక_scaled_c1 = hvx_vec_mul_f32_f32(v_కూడిక_sf_c1, v_scale_comb_c1);

        v_కూడిక_float_c0 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, v_కూడిక_scaled_c0);
        v_కూడిక_float_c1 = hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, v_కూడిక_scaled_c1);
    }

    v_కూడిక_float_c0 = hvx_vec_mul_f32_f32(v_కూడిక_float_c0, hvx_vec_splat_f32(0.5f));
    v_కూడిక_float_c1 = hvx_vec_mul_f32_f32(v_కూడిక_float_c1, hvx_vec_splat_f32(0.5f));

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక_float_c1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక_float_c1);
    }
}

స్థిర శూన్య tiled_vec_dot_f16_32x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz) {
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr  = (స్థిరరాశి HVX_సదిశ *) vx;
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act = (స్థిరరాశి HVX_సదిశ *) vy;
    స్థిరరాశి uint32_t n_vecs = ((n + 31) / 32) * 16;

    HVX_సదిశ_Pair acc = Q6_W_vzero();

    #సూచన unroll(4)
    కొరకు (uint32_t i = 0; i < n_vecs; i++) {
        acc = hvx_vec_mpyacc_f32_f16(acc, vptr[i], v_act[i]);
    }

    HVX_సదిశ v_కూడిక = hvx_vec_కూడు_f32_f32(Q6_V_lo_W(acc), Q6_V_hi_W(acc));

    ఒకవేళ (sz) {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_కూడిక, hvx_vmemu(sz)));
    } లేకపోతే {
        hvx_vec_store_u(s, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_కూడిక);
    }
}

స్థిర శూన్య tiled_vec_dot_f16_32x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు, స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1) {
    స్థిరరాశి HVX_సదిశ * పరిమిత vptr   = (స్థిరరాశి HVX_సదిశ *) vx;
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act0 = (స్థిరరాశి HVX_సదిశ *) vy0;
    స్థిరరాశి HVX_సదిశ * పరిమిత v_act1 = (స్థిరరాశి HVX_సదిశ *) vy1;
    స్థిరరాశి uint32_t n_vecs = ((n + 31) / 32) * 16;

    HVX_సదిశ_Pair acc0 = Q6_W_vzero();
    HVX_సదిశ_Pair acc1 = Q6_W_vzero();

    #సూచన unroll(4)
    కొరకు (uint32_t i = 0; i < n_vecs; i++) {
        HVX_సదిశ v_w = vptr[i];
        acc0 = hvx_vec_mpyacc_f32_f16(acc0, v_w, v_act0[i]);
        acc1 = hvx_vec_mpyacc_f32_f16(acc1, v_w, v_act1[i]);
    }

    HVX_సదిశ v_sum0 = hvx_vec_కూడు_f32_f32(Q6_V_lo_W(acc0), Q6_V_hi_W(acc0));
    HVX_సదిశ v_sum1 = hvx_vec_కూడు_f32_f32(Q6_V_lo_W(acc1), Q6_V_hi_W(acc1));

    ఒకవేళ (sz0) {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_sum0, hvx_vmemu(sz0)));
    } లేకపోతే {
        hvx_vec_store_u(s0, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_sum0);
    }
    ఒకవేళ (sz1) {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_sum1, hvx_vmemu(sz1)));
    } లేకపోతే {
        hvx_vec_store_u(s1, valid_అడ్డువరుసలు * పరిమాణం(దశాంశం), v_sum1);
    }
}


స్థిర అంతఃవరుస శూన్య quantize_f32_q8_0_tiled_kernel(
    స్థిరరాశి uint8_t * పరిమిత src_దత్తాంశం,
    uint8_t * పరిమిత dst_దత్తాంశం,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత
) {
    (శూన్య) tmp_దత్తాంశం;
    కొరకు (uint32_t i = 0; i < nrows; ++i) {
        quantize_అడ్డువరుస_f32_q8_0_tiled((దశాంశం *) src_దత్తాంశం, dst_దత్తాంశం, ne0);
        dst_దత్తాంశం += dst_అడ్డువరుస_కొలత;
        src_దత్తాంశం += src_అడ్డువరుస_కొలత;
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f32_q8_1_tiled_kernel(
    స్థిరరాశి uint8_t * పరిమిత src_దత్తాంశం,
    uint8_t * పరిమిత dst_దత్తాంశం,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత
) {
    (శూన్య) tmp_దత్తాంశం;
    కొరకు (uint32_t i = 0; i < nrows; ++i) {
        quantize_అడ్డువరుస_f32_q8_1_tiled((దశాంశం *) src_దత్తాంశం, dst_దత్తాంశం, ne0);
        dst_దత్తాంశం += dst_అడ్డువరుస_కొలత;
        src_దత్తాంశం += src_అడ్డువరుస_కొలత;
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f32_q8_1_s16_tiled_kernel(
    స్థిరరాశి uint8_t * పరిమిత src_దత్తాంశం,
    uint8_t * పరిమిత dst_దత్తాంశం,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత
) {
    (శూన్య) tmp_దత్తాంశం;
    కొరకు (uint32_t i = 0; i < nrows; ++i) {
        quantize_అడ్డువరుస_f32_q8_1_s16_tiled((దశాంశం *) src_దత్తాంశం, dst_దత్తాంశం, ne0);
        dst_దత్తాంశం += dst_అడ్డువరుస_కొలత;
        src_దత్తాంశం += src_అడ్డువరుస_కొలత;
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f32_f16_tiled_kernel(
    స్థిరరాశి uint8_t * పరిమిత src_దత్తాంశం,
    uint8_t * పరిమిత dst_దత్తాంశం,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత
) {
    (శూన్య) tmp_దత్తాంశం;
    కొరకు (uint32_t i = 0; i < nrows; ++i) {
        quantize_అడ్డువరుస_f32_f16_tiled((స్థిరరాశి దశాంశం *) src_దత్తాంశం, dst_దత్తాంశం, ne0);
        dst_దత్తాంశం += dst_అడ్డువరుస_కొలత;
        src_దత్తాంశం += src_అడ్డువరుస_కొలత;
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f16_f16_tiled_kernel(
    స్థిరరాశి uint8_t * పరిమిత src_దత్తాంశం,
    uint8_t * పరిమిత dst_దత్తాంశం,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత
) {
    (శూన్య) tmp_దత్తాంశం;
    కొరకు (uint32_t i = 0; i < nrows; ++i) {
        quantize_అడ్డువరుస_f16_f16_tiled((స్థిరరాశి __fp16 *) src_దత్తాంశం, dst_దత్తాంశం, ne0);
        dst_దత్తాంశం += dst_అడ్డువరుస_కొలత;
        src_దత్తాంశం += src_అడ్డువరుస_కొలత;
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f32_q8_0_tiled_ఖండం_kernel(
    స్థిరరాశి దశాంశం * పరిమిత src,
    uint8_t * పరిమిత dst,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t ib_మొదటి,
    uint32_t ib_చివరి,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత,
    uint32_t r,
    uint32_t c
) {
    (శూన్య) tmp_దత్తాంశం;
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (ne0 + qk - 1) / qk;

    కొరకు (uint32_t ib = ib_మొదటి; ib < ib_చివరి; ++ib) {
        స్థిరరాశి దశాంశం * పరిమిత src_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + r * src_అడ్డువరుస_కొలత + c * qk * పరిమాణం(దశాంశం));
        uint8_t * పరిమిత dst_ptr = dst + r * dst_అడ్డువరుస_కొలత + c * 4 * 1152;

        quantize_ఖండం_f32_q8_0_tiled((దశాంశం *) src_ptr, dst_ptr);

        c++;
        ఒకవేళ (c == nb) {
            c = 0;
            r++;
        }
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f32_q8_1_tiled_ఖండం_kernel(
    స్థిరరాశి దశాంశం * పరిమిత src,
    uint8_t * పరిమిత dst,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t ib_మొదటి,
    uint32_t ib_చివరి,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత,
    uint32_t r,
    uint32_t c
) {
    (శూన్య) tmp_దత్తాంశం;
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (ne0 + qk - 1) / qk;

    కొరకు (uint32_t ib = ib_మొదటి; ib < ib_చివరి; ++ib) {
        స్థిరరాశి దశాంశం * పరిమిత src_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + r * src_అడ్డువరుస_కొలత + c * qk * పరిమాణం(దశాంశం));
        uint8_t * పరిమిత dst_ptr = dst + r * dst_అడ్డువరుస_కొలత + c * 4 * 1280;

        quantize_ఖండం_f32_q8_1_tiled((దశాంశం *) src_ptr, dst_ptr);

        c++;
        ఒకవేళ (c == nb) {
            c = 0;
            r++;
        }
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f32_q8_1_s16_tiled_ఖండం_kernel(
    స్థిరరాశి దశాంశం * పరిమిత src,
    uint8_t * పరిమిత dst,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t ib_మొదటి,
    uint32_t ib_చివరి,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత,
    uint32_t r,
    uint32_t c
) {
    (శూన్య) tmp_దత్తాంశం;
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (ne0 + qk - 1) / qk;

    కొరకు (uint32_t ib = ib_మొదటి; ib < ib_చివరి; ++ib) {
        స్థిరరాశి దశాంశం * పరిమిత src_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + r * src_అడ్డువరుస_కొలత + c * qk * పరిమాణం(దశాంశం));
        uint8_t * పరిమిత dst_ptr = dst + r * dst_అడ్డువరుస_కొలత + c * 4 * 1280;

        quantize_ఖండం_f32_q8_1_s16_tiled((దశాంశం *) src_ptr, dst_ptr, src_ptr - (c & 1) * qk);  // super-block of 256 k

        c++;
        ఒకవేళ (c == nb) {
            c = 0;
            r++;
        }
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f32_f16_tiled_ఖండం_kernel(
    స్థిరరాశి దశాంశం * పరిమిత src,
    uint8_t * పరిమిత dst,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t ib_మొదటి,
    uint32_t ib_చివరి,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత,
    uint32_t r,
    uint32_t c
) {
    (శూన్య) tmp_దత్తాంశం;
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (ne0 + qk - 1) / qk;

    కొరకు (uint32_t ib = ib_మొదటి; ib < ib_చివరి; ++ib) {
        స్థిరరాశి దశాంశం * పరిమిత src_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src + r * src_అడ్డువరుస_కొలత + c * qk * పరిమాణం(దశాంశం));
        uint8_t * పరిమిత dst_ptr = dst + r * dst_అడ్డువరుస_కొలత + c * 4 * 2048;

        quantize_ఖండం_f32_f16_tiled(src_ptr, dst_ptr);

        c++;
        ఒకవేళ (c == nb) {
            c = 0;
            r++;
        }
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f16_f16_tiled_ఖండం_kernel(
    స్థిరరాశి __fp16 * పరిమిత src,
    uint8_t * పరిమిత dst,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t ib_మొదటి,
    uint32_t ib_చివరి,
    size_t src_అడ్డువరుస_కొలత,
    size_t dst_అడ్డువరుస_కొలత,
    uint32_t r,
    uint32_t c
) {
    (శూన్య) tmp_దత్తాంశం;
    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (ne0 + qk - 1) / qk;

    కొరకు (uint32_t ib = ib_మొదటి; ib < ib_చివరి; ++ib) {
        స్థిరరాశి __fp16 * పరిమిత src_ptr = (స్థిరరాశి __fp16 *) ((స్థిరరాశి uint8_t *) src + r * src_అడ్డువరుస_కొలత + c * qk * పరిమాణం(__fp16));
        uint8_t * పరిమిత dst_ptr = dst + r * dst_అడ్డువరుస_కొలత + c * 4 * 2048;

        quantize_ఖండం_f16_f16_tiled(src_ptr, dst_ptr);

        c++;
        ఒకవేళ (c == nb) {
            c = 0;
            r++;
        }
    }
}
