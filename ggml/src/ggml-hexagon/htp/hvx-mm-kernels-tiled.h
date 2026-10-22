// Dynamic quantizers that produce tiled activations

// q8_1: vector 9 holds d * sum(q) of the 32 k, replicated.
// q8_1_s16 (Q2_K, sums16): one fp16 d (vector 8) for the 256 k super-block starting at x_sb, and vector 9 holds the
// integer sums of k 0..15 (low halfword) and k 16..31 (high halfword) in every word
__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य quantize_खण्डम्_f32_q8_1_tiled_impl(भिन्न * सीमित x, uint8_t * सीमित y_खण्डम्, ध्रुव bool sums16,
                                                      ध्रुव भिन्न * सीमित x_sb) {
    प्रतिज्ञा((अचिह्नित दीर्घ) x % 128 == 0);
    प्रतिज्ञा((अचिह्नित दीर्घ) y_खण्डम् % 128 == 0);

    HVX_सदिशम् * vx = (HVX_सदिशम् *) x;
    HVX_सदिशम् zero = Q6_V_vzero();

    HVX_सदिशम् vmax0_sf, vmax1_sf, vmax2_sf, vmax3_sf;
    यदि (sums16) {
        ध्रुव HVX_सदिशम् * vsb = (ध्रुव HVX_सदिशम् *) x_sb;
        HVX_सदिशम् vm = hvx_vec_abs_f32(vsb[0]);
        प्रति (पूर्णाङ्क i = 1; i < 8; i++) {
            vm = Q6_Vsf_vmax_VsfVsf(vm, hvx_vec_abs_f32(vsb[i]));
        }
        vmax0_sf = vmax1_sf = vmax2_sf = vmax3_sf = hvx_vec_reduce_max_f32(vm);
    } अन्यथा {
        vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
        vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
        vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
        vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));
    }

    HVX_सदिशम् vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_सदिशम् vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_सदिशम् vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_सदिशम् vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);

    HVX_सदिशम् vmax01_hf   = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_सदिशम् vx01_hf     = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_सदिशम् vd01_qf16   = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_सदिशम् vd01_hf     = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_सदिशम् vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));

    HVX_सदिशम् vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_सदिशम् vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);
    HVX_सदिशम् vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_सदिशम् vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_सदिशम् vmax23_hf   = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));
    HVX_सदिशम् vx23_hf     = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));
    HVX_सदिशम् vd23_qf16   = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_सदिशम् vd23_hf     = Q6_Vhf_equals_Vqf16(vd23_qf16);
    HVX_सदिशम् vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_सदिशम् vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_सदिशम् vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_सदिशम् vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    ध्रुव HVX_सदिशम् ones = Q6_Vb_vsplat_R(1);
    HVX_सदिशम् v_sums = Q6_Vw_vrmpy_VbVb(vx_i8, ones);
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 4));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 8));
    यदि (!sums16) {
        v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 16));
    }
    // word 8b: sum of block b (sums16: k 0..15, and word 8b + 4: k 16..31)

    ध्रुव HVX_सदिशम् v_inv127 = hvx_vec_splat_f32(1.0f / 127.0f);
    HVX_सदिशम् vd0_sf = hvx_vec_mul_f32_f32(vmax0_sf, v_inv127);
    HVX_सदिशम् vd1_sf = hvx_vec_mul_f32_f32(vmax1_sf, v_inv127);
    HVX_सदिशम् vd2_sf = hvx_vec_mul_f32_f32(vmax2_sf, v_inv127);
    HVX_सदिशम् vd3_sf = hvx_vec_mul_f32_f32(vmax3_sf, v_inv127);

    HVX_सदिशम् v_sums_sf = Q6_Vsf_equals_Vw(v_sums);
    HVX_सदिशम् voff0_sf = hvx_vec_mul_f32_f32(vd0_sf, v_sums_sf);
    HVX_सदिशम् voff1_sf = hvx_vec_mul_f32_f32(vd1_sf, Q6_V_vror_VR(v_sums_sf, 32));
    HVX_सदिशम् voff2_sf = hvx_vec_mul_f32_f32(vd2_sf, Q6_V_vror_VR(v_sums_sf, 64));
    HVX_सदिशम् voff3_sf = hvx_vec_mul_f32_f32(vd3_sf, Q6_V_vror_VR(v_sums_sf, 96));

    HVX_सदिशम् voff01_hf = hvx_vec_f32_गन्तव्यम्_f16(voff0_sf, voff1_sf);
    HVX_सदिशम् voff23_hf = hvx_vec_f32_गन्तव्यम्_f16(voff2_sf, voff3_sf);

    int32_t __attribute__((aligned(128))) ws[32];
    यदि (sums16) {
        // |sum of 16 int8| <= 2032 fits a halfword
        *(HVX_सदिशम् *) ws = v_sums;
    }

    स्थिर ध्रुव uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_सदिशम् v_repl_ctrl = * (ध्रुव HVX_सदिशम् *) repl;

    #निर्देशः unroll
    प्रति (पूर्णाङ्क b = 0; b < 4; b++) {
        HVX_सदिशम् v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_सदिशम् vd_b   = (b < 2) ? vd01_hf   : vd23_hf;
        HVX_सदिशम् voff_b = (b < 2) ? voff01_hf : voff23_hf;
        HVX_सदिशम् r_scale  = hvx_vec_repl_f16((b & 1) ? Q6_V_vror_VR(vd_b, 64)   : vd_b);
        HVX_सदिशम् r_विस्थापनम् = sums16
            ? Q6_V_vsplat_R(((uint32_t) ws[8 * b + 4] << 16) | ((uint32_t) ws[8 * b] & 0xFFFF))
            : hvx_vec_repl_f16((b & 1) ? Q6_V_vror_VR(voff_b, 64) : voff_b);

        HVX_सदिशम् r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_सदिशम् r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_सदिशम् r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_सदिशम् r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_सदिशम् r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_सदिशम् r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_सदिशम् r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_सदिशम् r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_सदिशम् * सीमित dst = (HVX_सदिशम् *) (y_खण्डम् + b * 1280);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
        dst[9] = r_विस्थापनम्;
    }
}

स्थिर अन्तःस्थ शून्य quantize_खण्डम्_f32_q8_1_tiled(भिन्न * सीमित x, uint8_t * सीमित y_खण्डम्) {
    quantize_खण्डम्_f32_q8_1_tiled_impl(x, y_खण्डम्, false, रिक्त);
}

स्थिर अन्तःस्थ शून्य quantize_खण्डम्_f32_q8_1_s16_tiled(भिन्न * सीमित x, uint8_t * सीमित y_खण्डम्, ध्रुव भिन्न * सीमित x_sb) {
    quantize_खण्डम्_f32_q8_1_tiled_impl(x, y_खण्डम्, true, x_sb);
}

स्थिर अन्तःस्थ शून्य quantize_खण्डम्_f32_q8_0_tiled(भिन्न * सीमित x, uint8_t * सीमित y_खण्डम्) {
    प्रतिज्ञा((अचिह्नित दीर्घ) x % 128 == 0);
    प्रतिज्ञा((अचिह्नित दीर्घ) y_खण्डम् % 128 == 0);

    HVX_सदिशम् * vx = (HVX_सदिशम् *) x;
    HVX_सदिशम् zero   = Q6_V_vzero();

    HVX_सदिशम् vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_सदिशम् vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_सदिशम् vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_सदिशम् vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_सदिशम् vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_सदिशम् vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_सदिशम् vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_सदिशम् vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_सदिशम् vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_सदिशम् vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_सदिशम् vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_सदिशम् vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_सदिशम् vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_सदिशम् vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_सदिशम् vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_सदिशम् vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_सदिशम् vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_सदिशम् vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_सदिशम् vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_सदिशम् vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_सदिशम् vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_सदिशम् vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_सदिशम् vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_सदिशम् vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_सदिशम् vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    HVX_सदिशम्_Pair vp01 = Q6_W_vshuff_VVR(vd01_hf, vd01_hf, -64);
    HVX_सदिशम्_Pair vp23 = Q6_W_vshuff_VVR(vd23_hf, vd23_hf, -64);

    स्थिर ध्रुव uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_सदिशम् v_repl_ctrl = * (ध्रुव HVX_सदिशम् *) repl;

    #निर्देशः unroll
    प्रति (पूर्णाङ्क b = 0; b < 4; b++) {
        HVX_सदिशम् v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_सदिशम् r_scale;
        यदि (b == 0) {
            r_scale = Q6_V_lo_W(vp01);
        } अन्यथा यदि (b == 1) {
            r_scale = Q6_V_hi_W(vp01);
        } अन्यथा यदि (b == 2) {
            r_scale = Q6_V_lo_W(vp23);
        } अन्यथा {
            r_scale = Q6_V_hi_W(vp23);
        }

        HVX_सदिशम् r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_सदिशम् r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_सदिशम् r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_सदिशम् r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_सदिशम् r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_सदिशम् r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_सदिशम् r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_सदिशम् r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_सदिशम् * सीमित dst = (HVX_सदिशम् *) (y_खण्डम् + b * 1152);
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

स्थिर शून्य quantize_row_f32_q8_0_tiled(भिन्न * सीमित x, uint8_t * सीमित y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (k + qk - 1) / qk;

    प्रति (uint32_t i = 0; i < nb; i++) {
        uint8_t * सीमित y_खण्डम् = y + i * 4 * 1152;
        quantize_खण्डम्_f32_q8_0_tiled(x + i * qk, y_खण्डम्);
    }
}

स्थिर शून्य quantize_row_f32_q8_1_tiled(भिन्न * सीमित x, uint8_t * सीमित y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (k + qk - 1) / qk;

    प्रति (uint32_t i = 0; i < nb; i++) {
        uint8_t * सीमित y_खण्डम् = y + i * 4 * 1280;
        quantize_खण्डम्_f32_q8_1_tiled(x + i * qk, y_खण्डम्);
    }
}

स्थिर शून्य quantize_row_f32_q8_1_s16_tiled(भिन्न * सीमित x, uint8_t * सीमित y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (k + qk - 1) / qk;

    प्रति (uint32_t i = 0; i < nb; i++) {
        uint8_t * सीमित y_खण्डम् = y + i * 4 * 1280;
        quantize_खण्डम्_f32_q8_1_s16_tiled(x + i * qk, y_खण्डम्, x + (i & ~1u) * qk);  // k % 256 == 0 for Q2_K
    }
}

स्थिर अन्तःस्थ शून्य quantize_half_f16_tiled_splat(HVX_सदिशम् v_src, HVX_सदिशम् * सीमित dst) {
    स्थिर ध्रुव uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_सदिशम् v_repl_ctrl = *(ध्रुव HVX_सदिशम् *) repl;

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क cp = 0; cp < 32; cp++) {
        dst[cp] = Q6_V_vdelta_VV(cp == 0 ? v_src : Q6_V_vror_VR(v_src, cp * 4), v_repl_ctrl);
    }
}

स्थिर अन्तःस्थ शून्य quantize_खण्डम्_f32_f16_tiled(ध्रुव भिन्न * सीमित x, uint8_t * सीमित y_खण्डम्) {
    प्रतिज्ञा((अचिह्नित दीर्घ) x % 128 == 0);
    प्रतिज्ञा((अचिह्नित दीर्घ) y_खण्डम् % 128 == 0);

    ध्रुव HVX_सदिशम् * vx = (ध्रुव HVX_सदिशम् *) x;
    HVX_सदिशम् * सीमित dst = (HVX_सदिशम् *) y_खण्डम्;

    HVX_सदिशम् vx01_hf = hvx_vec_f32_गन्तव्यम्_f16(vx[0], vx[1]);
    quantize_half_f16_tiled_splat(vx01_hf, dst);

    HVX_सदिशम् vx23_hf = hvx_vec_f32_गन्तव्यम्_f16(vx[2], vx[3]);
    quantize_half_f16_tiled_splat(vx23_hf, dst + 32);
}

स्थिर अन्तःस्थ शून्य quantize_खण्डम्_f16_f16_tiled(ध्रुव __fp16 * सीमित x, uint8_t * सीमित y_खण्डम्) {
    प्रतिज्ञा((अचिह्नित दीर्घ) x % 128 == 0);
    प्रतिज्ञा((अचिह्नित दीर्घ) y_खण्डम् % 128 == 0);

    ध्रुव HVX_सदिशम् * vx = (ध्रुव HVX_सदिशम् *) x;
    HVX_सदिशम् * सीमित dst = (HVX_सदिशम् *) y_खण्डम्;

    quantize_half_f16_tiled_splat(vx[0], dst);
    quantize_half_f16_tiled_splat(vx[1], dst + 32);
}

स्थिर शून्य quantize_row_f32_f16_tiled(ध्रुव भिन्न * सीमित x, uint8_t * सीमित y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (k + qk - 1) / qk;

    प्रति (uint32_t i = 0; i < nb; i++) {
        uint8_t * सीमित y_खण्डम् = y + i * 4 * 2048;
        quantize_खण्डम्_f32_f16_tiled(x + i * qk, y_खण्डम्);
    }
}

स्थिर शून्य quantize_row_f16_f16_tiled(ध्रुव __fp16 * सीमित x, uint8_t * सीमित y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (k + qk - 1) / qk;

    प्रति (uint32_t i = 0; i < nb; i++) {
        uint8_t * सीमित y_खण्डम् = y + i * 4 * 2048;
        quantize_खण्डम्_f16_f16_tiled(x + i * qk, y_खण्डम्);
    }
}

// Dot kernels & helpers that consume tiled activations

स्थिर अन्तःस्थ HVX_सदिशम् hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(HVX_सदिशम् v1, HVX_सदिशम् v2) {
#यदि __HVX_ARCH__ >= 79
    HVX_सदिशम्_Pair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    प्रत्यागम Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4));
#अन्यथा
    HVX_सदिशम्_Pair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_सदिशम् hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_सदिशम् lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    प्रत्यागम Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -4));
#यदि_अन्तः
}

// both halves of hvx_vec_mul_f16_f16_to_f32_lower32: lo = products of lanes 0..31, hi = lanes 32..63
स्थिर अन्तःस्थ HVX_सदिशम्_Pair hvx_vec_mul_f16_f16_गन्तव्यम्_f32_pair(HVX_सदिशम् v1, HVX_सदिशम् v2) {
#यदि __HVX_ARCH__ >= 79
    HVX_सदिशम्_Pair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    प्रत्यागम Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4);
#अन्यथा
    HVX_सदिशम्_Pair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_सदिशम् hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_सदिशम् lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    प्रत्यागम Q6_W_vshuff_VVR(hi, lo, -4);
#यदि_अन्तः
}

स्थिर अन्तःस्थ HVX_सदिशम् unpack_and_interleave_4bit(HVX_सदिशम् v_a, HVX_सदिशम् v_b, HVX_सदिशम् आवरणम्_h4) {
    HVX_सदिशम् v_W0 = Q6_V_vand_VV(v_a, आवरणम्_h4);
    HVX_सदिशम् v_W1 = Q6_Vub_vlsr_VubR(v_a, 4);
    HVX_सदिशम् v_W2 = Q6_V_vand_VV(v_b, आवरणम्_h4);
    HVX_सदिशम् v_W3 = Q6_Vub_vlsr_VubR(v_b, 4);

    HVX_सदिशम्_Pair v01_pair = Q6_W_vshuff_VVR(v_W1, v_W0, -1);
    HVX_सदिशम्_Pair v23_pair = Q6_W_vshuff_VVR(v_W3, v_W2, -1);
    HVX_सदिशम्_Pair v0123_pair = Q6_W_vshuff_VVR(Q6_V_lo_W(v23_pair), Q6_V_lo_W(v01_pair), -2);
    प्रत्यागम Q6_V_lo_W(v0123_pair);
}

स्थिर अन्तःस्थ HVX_सदिशम्_Pair unpack_and_interleave_4bit_x2(HVX_सदिशम् v_src, HVX_सदिशम् आवरणम्_h4) {
    HVX_सदिशम् v_lo = Q6_V_vand_VV(v_src, आवरणम्_h4);
    HVX_सदिशम् v_hi = Q6_Vub_vlsr_VubR(v_src, 4);
    HVX_सदिशम्_Pair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_सदिशम् v01_lo = Q6_V_lo_W(v01_pair);
    HVX_सदिशम् v01_hi = Q6_V_hi_W(v01_pair);

    HVX_सदिशम् v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_सदिशम् v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_सदिशम् v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_सदिशम् v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    प्रत्यागम Q6_W_vcombine_VV(v_W1, v_W0);
}

स्थिर अन्तःस्थ HVX_सदिशम् unpack_q1_0_गणः(
    HVX_सदिशम् v_plane,
    पूर्णाङ्क g,
    HVX_सदिशम् v_neg_one,
    HVX_सदिशम् v_pos_one
) {
    HVX_सदिशम्_Pred q = Q6_Q_vand_VR(v_plane, 0x01010101u << g);
    प्रत्यागम Q6_V_vmux_QVV(q, v_pos_one, v_neg_one);
}

स्थिर अन्तःस्थ HVX_सदिशम् accum_q1_0_32x1(
    HVX_सदिशम् v_plane,
    ध्रुव HVX_सदिशम् * सीमित v_act,
    HVX_सदिशम् v_neg_one,
    HVX_सदिशम् v_pos_one
) {
    HVX_सदिशम् v_योग_lo = Q6_V_vzero();
    HVX_सदिशम् v_योग_hi = Q6_V_vzero();

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q1_0_गणः(v_plane, g,     v_neg_one, v_pos_one);
        HVX_सदिशम् v_W_hi = unpack_q1_0_गणः(v_plane, g + 4, v_neg_one, v_pos_one);
        v_योग_lo = Q6_Vw_vrmpyacc_VwVbVb(v_योग_lo, v_W_lo, v_act[g]);
        v_योग_hi = Q6_Vw_vrmpyacc_VwVbVb(v_योग_hi, v_W_hi, v_act[g + 4]);
    }

    प्रत्यागम Q6_Vw_vadd_VwVw(v_योग_lo, v_योग_hi);
}

स्थिर अन्तःस्थ HVX_सदिशम् accum_4bit_32x1(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act,
    HVX_सदिशम् i8
) {
    HVX_सदिशम् v_sum0 = Q6_V_vzero();
    HVX_सदिशम् v_sum1 = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_सदिशम्_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], आवरणम्_h4);
        HVX_सदिशम् v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_सदिशम् v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    प्रत्यागम Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

स्थिर अन्तःस्थ HVX_सदिशम् accum_4bit_32x1_lut(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act,
    HVX_सदिशम् आवरणम्_h4,
    HVX_सदिशम् lut
) {
    HVX_सदिशम् v_sum0 = Q6_V_vzero();
    HVX_सदिशम् v_sum1 = Q6_V_vzero();

    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_सदिशम्_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], आवरणम्_h4);
        HVX_सदिशम् v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_सदिशम् v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    प्रत्यागम Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

स्थिर अन्तःस्थ HVX_सदिशम्_Pair accum_q1_0_32x2(
    HVX_सदिशम् v_plane,
    ध्रुव HVX_सदिशम् * सीमित v_act0,
    ध्रुव HVX_सदिशम् * सीमित v_act1,
    HVX_सदिशम् v_neg_one,
    HVX_सदिशम् v_pos_one
) {
    HVX_सदिशम् v_sum0_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum0_hi = Q6_V_vzero();
    HVX_सदिशम् v_sum1_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum1_hi = Q6_V_vzero();

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q1_0_गणः(v_plane, g,     v_neg_one, v_pos_one);
        HVX_सदिशम् v_W_hi = unpack_q1_0_गणः(v_plane, g + 4, v_neg_one, v_pos_one);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    प्रत्यागम Q6_W_vcombine_VV(
        Q6_Vw_vadd_VwVw(v_sum1_lo, v_sum1_hi),
        Q6_Vw_vadd_VwVw(v_sum0_lo, v_sum0_hi));
}

स्थिर अन्तःस्थ HVX_सदिशम्_Pair accum_4bit_32x2(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act0,
    ध्रुव HVX_सदिशम् * सीमित v_act1,
    HVX_सदिशम् i8
) {
    HVX_सदिशम् v_sum0 = Q6_V_vzero();
    HVX_सदिशम् v_sum1 = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_सदिशम्_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], आवरणम्_h4);
        HVX_सदिशम् v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_सदिशम् v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    प्रत्यागम Q6_W_vcombine_VV(v_sum1, v_sum0);
}

स्थिर अन्तःस्थ HVX_सदिशम्_Pair accum_4bit_32x2_lut(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act0,
    ध्रुव HVX_सदिशम् * सीमित v_act1,
    HVX_सदिशम् आवरणम्_h4,
    HVX_सदिशम् lut
) {
    HVX_सदिशम् v_sum0 = Q6_V_vzero();
    HVX_सदिशम् v_sum1 = Q6_V_vzero();

    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_सदिशम्_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], आवरणम्_h4);
        HVX_सदिशम् v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_सदिशम् v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    प्रत्यागम Q6_W_vcombine_VV(v_sum1, v_sum0);
}

स्थिर अन्तःस्थ HVX_सदिशम् accum_q8_0_32x1(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act
) {
    HVX_सदिशम् v_योग = Q6_V_vzero();
    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 8; g++) {
        HVX_सदिशम् v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_सदिशम् v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_योग = Q6_Vw_vrmpyacc_VwVbVb(v_योग, v_W, v_act[g]);
    }
    प्रत्यागम v_योग;
}

स्थिर अन्तःस्थ HVX_सदिशम्_Pair accum_q8_0_32x2(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act0,
    ध्रुव HVX_सदिशम् * सीमित v_act1
) {
    HVX_सदिशम् v_sum0 = Q6_V_vzero();
    HVX_सदिशम् v_sum1 = Q6_V_vzero();
    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 8; g++) {
        HVX_सदिशम् v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_सदिशम् v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W, v_act0[g]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W, v_act1[g]);
    }
    प्रत्यागम Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q5_K: OR 0x10 into every lane of v whose flag j is set in the plane (see HTP_TENSOR_TILE_SIZE_Q5_K)
स्थिर अन्तःस्थ HVX_सदिशम् hvx_q5k_or_hibit(HVX_सदिशम् v, HVX_सदिशम् v_plane, पूर्णाङ्क j) {
    HVX_सदिशम्_Pred q = Q6_Q_vand_VR(v_plane, 0x01010101u << j);
    प्रत्यागम Q6_V_vandor_VQR(v, q, 0x10101010);
}

// 5-bit variant: the high bit comes from the plane, see hvx_q5k_or_hibit
स्थिर अन्तःस्थ HVX_सदिशम्_Pair unpack_and_interleave_5bit_x2(HVX_सदिशम् v_src, HVX_सदिशम् v_plane, पूर्णाङ्क i, HVX_सदिशम् आवरणम्_h4) {
    HVX_सदिशम् v_lo = hvx_q5k_or_hibit(Q6_V_vand_VV(v_src, आवरणम्_h4), v_plane, 2 * i);
    HVX_सदिशम् v_hi = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(v_src, 4), v_plane, 2 * i + 1);
    HVX_सदिशम्_Pair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_सदिशम् v01_lo = Q6_V_lo_W(v01_pair);
    HVX_सदिशम् v01_hi = Q6_V_hi_W(v01_pair);

    HVX_सदिशम् v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_सदिशम् v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_सदिशम् v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_सदिशम् v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    प्रत्यागम Q6_W_vcombine_VV(v_W1, v_W0);
}

स्थिर अन्तःस्थ HVX_सदिशम् accum_5bit_32x1(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act,
    HVX_सदिशम् i8
) {
    HVX_सदिशम् v_sum0 = Q6_V_vzero();
    HVX_सदिशम् v_sum1 = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_सदिशम् v_plane = vptr[5];

    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_सदिशम्_Pair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, आवरणम्_h4);
        HVX_सदिशम् v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_सदिशम् v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    प्रत्यागम Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

स्थिर अन्तःस्थ HVX_सदिशम्_Pair accum_5bit_32x2(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act0,
    ध्रुव HVX_सदिशम् * सीमित v_act1,
    HVX_सदिशम् i8
) {
    HVX_सदिशम् v_sum0 = Q6_V_vzero();
    HVX_सदिशम् v_sum1 = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_सदिशम् v_plane = vptr[5];

    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_सदिशम्_Pair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, आवरणम्_h4);
        HVX_सदिशम् v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_सदिशम् v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    प्रत्यागम Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q6_K weights are stored unsigned (0..63), see HTP_TENSOR_TILE_SIZE_Q6_K. Unpack k-group g of a tile to signed bytes (q - 32)
स्थिर अन्तःस्थ HVX_सदिशम् unpack_q6_k_गणः(ध्रुव HVX_सदिशम् * सीमित vptr, पूर्णाङ्क g, HVX_सदिशम् आवरणम्_0f, HVX_सदिशम् आवरणम्_03, HVX_सदिशम् i32) {
    HVX_सदिशम् v_lo = (g & 1) ? Q6_Vub_vlsr_VubR(vptr[g >> 1], 4) : Q6_V_vand_VV(vptr[g >> 1], आवरणम्_0f);
    HVX_सदिशम् v_hi = (g & 3) ? Q6_Vub_vlsr_VubR(vptr[4 + (g >> 2)], 2 * (g & 3)) : vptr[4 + (g >> 2)];
    HVX_सदिशम् v_q  = Q6_V_vor_VV(v_lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_hi, आवरणम्_03), 4));
    प्रत्यागम Q6_Vb_vsub_VbVb(v_q, i32);
}

// k 0..15 and k 16..31 of a Q6_K tile have different scales: lo half of the pair sums k 0..15, hi half sums k 16..31
स्थिर अन्तःस्थ HVX_सदिशम्_Pair accum_q6_k_32x1(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act,
    HVX_सदिशम् i32
) {
    HVX_सदिशम् v_योग_lo = Q6_V_vzero();
    HVX_सदिशम् v_योग_hi = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_सदिशम् आवरणम्_03 = Q6_Vb_vsplat_R(0x03);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q6_k_गणः(vptr, g,     आवरणम्_0f, आवरणम्_03, i32);
        HVX_सदिशम् v_W_hi = unpack_q6_k_गणः(vptr, g + 4, आवरणम्_0f, आवरणम्_03, i32);
        v_योग_lo = Q6_Vw_vrmpyacc_VwVbVb(v_योग_lo, v_W_lo, v_act[g]);
        v_योग_hi = Q6_Vw_vrmpyacc_VwVbVb(v_योग_hi, v_W_hi, v_act[g + 4]);
    }

    प्रत्यागम Q6_W_vcombine_VV(v_योग_hi, v_योग_lo);
}

स्थिर अन्तःस्थ शून्य accum_q6_k_32x2(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act0,
    ध्रुव HVX_सदिशम् * सीमित v_act1,
    HVX_सदिशम् i32,
    HVX_सदिशम्_Pair * v_sums0,
    HVX_सदिशम्_Pair * v_sums1
) {
    HVX_सदिशम् v_sum0_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum0_hi = Q6_V_vzero();
    HVX_सदिशम् v_sum1_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum1_hi = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_सदिशम् आवरणम्_03 = Q6_Vb_vsplat_R(0x03);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q6_k_गणः(vptr, g,     आवरणम्_0f, आवरणम्_03, i32);
        HVX_सदिशम् v_W_hi = unpack_q6_k_गणः(vptr, g + 4, आवरणम्_0f, आवरणम्_03, i32);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// scale the two half sums with the per-row tile scales (v_scale_w = vptr[6]) and the activation scale
स्थिर अन्तःस्थ HVX_सदिशम् scale_q6_k_32x1(HVX_सदिशम्_Pair v_sums, HVX_सदिशम् v_scale_w, HVX_सदिशम् v_scale_a) {
    HVX_सदिशम्_Pair v_scale = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_pair(v_scale_w, v_scale_a);
    HVX_सदिशम् v_lo = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums)), Q6_V_lo_W(v_scale));
    HVX_सदिशम् v_hi = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums)), Q6_V_hi_W(v_scale));
    प्रत्यागम hvx_vec_सङ्कलय_f32_f32(v_lo, v_hi);
}

// Q3_K / Q2_K: low 2 bits of k-group g, see HTP_TENSOR_TILE_SIZE_Q3_K
स्थिर अन्तःस्थ HVX_सदिशम् unpack_q3_k_low2(ध्रुव HVX_सदिशम् * सीमित vptr, पूर्णाङ्क g, HVX_सदिशम् आवरणम्_03) {
    HVX_सदिशम् v = vptr[g >> 2];
    यदि ((g & 3) == 3) {
        प्रत्यागम Q6_Vub_vlsr_VubR(v, 6);
    }
    यदि (g & 3) {
        v = Q6_Vub_vlsr_VubR(v, 2 * (g & 3));
    }
    प्रत्यागम Q6_V_vand_VV(v, आवरणम्_03);
}

// Q3_K k-group g as signed bytes: low2 | 0xFC (= low2 - 4) where bit g of vector 2 is set
स्थिर अन्तःस्थ HVX_सदिशम् unpack_q3_k_गणः(ध्रुव HVX_सदिशम् * सीमित vptr, पूर्णाङ्क g, HVX_सदिशम् आवरणम्_03) {
    HVX_सदिशम्_Pred q_neg = Q6_Q_vand_VR(vptr[2], 0x01010101u << g);
    प्रत्यागम Q6_V_vandor_VQR(unpack_q3_k_low2(vptr, g, आवरणम्_03), q_neg, 0xFCFCFCFC);
}

// same half split as accum_q6_k_32x1
स्थिर अन्तःस्थ HVX_सदिशम्_Pair accum_q3_k_32x1(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act
) {
    HVX_सदिशम् v_योग_lo = Q6_V_vzero();
    HVX_सदिशम् v_योग_hi = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_03  = Q6_Vb_vsplat_R(0x03);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q3_k_गणः(vptr, g,     आवरणम्_03);
        HVX_सदिशम् v_W_hi = unpack_q3_k_गणः(vptr, g + 4, आवरणम्_03);
        v_योग_lo = Q6_Vw_vrmpyacc_VwVbVb(v_योग_lo, v_W_lo, v_act[g]);
        v_योग_hi = Q6_Vw_vrmpyacc_VwVbVb(v_योग_hi, v_W_hi, v_act[g + 4]);
    }

    प्रत्यागम Q6_W_vcombine_VV(v_योग_hi, v_योग_lo);
}

स्थिर अन्तःस्थ शून्य accum_q3_k_32x2(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act0,
    ध्रुव HVX_सदिशम् * सीमित v_act1,
    HVX_सदिशम्_Pair * v_sums0,
    HVX_सदिशम्_Pair * v_sums1
) {
    HVX_सदिशम् v_sum0_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum0_hi = Q6_V_vzero();
    HVX_सदिशम् v_sum1_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum1_hi = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_03   = Q6_Vb_vsplat_R(0x03);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q3_k_गणः(vptr, g,     आवरणम्_03);
        HVX_सदिशम् v_W_hi = unpack_q3_k_गणः(vptr, g + 4, आवरणम्_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// Q2_K: the Q3_K dot products without the -4 flags, scales and mins are applied in q2_k_int_accum
स्थिर अन्तःस्थ HVX_सदिशम्_Pair accum_q2_k_32x1(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act
) {
    HVX_सदिशम् v_योग_lo = Q6_V_vzero();
    HVX_सदिशम् v_योग_hi = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_03  = Q6_Vb_vsplat_R(0x03);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q3_k_low2(vptr, g,     आवरणम्_03);
        HVX_सदिशम् v_W_hi = unpack_q3_k_low2(vptr, g + 4, आवरणम्_03);
        v_योग_lo = Q6_Vw_vrmpyacc_VwVbVb(v_योग_lo, v_W_lo, v_act[g]);
        v_योग_hi = Q6_Vw_vrmpyacc_VwVbVb(v_योग_hi, v_W_hi, v_act[g + 4]);
    }

    प्रत्यागम Q6_W_vcombine_VV(v_योग_hi, v_योग_lo);
}

स्थिर अन्तःस्थ शून्य accum_q2_k_32x2(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act0,
    ध्रुव HVX_सदिशम् * सीमित v_act1,
    HVX_सदिशम्_Pair * v_sums0,
    HVX_सदिशम्_Pair * v_sums1
) {
    HVX_सदिशम् v_sum0_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum0_hi = Q6_V_vzero();
    HVX_सदिशम् v_sum1_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum1_hi = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_03   = Q6_Vb_vsplat_R(0x03);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q3_k_low2(vptr, g,     आवरणम्_03);
        HVX_सदिशम् v_W_hi = unpack_q3_k_low2(vptr, g + 4, आवरणम्_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// Q2_0: unsigned codes 0..3; the -1 offset is applied via the Q8_1 block sum (v_act[9])
स्थिर अन्तःस्थ HVX_सदिशम् accum_q2_0_32x1(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act
) {
    HVX_सदिशम् v_योग_lo = Q6_V_vzero();
    HVX_सदिशम् v_योग_hi = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_03  = Q6_Vb_vsplat_R(0x03);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q3_k_low2(vptr, g,     आवरणम्_03);
        HVX_सदिशम् v_W_hi = unpack_q3_k_low2(vptr, g + 4, आवरणम्_03);
        v_योग_lo = Q6_Vw_vrmpyacc_VwVubVb(v_योग_lo, v_W_lo, v_act[g]);
        v_योग_hi = Q6_Vw_vrmpyacc_VwVubVb(v_योग_hi, v_W_hi, v_act[g + 4]);
    }

    प्रत्यागम Q6_Vw_vadd_VwVw(v_योग_lo, v_योग_hi);
}

स्थिर अन्तःस्थ शून्य accum_q2_0_32x2(
    ध्रुव HVX_सदिशम् * सीमित vptr,
    ध्रुव HVX_सदिशम् * सीमित v_act0,
    ध्रुव HVX_सदिशम् * सीमित v_act1,
    HVX_सदिशम् * v_sum0,
    HVX_सदिशम् * v_sum1
) {
    HVX_सदिशम् v_sum0_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum0_hi = Q6_V_vzero();
    HVX_सदिशम् v_sum1_lo = Q6_V_vzero();
    HVX_सदिशम् v_sum1_hi = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_03   = Q6_Vb_vsplat_R(0x03);

    #निर्देशः unroll
    प्रति (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_सदिशम् v_W_lo = unpack_q3_k_low2(vptr, g,     आवरणम्_03);
        HVX_सदिशम् v_W_hi = unpack_q3_k_low2(vptr, g + 4, आवरणम्_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVubVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVubVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVubVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVubVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sum0 = Q6_Vw_vadd_VwVw(v_sum0_lo, v_sum0_hi);
    *v_sum1 = Q6_Vw_vadd_VwVw(v_sum1_lo, v_sum1_hi);
}

// Q2_K: fp16 d (*v_d) and -dmin (*v_m) of the 32 rows of a super-block (8 tiles, VTCM stride 384), each in both vector halves
स्थिर अन्तःस्थ शून्य q2_k_आरोपय_dmin(ध्रुव uint8_t * सीमित sb, HVX_सदिशम् * सीमित v_d, HVX_सदिशम् * सीमित v_m) {
    HVX_सदिशम् v_dd = Q6_V_vzero();  // lanes 0..31: d, lanes 32..63: -dmin

    #निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < 8; j++) {
        HVX_सदिशम् v = hvx_vmem(sb + j * 384 + 256);  // the slice is at bytes 64..79
        v_dd = Q6_V_vmux_QVV(Q6_Q_vsetq_R(16 * j), v_dd, Q6_V_vror_VR(v, (64 - 16 * j) & 127));
    }

    HVX_सदिशम्_Pred q_lo = Q6_Q_vsetq_R(64);
    HVX_सदिशम्     v_r  = Q6_V_vror_VR(v_dd, 64);
    *v_d = Q6_V_vmux_QVV(q_lo, v_dd, v_r);
    *v_m = Q6_V_vmux_QVV(q_lo, v_r, v_dd);
}

// Q2_K: fp16 D = d * scale and M = -dmin * min of a tile from its scale bytes, in the lanes of the Q6_K scales
स्थिर अन्तःस्थ शून्य q2_k_unpack_scales(ध्रुव uint8_t * सीमित tile, HVX_सदिशम् v_d, HVX_सदिशम् v_m,
                                      HVX_सदिशम् * सीमित v_D, HVX_सदिशम् * सीमित v_M) {
    HVX_सदिशम् v_sc = hvx_vmem(tile + 256);
    HVX_सदिशम् v_l  = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(Q6_V_vand_VV(v_sc, Q6_Vb_vsplat_R(0x0F))));
    HVX_सदिशम् v_mn = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(Q6_Vub_vlsr_VubR(v_sc, 4)));
    // qf32 product: rounds like GGML_FP32_TO_FP16(d * scale) (qf16 is off by 1 ulp for many values on v73)
    *v_D = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(v_l), v_d));
    *v_M = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(v_mn), v_m));
}

// Q2_K scale and min nibbles of a tile as halfword pairs: word r = sub-block 1 << 16 | sub-block 0 of row r
स्थिर अन्तःस्थ शून्य q2_k_scale_युग्मानि(ध्रुव uint8_t * सीमित tile, HVX_सदिशम् * सीमित v_sc2, HVX_सदिशम् * सीमित v_mn2) {
    HVX_सदिशम् v_h = Q6_Vh_vshuff_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile + 256))));
    *v_sc2 = Q6_V_vand_VV(v_h, Q6_Vh_vsplat_R(0x000F));
    *v_mn2 = Q6_Vuh_vlsr_VuhR(v_h, 4);
}

// Q2_K with q8_1_s16 activations, one tile: isum += scale * (q . a), msum += min * sum(a), per row, in int32.
// |sum of 16 q * a| <= 3 * 128 * 16 fits a halfword, so both sub-blocks go through one vdmpy (no saturation)
स्थिर अन्तःस्थ शून्य q2_k_int_accum(HVX_सदिशम्_Pair v_sums, HVX_सदिशम् v_sc2, HVX_सदिशम् v_mn2, HVX_सदिशम् v_bsum,
                                  HVX_सदिशम् * सीमित v_isum, HVX_सदिशम् * सीमित v_msum) {
    HVX_सदिशम् v_s2 = Q6_Vh_vshuffe_VhVh(Q6_V_hi_W(v_sums), Q6_V_lo_W(v_sums));
    *v_isum = Q6_Vw_vadd_VwVw(*v_isum, Q6_Vw_vdmpy_VhVh_sat(v_s2, v_sc2));
    *v_msum = Q6_Vw_vadd_VwVw(*v_msum, Q6_Vw_vdmpy_VhVh_sat(v_mn2, v_bsum));
}

// Q2_K super-block result: d_a * (d * isum - dmin * msum), with v_d / v_m the fp16 d / -dmin of q2_k_load_dmin
स्थिर अन्तःस्थ HVX_सदिशम् q2_k_int_scale(HVX_सदिशम् v_isum, HVX_सदिशम् v_msum, HVX_सदिशम् v_d, HVX_सदिशम् v_m, HVX_सदिशम् v_act8) {
    HVX_सदिशम् v_d32  = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(v_d));
    HVX_सदिशम् v_m32  = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(v_m));
    HVX_सदिशम् v_da32 = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(v_act8));
    HVX_सदिशम् v_t = hvx_vec_सङ्कलय_f32_f32(hvx_vec_mul_f32_f32(v_d32, Q6_Vsf_equals_Vw(v_isum)),
                                         hvx_vec_mul_f32_f32(v_m32, Q6_Vsf_equals_Vw(v_msum)));
    प्रत्यागम hvx_vec_mul_f32_f32(v_t, v_da32);
}

स्थिर शून्य tiled_vec_dot_q1_0_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_qf  = Q6_V_vzero();
    HVX_सदिशम् v_neg_one = Q6_Vb_vsplat_R(-1);
    HVX_सदिशम् v_pos_one = Q6_Vb_vsplat_R(1);

    ध्रुव uint32_t n_k_tiles = n / HTP_TENSOR_TILE_K_Q1_0;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * htp_tensor_get_tile_padded_आकार(HTP_TYPE_Q1_0));
        HVX_सदिशम् v_tile_qf = Q6_V_vzero();

        #निर्देशः unroll
        प्रति (पूर्णाङ्क quarter = 0; quarter < 4; quarter++) {
            ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + (4 * kt + quarter) * HTP_MM_ACT_TILE_आकार_Q8_0);
            HVX_सदिशम् v_योग_sf = Q6_Vsf_equals_Vw(accum_q1_0_32x1(vptr[quarter], v_act, v_neg_one, v_pos_one));
            HVX_सदिशम् v_da_sf  = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32_shuff(v_act[8]));
            HVX_सदिशम् v_q_qf   = Q6_Vqf32_vmpy_VsfVsf(v_योग_sf, v_da_sf);
            v_tile_qf = (quarter == 0) ? v_q_qf : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf, v_q_qf);
        }

        HVX_सदिशम् v_dw_sf = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(vptr[4]));
        v_योग_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_योग_qf, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf), v_dw_sf));
    }

    HVX_सदिशम् v_योग_float = Q6_Vsf_equals_Vqf32(v_योग_qf);
    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_q1_0_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_qf_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_qf_c1 = Q6_V_vzero();
    HVX_सदिशम् v_neg_one   = Q6_Vb_vsplat_R(-1);
    HVX_सदिशम् v_pos_one   = Q6_Vb_vsplat_R(1);

    ध्रुव uint32_t n_k_tiles = n / HTP_TENSOR_TILE_K_Q1_0;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * htp_tensor_get_tile_padded_आकार(HTP_TYPE_Q1_0));
        HVX_सदिशम् v_tile_qf_c0 = Q6_V_vzero();
        HVX_सदिशम् v_tile_qf_c1 = Q6_V_vzero();

        प्रति (पूर्णाङ्क quarter = 0; quarter < 4; quarter++) {
            ध्रुव uint32_t act_tile = 4 * kt + quarter;
            ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + act_tile * HTP_MM_ACT_TILE_आकार_Q8_0);
            ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + act_tile * HTP_MM_ACT_TILE_आकार_Q8_0);
            HVX_सदिशम्_Pair v_sums = accum_q1_0_32x2(vptr[quarter], v_act0, v_act1, v_neg_one, v_pos_one);
            HVX_सदिशम् v_योग_sf_c0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums));
            HVX_सदिशम् v_योग_sf_c1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums));
            HVX_सदिशम् v_da_sf_c0  = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32_shuff(v_act0[8]));
            HVX_सदिशम् v_da_sf_c1  = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32_shuff(v_act1[8]));

            v_tile_qf_c0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c0, Q6_Vqf32_vmpy_VsfVsf(v_योग_sf_c0, v_da_sf_c0));
            v_tile_qf_c1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c1, Q6_Vqf32_vmpy_VsfVsf(v_योग_sf_c1, v_da_sf_c1));
        }

        HVX_सदिशम् v_dw_sf = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(vptr[4]));
        v_योग_qf_c0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_योग_qf_c0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf_c0), v_dw_sf));
        v_योग_qf_c1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_योग_qf_c1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf_c1), v_dw_sf));
    }

    HVX_सदिशम् v_योग_float_c0 = Q6_Vsf_equals_Vqf32(v_योग_qf_c0);
    HVX_सदिशम् v_योग_float_c1 = Q6_Vsf_equals_Vqf32(v_योग_qf_c1);
    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q4_0_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();
    HVX_सदिशम् i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 640);
        ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + kt * 1152);

        HVX_सदिशम् v_योग = accum_4bit_32x1(vptr, v_act, i8);
        HVX_सदिशम् v_योग_sf = Q6_Vsf_equals_Vw(v_योग);

        HVX_सदिशम् v_scale_w = vptr[4];
        HVX_सदिशम् v_scale_a = v_act[8];
        HVX_सदिशम् v_scale_comb = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale_w, v_scale_a);
        HVX_सदिशम् v_योग_scaled = hvx_vec_mul_f32_f32(v_योग_sf, v_scale_comb);

        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, v_योग_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_q4_0_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();
    HVX_सदिशम् i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 640);
        ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + kt * 1152);
        ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + kt * 1152);

        HVX_सदिशम्_Pair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, i8);
        HVX_सदिशम् v_योग_c0 = Q6_V_lo_W(v_sums);
        HVX_सदिशम् v_योग_c1 = Q6_V_hi_W(v_sums);

        HVX_सदिशम् v_योग_sf_c0 = Q6_Vsf_equals_Vw(v_योग_c0);
        HVX_सदिशम् v_योग_sf_c1 = Q6_Vsf_equals_Vw(v_योग_c1);

        HVX_सदिशम् v_scale_w = vptr[4];
        HVX_सदिशम् v_scale_a_c0 = v_act0[8];
        HVX_सदिशम् v_scale_a_c1 = v_act1[8];

        HVX_सदिशम् v_scale_comb_c0 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_सदिशम् v_scale_comb_c1 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_सदिशम् v_योग_scaled_c0 = hvx_vec_mul_f32_f32(v_योग_sf_c0, v_scale_comb_c0);
        HVX_सदिशम् v_योग_scaled_c1 = hvx_vec_mul_f32_f32(v_योग_sf_c1, v_scale_comb_c1);

        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, v_योग_scaled_c0);
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, v_योग_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q4_1_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 640);
        ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + kt * 1280);

        HVX_सदिशम् v_योग = accum_4bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_सदिशम् v_योग_sf = Q6_Vsf_equals_Vw(v_योग);

        HVX_सदिशम् v_scale_विस्थापनम् = vptr[4];
        HVX_सदिशम्_Pair p_deal = Q6_W_vdeal_VVR(v_scale_विस्थापनम्, v_scale_विस्थापनम्, -2);
        HVX_सदिशम् v_scale = Q6_V_lo_W(p_deal);
        HVX_सदिशम् v_विस्थापनम् = Q6_V_hi_W(p_deal);

        HVX_सदिशम् v_scale_a = v_act[8];
        HVX_सदिशम् v_योग_a   = v_act[9];

        HVX_सदिशम् v_scale_comb = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale, v_scale_a);
        HVX_सदिशम् v_विस्थापनम्_comb = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_विस्थापनम्, v_योग_a);

        HVX_सदिशम् v_scaled_dot = hvx_vec_mul_f32_f32(v_योग_sf, v_scale_comb);
        HVX_सदिशम् v_योग_scaled = hvx_vec_सङ्कलय_f32_f32(v_scaled_dot, v_विस्थापनम्_comb);

        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, v_योग_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_q4_1_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 640);
        ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + kt * 1280);
        ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + kt * 1280);

        HVX_सदिशम्_Pair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_सदिशम् v_योग_c0 = Q6_V_lo_W(v_sums);
        HVX_सदिशम् v_योग_c1 = Q6_V_hi_W(v_sums);

        HVX_सदिशम् v_योग_sf_c0 = Q6_Vsf_equals_Vw(v_योग_c0);
        HVX_सदिशम् v_योग_sf_c1 = Q6_Vsf_equals_Vw(v_योग_c1);

        HVX_सदिशम् v_scale_विस्थापनम् = vptr[4];
        HVX_सदिशम्_Pair p_deal = Q6_W_vdeal_VVR(v_scale_विस्थापनम्, v_scale_विस्थापनम्, -2);
        HVX_सदिशम् v_scale = Q6_V_lo_W(p_deal);
        HVX_सदिशम् v_विस्थापनम् = Q6_V_hi_W(p_deal);

        HVX_सदिशम् v_scale_a_c0 = v_act0[8];
        HVX_सदिशम् v_योग_a_c0   = v_act0[9];
        HVX_सदिशम् v_scale_a_c1 = v_act1[8];
        HVX_सदिशम् v_योग_a_c1   = v_act1[9];

        HVX_सदिशम् v_scale_comb_c0 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale, v_scale_a_c0);
        HVX_सदिशम् v_विस्थापनम्_comb_c0 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_विस्थापनम्, v_योग_a_c0);
        HVX_सदिशम् v_scale_comb_c1 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale, v_scale_a_c1);
        HVX_सदिशम् v_विस्थापनम्_comb_c1 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_विस्थापनम्, v_योग_a_c1);

        HVX_सदिशम् v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_योग_sf_c0, v_scale_comb_c0);
        HVX_सदिशम् v_योग_scaled_c0 = hvx_vec_सङ्कलय_f32_f32(v_scaled_dot_c0, v_विस्थापनम्_comb_c0);

        HVX_सदिशम् v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_योग_sf_c1, v_scale_comb_c1);
        HVX_सदिशम् v_योग_scaled_c1 = hvx_vec_सङ्कलय_f32_f32(v_scaled_dot_c1, v_विस्थापनम्_comb_c1);

        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, v_योग_scaled_c0);
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, v_योग_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q8_0_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 1152);
        ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + kt * 1152);

        HVX_सदिशम् v_योग = accum_q8_0_32x1(vptr, v_act);
        HVX_सदिशम् v_योग_sf = Q6_Vsf_equals_Vw(v_योग);

        HVX_सदिशम् v_scale_w = vptr[8];
        HVX_सदिशम् v_scale_a = v_act[8];
        HVX_सदिशम् v_scale_comb = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale_w, v_scale_a);
        HVX_सदिशम् v_योग_scaled = hvx_vec_mul_f32_f32(v_योग_sf, v_scale_comb);

        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, v_योग_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_q8_0_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 1152);
        ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + kt * 1152);
        ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + kt * 1152);

        HVX_सदिशम्_Pair v_sums = accum_q8_0_32x2(vptr, v_act0, v_act1);
        HVX_सदिशम् v_योग_c0 = Q6_V_lo_W(v_sums);
        HVX_सदिशम् v_योग_c1 = Q6_V_hi_W(v_sums);

        HVX_सदिशम् v_योग_sf_c0 = Q6_Vsf_equals_Vw(v_योग_c0);
        HVX_सदिशम् v_योग_sf_c1 = Q6_Vsf_equals_Vw(v_योग_c1);

        HVX_सदिशम् v_scale_w = vptr[8];
        HVX_सदिशम् v_scale_a_c0 = v_act0[8];
        HVX_सदिशम् v_scale_a_c1 = v_act1[8];

        HVX_सदिशम् v_scale_comb_c0 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_सदिशम् v_scale_comb_c1 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_सदिशम् v_योग_scaled_c0 = hvx_vec_mul_f32_f32(v_योग_sf_c0, v_scale_comb_c0);
        HVX_सदिशम् v_योग_scaled_c1 = hvx_vec_mul_f32_f32(v_योग_sf_c1, v_scale_comb_c1);

        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, v_योग_scaled_c0);
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, v_योग_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q5_k_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 768);
        ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + kt * 1280);

        HVX_सदिशम् v_योग = accum_5bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_सदिशम् v_योग_sf = Q6_Vsf_equals_Vw(v_योग);

        HVX_सदिशम् v_scale_विस्थापनम् = vptr[4];
        HVX_सदिशम्_Pair p_deal = Q6_W_vdeal_VVR(v_scale_विस्थापनम्, v_scale_विस्थापनम्, -2);
        HVX_सदिशम् v_scale = Q6_V_lo_W(p_deal);
        HVX_सदिशम् v_विस्थापनम् = Q6_V_hi_W(p_deal);

        HVX_सदिशम् v_scale_a = v_act[8];
        HVX_सदिशम् v_योग_a   = v_act[9];

        HVX_सदिशम् v_scale_comb = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale, v_scale_a);
        HVX_सदिशम् v_विस्थापनम्_comb = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_विस्थापनम्, v_योग_a);

        HVX_सदिशम् v_scaled_dot = hvx_vec_mul_f32_f32(v_योग_sf, v_scale_comb);
        HVX_सदिशम् v_योग_scaled = hvx_vec_सङ्कलय_f32_f32(v_scaled_dot, v_विस्थापनम्_comb);

        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, v_योग_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_q5_k_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 768);
        ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + kt * 1280);
        ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + kt * 1280);

        HVX_सदिशम्_Pair v_sums = accum_5bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_सदिशम् v_योग_c0 = Q6_V_lo_W(v_sums);
        HVX_सदिशम् v_योग_c1 = Q6_V_hi_W(v_sums);

        HVX_सदिशम् v_योग_sf_c0 = Q6_Vsf_equals_Vw(v_योग_c0);
        HVX_सदिशम् v_योग_sf_c1 = Q6_Vsf_equals_Vw(v_योग_c1);

        HVX_सदिशम् v_scale_विस्थापनम् = vptr[4];
        HVX_सदिशम्_Pair p_deal = Q6_W_vdeal_VVR(v_scale_विस्थापनम्, v_scale_विस्थापनम्, -2);
        HVX_सदिशम् v_scale = Q6_V_lo_W(p_deal);
        HVX_सदिशम् v_विस्थापनम् = Q6_V_hi_W(p_deal);

        HVX_सदिशम् v_scale_a_c0 = v_act0[8];
        HVX_सदिशम् v_योग_a_c0   = v_act0[9];
        HVX_सदिशम् v_scale_a_c1 = v_act1[8];
        HVX_सदिशम् v_योग_a_c1   = v_act1[9];

        HVX_सदिशम् v_scale_comb_c0 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale, v_scale_a_c0);
        HVX_सदिशम् v_विस्थापनम्_comb_c0 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_विस्थापनम्, v_योग_a_c0);
        HVX_सदिशम् v_scale_comb_c1 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale, v_scale_a_c1);
        HVX_सदिशम् v_विस्थापनम्_comb_c1 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_विस्थापनम्, v_योग_a_c1);

        HVX_सदिशम् v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_योग_sf_c0, v_scale_comb_c0);
        HVX_सदिशम् v_योग_scaled_c0 = hvx_vec_सङ्कलय_f32_f32(v_scaled_dot_c0, v_विस्थापनम्_comb_c0);

        HVX_सदिशम् v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_योग_sf_c1, v_scale_comb_c1);
        HVX_सदिशम् v_योग_scaled_c1 = hvx_vec_सङ्कलय_f32_f32(v_scaled_dot_c1, v_विस्थापनम्_comb_c1);

        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, v_योग_scaled_c0);
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, v_योग_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q6_k_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();
    HVX_सदिशम् i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 896);
        ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + kt * 1152);

        HVX_सदिशम्_Pair v_sums = accum_q6_k_32x1(vptr, v_act, i32);
        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, scale_q6_k_32x1(v_sums, vptr[6], v_act[8]));
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_q6_k_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();
    HVX_सदिशम् i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 896);
        ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + kt * 1152);
        ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + kt * 1152);

        HVX_सदिशम्_Pair v_sums0, v_sums1;
        accum_q6_k_32x2(vptr, v_act0, v_act1, i32, &v_sums0, &v_sums1);

        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, scale_q6_k_32x1(v_sums0, vptr[6], v_act0[8]));
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, scale_q6_k_32x1(v_sums1, vptr[6], v_act1[8]));
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q3_k_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 512);
        ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + kt * 1152);

        HVX_सदिशम्_Pair v_sums = accum_q3_k_32x1(vptr, v_act);
        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, scale_q6_k_32x1(v_sums, vptr[3], v_act[8]));
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_q3_k_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 512);
        ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + kt * 1152);
        ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + kt * 1152);

        HVX_सदिशम्_Pair v_sums0, v_sums1;
        accum_q3_k_32x2(vptr, v_act0, v_act1, &v_sums0, &v_sums1);

        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, scale_q6_k_32x1(v_sums0, vptr[3], v_act0[8]));
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, scale_q6_k_32x1(v_sums1, vptr[3], v_act1[8]));
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q2_k_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kb = 0; kb < n_k_tiles; kb += 8) {
        HVX_सदिशम् v_d, v_m;
        q2_k_आरोपय_dmin(tile_ptr + kb * 384, &v_d, &v_m);

        HVX_सदिशम् v_isum = Q6_V_vzero();
        HVX_सदिशम् v_msum = Q6_V_vzero();
        प्रति (uint32_t kt = kb; kt < kb + 8; kt++) {
            ध्रुव uint8_t * सीमित tile = tile_ptr + kt * 384;
            ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + kt * 1280);

            HVX_सदिशम् v_sc2, v_mn2;
            q2_k_scale_युग्मानि(tile, &v_sc2, &v_mn2);
            q2_k_int_accum(accum_q2_k_32x1((ध्रुव HVX_सदिशम् *) tile, v_act), v_sc2, v_mn2, v_act[9], &v_isum, &v_msum);
        }
        ध्रुव HVX_सदिशम् * सीमित v_act_sb = (ध्रुव HVX_सदिशम् *) (y_q + kb * 1280);
        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, q2_k_int_scale(v_isum, v_msum, v_d, v_m, v_act_sb[8]));
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_q2_k_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kb = 0; kb < n_k_tiles; kb += 8) {
        HVX_सदिशम् v_d, v_m;
        q2_k_आरोपय_dmin(tile_ptr + kb * 384, &v_d, &v_m);

        HVX_सदिशम् v_isum0 = Q6_V_vzero();
        HVX_सदिशम् v_msum0 = Q6_V_vzero();
        HVX_सदिशम् v_isum1 = Q6_V_vzero();
        HVX_सदिशम् v_msum1 = Q6_V_vzero();
        प्रति (uint32_t kt = kb; kt < kb + 8; kt++) {
            ध्रुव uint8_t * सीमित tile = tile_ptr + kt * 384;
            ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + kt * 1280);
            ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + kt * 1280);

            HVX_सदिशम् v_sc2, v_mn2;
            q2_k_scale_युग्मानि(tile, &v_sc2, &v_mn2);

            HVX_सदिशम्_Pair v_sums0, v_sums1;
            accum_q2_k_32x2((ध्रुव HVX_सदिशम् *) tile, v_act0, v_act1, &v_sums0, &v_sums1);
            q2_k_int_accum(v_sums0, v_sc2, v_mn2, v_act0[9], &v_isum0, &v_msum0);
            q2_k_int_accum(v_sums1, v_sc2, v_mn2, v_act1[9], &v_isum1, &v_msum1);
        }
        ध्रुव HVX_सदिशम् * सीमित v_act0_sb = (ध्रुव HVX_सदिशम् *) (y0_q + kb * 1280);
        ध्रुव HVX_सदिशम् * सीमित v_act1_sb = (ध्रुव HVX_सदिशम् *) (y1_q + kb * 1280);
        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, q2_k_int_scale(v_isum0, v_msum0, v_d, v_m, v_act0_sb[8]));
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, q2_k_int_scale(v_isum1, v_msum1, v_d, v_m, v_act1_sb[8]));
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

// One Q2_0 weight tile covers 64 k: act tiles 2 * kt and 2 * kt + 1, each 32 k wide (Q8_1)
स्थिर शून्य tiled_vec_dot_q2_0_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();
    HVX_सदिशम् v_one_f16   = hvx_vec_splat_f16(1.0);

    uint32_t n_weight_tiles = n / HTP_MM_WEIGHT_TILE_K_Q2_0;
    प्रति (uint32_t kt = 0; kt < n_weight_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * HTP_MM_WEIGHT_ALIGNED_TILE_आकार_Q2_0);
        HVX_सदिशम् v_tile_qf = Q6_V_vzero();

        #निर्देशः unroll
        प्रति (पूर्णाङ्क half = 0; half < 2; half++) {
            ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + (2 * kt + half) * HTP_MM_ACT_TILE_आकार_Q8_1);

            HVX_सदिशम् v_योग_sf = Q6_Vsf_equals_Vw(accum_q2_0_32x1(vptr + 2 * half, v_act));
            HVX_सदिशम् v_da_sf  = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32_shuff(v_act[8]));
            HVX_सदिशम् v_sa_qf  = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act[9], v_one_f16));
            HVX_सदिशम् v_h_qf   = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_योग_sf, v_da_sf), v_sa_qf);
            v_tile_qf = (half == 0) ? v_h_qf : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf, v_h_qf);
        }

        HVX_सदिशम् v_dw_sf = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(vptr[4]));
        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf), v_dw_sf));
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_q2_0_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();
    HVX_सदिशम् v_one_f16      = hvx_vec_splat_f16(1.0);

    uint32_t n_weight_tiles = n / HTP_MM_WEIGHT_TILE_K_Q2_0;
    प्रति (uint32_t kt = 0; kt < n_weight_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * HTP_MM_WEIGHT_ALIGNED_TILE_आकार_Q2_0);
        HVX_सदिशम् v_tile_qf_c0 = Q6_V_vzero();
        HVX_सदिशम् v_tile_qf_c1 = Q6_V_vzero();

        #निर्देशः unroll
        प्रति (पूर्णाङ्क half = 0; half < 2; half++) {
            ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + (2 * kt + half) * HTP_MM_ACT_TILE_आकार_Q8_1);
            ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + (2 * kt + half) * HTP_MM_ACT_TILE_आकार_Q8_1);

            HVX_सदिशम् v_योग_c0, v_योग_c1;
            accum_q2_0_32x2(vptr + 2 * half, v_act0, v_act1, &v_योग_c0, &v_योग_c1);

            HVX_सदिशम् v_da_sf_c0 = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32_shuff(v_act0[8]));
            HVX_सदिशम् v_da_sf_c1 = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32_shuff(v_act1[8]));
            HVX_सदिशम् v_sa_qf_c0 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act0[9], v_one_f16));
            HVX_सदिशम् v_sa_qf_c1 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act1[9], v_one_f16));

            HVX_सदिशम् v_h_qf_c0 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(v_योग_c0), v_da_sf_c0), v_sa_qf_c0);
            HVX_सदिशम् v_h_qf_c1 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(v_योग_c1), v_da_sf_c1), v_sa_qf_c1);

            v_tile_qf_c0 = (half == 0) ? v_h_qf_c0 : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c0, v_h_qf_c0);
            v_tile_qf_c1 = (half == 0) ? v_h_qf_c1 : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c1, v_h_qf_c1);
        }

        HVX_सदिशम् v_dw_sf = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(vptr[4]));
        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf_c0), v_dw_sf));
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf_c1), v_dw_sf));
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_iq4nl_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_सदिशम् lut = *(ध्रुव HVX_सदिशम् *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 640);
        ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + kt * 1152);

        HVX_सदिशम् v_योग = accum_4bit_32x1_lut(vptr, v_act, आवरणम्_h4, lut);
        HVX_सदिशम् v_योग_sf = Q6_Vsf_equals_Vw(v_योग);

        HVX_सदिशम् v_scale_w = vptr[4];
        HVX_सदिशम् v_scale_a = v_act[8];
        HVX_सदिशम् v_scale_comb = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale_w, v_scale_a);
        HVX_सदिशम् v_योग_scaled = hvx_vec_mul_f32_f32(v_योग_sf, v_scale_comb);

        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, v_योग_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_iq4nl_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_सदिशम् lut = *(ध्रुव HVX_सदिशम् *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 640);
        ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + kt * 1152);
        ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + kt * 1152);

        HVX_सदिशम्_Pair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, आवरणम्_h4, lut);
        HVX_सदिशम् v_योग_c0 = Q6_V_lo_W(v_sums);
        HVX_सदिशम् v_योग_c1 = Q6_V_hi_W(v_sums);

        HVX_सदिशम् v_योग_sf_c0 = Q6_Vsf_equals_Vw(v_योग_c0);
        HVX_सदिशम् v_योग_sf_c1 = Q6_Vsf_equals_Vw(v_योग_c1);

        HVX_सदिशम् v_scale_w = vptr[4];
        HVX_सदिशम् v_scale_a_c0 = v_act0[8];
        HVX_सदिशम् v_scale_a_c1 = v_act1[8];

        HVX_सदिशम् v_scale_comb_c0 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_सदिशम् v_scale_comb_c1 = hvx_vec_mul_f16_f16_गन्तव्यम्_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_सदिशम् v_योग_scaled_c0 = hvx_vec_mul_f32_f32(v_योग_sf_c0, v_scale_comb_c0);
        HVX_सदिशम् v_योग_scaled_c1 = hvx_vec_mul_f32_f32(v_योग_sf_c1, v_scale_comb_c1);

        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, v_योग_scaled_c0);
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, v_योग_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_mxfp4_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y_q = vy;

    HVX_सदिशम् v_योग_float = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_सदिशम् lut = *(ध्रुव HVX_सदिशम् *) kvalues_mxfp4_lut;
    HVX_सदिशम् expand = *(ध्रुव HVX_सदिशम् *) expand_x32_e8m0;
    HVX_सदिशम् e8m0_आवरणम् = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 640);
        ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) (y_q + kt * 1152);

        HVX_सदिशम् v_योग = accum_4bit_32x1_lut(vptr, v_act, आवरणम्_h4, lut);
        HVX_सदिशम् v_योग_sf = Q6_Vsf_equals_Vw(v_योग);

        HVX_सदिशम् v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_सदिशम् r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_आवरणम्);
        HVX_सदिशम् v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_सदिशम् v_scale_a_f16 = v_act[8];
        HVX_सदिशम्_Pair p_scale_a_f32 = hvx_vec_f16_गन्तव्यम्_f32_shuff(v_scale_a_f16);
        HVX_सदिशम् v_scale_a = Q6_V_lo_W(p_scale_a_f32);

        HVX_सदिशम् v_scale_comb = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a);
        HVX_सदिशम् v_योग_scaled = hvx_vec_mul_f32_f32(v_योग_sf, v_scale_comb);

        v_योग_float = hvx_vec_सङ्कलय_f32_f32(v_योग_float, v_योग_scaled);
    }

    v_योग_float = hvx_vec_mul_f32_f32(v_योग_float, hvx_vec_splat_f32(0.5f));

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float);
    }
}

स्थिर शून्य tiled_vec_dot_mxfp4_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव uint8_t * सीमित tile_ptr = vx;
    ध्रुव uint8_t * सीमित y0_q = vy0;
    ध्रुव uint8_t * सीमित y1_q = vy1;

    HVX_सदिशम् v_योग_float_c0 = Q6_V_vzero();
    HVX_सदिशम् v_योग_float_c1 = Q6_V_vzero();
    HVX_सदिशम् आवरणम्_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_सदिशम् lut = *(ध्रुव HVX_सदिशम् *) kvalues_mxfp4_lut;
    HVX_सदिशम् expand = *(ध्रुव HVX_सदिशम् *) expand_x32_e8m0;
    HVX_सदिशम् e8m0_आवरणम् = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    प्रति (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ध्रुव HVX_सदिशम् * सीमित vptr = (ध्रुव HVX_सदिशम् *) (tile_ptr + kt * 640);
        ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) (y0_q + kt * 1152);
        ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) (y1_q + kt * 1152);

        HVX_सदिशम्_Pair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, आवरणम्_h4, lut);
        HVX_सदिशम् v_योग_c0 = Q6_V_lo_W(v_sums);
        HVX_सदिशम् v_योग_c1 = Q6_V_hi_W(v_sums);

        HVX_सदिशम् v_योग_sf_c0 = Q6_Vsf_equals_Vw(v_योग_c0);
        HVX_सदिशम् v_योग_sf_c1 = Q6_Vsf_equals_Vw(v_योग_c1);

        HVX_सदिशम् v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_सदिशम् r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_आवरणम्);
        HVX_सदिशम् v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_सदिशम् v_scale_a_c0_f16 = v_act0[8];
        HVX_सदिशम् v_scale_a_c1_f16 = v_act1[8];

        HVX_सदिशम्_Pair p_scale_a_c0_f32 = hvx_vec_f16_गन्तव्यम्_f32_shuff(v_scale_a_c0_f16);
        HVX_सदिशम्_Pair p_scale_a_c1_f32 = hvx_vec_f16_गन्तव्यम्_f32_shuff(v_scale_a_c1_f16);

        HVX_सदिशम् v_scale_a_c0 = Q6_V_lo_W(p_scale_a_c0_f32);
        HVX_सदिशम् v_scale_a_c1 = Q6_V_lo_W(p_scale_a_c1_f32);

        HVX_सदिशम् v_scale_comb_c0 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c0);
        HVX_सदिशम् v_scale_comb_c1 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c1);

        HVX_सदिशम् v_योग_scaled_c0 = hvx_vec_mul_f32_f32(v_योग_sf_c0, v_scale_comb_c0);
        HVX_सदिशम् v_योग_scaled_c1 = hvx_vec_mul_f32_f32(v_योग_sf_c1, v_scale_comb_c1);

        v_योग_float_c0 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, v_योग_scaled_c0);
        v_योग_float_c1 = hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, v_योग_scaled_c1);
    }

    v_योग_float_c0 = hvx_vec_mul_f32_f32(v_योग_float_c0, hvx_vec_splat_f32(0.5f));
    v_योग_float_c1 = hvx_vec_mul_f32_f32(v_योग_float_c1, hvx_vec_splat_f32(0.5f));

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_f16_32x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz) {
    ध्रुव HVX_सदिशम् * सीमित vptr  = (ध्रुव HVX_सदिशम् *) vx;
    ध्रुव HVX_सदिशम् * सीमित v_act = (ध्रुव HVX_सदिशम् *) vy;
    ध्रुव uint32_t n_vecs = ((n + 31) / 32) * 16;

    HVX_सदिशम्_Pair acc = Q6_W_vzero();

    #निर्देशः unroll(4)
    प्रति (uint32_t i = 0; i < n_vecs; i++) {
        acc = hvx_vec_mpyacc_f32_f16(acc, vptr[i], v_act[i]);
    }

    HVX_सदिशम् v_योग = hvx_vec_सङ्कलय_f32_f32(Q6_V_lo_W(acc), Q6_V_hi_W(acc));

    यदि (sz) {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_योग, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_योग);
    }
}

स्थिर शून्य tiled_vec_dot_f16_32x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः, ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1) {
    ध्रुव HVX_सदिशम् * सीमित vptr   = (ध्रुव HVX_सदिशम् *) vx;
    ध्रुव HVX_सदिशम् * सीमित v_act0 = (ध्रुव HVX_सदिशम् *) vy0;
    ध्रुव HVX_सदिशम् * सीमित v_act1 = (ध्रुव HVX_सदिशम् *) vy1;
    ध्रुव uint32_t n_vecs = ((n + 31) / 32) * 16;

    HVX_सदिशम्_Pair acc0 = Q6_W_vzero();
    HVX_सदिशम्_Pair acc1 = Q6_W_vzero();

    #निर्देशः unroll(4)
    प्रति (uint32_t i = 0; i < n_vecs; i++) {
        HVX_सदिशम् v_w = vptr[i];
        acc0 = hvx_vec_mpyacc_f32_f16(acc0, v_w, v_act0[i]);
        acc1 = hvx_vec_mpyacc_f32_f16(acc1, v_w, v_act1[i]);
    }

    HVX_सदिशम् v_sum0 = hvx_vec_सङ्कलय_f32_f32(Q6_V_lo_W(acc0), Q6_V_hi_W(acc0));
    HVX_सदिशम् v_sum1 = hvx_vec_सङ्कलय_f32_f32(Q6_V_lo_W(acc1), Q6_V_hi_W(acc1));

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_sum0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_sum0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_sum1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_पङ्क्तिकाः * परिमाण(भिन्न), v_sum1);
    }
}


स्थिर अन्तःस्थ शून्य quantize_f32_q8_0_tiled_kernel(
    ध्रुव uint8_t * सीमित src_दत्त,
    uint8_t * सीमित dst_दत्त,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_आकार,
    size_t dst_row_आकार
) {
    (शून्य) tmp_दत्त;
    प्रति (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_0_tiled((भिन्न *) src_दत्त, dst_दत्त, ne0);
        dst_दत्त += dst_row_आकार;
        src_दत्त += src_row_आकार;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_1_tiled_kernel(
    ध्रुव uint8_t * सीमित src_दत्त,
    uint8_t * सीमित dst_दत्त,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_आकार,
    size_t dst_row_आकार
) {
    (शून्य) tmp_दत्त;
    प्रति (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_1_tiled((भिन्न *) src_दत्त, dst_दत्त, ne0);
        dst_दत्त += dst_row_आकार;
        src_दत्त += src_row_आकार;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_1_s16_tiled_kernel(
    ध्रुव uint8_t * सीमित src_दत्त,
    uint8_t * सीमित dst_दत्त,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_आकार,
    size_t dst_row_आकार
) {
    (शून्य) tmp_दत्त;
    प्रति (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_1_s16_tiled((भिन्न *) src_दत्त, dst_दत्त, ne0);
        dst_दत्त += dst_row_आकार;
        src_दत्त += src_row_आकार;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_f16_tiled_kernel(
    ध्रुव uint8_t * सीमित src_दत्त,
    uint8_t * सीमित dst_दत्त,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_आकार,
    size_t dst_row_आकार
) {
    (शून्य) tmp_दत्त;
    प्रति (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_f16_tiled((ध्रुव भिन्न *) src_दत्त, dst_दत्त, ne0);
        dst_दत्त += dst_row_आकार;
        src_दत्त += src_row_आकार;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f16_f16_tiled_kernel(
    ध्रुव uint8_t * सीमित src_दत्त,
    uint8_t * सीमित dst_दत्त,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_आकार,
    size_t dst_row_आकार
) {
    (शून्य) tmp_दत्त;
    प्रति (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f16_f16_tiled((ध्रुव __fp16 *) src_दत्त, dst_दत्त, ne0);
        dst_दत्त += dst_row_आकार;
        src_दत्त += src_row_आकार;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_0_tiled_खण्डम्_kernel(
    ध्रुव भिन्न * सीमित src,
    uint8_t * सीमित dst,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_चरमः,
    size_t src_row_आकार,
    size_t dst_row_आकार,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_दत्त;
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (ne0 + qk - 1) / qk;

    प्रति (uint32_t ib = ib_first; ib < ib_चरमः; ++ib) {
        ध्रुव भिन्न * सीमित src_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + r * src_row_आकार + c * qk * परिमाण(भिन्न));
        uint8_t * सीमित dst_ptr = dst + r * dst_row_आकार + c * 4 * 1152;

        quantize_खण्डम्_f32_q8_0_tiled((भिन्न *) src_ptr, dst_ptr);

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_1_tiled_खण्डम्_kernel(
    ध्रुव भिन्न * सीमित src,
    uint8_t * सीमित dst,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_चरमः,
    size_t src_row_आकार,
    size_t dst_row_आकार,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_दत्त;
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (ne0 + qk - 1) / qk;

    प्रति (uint32_t ib = ib_first; ib < ib_चरमः; ++ib) {
        ध्रुव भिन्न * सीमित src_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + r * src_row_आकार + c * qk * परिमाण(भिन्न));
        uint8_t * सीमित dst_ptr = dst + r * dst_row_आकार + c * 4 * 1280;

        quantize_खण्डम्_f32_q8_1_tiled((भिन्न *) src_ptr, dst_ptr);

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_1_s16_tiled_खण्डम्_kernel(
    ध्रुव भिन्न * सीमित src,
    uint8_t * सीमित dst,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_चरमः,
    size_t src_row_आकार,
    size_t dst_row_आकार,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_दत्त;
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (ne0 + qk - 1) / qk;

    प्रति (uint32_t ib = ib_first; ib < ib_चरमः; ++ib) {
        ध्रुव भिन्न * सीमित src_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + r * src_row_आकार + c * qk * परिमाण(भिन्न));
        uint8_t * सीमित dst_ptr = dst + r * dst_row_आकार + c * 4 * 1280;

        quantize_खण्डम्_f32_q8_1_s16_tiled((भिन्न *) src_ptr, dst_ptr, src_ptr - (c & 1) * qk);  // super-block of 256 k

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_f16_tiled_खण्डम्_kernel(
    ध्रुव भिन्न * सीमित src,
    uint8_t * सीमित dst,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_चरमः,
    size_t src_row_आकार,
    size_t dst_row_आकार,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_दत्त;
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (ne0 + qk - 1) / qk;

    प्रति (uint32_t ib = ib_first; ib < ib_चरमः; ++ib) {
        ध्रुव भिन्न * सीमित src_ptr = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src + r * src_row_आकार + c * qk * परिमाण(भिन्न));
        uint8_t * सीमित dst_ptr = dst + r * dst_row_आकार + c * 4 * 2048;

        quantize_खण्डम्_f32_f16_tiled(src_ptr, dst_ptr);

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}

स्थिर अन्तःस्थ शून्य quantize_f16_f16_tiled_खण्डम्_kernel(
    ध्रुव __fp16 * सीमित src,
    uint8_t * सीमित dst,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_चरमः,
    size_t src_row_आकार,
    size_t dst_row_आकार,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_दत्त;
    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (ne0 + qk - 1) / qk;

    प्रति (uint32_t ib = ib_first; ib < ib_चरमः; ++ib) {
        ध्रुव __fp16 * सीमित src_ptr = (ध्रुव __fp16 *) ((ध्रुव uint8_t *) src + r * src_row_आकार + c * qk * परिमाण(__fp16));
        uint8_t * सीमित dst_ptr = dst + r * dst_row_आकार + c * 4 * 2048;

        quantize_खण्डम्_f16_f16_tiled(src_ptr, dst_ptr);

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}
