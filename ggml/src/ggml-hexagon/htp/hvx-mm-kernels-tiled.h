// Dynamic quantizers that produce tiled activations

// q8_1: vector 9 holds d * sum(q) of the 32 k, replicated.
// q8_1_s16 (Q2_K, sums16): one fp16 d (vector 8) for the 256 k super-block starting at x_sb, and vector 9 holds the
// integer sums of k 0..15 (low halfword) and k 16..31 (high halfword) in every word
__attribute__((always_inline))
ساكن مضمن فراغ quantize_بلاک_f32_q8_1_tiled_impl(عائم * محدود x, uint8_t * محدود y_بلاک, ثابت bool sums16,
                                                      ثابت عائم * محدود x_sb) {
    أكد((بنانشانی لما) x % 128 == 0);
    أكد((بنانشانی لما) y_بلاک % 128 == 0);

    HVX_سمتیہ * vx = (HVX_سمتیہ *) x;
    HVX_سمتیہ zero = Q6_V_vzero();

    HVX_سمتیہ vmax0_sf, vmax1_sf, vmax2_sf, vmax3_sf;
    إذا (sums16) {
        ثابت HVX_سمتیہ * vsb = (ثابت HVX_سمتیہ *) x_sb;
        HVX_سمتیہ vm = hvx_vec_abs_f32(vsb[0]);
        لكل (صحيح i = 1; i < 8; i++) {
            vm = Q6_Vsf_vmax_VsfVsf(vm, hvx_vec_abs_f32(vsb[i]));
        }
        vmax0_sf = vmax1_sf = vmax2_sf = vmax3_sf = hvx_vec_reduce_max_f32(vm);
    } وإلا {
        vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
        vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
        vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
        vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));
    }

    HVX_سمتیہ vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_سمتیہ vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_سمتیہ vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_سمتیہ vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);

    HVX_سمتیہ vmax01_hf   = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_سمتیہ vx01_hf     = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_سمتیہ vd01_qf16   = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_سمتیہ vd01_hf     = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_سمتیہ vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));

    HVX_سمتیہ vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_سمتیہ vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);
    HVX_سمتیہ vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_سمتیہ vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_سمتیہ vmax23_hf   = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));
    HVX_سمتیہ vx23_hf     = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));
    HVX_سمتیہ vd23_qf16   = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_سمتیہ vd23_hf     = Q6_Vhf_equals_Vqf16(vd23_qf16);
    HVX_سمتیہ vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_سمتیہ vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_سمتیہ vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_سمتیہ vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    ثابت HVX_سمتیہ ones = Q6_Vb_vsplat_R(1);
    HVX_سمتیہ v_sums = Q6_Vw_vrmpy_VbVb(vx_i8, ones);
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 4));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 8));
    إذا (!sums16) {
        v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 16));
    }
    // word 8b: sum of block b (sums16: k 0..15, and word 8b + 4: k 16..31)

    ثابت HVX_سمتیہ v_inv127 = hvx_vec_splat_f32(1.0f / 127.0f);
    HVX_سمتیہ vd0_sf = hvx_vec_mul_f32_f32(vmax0_sf, v_inv127);
    HVX_سمتیہ vd1_sf = hvx_vec_mul_f32_f32(vmax1_sf, v_inv127);
    HVX_سمتیہ vd2_sf = hvx_vec_mul_f32_f32(vmax2_sf, v_inv127);
    HVX_سمتیہ vd3_sf = hvx_vec_mul_f32_f32(vmax3_sf, v_inv127);

    HVX_سمتیہ v_sums_sf = Q6_Vsf_equals_Vw(v_sums);
    HVX_سمتیہ voff0_sf = hvx_vec_mul_f32_f32(vd0_sf, v_sums_sf);
    HVX_سمتیہ voff1_sf = hvx_vec_mul_f32_f32(vd1_sf, Q6_V_vror_VR(v_sums_sf, 32));
    HVX_سمتیہ voff2_sf = hvx_vec_mul_f32_f32(vd2_sf, Q6_V_vror_VR(v_sums_sf, 64));
    HVX_سمتیہ voff3_sf = hvx_vec_mul_f32_f32(vd3_sf, Q6_V_vror_VR(v_sums_sf, 96));

    HVX_سمتیہ voff01_hf = hvx_vec_f32_منزل_f16(voff0_sf, voff1_sf);
    HVX_سمتیہ voff23_hf = hvx_vec_f32_منزل_f16(voff2_sf, voff3_sf);

    int32_t __attribute__((aligned(128))) ws[32];
    إذا (sums16) {
        // |sum of 16 int8| <= 2032 fits a halfword
        *(HVX_سمتیہ *) ws = v_sums;
    }

    ساكن ثابت uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_سمتیہ v_repl_ctrl = * (ثابت HVX_سمتیہ *) repl;

    #ہدایت unroll
    لكل (صحيح b = 0; b < 4; b++) {
        HVX_سمتیہ v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_سمتیہ vd_b   = (b < 2) ? vd01_hf   : vd23_hf;
        HVX_سمتیہ voff_b = (b < 2) ? voff01_hf : voff23_hf;
        HVX_سمتیہ r_scale  = hvx_vec_repl_f16((b & 1) ? Q6_V_vror_VR(vd_b, 64)   : vd_b);
        HVX_سمتیہ r_وتھ = sums16
            ? Q6_V_vsplat_R(((uint32_t) ws[8 * b + 4] << 16) | ((uint32_t) ws[8 * b] & 0xFFFF))
            : hvx_vec_repl_f16((b & 1) ? Q6_V_vror_VR(voff_b, 64) : voff_b);

        HVX_سمتیہ r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_سمتیہ r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_سمتیہ r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_سمتیہ r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_سمتیہ r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_سمتیہ r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_سمتیہ r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_سمتیہ r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_سمتیہ * محدود dst = (HVX_سمتیہ *) (y_بلاک + b * 1280);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
        dst[9] = r_وتھ;
    }
}

ساكن مضمن فراغ quantize_بلاک_f32_q8_1_tiled(عائم * محدود x, uint8_t * محدود y_بلاک) {
    quantize_بلاک_f32_q8_1_tiled_impl(x, y_بلاک, false, عدم);
}

ساكن مضمن فراغ quantize_بلاک_f32_q8_1_s16_tiled(عائم * محدود x, uint8_t * محدود y_بلاک, ثابت عائم * محدود x_sb) {
    quantize_بلاک_f32_q8_1_tiled_impl(x, y_بلاک, true, x_sb);
}

ساكن مضمن فراغ quantize_بلاک_f32_q8_0_tiled(عائم * محدود x, uint8_t * محدود y_بلاک) {
    أكد((بنانشانی لما) x % 128 == 0);
    أكد((بنانشانی لما) y_بلاک % 128 == 0);

    HVX_سمتیہ * vx = (HVX_سمتیہ *) x;
    HVX_سمتیہ zero   = Q6_V_vzero();

    HVX_سمتیہ vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_سمتیہ vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_سمتیہ vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_سمتیہ vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_سمتیہ vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_سمتیہ vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_سمتیہ vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_سمتیہ vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_سمتیہ vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_سمتیہ vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_سمتیہ vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_سمتیہ vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_سمتیہ vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_سمتیہ vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_سمتیہ vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_سمتیہ vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_سمتیہ vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_سمتیہ vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_سمتیہ vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_سمتیہ vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_سمتیہ vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_سمتیہ vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_سمتیہ vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_سمتیہ vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_سمتیہ vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    HVX_سمتیہ_Pair vp01 = Q6_W_vshuff_VVR(vd01_hf, vd01_hf, -64);
    HVX_سمتیہ_Pair vp23 = Q6_W_vshuff_VVR(vd23_hf, vd23_hf, -64);

    ساكن ثابت uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_سمتیہ v_repl_ctrl = * (ثابت HVX_سمتیہ *) repl;

    #ہدایت unroll
    لكل (صحيح b = 0; b < 4; b++) {
        HVX_سمتیہ v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_سمتیہ r_scale;
        إذا (b == 0) {
            r_scale = Q6_V_lo_W(vp01);
        } وإلا إذا (b == 1) {
            r_scale = Q6_V_hi_W(vp01);
        } وإلا إذا (b == 2) {
            r_scale = Q6_V_lo_W(vp23);
        } وإلا {
            r_scale = Q6_V_hi_W(vp23);
        }

        HVX_سمتیہ r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_سمتیہ r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_سمتیہ r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_سمتیہ r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_سمتیہ r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_سمتیہ r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_سمتیہ r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_سمتیہ r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_سمتیہ * محدود dst = (HVX_سمتیہ *) (y_بلاک + b * 1152);
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

ساكن فراغ quantize_صف_f32_q8_0_tiled(عائم * محدود x, uint8_t * محدود y, uint32_t k) {
    أكد(k % 32 == 0);
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (k + qk - 1) / qk;

    لكل (uint32_t i = 0; i < nb; i++) {
        uint8_t * محدود y_بلاک = y + i * 4 * 1152;
        quantize_بلاک_f32_q8_0_tiled(x + i * qk, y_بلاک);
    }
}

ساكن فراغ quantize_صف_f32_q8_1_tiled(عائم * محدود x, uint8_t * محدود y, uint32_t k) {
    أكد(k % 32 == 0);
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (k + qk - 1) / qk;

    لكل (uint32_t i = 0; i < nb; i++) {
        uint8_t * محدود y_بلاک = y + i * 4 * 1280;
        quantize_بلاک_f32_q8_1_tiled(x + i * qk, y_بلاک);
    }
}

ساكن فراغ quantize_صف_f32_q8_1_s16_tiled(عائم * محدود x, uint8_t * محدود y, uint32_t k) {
    أكد(k % 32 == 0);
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (k + qk - 1) / qk;

    لكل (uint32_t i = 0; i < nb; i++) {
        uint8_t * محدود y_بلاک = y + i * 4 * 1280;
        quantize_بلاک_f32_q8_1_s16_tiled(x + i * qk, y_بلاک, x + (i & ~1u) * qk);  // k % 256 == 0 for Q2_K
    }
}

ساكن مضمن فراغ quantize_half_f16_tiled_splat(HVX_سمتیہ v_src, HVX_سمتیہ * محدود dst) {
    ساكن ثابت uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_سمتیہ v_repl_ctrl = *(ثابت HVX_سمتیہ *) repl;

    #ہدایت unroll(4)
    لكل (صحيح cp = 0; cp < 32; cp++) {
        dst[cp] = Q6_V_vdelta_VV(cp == 0 ? v_src : Q6_V_vror_VR(v_src, cp * 4), v_repl_ctrl);
    }
}

ساكن مضمن فراغ quantize_بلاک_f32_f16_tiled(ثابت عائم * محدود x, uint8_t * محدود y_بلاک) {
    أكد((بنانشانی لما) x % 128 == 0);
    أكد((بنانشانی لما) y_بلاک % 128 == 0);

    ثابت HVX_سمتیہ * vx = (ثابت HVX_سمتیہ *) x;
    HVX_سمتیہ * محدود dst = (HVX_سمتیہ *) y_بلاک;

    HVX_سمتیہ vx01_hf = hvx_vec_f32_منزل_f16(vx[0], vx[1]);
    quantize_half_f16_tiled_splat(vx01_hf, dst);

    HVX_سمتیہ vx23_hf = hvx_vec_f32_منزل_f16(vx[2], vx[3]);
    quantize_half_f16_tiled_splat(vx23_hf, dst + 32);
}

ساكن مضمن فراغ quantize_بلاک_f16_f16_tiled(ثابت __fp16 * محدود x, uint8_t * محدود y_بلاک) {
    أكد((بنانشانی لما) x % 128 == 0);
    أكد((بنانشانی لما) y_بلاک % 128 == 0);

    ثابت HVX_سمتیہ * vx = (ثابت HVX_سمتیہ *) x;
    HVX_سمتیہ * محدود dst = (HVX_سمتیہ *) y_بلاک;

    quantize_half_f16_tiled_splat(vx[0], dst);
    quantize_half_f16_tiled_splat(vx[1], dst + 32);
}

ساكن فراغ quantize_صف_f32_f16_tiled(ثابت عائم * محدود x, uint8_t * محدود y, uint32_t k) {
    أكد(k % 32 == 0);
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (k + qk - 1) / qk;

    لكل (uint32_t i = 0; i < nb; i++) {
        uint8_t * محدود y_بلاک = y + i * 4 * 2048;
        quantize_بلاک_f32_f16_tiled(x + i * qk, y_بلاک);
    }
}

ساكن فراغ quantize_صف_f16_f16_tiled(ثابت __fp16 * محدود x, uint8_t * محدود y, uint32_t k) {
    أكد(k % 32 == 0);
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (k + qk - 1) / qk;

    لكل (uint32_t i = 0; i < nb; i++) {
        uint8_t * محدود y_بلاک = y + i * 4 * 2048;
        quantize_بلاک_f16_f16_tiled(x + i * qk, y_بلاک);
    }
}

// Dot kernels & helpers that consume tiled activations

ساكن مضمن HVX_سمتیہ hvx_vec_mul_f16_f16_منزل_f32_lower32(HVX_سمتیہ v1, HVX_سمتیہ v2) {
#إذا __HVX_ARCH__ >= 79
    HVX_سمتیہ_Pair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    أرجع Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4));
#وإلا
    HVX_سمتیہ_Pair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_سمتیہ hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_سمتیہ lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    أرجع Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -4));
#جے_انت
}

// both halves of hvx_vec_mul_f16_f16_to_f32_lower32: lo = products of lanes 0..31, hi = lanes 32..63
ساكن مضمن HVX_سمتیہ_Pair hvx_vec_mul_f16_f16_منزل_f32_pair(HVX_سمتیہ v1, HVX_سمتیہ v2) {
#إذا __HVX_ARCH__ >= 79
    HVX_سمتیہ_Pair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    أرجع Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4);
#وإلا
    HVX_سمتیہ_Pair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_سمتیہ hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_سمتیہ lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    أرجع Q6_W_vshuff_VVR(hi, lo, -4);
#جے_انت
}

ساكن مضمن HVX_سمتیہ unpack_and_interleave_4bit(HVX_سمتیہ v_a, HVX_سمتیہ v_b, HVX_سمتیہ نقاب_h4) {
    HVX_سمتیہ v_W0 = Q6_V_vand_VV(v_a, نقاب_h4);
    HVX_سمتیہ v_W1 = Q6_Vub_vlsr_VubR(v_a, 4);
    HVX_سمتیہ v_W2 = Q6_V_vand_VV(v_b, نقاب_h4);
    HVX_سمتیہ v_W3 = Q6_Vub_vlsr_VubR(v_b, 4);

    HVX_سمتیہ_Pair v01_pair = Q6_W_vshuff_VVR(v_W1, v_W0, -1);
    HVX_سمتیہ_Pair v23_pair = Q6_W_vshuff_VVR(v_W3, v_W2, -1);
    HVX_سمتیہ_Pair v0123_pair = Q6_W_vshuff_VVR(Q6_V_lo_W(v23_pair), Q6_V_lo_W(v01_pair), -2);
    أرجع Q6_V_lo_W(v0123_pair);
}

ساكن مضمن HVX_سمتیہ_Pair unpack_and_interleave_4bit_x2(HVX_سمتیہ v_src, HVX_سمتیہ نقاب_h4) {
    HVX_سمتیہ v_lo = Q6_V_vand_VV(v_src, نقاب_h4);
    HVX_سمتیہ v_hi = Q6_Vub_vlsr_VubR(v_src, 4);
    HVX_سمتیہ_Pair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_سمتیہ v01_lo = Q6_V_lo_W(v01_pair);
    HVX_سمتیہ v01_hi = Q6_V_hi_W(v01_pair);

    HVX_سمتیہ v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_سمتیہ v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_سمتیہ v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_سمتیہ v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    أرجع Q6_W_vcombine_VV(v_W1, v_W0);
}

ساكن مضمن HVX_سمتیہ unpack_q1_0_گروہ(
    HVX_سمتیہ v_plane,
    صحيح g,
    HVX_سمتیہ v_neg_one,
    HVX_سمتیہ v_pos_one
) {
    HVX_سمتیہ_Pred q = Q6_Q_vand_VR(v_plane, 0x01010101u << g);
    أرجع Q6_V_vmux_QVV(q, v_pos_one, v_neg_one);
}

ساكن مضمن HVX_سمتیہ accum_q1_0_32x1(
    HVX_سمتیہ v_plane,
    ثابت HVX_سمتیہ * محدود v_act,
    HVX_سمتیہ v_neg_one,
    HVX_سمتیہ v_pos_one
) {
    HVX_سمتیہ v_جوڑ_lo = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_hi = Q6_V_vzero();

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q1_0_گروہ(v_plane, g,     v_neg_one, v_pos_one);
        HVX_سمتیہ v_W_hi = unpack_q1_0_گروہ(v_plane, g + 4, v_neg_one, v_pos_one);
        v_جوڑ_lo = Q6_Vw_vrmpyacc_VwVbVb(v_جوڑ_lo, v_W_lo, v_act[g]);
        v_جوڑ_hi = Q6_Vw_vrmpyacc_VwVbVb(v_جوڑ_hi, v_W_hi, v_act[g + 4]);
    }

    أرجع Q6_Vw_vadd_VwVw(v_جوڑ_lo, v_جوڑ_hi);
}

ساكن مضمن HVX_سمتیہ accum_4bit_32x1(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act,
    HVX_سمتیہ i8
) {
    HVX_سمتیہ v_sum0 = Q6_V_vzero();
    HVX_سمتیہ v_sum1 = Q6_V_vzero();
    HVX_سمتیہ نقاب_h4 = Q6_Vb_vsplat_R(0x0F);

    #ہدایت unroll
    لكل (صحيح i = 0; i < 4; i++) {
        HVX_سمتیہ_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], نقاب_h4);
        HVX_سمتیہ v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_سمتیہ v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    أرجع Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

ساكن مضمن HVX_سمتیہ accum_4bit_32x1_lut(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act,
    HVX_سمتیہ نقاب_h4,
    HVX_سمتیہ lut
) {
    HVX_سمتیہ v_sum0 = Q6_V_vzero();
    HVX_سمتیہ v_sum1 = Q6_V_vzero();

    #ہدایت unroll
    لكل (صحيح i = 0; i < 4; i++) {
        HVX_سمتیہ_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], نقاب_h4);
        HVX_سمتیہ v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_سمتیہ v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    أرجع Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

ساكن مضمن HVX_سمتیہ_Pair accum_q1_0_32x2(
    HVX_سمتیہ v_plane,
    ثابت HVX_سمتیہ * محدود v_act0,
    ثابت HVX_سمتیہ * محدود v_act1,
    HVX_سمتیہ v_neg_one,
    HVX_سمتیہ v_pos_one
) {
    HVX_سمتیہ v_sum0_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum0_hi = Q6_V_vzero();
    HVX_سمتیہ v_sum1_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum1_hi = Q6_V_vzero();

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q1_0_گروہ(v_plane, g,     v_neg_one, v_pos_one);
        HVX_سمتیہ v_W_hi = unpack_q1_0_گروہ(v_plane, g + 4, v_neg_one, v_pos_one);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    أرجع Q6_W_vcombine_VV(
        Q6_Vw_vadd_VwVw(v_sum1_lo, v_sum1_hi),
        Q6_Vw_vadd_VwVw(v_sum0_lo, v_sum0_hi));
}

ساكن مضمن HVX_سمتیہ_Pair accum_4bit_32x2(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act0,
    ثابت HVX_سمتیہ * محدود v_act1,
    HVX_سمتیہ i8
) {
    HVX_سمتیہ v_sum0 = Q6_V_vzero();
    HVX_سمتیہ v_sum1 = Q6_V_vzero();
    HVX_سمتیہ نقاب_h4 = Q6_Vb_vsplat_R(0x0F);

    #ہدایت unroll
    لكل (صحيح i = 0; i < 4; i++) {
        HVX_سمتیہ_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], نقاب_h4);
        HVX_سمتیہ v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_سمتیہ v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    أرجع Q6_W_vcombine_VV(v_sum1, v_sum0);
}

ساكن مضمن HVX_سمتیہ_Pair accum_4bit_32x2_lut(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act0,
    ثابت HVX_سمتیہ * محدود v_act1,
    HVX_سمتیہ نقاب_h4,
    HVX_سمتیہ lut
) {
    HVX_سمتیہ v_sum0 = Q6_V_vzero();
    HVX_سمتیہ v_sum1 = Q6_V_vzero();

    #ہدایت unroll
    لكل (صحيح i = 0; i < 4; i++) {
        HVX_سمتیہ_Pair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], نقاب_h4);
        HVX_سمتیہ v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_سمتیہ v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    أرجع Q6_W_vcombine_VV(v_sum1, v_sum0);
}

ساكن مضمن HVX_سمتیہ accum_q8_0_32x1(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act
) {
    HVX_سمتیہ v_جوڑ = Q6_V_vzero();
    #ہدایت unroll
    لكل (صحيح g = 0; g < 8; g++) {
        HVX_سمتیہ v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_سمتیہ v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_جوڑ = Q6_Vw_vrmpyacc_VwVbVb(v_جوڑ, v_W, v_act[g]);
    }
    أرجع v_جوڑ;
}

ساكن مضمن HVX_سمتیہ_Pair accum_q8_0_32x2(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act0,
    ثابت HVX_سمتیہ * محدود v_act1
) {
    HVX_سمتیہ v_sum0 = Q6_V_vzero();
    HVX_سمتیہ v_sum1 = Q6_V_vzero();
    #ہدایت unroll
    لكل (صحيح g = 0; g < 8; g++) {
        HVX_سمتیہ v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_سمتیہ v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W, v_act0[g]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W, v_act1[g]);
    }
    أرجع Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q5_K: OR 0x10 into every lane of v whose flag j is set in the plane (see HTP_TENSOR_TILE_SIZE_Q5_K)
ساكن مضمن HVX_سمتیہ hvx_q5k_or_hibit(HVX_سمتیہ v, HVX_سمتیہ v_plane, صحيح j) {
    HVX_سمتیہ_Pred q = Q6_Q_vand_VR(v_plane, 0x01010101u << j);
    أرجع Q6_V_vandor_VQR(v, q, 0x10101010);
}

// 5-bit variant: the high bit comes from the plane, see hvx_q5k_or_hibit
ساكن مضمن HVX_سمتیہ_Pair unpack_and_interleave_5bit_x2(HVX_سمتیہ v_src, HVX_سمتیہ v_plane, صحيح i, HVX_سمتیہ نقاب_h4) {
    HVX_سمتیہ v_lo = hvx_q5k_or_hibit(Q6_V_vand_VV(v_src, نقاب_h4), v_plane, 2 * i);
    HVX_سمتیہ v_hi = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(v_src, 4), v_plane, 2 * i + 1);
    HVX_سمتیہ_Pair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_سمتیہ v01_lo = Q6_V_lo_W(v01_pair);
    HVX_سمتیہ v01_hi = Q6_V_hi_W(v01_pair);

    HVX_سمتیہ v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_سمتیہ v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_سمتیہ v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_سمتیہ v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    أرجع Q6_W_vcombine_VV(v_W1, v_W0);
}

ساكن مضمن HVX_سمتیہ accum_5bit_32x1(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act,
    HVX_سمتیہ i8
) {
    HVX_سمتیہ v_sum0 = Q6_V_vzero();
    HVX_سمتیہ v_sum1 = Q6_V_vzero();
    HVX_سمتیہ نقاب_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_سمتیہ v_plane = vptr[5];

    #ہدایت unroll
    لكل (صحيح i = 0; i < 4; i++) {
        HVX_سمتیہ_Pair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, نقاب_h4);
        HVX_سمتیہ v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_سمتیہ v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    أرجع Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

ساكن مضمن HVX_سمتیہ_Pair accum_5bit_32x2(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act0,
    ثابت HVX_سمتیہ * محدود v_act1,
    HVX_سمتیہ i8
) {
    HVX_سمتیہ v_sum0 = Q6_V_vzero();
    HVX_سمتیہ v_sum1 = Q6_V_vzero();
    HVX_سمتیہ نقاب_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_سمتیہ v_plane = vptr[5];

    #ہدایت unroll
    لكل (صحيح i = 0; i < 4; i++) {
        HVX_سمتیہ_Pair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, نقاب_h4);
        HVX_سمتیہ v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_سمتیہ v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    أرجع Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q6_K weights are stored unsigned (0..63), see HTP_TENSOR_TILE_SIZE_Q6_K. Unpack k-group g of a tile to signed bytes (q - 32)
ساكن مضمن HVX_سمتیہ unpack_q6_k_گروہ(ثابت HVX_سمتیہ * محدود vptr, صحيح g, HVX_سمتیہ نقاب_0f, HVX_سمتیہ نقاب_03, HVX_سمتیہ i32) {
    HVX_سمتیہ v_lo = (g & 1) ? Q6_Vub_vlsr_VubR(vptr[g >> 1], 4) : Q6_V_vand_VV(vptr[g >> 1], نقاب_0f);
    HVX_سمتیہ v_hi = (g & 3) ? Q6_Vub_vlsr_VubR(vptr[4 + (g >> 2)], 2 * (g & 3)) : vptr[4 + (g >> 2)];
    HVX_سمتیہ v_q  = Q6_V_vor_VV(v_lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_hi, نقاب_03), 4));
    أرجع Q6_Vb_vsub_VbVb(v_q, i32);
}

// k 0..15 and k 16..31 of a Q6_K tile have different scales: lo half of the pair sums k 0..15, hi half sums k 16..31
ساكن مضمن HVX_سمتیہ_Pair accum_q6_k_32x1(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act,
    HVX_سمتیہ i32
) {
    HVX_سمتیہ v_جوڑ_lo = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_hi = Q6_V_vzero();
    HVX_سمتیہ نقاب_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_سمتیہ نقاب_03 = Q6_Vb_vsplat_R(0x03);

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q6_k_گروہ(vptr, g,     نقاب_0f, نقاب_03, i32);
        HVX_سمتیہ v_W_hi = unpack_q6_k_گروہ(vptr, g + 4, نقاب_0f, نقاب_03, i32);
        v_جوڑ_lo = Q6_Vw_vrmpyacc_VwVbVb(v_جوڑ_lo, v_W_lo, v_act[g]);
        v_جوڑ_hi = Q6_Vw_vrmpyacc_VwVbVb(v_جوڑ_hi, v_W_hi, v_act[g + 4]);
    }

    أرجع Q6_W_vcombine_VV(v_جوڑ_hi, v_جوڑ_lo);
}

ساكن مضمن فراغ accum_q6_k_32x2(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act0,
    ثابت HVX_سمتیہ * محدود v_act1,
    HVX_سمتیہ i32,
    HVX_سمتیہ_Pair * v_sums0,
    HVX_سمتیہ_Pair * v_sums1
) {
    HVX_سمتیہ v_sum0_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum0_hi = Q6_V_vzero();
    HVX_سمتیہ v_sum1_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum1_hi = Q6_V_vzero();
    HVX_سمتیہ نقاب_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_سمتیہ نقاب_03 = Q6_Vb_vsplat_R(0x03);

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q6_k_گروہ(vptr, g,     نقاب_0f, نقاب_03, i32);
        HVX_سمتیہ v_W_hi = unpack_q6_k_گروہ(vptr, g + 4, نقاب_0f, نقاب_03, i32);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// scale the two half sums with the per-row tile scales (v_scale_w = vptr[6]) and the activation scale
ساكن مضمن HVX_سمتیہ scale_q6_k_32x1(HVX_سمتیہ_Pair v_sums, HVX_سمتیہ v_scale_w, HVX_سمتیہ v_scale_a) {
    HVX_سمتیہ_Pair v_scale = hvx_vec_mul_f16_f16_منزل_f32_pair(v_scale_w, v_scale_a);
    HVX_سمتیہ v_lo = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums)), Q6_V_lo_W(v_scale));
    HVX_سمتیہ v_hi = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums)), Q6_V_hi_W(v_scale));
    أرجع hvx_vec_جوڑو_f32_f32(v_lo, v_hi);
}

// Q3_K / Q2_K: low 2 bits of k-group g, see HTP_TENSOR_TILE_SIZE_Q3_K
ساكن مضمن HVX_سمتیہ unpack_q3_k_low2(ثابت HVX_سمتیہ * محدود vptr, صحيح g, HVX_سمتیہ نقاب_03) {
    HVX_سمتیہ v = vptr[g >> 2];
    إذا ((g & 3) == 3) {
        أرجع Q6_Vub_vlsr_VubR(v, 6);
    }
    إذا (g & 3) {
        v = Q6_Vub_vlsr_VubR(v, 2 * (g & 3));
    }
    أرجع Q6_V_vand_VV(v, نقاب_03);
}

// Q3_K k-group g as signed bytes: low2 | 0xFC (= low2 - 4) where bit g of vector 2 is set
ساكن مضمن HVX_سمتیہ unpack_q3_k_گروہ(ثابت HVX_سمتیہ * محدود vptr, صحيح g, HVX_سمتیہ نقاب_03) {
    HVX_سمتیہ_Pred q_neg = Q6_Q_vand_VR(vptr[2], 0x01010101u << g);
    أرجع Q6_V_vandor_VQR(unpack_q3_k_low2(vptr, g, نقاب_03), q_neg, 0xFCFCFCFC);
}

// same half split as accum_q6_k_32x1
ساكن مضمن HVX_سمتیہ_Pair accum_q3_k_32x1(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act
) {
    HVX_سمتیہ v_جوڑ_lo = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_hi = Q6_V_vzero();
    HVX_سمتیہ نقاب_03  = Q6_Vb_vsplat_R(0x03);

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q3_k_گروہ(vptr, g,     نقاب_03);
        HVX_سمتیہ v_W_hi = unpack_q3_k_گروہ(vptr, g + 4, نقاب_03);
        v_جوڑ_lo = Q6_Vw_vrmpyacc_VwVbVb(v_جوڑ_lo, v_W_lo, v_act[g]);
        v_جوڑ_hi = Q6_Vw_vrmpyacc_VwVbVb(v_جوڑ_hi, v_W_hi, v_act[g + 4]);
    }

    أرجع Q6_W_vcombine_VV(v_جوڑ_hi, v_جوڑ_lo);
}

ساكن مضمن فراغ accum_q3_k_32x2(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act0,
    ثابت HVX_سمتیہ * محدود v_act1,
    HVX_سمتیہ_Pair * v_sums0,
    HVX_سمتیہ_Pair * v_sums1
) {
    HVX_سمتیہ v_sum0_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum0_hi = Q6_V_vzero();
    HVX_سمتیہ v_sum1_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum1_hi = Q6_V_vzero();
    HVX_سمتیہ نقاب_03   = Q6_Vb_vsplat_R(0x03);

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q3_k_گروہ(vptr, g,     نقاب_03);
        HVX_سمتیہ v_W_hi = unpack_q3_k_گروہ(vptr, g + 4, نقاب_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// Q2_K: the Q3_K dot products without the -4 flags, scales and mins are applied in q2_k_int_accum
ساكن مضمن HVX_سمتیہ_Pair accum_q2_k_32x1(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act
) {
    HVX_سمتیہ v_جوڑ_lo = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_hi = Q6_V_vzero();
    HVX_سمتیہ نقاب_03  = Q6_Vb_vsplat_R(0x03);

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q3_k_low2(vptr, g,     نقاب_03);
        HVX_سمتیہ v_W_hi = unpack_q3_k_low2(vptr, g + 4, نقاب_03);
        v_جوڑ_lo = Q6_Vw_vrmpyacc_VwVbVb(v_جوڑ_lo, v_W_lo, v_act[g]);
        v_جوڑ_hi = Q6_Vw_vrmpyacc_VwVbVb(v_جوڑ_hi, v_W_hi, v_act[g + 4]);
    }

    أرجع Q6_W_vcombine_VV(v_جوڑ_hi, v_جوڑ_lo);
}

ساكن مضمن فراغ accum_q2_k_32x2(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act0,
    ثابت HVX_سمتیہ * محدود v_act1,
    HVX_سمتیہ_Pair * v_sums0,
    HVX_سمتیہ_Pair * v_sums1
) {
    HVX_سمتیہ v_sum0_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum0_hi = Q6_V_vzero();
    HVX_سمتیہ v_sum1_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum1_hi = Q6_V_vzero();
    HVX_سمتیہ نقاب_03   = Q6_Vb_vsplat_R(0x03);

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q3_k_low2(vptr, g,     نقاب_03);
        HVX_سمتیہ v_W_hi = unpack_q3_k_low2(vptr, g + 4, نقاب_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// Q2_0: unsigned codes 0..3; the -1 offset is applied via the Q8_1 block sum (v_act[9])
ساكن مضمن HVX_سمتیہ accum_q2_0_32x1(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act
) {
    HVX_سمتیہ v_جوڑ_lo = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_hi = Q6_V_vzero();
    HVX_سمتیہ نقاب_03  = Q6_Vb_vsplat_R(0x03);

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q3_k_low2(vptr, g,     نقاب_03);
        HVX_سمتیہ v_W_hi = unpack_q3_k_low2(vptr, g + 4, نقاب_03);
        v_جوڑ_lo = Q6_Vw_vrmpyacc_VwVubVb(v_جوڑ_lo, v_W_lo, v_act[g]);
        v_جوڑ_hi = Q6_Vw_vrmpyacc_VwVubVb(v_جوڑ_hi, v_W_hi, v_act[g + 4]);
    }

    أرجع Q6_Vw_vadd_VwVw(v_جوڑ_lo, v_جوڑ_hi);
}

ساكن مضمن فراغ accum_q2_0_32x2(
    ثابت HVX_سمتیہ * محدود vptr,
    ثابت HVX_سمتیہ * محدود v_act0,
    ثابت HVX_سمتیہ * محدود v_act1,
    HVX_سمتیہ * v_sum0,
    HVX_سمتیہ * v_sum1
) {
    HVX_سمتیہ v_sum0_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum0_hi = Q6_V_vzero();
    HVX_سمتیہ v_sum1_lo = Q6_V_vzero();
    HVX_سمتیہ v_sum1_hi = Q6_V_vzero();
    HVX_سمتیہ نقاب_03   = Q6_Vb_vsplat_R(0x03);

    #ہدایت unroll
    لكل (صحيح g = 0; g < 4; g++) {
        HVX_سمتیہ v_W_lo = unpack_q3_k_low2(vptr, g,     نقاب_03);
        HVX_سمتیہ v_W_hi = unpack_q3_k_low2(vptr, g + 4, نقاب_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVubVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVubVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVubVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVubVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sum0 = Q6_Vw_vadd_VwVw(v_sum0_lo, v_sum0_hi);
    *v_sum1 = Q6_Vw_vadd_VwVw(v_sum1_lo, v_sum1_hi);
}

// Q2_K: fp16 d (*v_d) and -dmin (*v_m) of the 32 rows of a super-block (8 tiles, VTCM stride 384), each in both vector halves
ساكن مضمن فراغ q2_k_لدو_dmin(ثابت uint8_t * محدود sb, HVX_سمتیہ * محدود v_d, HVX_سمتیہ * محدود v_m) {
    HVX_سمتیہ v_dd = Q6_V_vzero();  // lanes 0..31: d, lanes 32..63: -dmin

    #ہدایت unroll
    لكل (صحيح j = 0; j < 8; j++) {
        HVX_سمتیہ v = hvx_vmem(sb + j * 384 + 256);  // the slice is at bytes 64..79
        v_dd = Q6_V_vmux_QVV(Q6_Q_vsetq_R(16 * j), v_dd, Q6_V_vror_VR(v, (64 - 16 * j) & 127));
    }

    HVX_سمتیہ_Pred q_lo = Q6_Q_vsetq_R(64);
    HVX_سمتیہ     v_r  = Q6_V_vror_VR(v_dd, 64);
    *v_d = Q6_V_vmux_QVV(q_lo, v_dd, v_r);
    *v_m = Q6_V_vmux_QVV(q_lo, v_r, v_dd);
}

// Q2_K: fp16 D = d * scale and M = -dmin * min of a tile from its scale bytes, in the lanes of the Q6_K scales
ساكن مضمن فراغ q2_k_unpack_scales(ثابت uint8_t * محدود tile, HVX_سمتیہ v_d, HVX_سمتیہ v_m,
                                      HVX_سمتیہ * محدود v_D, HVX_سمتیہ * محدود v_M) {
    HVX_سمتیہ v_sc = hvx_vmem(tile + 256);
    HVX_سمتیہ v_l  = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(Q6_V_vand_VV(v_sc, Q6_Vb_vsplat_R(0x0F))));
    HVX_سمتیہ v_mn = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(Q6_Vub_vlsr_VubR(v_sc, 4)));
    // qf32 product: rounds like GGML_FP32_TO_FP16(d * scale) (qf16 is off by 1 ulp for many values on v73)
    *v_D = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(v_l), v_d));
    *v_M = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(v_mn), v_m));
}

// Q2_K scale and min nibbles of a tile as halfword pairs: word r = sub-block 1 << 16 | sub-block 0 of row r
ساكن مضمن فراغ q2_k_scale_جوڑے(ثابت uint8_t * محدود tile, HVX_سمتیہ * محدود v_sc2, HVX_سمتیہ * محدود v_mn2) {
    HVX_سمتیہ v_h = Q6_Vh_vshuff_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile + 256))));
    *v_sc2 = Q6_V_vand_VV(v_h, Q6_Vh_vsplat_R(0x000F));
    *v_mn2 = Q6_Vuh_vlsr_VuhR(v_h, 4);
}

// Q2_K with q8_1_s16 activations, one tile: isum += scale * (q . a), msum += min * sum(a), per row, in int32.
// |sum of 16 q * a| <= 3 * 128 * 16 fits a halfword, so both sub-blocks go through one vdmpy (no saturation)
ساكن مضمن فراغ q2_k_int_accum(HVX_سمتیہ_Pair v_sums, HVX_سمتیہ v_sc2, HVX_سمتیہ v_mn2, HVX_سمتیہ v_bsum,
                                  HVX_سمتیہ * محدود v_isum, HVX_سمتیہ * محدود v_msum) {
    HVX_سمتیہ v_s2 = Q6_Vh_vshuffe_VhVh(Q6_V_hi_W(v_sums), Q6_V_lo_W(v_sums));
    *v_isum = Q6_Vw_vadd_VwVw(*v_isum, Q6_Vw_vdmpy_VhVh_sat(v_s2, v_sc2));
    *v_msum = Q6_Vw_vadd_VwVw(*v_msum, Q6_Vw_vdmpy_VhVh_sat(v_mn2, v_bsum));
}

// Q2_K super-block result: d_a * (d * isum - dmin * msum), with v_d / v_m the fp16 d / -dmin of q2_k_load_dmin
ساكن مضمن HVX_سمتیہ q2_k_int_scale(HVX_سمتیہ v_isum, HVX_سمتیہ v_msum, HVX_سمتیہ v_d, HVX_سمتیہ v_m, HVX_سمتیہ v_act8) {
    HVX_سمتیہ v_d32  = Q6_V_lo_W(hvx_vec_f16_منزل_f32(v_d));
    HVX_سمتیہ v_m32  = Q6_V_lo_W(hvx_vec_f16_منزل_f32(v_m));
    HVX_سمتیہ v_da32 = Q6_V_lo_W(hvx_vec_f16_منزل_f32(v_act8));
    HVX_سمتیہ v_t = hvx_vec_جوڑو_f32_f32(hvx_vec_mul_f32_f32(v_d32, Q6_Vsf_equals_Vw(v_isum)),
                                         hvx_vec_mul_f32_f32(v_m32, Q6_Vsf_equals_Vw(v_msum)));
    أرجع hvx_vec_mul_f32_f32(v_t, v_da32);
}

ساكن فراغ tiled_vec_dot_q1_0_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_qf  = Q6_V_vzero();
    HVX_سمتیہ v_neg_one = Q6_Vb_vsplat_R(-1);
    HVX_سمتیہ v_pos_one = Q6_Vb_vsplat_R(1);

    ثابت uint32_t n_k_tiles = n / HTP_TENSOR_TILE_K_Q1_0;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * htp_tensor_get_tile_padded_ناپ(HTP_TYPE_Q1_0));
        HVX_سمتیہ v_tile_qf = Q6_V_vzero();

        #ہدایت unroll
        لكل (صحيح quarter = 0; quarter < 4; quarter++) {
            ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + (4 * kt + quarter) * HTP_MM_ACT_TILE_ناپ_Q8_0);
            HVX_سمتیہ v_جوڑ_sf = Q6_Vsf_equals_Vw(accum_q1_0_32x1(vptr[quarter], v_act, v_neg_one, v_pos_one));
            HVX_سمتیہ v_da_sf  = Q6_V_lo_W(hvx_vec_f16_منزل_f32_shuff(v_act[8]));
            HVX_سمتیہ v_q_qf   = Q6_Vqf32_vmpy_VsfVsf(v_جوڑ_sf, v_da_sf);
            v_tile_qf = (quarter == 0) ? v_q_qf : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf, v_q_qf);
        }

        HVX_سمتیہ v_dw_sf = Q6_V_lo_W(hvx_vec_f16_منزل_f32(vptr[4]));
        v_جوڑ_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_جوڑ_qf, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf), v_dw_sf));
    }

    HVX_سمتیہ v_جوڑ_float = Q6_Vsf_equals_Vqf32(v_جوڑ_qf);
    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_q1_0_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_qf_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_qf_c1 = Q6_V_vzero();
    HVX_سمتیہ v_neg_one   = Q6_Vb_vsplat_R(-1);
    HVX_سمتیہ v_pos_one   = Q6_Vb_vsplat_R(1);

    ثابت uint32_t n_k_tiles = n / HTP_TENSOR_TILE_K_Q1_0;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * htp_tensor_get_tile_padded_ناپ(HTP_TYPE_Q1_0));
        HVX_سمتیہ v_tile_qf_c0 = Q6_V_vzero();
        HVX_سمتیہ v_tile_qf_c1 = Q6_V_vzero();

        لكل (صحيح quarter = 0; quarter < 4; quarter++) {
            ثابت uint32_t act_tile = 4 * kt + quarter;
            ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + act_tile * HTP_MM_ACT_TILE_ناپ_Q8_0);
            ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + act_tile * HTP_MM_ACT_TILE_ناپ_Q8_0);
            HVX_سمتیہ_Pair v_sums = accum_q1_0_32x2(vptr[quarter], v_act0, v_act1, v_neg_one, v_pos_one);
            HVX_سمتیہ v_جوڑ_sf_c0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums));
            HVX_سمتیہ v_جوڑ_sf_c1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums));
            HVX_سمتیہ v_da_sf_c0  = Q6_V_lo_W(hvx_vec_f16_منزل_f32_shuff(v_act0[8]));
            HVX_سمتیہ v_da_sf_c1  = Q6_V_lo_W(hvx_vec_f16_منزل_f32_shuff(v_act1[8]));

            v_tile_qf_c0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c0, Q6_Vqf32_vmpy_VsfVsf(v_جوڑ_sf_c0, v_da_sf_c0));
            v_tile_qf_c1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c1, Q6_Vqf32_vmpy_VsfVsf(v_جوڑ_sf_c1, v_da_sf_c1));
        }

        HVX_سمتیہ v_dw_sf = Q6_V_lo_W(hvx_vec_f16_منزل_f32(vptr[4]));
        v_جوڑ_qf_c0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_جوڑ_qf_c0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf_c0), v_dw_sf));
        v_جوڑ_qf_c1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_جوڑ_qf_c1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf_c1), v_dw_sf));
    }

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_Vsf_equals_Vqf32(v_جوڑ_qf_c0);
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_Vsf_equals_Vqf32(v_جوڑ_qf_c1);
    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_q4_0_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();
    HVX_سمتیہ i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 640);
        ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + kt * 1152);

        HVX_سمتیہ v_جوڑ = accum_4bit_32x1(vptr, v_act, i8);
        HVX_سمتیہ v_جوڑ_sf = Q6_Vsf_equals_Vw(v_جوڑ);

        HVX_سمتیہ v_scale_w = vptr[4];
        HVX_سمتیہ v_scale_a = v_act[8];
        HVX_سمتیہ v_scale_comb = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale_w, v_scale_a);
        HVX_سمتیہ v_جوڑ_scaled = hvx_vec_mul_f32_f32(v_جوڑ_sf, v_scale_comb);

        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, v_جوڑ_scaled);
    }

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_q4_0_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();
    HVX_سمتیہ i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 640);
        ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + kt * 1152);
        ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + kt * 1152);

        HVX_سمتیہ_Pair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, i8);
        HVX_سمتیہ v_جوڑ_c0 = Q6_V_lo_W(v_sums);
        HVX_سمتیہ v_جوڑ_c1 = Q6_V_hi_W(v_sums);

        HVX_سمتیہ v_جوڑ_sf_c0 = Q6_Vsf_equals_Vw(v_جوڑ_c0);
        HVX_سمتیہ v_جوڑ_sf_c1 = Q6_Vsf_equals_Vw(v_جوڑ_c1);

        HVX_سمتیہ v_scale_w = vptr[4];
        HVX_سمتیہ v_scale_a_c0 = v_act0[8];
        HVX_سمتیہ v_scale_a_c1 = v_act1[8];

        HVX_سمتیہ v_scale_comb_c0 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_سمتیہ v_scale_comb_c1 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_سمتیہ v_جوڑ_scaled_c0 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c0, v_scale_comb_c0);
        HVX_سمتیہ v_جوڑ_scaled_c1 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c1, v_scale_comb_c1);

        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, v_جوڑ_scaled_c0);
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, v_جوڑ_scaled_c1);
    }

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_q4_1_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 640);
        ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + kt * 1280);

        HVX_سمتیہ v_جوڑ = accum_4bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_سمتیہ v_جوڑ_sf = Q6_Vsf_equals_Vw(v_جوڑ);

        HVX_سمتیہ v_scale_وتھ = vptr[4];
        HVX_سمتیہ_Pair p_deal = Q6_W_vdeal_VVR(v_scale_وتھ, v_scale_وتھ, -2);
        HVX_سمتیہ v_scale = Q6_V_lo_W(p_deal);
        HVX_سمتیہ v_وتھ = Q6_V_hi_W(p_deal);

        HVX_سمتیہ v_scale_a = v_act[8];
        HVX_سمتیہ v_جوڑ_a   = v_act[9];

        HVX_سمتیہ v_scale_comb = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale, v_scale_a);
        HVX_سمتیہ v_وتھ_comb = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_وتھ, v_جوڑ_a);

        HVX_سمتیہ v_scaled_dot = hvx_vec_mul_f32_f32(v_جوڑ_sf, v_scale_comb);
        HVX_سمتیہ v_جوڑ_scaled = hvx_vec_جوڑو_f32_f32(v_scaled_dot, v_وتھ_comb);

        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, v_جوڑ_scaled);
    }

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_q4_1_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 640);
        ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + kt * 1280);
        ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + kt * 1280);

        HVX_سمتیہ_Pair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_سمتیہ v_جوڑ_c0 = Q6_V_lo_W(v_sums);
        HVX_سمتیہ v_جوڑ_c1 = Q6_V_hi_W(v_sums);

        HVX_سمتیہ v_جوڑ_sf_c0 = Q6_Vsf_equals_Vw(v_جوڑ_c0);
        HVX_سمتیہ v_جوڑ_sf_c1 = Q6_Vsf_equals_Vw(v_جوڑ_c1);

        HVX_سمتیہ v_scale_وتھ = vptr[4];
        HVX_سمتیہ_Pair p_deal = Q6_W_vdeal_VVR(v_scale_وتھ, v_scale_وتھ, -2);
        HVX_سمتیہ v_scale = Q6_V_lo_W(p_deal);
        HVX_سمتیہ v_وتھ = Q6_V_hi_W(p_deal);

        HVX_سمتیہ v_scale_a_c0 = v_act0[8];
        HVX_سمتیہ v_جوڑ_a_c0   = v_act0[9];
        HVX_سمتیہ v_scale_a_c1 = v_act1[8];
        HVX_سمتیہ v_جوڑ_a_c1   = v_act1[9];

        HVX_سمتیہ v_scale_comb_c0 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale, v_scale_a_c0);
        HVX_سمتیہ v_وتھ_comb_c0 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_وتھ, v_جوڑ_a_c0);
        HVX_سمتیہ v_scale_comb_c1 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale, v_scale_a_c1);
        HVX_سمتیہ v_وتھ_comb_c1 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_وتھ, v_جوڑ_a_c1);

        HVX_سمتیہ v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c0, v_scale_comb_c0);
        HVX_سمتیہ v_جوڑ_scaled_c0 = hvx_vec_جوڑو_f32_f32(v_scaled_dot_c0, v_وتھ_comb_c0);

        HVX_سمتیہ v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c1, v_scale_comb_c1);
        HVX_سمتیہ v_جوڑ_scaled_c1 = hvx_vec_جوڑو_f32_f32(v_scaled_dot_c1, v_وتھ_comb_c1);

        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, v_جوڑ_scaled_c0);
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, v_جوڑ_scaled_c1);
    }

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_q8_0_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 1152);
        ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + kt * 1152);

        HVX_سمتیہ v_جوڑ = accum_q8_0_32x1(vptr, v_act);
        HVX_سمتیہ v_جوڑ_sf = Q6_Vsf_equals_Vw(v_جوڑ);

        HVX_سمتیہ v_scale_w = vptr[8];
        HVX_سمتیہ v_scale_a = v_act[8];
        HVX_سمتیہ v_scale_comb = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale_w, v_scale_a);
        HVX_سمتیہ v_جوڑ_scaled = hvx_vec_mul_f32_f32(v_جوڑ_sf, v_scale_comb);

        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, v_جوڑ_scaled);
    }

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_q8_0_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 1152);
        ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + kt * 1152);
        ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + kt * 1152);

        HVX_سمتیہ_Pair v_sums = accum_q8_0_32x2(vptr, v_act0, v_act1);
        HVX_سمتیہ v_جوڑ_c0 = Q6_V_lo_W(v_sums);
        HVX_سمتیہ v_جوڑ_c1 = Q6_V_hi_W(v_sums);

        HVX_سمتیہ v_جوڑ_sf_c0 = Q6_Vsf_equals_Vw(v_جوڑ_c0);
        HVX_سمتیہ v_جوڑ_sf_c1 = Q6_Vsf_equals_Vw(v_جوڑ_c1);

        HVX_سمتیہ v_scale_w = vptr[8];
        HVX_سمتیہ v_scale_a_c0 = v_act0[8];
        HVX_سمتیہ v_scale_a_c1 = v_act1[8];

        HVX_سمتیہ v_scale_comb_c0 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_سمتیہ v_scale_comb_c1 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_سمتیہ v_جوڑ_scaled_c0 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c0, v_scale_comb_c0);
        HVX_سمتیہ v_جوڑ_scaled_c1 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c1, v_scale_comb_c1);

        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, v_جوڑ_scaled_c0);
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, v_جوڑ_scaled_c1);
    }

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_q5_k_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 768);
        ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + kt * 1280);

        HVX_سمتیہ v_جوڑ = accum_5bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_سمتیہ v_جوڑ_sf = Q6_Vsf_equals_Vw(v_جوڑ);

        HVX_سمتیہ v_scale_وتھ = vptr[4];
        HVX_سمتیہ_Pair p_deal = Q6_W_vdeal_VVR(v_scale_وتھ, v_scale_وتھ, -2);
        HVX_سمتیہ v_scale = Q6_V_lo_W(p_deal);
        HVX_سمتیہ v_وتھ = Q6_V_hi_W(p_deal);

        HVX_سمتیہ v_scale_a = v_act[8];
        HVX_سمتیہ v_جوڑ_a   = v_act[9];

        HVX_سمتیہ v_scale_comb = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale, v_scale_a);
        HVX_سمتیہ v_وتھ_comb = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_وتھ, v_جوڑ_a);

        HVX_سمتیہ v_scaled_dot = hvx_vec_mul_f32_f32(v_جوڑ_sf, v_scale_comb);
        HVX_سمتیہ v_جوڑ_scaled = hvx_vec_جوڑو_f32_f32(v_scaled_dot, v_وتھ_comb);

        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, v_جوڑ_scaled);
    }

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_q5_k_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 768);
        ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + kt * 1280);
        ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + kt * 1280);

        HVX_سمتیہ_Pair v_sums = accum_5bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_سمتیہ v_جوڑ_c0 = Q6_V_lo_W(v_sums);
        HVX_سمتیہ v_جوڑ_c1 = Q6_V_hi_W(v_sums);

        HVX_سمتیہ v_جوڑ_sf_c0 = Q6_Vsf_equals_Vw(v_جوڑ_c0);
        HVX_سمتیہ v_جوڑ_sf_c1 = Q6_Vsf_equals_Vw(v_جوڑ_c1);

        HVX_سمتیہ v_scale_وتھ = vptr[4];
        HVX_سمتیہ_Pair p_deal = Q6_W_vdeal_VVR(v_scale_وتھ, v_scale_وتھ, -2);
        HVX_سمتیہ v_scale = Q6_V_lo_W(p_deal);
        HVX_سمتیہ v_وتھ = Q6_V_hi_W(p_deal);

        HVX_سمتیہ v_scale_a_c0 = v_act0[8];
        HVX_سمتیہ v_جوڑ_a_c0   = v_act0[9];
        HVX_سمتیہ v_scale_a_c1 = v_act1[8];
        HVX_سمتیہ v_جوڑ_a_c1   = v_act1[9];

        HVX_سمتیہ v_scale_comb_c0 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale, v_scale_a_c0);
        HVX_سمتیہ v_وتھ_comb_c0 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_وتھ, v_جوڑ_a_c0);
        HVX_سمتیہ v_scale_comb_c1 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale, v_scale_a_c1);
        HVX_سمتیہ v_وتھ_comb_c1 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_وتھ, v_جوڑ_a_c1);

        HVX_سمتیہ v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c0, v_scale_comb_c0);
        HVX_سمتیہ v_جوڑ_scaled_c0 = hvx_vec_جوڑو_f32_f32(v_scaled_dot_c0, v_وتھ_comb_c0);

        HVX_سمتیہ v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c1, v_scale_comb_c1);
        HVX_سمتیہ v_جوڑ_scaled_c1 = hvx_vec_جوڑو_f32_f32(v_scaled_dot_c1, v_وتھ_comb_c1);

        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, v_جوڑ_scaled_c0);
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, v_جوڑ_scaled_c1);
    }

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_q6_k_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();
    HVX_سمتیہ i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 896);
        ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + kt * 1152);

        HVX_سمتیہ_Pair v_sums = accum_q6_k_32x1(vptr, v_act, i32);
        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, scale_q6_k_32x1(v_sums, vptr[6], v_act[8]));
    }

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_q6_k_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();
    HVX_سمتیہ i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 896);
        ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + kt * 1152);
        ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + kt * 1152);

        HVX_سمتیہ_Pair v_sums0, v_sums1;
        accum_q6_k_32x2(vptr, v_act0, v_act1, i32, &v_sums0, &v_sums1);

        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, scale_q6_k_32x1(v_sums0, vptr[6], v_act0[8]));
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, scale_q6_k_32x1(v_sums1, vptr[6], v_act1[8]));
    }

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_q3_k_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 512);
        ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + kt * 1152);

        HVX_سمتیہ_Pair v_sums = accum_q3_k_32x1(vptr, v_act);
        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, scale_q6_k_32x1(v_sums, vptr[3], v_act[8]));
    }

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_q3_k_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 512);
        ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + kt * 1152);
        ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + kt * 1152);

        HVX_سمتیہ_Pair v_sums0, v_sums1;
        accum_q3_k_32x2(vptr, v_act0, v_act1, &v_sums0, &v_sums1);

        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, scale_q6_k_32x1(v_sums0, vptr[3], v_act0[8]));
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, scale_q6_k_32x1(v_sums1, vptr[3], v_act1[8]));
    }

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_q2_k_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kb = 0; kb < n_k_tiles; kb += 8) {
        HVX_سمتیہ v_d, v_m;
        q2_k_لدو_dmin(tile_ptr + kb * 384, &v_d, &v_m);

        HVX_سمتیہ v_isum = Q6_V_vzero();
        HVX_سمتیہ v_msum = Q6_V_vzero();
        لكل (uint32_t kt = kb; kt < kb + 8; kt++) {
            ثابت uint8_t * محدود tile = tile_ptr + kt * 384;
            ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + kt * 1280);

            HVX_سمتیہ v_sc2, v_mn2;
            q2_k_scale_جوڑے(tile, &v_sc2, &v_mn2);
            q2_k_int_accum(accum_q2_k_32x1((ثابت HVX_سمتیہ *) tile, v_act), v_sc2, v_mn2, v_act[9], &v_isum, &v_msum);
        }
        ثابت HVX_سمتیہ * محدود v_act_sb = (ثابت HVX_سمتیہ *) (y_q + kb * 1280);
        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, q2_k_int_scale(v_isum, v_msum, v_d, v_m, v_act_sb[8]));
    }

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_q2_k_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kb = 0; kb < n_k_tiles; kb += 8) {
        HVX_سمتیہ v_d, v_m;
        q2_k_لدو_dmin(tile_ptr + kb * 384, &v_d, &v_m);

        HVX_سمتیہ v_isum0 = Q6_V_vzero();
        HVX_سمتیہ v_msum0 = Q6_V_vzero();
        HVX_سمتیہ v_isum1 = Q6_V_vzero();
        HVX_سمتیہ v_msum1 = Q6_V_vzero();
        لكل (uint32_t kt = kb; kt < kb + 8; kt++) {
            ثابت uint8_t * محدود tile = tile_ptr + kt * 384;
            ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + kt * 1280);
            ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + kt * 1280);

            HVX_سمتیہ v_sc2, v_mn2;
            q2_k_scale_جوڑے(tile, &v_sc2, &v_mn2);

            HVX_سمتیہ_Pair v_sums0, v_sums1;
            accum_q2_k_32x2((ثابت HVX_سمتیہ *) tile, v_act0, v_act1, &v_sums0, &v_sums1);
            q2_k_int_accum(v_sums0, v_sc2, v_mn2, v_act0[9], &v_isum0, &v_msum0);
            q2_k_int_accum(v_sums1, v_sc2, v_mn2, v_act1[9], &v_isum1, &v_msum1);
        }
        ثابت HVX_سمتیہ * محدود v_act0_sb = (ثابت HVX_سمتیہ *) (y0_q + kb * 1280);
        ثابت HVX_سمتیہ * محدود v_act1_sb = (ثابت HVX_سمتیہ *) (y1_q + kb * 1280);
        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, q2_k_int_scale(v_isum0, v_msum0, v_d, v_m, v_act0_sb[8]));
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, q2_k_int_scale(v_isum1, v_msum1, v_d, v_m, v_act1_sb[8]));
    }

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

// One Q2_0 weight tile covers 64 k: act tiles 2 * kt and 2 * kt + 1, each 32 k wide (Q8_1)
ساكن فراغ tiled_vec_dot_q2_0_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();
    HVX_سمتیہ v_one_f16   = hvx_vec_splat_f16(1.0);

    uint32_t n_بھار_tiles = n / HTP_MM_بھار_TILE_K_Q2_0;
    لكل (uint32_t kt = 0; kt < n_بھار_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * HTP_MM_بھار_ALIGNED_TILE_ناپ_Q2_0);
        HVX_سمتیہ v_tile_qf = Q6_V_vzero();

        #ہدایت unroll
        لكل (صحيح half = 0; half < 2; half++) {
            ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + (2 * kt + half) * HTP_MM_ACT_TILE_ناپ_Q8_1);

            HVX_سمتیہ v_جوڑ_sf = Q6_Vsf_equals_Vw(accum_q2_0_32x1(vptr + 2 * half, v_act));
            HVX_سمتیہ v_da_sf  = Q6_V_lo_W(hvx_vec_f16_منزل_f32_shuff(v_act[8]));
            HVX_سمتیہ v_sa_qf  = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act[9], v_one_f16));
            HVX_سمتیہ v_h_qf   = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_جوڑ_sf, v_da_sf), v_sa_qf);
            v_tile_qf = (half == 0) ? v_h_qf : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf, v_h_qf);
        }

        HVX_سمتیہ v_dw_sf = Q6_V_lo_W(hvx_vec_f16_منزل_f32(vptr[4]));
        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf), v_dw_sf));
    }

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_q2_0_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();
    HVX_سمتیہ v_one_f16      = hvx_vec_splat_f16(1.0);

    uint32_t n_بھار_tiles = n / HTP_MM_بھار_TILE_K_Q2_0;
    لكل (uint32_t kt = 0; kt < n_بھار_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * HTP_MM_بھار_ALIGNED_TILE_ناپ_Q2_0);
        HVX_سمتیہ v_tile_qf_c0 = Q6_V_vzero();
        HVX_سمتیہ v_tile_qf_c1 = Q6_V_vzero();

        #ہدایت unroll
        لكل (صحيح half = 0; half < 2; half++) {
            ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + (2 * kt + half) * HTP_MM_ACT_TILE_ناپ_Q8_1);
            ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + (2 * kt + half) * HTP_MM_ACT_TILE_ناپ_Q8_1);

            HVX_سمتیہ v_جوڑ_c0, v_جوڑ_c1;
            accum_q2_0_32x2(vptr + 2 * half, v_act0, v_act1, &v_جوڑ_c0, &v_جوڑ_c1);

            HVX_سمتیہ v_da_sf_c0 = Q6_V_lo_W(hvx_vec_f16_منزل_f32_shuff(v_act0[8]));
            HVX_سمتیہ v_da_sf_c1 = Q6_V_lo_W(hvx_vec_f16_منزل_f32_shuff(v_act1[8]));
            HVX_سمتیہ v_sa_qf_c0 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act0[9], v_one_f16));
            HVX_سمتیہ v_sa_qf_c1 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act1[9], v_one_f16));

            HVX_سمتیہ v_h_qf_c0 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(v_جوڑ_c0), v_da_sf_c0), v_sa_qf_c0);
            HVX_سمتیہ v_h_qf_c1 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(v_جوڑ_c1), v_da_sf_c1), v_sa_qf_c1);

            v_tile_qf_c0 = (half == 0) ? v_h_qf_c0 : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c0, v_h_qf_c0);
            v_tile_qf_c1 = (half == 0) ? v_h_qf_c1 : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c1, v_h_qf_c1);
        }

        HVX_سمتیہ v_dw_sf = Q6_V_lo_W(hvx_vec_f16_منزل_f32(vptr[4]));
        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf_c0), v_dw_sf));
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf_c1), v_dw_sf));
    }

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_iq4nl_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();
    HVX_سمتیہ نقاب_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_سمتیہ lut = *(ثابت HVX_سمتیہ *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 640);
        ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + kt * 1152);

        HVX_سمتیہ v_جوڑ = accum_4bit_32x1_lut(vptr, v_act, نقاب_h4, lut);
        HVX_سمتیہ v_جوڑ_sf = Q6_Vsf_equals_Vw(v_جوڑ);

        HVX_سمتیہ v_scale_w = vptr[4];
        HVX_سمتیہ v_scale_a = v_act[8];
        HVX_سمتیہ v_scale_comb = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale_w, v_scale_a);
        HVX_سمتیہ v_جوڑ_scaled = hvx_vec_mul_f32_f32(v_جوڑ_sf, v_scale_comb);

        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, v_جوڑ_scaled);
    }

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_iq4nl_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();
    HVX_سمتیہ نقاب_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_سمتیہ lut = *(ثابت HVX_سمتیہ *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 640);
        ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + kt * 1152);
        ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + kt * 1152);

        HVX_سمتیہ_Pair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, نقاب_h4, lut);
        HVX_سمتیہ v_جوڑ_c0 = Q6_V_lo_W(v_sums);
        HVX_سمتیہ v_جوڑ_c1 = Q6_V_hi_W(v_sums);

        HVX_سمتیہ v_جوڑ_sf_c0 = Q6_Vsf_equals_Vw(v_جوڑ_c0);
        HVX_سمتیہ v_جوڑ_sf_c1 = Q6_Vsf_equals_Vw(v_جوڑ_c1);

        HVX_سمتیہ v_scale_w = vptr[4];
        HVX_سمتیہ v_scale_a_c0 = v_act0[8];
        HVX_سمتیہ v_scale_a_c1 = v_act1[8];

        HVX_سمتیہ v_scale_comb_c0 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_سمتیہ v_scale_comb_c1 = hvx_vec_mul_f16_f16_منزل_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_سمتیہ v_جوڑ_scaled_c0 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c0, v_scale_comb_c0);
        HVX_سمتیہ v_جوڑ_scaled_c1 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c1, v_scale_comb_c1);

        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, v_جوڑ_scaled_c0);
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, v_جوڑ_scaled_c1);
    }

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_mxfp4_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y_q = vy;

    HVX_سمتیہ v_جوڑ_float = Q6_V_vzero();
    HVX_سمتیہ نقاب_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_سمتیہ lut = *(ثابت HVX_سمتیہ *) kvalues_mxfp4_lut;
    HVX_سمتیہ expand = *(ثابت HVX_سمتیہ *) expand_x32_e8m0;
    HVX_سمتیہ e8m0_نقاب = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 640);
        ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) (y_q + kt * 1152);

        HVX_سمتیہ v_جوڑ = accum_4bit_32x1_lut(vptr, v_act, نقاب_h4, lut);
        HVX_سمتیہ v_جوڑ_sf = Q6_Vsf_equals_Vw(v_جوڑ);

        HVX_سمتیہ v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_سمتیہ r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_نقاب);
        HVX_سمتیہ v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_سمتیہ v_scale_a_f16 = v_act[8];
        HVX_سمتیہ_Pair p_scale_a_f32 = hvx_vec_f16_منزل_f32_shuff(v_scale_a_f16);
        HVX_سمتیہ v_scale_a = Q6_V_lo_W(p_scale_a_f32);

        HVX_سمتیہ v_scale_comb = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a);
        HVX_سمتیہ v_جوڑ_scaled = hvx_vec_mul_f32_f32(v_جوڑ_sf, v_scale_comb);

        v_جوڑ_float = hvx_vec_جوڑو_f32_f32(v_جوڑ_float, v_جوڑ_scaled);
    }

    v_جوڑ_float = hvx_vec_mul_f32_f32(v_جوڑ_float, hvx_vec_splat_f32(0.5f));

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ_float);
    }
}

ساكن فراغ tiled_vec_dot_mxfp4_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت uint8_t * محدود tile_ptr = vx;
    ثابت uint8_t * محدود y0_q = vy0;
    ثابت uint8_t * محدود y1_q = vy1;

    HVX_سمتیہ v_جوڑ_float_c0 = Q6_V_vzero();
    HVX_سمتیہ v_جوڑ_float_c1 = Q6_V_vzero();
    HVX_سمتیہ نقاب_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_سمتیہ lut = *(ثابت HVX_سمتیہ *) kvalues_mxfp4_lut;
    HVX_سمتیہ expand = *(ثابت HVX_سمتیہ *) expand_x32_e8m0;
    HVX_سمتیہ e8m0_نقاب = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    لكل (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        ثابت HVX_سمتیہ * محدود vptr = (ثابت HVX_سمتیہ *) (tile_ptr + kt * 640);
        ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) (y0_q + kt * 1152);
        ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) (y1_q + kt * 1152);

        HVX_سمتیہ_Pair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, نقاب_h4, lut);
        HVX_سمتیہ v_جوڑ_c0 = Q6_V_lo_W(v_sums);
        HVX_سمتیہ v_جوڑ_c1 = Q6_V_hi_W(v_sums);

        HVX_سمتیہ v_جوڑ_sf_c0 = Q6_Vsf_equals_Vw(v_جوڑ_c0);
        HVX_سمتیہ v_جوڑ_sf_c1 = Q6_Vsf_equals_Vw(v_جوڑ_c1);

        HVX_سمتیہ v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_سمتیہ r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_نقاب);
        HVX_سمتیہ v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_سمتیہ v_scale_a_c0_f16 = v_act0[8];
        HVX_سمتیہ v_scale_a_c1_f16 = v_act1[8];

        HVX_سمتیہ_Pair p_scale_a_c0_f32 = hvx_vec_f16_منزل_f32_shuff(v_scale_a_c0_f16);
        HVX_سمتیہ_Pair p_scale_a_c1_f32 = hvx_vec_f16_منزل_f32_shuff(v_scale_a_c1_f16);

        HVX_سمتیہ v_scale_a_c0 = Q6_V_lo_W(p_scale_a_c0_f32);
        HVX_سمتیہ v_scale_a_c1 = Q6_V_lo_W(p_scale_a_c1_f32);

        HVX_سمتیہ v_scale_comb_c0 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c0);
        HVX_سمتیہ v_scale_comb_c1 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c1);

        HVX_سمتیہ v_جوڑ_scaled_c0 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c0, v_scale_comb_c0);
        HVX_سمتیہ v_جوڑ_scaled_c1 = hvx_vec_mul_f32_f32(v_جوڑ_sf_c1, v_scale_comb_c1);

        v_جوڑ_float_c0 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, v_جوڑ_scaled_c0);
        v_جوڑ_float_c1 = hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, v_جوڑ_scaled_c1);
    }

    v_جوڑ_float_c0 = hvx_vec_mul_f32_f32(v_جوڑ_float_c0, hvx_vec_splat_f32(0.5f));
    v_جوڑ_float_c1 = hvx_vec_mul_f32_f32(v_جوڑ_float_c1, hvx_vec_splat_f32(0.5f));

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_جوڑ_float_c0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ_float_c1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_جوڑ_float_c1);
    }
}

ساكن فراغ tiled_vec_dot_f16_32x1(ثابت uint32_t n, عائم * محدود s, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy, uint32_t valid_صفاں, ثابت عائم * محدود sz) {
    ثابت HVX_سمتیہ * محدود vptr  = (ثابت HVX_سمتیہ *) vx;
    ثابت HVX_سمتیہ * محدود v_act = (ثابت HVX_سمتیہ *) vy;
    ثابت uint32_t n_vecs = ((n + 31) / 32) * 16;

    HVX_سمتیہ_Pair acc = Q6_W_vzero();

    #ہدایت unroll(4)
    لكل (uint32_t i = 0; i < n_vecs; i++) {
        acc = hvx_vec_mpyacc_f32_f16(acc, vptr[i], v_act[i]);
    }

    HVX_سمتیہ v_جوڑ = hvx_vec_جوڑو_f32_f32(Q6_V_lo_W(acc), Q6_V_hi_W(acc));

    إذا (sz) {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_جوڑ, hvx_vmemu(sz)));
    } وإلا {
        hvx_vec_store_u(s, valid_صفاں * حجم(عائم), v_جوڑ);
    }
}

ساكن فراغ tiled_vec_dot_f16_32x2(ثابت uint32_t n, عائم * محدود s0, عائم * محدود s1, ثابت فراغ * محدود vx, ثابت فراغ * محدود vy0, ثابت فراغ * محدود vy1, uint32_t valid_صفاں, ثابت عائم * محدود sz0, ثابت عائم * محدود sz1) {
    ثابت HVX_سمتیہ * محدود vptr   = (ثابت HVX_سمتیہ *) vx;
    ثابت HVX_سمتیہ * محدود v_act0 = (ثابت HVX_سمتیہ *) vy0;
    ثابت HVX_سمتیہ * محدود v_act1 = (ثابت HVX_سمتیہ *) vy1;
    ثابت uint32_t n_vecs = ((n + 31) / 32) * 16;

    HVX_سمتیہ_Pair acc0 = Q6_W_vzero();
    HVX_سمتیہ_Pair acc1 = Q6_W_vzero();

    #ہدایت unroll(4)
    لكل (uint32_t i = 0; i < n_vecs; i++) {
        HVX_سمتیہ v_w = vptr[i];
        acc0 = hvx_vec_mpyacc_f32_f16(acc0, v_w, v_act0[i]);
        acc1 = hvx_vec_mpyacc_f32_f16(acc1, v_w, v_act1[i]);
    }

    HVX_سمتیہ v_sum0 = hvx_vec_جوڑو_f32_f32(Q6_V_lo_W(acc0), Q6_V_hi_W(acc0));
    HVX_سمتیہ v_sum1 = hvx_vec_جوڑو_f32_f32(Q6_V_lo_W(acc1), Q6_V_hi_W(acc1));

    إذا (sz0) {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_sum0, hvx_vmemu(sz0)));
    } وإلا {
        hvx_vec_store_u(s0, valid_صفاں * حجم(عائم), v_sum0);
    }
    إذا (sz1) {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), hvx_vec_جوڑو_f32_f32(v_sum1, hvx_vmemu(sz1)));
    } وإلا {
        hvx_vec_store_u(s1, valid_صفاں * حجم(عائم), v_sum1);
    }
}


ساكن مضمن فراغ quantize_f32_q8_0_tiled_kernel(
    ثابت uint8_t * محدود src_انکڑے,
    uint8_t * محدود dst_انکڑے,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ
) {
    (فراغ) tmp_انکڑے;
    لكل (uint32_t i = 0; i < nrows; ++i) {
        quantize_صف_f32_q8_0_tiled((عائم *) src_انکڑے, dst_انکڑے, ne0);
        dst_انکڑے += dst_صف_ناپ;
        src_انکڑے += src_صف_ناپ;
    }
}

ساكن مضمن فراغ quantize_f32_q8_1_tiled_kernel(
    ثابت uint8_t * محدود src_انکڑے,
    uint8_t * محدود dst_انکڑے,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ
) {
    (فراغ) tmp_انکڑے;
    لكل (uint32_t i = 0; i < nrows; ++i) {
        quantize_صف_f32_q8_1_tiled((عائم *) src_انکڑے, dst_انکڑے, ne0);
        dst_انکڑے += dst_صف_ناپ;
        src_انکڑے += src_صف_ناپ;
    }
}

ساكن مضمن فراغ quantize_f32_q8_1_s16_tiled_kernel(
    ثابت uint8_t * محدود src_انکڑے,
    uint8_t * محدود dst_انکڑے,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ
) {
    (فراغ) tmp_انکڑے;
    لكل (uint32_t i = 0; i < nrows; ++i) {
        quantize_صف_f32_q8_1_s16_tiled((عائم *) src_انکڑے, dst_انکڑے, ne0);
        dst_انکڑے += dst_صف_ناپ;
        src_انکڑے += src_صف_ناپ;
    }
}

ساكن مضمن فراغ quantize_f32_f16_tiled_kernel(
    ثابت uint8_t * محدود src_انکڑے,
    uint8_t * محدود dst_انکڑے,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ
) {
    (فراغ) tmp_انکڑے;
    لكل (uint32_t i = 0; i < nrows; ++i) {
        quantize_صف_f32_f16_tiled((ثابت عائم *) src_انکڑے, dst_انکڑے, ne0);
        dst_انکڑے += dst_صف_ناپ;
        src_انکڑے += src_صف_ناپ;
    }
}

ساكن مضمن فراغ quantize_f16_f16_tiled_kernel(
    ثابت uint8_t * محدود src_انکڑے,
    uint8_t * محدود dst_انکڑے,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ
) {
    (فراغ) tmp_انکڑے;
    لكل (uint32_t i = 0; i < nrows; ++i) {
        quantize_صف_f16_f16_tiled((ثابت __fp16 *) src_انکڑے, dst_انکڑے, ne0);
        dst_انکڑے += dst_صف_ناپ;
        src_انکڑے += src_صف_ناپ;
    }
}

ساكن مضمن فراغ quantize_f32_q8_0_tiled_بلاک_kernel(
    ثابت عائم * محدود src,
    uint8_t * محدود dst,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t ib_پہلا,
    uint32_t ib_آخری,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ,
    uint32_t r,
    uint32_t c
) {
    (فراغ) tmp_انکڑے;
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne0 + qk - 1) / qk;

    لكل (uint32_t ib = ib_پہلا; ib < ib_آخری; ++ib) {
        ثابت عائم * محدود src_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + r * src_صف_ناپ + c * qk * حجم(عائم));
        uint8_t * محدود dst_ptr = dst + r * dst_صف_ناپ + c * 4 * 1152;

        quantize_بلاک_f32_q8_0_tiled((عائم *) src_ptr, dst_ptr);

        c++;
        إذا (c == nb) {
            c = 0;
            r++;
        }
    }
}

ساكن مضمن فراغ quantize_f32_q8_1_tiled_بلاک_kernel(
    ثابت عائم * محدود src,
    uint8_t * محدود dst,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t ib_پہلا,
    uint32_t ib_آخری,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ,
    uint32_t r,
    uint32_t c
) {
    (فراغ) tmp_انکڑے;
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne0 + qk - 1) / qk;

    لكل (uint32_t ib = ib_پہلا; ib < ib_آخری; ++ib) {
        ثابت عائم * محدود src_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + r * src_صف_ناپ + c * qk * حجم(عائم));
        uint8_t * محدود dst_ptr = dst + r * dst_صف_ناپ + c * 4 * 1280;

        quantize_بلاک_f32_q8_1_tiled((عائم *) src_ptr, dst_ptr);

        c++;
        إذا (c == nb) {
            c = 0;
            r++;
        }
    }
}

ساكن مضمن فراغ quantize_f32_q8_1_s16_tiled_بلاک_kernel(
    ثابت عائم * محدود src,
    uint8_t * محدود dst,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t ib_پہلا,
    uint32_t ib_آخری,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ,
    uint32_t r,
    uint32_t c
) {
    (فراغ) tmp_انکڑے;
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne0 + qk - 1) / qk;

    لكل (uint32_t ib = ib_پہلا; ib < ib_آخری; ++ib) {
        ثابت عائم * محدود src_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + r * src_صف_ناپ + c * qk * حجم(عائم));
        uint8_t * محدود dst_ptr = dst + r * dst_صف_ناپ + c * 4 * 1280;

        quantize_بلاک_f32_q8_1_s16_tiled((عائم *) src_ptr, dst_ptr, src_ptr - (c & 1) * qk);  // super-block of 256 k

        c++;
        إذا (c == nb) {
            c = 0;
            r++;
        }
    }
}

ساكن مضمن فراغ quantize_f32_f16_tiled_بلاک_kernel(
    ثابت عائم * محدود src,
    uint8_t * محدود dst,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t ib_پہلا,
    uint32_t ib_آخری,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ,
    uint32_t r,
    uint32_t c
) {
    (فراغ) tmp_انکڑے;
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne0 + qk - 1) / qk;

    لكل (uint32_t ib = ib_پہلا; ib < ib_آخری; ++ib) {
        ثابت عائم * محدود src_ptr = (ثابت عائم *) ((ثابت uint8_t *) src + r * src_صف_ناپ + c * qk * حجم(عائم));
        uint8_t * محدود dst_ptr = dst + r * dst_صف_ناپ + c * 4 * 2048;

        quantize_بلاک_f32_f16_tiled(src_ptr, dst_ptr);

        c++;
        إذا (c == nb) {
            c = 0;
            r++;
        }
    }
}

ساكن مضمن فراغ quantize_f16_f16_tiled_بلاک_kernel(
    ثابت __fp16 * محدود src,
    uint8_t * محدود dst,
    uint8_t * محدود tmp_انکڑے,
    uint32_t ne0,
    uint32_t ib_پہلا,
    uint32_t ib_آخری,
    size_t src_صف_ناپ,
    size_t dst_صف_ناپ,
    uint32_t r,
    uint32_t c
) {
    (فراغ) tmp_انکڑے;
    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne0 + qk - 1) / qk;

    لكل (uint32_t ib = ib_پہلا; ib < ib_آخری; ++ib) {
        ثابت __fp16 * محدود src_ptr = (ثابت __fp16 *) ((ثابت uint8_t *) src + r * src_صف_ناپ + c * qk * حجم(__fp16));
        uint8_t * محدود dst_ptr = dst + r * dst_صف_ناپ + c * 4 * 2048;

        quantize_بلاک_f16_f16_tiled(src_ptr, dst_ptr);

        c++;
        إذا (c == nb) {
            c = 0;
            r++;
        }
    }
}
