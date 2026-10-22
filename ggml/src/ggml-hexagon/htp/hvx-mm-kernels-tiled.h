// Dynamic quantizers that produce tiled activations

// q8_1: vector 9 holds d * sum(q) of the 32 k, replicated.
// q8_1_s16 (Q2_K, sums16): one fp16 d (vector 8) for the 256 k super-block starting at x_sb, and vector 9 holds the
// integer sums of k 0..15 (low halfword) and k 16..31 (high halfword) in every word
__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य quantize_block_f32_q8_1_tiled_impl(भिन्न * restrict x, uint8_t * restrict y_block, अचर bool sums16,
                                                      अचर भिन्न * restrict x_sb) {
    प्रतिज्ञा((अचिन्हित दीर्घ) x % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero = Q6_V_vzero();

    HVX_Vector vmax0_sf, vmax1_sf, vmax2_sf, vmax3_sf;
    यदि (sums16) {
        अचर HVX_Vector * vsb = (अचर HVX_Vector *) x_sb;
        HVX_Vector vm = hvx_vec_abs_f32(vsb[0]);
        लागि (पूर्णाङ्क i = 1; i < 8; i++) {
            vm = Q6_Vsf_vmax_VsfVsf(vm, hvx_vec_abs_f32(vsb[i]));
        }
        vmax0_sf = vmax1_sf = vmax2_sf = vmax3_sf = hvx_vec_reduce_max_f32(vm);
    } अन्यथा {
        vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
        vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
        vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
        vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));
    }

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);

    HVX_Vector vmax01_hf   = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vx01_hf     = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vd01_qf16   = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd01_hf     = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));

    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_Vector vmax23_hf   = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));
    HVX_Vector vx23_hf     = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));
    HVX_Vector vd23_qf16   = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd23_hf     = Q6_Vhf_equals_Vqf16(vd23_qf16);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    अचर HVX_Vector ones = Q6_Vb_vsplat_R(1);
    HVX_Vector v_sums = Q6_Vw_vrmpy_VbVb(vx_i8, ones);
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 4));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 8));
    यदि (!sums16) {
        v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 16));
    }
    // word 8b: sum of block b (sums16: k 0..15, and word 8b + 4: k 16..31)

    अचर HVX_Vector v_inv127 = hvx_vec_splat_f32(1.0f / 127.0f);
    HVX_Vector vd0_sf = hvx_vec_mul_f32_f32(vmax0_sf, v_inv127);
    HVX_Vector vd1_sf = hvx_vec_mul_f32_f32(vmax1_sf, v_inv127);
    HVX_Vector vd2_sf = hvx_vec_mul_f32_f32(vmax2_sf, v_inv127);
    HVX_Vector vd3_sf = hvx_vec_mul_f32_f32(vmax3_sf, v_inv127);

    HVX_Vector v_sums_sf = Q6_Vsf_equals_Vw(v_sums);
    HVX_Vector voff0_sf = hvx_vec_mul_f32_f32(vd0_sf, v_sums_sf);
    HVX_Vector voff1_sf = hvx_vec_mul_f32_f32(vd1_sf, Q6_V_vror_VR(v_sums_sf, 32));
    HVX_Vector voff2_sf = hvx_vec_mul_f32_f32(vd2_sf, Q6_V_vror_VR(v_sums_sf, 64));
    HVX_Vector voff3_sf = hvx_vec_mul_f32_f32(vd3_sf, Q6_V_vror_VR(v_sums_sf, 96));

    HVX_Vector voff01_hf = hvx_vec_f32_to_f16(voff0_sf, voff1_sf);
    HVX_Vector voff23_hf = hvx_vec_f32_to_f16(voff2_sf, voff3_sf);

    int32_t __attribute__((aligned(128))) ws[32];
    यदि (sums16) {
        // |sum of 16 int8| <= 2032 fits a halfword
        *(HVX_Vector *) ws = v_sums;
    }

    स्थिर अचर uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_Vector v_repl_ctrl = * (अचर HVX_Vector *) repl;

    #pragma unroll
    लागि (पूर्णाङ्क b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_Vector vd_b   = (b < 2) ? vd01_hf   : vd23_hf;
        HVX_Vector voff_b = (b < 2) ? voff01_hf : voff23_hf;
        HVX_Vector r_scale  = hvx_vec_repl_f16((b & 1) ? Q6_V_vror_VR(vd_b, 64)   : vd_b);
        HVX_Vector r_offset = sums16
            ? Q6_V_vsplat_R(((uint32_t) ws[8 * b + 4] << 16) | ((uint32_t) ws[8 * b] & 0xFFFF))
            : hvx_vec_repl_f16((b & 1) ? Q6_V_vror_VR(voff_b, 64) : voff_b);

        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1280);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
        dst[9] = r_offset;
    }
}

स्थिर अन्तःस्थ शून्य quantize_block_f32_q8_1_tiled(भिन्न * restrict x, uint8_t * restrict y_block) {
    quantize_block_f32_q8_1_tiled_impl(x, y_block, false, रिक्त);
}

स्थिर अन्तःस्थ शून्य quantize_block_f32_q8_1_s16_tiled(भिन्न * restrict x, uint8_t * restrict y_block, अचर भिन्न * restrict x_sb) {
    quantize_block_f32_q8_1_tiled_impl(x, y_block, true, x_sb);
}

स्थिर अन्तःस्थ शून्य quantize_block_f32_q8_0_tiled(भिन्न * restrict x, uint8_t * restrict y_block) {
    प्रतिज्ञा((अचिन्हित दीर्घ) x % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero   = Q6_V_vzero();

    HVX_Vector vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_Vector vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_Vector vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_Vector vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    HVX_VectorPair vp01 = Q6_W_vshuff_VVR(vd01_hf, vd01_hf, -64);
    HVX_VectorPair vp23 = Q6_W_vshuff_VVR(vd23_hf, vd23_hf, -64);

    स्थिर अचर uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_Vector v_repl_ctrl = * (अचर HVX_Vector *) repl;

    #pragma unroll
    लागि (पूर्णाङ्क b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_Vector r_scale;
        यदि (b == 0) {
            r_scale = Q6_V_lo_W(vp01);
        } अन्यथा यदि (b == 1) {
            r_scale = Q6_V_hi_W(vp01);
        } अन्यथा यदि (b == 2) {
            r_scale = Q6_V_lo_W(vp23);
        } अन्यथा {
            r_scale = Q6_V_hi_W(vp23);
        }

        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1152);
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

स्थिर शून्य quantize_row_f32_q8_0_tiled(भिन्न * restrict x, uint8_t * restrict y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (k + qk - 1) / qk;

    लागि (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 1152;
        quantize_block_f32_q8_0_tiled(x + i * qk, y_block);
    }
}

स्थिर शून्य quantize_row_f32_q8_1_tiled(भिन्न * restrict x, uint8_t * restrict y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (k + qk - 1) / qk;

    लागि (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 1280;
        quantize_block_f32_q8_1_tiled(x + i * qk, y_block);
    }
}

स्थिर शून्य quantize_row_f32_q8_1_s16_tiled(भिन्न * restrict x, uint8_t * restrict y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (k + qk - 1) / qk;

    लागि (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 1280;
        quantize_block_f32_q8_1_s16_tiled(x + i * qk, y_block, x + (i & ~1u) * qk);  // k % 256 == 0 for Q2_K
    }
}

स्थिर अन्तःस्थ शून्य quantize_half_f16_tiled_splat(HVX_Vector v_src, HVX_Vector * restrict dst) {
    स्थिर अचर uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_Vector v_repl_ctrl = *(अचर HVX_Vector *) repl;

    #pragma unroll(4)
    लागि (पूर्णाङ्क cp = 0; cp < 32; cp++) {
        dst[cp] = Q6_V_vdelta_VV(cp == 0 ? v_src : Q6_V_vror_VR(v_src, cp * 4), v_repl_ctrl);
    }
}

स्थिर अन्तःस्थ शून्य quantize_block_f32_f16_tiled(अचर भिन्न * restrict x, uint8_t * restrict y_block) {
    प्रतिज्ञा((अचिन्हित दीर्घ) x % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) y_block % 128 == 0);

    अचर HVX_Vector * vx = (अचर HVX_Vector *) x;
    HVX_Vector * restrict dst = (HVX_Vector *) y_block;

    HVX_Vector vx01_hf = hvx_vec_f32_to_f16(vx[0], vx[1]);
    quantize_half_f16_tiled_splat(vx01_hf, dst);

    HVX_Vector vx23_hf = hvx_vec_f32_to_f16(vx[2], vx[3]);
    quantize_half_f16_tiled_splat(vx23_hf, dst + 32);
}

स्थिर अन्तःस्थ शून्य quantize_block_f16_f16_tiled(अचर __fp16 * restrict x, uint8_t * restrict y_block) {
    प्रतिज्ञा((अचिन्हित दीर्घ) x % 128 == 0);
    प्रतिज्ञा((अचिन्हित दीर्घ) y_block % 128 == 0);

    अचर HVX_Vector * vx = (अचर HVX_Vector *) x;
    HVX_Vector * restrict dst = (HVX_Vector *) y_block;

    quantize_half_f16_tiled_splat(vx[0], dst);
    quantize_half_f16_tiled_splat(vx[1], dst + 32);
}

स्थिर शून्य quantize_row_f32_f16_tiled(अचर भिन्न * restrict x, uint8_t * restrict y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (k + qk - 1) / qk;

    लागि (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 2048;
        quantize_block_f32_f16_tiled(x + i * qk, y_block);
    }
}

स्थिर शून्य quantize_row_f16_f16_tiled(अचर __fp16 * restrict x, uint8_t * restrict y, uint32_t k) {
    प्रतिज्ञा(k % 32 == 0);
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (k + qk - 1) / qk;

    लागि (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 2048;
        quantize_block_f16_f16_tiled(x + i * qk, y_block);
    }
}

// Dot kernels & helpers that consume tiled activations

स्थिर अन्तःस्थ HVX_Vector hvx_vec_mul_f16_f16_to_f32_lower32(HVX_Vector v1, HVX_Vector v2) {
#यदि __HVX_ARCH__ >= 79
    HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    फर्काऊ Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4));
#अन्यथा
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_Vector hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_Vector lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    फर्काऊ Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -4));
#endif
}

// both halves of hvx_vec_mul_f16_f16_to_f32_lower32: lo = products of lanes 0..31, hi = lanes 32..63
स्थिर अन्तःस्थ HVX_VectorPair hvx_vec_mul_f16_f16_to_f32_pair(HVX_Vector v1, HVX_Vector v2) {
#यदि __HVX_ARCH__ >= 79
    HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    फर्काऊ Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4);
#अन्यथा
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_Vector hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_Vector lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    फर्काऊ Q6_W_vshuff_VVR(hi, lo, -4);
#endif
}

स्थिर अन्तःस्थ HVX_Vector unpack_and_interleave_4bit(HVX_Vector v_a, HVX_Vector v_b, HVX_Vector mask_h4) {
    HVX_Vector v_W0 = Q6_V_vand_VV(v_a, mask_h4);
    HVX_Vector v_W1 = Q6_Vub_vlsr_VubR(v_a, 4);
    HVX_Vector v_W2 = Q6_V_vand_VV(v_b, mask_h4);
    HVX_Vector v_W3 = Q6_Vub_vlsr_VubR(v_b, 4);

    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_W1, v_W0, -1);
    HVX_VectorPair v23_pair = Q6_W_vshuff_VVR(v_W3, v_W2, -1);
    HVX_VectorPair v0123_pair = Q6_W_vshuff_VVR(Q6_V_lo_W(v23_pair), Q6_V_lo_W(v01_pair), -2);
    फर्काऊ Q6_V_lo_W(v0123_pair);
}

स्थिर अन्तःस्थ HVX_VectorPair unpack_and_interleave_4bit_x2(HVX_Vector v_src, HVX_Vector mask_h4) {
    HVX_Vector v_lo = Q6_V_vand_VV(v_src, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(v_src, 4);
    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_Vector v01_lo = Q6_V_lo_W(v01_pair);
    HVX_Vector v01_hi = Q6_V_hi_W(v01_pair);

    HVX_Vector v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_Vector v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_Vector v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_Vector v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    फर्काऊ Q6_W_vcombine_VV(v_W1, v_W0);
}

स्थिर अन्तःस्थ HVX_Vector unpack_q1_0_group(
    HVX_Vector v_plane,
    पूर्णाङ्क g,
    HVX_Vector v_neg_one,
    HVX_Vector v_pos_one
) {
    HVX_VectorPred q = Q6_Q_vand_VR(v_plane, 0x01010101u << g);
    फर्काऊ Q6_V_vmux_QVV(q, v_pos_one, v_neg_one);
}

स्थिर अन्तःस्थ HVX_Vector accum_q1_0_32x1(
    HVX_Vector v_plane,
    अचर HVX_Vector * restrict v_act,
    HVX_Vector v_neg_one,
    HVX_Vector v_pos_one
) {
    HVX_Vector v_sum_lo = Q6_V_vzero();
    HVX_Vector v_sum_hi = Q6_V_vzero();

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q1_0_group(v_plane, g,     v_neg_one, v_pos_one);
        HVX_Vector v_W_hi = unpack_q1_0_group(v_plane, g + 4, v_neg_one, v_pos_one);
        v_sum_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum_lo, v_W_lo, v_act[g]);
        v_sum_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum_hi, v_W_hi, v_act[g + 4]);
    }

    फर्काऊ Q6_Vw_vadd_VwVw(v_sum_lo, v_sum_hi);
}

स्थिर अन्तःस्थ HVX_Vector accum_4bit_32x1(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    लागि (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    फर्काऊ Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

स्थिर अन्तःस्थ HVX_Vector accum_4bit_32x1_lut(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    लागि (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    फर्काऊ Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

स्थिर अन्तःस्थ HVX_VectorPair accum_q1_0_32x2(
    HVX_Vector v_plane,
    अचर HVX_Vector * restrict v_act0,
    अचर HVX_Vector * restrict v_act1,
    HVX_Vector v_neg_one,
    HVX_Vector v_pos_one
) {
    HVX_Vector v_sum0_lo = Q6_V_vzero();
    HVX_Vector v_sum0_hi = Q6_V_vzero();
    HVX_Vector v_sum1_lo = Q6_V_vzero();
    HVX_Vector v_sum1_hi = Q6_V_vzero();

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q1_0_group(v_plane, g,     v_neg_one, v_pos_one);
        HVX_Vector v_W_hi = unpack_q1_0_group(v_plane, g + 4, v_neg_one, v_pos_one);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    फर्काऊ Q6_W_vcombine_VV(
        Q6_Vw_vadd_VwVw(v_sum1_lo, v_sum1_hi),
        Q6_Vw_vadd_VwVw(v_sum0_lo, v_sum0_hi));
}

स्थिर अन्तःस्थ HVX_VectorPair accum_4bit_32x2(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act0,
    अचर HVX_Vector * restrict v_act1,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    लागि (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    फर्काऊ Q6_W_vcombine_VV(v_sum1, v_sum0);
}

स्थिर अन्तःस्थ HVX_VectorPair accum_4bit_32x2_lut(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act0,
    अचर HVX_Vector * restrict v_act1,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    लागि (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    फर्काऊ Q6_W_vcombine_VV(v_sum1, v_sum0);
}

स्थिर अन्तःस्थ HVX_Vector accum_q8_0_32x1(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act
) {
    HVX_Vector v_sum = Q6_V_vzero();
    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum = Q6_Vw_vrmpyacc_VwVbVb(v_sum, v_W, v_act[g]);
    }
    फर्काऊ v_sum;
}

स्थिर अन्तःस्थ HVX_VectorPair accum_q8_0_32x2(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act0,
    अचर HVX_Vector * restrict v_act1
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W, v_act0[g]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W, v_act1[g]);
    }
    फर्काऊ Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q5_K: OR 0x10 into every lane of v whose flag j is set in the plane (see HTP_TENSOR_TILE_SIZE_Q5_K)
स्थिर अन्तःस्थ HVX_Vector hvx_q5k_or_hibit(HVX_Vector v, HVX_Vector v_plane, पूर्णाङ्क j) {
    HVX_VectorPred q = Q6_Q_vand_VR(v_plane, 0x01010101u << j);
    फर्काऊ Q6_V_vandor_VQR(v, q, 0x10101010);
}

// 5-bit variant: the high bit comes from the plane, see hvx_q5k_or_hibit
स्थिर अन्तःस्थ HVX_VectorPair unpack_and_interleave_5bit_x2(HVX_Vector v_src, HVX_Vector v_plane, पूर्णाङ्क i, HVX_Vector mask_h4) {
    HVX_Vector v_lo = hvx_q5k_or_hibit(Q6_V_vand_VV(v_src, mask_h4), v_plane, 2 * i);
    HVX_Vector v_hi = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(v_src, 4), v_plane, 2 * i + 1);
    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_Vector v01_lo = Q6_V_lo_W(v01_pair);
    HVX_Vector v01_hi = Q6_V_hi_W(v01_pair);

    HVX_Vector v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_Vector v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_Vector v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_Vector v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    फर्काऊ Q6_W_vcombine_VV(v_W1, v_W0);
}

स्थिर अन्तःस्थ HVX_Vector accum_5bit_32x1(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_plane = vptr[5];

    #pragma unroll
    लागि (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    फर्काऊ Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

स्थिर अन्तःस्थ HVX_VectorPair accum_5bit_32x2(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act0,
    अचर HVX_Vector * restrict v_act1,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_plane = vptr[5];

    #pragma unroll
    लागि (पूर्णाङ्क i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    फर्काऊ Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q6_K weights are stored unsigned (0..63), see HTP_TENSOR_TILE_SIZE_Q6_K. Unpack k-group g of a tile to signed bytes (q - 32)
स्थिर अन्तःस्थ HVX_Vector unpack_q6_k_group(अचर HVX_Vector * restrict vptr, पूर्णाङ्क g, HVX_Vector mask_0f, HVX_Vector mask_03, HVX_Vector i32) {
    HVX_Vector v_lo = (g & 1) ? Q6_Vub_vlsr_VubR(vptr[g >> 1], 4) : Q6_V_vand_VV(vptr[g >> 1], mask_0f);
    HVX_Vector v_hi = (g & 3) ? Q6_Vub_vlsr_VubR(vptr[4 + (g >> 2)], 2 * (g & 3)) : vptr[4 + (g >> 2)];
    HVX_Vector v_q  = Q6_V_vor_VV(v_lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_hi, mask_03), 4));
    फर्काऊ Q6_Vb_vsub_VbVb(v_q, i32);
}

// k 0..15 and k 16..31 of a Q6_K tile have different scales: lo half of the pair sums k 0..15, hi half sums k 16..31
स्थिर अन्तःस्थ HVX_VectorPair accum_q6_k_32x1(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act,
    HVX_Vector i32
) {
    HVX_Vector v_sum_lo = Q6_V_vzero();
    HVX_Vector v_sum_hi = Q6_V_vzero();
    HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q6_k_group(vptr, g,     mask_0f, mask_03, i32);
        HVX_Vector v_W_hi = unpack_q6_k_group(vptr, g + 4, mask_0f, mask_03, i32);
        v_sum_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum_lo, v_W_lo, v_act[g]);
        v_sum_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum_hi, v_W_hi, v_act[g + 4]);
    }

    फर्काऊ Q6_W_vcombine_VV(v_sum_hi, v_sum_lo);
}

स्थिर अन्तःस्थ शून्य accum_q6_k_32x2(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act0,
    अचर HVX_Vector * restrict v_act1,
    HVX_Vector i32,
    HVX_VectorPair * v_sums0,
    HVX_VectorPair * v_sums1
) {
    HVX_Vector v_sum0_lo = Q6_V_vzero();
    HVX_Vector v_sum0_hi = Q6_V_vzero();
    HVX_Vector v_sum1_lo = Q6_V_vzero();
    HVX_Vector v_sum1_hi = Q6_V_vzero();
    HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q6_k_group(vptr, g,     mask_0f, mask_03, i32);
        HVX_Vector v_W_hi = unpack_q6_k_group(vptr, g + 4, mask_0f, mask_03, i32);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// scale the two half sums with the per-row tile scales (v_scale_w = vptr[6]) and the activation scale
स्थिर अन्तःस्थ HVX_Vector scale_q6_k_32x1(HVX_VectorPair v_sums, HVX_Vector v_scale_w, HVX_Vector v_scale_a) {
    HVX_VectorPair v_scale = hvx_vec_mul_f16_f16_to_f32_pair(v_scale_w, v_scale_a);
    HVX_Vector v_lo = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums)), Q6_V_lo_W(v_scale));
    HVX_Vector v_hi = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums)), Q6_V_hi_W(v_scale));
    फर्काऊ hvx_vec_add_f32_f32(v_lo, v_hi);
}

// Q3_K / Q2_K: low 2 bits of k-group g, see HTP_TENSOR_TILE_SIZE_Q3_K
स्थिर अन्तःस्थ HVX_Vector unpack_q3_k_low2(अचर HVX_Vector * restrict vptr, पूर्णाङ्क g, HVX_Vector mask_03) {
    HVX_Vector v = vptr[g >> 2];
    यदि ((g & 3) == 3) {
        फर्काऊ Q6_Vub_vlsr_VubR(v, 6);
    }
    यदि (g & 3) {
        v = Q6_Vub_vlsr_VubR(v, 2 * (g & 3));
    }
    फर्काऊ Q6_V_vand_VV(v, mask_03);
}

// Q3_K k-group g as signed bytes: low2 | 0xFC (= low2 - 4) where bit g of vector 2 is set
स्थिर अन्तःस्थ HVX_Vector unpack_q3_k_group(अचर HVX_Vector * restrict vptr, पूर्णाङ्क g, HVX_Vector mask_03) {
    HVX_VectorPred q_neg = Q6_Q_vand_VR(vptr[2], 0x01010101u << g);
    फर्काऊ Q6_V_vandor_VQR(unpack_q3_k_low2(vptr, g, mask_03), q_neg, 0xFCFCFCFC);
}

// same half split as accum_q6_k_32x1
स्थिर अन्तःस्थ HVX_VectorPair accum_q3_k_32x1(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act
) {
    HVX_Vector v_sum_lo = Q6_V_vzero();
    HVX_Vector v_sum_hi = Q6_V_vzero();
    HVX_Vector mask_03  = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q3_k_group(vptr, g,     mask_03);
        HVX_Vector v_W_hi = unpack_q3_k_group(vptr, g + 4, mask_03);
        v_sum_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum_lo, v_W_lo, v_act[g]);
        v_sum_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum_hi, v_W_hi, v_act[g + 4]);
    }

    फर्काऊ Q6_W_vcombine_VV(v_sum_hi, v_sum_lo);
}

स्थिर अन्तःस्थ शून्य accum_q3_k_32x2(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act0,
    अचर HVX_Vector * restrict v_act1,
    HVX_VectorPair * v_sums0,
    HVX_VectorPair * v_sums1
) {
    HVX_Vector v_sum0_lo = Q6_V_vzero();
    HVX_Vector v_sum0_hi = Q6_V_vzero();
    HVX_Vector v_sum1_lo = Q6_V_vzero();
    HVX_Vector v_sum1_hi = Q6_V_vzero();
    HVX_Vector mask_03   = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q3_k_group(vptr, g,     mask_03);
        HVX_Vector v_W_hi = unpack_q3_k_group(vptr, g + 4, mask_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// Q2_K: the Q3_K dot products without the -4 flags, scales and mins are applied in q2_k_int_accum
स्थिर अन्तःस्थ HVX_VectorPair accum_q2_k_32x1(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act
) {
    HVX_Vector v_sum_lo = Q6_V_vzero();
    HVX_Vector v_sum_hi = Q6_V_vzero();
    HVX_Vector mask_03  = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q3_k_low2(vptr, g,     mask_03);
        HVX_Vector v_W_hi = unpack_q3_k_low2(vptr, g + 4, mask_03);
        v_sum_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum_lo, v_W_lo, v_act[g]);
        v_sum_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum_hi, v_W_hi, v_act[g + 4]);
    }

    फर्काऊ Q6_W_vcombine_VV(v_sum_hi, v_sum_lo);
}

स्थिर अन्तःस्थ शून्य accum_q2_k_32x2(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act0,
    अचर HVX_Vector * restrict v_act1,
    HVX_VectorPair * v_sums0,
    HVX_VectorPair * v_sums1
) {
    HVX_Vector v_sum0_lo = Q6_V_vzero();
    HVX_Vector v_sum0_hi = Q6_V_vzero();
    HVX_Vector v_sum1_lo = Q6_V_vzero();
    HVX_Vector v_sum1_hi = Q6_V_vzero();
    HVX_Vector mask_03   = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q3_k_low2(vptr, g,     mask_03);
        HVX_Vector v_W_hi = unpack_q3_k_low2(vptr, g + 4, mask_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// Q2_0: unsigned codes 0..3; the -1 offset is applied via the Q8_1 block sum (v_act[9])
स्थिर अन्तःस्थ HVX_Vector accum_q2_0_32x1(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act
) {
    HVX_Vector v_sum_lo = Q6_V_vzero();
    HVX_Vector v_sum_hi = Q6_V_vzero();
    HVX_Vector mask_03  = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q3_k_low2(vptr, g,     mask_03);
        HVX_Vector v_W_hi = unpack_q3_k_low2(vptr, g + 4, mask_03);
        v_sum_lo = Q6_Vw_vrmpyacc_VwVubVb(v_sum_lo, v_W_lo, v_act[g]);
        v_sum_hi = Q6_Vw_vrmpyacc_VwVubVb(v_sum_hi, v_W_hi, v_act[g + 4]);
    }

    फर्काऊ Q6_Vw_vadd_VwVw(v_sum_lo, v_sum_hi);
}

स्थिर अन्तःस्थ शून्य accum_q2_0_32x2(
    अचर HVX_Vector * restrict vptr,
    अचर HVX_Vector * restrict v_act0,
    अचर HVX_Vector * restrict v_act1,
    HVX_Vector * v_sum0,
    HVX_Vector * v_sum1
) {
    HVX_Vector v_sum0_lo = Q6_V_vzero();
    HVX_Vector v_sum0_hi = Q6_V_vzero();
    HVX_Vector v_sum1_lo = Q6_V_vzero();
    HVX_Vector v_sum1_hi = Q6_V_vzero();
    HVX_Vector mask_03   = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    लागि (पूर्णाङ्क g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q3_k_low2(vptr, g,     mask_03);
        HVX_Vector v_W_hi = unpack_q3_k_low2(vptr, g + 4, mask_03);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVubVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVubVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVubVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVubVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sum0 = Q6_Vw_vadd_VwVw(v_sum0_lo, v_sum0_hi);
    *v_sum1 = Q6_Vw_vadd_VwVw(v_sum1_lo, v_sum1_hi);
}

// Q2_K: fp16 d (*v_d) and -dmin (*v_m) of the 32 rows of a super-block (8 tiles, VTCM stride 384), each in both vector halves
स्थिर अन्तःस्थ शून्य q2_k_load_dmin(अचर uint8_t * restrict sb, HVX_Vector * restrict v_d, HVX_Vector * restrict v_m) {
    HVX_Vector v_dd = Q6_V_vzero();  // lanes 0..31: d, lanes 32..63: -dmin

    #pragma unroll
    लागि (पूर्णाङ्क j = 0; j < 8; j++) {
        HVX_Vector v = hvx_vmem(sb + j * 384 + 256);  // the slice is at bytes 64..79
        v_dd = Q6_V_vmux_QVV(Q6_Q_vsetq_R(16 * j), v_dd, Q6_V_vror_VR(v, (64 - 16 * j) & 127));
    }

    HVX_VectorPred q_lo = Q6_Q_vsetq_R(64);
    HVX_Vector     v_r  = Q6_V_vror_VR(v_dd, 64);
    *v_d = Q6_V_vmux_QVV(q_lo, v_dd, v_r);
    *v_m = Q6_V_vmux_QVV(q_lo, v_r, v_dd);
}

// Q2_K: fp16 D = d * scale and M = -dmin * min of a tile from its scale bytes, in the lanes of the Q6_K scales
स्थिर अन्तःस्थ शून्य q2_k_unpack_scales(अचर uint8_t * restrict tile, HVX_Vector v_d, HVX_Vector v_m,
                                      HVX_Vector * restrict v_D, HVX_Vector * restrict v_M) {
    HVX_Vector v_sc = hvx_vmem(tile + 256);
    HVX_Vector v_l  = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(Q6_V_vand_VV(v_sc, Q6_Vb_vsplat_R(0x0F))));
    HVX_Vector v_mn = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(Q6_Vub_vlsr_VubR(v_sc, 4)));
    // qf32 product: rounds like GGML_FP32_TO_FP16(d * scale) (qf16 is off by 1 ulp for many values on v73)
    *v_D = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(v_l), v_d));
    *v_M = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(v_mn), v_m));
}

// Q2_K scale and min nibbles of a tile as halfword pairs: word r = sub-block 1 << 16 | sub-block 0 of row r
स्थिर अन्तःस्थ शून्य q2_k_scale_pairs(अचर uint8_t * restrict tile, HVX_Vector * restrict v_sc2, HVX_Vector * restrict v_mn2) {
    HVX_Vector v_h = Q6_Vh_vshuff_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmem(tile + 256))));
    *v_sc2 = Q6_V_vand_VV(v_h, Q6_Vh_vsplat_R(0x000F));
    *v_mn2 = Q6_Vuh_vlsr_VuhR(v_h, 4);
}

// Q2_K with q8_1_s16 activations, one tile: isum += scale * (q . a), msum += min * sum(a), per row, in int32.
// |sum of 16 q * a| <= 3 * 128 * 16 fits a halfword, so both sub-blocks go through one vdmpy (no saturation)
स्थिर अन्तःस्थ शून्य q2_k_int_accum(HVX_VectorPair v_sums, HVX_Vector v_sc2, HVX_Vector v_mn2, HVX_Vector v_bsum,
                                  HVX_Vector * restrict v_isum, HVX_Vector * restrict v_msum) {
    HVX_Vector v_s2 = Q6_Vh_vshuffe_VhVh(Q6_V_hi_W(v_sums), Q6_V_lo_W(v_sums));
    *v_isum = Q6_Vw_vadd_VwVw(*v_isum, Q6_Vw_vdmpy_VhVh_sat(v_s2, v_sc2));
    *v_msum = Q6_Vw_vadd_VwVw(*v_msum, Q6_Vw_vdmpy_VhVh_sat(v_mn2, v_bsum));
}

// Q2_K super-block result: d_a * (d * isum - dmin * msum), with v_d / v_m the fp16 d / -dmin of q2_k_load_dmin
स्थिर अन्तःस्थ HVX_Vector q2_k_int_scale(HVX_Vector v_isum, HVX_Vector v_msum, HVX_Vector v_d, HVX_Vector v_m, HVX_Vector v_act8) {
    HVX_Vector v_d32  = Q6_V_lo_W(hvx_vec_f16_to_f32(v_d));
    HVX_Vector v_m32  = Q6_V_lo_W(hvx_vec_f16_to_f32(v_m));
    HVX_Vector v_da32 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_act8));
    HVX_Vector v_t = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(v_d32, Q6_Vsf_equals_Vw(v_isum)),
                                         hvx_vec_mul_f32_f32(v_m32, Q6_Vsf_equals_Vw(v_msum)));
    फर्काऊ hvx_vec_mul_f32_f32(v_t, v_da32);
}

स्थिर शून्य tiled_vec_dot_q1_0_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_qf  = Q6_V_vzero();
    HVX_Vector v_neg_one = Q6_Vb_vsplat_R(-1);
    HVX_Vector v_pos_one = Q6_Vb_vsplat_R(1);

    अचर uint32_t n_k_tiles = n / HTP_TENSOR_TILE_K_Q1_0;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * htp_tensor_get_tile_padded_size(HTP_TYPE_Q1_0));
        HVX_Vector v_tile_qf = Q6_V_vzero();

        #pragma unroll
        लागि (पूर्णाङ्क quarter = 0; quarter < 4; quarter++) {
            अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + (4 * kt + quarter) * HTP_MM_ACT_TILE_SIZE_Q8_0);
            HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(accum_q1_0_32x1(vptr[quarter], v_act, v_neg_one, v_pos_one));
            HVX_Vector v_da_sf  = Q6_V_lo_W(hvx_vec_f16_to_f32_shuff(v_act[8]));
            HVX_Vector v_q_qf   = Q6_Vqf32_vmpy_VsfVsf(v_sum_sf, v_da_sf);
            v_tile_qf = (quarter == 0) ? v_q_qf : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf, v_q_qf);
        }

        HVX_Vector v_dw_sf = Q6_V_lo_W(hvx_vec_f16_to_f32(vptr[4]));
        v_sum_qf = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum_qf, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf), v_dw_sf));
    }

    HVX_Vector v_sum_float = Q6_Vsf_equals_Vqf32(v_sum_qf);
    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_q1_0_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_qf_c0 = Q6_V_vzero();
    HVX_Vector v_sum_qf_c1 = Q6_V_vzero();
    HVX_Vector v_neg_one   = Q6_Vb_vsplat_R(-1);
    HVX_Vector v_pos_one   = Q6_Vb_vsplat_R(1);

    अचर uint32_t n_k_tiles = n / HTP_TENSOR_TILE_K_Q1_0;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * htp_tensor_get_tile_padded_size(HTP_TYPE_Q1_0));
        HVX_Vector v_tile_qf_c0 = Q6_V_vzero();
        HVX_Vector v_tile_qf_c1 = Q6_V_vzero();

        लागि (पूर्णाङ्क quarter = 0; quarter < 4; quarter++) {
            अचर uint32_t act_tile = 4 * kt + quarter;
            अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + act_tile * HTP_MM_ACT_TILE_SIZE_Q8_0);
            अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + act_tile * HTP_MM_ACT_TILE_SIZE_Q8_0);
            HVX_VectorPair v_sums = accum_q1_0_32x2(vptr[quarter], v_act0, v_act1, v_neg_one, v_pos_one);
            HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums));
            HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums));
            HVX_Vector v_da_sf_c0  = Q6_V_lo_W(hvx_vec_f16_to_f32_shuff(v_act0[8]));
            HVX_Vector v_da_sf_c1  = Q6_V_lo_W(hvx_vec_f16_to_f32_shuff(v_act1[8]));

            v_tile_qf_c0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c0, Q6_Vqf32_vmpy_VsfVsf(v_sum_sf_c0, v_da_sf_c0));
            v_tile_qf_c1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c1, Q6_Vqf32_vmpy_VsfVsf(v_sum_sf_c1, v_da_sf_c1));
        }

        HVX_Vector v_dw_sf = Q6_V_lo_W(hvx_vec_f16_to_f32(vptr[4]));
        v_sum_qf_c0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum_qf_c0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf_c0), v_dw_sf));
        v_sum_qf_c1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum_qf_c1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v_tile_qf_c1), v_dw_sf));
    }

    HVX_Vector v_sum_float_c0 = Q6_Vsf_equals_Vqf32(v_sum_qf_c0);
    HVX_Vector v_sum_float_c1 = Q6_Vsf_equals_Vqf32(v_sum_qf_c1);
    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q4_0_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 640);
        अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, i8);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_q4_0_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 640);
        अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + kt * 1152);
        अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, i8);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q4_1_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 640);
        अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + kt * 1280);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_sum_a   = v_act[9];

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);

        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_q4_1_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 640);
        अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + kt * 1280);
        अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + kt * 1280);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_sum_a_c0   = v_act0[9];
        HVX_Vector v_scale_a_c1 = v_act1[8];
        HVX_Vector v_sum_a_c1   = v_act1[9];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c0);
        HVX_Vector v_offset_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c1);
        HVX_Vector v_offset_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c1);

        HVX_Vector v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c0 = hvx_vec_add_f32_f32(v_scaled_dot_c0, v_offset_comb_c0);

        HVX_Vector v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_add_f32_f32(v_scaled_dot_c1, v_offset_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q8_0_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 1152);
        अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_q8_0_32x1(vptr, v_act);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[8];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_q8_0_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 1152);
        अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + kt * 1152);
        अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q8_0_32x2(vptr, v_act0, v_act1);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[8];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q5_k_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 768);
        अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + kt * 1280);

        HVX_Vector v_sum = accum_5bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_sum_a   = v_act[9];

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);

        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_q5_k_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 768);
        अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + kt * 1280);
        अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + kt * 1280);

        HVX_VectorPair v_sums = accum_5bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_sum_a_c0   = v_act0[9];
        HVX_Vector v_scale_a_c1 = v_act1[8];
        HVX_Vector v_sum_a_c1   = v_act1[9];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c0);
        HVX_Vector v_offset_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c1);
        HVX_Vector v_offset_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c1);

        HVX_Vector v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c0 = hvx_vec_add_f32_f32(v_scaled_dot_c0, v_offset_comb_c0);

        HVX_Vector v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_add_f32_f32(v_scaled_dot_c1, v_offset_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q6_k_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 896);
        अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q6_k_32x1(vptr, v_act, i32);
        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, scale_q6_k_32x1(v_sums, vptr[6], v_act[8]));
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_q6_k_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 896);
        अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + kt * 1152);
        अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums0, v_sums1;
        accum_q6_k_32x2(vptr, v_act0, v_act1, i32, &v_sums0, &v_sums1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, scale_q6_k_32x1(v_sums0, vptr[6], v_act0[8]));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, scale_q6_k_32x1(v_sums1, vptr[6], v_act1[8]));
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q3_k_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 512);
        अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q3_k_32x1(vptr, v_act);
        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, scale_q6_k_32x1(v_sums, vptr[3], v_act[8]));
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_q3_k_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 512);
        अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + kt * 1152);
        अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums0, v_sums1;
        accum_q3_k_32x2(vptr, v_act0, v_act1, &v_sums0, &v_sums1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, scale_q6_k_32x1(v_sums0, vptr[3], v_act0[8]));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, scale_q6_k_32x1(v_sums1, vptr[3], v_act1[8]));
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_q2_k_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kb = 0; kb < n_k_tiles; kb += 8) {
        HVX_Vector v_d, v_m;
        q2_k_load_dmin(tile_ptr + kb * 384, &v_d, &v_m);

        HVX_Vector v_isum = Q6_V_vzero();
        HVX_Vector v_msum = Q6_V_vzero();
        लागि (uint32_t kt = kb; kt < kb + 8; kt++) {
            अचर uint8_t * restrict tile = tile_ptr + kt * 384;
            अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + kt * 1280);

            HVX_Vector v_sc2, v_mn2;
            q2_k_scale_pairs(tile, &v_sc2, &v_mn2);
            q2_k_int_accum(accum_q2_k_32x1((अचर HVX_Vector *) tile, v_act), v_sc2, v_mn2, v_act[9], &v_isum, &v_msum);
        }
        अचर HVX_Vector * restrict v_act_sb = (अचर HVX_Vector *) (y_q + kb * 1280);
        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, q2_k_int_scale(v_isum, v_msum, v_d, v_m, v_act_sb[8]));
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_q2_k_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kb = 0; kb < n_k_tiles; kb += 8) {
        HVX_Vector v_d, v_m;
        q2_k_load_dmin(tile_ptr + kb * 384, &v_d, &v_m);

        HVX_Vector v_isum0 = Q6_V_vzero();
        HVX_Vector v_msum0 = Q6_V_vzero();
        HVX_Vector v_isum1 = Q6_V_vzero();
        HVX_Vector v_msum1 = Q6_V_vzero();
        लागि (uint32_t kt = kb; kt < kb + 8; kt++) {
            अचर uint8_t * restrict tile = tile_ptr + kt * 384;
            अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + kt * 1280);
            अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + kt * 1280);

            HVX_Vector v_sc2, v_mn2;
            q2_k_scale_pairs(tile, &v_sc2, &v_mn2);

            HVX_VectorPair v_sums0, v_sums1;
            accum_q2_k_32x2((अचर HVX_Vector *) tile, v_act0, v_act1, &v_sums0, &v_sums1);
            q2_k_int_accum(v_sums0, v_sc2, v_mn2, v_act0[9], &v_isum0, &v_msum0);
            q2_k_int_accum(v_sums1, v_sc2, v_mn2, v_act1[9], &v_isum1, &v_msum1);
        }
        अचर HVX_Vector * restrict v_act0_sb = (अचर HVX_Vector *) (y0_q + kb * 1280);
        अचर HVX_Vector * restrict v_act1_sb = (अचर HVX_Vector *) (y1_q + kb * 1280);
        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, q2_k_int_scale(v_isum0, v_msum0, v_d, v_m, v_act0_sb[8]));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, q2_k_int_scale(v_isum1, v_msum1, v_d, v_m, v_act1_sb[8]));
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

// One Q2_0 weight tile covers 64 k: act tiles 2 * kt and 2 * kt + 1, each 32 k wide (Q8_1)
स्थिर शून्य tiled_vec_dot_q2_0_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector v_one_f16   = hvx_vec_splat_f16(1.0);

    uint32_t n_weight_tiles = n / HTP_MM_WEIGHT_TILE_K_Q2_0;
    लागि (uint32_t kt = 0; kt < n_weight_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q2_0);
        HVX_Vector v_tile_qf = Q6_V_vzero();

        #pragma unroll
        लागि (पूर्णाङ्क half = 0; half < 2; half++) {
            अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + (2 * kt + half) * HTP_MM_ACT_TILE_SIZE_Q8_1);

            HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(accum_q2_0_32x1(vptr + 2 * half, v_act));
            HVX_Vector v_da_sf  = Q6_V_lo_W(hvx_vec_f16_to_f32_shuff(v_act[8]));
            HVX_Vector v_sa_qf  = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act[9], v_one_f16));
            HVX_Vector v_h_qf   = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_sum_sf, v_da_sf), v_sa_qf);
            v_tile_qf = (half == 0) ? v_h_qf : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf, v_h_qf);
        }

        HVX_Vector v_dw_sf = Q6_V_lo_W(hvx_vec_f16_to_f32(vptr[4]));
        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf), v_dw_sf));
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_q2_0_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector v_one_f16      = hvx_vec_splat_f16(1.0);

    uint32_t n_weight_tiles = n / HTP_MM_WEIGHT_TILE_K_Q2_0;
    लागि (uint32_t kt = 0; kt < n_weight_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q2_0);
        HVX_Vector v_tile_qf_c0 = Q6_V_vzero();
        HVX_Vector v_tile_qf_c1 = Q6_V_vzero();

        #pragma unroll
        लागि (पूर्णाङ्क half = 0; half < 2; half++) {
            अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + (2 * kt + half) * HTP_MM_ACT_TILE_SIZE_Q8_1);
            अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + (2 * kt + half) * HTP_MM_ACT_TILE_SIZE_Q8_1);

            HVX_Vector v_sum_c0, v_sum_c1;
            accum_q2_0_32x2(vptr + 2 * half, v_act0, v_act1, &v_sum_c0, &v_sum_c1);

            HVX_Vector v_da_sf_c0 = Q6_V_lo_W(hvx_vec_f16_to_f32_shuff(v_act0[8]));
            HVX_Vector v_da_sf_c1 = Q6_V_lo_W(hvx_vec_f16_to_f32_shuff(v_act1[8]));
            HVX_Vector v_sa_qf_c0 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act0[9], v_one_f16));
            HVX_Vector v_sa_qf_c1 = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(v_act1[9], v_one_f16));

            HVX_Vector v_h_qf_c0 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(v_sum_c0), v_da_sf_c0), v_sa_qf_c0);
            HVX_Vector v_h_qf_c1 = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(v_sum_c1), v_da_sf_c1), v_sa_qf_c1);

            v_tile_qf_c0 = (half == 0) ? v_h_qf_c0 : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c0, v_h_qf_c0);
            v_tile_qf_c1 = (half == 0) ? v_h_qf_c1 : Q6_Vqf32_vadd_Vqf32Vqf32(v_tile_qf_c1, v_h_qf_c1);
        }

        HVX_Vector v_dw_sf = Q6_V_lo_W(hvx_vec_f16_to_f32(vptr[4]));
        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf_c0), v_dw_sf));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vqf32(v_tile_qf_c1), v_dw_sf));
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_iq4nl_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(अचर HVX_Vector *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 640);
        अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_iq4nl_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(अचर HVX_Vector *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 640);
        अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + kt * 1152);
        अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_mxfp4_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(अचर HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(अचर HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 640);
        अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_f16 = v_act[8];
        HVX_VectorPair p_scale_a_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_f16);
        HVX_Vector v_scale_a = Q6_V_lo_W(p_scale_a_f32);

        HVX_Vector v_scale_comb = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    v_sum_float = hvx_vec_mul_f32_f32(v_sum_float, hvx_vec_splat_f32(0.5f));

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum_float);
    }
}

स्थिर शून्य tiled_vec_dot_mxfp4_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर uint8_t * restrict tile_ptr = vx;
    अचर uint8_t * restrict y0_q = vy0;
    अचर uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(अचर HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(अचर HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    लागि (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        अचर HVX_Vector * restrict vptr = (अचर HVX_Vector *) (tile_ptr + kt * 640);
        अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) (y0_q + kt * 1152);
        अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_c0_f16 = v_act0[8];
        HVX_Vector v_scale_a_c1_f16 = v_act1[8];

        HVX_VectorPair p_scale_a_c0_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c0_f16);
        HVX_VectorPair p_scale_a_c1_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c1_f16);

        HVX_Vector v_scale_a_c0 = Q6_V_lo_W(p_scale_a_c0_f32);
        HVX_Vector v_scale_a_c1 = Q6_V_lo_W(p_scale_a_c1_f32);

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    v_sum_float_c0 = hvx_vec_mul_f32_f32(v_sum_float_c0, hvx_vec_splat_f32(0.5f));
    v_sum_float_c1 = hvx_vec_mul_f32_f32(v_sum_float_c1, hvx_vec_splat_f32(0.5f));

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum_float_c0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum_float_c1);
    }
}

स्थिर शून्य tiled_vec_dot_f16_32x1(अचर uint32_t n, भिन्न * restrict s, अचर शून्य * restrict vx, अचर शून्य * restrict vy, uint32_t valid_rows, अचर भिन्न * restrict sz) {
    अचर HVX_Vector * restrict vptr  = (अचर HVX_Vector *) vx;
    अचर HVX_Vector * restrict v_act = (अचर HVX_Vector *) vy;
    अचर uint32_t n_vecs = ((n + 31) / 32) * 16;

    HVX_VectorPair acc = Q6_W_vzero();

    #pragma unroll(4)
    लागि (uint32_t i = 0; i < n_vecs; i++) {
        acc = hvx_vec_mpyacc_f32_f16(acc, vptr[i], v_act[i]);
    }

    HVX_Vector v_sum = hvx_vec_add_f32_f32(Q6_V_lo_W(acc), Q6_V_hi_W(acc));

    यदि (sz) {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum, hvx_vmemu(sz)));
    } अन्यथा {
        hvx_vec_store_u(s, valid_rows * परिमाण(भिन्न), v_sum);
    }
}

स्थिर शून्य tiled_vec_dot_f16_32x2(अचर uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1, अचर शून्य * restrict vx, अचर शून्य * restrict vy0, अचर शून्य * restrict vy1, uint32_t valid_rows, अचर भिन्न * restrict sz0, अचर भिन्न * restrict sz1) {
    अचर HVX_Vector * restrict vptr   = (अचर HVX_Vector *) vx;
    अचर HVX_Vector * restrict v_act0 = (अचर HVX_Vector *) vy0;
    अचर HVX_Vector * restrict v_act1 = (अचर HVX_Vector *) vy1;
    अचर uint32_t n_vecs = ((n + 31) / 32) * 16;

    HVX_VectorPair acc0 = Q6_W_vzero();
    HVX_VectorPair acc1 = Q6_W_vzero();

    #pragma unroll(4)
    लागि (uint32_t i = 0; i < n_vecs; i++) {
        HVX_Vector v_w = vptr[i];
        acc0 = hvx_vec_mpyacc_f32_f16(acc0, v_w, v_act0[i]);
        acc1 = hvx_vec_mpyacc_f32_f16(acc1, v_w, v_act1[i]);
    }

    HVX_Vector v_sum0 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc0), Q6_V_hi_W(acc0));
    HVX_Vector v_sum1 = hvx_vec_add_f32_f32(Q6_V_lo_W(acc1), Q6_V_hi_W(acc1));

    यदि (sz0) {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum0, hvx_vmemu(sz0)));
    } अन्यथा {
        hvx_vec_store_u(s0, valid_rows * परिमाण(भिन्न), v_sum0);
    }
    यदि (sz1) {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), hvx_vec_add_f32_f32(v_sum1, hvx_vmemu(sz1)));
    } अन्यथा {
        hvx_vec_store_u(s1, valid_rows * परिमाण(भिन्न), v_sum1);
    }
}


स्थिर अन्तःस्थ शून्य quantize_f32_q8_0_tiled_kernel(
    अचर uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size
) {
    (शून्य) tmp_data;
    लागि (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_0_tiled((भिन्न *) src_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_1_tiled_kernel(
    अचर uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size
) {
    (शून्य) tmp_data;
    लागि (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_1_tiled((भिन्न *) src_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_1_s16_tiled_kernel(
    अचर uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size
) {
    (शून्य) tmp_data;
    लागि (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_1_s16_tiled((भिन्न *) src_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_f16_tiled_kernel(
    अचर uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size
) {
    (शून्य) tmp_data;
    लागि (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_f16_tiled((अचर भिन्न *) src_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f16_f16_tiled_kernel(
    अचर uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size
) {
    (शून्य) tmp_data;
    लागि (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f16_f16_tiled((अचर __fp16 *) src_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_0_tiled_block_kernel(
    अचर भिन्न * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_data;
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (ne0 + qk - 1) / qk;

    लागि (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        अचर भिन्न * restrict src_ptr = (अचर भिन्न *) ((अचर uint8_t *) src + r * src_row_size + c * qk * परिमाण(भिन्न));
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1152;

        quantize_block_f32_q8_0_tiled((भिन्न *) src_ptr, dst_ptr);

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_1_tiled_block_kernel(
    अचर भिन्न * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_data;
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (ne0 + qk - 1) / qk;

    लागि (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        अचर भिन्न * restrict src_ptr = (अचर भिन्न *) ((अचर uint8_t *) src + r * src_row_size + c * qk * परिमाण(भिन्न));
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1280;

        quantize_block_f32_q8_1_tiled((भिन्न *) src_ptr, dst_ptr);

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_q8_1_s16_tiled_block_kernel(
    अचर भिन्न * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_data;
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (ne0 + qk - 1) / qk;

    लागि (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        अचर भिन्न * restrict src_ptr = (अचर भिन्न *) ((अचर uint8_t *) src + r * src_row_size + c * qk * परिमाण(भिन्न));
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1280;

        quantize_block_f32_q8_1_s16_tiled((भिन्न *) src_ptr, dst_ptr, src_ptr - (c & 1) * qk);  // super-block of 256 k

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_f16_tiled_block_kernel(
    अचर भिन्न * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_data;
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (ne0 + qk - 1) / qk;

    लागि (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        अचर भिन्न * restrict src_ptr = (अचर भिन्न *) ((अचर uint8_t *) src + r * src_row_size + c * qk * परिमाण(भिन्न));
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 2048;

        quantize_block_f32_f16_tiled(src_ptr, dst_ptr);

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}

स्थिर अन्तःस्थ शून्य quantize_f16_f16_tiled_block_kernel(
    अचर __fp16 * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    माप_प्रकार src_row_size,
    माप_प्रकार dst_row_size,
    uint32_t r,
    uint32_t c
) {
    (शून्य) tmp_data;
    अचर uint32_t qk = QK_Q8_0_TILED;
    अचर uint32_t nb = (ne0 + qk - 1) / qk;

    लागि (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        अचर __fp16 * restrict src_ptr = (अचर __fp16 *) ((अचर uint8_t *) src + r * src_row_size + c * qk * परिमाण(__fp16));
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 2048;

        quantize_block_f16_f16_tiled(src_ptr, dst_ptr);

        c++;
        यदि (c == nb) {
            c = 0;
            r++;
        }
    }
}
