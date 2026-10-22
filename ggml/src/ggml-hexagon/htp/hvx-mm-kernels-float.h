#اگر_غیر_تعریف_شدہ HVX_MM_KERNELS_FLOAT_H
#تعریف_کرو HVX_MM_KERNELS_FLOAT_H

#شامل "hvx-utils.h"
#شامل "htp-tensor.h"

// Float activation copy/quantization kernels (DDR -> VTCM)

ساکن اندرونی خالی quantize_f32_f32_kernel(
    ثابت uint8_t * محدود src_ڈیٹا,
    uint8_t * محدود dst_ڈیٹا,
    uint8_t * محدود tmp_ڈیٹا,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (خالی) tmp_ڈیٹا;
    ثابت size_t src_صف_سائز = ne0 * حجم(اعشاری);
    برائے (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_ڈیٹا, src_صف_سائز, src_stride, 2);
        hvx_copy_f32_au(dst_ڈیٹا, src_ڈیٹا, ne0);

        dst_ڈیٹا += dst_stride;
        src_ڈیٹا += src_stride;
    }
}

ساکن اندرونی خالی quantize_f32_f16_kernel(
    ثابت uint8_t * محدود src_ڈیٹا,
    uint8_t * محدود dst_ڈیٹا,
    uint8_t * محدود tmp_ڈیٹا,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (خالی) tmp_ڈیٹا;
    ثابت size_t src_صف_سائز = ne0 * حجم(اعشاری);
    برائے (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_ڈیٹا, src_صف_سائز, src_stride, 2);
        hvx_copy_f16_f32_au(dst_ڈیٹا, src_ڈیٹا, ne0);

        dst_ڈیٹا += dst_stride;
        src_ڈیٹا += src_stride;
    }
}

ساکن اندرونی خالی quantize_f16_f16_kernel(
    ثابت uint8_t * محدود src_ڈیٹا,
    uint8_t * محدود dst_ڈیٹا,
    uint8_t * محدود tmp_ڈیٹا,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (خالی) tmp_ڈیٹا;
    ثابت size_t src_صف_سائز = ne0 * حجم(اعشاری);
    برائے (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_ڈیٹا, src_صف_سائز, src_stride, 2);
        hvx_copy_f16_au(dst_ڈیٹا, src_ڈیٹا, ne0);

        dst_ڈیٹا += dst_stride;
        src_ڈیٹا += src_stride;
    }
}

// Float dot product kernels (HVX)

#اگر __HVX_ARCH__ < 79
#تعریف_کرو HVX_OP_جوڑو_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#تعریف_کرو HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ورنہ
#تعریف_کرو HVX_OP_جوڑو_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#تعریف_کرو HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#اگر_ختم

ساکن اندرونی خالی flat_vec_dot_f32_f32_aa_1x1(ثابت uint32_t n, اعشاری * محدود s, ثابت خالی * محدود vx, ثابت خالی * محدود vy) {
    ثابت HVX_سمتیہ * محدود x = (ثابت HVX_سمتیہ *) vx;
    ثابت HVX_سمتیہ * محدود y = (ثابت HVX_سمتیہ *) vy;

    uint32_t nvec = n / VLEN_FP32; // num full fp32 hvx vectors
    uint32_t nloe = n % VLEN_FP32; // leftover elements

    HVX_سمتیہ rsum = Q6_V_vzero();

    uint32_t i = 0;

    #ہدایت unroll(4)
    برائے (i = 0; i < nvec; i++) {
        HVX_سمتیہ prod = HVX_OP_MUL_F32(x[i], y[i]);
        rsum = HVX_OP_جوڑو_F32(rsum, prod);
    }

    اگر (nloe) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ x_sf = Q6_V_vand_QV(bmask, x[i]);
        HVX_سمتیہ y_sf = Q6_V_vand_QV(bmask, y[i]);
        HVX_سمتیہ prod = HVX_OP_MUL_F32(x_sf, y_sf);
        rsum = HVX_OP_جوڑو_F32(rsum, prod);
    }

    *s = hvx_vec_get_f32(hvx_vec_reduce_مجموع_f32(rsum));
}

ساکن اندرونی خالی flat_vec_dot_f32_f32_aa_2x1(ثابت uint32_t n, اعشاری * محدود s0,
                                ثابت خالی * محدود vx0, ثابت خالی * محدود vx1,
                                ثابت خالی * محدود vy0) {
    ثابت HVX_سمتیہ * محدود x0 = (ثابت HVX_سمتیہ *) vx0;
    ثابت HVX_سمتیہ * محدود x1 = (ثابت HVX_سمتیہ *) vx1;
    ثابت HVX_سمتیہ * محدود y  = (ثابت HVX_سمتیہ *) vy0;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_سمتیہ rsum0 = Q6_V_vzero();
    HVX_سمتیہ rsum1 = Q6_V_vzero();

    uint32_t i = 0;

    #ہدایت unroll(2)
    برائے (i = 0; i < nvec; i++) {
        HVX_سمتیہ y_sf = y[i];
        HVX_سمتیہ prod0 = HVX_OP_MUL_F32(x0[i], y_sf);
        HVX_سمتیہ prod1 = HVX_OP_MUL_F32(x1[i], y_sf);
        rsum0 = HVX_OP_جوڑو_F32(rsum0, prod0);
        rsum1 = HVX_OP_جوڑو_F32(rsum1, prod1);
    }

    اگر (nloe) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_سمتیہ y_sf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_سمتیہ x0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_سمتیہ x1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_سمتیہ prod0 = HVX_OP_MUL_F32(x0_sf, y_sf);
        HVX_سمتیہ prod1 = HVX_OP_MUL_F32(x1_sf, y_sf);
        rsum0 = HVX_OP_جوڑو_F32(rsum0, prod0);
        rsum1 = HVX_OP_جوڑو_F32(rsum1, prod1);
    }

    HVX_سمتیہ rsum = hvx_vec_reduce_مجموع_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ساکن اندرونی خالی flat_vec_dot_f32_f32_aa_2x2(ثابت uint32_t n, اعشاری * محدود s0, اعشاری * محدود s1,
                                ثابت خالی * محدود vx0, ثابت خالی * محدود vx1,
                                ثابت خالی * محدود vy0, ثابت خالی * محدود vy1) {
    ثابت HVX_سمتیہ * محدود x0 = (ثابت HVX_سمتیہ *) vx0;
    ثابت HVX_سمتیہ * محدود x1 = (ثابت HVX_سمتیہ *) vx1;
    ثابت HVX_سمتیہ * محدود y0 = (ثابت HVX_سمتیہ *) vy0;
    ثابت HVX_سمتیہ * محدود y1 = (ثابت HVX_سمتیہ *) vy1;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_سمتیہ r0_c0_مجموع = Q6_V_vzero();
    HVX_سمتیہ r0_c1_مجموع = Q6_V_vzero();
    HVX_سمتیہ r1_c0_مجموع = Q6_V_vzero();
    HVX_سمتیہ r1_c1_مجموع = Q6_V_vzero();

    uint32_t i = 0;

    #ہدایت unroll(2)
    برائے (i = 0; i < nvec; i++) {
        HVX_سمتیہ r0_sf = x0[i];
        HVX_سمتیہ r1_sf = x1[i];
        HVX_سمتیہ c0_sf = y0[i];
        HVX_سمتیہ c1_sf = y1[i];

        r0_c0_مجموع = HVX_OP_جوڑو_F32(r0_c0_مجموع, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_مجموع = HVX_OP_جوڑو_F32(r0_c1_مجموع, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_مجموع = HVX_OP_جوڑو_F32(r1_c0_مجموع, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_مجموع = HVX_OP_جوڑو_F32(r1_c1_مجموع, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    اگر (nloe) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);

        HVX_سمتیہ r0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_سمتیہ r1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_سمتیہ c0_sf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_سمتیہ c1_sf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_مجموع = HVX_OP_جوڑو_F32(r0_c0_مجموع, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_مجموع = HVX_OP_جوڑو_F32(r0_c1_مجموع, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_مجموع = HVX_OP_جوڑو_F32(r1_c0_مجموع, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_مجموع = HVX_OP_جوڑو_F32(r1_c1_مجموع, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    // Reduce and store results
    HVX_سمتیہ r0_r1_c0_مجموع = hvx_vec_reduce_مجموع_f32x2(r0_c0_مجموع, r1_c0_مجموع);
    HVX_سمتیہ r0_r1_c1_مجموع = hvx_vec_reduce_مجموع_f32x2(r0_c1_مجموع, r1_c1_مجموع);

    hvx_vec_store_u(s0, 8, r0_r1_c0_مجموع);
    hvx_vec_store_u(s1, 8, r0_r1_c1_مجموع);
}

ساکن اندرونی خالی flat_vec_dot_bf16_f32_aa_1x1(ثابت uint32_t n, اعشاری * محدود s, ثابت خالی * محدود vx, ثابت خالی * محدود vy) {
    ثابت HVX_سمتیہ * محدود x = (ثابت HVX_سمتیہ *) vx;
    ثابت HVX_سمتیہ * محدود y = (ثابت HVX_سمتیہ *) vy;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_سمتیہ rsum0 = Q6_V_vzero();
    HVX_سمتیہ rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #ہدایت unroll(2)
    برائے (i = 0; i < nvec; i++) {
        HVX_سمتیہ v_bf16 = x[i];
        HVX_سمتیہ_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_سمتیہ w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_سمتیہ w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        rsum0 = HVX_OP_جوڑو_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y[2 * i + 0]));
        rsum1 = HVX_OP_جوڑو_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y[2 * i + 1]));
    }

    اگر (rem) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_سمتیہ v_bf16 = Q6_V_vand_QV(bmask, x[i]);
        HVX_سمتیہ_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_سمتیہ w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_سمتیہ w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        اگر (rem <= 32) {
            HVX_سمتیہ_Pred a_نقاب = Q6_Q_vsetq2_R(rem * 4);
            HVX_سمتیہ y0 = Q6_V_vand_QV(a_نقاب, y[2 * i + 0]);
            rsum0 = HVX_OP_جوڑو_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));
        } ورنہ {
            HVX_سمتیہ y0 = y[2 * i + 0];
            rsum0 = HVX_OP_جوڑو_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));

            HVX_سمتیہ_Pred a_نقاب = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_سمتیہ y1 = Q6_V_vand_QV(a_نقاب, y[2 * i + 1]);
            rsum1 = HVX_OP_جوڑو_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y1));
        }
    }

    HVX_سمتیہ rsum = HVX_OP_جوڑو_F32(rsum0, rsum1);
    *s = hvx_vec_get_f32(hvx_vec_reduce_مجموع_f32(rsum));
}

ساکن اندرونی خالی flat_vec_dot_bf16_f32_aa_2x1(ثابت uint32_t n, اعشاری * محدود s0,
                                ثابت خالی * محدود vx0, ثابت خالی * محدود vx1,
                                ثابت خالی * محدود vy0) {
    ثابت HVX_سمتیہ * محدود x0 = (ثابت HVX_سمتیہ *) vx0;
    ثابت HVX_سمتیہ * محدود x1 = (ثابت HVX_سمتیہ *) vx1;
    ثابت HVX_سمتیہ * محدود y  = (ثابت HVX_سمتیہ *) vy0;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_سمتیہ rsum0 = Q6_V_vzero();
    HVX_سمتیہ rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #ہدایت unroll(2)
    برائے (i = 0; i < nvec; i++) {
        HVX_سمتیہ_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_سمتیہ_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_سمتیہ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_سمتیہ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_سمتیہ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_سمتیہ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_سمتیہ y0 = y[2 * i + 0];
        HVX_سمتیہ y1 = y[2 * i + 1];

        rsum0 = HVX_OP_جوڑو_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
        rsum1 = HVX_OP_جوڑو_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        rsum0 = HVX_OP_جوڑو_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
        rsum1 = HVX_OP_جوڑو_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
    }

    اگر (rem) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_سمتیہ_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_سمتیہ_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_سمتیہ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_سمتیہ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_سمتیہ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_سمتیہ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        اگر (rem <= 32) {
            HVX_سمتیہ_Pred a_نقاب = Q6_Q_vsetq2_R(rem * 4);
            HVX_سمتیہ y0 = Q6_V_vand_QV(a_نقاب, y[2 * i + 0]);
            rsum0 = HVX_OP_جوڑو_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_جوڑو_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        } ورنہ {
            HVX_سمتیہ y0 = y[2 * i + 0];
            rsum0 = HVX_OP_جوڑو_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_جوڑو_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));

            HVX_سمتیہ_Pred a_نقاب = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_سمتیہ y1 = Q6_V_vand_QV(a_نقاب, y[2 * i + 1]);
            rsum0 = HVX_OP_جوڑو_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
            rsum1 = HVX_OP_جوڑو_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
        }
    }

    HVX_سمتیہ rsum = hvx_vec_reduce_مجموع_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ساکن اندرونی خالی flat_vec_dot_bf16_f32_aa_2x2(ثابت uint32_t n, اعشاری * محدود s0, اعشاری * محدود s1,
                                ثابت خالی * محدود vx0, ثابت خالی * محدود vx1,
                                ثابت خالی * محدود vy0, ثابت خالی * محدود vy1) {
    ثابت HVX_سمتیہ * محدود x0 = (ثابت HVX_سمتیہ *) vx0;
    ثابت HVX_سمتیہ * محدود x1 = (ثابت HVX_سمتیہ *) vx1;
    ثابت HVX_سمتیہ * محدود y0 = (ثابت HVX_سمتیہ *) vy0;
    ثابت HVX_سمتیہ * محدود y1 = (ثابت HVX_سمتیہ *) vy1;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_سمتیہ r0_c0_مجموع = Q6_V_vzero();
    HVX_سمتیہ r0_c1_مجموع = Q6_V_vzero();
    HVX_سمتیہ r1_c0_مجموع = Q6_V_vzero();
    HVX_سمتیہ r1_c1_مجموع = Q6_V_vzero();

    uint32_t i = 0;
    #ہدایت unroll(2)
    برائے (i = 0; i < nvec; i++) {
        HVX_سمتیہ_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_سمتیہ_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_سمتیہ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_سمتیہ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_سمتیہ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_سمتیہ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_سمتیہ c0_0 = y0[2 * i + 0];
        HVX_سمتیہ c0_1 = y0[2 * i + 1];
        HVX_سمتیہ c1_0 = y1[2 * i + 0];
        HVX_سمتیہ c1_1 = y1[2 * i + 1];

        r0_c0_مجموع = HVX_OP_جوڑو_F32(r0_c0_مجموع, HVX_OP_MUL_F32(w00_f32, c0_0));
        r0_c1_مجموع = HVX_OP_جوڑو_F32(r0_c1_مجموع, HVX_OP_MUL_F32(w00_f32, c1_0));
        r1_c0_مجموع = HVX_OP_جوڑو_F32(r1_c0_مجموع, HVX_OP_MUL_F32(w10_f32, c0_0));
        r1_c1_مجموع = HVX_OP_جوڑو_F32(r1_c1_مجموع, HVX_OP_MUL_F32(w10_f32, c1_0));

        r0_c0_مجموع = HVX_OP_جوڑو_F32(r0_c0_مجموع, HVX_OP_MUL_F32(w01_f32, c0_1));
        r0_c1_مجموع = HVX_OP_جوڑو_F32(r0_c1_مجموع, HVX_OP_MUL_F32(w01_f32, c1_1));
        r1_c0_مجموع = HVX_OP_جوڑو_F32(r1_c0_مجموع, HVX_OP_MUL_F32(w11_f32, c0_1));
        r1_c1_مجموع = HVX_OP_جوڑو_F32(r1_c1_مجموع, HVX_OP_MUL_F32(w11_f32, c1_1));
    }

    اگر (rem) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_سمتیہ_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_سمتیہ_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_سمتیہ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_سمتیہ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_سمتیہ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_سمتیہ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        اگر (rem <= 32) {
            HVX_سمتیہ_Pred a_نقاب = Q6_Q_vsetq2_R(rem * 4);
            HVX_سمتیہ c0_0 = Q6_V_vand_QV(a_نقاب, y0[2 * i + 0]);
            HVX_سمتیہ c1_0 = Q6_V_vand_QV(a_نقاب, y1[2 * i + 0]);
            r0_c0_مجموع = HVX_OP_جوڑو_F32(r0_c0_مجموع, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_مجموع = HVX_OP_جوڑو_F32(r0_c1_مجموع, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_مجموع = HVX_OP_جوڑو_F32(r1_c0_مجموع, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_مجموع = HVX_OP_جوڑو_F32(r1_c1_مجموع, HVX_OP_MUL_F32(w10_f32, c1_0));
        } ورنہ {
            HVX_سمتیہ c0_0 = y0[2 * i + 0];
            HVX_سمتیہ c1_0 = y1[2 * i + 0];
            r0_c0_مجموع = HVX_OP_جوڑو_F32(r0_c0_مجموع, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_مجموع = HVX_OP_جوڑو_F32(r0_c1_مجموع, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_مجموع = HVX_OP_جوڑو_F32(r1_c0_مجموع, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_مجموع = HVX_OP_جوڑو_F32(r1_c1_مجموع, HVX_OP_MUL_F32(w10_f32, c1_0));

            HVX_سمتیہ_Pred a_نقاب = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_سمتیہ c0_1 = Q6_V_vand_QV(a_نقاب, y0[2 * i + 1]);
            HVX_سمتیہ c1_1 = Q6_V_vand_QV(a_نقاب, y1[2 * i + 1]);
            r0_c0_مجموع = HVX_OP_جوڑو_F32(r0_c0_مجموع, HVX_OP_MUL_F32(w01_f32, c0_1));
            r0_c1_مجموع = HVX_OP_جوڑو_F32(r0_c1_مجموع, HVX_OP_MUL_F32(w01_f32, c1_1));
            r1_c0_مجموع = HVX_OP_جوڑو_F32(r1_c0_مجموع, HVX_OP_MUL_F32(w11_f32, c0_1));
            r1_c1_مجموع = HVX_OP_جوڑو_F32(r1_c1_مجموع, HVX_OP_MUL_F32(w11_f32, c1_1));
        }
    }

    HVX_سمتیہ r0_r1_c0_مجموع = hvx_vec_reduce_مجموع_f32x2(r0_c0_مجموع, r1_c0_مجموع);
    HVX_سمتیہ r0_r1_c1_مجموع = hvx_vec_reduce_مجموع_f32x2(r0_c1_مجموع, r1_c1_مجموع);

    hvx_vec_store_u(s0, 8, r0_r1_c0_مجموع);
    hvx_vec_store_u(s1, 8, r0_r1_c1_مجموع);
}

#تعریف_منسوخ HVX_OP_جوڑو_F32
#تعریف_منسوخ HVX_OP_MUL_F32

ساکن اندرونی خالی flat_vec_dot_f16_f16_aa_1x1(ثابت uint32_t n, اعشاری * محدود s, ثابت خالی * محدود vx, ثابت خالی * محدود vy) {
    ثابت HVX_سمتیہ * محدود x = (ثابت HVX_سمتیہ *) vx;
    ثابت HVX_سمتیہ * محدود y = (ثابت HVX_سمتیہ *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_سمتیہ_Pair rsum_p = Q6_W_vzero();

    uint32_t i = 0;

    #ہدایت unroll(4)
    برائے (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x[i], y[i]);
    }

    اگر (nloe) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_سمتیہ x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_سمتیہ y_hf = Q6_V_vand_QV(bmask, y[i]);
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_سمتیہ rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));
    hvx_vec_store_u(s, 4, hvx_vec_reduce_مجموع_f32(rsum));
}

ساکن اندرونی خالی flat_vec_dot_f16_f16_aa_2x1(ثابت uint32_t n, اعشاری * محدود s0,
                                ثابت خالی * محدود vx0, ثابت خالی * محدود vx1,
                                ثابت خالی * محدود vy0) {
    ثابت HVX_سمتیہ * محدود x0 = (ثابت HVX_سمتیہ *) vx0;
    ثابت HVX_سمتیہ * محدود x1 = (ثابت HVX_سمتیہ *) vx1;
    ثابت HVX_سمتیہ * محدود y  = (ثابت HVX_سمتیہ *) vy0;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_سمتیہ_Pair rsum0_p = Q6_W_vzero();
    HVX_سمتیہ_Pair rsum1_p = Q6_W_vzero();

    uint32_t i = 0;

    #ہدایت unroll(2)
    برائے (i = 0; i < nvec; i++) {
        HVX_سمتیہ y_hf = y[i];
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0[i], y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1[i], y_hf);
    }

    اگر (nloe) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_سمتیہ y_hf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_سمتیہ x0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_سمتیہ x1_hf = Q6_V_vand_QV(bmask, x1[i]);
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
    }

    HVX_سمتیہ rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p)));
    HVX_سمتیہ rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p)));
    HVX_سمتیہ rsum  = hvx_vec_reduce_مجموع_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ساکن اندرونی خالی flat_vec_dot_f16_f16_aa_2x2(ثابت uint32_t n, اعشاری * محدود s0, اعشاری * محدود s1,
                                ثابت خالی * محدود vx0, ثابت خالی * محدود vx1,
                                ثابت خالی * محدود vy0, ثابت خالی * محدود vy1) {
    ثابت HVX_سمتیہ * محدود x0 = (ثابت HVX_سمتیہ *) vx0;
    ثابت HVX_سمتیہ * محدود x1 = (ثابت HVX_سمتیہ *) vx1;
    ثابت HVX_سمتیہ * محدود y0 = (ثابت HVX_سمتیہ *) vy0;
    ثابت HVX_سمتیہ * محدود y1 = (ثابت HVX_سمتیہ *) vy1;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    // Row sums (sf) - 4 accumulators for 2x2 tile
    HVX_سمتیہ_Pair r0_c0_مجموع_p = Q6_W_vzero();
    HVX_سمتیہ_Pair r0_c1_مجموع_p = Q6_W_vzero();
    HVX_سمتیہ_Pair r1_c0_مجموع_p = Q6_W_vzero();
    HVX_سمتیہ_Pair r1_c1_مجموع_p = Q6_W_vzero();

    uint32_t i = 0;

    #ہدایت unroll(2)
    برائے (i = 0; i < nvec; i++) {
        HVX_سمتیہ r0_hf = x0[i];
        HVX_سمتیہ r1_hf = x1[i];
        HVX_سمتیہ c0_hf = y0[i];
        HVX_سمتیہ c1_hf = y1[i];

        // Compute 4 dot products: r0xc0, r0xc1, r1xc0, r1xc1
        r0_c0_مجموع_p = hvx_vec_mpyacc_f32_f16(r0_c0_مجموع_p, r0_hf, c0_hf);
        r0_c1_مجموع_p = hvx_vec_mpyacc_f32_f16(r0_c1_مجموع_p, r0_hf, c1_hf);
        r1_c0_مجموع_p = hvx_vec_mpyacc_f32_f16(r1_c0_مجموع_p, r1_hf, c0_hf);
        r1_c1_مجموع_p = hvx_vec_mpyacc_f32_f16(r1_c1_مجموع_p, r1_hf, c1_hf);
    }

    اگر (nloe) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);

        HVX_سمتیہ r0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_سمتیہ r1_hf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_سمتیہ c0_hf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_سمتیہ c1_hf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_مجموع_p = hvx_vec_mpyacc_f32_f16(r0_c0_مجموع_p, r0_hf, c0_hf);
        r0_c1_مجموع_p = hvx_vec_mpyacc_f32_f16(r0_c1_مجموع_p, r0_hf, c1_hf);
        r1_c0_مجموع_p = hvx_vec_mpyacc_f32_f16(r1_c0_مجموع_p, r1_hf, c0_hf);
        r1_c1_مجموع_p = hvx_vec_mpyacc_f32_f16(r1_c1_مجموع_p, r1_hf, c1_hf);
    }

    HVX_سمتیہ r0_c0_مجموع = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c0_مجموع_p), Q6_V_hi_W(r0_c0_مجموع_p)));
    HVX_سمتیہ r0_c1_مجموع = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c1_مجموع_p), Q6_V_hi_W(r0_c1_مجموع_p)));
    HVX_سمتیہ r1_c0_مجموع = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c0_مجموع_p), Q6_V_hi_W(r1_c0_مجموع_p)));
    HVX_سمتیہ r1_c1_مجموع = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c1_مجموع_p), Q6_V_hi_W(r1_c1_مجموع_p)));

    // Reduce and store results
    HVX_سمتیہ r0_r1_c0_مجموع = hvx_vec_reduce_مجموع_f32x2(r0_c0_مجموع, r1_c0_مجموع);
    HVX_سمتیہ r0_r1_c1_مجموع = hvx_vec_reduce_مجموع_f32x2(r0_c1_مجموع, r1_c1_مجموع);

    hvx_vec_store_u(&s0[0], 8, r0_r1_c0_مجموع);  // row0,col0 row1,col0
    hvx_vec_store_u(&s1[0], 8, r0_r1_c1_مجموع);  // row0,col1 row1,col1
}



ساکن اندرونی خالی hvx_tensor_جوڑو_f32_grid(
    ثابت ساخت htp_tensor * محدود dst,
    ثابت ساخت htp_tensor * محدود src2,
    uint32_t آغاز_صف,
    uint32_t انتہا_صف,
    uint32_t آغاز_ستون,
    uint32_t انتہا_ستون,
    ثابت ساخت fastdiv_اقدار * div_ne11_12,
    ثابت ساخت fastdiv_اقدار * div_ne11
) {
    اگر (آغاز_صف >= انتہا_صف || آغاز_ستون >= انتہا_ستون) واپس;
    ثابت uint32_t nb1 = dst->nb[1]; // row stride in bytes

    ثابت uint32_t ne11 = dst->ne[1];
    ثابت uint32_t ne12 = dst->ne[2];
    ثابت uint32_t ne11_12 = ne11 * ne12;

    ثابت bool is_broadcast1 = (src2->ne[1] == 1);
    ثابت bool is_broadcast2 = (src2->ne[2] == 1);
    ثابت bool is_broadcast3 = (src2->ne[3] == 1);

    برائے (uint32_t r = آغاز_صف; r < انتہا_صف; r++) {
        اعشاری * dst_صف = (اعشاری *) ((uint8_t *) dst->ڈیٹا + (size_t) r * nb1);

        uint32_t i13 = fastdiv(r, div_ne11_12);
        uint32_t i12 = fastdiv(r - i13 * ne11_12, div_ne11);
        uint32_t i11 = r - i13 * ne11_12 - i12 * ne11;

        uint32_t i23 = is_broadcast3 ? 0 : i13;
        uint32_t i22 = is_broadcast2 ? 0 : i12;
        uint32_t i21 = is_broadcast1 ? 0 : i11;

        ثابت اعشاری * src2_صف = (ثابت اعشاری *) ((ثابت uint8_t *) src2->ڈیٹا +
                                  (size_t) i21 * src2->nb[1] + (size_t) i22 * src2->nb[2] + (size_t) i23 * src2->nb[3]);

        اعشاری * dst_ptr = &dst_صف[آغاز_ستون];
        ثابت اعشاری * src2_ptr = &src2_صف[آغاز_ستون];
        صحیح بقایا = انتہا_ستون - آغاز_ستون;
        جب (بقایا >= 32) {
            HVX_سمتیہ v_برآمد = hvx_vmemu(dst_ptr);
            HVX_سمتیہ v_z   = hvx_vmemu(src2_ptr);
            hvx_vmemu(dst_ptr) = hvx_vec_جوڑو_f32_f32(v_برآمد, v_z);
            dst_ptr += 32;
            src2_ptr += 32;
            بقایا -= 32;
        }
        اگر (بقایا > 0) {
            HVX_سمتیہ v_برآمد = hvx_vmemu(dst_ptr);
            HVX_سمتیہ v_z   = hvx_vmemu(src2_ptr);
            hvx_vec_store_u(dst_ptr, بقایا * حجم(اعشاری), hvx_vec_جوڑو_f32_f32(v_برآمد, v_z));
        }
    }
}

#اگر_ختم // HVX_MM_KERNELS_FLOAT_H
