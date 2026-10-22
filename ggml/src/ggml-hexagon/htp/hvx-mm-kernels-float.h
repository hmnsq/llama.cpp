#यदि_अपरिभाषितम् HVX_MM_KERNELS_FLOAT_H
#परिभाषय HVX_MM_KERNELS_FLOAT_H

#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "htp-tensor.h"

// Float activation copy/quantization kernels (DDR -> VTCM)

स्थिर अन्तःस्थ शून्य quantize_f32_f32_kernel(
    ध्रुव uint8_t * सीमित src_दत्त,
    uint8_t * सीमित dst_दत्त,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (शून्य) tmp_दत्त;
    ध्रुव size_t src_row_आकार = ne0 * परिमाण(भिन्न);
    प्रति (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_दत्त, src_row_आकार, src_stride, 2);
        hvx_copy_f32_au(dst_दत्त, src_दत्त, ne0);

        dst_दत्त += dst_stride;
        src_दत्त += src_stride;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f32_f16_kernel(
    ध्रुव uint8_t * सीमित src_दत्त,
    uint8_t * सीमित dst_दत्त,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (शून्य) tmp_दत्त;
    ध्रुव size_t src_row_आकार = ne0 * परिमाण(भिन्न);
    प्रति (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_दत्त, src_row_आकार, src_stride, 2);
        hvx_copy_f16_f32_au(dst_दत्त, src_दत्त, ne0);

        dst_दत्त += dst_stride;
        src_दत्त += src_stride;
    }
}

स्थिर अन्तःस्थ शून्य quantize_f16_f16_kernel(
    ध्रुव uint8_t * सीमित src_दत्त,
    uint8_t * सीमित dst_दत्त,
    uint8_t * सीमित tmp_दत्त,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (शून्य) tmp_दत्त;
    ध्रुव size_t src_row_आकार = ne0 * परिमाण(भिन्न);
    प्रति (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_दत्त, src_row_आकार, src_stride, 2);
        hvx_copy_f16_au(dst_दत्त, src_दत्त, ne0);

        dst_दत्त += dst_stride;
        src_दत्त += src_stride;
    }
}

// Float dot product kernels (HVX)

#यदि __HVX_ARCH__ < 79
#परिभाषय HVX_OP_सङ्कलय_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#परिभाषय HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#अन्यथा
#परिभाषय HVX_OP_सङ्कलय_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#परिभाषय HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#यदि_अन्तः

स्थिर अन्तःस्थ शून्य flat_vec_dot_f32_f32_aa_1x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy) {
    ध्रुव HVX_सदिशम् * सीमित x = (ध्रुव HVX_सदिशम् *) vx;
    ध्रुव HVX_सदिशम् * सीमित y = (ध्रुव HVX_सदिशम् *) vy;

    uint32_t nvec = n / VLEN_FP32; // num full fp32 hvx vectors
    uint32_t nloe = n % VLEN_FP32; // leftover elements

    HVX_सदिशम् rsum = Q6_V_vzero();

    uint32_t i = 0;

    #निर्देशः unroll(4)
    प्रति (i = 0; i < nvec; i++) {
        HVX_सदिशम् prod = HVX_OP_MUL_F32(x[i], y[i]);
        rsum = HVX_OP_सङ्कलय_F32(rsum, prod);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् x_sf = Q6_V_vand_QV(bmask, x[i]);
        HVX_सदिशम् y_sf = Q6_V_vand_QV(bmask, y[i]);
        HVX_सदिशम् prod = HVX_OP_MUL_F32(x_sf, y_sf);
        rsum = HVX_OP_सङ्कलय_F32(rsum, prod);
    }

    *s = hvx_vec_get_f32(hvx_vec_reduce_योग_f32(rsum));
}

स्थिर अन्तःस्थ शून्य flat_vec_dot_f32_f32_aa_2x1(ध्रुव uint32_t n, भिन्न * सीमित s0,
                                ध्रुव शून्य * सीमित vx0, ध्रुव शून्य * सीमित vx1,
                                ध्रुव शून्य * सीमित vy0) {
    ध्रुव HVX_सदिशम् * सीमित x0 = (ध्रुव HVX_सदिशम् *) vx0;
    ध्रुव HVX_सदिशम् * सीमित x1 = (ध्रुव HVX_सदिशम् *) vx1;
    ध्रुव HVX_सदिशम् * सीमित y  = (ध्रुव HVX_सदिशम् *) vy0;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_सदिशम् rsum0 = Q6_V_vzero();
    HVX_सदिशम् rsum1 = Q6_V_vzero();

    uint32_t i = 0;

    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; i++) {
        HVX_सदिशम् y_sf = y[i];
        HVX_सदिशम् prod0 = HVX_OP_MUL_F32(x0[i], y_sf);
        HVX_सदिशम् prod1 = HVX_OP_MUL_F32(x1[i], y_sf);
        rsum0 = HVX_OP_सङ्कलय_F32(rsum0, prod0);
        rsum1 = HVX_OP_सङ्कलय_F32(rsum1, prod1);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् y_sf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_सदिशम् x0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_सदिशम् x1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_सदिशम् prod0 = HVX_OP_MUL_F32(x0_sf, y_sf);
        HVX_सदिशम् prod1 = HVX_OP_MUL_F32(x1_sf, y_sf);
        rsum0 = HVX_OP_सङ्कलय_F32(rsum0, prod0);
        rsum1 = HVX_OP_सङ्कलय_F32(rsum1, prod1);
    }

    HVX_सदिशम् rsum = hvx_vec_reduce_योग_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

स्थिर अन्तःस्थ शून्य flat_vec_dot_f32_f32_aa_2x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1,
                                ध्रुव शून्य * सीमित vx0, ध्रुव शून्य * सीमित vx1,
                                ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1) {
    ध्रुव HVX_सदिशम् * सीमित x0 = (ध्रुव HVX_सदिशम् *) vx0;
    ध्रुव HVX_सदिशम् * सीमित x1 = (ध्रुव HVX_सदिशम् *) vx1;
    ध्रुव HVX_सदिशम् * सीमित y0 = (ध्रुव HVX_सदिशम् *) vy0;
    ध्रुव HVX_सदिशम् * सीमित y1 = (ध्रुव HVX_सदिशम् *) vy1;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_सदिशम् r0_c0_योग = Q6_V_vzero();
    HVX_सदिशम् r0_c1_योग = Q6_V_vzero();
    HVX_सदिशम् r1_c0_योग = Q6_V_vzero();
    HVX_सदिशम् r1_c1_योग = Q6_V_vzero();

    uint32_t i = 0;

    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; i++) {
        HVX_सदिशम् r0_sf = x0[i];
        HVX_सदिशम् r1_sf = x1[i];
        HVX_सदिशम् c0_sf = y0[i];
        HVX_सदिशम् c1_sf = y1[i];

        r0_c0_योग = HVX_OP_सङ्कलय_F32(r0_c0_योग, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_योग = HVX_OP_सङ्कलय_F32(r0_c1_योग, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_योग = HVX_OP_सङ्कलय_F32(r1_c0_योग, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_योग = HVX_OP_सङ्कलय_F32(r1_c1_योग, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    यदि (nloe) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);

        HVX_सदिशम् r0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_सदिशम् r1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_सदिशम् c0_sf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_सदिशम् c1_sf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_योग = HVX_OP_सङ्कलय_F32(r0_c0_योग, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_योग = HVX_OP_सङ्कलय_F32(r0_c1_योग, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_योग = HVX_OP_सङ्कलय_F32(r1_c0_योग, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_योग = HVX_OP_सङ्कलय_F32(r1_c1_योग, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    // Reduce and store results
    HVX_सदिशम् r0_r1_c0_योग = hvx_vec_reduce_योग_f32x2(r0_c0_योग, r1_c0_योग);
    HVX_सदिशम् r0_r1_c1_योग = hvx_vec_reduce_योग_f32x2(r0_c1_योग, r1_c1_योग);

    hvx_vec_store_u(s0, 8, r0_r1_c0_योग);
    hvx_vec_store_u(s1, 8, r0_r1_c1_योग);
}

स्थिर अन्तःस्थ शून्य flat_vec_dot_bf16_f32_aa_1x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy) {
    ध्रुव HVX_सदिशम् * सीमित x = (ध्रुव HVX_सदिशम् *) vx;
    ध्रुव HVX_सदिशम् * सीमित y = (ध्रुव HVX_सदिशम् *) vy;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_सदिशम् rsum0 = Q6_V_vzero();
    HVX_सदिशम् rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; i++) {
        HVX_सदिशम् v_bf16 = x[i];
        HVX_सदिशम्_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_सदिशम् w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_सदिशम् w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        rsum0 = HVX_OP_सङ्कलय_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y[2 * i + 0]));
        rsum1 = HVX_OP_सङ्कलय_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y[2 * i + 1]));
    }

    यदि (rem) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_सदिशम् v_bf16 = Q6_V_vand_QV(bmask, x[i]);
        HVX_सदिशम्_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_सदिशम् w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_सदिशम् w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        यदि (rem <= 32) {
            HVX_सदिशम्_Pred a_आवरणम् = Q6_Q_vsetq2_R(rem * 4);
            HVX_सदिशम् y0 = Q6_V_vand_QV(a_आवरणम्, y[2 * i + 0]);
            rsum0 = HVX_OP_सङ्कलय_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));
        } अन्यथा {
            HVX_सदिशम् y0 = y[2 * i + 0];
            rsum0 = HVX_OP_सङ्कलय_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));

            HVX_सदिशम्_Pred a_आवरणम् = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_सदिशम् y1 = Q6_V_vand_QV(a_आवरणम्, y[2 * i + 1]);
            rsum1 = HVX_OP_सङ्कलय_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y1));
        }
    }

    HVX_सदिशम् rsum = HVX_OP_सङ्कलय_F32(rsum0, rsum1);
    *s = hvx_vec_get_f32(hvx_vec_reduce_योग_f32(rsum));
}

स्थिर अन्तःस्थ शून्य flat_vec_dot_bf16_f32_aa_2x1(ध्रुव uint32_t n, भिन्न * सीमित s0,
                                ध्रुव शून्य * सीमित vx0, ध्रुव शून्य * सीमित vx1,
                                ध्रुव शून्य * सीमित vy0) {
    ध्रुव HVX_सदिशम् * सीमित x0 = (ध्रुव HVX_सदिशम् *) vx0;
    ध्रुव HVX_सदिशम् * सीमित x1 = (ध्रुव HVX_सदिशम् *) vx1;
    ध्रुव HVX_सदिशम् * सीमित y  = (ध्रुव HVX_सदिशम् *) vy0;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_सदिशम् rsum0 = Q6_V_vzero();
    HVX_सदिशम् rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; i++) {
        HVX_सदिशम्_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_सदिशम्_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_सदिशम् w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_सदिशम् w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_सदिशम् w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_सदिशम् w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_सदिशम् y0 = y[2 * i + 0];
        HVX_सदिशम् y1 = y[2 * i + 1];

        rsum0 = HVX_OP_सङ्कलय_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
        rsum1 = HVX_OP_सङ्कलय_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        rsum0 = HVX_OP_सङ्कलय_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
        rsum1 = HVX_OP_सङ्कलय_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
    }

    यदि (rem) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_सदिशम्_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_सदिशम्_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_सदिशम् w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_सदिशम् w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_सदिशम् w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_सदिशम् w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        यदि (rem <= 32) {
            HVX_सदिशम्_Pred a_आवरणम् = Q6_Q_vsetq2_R(rem * 4);
            HVX_सदिशम् y0 = Q6_V_vand_QV(a_आवरणम्, y[2 * i + 0]);
            rsum0 = HVX_OP_सङ्कलय_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_सङ्कलय_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        } अन्यथा {
            HVX_सदिशम् y0 = y[2 * i + 0];
            rsum0 = HVX_OP_सङ्कलय_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_सङ्कलय_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));

            HVX_सदिशम्_Pred a_आवरणम् = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_सदिशम् y1 = Q6_V_vand_QV(a_आवरणम्, y[2 * i + 1]);
            rsum0 = HVX_OP_सङ्कलय_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
            rsum1 = HVX_OP_सङ्कलय_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
        }
    }

    HVX_सदिशम् rsum = hvx_vec_reduce_योग_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

स्थिर अन्तःस्थ शून्य flat_vec_dot_bf16_f32_aa_2x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1,
                                ध्रुव शून्य * सीमित vx0, ध्रुव शून्य * सीमित vx1,
                                ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1) {
    ध्रुव HVX_सदिशम् * सीमित x0 = (ध्रुव HVX_सदिशम् *) vx0;
    ध्रुव HVX_सदिशम् * सीमित x1 = (ध्रुव HVX_सदिशम् *) vx1;
    ध्रुव HVX_सदिशम् * सीमित y0 = (ध्रुव HVX_सदिशम् *) vy0;
    ध्रुव HVX_सदिशम् * सीमित y1 = (ध्रुव HVX_सदिशम् *) vy1;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_सदिशम् r0_c0_योग = Q6_V_vzero();
    HVX_सदिशम् r0_c1_योग = Q6_V_vzero();
    HVX_सदिशम् r1_c0_योग = Q6_V_vzero();
    HVX_सदिशम् r1_c1_योग = Q6_V_vzero();

    uint32_t i = 0;
    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; i++) {
        HVX_सदिशम्_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_सदिशम्_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_सदिशम् w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_सदिशम् w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_सदिशम् w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_सदिशम् w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_सदिशम् c0_0 = y0[2 * i + 0];
        HVX_सदिशम् c0_1 = y0[2 * i + 1];
        HVX_सदिशम् c1_0 = y1[2 * i + 0];
        HVX_सदिशम् c1_1 = y1[2 * i + 1];

        r0_c0_योग = HVX_OP_सङ्कलय_F32(r0_c0_योग, HVX_OP_MUL_F32(w00_f32, c0_0));
        r0_c1_योग = HVX_OP_सङ्कलय_F32(r0_c1_योग, HVX_OP_MUL_F32(w00_f32, c1_0));
        r1_c0_योग = HVX_OP_सङ्कलय_F32(r1_c0_योग, HVX_OP_MUL_F32(w10_f32, c0_0));
        r1_c1_योग = HVX_OP_सङ्कलय_F32(r1_c1_योग, HVX_OP_MUL_F32(w10_f32, c1_0));

        r0_c0_योग = HVX_OP_सङ्कलय_F32(r0_c0_योग, HVX_OP_MUL_F32(w01_f32, c0_1));
        r0_c1_योग = HVX_OP_सङ्कलय_F32(r0_c1_योग, HVX_OP_MUL_F32(w01_f32, c1_1));
        r1_c0_योग = HVX_OP_सङ्कलय_F32(r1_c0_योग, HVX_OP_MUL_F32(w11_f32, c0_1));
        r1_c1_योग = HVX_OP_सङ्कलय_F32(r1_c1_योग, HVX_OP_MUL_F32(w11_f32, c1_1));
    }

    यदि (rem) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_सदिशम्_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_सदिशम्_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_सदिशम् w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_सदिशम् w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_सदिशम् w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_सदिशम् w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        यदि (rem <= 32) {
            HVX_सदिशम्_Pred a_आवरणम् = Q6_Q_vsetq2_R(rem * 4);
            HVX_सदिशम् c0_0 = Q6_V_vand_QV(a_आवरणम्, y0[2 * i + 0]);
            HVX_सदिशम् c1_0 = Q6_V_vand_QV(a_आवरणम्, y1[2 * i + 0]);
            r0_c0_योग = HVX_OP_सङ्कलय_F32(r0_c0_योग, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_योग = HVX_OP_सङ्कलय_F32(r0_c1_योग, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_योग = HVX_OP_सङ्कलय_F32(r1_c0_योग, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_योग = HVX_OP_सङ्कलय_F32(r1_c1_योग, HVX_OP_MUL_F32(w10_f32, c1_0));
        } अन्यथा {
            HVX_सदिशम् c0_0 = y0[2 * i + 0];
            HVX_सदिशम् c1_0 = y1[2 * i + 0];
            r0_c0_योग = HVX_OP_सङ्कलय_F32(r0_c0_योग, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_योग = HVX_OP_सङ्कलय_F32(r0_c1_योग, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_योग = HVX_OP_सङ्कलय_F32(r1_c0_योग, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_योग = HVX_OP_सङ्कलय_F32(r1_c1_योग, HVX_OP_MUL_F32(w10_f32, c1_0));

            HVX_सदिशम्_Pred a_आवरणम् = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_सदिशम् c0_1 = Q6_V_vand_QV(a_आवरणम्, y0[2 * i + 1]);
            HVX_सदिशम् c1_1 = Q6_V_vand_QV(a_आवरणम्, y1[2 * i + 1]);
            r0_c0_योग = HVX_OP_सङ्कलय_F32(r0_c0_योग, HVX_OP_MUL_F32(w01_f32, c0_1));
            r0_c1_योग = HVX_OP_सङ्कलय_F32(r0_c1_योग, HVX_OP_MUL_F32(w01_f32, c1_1));
            r1_c0_योग = HVX_OP_सङ्कलय_F32(r1_c0_योग, HVX_OP_MUL_F32(w11_f32, c0_1));
            r1_c1_योग = HVX_OP_सङ्कलय_F32(r1_c1_योग, HVX_OP_MUL_F32(w11_f32, c1_1));
        }
    }

    HVX_सदिशम् r0_r1_c0_योग = hvx_vec_reduce_योग_f32x2(r0_c0_योग, r1_c0_योग);
    HVX_सदिशम् r0_r1_c1_योग = hvx_vec_reduce_योग_f32x2(r0_c1_योग, r1_c1_योग);

    hvx_vec_store_u(s0, 8, r0_r1_c0_योग);
    hvx_vec_store_u(s1, 8, r0_r1_c1_योग);
}

#अपरिभाषय HVX_OP_सङ्कलय_F32
#अपरिभाषय HVX_OP_MUL_F32

स्थिर अन्तःस्थ शून्य flat_vec_dot_f16_f16_aa_1x1(ध्रुव uint32_t n, भिन्न * सीमित s, ध्रुव शून्य * सीमित vx, ध्रुव शून्य * सीमित vy) {
    ध्रुव HVX_सदिशम् * सीमित x = (ध्रुव HVX_सदिशम् *) vx;
    ध्रुव HVX_सदिशम् * सीमित y = (ध्रुव HVX_सदिशम् *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_सदिशम्_Pair rsum_p = Q6_W_vzero();

    uint32_t i = 0;

    #निर्देशः unroll(4)
    प्रति (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x[i], y[i]);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_सदिशम् x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_सदिशम् y_hf = Q6_V_vand_QV(bmask, y[i]);
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_सदिशम् rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));
    hvx_vec_store_u(s, 4, hvx_vec_reduce_योग_f32(rsum));
}

स्थिर अन्तःस्थ शून्य flat_vec_dot_f16_f16_aa_2x1(ध्रुव uint32_t n, भिन्न * सीमित s0,
                                ध्रुव शून्य * सीमित vx0, ध्रुव शून्य * सीमित vx1,
                                ध्रुव शून्य * सीमित vy0) {
    ध्रुव HVX_सदिशम् * सीमित x0 = (ध्रुव HVX_सदिशम् *) vx0;
    ध्रुव HVX_सदिशम् * सीमित x1 = (ध्रुव HVX_सदिशम् *) vx1;
    ध्रुव HVX_सदिशम् * सीमित y  = (ध्रुव HVX_सदिशम् *) vy0;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_सदिशम्_Pair rsum0_p = Q6_W_vzero();
    HVX_सदिशम्_Pair rsum1_p = Q6_W_vzero();

    uint32_t i = 0;

    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; i++) {
        HVX_सदिशम् y_hf = y[i];
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0[i], y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1[i], y_hf);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_सदिशम् y_hf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_सदिशम् x0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_सदिशम् x1_hf = Q6_V_vand_QV(bmask, x1[i]);
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
    }

    HVX_सदिशम् rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p)));
    HVX_सदिशम् rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p)));
    HVX_सदिशम् rsum  = hvx_vec_reduce_योग_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

स्थिर अन्तःस्थ शून्य flat_vec_dot_f16_f16_aa_2x2(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1,
                                ध्रुव शून्य * सीमित vx0, ध्रुव शून्य * सीमित vx1,
                                ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1) {
    ध्रुव HVX_सदिशम् * सीमित x0 = (ध्रुव HVX_सदिशम् *) vx0;
    ध्रुव HVX_सदिशम् * सीमित x1 = (ध्रुव HVX_सदिशम् *) vx1;
    ध्रुव HVX_सदिशम् * सीमित y0 = (ध्रुव HVX_सदिशम् *) vy0;
    ध्रुव HVX_सदिशम् * सीमित y1 = (ध्रुव HVX_सदिशम् *) vy1;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    // Row sums (sf) - 4 accumulators for 2x2 tile
    HVX_सदिशम्_Pair r0_c0_योग_p = Q6_W_vzero();
    HVX_सदिशम्_Pair r0_c1_योग_p = Q6_W_vzero();
    HVX_सदिशम्_Pair r1_c0_योग_p = Q6_W_vzero();
    HVX_सदिशम्_Pair r1_c1_योग_p = Q6_W_vzero();

    uint32_t i = 0;

    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; i++) {
        HVX_सदिशम् r0_hf = x0[i];
        HVX_सदिशम् r1_hf = x1[i];
        HVX_सदिशम् c0_hf = y0[i];
        HVX_सदिशम् c1_hf = y1[i];

        // Compute 4 dot products: r0xc0, r0xc1, r1xc0, r1xc1
        r0_c0_योग_p = hvx_vec_mpyacc_f32_f16(r0_c0_योग_p, r0_hf, c0_hf);
        r0_c1_योग_p = hvx_vec_mpyacc_f32_f16(r0_c1_योग_p, r0_hf, c1_hf);
        r1_c0_योग_p = hvx_vec_mpyacc_f32_f16(r1_c0_योग_p, r1_hf, c0_hf);
        r1_c1_योग_p = hvx_vec_mpyacc_f32_f16(r1_c1_योग_p, r1_hf, c1_hf);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 2);

        HVX_सदिशम् r0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_सदिशम् r1_hf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_सदिशम् c0_hf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_सदिशम् c1_hf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_योग_p = hvx_vec_mpyacc_f32_f16(r0_c0_योग_p, r0_hf, c0_hf);
        r0_c1_योग_p = hvx_vec_mpyacc_f32_f16(r0_c1_योग_p, r0_hf, c1_hf);
        r1_c0_योग_p = hvx_vec_mpyacc_f32_f16(r1_c0_योग_p, r1_hf, c0_hf);
        r1_c1_योग_p = hvx_vec_mpyacc_f32_f16(r1_c1_योग_p, r1_hf, c1_hf);
    }

    HVX_सदिशम् r0_c0_योग = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c0_योग_p), Q6_V_hi_W(r0_c0_योग_p)));
    HVX_सदिशम् r0_c1_योग = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c1_योग_p), Q6_V_hi_W(r0_c1_योग_p)));
    HVX_सदिशम् r1_c0_योग = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c0_योग_p), Q6_V_hi_W(r1_c0_योग_p)));
    HVX_सदिशम् r1_c1_योग = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c1_योग_p), Q6_V_hi_W(r1_c1_योग_p)));

    // Reduce and store results
    HVX_सदिशम् r0_r1_c0_योग = hvx_vec_reduce_योग_f32x2(r0_c0_योग, r1_c0_योग);
    HVX_सदिशम् r0_r1_c1_योग = hvx_vec_reduce_योग_f32x2(r0_c1_योग, r1_c1_योग);

    hvx_vec_store_u(&s0[0], 8, r0_r1_c0_योग);  // row0,col0 row1,col0
    hvx_vec_store_u(&s1[0], 8, r0_r1_c1_योग);  // row0,col1 row1,col1
}



स्थिर अन्तःस्थ शून्य hvx_tensor_सङ्कलय_f32_grid(
    ध्रुव संरचना htp_tensor * सीमित dst,
    ध्रुव संरचना htp_tensor * सीमित src2,
    uint32_t start_row,
    uint32_t अन्तः_row,
    uint32_t start_col,
    uint32_t अन्तः_col,
    ध्रुव संरचना fastdiv_मानानि * div_ne11_12,
    ध्रुव संरचना fastdiv_मानानि * div_ne11
) {
    यदि (start_row >= अन्तः_row || start_col >= अन्तः_col) प्रत्यागम;
    ध्रुव uint32_t nb1 = dst->nb[1]; // row stride in bytes

    ध्रुव uint32_t ne11 = dst->ne[1];
    ध्रुव uint32_t ne12 = dst->ne[2];
    ध्रुव uint32_t ne11_12 = ne11 * ne12;

    ध्रुव bool is_broadcast1 = (src2->ne[1] == 1);
    ध्रुव bool is_broadcast2 = (src2->ne[2] == 1);
    ध्रुव bool is_broadcast3 = (src2->ne[3] == 1);

    प्रति (uint32_t r = start_row; r < अन्तः_row; r++) {
        भिन्न * dst_row = (भिन्न *) ((uint8_t *) dst->दत्त + (size_t) r * nb1);

        uint32_t i13 = fastdiv(r, div_ne11_12);
        uint32_t i12 = fastdiv(r - i13 * ne11_12, div_ne11);
        uint32_t i11 = r - i13 * ne11_12 - i12 * ne11;

        uint32_t i23 = is_broadcast3 ? 0 : i13;
        uint32_t i22 = is_broadcast2 ? 0 : i12;
        uint32_t i21 = is_broadcast1 ? 0 : i11;

        ध्रुव भिन्न * src2_row = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src2->दत्त +
                                  (size_t) i21 * src2->nb[1] + (size_t) i22 * src2->nb[2] + (size_t) i23 * src2->nb[3]);

        भिन्न * dst_ptr = &dst_row[start_col];
        ध्रुव भिन्न * src2_ptr = &src2_row[start_col];
        पूर्णाङ्क remaining = अन्तः_col - start_col;
        यावत् (remaining >= 32) {
            HVX_सदिशम् v_बहिर्गमनम् = hvx_vmemu(dst_ptr);
            HVX_सदिशम् v_z   = hvx_vmemu(src2_ptr);
            hvx_vmemu(dst_ptr) = hvx_vec_सङ्कलय_f32_f32(v_बहिर्गमनम्, v_z);
            dst_ptr += 32;
            src2_ptr += 32;
            remaining -= 32;
        }
        यदि (remaining > 0) {
            HVX_सदिशम् v_बहिर्गमनम् = hvx_vmemu(dst_ptr);
            HVX_सदिशम् v_z   = hvx_vmemu(src2_ptr);
            hvx_vec_store_u(dst_ptr, remaining * परिमाण(भिन्न), hvx_vec_सङ्कलय_f32_f32(v_बहिर्गमनम्, v_z));
        }
    }
}

#यदि_अन्तः // HVX_MM_KERNELS_FLOAT_H
