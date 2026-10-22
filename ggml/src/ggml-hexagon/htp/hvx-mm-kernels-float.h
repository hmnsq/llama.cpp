#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_MM_KERNELS_FLOAT_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_MM_KERNELS_FLOAT_H

#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

// Float activation copy/quantization kernels (DDR -> VTCM)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ quantize_f32_f32_kernel(
    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_ದತ್ತಾಂಶ,
    uint8_t * ಸೀಮಿತ dst_ದತ್ತಾಂಶ,
    uint8_t * ಸೀಮಿತ tmp_ದತ್ತಾಂಶ,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (ಶೂನ್ಯ) tmp_ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_ದತ್ತಾಂಶ, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src_stride, 2);
        hvx_copy_f32_au(dst_ದತ್ತಾಂಶ, src_ದತ್ತಾಂಶ, ne0);

        dst_ದತ್ತಾಂಶ += dst_stride;
        src_ದತ್ತಾಂಶ += src_stride;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ quantize_f32_f16_kernel(
    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_ದತ್ತಾಂಶ,
    uint8_t * ಸೀಮಿತ dst_ದತ್ತಾಂಶ,
    uint8_t * ಸೀಮಿತ tmp_ದತ್ತಾಂಶ,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (ಶೂನ್ಯ) tmp_ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_ದತ್ತಾಂಶ, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src_stride, 2);
        hvx_copy_f16_f32_au(dst_ದತ್ತಾಂಶ, src_ದತ್ತಾಂಶ, ne0);

        dst_ದತ್ತಾಂಶ += dst_stride;
        src_ದತ್ತಾಂಶ += src_stride;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ quantize_f16_f16_kernel(
    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_ದತ್ತಾಂಶ,
    uint8_t * ಸೀಮಿತ dst_ದತ್ತಾಂಶ,
    uint8_t * ಸೀಮಿತ tmp_ದತ್ತಾಂಶ,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (ಶೂನ್ಯ) tmp_ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_ದತ್ತಾಂಶ, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src_stride, 2);
        hvx_copy_f16_au(dst_ದತ್ತಾಂಶ, src_ದತ್ತಾಂಶ, ne0);

        dst_ದತ್ತಾಂಶ += dst_stride;
        src_ದತ್ತಾಂಶ += src_stride;
    }
}

// Float dot product kernels (HVX)

#ಆದರೆ __HVX_ARCH__ < 79
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_ಕೂಡಿಸು_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_ಕೂಡಿಸು_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flat_vec_dot_f32_f32_aa_1x1(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy;

    uint32_t nvec = n / VLEN_FP32; // num full fp32 hvx vectors
    uint32_t nloe = n % VLEN_FP32; // leftover elements

    HVX_ಸದಿಶ rsum = Q6_V_vzero();

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ prod = HVX_OP_MUL_F32(x[i], y[i]);
        rsum = HVX_OP_ಕೂಡಿಸು_F32(rsum, prod);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ x_sf = Q6_V_vand_QV(bmask, x[i]);
        HVX_ಸದಿಶ y_sf = Q6_V_vand_QV(bmask, y[i]);
        HVX_ಸದಿಶ prod = HVX_OP_MUL_F32(x_sf, y_sf);
        rsum = HVX_OP_ಕೂಡಿಸು_F32(rsum, prod);
    }

    *s = hvx_vec_get_f32(hvx_vec_reduce_ಮೊತ್ತ_f32(rsum));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flat_vec_dot_f32_f32_aa_2x1(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx1,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx1;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y  = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy0;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_ಸದಿಶ rsum0 = Q6_V_vzero();
    HVX_ಸದಿಶ rsum1 = Q6_V_vzero();

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ y_sf = y[i];
        HVX_ಸದಿಶ prod0 = HVX_OP_MUL_F32(x0[i], y_sf);
        HVX_ಸದಿಶ prod1 = HVX_OP_MUL_F32(x1[i], y_sf);
        rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, prod0);
        rsum1 = HVX_OP_ಕೂಡಿಸು_F32(rsum1, prod1);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ y_sf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_ಸದಿಶ x0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_ಸದಿಶ x1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_ಸದಿಶ prod0 = HVX_OP_MUL_F32(x0_sf, y_sf);
        HVX_ಸದಿಶ prod1 = HVX_OP_MUL_F32(x1_sf, y_sf);
        rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, prod0);
        rsum1 = HVX_OP_ಕೂಡಿಸು_F32(rsum1, prod1);
    }

    HVX_ಸದಿಶ rsum = hvx_vec_reduce_ಮೊತ್ತ_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flat_vec_dot_f32_f32_aa_2x2(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0, ದಶಮಾಂಶ * ಸೀಮಿತ s1,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx1,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy1) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx1;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy1;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_ಸದಿಶ r0_c0_ಮೊತ್ತ = Q6_V_vzero();
    HVX_ಸದಿಶ r0_c1_ಮೊತ್ತ = Q6_V_vzero();
    HVX_ಸದಿಶ r1_c0_ಮೊತ್ತ = Q6_V_vzero();
    HVX_ಸದಿಶ r1_c1_ಮೊತ್ತ = Q6_V_vzero();

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ r0_sf = x0[i];
        HVX_ಸದಿಶ r1_sf = x1[i];
        HVX_ಸದಿಶ c0_sf = y0[i];
        HVX_ಸದಿಶ c1_sf = y1[i];

        r0_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);

        HVX_ಸದಿಶ r0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_ಸದಿಶ r1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_ಸದಿಶ c0_sf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_ಸದಿಶ c1_sf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    // Reduce and store results
    HVX_ಸದಿಶ r0_r1_c0_ಮೊತ್ತ = hvx_vec_reduce_ಮೊತ್ತ_f32x2(r0_c0_ಮೊತ್ತ, r1_c0_ಮೊತ್ತ);
    HVX_ಸದಿಶ r0_r1_c1_ಮೊತ್ತ = hvx_vec_reduce_ಮೊತ್ತ_f32x2(r0_c1_ಮೊತ್ತ, r1_c1_ಮೊತ್ತ);

    hvx_vec_store_u(s0, 8, r0_r1_c0_ಮೊತ್ತ);
    hvx_vec_store_u(s1, 8, r0_r1_c1_ಮೊತ್ತ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flat_vec_dot_bf16_f32_aa_1x1(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_ಸದಿಶ rsum0 = Q6_V_vzero();
    HVX_ಸದಿಶ rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v_bf16 = x[i];
        HVX_ಸದಿಶ_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_ಸದಿಶ w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_ಸದಿಶ w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y[2 * i + 0]));
        rsum1 = HVX_OP_ಕೂಡಿಸು_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y[2 * i + 1]));
    }

    ಆದರೆ (rem) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_ಸದಿಶ v_bf16 = Q6_V_vand_QV(bmask, x[i]);
        HVX_ಸದಿಶ_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_ಸದಿಶ w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_ಸದಿಶ w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        ಆದರೆ (rem <= 32) {
            HVX_ಸದಿಶ_Pred a_ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem * 4);
            HVX_ಸದಿಶ y0 = Q6_V_vand_QV(a_ಮುಖವಾಡ, y[2 * i + 0]);
            rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));
        } ಇಲ್ಲದಿದ್ದರೆ {
            HVX_ಸದಿಶ y0 = y[2 * i + 0];
            rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));

            HVX_ಸದಿಶ_Pred a_ಮುಖವಾಡ = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_ಸದಿಶ y1 = Q6_V_vand_QV(a_ಮುಖವಾಡ, y[2 * i + 1]);
            rsum1 = HVX_OP_ಕೂಡಿಸು_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y1));
        }
    }

    HVX_ಸದಿಶ rsum = HVX_OP_ಕೂಡಿಸು_F32(rsum0, rsum1);
    *s = hvx_vec_get_f32(hvx_vec_reduce_ಮೊತ್ತ_f32(rsum));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flat_vec_dot_bf16_f32_aa_2x1(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx1,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx1;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y  = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy0;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_ಸದಿಶ rsum0 = Q6_V_vzero();
    HVX_ಸದಿಶ rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_ಸದಿಶ_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_ಸದಿಶ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_ಸದಿಶ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_ಸದಿಶ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_ಸದಿಶ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_ಸದಿಶ y0 = y[2 * i + 0];
        HVX_ಸದಿಶ y1 = y[2 * i + 1];

        rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
        rsum1 = HVX_OP_ಕೂಡಿಸು_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
        rsum1 = HVX_OP_ಕೂಡಿಸು_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
    }

    ಆದರೆ (rem) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_ಸದಿಶ_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_ಸದಿಶ_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_ಸದಿಶ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_ಸದಿಶ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_ಸದಿಶ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_ಸದಿಶ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        ಆದರೆ (rem <= 32) {
            HVX_ಸದಿಶ_Pred a_ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem * 4);
            HVX_ಸದಿಶ y0 = Q6_V_vand_QV(a_ಮುಖವಾಡ, y[2 * i + 0]);
            rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_ಕೂಡಿಸು_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        } ಇಲ್ಲದಿದ್ದರೆ {
            HVX_ಸದಿಶ y0 = y[2 * i + 0];
            rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_ಕೂಡಿಸು_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));

            HVX_ಸದಿಶ_Pred a_ಮುಖವಾಡ = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_ಸದಿಶ y1 = Q6_V_vand_QV(a_ಮುಖವಾಡ, y[2 * i + 1]);
            rsum0 = HVX_OP_ಕೂಡಿಸು_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
            rsum1 = HVX_OP_ಕೂಡಿಸು_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
        }
    }

    HVX_ಸದಿಶ rsum = hvx_vec_reduce_ಮೊತ್ತ_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flat_vec_dot_bf16_f32_aa_2x2(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0, ದಶಮಾಂಶ * ಸೀಮಿತ s1,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx1,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy1) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx1;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy1;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_ಸದಿಶ r0_c0_ಮೊತ್ತ = Q6_V_vzero();
    HVX_ಸದಿಶ r0_c1_ಮೊತ್ತ = Q6_V_vzero();
    HVX_ಸದಿಶ r1_c0_ಮೊತ್ತ = Q6_V_vzero();
    HVX_ಸದಿಶ r1_c1_ಮೊತ್ತ = Q6_V_vzero();

    uint32_t i = 0;
    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_ಸದಿಶ_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_ಸದಿಶ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_ಸದಿಶ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_ಸದಿಶ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_ಸದಿಶ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_ಸದಿಶ c0_0 = y0[2 * i + 0];
        HVX_ಸದಿಶ c0_1 = y0[2 * i + 1];
        HVX_ಸದಿಶ c1_0 = y1[2 * i + 0];
        HVX_ಸದಿಶ c1_1 = y1[2 * i + 1];

        r0_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w00_f32, c0_0));
        r0_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w00_f32, c1_0));
        r1_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w10_f32, c0_0));
        r1_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w10_f32, c1_0));

        r0_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w01_f32, c0_1));
        r0_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w01_f32, c1_1));
        r1_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w11_f32, c0_1));
        r1_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w11_f32, c1_1));
    }

    ಆದರೆ (rem) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_ಸದಿಶ_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_ಸದಿಶ_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_ಸದಿಶ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_ಸದಿಶ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_ಸದಿಶ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_ಸದಿಶ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        ಆದರೆ (rem <= 32) {
            HVX_ಸದಿಶ_Pred a_ಮುಖವಾಡ = Q6_Q_vsetq2_R(rem * 4);
            HVX_ಸದಿಶ c0_0 = Q6_V_vand_QV(a_ಮುಖವಾಡ, y0[2 * i + 0]);
            HVX_ಸದಿಶ c1_0 = Q6_V_vand_QV(a_ಮುಖವಾಡ, y1[2 * i + 0]);
            r0_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w10_f32, c1_0));
        } ಇಲ್ಲದಿದ್ದರೆ {
            HVX_ಸದಿಶ c0_0 = y0[2 * i + 0];
            HVX_ಸದಿಶ c1_0 = y1[2 * i + 0];
            r0_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w10_f32, c1_0));

            HVX_ಸದಿಶ_Pred a_ಮುಖವಾಡ = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_ಸದಿಶ c0_1 = Q6_V_vand_QV(a_ಮುಖವಾಡ, y0[2 * i + 1]);
            HVX_ಸದಿಶ c1_1 = Q6_V_vand_QV(a_ಮುಖವಾಡ, y1[2 * i + 1]);
            r0_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w01_f32, c0_1));
            r0_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r0_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w01_f32, c1_1));
            r1_c0_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c0_ಮೊತ್ತ, HVX_OP_MUL_F32(w11_f32, c0_1));
            r1_c1_ಮೊತ್ತ = HVX_OP_ಕೂಡಿಸು_F32(r1_c1_ಮೊತ್ತ, HVX_OP_MUL_F32(w11_f32, c1_1));
        }
    }

    HVX_ಸದಿಶ r0_r1_c0_ಮೊತ್ತ = hvx_vec_reduce_ಮೊತ್ತ_f32x2(r0_c0_ಮೊತ್ತ, r1_c0_ಮೊತ್ತ);
    HVX_ಸದಿಶ r0_r1_c1_ಮೊತ್ತ = hvx_vec_reduce_ಮೊತ್ತ_f32x2(r0_c1_ಮೊತ್ತ, r1_c1_ಮೊತ್ತ);

    hvx_vec_store_u(s0, 8, r0_r1_c0_ಮೊತ್ತ);
    hvx_vec_store_u(s1, 8, r0_r1_c1_ಮೊತ್ತ);
}

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_ಕೂಡಿಸು_F32
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು HVX_OP_MUL_F32

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flat_vec_dot_f16_f16_aa_1x1(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ಸದಿಶ_Pair rsum_p = Q6_W_vzero();

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x[i], y[i]);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ಸದಿಶ x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_ಸದಿಶ y_hf = Q6_V_vand_QV(bmask, y[i]);
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_ಸದಿಶ rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));
    hvx_vec_store_u(s, 4, hvx_vec_reduce_ಮೊತ್ತ_f32(rsum));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flat_vec_dot_f16_f16_aa_2x1(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx1,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx1;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y  = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy0;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_ಸದಿಶ_Pair rsum0_p = Q6_W_vzero();
    HVX_ಸದಿಶ_Pair rsum1_p = Q6_W_vzero();

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ y_hf = y[i];
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0[i], y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1[i], y_hf);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ಸದಿಶ y_hf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_ಸದಿಶ x0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_ಸದಿಶ x1_hf = Q6_V_vand_QV(bmask, x1[i]);
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
    }

    HVX_ಸದಿಶ rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p)));
    HVX_ಸದಿಶ rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p)));
    HVX_ಸದಿಶ rsum  = hvx_vec_reduce_ಮೊತ್ತ_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flat_vec_dot_f16_f16_aa_2x2(ಸ್ಥಿರಾಂಕ uint32_t n, ದಶಮಾಂಶ * ಸೀಮಿತ s0, ದಶಮಾಂಶ * ಸೀಮಿತ s1,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vx1,
                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ vy1) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ x1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vx1;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ y1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) vy1;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    // Row sums (sf) - 4 accumulators for 2x2 tile
    HVX_ಸದಿಶ_Pair r0_c0_ಮೊತ್ತ_p = Q6_W_vzero();
    HVX_ಸದಿಶ_Pair r0_c1_ಮೊತ್ತ_p = Q6_W_vzero();
    HVX_ಸದಿಶ_Pair r1_c0_ಮೊತ್ತ_p = Q6_W_vzero();
    HVX_ಸದಿಶ_Pair r1_c1_ಮೊತ್ತ_p = Q6_W_vzero();

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ r0_hf = x0[i];
        HVX_ಸದಿಶ r1_hf = x1[i];
        HVX_ಸದಿಶ c0_hf = y0[i];
        HVX_ಸದಿಶ c1_hf = y1[i];

        // Compute 4 dot products: r0xc0, r0xc1, r1xc0, r1xc1
        r0_c0_ಮೊತ್ತ_p = hvx_vec_mpyacc_f32_f16(r0_c0_ಮೊತ್ತ_p, r0_hf, c0_hf);
        r0_c1_ಮೊತ್ತ_p = hvx_vec_mpyacc_f32_f16(r0_c1_ಮೊತ್ತ_p, r0_hf, c1_hf);
        r1_c0_ಮೊತ್ತ_p = hvx_vec_mpyacc_f32_f16(r1_c0_ಮೊತ್ತ_p, r1_hf, c0_hf);
        r1_c1_ಮೊತ್ತ_p = hvx_vec_mpyacc_f32_f16(r1_c1_ಮೊತ್ತ_p, r1_hf, c1_hf);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);

        HVX_ಸದಿಶ r0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_ಸದಿಶ r1_hf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_ಸದಿಶ c0_hf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_ಸದಿಶ c1_hf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_ಮೊತ್ತ_p = hvx_vec_mpyacc_f32_f16(r0_c0_ಮೊತ್ತ_p, r0_hf, c0_hf);
        r0_c1_ಮೊತ್ತ_p = hvx_vec_mpyacc_f32_f16(r0_c1_ಮೊತ್ತ_p, r0_hf, c1_hf);
        r1_c0_ಮೊತ್ತ_p = hvx_vec_mpyacc_f32_f16(r1_c0_ಮೊತ್ತ_p, r1_hf, c0_hf);
        r1_c1_ಮೊತ್ತ_p = hvx_vec_mpyacc_f32_f16(r1_c1_ಮೊತ್ತ_p, r1_hf, c1_hf);
    }

    HVX_ಸದಿಶ r0_c0_ಮೊತ್ತ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c0_ಮೊತ್ತ_p), Q6_V_hi_W(r0_c0_ಮೊತ್ತ_p)));
    HVX_ಸದಿಶ r0_c1_ಮೊತ್ತ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c1_ಮೊತ್ತ_p), Q6_V_hi_W(r0_c1_ಮೊತ್ತ_p)));
    HVX_ಸದಿಶ r1_c0_ಮೊತ್ತ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c0_ಮೊತ್ತ_p), Q6_V_hi_W(r1_c0_ಮೊತ್ತ_p)));
    HVX_ಸದಿಶ r1_c1_ಮೊತ್ತ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c1_ಮೊತ್ತ_p), Q6_V_hi_W(r1_c1_ಮೊತ್ತ_p)));

    // Reduce and store results
    HVX_ಸದಿಶ r0_r1_c0_ಮೊತ್ತ = hvx_vec_reduce_ಮೊತ್ತ_f32x2(r0_c0_ಮೊತ್ತ, r1_c0_ಮೊತ್ತ);
    HVX_ಸದಿಶ r0_r1_c1_ಮೊತ್ತ = hvx_vec_reduce_ಮೊತ್ತ_f32x2(r0_c1_ಮೊತ್ತ, r1_c1_ಮೊತ್ತ);

    hvx_vec_store_u(&s0[0], 8, r0_r1_c0_ಮೊತ್ತ);  // row0,col0 row1,col0
    hvx_vec_store_u(&s1[0], 8, r0_r1_c1_ಮೊತ್ತ);  // row0,col1 row1,col1
}



ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_tensor_ಕೂಡಿಸು_f32_grid(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src2,
    uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು,
    uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು,
    uint32_t ಆರಂಭ_ಲಂಬಸಾಲು,
    uint32_t ಕೊನೆ_ಲಂಬಸಾಲು,
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne11_12,
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne11
) {
    ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಕೊನೆ_ಅಡ್ಡಸಾಲು || ಆರಂಭ_ಲಂಬಸಾಲು >= ಕೊನೆ_ಲಂಬಸಾಲು) ಹಿಂತಿರುಗಿ;
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1]; // row stride in bytes

    ಸ್ಥಿರಾಂಕ uint32_t ne11 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne11_12 = ne11 * ne12;

    ಸ್ಥಿರಾಂಕ bool is_broadcast1 = (src2->ne[1] == 1);
    ಸ್ಥಿರಾಂಕ bool is_broadcast2 = (src2->ne[2] == 1);
    ಸ್ಥಿರಾಂಕ bool is_broadcast3 = (src2->ne[3] == 1);

    ಗಾಗಿ (uint32_t r = ಆರಂಭ_ಅಡ್ಡಸಾಲು; r < ಕೊನೆ_ಅಡ್ಡಸಾಲು; r++) {
        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((uint8_t *) dst->ದತ್ತಾಂಶ + (size_t) r * nb1);

        uint32_t i13 = fastdiv(r, div_ne11_12);
        uint32_t i12 = fastdiv(r - i13 * ne11_12, div_ne11);
        uint32_t i11 = r - i13 * ne11_12 - i12 * ne11;

        uint32_t i23 = is_broadcast3 ? 0 : i13;
        uint32_t i22 = is_broadcast2 ? 0 : i12;
        uint32_t i21 = is_broadcast1 ? 0 : i11;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src2_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src2->ದತ್ತಾಂಶ +
                                  (size_t) i21 * src2->nb[1] + (size_t) i22 * src2->nb[2] + (size_t) i23 * src2->nb[3]);

        ದಶಮಾಂಶ * dst_ptr = &dst_ಅಡ್ಡಸಾಲು[ಆರಂಭ_ಲಂಬಸಾಲು];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src2_ptr = &src2_ಅಡ್ಡಸಾಲು[ಆರಂಭ_ಲಂಬಸಾಲು];
        ಪೂರ್ಣಾಂಕ ಉಳಿದ = ಕೊನೆ_ಲಂಬಸಾಲು - ಆರಂಭ_ಲಂಬಸಾಲು;
        ಆಗ (ಉಳಿದ >= 32) {
            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vmemu(dst_ptr);
            HVX_ಸದಿಶ v_z   = hvx_vmemu(src2_ptr);
            hvx_vmemu(dst_ptr) = hvx_vec_ಕೂಡಿಸು_f32_f32(v_ಹೊರಹೋಗು, v_z);
            dst_ptr += 32;
            src2_ptr += 32;
            ಉಳಿದ -= 32;
        }
        ಆದರೆ (ಉಳಿದ > 0) {
            HVX_ಸದಿಶ v_ಹೊರಹೋಗು = hvx_vmemu(dst_ptr);
            HVX_ಸದಿಶ v_z   = hvx_vmemu(src2_ptr);
            hvx_vec_store_u(dst_ptr, ಉಳಿದ * ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_ಕೂಡಿಸು_f32_f32(v_ಹೊರಹೋಗು, v_z));
        }
    }
}

#ಆದರೆ_ಅಂತ್ಯ // HVX_MM_KERNELS_FLOAT_H
