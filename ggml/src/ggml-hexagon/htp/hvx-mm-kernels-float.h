#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_MM_KERNELS_FLOAT_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_MM_KERNELS_FLOAT_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"

// Float activation copy/quantization kernels (DDR -> VTCM)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ quantize_f32_f32_kernel(
    ਅਚਲ uint8_t * ਸੀਮਿਤ src_ਅੰਕੜੇ,
    uint8_t * ਸੀਮਿਤ dst_ਅੰਕੜੇ,
    uint8_t * ਸੀਮਿਤ tmp_ਅੰਕੜੇ,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (ਖਾਲੀ) tmp_ਅੰਕੜੇ;
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ = ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_ਅੰਕੜੇ, src_ਪੰਕਤੀ_ਮਾਪ, src_stride, 2);
        hvx_copy_f32_au(dst_ਅੰਕੜੇ, src_ਅੰਕੜੇ, ne0);

        dst_ਅੰਕੜੇ += dst_stride;
        src_ਅੰਕੜੇ += src_stride;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ quantize_f32_f16_kernel(
    ਅਚਲ uint8_t * ਸੀਮਿਤ src_ਅੰਕੜੇ,
    uint8_t * ਸੀਮਿਤ dst_ਅੰਕੜੇ,
    uint8_t * ਸੀਮਿਤ tmp_ਅੰਕੜੇ,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (ਖਾਲੀ) tmp_ਅੰਕੜੇ;
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ = ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_ਅੰਕੜੇ, src_ਪੰਕਤੀ_ਮਾਪ, src_stride, 2);
        hvx_copy_f16_f32_au(dst_ਅੰਕੜੇ, src_ਅੰਕੜੇ, ne0);

        dst_ਅੰਕੜੇ += dst_stride;
        src_ਅੰਕੜੇ += src_stride;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ quantize_f16_f16_kernel(
    ਅਚਲ uint8_t * ਸੀਮਿਤ src_ਅੰਕੜੇ,
    uint8_t * ਸੀਮਿਤ dst_ਅੰਕੜੇ,
    uint8_t * ਸੀਮਿਤ tmp_ਅੰਕੜੇ,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (ਖਾਲੀ) tmp_ਅੰਕੜੇ;
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ = ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_ਅੰਕੜੇ, src_ਪੰਕਤੀ_ਮਾਪ, src_stride, 2);
        hvx_copy_f16_au(dst_ਅੰਕੜੇ, src_ਅੰਕੜੇ, ne0);

        dst_ਅੰਕੜੇ += dst_stride;
        src_ਅੰਕੜੇ += src_stride;
    }
}

// Float dot product kernels (HVX)

#ਜੇ __HVX_ARCH__ < 79
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_ਜੋੜੋ_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_ਜੋੜੋ_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flat_vec_dot_f32_f32_aa_1x1(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy;

    uint32_t nvec = n / VLEN_FP32; // num full fp32 hvx vectors
    uint32_t nloe = n % VLEN_FP32; // leftover elements

    HVX_ਸਦਿਸ਼ rsum = Q6_V_vzero();

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ prod = HVX_OP_MUL_F32(x[i], y[i]);
        rsum = HVX_OP_ਜੋੜੋ_F32(rsum, prod);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ x_sf = Q6_V_vand_QV(bmask, x[i]);
        HVX_ਸਦਿਸ਼ y_sf = Q6_V_vand_QV(bmask, y[i]);
        HVX_ਸਦਿਸ਼ prod = HVX_OP_MUL_F32(x_sf, y_sf);
        rsum = HVX_OP_ਜੋੜੋ_F32(rsum, prod);
    }

    *s = hvx_vec_get_f32(hvx_vec_reduce_ਜੋੜ_f32(rsum));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flat_vec_dot_f32_f32_aa_2x1(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx1,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx1;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y  = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy0;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_ਸਦਿਸ਼ rsum0 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ rsum1 = Q6_V_vzero();

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ y_sf = y[i];
        HVX_ਸਦਿਸ਼ prod0 = HVX_OP_MUL_F32(x0[i], y_sf);
        HVX_ਸਦਿਸ਼ prod1 = HVX_OP_MUL_F32(x1[i], y_sf);
        rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, prod0);
        rsum1 = HVX_OP_ਜੋੜੋ_F32(rsum1, prod1);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ਸਦਿਸ਼ y_sf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_ਸਦਿਸ਼ x0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_ਸਦਿਸ਼ x1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_ਸਦਿਸ਼ prod0 = HVX_OP_MUL_F32(x0_sf, y_sf);
        HVX_ਸਦਿਸ਼ prod1 = HVX_OP_MUL_F32(x1_sf, y_sf);
        rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, prod0);
        rsum1 = HVX_OP_ਜੋੜੋ_F32(rsum1, prod1);
    }

    HVX_ਸਦਿਸ਼ rsum = hvx_vec_reduce_ਜੋੜ_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flat_vec_dot_f32_f32_aa_2x2(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s1,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx1,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy1) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx1;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy1;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_ਸਦਿਸ਼ r0_c0_ਜੋੜ = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ r0_c1_ਜੋੜ = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ r1_c0_ਜੋੜ = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ r1_c1_ਜੋੜ = Q6_V_vzero();

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ r0_sf = x0[i];
        HVX_ਸਦਿਸ਼ r1_sf = x1[i];
        HVX_ਸਦਿਸ਼ c0_sf = y0[i];
        HVX_ਸਦਿਸ਼ c1_sf = y1[i];

        r0_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c0_ਜੋੜ, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c1_ਜੋੜ, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c0_ਜੋੜ, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c1_ਜੋੜ, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 4);

        HVX_ਸਦਿਸ਼ r0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_ਸਦਿਸ਼ r1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_ਸਦਿਸ਼ c0_sf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_ਸਦਿਸ਼ c1_sf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c0_ਜੋੜ, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c1_ਜੋੜ, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c0_ਜੋੜ, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c1_ਜੋੜ, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    // Reduce and store results
    HVX_ਸਦਿਸ਼ r0_r1_c0_ਜੋੜ = hvx_vec_reduce_ਜੋੜ_f32x2(r0_c0_ਜੋੜ, r1_c0_ਜੋੜ);
    HVX_ਸਦਿਸ਼ r0_r1_c1_ਜੋੜ = hvx_vec_reduce_ਜੋੜ_f32x2(r0_c1_ਜੋੜ, r1_c1_ਜੋੜ);

    hvx_vec_store_u(s0, 8, r0_r1_c0_ਜੋੜ);
    hvx_vec_store_u(s1, 8, r0_r1_c1_ਜੋੜ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flat_vec_dot_bf16_f32_aa_1x1(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_ਸਦਿਸ਼ rsum0 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v_bf16 = x[i];
        HVX_ਸਦਿਸ਼_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_ਸਦਿਸ਼ w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_ਸਦਿਸ਼ w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y[2 * i + 0]));
        rsum1 = HVX_OP_ਜੋੜੋ_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y[2 * i + 1]));
    }

    ਜੇ (rem) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_ਸਦਿਸ਼ v_bf16 = Q6_V_vand_QV(bmask, x[i]);
        HVX_ਸਦਿਸ਼_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_ਸਦਿਸ਼ w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_ਸਦਿਸ਼ w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        ਜੇ (rem <= 32) {
            HVX_ਸਦਿਸ਼_Pred a_ਨਕਾਬ = Q6_Q_vsetq2_R(rem * 4);
            HVX_ਸਦਿਸ਼ y0 = Q6_V_vand_QV(a_ਨਕਾਬ, y[2 * i + 0]);
            rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));
        } ਨਹੀਂਤਾਂ {
            HVX_ਸਦਿਸ਼ y0 = y[2 * i + 0];
            rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));

            HVX_ਸਦਿਸ਼_Pred a_ਨਕਾਬ = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_ਸਦਿਸ਼ y1 = Q6_V_vand_QV(a_ਨਕਾਬ, y[2 * i + 1]);
            rsum1 = HVX_OP_ਜੋੜੋ_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y1));
        }
    }

    HVX_ਸਦਿਸ਼ rsum = HVX_OP_ਜੋੜੋ_F32(rsum0, rsum1);
    *s = hvx_vec_get_f32(hvx_vec_reduce_ਜੋੜ_f32(rsum));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flat_vec_dot_bf16_f32_aa_2x1(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx1,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx1;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y  = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy0;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_ਸਦਿਸ਼ rsum0 = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_ਸਦਿਸ਼_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_ਸਦਿਸ਼ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_ਸਦਿਸ਼ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_ਸਦਿਸ਼ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_ਸਦਿਸ਼ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_ਸਦਿਸ਼ y0 = y[2 * i + 0];
        HVX_ਸਦਿਸ਼ y1 = y[2 * i + 1];

        rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
        rsum1 = HVX_OP_ਜੋੜੋ_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
        rsum1 = HVX_OP_ਜੋੜੋ_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
    }

    ਜੇ (rem) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_ਸਦਿਸ਼_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_ਸਦਿਸ਼_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_ਸਦਿਸ਼ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_ਸਦਿਸ਼ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_ਸਦਿਸ਼ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_ਸਦਿਸ਼ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        ਜੇ (rem <= 32) {
            HVX_ਸਦਿਸ਼_Pred a_ਨਕਾਬ = Q6_Q_vsetq2_R(rem * 4);
            HVX_ਸਦਿਸ਼ y0 = Q6_V_vand_QV(a_ਨਕਾਬ, y[2 * i + 0]);
            rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_ਜੋੜੋ_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        } ਨਹੀਂਤਾਂ {
            HVX_ਸਦਿਸ਼ y0 = y[2 * i + 0];
            rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_ਜੋੜੋ_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));

            HVX_ਸਦਿਸ਼_Pred a_ਨਕਾਬ = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_ਸਦਿਸ਼ y1 = Q6_V_vand_QV(a_ਨਕਾਬ, y[2 * i + 1]);
            rsum0 = HVX_OP_ਜੋੜੋ_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
            rsum1 = HVX_OP_ਜੋੜੋ_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
        }
    }

    HVX_ਸਦਿਸ਼ rsum = hvx_vec_reduce_ਜੋੜ_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flat_vec_dot_bf16_f32_aa_2x2(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s1,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx1,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy1) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx1;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy1;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_ਸਦਿਸ਼ r0_c0_ਜੋੜ = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ r0_c1_ਜੋੜ = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ r1_c0_ਜੋੜ = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ r1_c1_ਜੋੜ = Q6_V_vzero();

    uint32_t i = 0;
    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_ਸਦਿਸ਼_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_ਸਦਿਸ਼ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_ਸਦਿਸ਼ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_ਸਦਿਸ਼ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_ਸਦਿਸ਼ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_ਸਦਿਸ਼ c0_0 = y0[2 * i + 0];
        HVX_ਸਦਿਸ਼ c0_1 = y0[2 * i + 1];
        HVX_ਸਦਿਸ਼ c1_0 = y1[2 * i + 0];
        HVX_ਸਦਿਸ਼ c1_1 = y1[2 * i + 1];

        r0_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c0_ਜੋੜ, HVX_OP_MUL_F32(w00_f32, c0_0));
        r0_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c1_ਜੋੜ, HVX_OP_MUL_F32(w00_f32, c1_0));
        r1_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c0_ਜੋੜ, HVX_OP_MUL_F32(w10_f32, c0_0));
        r1_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c1_ਜੋੜ, HVX_OP_MUL_F32(w10_f32, c1_0));

        r0_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c0_ਜੋੜ, HVX_OP_MUL_F32(w01_f32, c0_1));
        r0_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c1_ਜੋੜ, HVX_OP_MUL_F32(w01_f32, c1_1));
        r1_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c0_ਜੋੜ, HVX_OP_MUL_F32(w11_f32, c0_1));
        r1_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c1_ਜੋੜ, HVX_OP_MUL_F32(w11_f32, c1_1));
    }

    ਜੇ (rem) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_ਸਦਿਸ਼_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_ਸਦਿਸ਼_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_ਸਦਿਸ਼ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_ਸਦਿਸ਼ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_ਸਦਿਸ਼ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_ਸਦਿਸ਼ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        ਜੇ (rem <= 32) {
            HVX_ਸਦਿਸ਼_Pred a_ਨਕਾਬ = Q6_Q_vsetq2_R(rem * 4);
            HVX_ਸਦਿਸ਼ c0_0 = Q6_V_vand_QV(a_ਨਕਾਬ, y0[2 * i + 0]);
            HVX_ਸਦਿਸ਼ c1_0 = Q6_V_vand_QV(a_ਨਕਾਬ, y1[2 * i + 0]);
            r0_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c0_ਜੋੜ, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c1_ਜੋੜ, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c0_ਜੋੜ, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c1_ਜੋੜ, HVX_OP_MUL_F32(w10_f32, c1_0));
        } ਨਹੀਂਤਾਂ {
            HVX_ਸਦਿਸ਼ c0_0 = y0[2 * i + 0];
            HVX_ਸਦਿਸ਼ c1_0 = y1[2 * i + 0];
            r0_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c0_ਜੋੜ, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c1_ਜੋੜ, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c0_ਜੋੜ, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c1_ਜੋੜ, HVX_OP_MUL_F32(w10_f32, c1_0));

            HVX_ਸਦਿਸ਼_Pred a_ਨਕਾਬ = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_ਸਦਿਸ਼ c0_1 = Q6_V_vand_QV(a_ਨਕਾਬ, y0[2 * i + 1]);
            HVX_ਸਦਿਸ਼ c1_1 = Q6_V_vand_QV(a_ਨਕਾਬ, y1[2 * i + 1]);
            r0_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c0_ਜੋੜ, HVX_OP_MUL_F32(w01_f32, c0_1));
            r0_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r0_c1_ਜੋੜ, HVX_OP_MUL_F32(w01_f32, c1_1));
            r1_c0_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c0_ਜੋੜ, HVX_OP_MUL_F32(w11_f32, c0_1));
            r1_c1_ਜੋੜ = HVX_OP_ਜੋੜੋ_F32(r1_c1_ਜੋੜ, HVX_OP_MUL_F32(w11_f32, c1_1));
        }
    }

    HVX_ਸਦਿਸ਼ r0_r1_c0_ਜੋੜ = hvx_vec_reduce_ਜੋੜ_f32x2(r0_c0_ਜੋੜ, r1_c0_ਜੋੜ);
    HVX_ਸਦਿਸ਼ r0_r1_c1_ਜੋੜ = hvx_vec_reduce_ਜੋੜ_f32x2(r0_c1_ਜੋੜ, r1_c1_ਜੋੜ);

    hvx_vec_store_u(s0, 8, r0_r1_c0_ਜੋੜ);
    hvx_vec_store_u(s1, 8, r0_r1_c1_ਜੋੜ);
}

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_ਜੋੜੋ_F32
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ HVX_OP_MUL_F32

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flat_vec_dot_f16_f16_aa_1x1(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ਸਦਿਸ਼_Pair rsum_p = Q6_W_vzero();

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x[i], y[i]);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ਸਦਿਸ਼ x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_ਸਦਿਸ਼ y_hf = Q6_V_vand_QV(bmask, y[i]);
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_ਸਦਿਸ਼ rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));
    hvx_vec_store_u(s, 4, hvx_vec_reduce_ਜੋੜ_f32(rsum));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flat_vec_dot_f16_f16_aa_2x1(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx1,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx1;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y  = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy0;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_ਸਦਿਸ਼_Pair rsum0_p = Q6_W_vzero();
    HVX_ਸਦਿਸ਼_Pair rsum1_p = Q6_W_vzero();

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ y_hf = y[i];
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0[i], y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1[i], y_hf);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ਸਦਿਸ਼ y_hf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_ਸਦਿਸ਼ x0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_ਸਦਿਸ਼ x1_hf = Q6_V_vand_QV(bmask, x1[i]);
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
    }

    HVX_ਸਦਿਸ਼ rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p)));
    HVX_ਸਦਿਸ਼ rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p)));
    HVX_ਸਦਿਸ਼ rsum  = hvx_vec_reduce_ਜੋੜ_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flat_vec_dot_f16_f16_aa_2x2(ਅਚਲ uint32_t n, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s0, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ s1,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vx1,
                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ vy1) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ x1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vx1;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ y1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) vy1;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    // Row sums (sf) - 4 accumulators for 2x2 tile
    HVX_ਸਦਿਸ਼_Pair r0_c0_ਜੋੜ_p = Q6_W_vzero();
    HVX_ਸਦਿਸ਼_Pair r0_c1_ਜੋੜ_p = Q6_W_vzero();
    HVX_ਸਦਿਸ਼_Pair r1_c0_ਜੋੜ_p = Q6_W_vzero();
    HVX_ਸਦਿਸ਼_Pair r1_c1_ਜੋੜ_p = Q6_W_vzero();

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ r0_hf = x0[i];
        HVX_ਸਦਿਸ਼ r1_hf = x1[i];
        HVX_ਸਦਿਸ਼ c0_hf = y0[i];
        HVX_ਸਦਿਸ਼ c1_hf = y1[i];

        // Compute 4 dot products: r0xc0, r0xc1, r1xc0, r1xc1
        r0_c0_ਜੋੜ_p = hvx_vec_mpyacc_f32_f16(r0_c0_ਜੋੜ_p, r0_hf, c0_hf);
        r0_c1_ਜੋੜ_p = hvx_vec_mpyacc_f32_f16(r0_c1_ਜੋੜ_p, r0_hf, c1_hf);
        r1_c0_ਜੋੜ_p = hvx_vec_mpyacc_f32_f16(r1_c0_ਜੋੜ_p, r1_hf, c0_hf);
        r1_c1_ਜੋੜ_p = hvx_vec_mpyacc_f32_f16(r1_c1_ਜੋੜ_p, r1_hf, c1_hf);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 2);

        HVX_ਸਦਿਸ਼ r0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_ਸਦਿਸ਼ r1_hf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_ਸਦਿਸ਼ c0_hf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_ਸਦਿਸ਼ c1_hf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_ਜੋੜ_p = hvx_vec_mpyacc_f32_f16(r0_c0_ਜੋੜ_p, r0_hf, c0_hf);
        r0_c1_ਜੋੜ_p = hvx_vec_mpyacc_f32_f16(r0_c1_ਜੋੜ_p, r0_hf, c1_hf);
        r1_c0_ਜੋੜ_p = hvx_vec_mpyacc_f32_f16(r1_c0_ਜੋੜ_p, r1_hf, c0_hf);
        r1_c1_ਜੋੜ_p = hvx_vec_mpyacc_f32_f16(r1_c1_ਜੋੜ_p, r1_hf, c1_hf);
    }

    HVX_ਸਦਿਸ਼ r0_c0_ਜੋੜ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c0_ਜੋੜ_p), Q6_V_hi_W(r0_c0_ਜੋੜ_p)));
    HVX_ਸਦਿਸ਼ r0_c1_ਜੋੜ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c1_ਜੋੜ_p), Q6_V_hi_W(r0_c1_ਜੋੜ_p)));
    HVX_ਸਦਿਸ਼ r1_c0_ਜੋੜ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c0_ਜੋੜ_p), Q6_V_hi_W(r1_c0_ਜੋੜ_p)));
    HVX_ਸਦਿਸ਼ r1_c1_ਜੋੜ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c1_ਜੋੜ_p), Q6_V_hi_W(r1_c1_ਜੋੜ_p)));

    // Reduce and store results
    HVX_ਸਦਿਸ਼ r0_r1_c0_ਜੋੜ = hvx_vec_reduce_ਜੋੜ_f32x2(r0_c0_ਜੋੜ, r1_c0_ਜੋੜ);
    HVX_ਸਦਿਸ਼ r0_r1_c1_ਜੋੜ = hvx_vec_reduce_ਜੋੜ_f32x2(r0_c1_ਜੋੜ, r1_c1_ਜੋੜ);

    hvx_vec_store_u(&s0[0], 8, r0_r1_c0_ਜੋੜ);  // row0,col0 row1,col0
    hvx_vec_store_u(&s1[0], 8, r0_r1_c1_ਜੋੜ);  // row0,col1 row1,col1
}



ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_tensor_ਜੋੜੋ_f32_grid(
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst,
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src2,
    uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ,
    uint32_t ਸਿਰਾ_ਪੰਕਤੀ,
    uint32_t ਸ਼ੁਰੂ_ਖਾਨਾ,
    uint32_t ਸਿਰਾ_ਖਾਨਾ,
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne11_12,
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne11
) {
    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਸਿਰਾ_ਪੰਕਤੀ || ਸ਼ੁਰੂ_ਖਾਨਾ >= ਸਿਰਾ_ਖਾਨਾ) ਵਾਪਸ;
    ਅਚਲ uint32_t nb1 = dst->nb[1]; // row stride in bytes

    ਅਚਲ uint32_t ne11 = dst->ne[1];
    ਅਚਲ uint32_t ne12 = dst->ne[2];
    ਅਚਲ uint32_t ne11_12 = ne11 * ne12;

    ਅਚਲ bool is_broadcast1 = (src2->ne[1] == 1);
    ਅਚਲ bool is_broadcast2 = (src2->ne[2] == 1);
    ਅਚਲ bool is_broadcast3 = (src2->ne[3] == 1);

    ਲਈ (uint32_t r = ਸ਼ੁਰੂ_ਪੰਕਤੀ; r < ਸਿਰਾ_ਪੰਕਤੀ; r++) {
        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((uint8_t *) dst->ਅੰਕੜੇ + (size_t) r * nb1);

        uint32_t i13 = fastdiv(r, div_ne11_12);
        uint32_t i12 = fastdiv(r - i13 * ne11_12, div_ne11);
        uint32_t i11 = r - i13 * ne11_12 - i12 * ne11;

        uint32_t i23 = is_broadcast3 ? 0 : i13;
        uint32_t i22 = is_broadcast2 ? 0 : i12;
        uint32_t i21 = is_broadcast1 ? 0 : i11;

        ਅਚਲ ਦਸ਼ਮਲਵ * src2_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src2->ਅੰਕੜੇ +
                                  (size_t) i21 * src2->nb[1] + (size_t) i22 * src2->nb[2] + (size_t) i23 * src2->nb[3]);

        ਦਸ਼ਮਲਵ * dst_ptr = &dst_ਪੰਕਤੀ[ਸ਼ੁਰੂ_ਖਾਨਾ];
        ਅਚਲ ਦਸ਼ਮਲਵ * src2_ptr = &src2_ਪੰਕਤੀ[ਸ਼ੁਰੂ_ਖਾਨਾ];
        ਪੂਰਨ ਬਚਿਆ = ਸਿਰਾ_ਖਾਨਾ - ਸ਼ੁਰੂ_ਖਾਨਾ;
        ਜਦੋਂ (ਬਚਿਆ >= 32) {
            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vmemu(dst_ptr);
            HVX_ਸਦਿਸ਼ v_z   = hvx_vmemu(src2_ptr);
            hvx_vmemu(dst_ptr) = hvx_vec_ਜੋੜੋ_f32_f32(v_ਬਾਹਰਲਾ, v_z);
            dst_ptr += 32;
            src2_ptr += 32;
            ਬਚਿਆ -= 32;
        }
        ਜੇ (ਬਚਿਆ > 0) {
            HVX_ਸਦਿਸ਼ v_ਬਾਹਰਲਾ = hvx_vmemu(dst_ptr);
            HVX_ਸਦਿਸ਼ v_z   = hvx_vmemu(src2_ptr);
            hvx_vec_store_u(dst_ptr, ਬਚਿਆ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_ਜੋੜੋ_f32_f32(v_ਬਾਹਰਲਾ, v_z));
        }
    }
}

#ਜੇ_ਅੰਤ // HVX_MM_KERNELS_FLOAT_H
