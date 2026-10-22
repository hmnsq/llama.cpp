#నిర్వచితం_కాకపోతే HVX_MM_KERNELS_FLOAT_H
#నిర్వచించు HVX_MM_KERNELS_FLOAT_H

#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "htp-tensor.h"

// Float activation copy/quantization kernels (DDR -> VTCM)

స్థిర అంతఃవరుస శూన్య quantize_f32_f32_kernel(
    స్థిరరాశి uint8_t * పరిమిత src_దత్తాంశం,
    uint8_t * పరిమిత dst_దత్తాంశం,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (శూన్య) tmp_దత్తాంశం;
    స్థిరరాశి size_t src_అడ్డువరుస_కొలత = ne0 * పరిమాణం(దశాంశం);
    కొరకు (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_దత్తాంశం, src_అడ్డువరుస_కొలత, src_stride, 2);
        hvx_copy_f32_au(dst_దత్తాంశం, src_దత్తాంశం, ne0);

        dst_దత్తాంశం += dst_stride;
        src_దత్తాంశం += src_stride;
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f32_f16_kernel(
    స్థిరరాశి uint8_t * పరిమిత src_దత్తాంశం,
    uint8_t * పరిమిత dst_దత్తాంశం,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (శూన్య) tmp_దత్తాంశం;
    స్థిరరాశి size_t src_అడ్డువరుస_కొలత = ne0 * పరిమాణం(దశాంశం);
    కొరకు (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_దత్తాంశం, src_అడ్డువరుస_కొలత, src_stride, 2);
        hvx_copy_f16_f32_au(dst_దత్తాంశం, src_దత్తాంశం, ne0);

        dst_దత్తాంశం += dst_stride;
        src_దత్తాంశం += src_stride;
    }
}

స్థిర అంతఃవరుస శూన్య quantize_f16_f16_kernel(
    స్థిరరాశి uint8_t * పరిమిత src_దత్తాంశం,
    uint8_t * పరిమిత dst_దత్తాంశం,
    uint8_t * పరిమిత tmp_దత్తాంశం,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (శూన్య) tmp_దత్తాంశం;
    స్థిరరాశి size_t src_అడ్డువరుస_కొలత = ne0 * పరిమాణం(దశాంశం);
    కొరకు (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_దత్తాంశం, src_అడ్డువరుస_కొలత, src_stride, 2);
        hvx_copy_f16_au(dst_దత్తాంశం, src_దత్తాంశం, ne0);

        dst_దత్తాంశం += dst_stride;
        src_దత్తాంశం += src_stride;
    }
}

// Float dot product kernels (HVX)

#ఒకవేళ __HVX_ARCH__ < 79
#నిర్వచించు HVX_OP_కూడు_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#నిర్వచించు HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#లేకపోతే
#నిర్వచించు HVX_OP_కూడు_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#నిర్వచించు HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#అయితే_ముగింపు

స్థిర అంతఃవరుస శూన్య flat_vec_dot_f32_f32_aa_1x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy) {
    స్థిరరాశి HVX_సదిశ * పరిమిత x = (స్థిరరాశి HVX_సదిశ *) vx;
    స్థిరరాశి HVX_సదిశ * పరిమిత y = (స్థిరరాశి HVX_సదిశ *) vy;

    uint32_t nvec = n / VLEN_FP32; // num full fp32 hvx vectors
    uint32_t nloe = n % VLEN_FP32; // leftover elements

    HVX_సదిశ rsum = Q6_V_vzero();

    uint32_t i = 0;

    #సూచన unroll(4)
    కొరకు (i = 0; i < nvec; i++) {
        HVX_సదిశ prod = HVX_OP_MUL_F32(x[i], y[i]);
        rsum = HVX_OP_కూడు_F32(rsum, prod);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ x_sf = Q6_V_vand_QV(bmask, x[i]);
        HVX_సదిశ y_sf = Q6_V_vand_QV(bmask, y[i]);
        HVX_సదిశ prod = HVX_OP_MUL_F32(x_sf, y_sf);
        rsum = HVX_OP_కూడు_F32(rsum, prod);
    }

    *s = hvx_vec_get_f32(hvx_vec_reduce_కూడిక_f32(rsum));
}

స్థిర అంతఃవరుస శూన్య flat_vec_dot_f32_f32_aa_2x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0,
                                స్థిరరాశి శూన్య * పరిమిత vx0, స్థిరరాశి శూన్య * పరిమిత vx1,
                                స్థిరరాశి శూన్య * పరిమిత vy0) {
    స్థిరరాశి HVX_సదిశ * పరిమిత x0 = (స్థిరరాశి HVX_సదిశ *) vx0;
    స్థిరరాశి HVX_సదిశ * పరిమిత x1 = (స్థిరరాశి HVX_సదిశ *) vx1;
    స్థిరరాశి HVX_సదిశ * పరిమిత y  = (స్థిరరాశి HVX_సదిశ *) vy0;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_సదిశ rsum0 = Q6_V_vzero();
    HVX_సదిశ rsum1 = Q6_V_vzero();

    uint32_t i = 0;

    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; i++) {
        HVX_సదిశ y_sf = y[i];
        HVX_సదిశ prod0 = HVX_OP_MUL_F32(x0[i], y_sf);
        HVX_సదిశ prod1 = HVX_OP_MUL_F32(x1[i], y_sf);
        rsum0 = HVX_OP_కూడు_F32(rsum0, prod0);
        rsum1 = HVX_OP_కూడు_F32(rsum1, prod1);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ y_sf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_సదిశ x0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_సదిశ x1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_సదిశ prod0 = HVX_OP_MUL_F32(x0_sf, y_sf);
        HVX_సదిశ prod1 = HVX_OP_MUL_F32(x1_sf, y_sf);
        rsum0 = HVX_OP_కూడు_F32(rsum0, prod0);
        rsum1 = HVX_OP_కూడు_F32(rsum1, prod1);
    }

    HVX_సదిశ rsum = hvx_vec_reduce_కూడిక_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

స్థిర అంతఃవరుస శూన్య flat_vec_dot_f32_f32_aa_2x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1,
                                స్థిరరాశి శూన్య * పరిమిత vx0, స్థిరరాశి శూన్య * పరిమిత vx1,
                                స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1) {
    స్థిరరాశి HVX_సదిశ * పరిమిత x0 = (స్థిరరాశి HVX_సదిశ *) vx0;
    స్థిరరాశి HVX_సదిశ * పరిమిత x1 = (స్థిరరాశి HVX_సదిశ *) vx1;
    స్థిరరాశి HVX_సదిశ * పరిమిత y0 = (స్థిరరాశి HVX_సదిశ *) vy0;
    స్థిరరాశి HVX_సదిశ * పరిమిత y1 = (స్థిరరాశి HVX_సదిశ *) vy1;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_సదిశ r0_c0_కూడిక = Q6_V_vzero();
    HVX_సదిశ r0_c1_కూడిక = Q6_V_vzero();
    HVX_సదిశ r1_c0_కూడిక = Q6_V_vzero();
    HVX_సదిశ r1_c1_కూడిక = Q6_V_vzero();

    uint32_t i = 0;

    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; i++) {
        HVX_సదిశ r0_sf = x0[i];
        HVX_సదిశ r1_sf = x1[i];
        HVX_సదిశ c0_sf = y0[i];
        HVX_సదిశ c1_sf = y1[i];

        r0_c0_కూడిక = HVX_OP_కూడు_F32(r0_c0_కూడిక, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_కూడిక = HVX_OP_కూడు_F32(r0_c1_కూడిక, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_కూడిక = HVX_OP_కూడు_F32(r1_c0_కూడిక, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_కూడిక = HVX_OP_కూడు_F32(r1_c1_కూడిక, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);

        HVX_సదిశ r0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_సదిశ r1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_సదిశ c0_sf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_సదిశ c1_sf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_కూడిక = HVX_OP_కూడు_F32(r0_c0_కూడిక, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_కూడిక = HVX_OP_కూడు_F32(r0_c1_కూడిక, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_కూడిక = HVX_OP_కూడు_F32(r1_c0_కూడిక, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_కూడిక = HVX_OP_కూడు_F32(r1_c1_కూడిక, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    // Reduce and store results
    HVX_సదిశ r0_r1_c0_కూడిక = hvx_vec_reduce_కూడిక_f32x2(r0_c0_కూడిక, r1_c0_కూడిక);
    HVX_సదిశ r0_r1_c1_కూడిక = hvx_vec_reduce_కూడిక_f32x2(r0_c1_కూడిక, r1_c1_కూడిక);

    hvx_vec_store_u(s0, 8, r0_r1_c0_కూడిక);
    hvx_vec_store_u(s1, 8, r0_r1_c1_కూడిక);
}

స్థిర అంతఃవరుస శూన్య flat_vec_dot_bf16_f32_aa_1x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy) {
    స్థిరరాశి HVX_సదిశ * పరిమిత x = (స్థిరరాశి HVX_సదిశ *) vx;
    స్థిరరాశి HVX_సదిశ * పరిమిత y = (స్థిరరాశి HVX_సదిశ *) vy;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_సదిశ rsum0 = Q6_V_vzero();
    HVX_సదిశ rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; i++) {
        HVX_సదిశ v_bf16 = x[i];
        HVX_సదిశ_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_సదిశ w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_సదిశ w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        rsum0 = HVX_OP_కూడు_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y[2 * i + 0]));
        rsum1 = HVX_OP_కూడు_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y[2 * i + 1]));
    }

    ఒకవేళ (rem) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_సదిశ v_bf16 = Q6_V_vand_QV(bmask, x[i]);
        HVX_సదిశ_Pair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_సదిశ w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_సదిశ w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        ఒకవేళ (rem <= 32) {
            HVX_సదిశ_Pred a_ముసుగు = Q6_Q_vsetq2_R(rem * 4);
            HVX_సదిశ y0 = Q6_V_vand_QV(a_ముసుగు, y[2 * i + 0]);
            rsum0 = HVX_OP_కూడు_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));
        } లేకపోతే {
            HVX_సదిశ y0 = y[2 * i + 0];
            rsum0 = HVX_OP_కూడు_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));

            HVX_సదిశ_Pred a_ముసుగు = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_సదిశ y1 = Q6_V_vand_QV(a_ముసుగు, y[2 * i + 1]);
            rsum1 = HVX_OP_కూడు_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y1));
        }
    }

    HVX_సదిశ rsum = HVX_OP_కూడు_F32(rsum0, rsum1);
    *s = hvx_vec_get_f32(hvx_vec_reduce_కూడిక_f32(rsum));
}

స్థిర అంతఃవరుస శూన్య flat_vec_dot_bf16_f32_aa_2x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0,
                                స్థిరరాశి శూన్య * పరిమిత vx0, స్థిరరాశి శూన్య * పరిమిత vx1,
                                స్థిరరాశి శూన్య * పరిమిత vy0) {
    స్థిరరాశి HVX_సదిశ * పరిమిత x0 = (స్థిరరాశి HVX_సదిశ *) vx0;
    స్థిరరాశి HVX_సదిశ * పరిమిత x1 = (స్థిరరాశి HVX_సదిశ *) vx1;
    స్థిరరాశి HVX_సదిశ * పరిమిత y  = (స్థిరరాశి HVX_సదిశ *) vy0;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_సదిశ rsum0 = Q6_V_vzero();
    HVX_సదిశ rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; i++) {
        HVX_సదిశ_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_సదిశ_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_సదిశ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_సదిశ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_సదిశ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_సదిశ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_సదిశ y0 = y[2 * i + 0];
        HVX_సదిశ y1 = y[2 * i + 1];

        rsum0 = HVX_OP_కూడు_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
        rsum1 = HVX_OP_కూడు_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        rsum0 = HVX_OP_కూడు_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
        rsum1 = HVX_OP_కూడు_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
    }

    ఒకవేళ (rem) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_సదిశ_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_సదిశ_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_సదిశ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_సదిశ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_సదిశ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_సదిశ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        ఒకవేళ (rem <= 32) {
            HVX_సదిశ_Pred a_ముసుగు = Q6_Q_vsetq2_R(rem * 4);
            HVX_సదిశ y0 = Q6_V_vand_QV(a_ముసుగు, y[2 * i + 0]);
            rsum0 = HVX_OP_కూడు_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_కూడు_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        } లేకపోతే {
            HVX_సదిశ y0 = y[2 * i + 0];
            rsum0 = HVX_OP_కూడు_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_కూడు_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));

            HVX_సదిశ_Pred a_ముసుగు = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_సదిశ y1 = Q6_V_vand_QV(a_ముసుగు, y[2 * i + 1]);
            rsum0 = HVX_OP_కూడు_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
            rsum1 = HVX_OP_కూడు_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
        }
    }

    HVX_సదిశ rsum = hvx_vec_reduce_కూడిక_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

స్థిర అంతఃవరుస శూన్య flat_vec_dot_bf16_f32_aa_2x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1,
                                స్థిరరాశి శూన్య * పరిమిత vx0, స్థిరరాశి శూన్య * పరిమిత vx1,
                                స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1) {
    స్థిరరాశి HVX_సదిశ * పరిమిత x0 = (స్థిరరాశి HVX_సదిశ *) vx0;
    స్థిరరాశి HVX_సదిశ * పరిమిత x1 = (స్థిరరాశి HVX_సదిశ *) vx1;
    స్థిరరాశి HVX_సదిశ * పరిమిత y0 = (స్థిరరాశి HVX_సదిశ *) vy0;
    స్థిరరాశి HVX_సదిశ * పరిమిత y1 = (స్థిరరాశి HVX_సదిశ *) vy1;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_సదిశ r0_c0_కూడిక = Q6_V_vzero();
    HVX_సదిశ r0_c1_కూడిక = Q6_V_vzero();
    HVX_సదిశ r1_c0_కూడిక = Q6_V_vzero();
    HVX_సదిశ r1_c1_కూడిక = Q6_V_vzero();

    uint32_t i = 0;
    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; i++) {
        HVX_సదిశ_Pair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_సదిశ_Pair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_సదిశ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_సదిశ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_సదిశ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_సదిశ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_సదిశ c0_0 = y0[2 * i + 0];
        HVX_సదిశ c0_1 = y0[2 * i + 1];
        HVX_సదిశ c1_0 = y1[2 * i + 0];
        HVX_సదిశ c1_1 = y1[2 * i + 1];

        r0_c0_కూడిక = HVX_OP_కూడు_F32(r0_c0_కూడిక, HVX_OP_MUL_F32(w00_f32, c0_0));
        r0_c1_కూడిక = HVX_OP_కూడు_F32(r0_c1_కూడిక, HVX_OP_MUL_F32(w00_f32, c1_0));
        r1_c0_కూడిక = HVX_OP_కూడు_F32(r1_c0_కూడిక, HVX_OP_MUL_F32(w10_f32, c0_0));
        r1_c1_కూడిక = HVX_OP_కూడు_F32(r1_c1_కూడిక, HVX_OP_MUL_F32(w10_f32, c1_0));

        r0_c0_కూడిక = HVX_OP_కూడు_F32(r0_c0_కూడిక, HVX_OP_MUL_F32(w01_f32, c0_1));
        r0_c1_కూడిక = HVX_OP_కూడు_F32(r0_c1_కూడిక, HVX_OP_MUL_F32(w01_f32, c1_1));
        r1_c0_కూడిక = HVX_OP_కూడు_F32(r1_c0_కూడిక, HVX_OP_MUL_F32(w11_f32, c0_1));
        r1_c1_కూడిక = HVX_OP_కూడు_F32(r1_c1_కూడిక, HVX_OP_MUL_F32(w11_f32, c1_1));
    }

    ఒకవేళ (rem) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_సదిశ_Pair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_సదిశ_Pair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_సదిశ w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_సదిశ w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_సదిశ w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_సదిశ w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        ఒకవేళ (rem <= 32) {
            HVX_సదిశ_Pred a_ముసుగు = Q6_Q_vsetq2_R(rem * 4);
            HVX_సదిశ c0_0 = Q6_V_vand_QV(a_ముసుగు, y0[2 * i + 0]);
            HVX_సదిశ c1_0 = Q6_V_vand_QV(a_ముసుగు, y1[2 * i + 0]);
            r0_c0_కూడిక = HVX_OP_కూడు_F32(r0_c0_కూడిక, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_కూడిక = HVX_OP_కూడు_F32(r0_c1_కూడిక, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_కూడిక = HVX_OP_కూడు_F32(r1_c0_కూడిక, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_కూడిక = HVX_OP_కూడు_F32(r1_c1_కూడిక, HVX_OP_MUL_F32(w10_f32, c1_0));
        } లేకపోతే {
            HVX_సదిశ c0_0 = y0[2 * i + 0];
            HVX_సదిశ c1_0 = y1[2 * i + 0];
            r0_c0_కూడిక = HVX_OP_కూడు_F32(r0_c0_కూడిక, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_కూడిక = HVX_OP_కూడు_F32(r0_c1_కూడిక, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_కూడిక = HVX_OP_కూడు_F32(r1_c0_కూడిక, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_కూడిక = HVX_OP_కూడు_F32(r1_c1_కూడిక, HVX_OP_MUL_F32(w10_f32, c1_0));

            HVX_సదిశ_Pred a_ముసుగు = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_సదిశ c0_1 = Q6_V_vand_QV(a_ముసుగు, y0[2 * i + 1]);
            HVX_సదిశ c1_1 = Q6_V_vand_QV(a_ముసుగు, y1[2 * i + 1]);
            r0_c0_కూడిక = HVX_OP_కూడు_F32(r0_c0_కూడిక, HVX_OP_MUL_F32(w01_f32, c0_1));
            r0_c1_కూడిక = HVX_OP_కూడు_F32(r0_c1_కూడిక, HVX_OP_MUL_F32(w01_f32, c1_1));
            r1_c0_కూడిక = HVX_OP_కూడు_F32(r1_c0_కూడిక, HVX_OP_MUL_F32(w11_f32, c0_1));
            r1_c1_కూడిక = HVX_OP_కూడు_F32(r1_c1_కూడిక, HVX_OP_MUL_F32(w11_f32, c1_1));
        }
    }

    HVX_సదిశ r0_r1_c0_కూడిక = hvx_vec_reduce_కూడిక_f32x2(r0_c0_కూడిక, r1_c0_కూడిక);
    HVX_సదిశ r0_r1_c1_కూడిక = hvx_vec_reduce_కూడిక_f32x2(r0_c1_కూడిక, r1_c1_కూడిక);

    hvx_vec_store_u(s0, 8, r0_r1_c0_కూడిక);
    hvx_vec_store_u(s1, 8, r0_r1_c1_కూడిక);
}

#నిర్వచనం_రద్దు HVX_OP_కూడు_F32
#నిర్వచనం_రద్దు HVX_OP_MUL_F32

స్థిర అంతఃవరుస శూన్య flat_vec_dot_f16_f16_aa_1x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s, స్థిరరాశి శూన్య * పరిమిత vx, స్థిరరాశి శూన్య * పరిమిత vy) {
    స్థిరరాశి HVX_సదిశ * పరిమిత x = (స్థిరరాశి HVX_సదిశ *) vx;
    స్థిరరాశి HVX_సదిశ * పరిమిత y = (స్థిరరాశి HVX_సదిశ *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_సదిశ_Pair rsum_p = Q6_W_vzero();

    uint32_t i = 0;

    #సూచన unroll(4)
    కొరకు (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x[i], y[i]);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_సదిశ x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_సదిశ y_hf = Q6_V_vand_QV(bmask, y[i]);
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_సదిశ rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));
    hvx_vec_store_u(s, 4, hvx_vec_reduce_కూడిక_f32(rsum));
}

స్థిర అంతఃవరుస శూన్య flat_vec_dot_f16_f16_aa_2x1(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0,
                                స్థిరరాశి శూన్య * పరిమిత vx0, స్థిరరాశి శూన్య * పరిమిత vx1,
                                స్థిరరాశి శూన్య * పరిమిత vy0) {
    స్థిరరాశి HVX_సదిశ * పరిమిత x0 = (స్థిరరాశి HVX_సదిశ *) vx0;
    స్థిరరాశి HVX_సదిశ * పరిమిత x1 = (స్థిరరాశి HVX_సదిశ *) vx1;
    స్థిరరాశి HVX_సదిశ * పరిమిత y  = (స్థిరరాశి HVX_సదిశ *) vy0;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_సదిశ_Pair rsum0_p = Q6_W_vzero();
    HVX_సదిశ_Pair rsum1_p = Q6_W_vzero();

    uint32_t i = 0;

    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; i++) {
        HVX_సదిశ y_hf = y[i];
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0[i], y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1[i], y_hf);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_సదిశ y_hf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_సదిశ x0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_సదిశ x1_hf = Q6_V_vand_QV(bmask, x1[i]);
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
    }

    HVX_సదిశ rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p)));
    HVX_సదిశ rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p)));
    HVX_సదిశ rsum  = hvx_vec_reduce_కూడిక_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

స్థిర అంతఃవరుస శూన్య flat_vec_dot_f16_f16_aa_2x2(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1,
                                స్థిరరాశి శూన్య * పరిమిత vx0, స్థిరరాశి శూన్య * పరిమిత vx1,
                                స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1) {
    స్థిరరాశి HVX_సదిశ * పరిమిత x0 = (స్థిరరాశి HVX_సదిశ *) vx0;
    స్థిరరాశి HVX_సదిశ * పరిమిత x1 = (స్థిరరాశి HVX_సదిశ *) vx1;
    స్థిరరాశి HVX_సదిశ * పరిమిత y0 = (స్థిరరాశి HVX_సదిశ *) vy0;
    స్థిరరాశి HVX_సదిశ * పరిమిత y1 = (స్థిరరాశి HVX_సదిశ *) vy1;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    // Row sums (sf) - 4 accumulators for 2x2 tile
    HVX_సదిశ_Pair r0_c0_కూడిక_p = Q6_W_vzero();
    HVX_సదిశ_Pair r0_c1_కూడిక_p = Q6_W_vzero();
    HVX_సదిశ_Pair r1_c0_కూడిక_p = Q6_W_vzero();
    HVX_సదిశ_Pair r1_c1_కూడిక_p = Q6_W_vzero();

    uint32_t i = 0;

    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; i++) {
        HVX_సదిశ r0_hf = x0[i];
        HVX_సదిశ r1_hf = x1[i];
        HVX_సదిశ c0_hf = y0[i];
        HVX_సదిశ c1_hf = y1[i];

        // Compute 4 dot products: r0xc0, r0xc1, r1xc0, r1xc1
        r0_c0_కూడిక_p = hvx_vec_mpyacc_f32_f16(r0_c0_కూడిక_p, r0_hf, c0_hf);
        r0_c1_కూడిక_p = hvx_vec_mpyacc_f32_f16(r0_c1_కూడిక_p, r0_hf, c1_hf);
        r1_c0_కూడిక_p = hvx_vec_mpyacc_f32_f16(r1_c0_కూడిక_p, r1_hf, c0_hf);
        r1_c1_కూడిక_p = hvx_vec_mpyacc_f32_f16(r1_c1_కూడిక_p, r1_hf, c1_hf);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);

        HVX_సదిశ r0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_సదిశ r1_hf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_సదిశ c0_hf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_సదిశ c1_hf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_కూడిక_p = hvx_vec_mpyacc_f32_f16(r0_c0_కూడిక_p, r0_hf, c0_hf);
        r0_c1_కూడిక_p = hvx_vec_mpyacc_f32_f16(r0_c1_కూడిక_p, r0_hf, c1_hf);
        r1_c0_కూడిక_p = hvx_vec_mpyacc_f32_f16(r1_c0_కూడిక_p, r1_hf, c0_hf);
        r1_c1_కూడిక_p = hvx_vec_mpyacc_f32_f16(r1_c1_కూడిక_p, r1_hf, c1_hf);
    }

    HVX_సదిశ r0_c0_కూడిక = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c0_కూడిక_p), Q6_V_hi_W(r0_c0_కూడిక_p)));
    HVX_సదిశ r0_c1_కూడిక = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c1_కూడిక_p), Q6_V_hi_W(r0_c1_కూడిక_p)));
    HVX_సదిశ r1_c0_కూడిక = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c0_కూడిక_p), Q6_V_hi_W(r1_c0_కూడిక_p)));
    HVX_సదిశ r1_c1_కూడిక = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c1_కూడిక_p), Q6_V_hi_W(r1_c1_కూడిక_p)));

    // Reduce and store results
    HVX_సదిశ r0_r1_c0_కూడిక = hvx_vec_reduce_కూడిక_f32x2(r0_c0_కూడిక, r1_c0_కూడిక);
    HVX_సదిశ r0_r1_c1_కూడిక = hvx_vec_reduce_కూడిక_f32x2(r0_c1_కూడిక, r1_c1_కూడిక);

    hvx_vec_store_u(&s0[0], 8, r0_r1_c0_కూడిక);  // row0,col0 row1,col0
    hvx_vec_store_u(&s1[0], 8, r0_r1_c1_కూడిక);  // row0,col1 row1,col1
}



స్థిర అంతఃవరుస శూన్య hvx_tensor_కూడు_f32_grid(
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst,
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src2,
    uint32_t ప్రారంభం_అడ్డువరుస,
    uint32_t చివర_అడ్డువరుస,
    uint32_t ప్రారంభం_నిలువు,
    uint32_t చివర_నిలువు,
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne11_12,
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne11
) {
    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= చివర_అడ్డువరుస || ప్రారంభం_నిలువు >= చివర_నిలువు) తిరిగి;
    స్థిరరాశి uint32_t nb1 = dst->nb[1]; // row stride in bytes

    స్థిరరాశి uint32_t ne11 = dst->ne[1];
    స్థిరరాశి uint32_t ne12 = dst->ne[2];
    స్థిరరాశి uint32_t ne11_12 = ne11 * ne12;

    స్థిరరాశి bool is_broadcast1 = (src2->ne[1] == 1);
    స్థిరరాశి bool is_broadcast2 = (src2->ne[2] == 1);
    స్థిరరాశి bool is_broadcast3 = (src2->ne[3] == 1);

    కొరకు (uint32_t r = ప్రారంభం_అడ్డువరుస; r < చివర_అడ్డువరుస; r++) {
        దశాంశం * dst_అడ్డువరుస = (దశాంశం *) ((uint8_t *) dst->దత్తాంశం + (size_t) r * nb1);

        uint32_t i13 = fastdiv(r, div_ne11_12);
        uint32_t i12 = fastdiv(r - i13 * ne11_12, div_ne11);
        uint32_t i11 = r - i13 * ne11_12 - i12 * ne11;

        uint32_t i23 = is_broadcast3 ? 0 : i13;
        uint32_t i22 = is_broadcast2 ? 0 : i12;
        uint32_t i21 = is_broadcast1 ? 0 : i11;

        స్థిరరాశి దశాంశం * src2_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src2->దత్తాంశం +
                                  (size_t) i21 * src2->nb[1] + (size_t) i22 * src2->nb[2] + (size_t) i23 * src2->nb[3]);

        దశాంశం * dst_ptr = &dst_అడ్డువరుస[ప్రారంభం_నిలువు];
        స్థిరరాశి దశాంశం * src2_ptr = &src2_అడ్డువరుస[ప్రారంభం_నిలువు];
        పూర్ణాంకం మిగిలిన = చివర_నిలువు - ప్రారంభం_నిలువు;
        అయితే (మిగిలిన >= 32) {
            HVX_సదిశ v_బహిర్గతం = hvx_vmemu(dst_ptr);
            HVX_సదిశ v_z   = hvx_vmemu(src2_ptr);
            hvx_vmemu(dst_ptr) = hvx_vec_కూడు_f32_f32(v_బహిర్గతం, v_z);
            dst_ptr += 32;
            src2_ptr += 32;
            మిగిలిన -= 32;
        }
        ఒకవేళ (మిగిలిన > 0) {
            HVX_సదిశ v_బహిర్గతం = hvx_vmemu(dst_ptr);
            HVX_సదిశ v_z   = hvx_vmemu(src2_ptr);
            hvx_vec_store_u(dst_ptr, మిగిలిన * పరిమాణం(దశాంశం), hvx_vec_కూడు_f32_f32(v_బహిర్గతం, v_z));
        }
    }
}

#అయితే_ముగింపు // HVX_MM_KERNELS_FLOAT_H
