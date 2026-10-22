#నిర్వచితం_కాకపోతే HVX_FA_KERNELS_H
#నిర్వచించు HVX_FA_KERNELS_H

#అన్తర్భావ <assert.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ "hvx-utils.h"

// Little inner kernels for HVX

#ఒకవేళ __HVX_ARCH__ < 79
#నిర్వచించు HVX_OP_కూడు_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#నిర్వచించు HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#నిర్వచించు HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#లేకపోతే
#నిర్వచించు HVX_OP_కూడు_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#నిర్వచించు HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#నిర్వచించు HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#అయితే_ముగింపు

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
స్థిర __attribute__((వాడని_2)) __attribute__((noinline)) శూన్య hvx_vec_f32_గమ్యం_f16_a(శూన్య *ptr, HVX_సదిశ v0, HVX_సదిశ v1)
{
    *(HVX_సదిశ *) ptr = hvx_vec_f32_గమ్యం_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
స్థిర అంతఃవరుస శూన్య hvx_dot_f16_f16_aa(దశాంశం * పరిమిత r, స్థిరరాశి శూన్య * పరిమిత x, స్థిరరాశి శూన్య * పరిమిత y, గుర్తులేని పూర్ణాంకం n, దశాంశం s) {
    స్థిరరాశి HVX_సదిశ * పరిమిత vx = (స్థిరరాశి HVX_సదిశ * పరిమిత) x; // fp16
    స్థిరరాశి HVX_సదిశ * పరిమిత vy = (స్థిరరాశి HVX_సదిశ * పరిమిత) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_సదిశ_Pair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #సూచన unroll(4)
    కొరకు (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_సదిశ y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_సదిశ x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_సదిశ rsum = HVX_OP_కూడు_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_కూడిక_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_dot_f16_f16_aa_rx4(స్థిరరాశి శూన్య * పరిమిత y,
                                                స్థిరరాశి uint8_t * పరిమిత x,
                                                స్థిరరాశి size_t stride_x,
                                                స్థిరరాశి size_t nvec,
                                                స్థిరరాశి size_t nloe) {
    స్థిరరాశి HVX_సదిశ * పరిమిత vx0 = (స్థిరరాశి HVX_సదిశ * పరిమిత) x;                   // fp16
    స్థిరరాశి HVX_సదిశ * పరిమిత vx1 = (స్థిరరాశి HVX_సదిశ * పరిమిత) (x + stride_x);      // fp16
    స్థిరరాశి HVX_సదిశ * పరిమిత vx2 = (స్థిరరాశి HVX_సదిశ * పరిమిత) (x + stride_x * 2);  // fp16
    స్థిరరాశి HVX_సదిశ * పరిమిత vx3 = (స్థిరరాశి HVX_సదిశ * పరిమిత) (x + stride_x * 3);  // fp16
    స్థిరరాశి HVX_సదిశ * పరిమిత vy  = (స్థిరరాశి HVX_సదిశ * పరిమిత) y;                   // fp16

    HVX_సదిశ_Pair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_సదిశ_Pair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_సదిశ_Pair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_సదిశ_Pair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    కొరకు (i = 0; i < nvec; i++) {
        HVX_సదిశ y_hf  = vy[i];
        HVX_సదిశ x0_hf = vx0[i];
        HVX_సదిశ x1_hf = vx1[i];
        HVX_సదిశ x2_hf = vx2[i];
        HVX_సదిశ x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    ఒకవేళ (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_సదిశ     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_సదిశ     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_సదిశ     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_సదిశ     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_సదిశ     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_సదిశ rsum0 = HVX_OP_కూడు_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_సదిశ rsum1 = HVX_OP_కూడు_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_సదిశ rsum2 = HVX_OP_కూడు_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_సదిశ rsum3 = HVX_OP_కూడు_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_సదిశ_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    తిరిగి hvx_vec_reduce_కూడిక_f32x4(rsum0123);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_dot_f16_f16_aa_rx32(స్థిరరాశి శూన్య * పరిమిత y,
                                                 స్థిరరాశి uint8_t * పరిమిత x,
                                                 స్థిరరాశి size_t stride_x,
                                                 స్థిరరాశి size_t n,
                                                 దశాంశం        s) {

    స్థిరరాశి size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    స్థిరరాశి size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_సదిశ   sums = Q6_V_vzero();
    స్థిరరాశి size_t stride_x_4 = stride_x * 4;
    కొరకు (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_సదిశ     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_సదిశ_Pred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    తిరిగి HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// MAD: y (F32) += x (F16) * s (F16)
స్థిర అంతఃవరుస శూన్య hvx_mad_f32_f16_aa(దశాంశం * పరిమిత y, స్థిరరాశి శూన్య * పరిమిత x, స్థిరరాశి __fp16 * పరిమిత s, uint32_t n) {
    స్థిరరాశి HVX_సదిశ * పరిమిత vx0 = (స్థిరరాశి HVX_సదిశ *) x;

    HVX_సదిశ_Pair * పరిమిత vy_p = (HVX_సదిశ_Pair *) y;
    HVX_సదిశ * పరిమిత vy = (HVX_సదిశ *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_సదిశ S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_సదిశ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ఒకవేళ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ఒకవేళ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
స్థిర అంతఃవరుస శూన్య hvx_mad_f32_f16_aa_rx2(దశాంశం * పరిమిత y, స్థిరరాశి శూన్య * పరిమిత x0, స్థిరరాశి శూన్య * పరిమిత x1,
                                          స్థిరరాశి __fp16 * పరిమిత s0, స్థిరరాశి __fp16 * పరిమిత s1, uint32_t n) {
    స్థిరరాశి HVX_సదిశ * పరిమిత vx0 = (స్థిరరాశి HVX_సదిశ *) x0;
    స్థిరరాశి HVX_సదిశ * పరిమిత vx1 = (స్థిరరాశి HVX_సదిశ *) x1;

    HVX_సదిశ_Pair * పరిమిత vy_p  = (HVX_సదిశ_Pair *) y;
    HVX_సదిశ * పరిమిత vy        = (HVX_సదిశ *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_సదిశ S0 = hvx_vec_splat_f16(*s0);
    HVX_సదిశ S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_సదిశ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ఒకవేళ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ఒకవేళ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}
స్థిర అంతఃవరుస శూన్య hvx_mad_f32_f16_aa_vec(దశాంశం * పరిమిత y, స్థిరరాశి శూన్య * పరిమిత x, HVX_సదిశ S0, uint32_t n) {
    స్థిరరాశి HVX_సదిశ * పరిమిత vx0 = (స్థిరరాశి HVX_సదిశ *) x;

    HVX_సదిశ_Pair * పరిమిత vy_p = (HVX_సదిశ_Pair *) y;
    HVX_సదిశ * పరిమిత vy = (HVX_సదిశ *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    uint32_t i = 0;

    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_సదిశ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ఒకవేళ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ఒకవేళ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

స్థిర అంతఃవరుస శూన్య hvx_mad_f32_f16_aa_rx2_vec(దశాంశం * పరిమిత y, స్థిరరాశి శూన్య * పరిమిత x0, స్థిరరాశి శూన్య * పరిమిత x1,
                                          HVX_సదిశ S0, HVX_సదిశ S1, uint32_t n) {
    స్థిరరాశి HVX_సదిశ * పరిమిత vx0 = (స్థిరరాశి HVX_సదిశ *) x0;
    స్థిరరాశి HVX_సదిశ * పరిమిత vx1 = (స్థిరరాశి HVX_సదిశ *) x1;

    HVX_సదిశ_Pair * పరిమిత vy_p  = (HVX_సదిశ_Pair *) y;
    HVX_సదిశ * పరిమిత vy        = (HVX_సదిశ *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    uint32_t i = 0;

    #సూచన unroll(2)
    కొరకు (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_సదిశ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ఒకవేళ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ఒకవేళ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

స్థిర అంతఃవరుస శూన్య hvx_scale_vec_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, స్థిరరాశి uint32_t n, HVX_సదిశ vs) {
    ప్రతిజ్ఞా((size_t) dst % 128 == 0);
    ప్రతిజ్ఞా((size_t) src % 128 == 0);

    స్థిరరాశి HVX_సదిశ * పరిమిత vsrc = (స్థిరరాశి HVX_సదిశ * పరిమిత) src;
    HVX_సదిశ * పరిమిత vdst       = (HVX_సదిశ * పరిమిత) dst;

    స్థిరరాశి uint32_t nvec = n / VLEN_FP32;
    స్థిరరాశి uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #సూచన unroll(4)
    కొరకు (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    ఒకవేళ (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * పరిమాణం(దశాంశం), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

#అయితే_ముగింపు /* HVX_FA_KERNELS_H */
