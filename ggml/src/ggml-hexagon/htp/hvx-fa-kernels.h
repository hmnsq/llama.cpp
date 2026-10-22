#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_FA_KERNELS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_FA_KERNELS_H

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

// Little inner kernels for HVX

#ಆದರೆ __HVX_ARCH__ < 79
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_ಕೂಡಿಸು_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_ಕೂಡಿಸು_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#ಆದರೆ_ಅಂತ್ಯ

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
ಸ್ಥಿರ __attribute__((ಬಳಸದ_2)) __attribute__((noinline)) ಶೂನ್ಯ hvx_vec_f32_ಗಮ್ಯ_f16_a(ಶೂನ್ಯ *ptr, HVX_ಸದಿಶ v0, HVX_ಸದಿಶ v1)
{
    *(HVX_ಸದಿಶ *) ptr = hvx_vec_f32_ಗಮ್ಯ_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_dot_f16_f16_aa(ದಶಮಾಂಶ * ಸೀಮಿತ r, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ x, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ y, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ s) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ) x; // fp16
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vy = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ಸದಿಶ_Pair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ಸದಿಶ y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_ಸದಿಶ x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_ಸದಿಶ rsum = HVX_OP_ಕೂಡಿಸು_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_ಮೊತ್ತ_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_dot_f16_f16_aa_rx4(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ y,
                                                ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ x,
                                                ಸ್ಥಿರಾಂಕ size_t stride_x,
                                                ಸ್ಥಿರಾಂಕ size_t nvec,
                                                ಸ್ಥಿರಾಂಕ size_t nloe) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ) x;                   // fp16
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ) (x + stride_x);      // fp16
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx2 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ) (x + stride_x * 2);  // fp16
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx3 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ) (x + stride_x * 3);  // fp16
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vy  = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ) y;                   // fp16

    HVX_ಸದಿಶ_Pair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_ಸದಿಶ_Pair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_ಸದಿಶ_Pair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_ಸದಿಶ_Pair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    ಗಾಗಿ (i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ y_hf  = vy[i];
        HVX_ಸದಿಶ x0_hf = vx0[i];
        HVX_ಸದಿಶ x1_hf = vx1[i];
        HVX_ಸದಿಶ x2_hf = vx2[i];
        HVX_ಸದಿಶ x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    ಆದರೆ (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ಸದಿಶ     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_ಸದಿಶ     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_ಸದಿಶ     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_ಸದಿಶ     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_ಸದಿಶ     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_ಸದಿಶ rsum0 = HVX_OP_ಕೂಡಿಸು_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_ಸದಿಶ rsum1 = HVX_OP_ಕೂಡಿಸು_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_ಸದಿಶ rsum2 = HVX_OP_ಕೂಡಿಸು_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_ಸದಿಶ rsum3 = HVX_OP_ಕೂಡಿಸು_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_ಸದಿಶ_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    ಹಿಂತಿರುಗಿ hvx_vec_reduce_ಮೊತ್ತ_f32x4(rsum0123);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_dot_f16_f16_aa_rx32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ y,
                                                 ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ x,
                                                 ಸ್ಥಿರಾಂಕ size_t stride_x,
                                                 ಸ್ಥಿರಾಂಕ size_t n,
                                                 ದಶಮಾಂಶ        s) {

    ಸ್ಥಿರಾಂಕ size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    ಸ್ಥಿರಾಂಕ size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ಸದಿಶ   sums = Q6_V_vzero();
    ಸ್ಥಿರಾಂಕ size_t stride_x_4 = stride_x * 4;
    ಗಾಗಿ (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_ಸದಿಶ     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_ಸದಿಶ_Pred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    ಹಿಂತಿರುಗಿ HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// MAD: y (F32) += x (F16) * s (F16)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_mad_f32_f16_aa(ದಶಮಾಂಶ * ಸೀಮಿತ y, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ x, ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ s, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) x;

    HVX_ಸದಿಶ_Pair * ಸೀಮಿತ vy_p = (HVX_ಸದಿಶ_Pair *) y;
    HVX_ಸದಿಶ * ಸೀಮಿತ vy = (HVX_ಸದಿಶ *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ಸದಿಶ S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_ಸದಿಶ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ಆದರೆ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ಆದರೆ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_mad_f32_f16_aa_rx2(ದಶಮಾಂಶ * ಸೀಮಿತ y, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ x0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ x1,
                                          ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ s0, ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ s1, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) x0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) x1;

    HVX_ಸದಿಶ_Pair * ಸೀಮಿತ vy_p  = (HVX_ಸದಿಶ_Pair *) y;
    HVX_ಸದಿಶ * ಸೀಮಿತ vy        = (HVX_ಸದಿಶ *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_ಸದಿಶ S0 = hvx_vec_splat_f16(*s0);
    HVX_ಸದಿಶ S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_ಸದಿಶ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ಆದರೆ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ಆದರೆ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_mad_f32_f16_aa_vec(ದಶಮಾಂಶ * ಸೀಮಿತ y, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ x, HVX_ಸದಿಶ S0, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) x;

    HVX_ಸದಿಶ_Pair * ಸೀಮಿತ vy_p = (HVX_ಸದಿಶ_Pair *) y;
    HVX_ಸದಿಶ * ಸೀಮಿತ vy = (HVX_ಸದಿಶ *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_ಸದಿಶ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ಆದರೆ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ಆದರೆ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_mad_f32_f16_aa_rx2_vec(ದಶಮಾಂಶ * ಸೀಮಿತ y, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ x0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ x1,
                                          HVX_ಸದಿಶ S0, HVX_ಸದಿಶ S1, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) x0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vx1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) x1;

    HVX_ಸದಿಶ_Pair * ಸೀಮಿತ vy_p  = (HVX_ಸದಿಶ_Pair *) y;
    HVX_ಸದಿಶ * ಸೀಮಿತ vy        = (HVX_ಸದಿಶ *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    uint32_t i = 0;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_ಸದಿಶ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ಆದರೆ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ಆದರೆ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_scale_vec_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ uint32_t n, HVX_ಸದಿಶ vs) {
    ಪ್ರತಿಜ್ಞಾ((size_t) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((size_t) src % 128 == 0);

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vsrc = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ vdst       = (HVX_ಸದಿಶ * ಸೀಮಿತ) dst;

    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    ಆದರೆ (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_FA_KERNELS_H */
