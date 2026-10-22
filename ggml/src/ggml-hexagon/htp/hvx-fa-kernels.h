#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HVX_FA_KERNELS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_FA_KERNELS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"

// Little inner kernels for HVX

#ਜੇ __HVX_ARCH__ < 79
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_ਜੋੜੋ_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_ਜੋੜੋ_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#ਜੇ_ਅੰਤ

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
ਸਥਿਰ __attribute__((ਅਣਵਰਤਿਆ_2)) __attribute__((noinline)) ਖਾਲੀ hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_a(ਖਾਲੀ *ptr, HVX_ਸਦਿਸ਼ v0, HVX_ਸਦਿਸ਼ v1)
{
    *(HVX_ਸਦਿਸ਼ *) ptr = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_dot_f16_f16_aa(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ r, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ x, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ y, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਦਸ਼ਮਲਵ s) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx = (ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ) x; // fp16
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vy = (ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ਸਦਿਸ਼_Pair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ਸਦਿਸ਼ y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_ਸਦਿਸ਼ x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_ਸਦਿਸ਼ rsum = HVX_OP_ਜੋੜੋ_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_ਜੋੜ_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_dot_f16_f16_aa_rx4(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ y,
                                                ਅਚਲ uint8_t * ਸੀਮਿਤ x,
                                                ਅਚਲ size_t stride_x,
                                                ਅਚਲ size_t nvec,
                                                ਅਚਲ size_t nloe) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx0 = (ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ) x;                   // fp16
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx1 = (ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ) (x + stride_x);      // fp16
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx2 = (ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ) (x + stride_x * 2);  // fp16
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx3 = (ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ) (x + stride_x * 3);  // fp16
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vy  = (ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ) y;                   // fp16

    HVX_ਸਦਿਸ਼_Pair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_ਸਦਿਸ਼_Pair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_ਸਦਿਸ਼_Pair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_ਸਦਿਸ਼_Pair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    ਲਈ (i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ y_hf  = vy[i];
        HVX_ਸਦਿਸ਼ x0_hf = vx0[i];
        HVX_ਸਦਿਸ਼ x1_hf = vx1[i];
        HVX_ਸਦਿਸ਼ x2_hf = vx2[i];
        HVX_ਸਦਿਸ਼ x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    ਜੇ (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_ਸਦਿਸ਼_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ਸਦਿਸ਼     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_ਸਦਿਸ਼     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_ਸਦਿਸ਼     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_ਸਦਿਸ਼     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_ਸਦਿਸ਼     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_ਸਦਿਸ਼ rsum0 = HVX_OP_ਜੋੜੋ_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_ਸਦਿਸ਼ rsum1 = HVX_OP_ਜੋੜੋ_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_ਸਦਿਸ਼ rsum2 = HVX_OP_ਜੋੜੋ_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_ਸਦਿਸ਼ rsum3 = HVX_OP_ਜੋੜੋ_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_ਸਦਿਸ਼_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    ਵਾਪਸ hvx_vec_reduce_ਜੋੜ_f32x4(rsum0123);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_dot_f16_f16_aa_rx32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ y,
                                                 ਅਚਲ uint8_t * ਸੀਮਿਤ x,
                                                 ਅਚਲ size_t stride_x,
                                                 ਅਚਲ size_t n,
                                                 ਦਸ਼ਮਲਵ        s) {

    ਅਚਲ size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    ਅਚਲ size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ਸਦਿਸ਼   sums = Q6_V_vzero();
    ਅਚਲ size_t stride_x_4 = stride_x * 4;
    ਲਈ (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_ਸਦਿਸ਼     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_ਸਦਿਸ਼_Pred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    ਵਾਪਸ HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// MAD: y (F32) += x (F16) * s (F16)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_mad_f32_f16_aa(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ y, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ x, ਅਚਲ __fp16 * ਸੀਮਿਤ s, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) x;

    HVX_ਸਦਿਸ਼_Pair * ਸੀਮਿਤ vy_p = (HVX_ਸਦਿਸ਼_Pair *) y;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vy = (HVX_ਸਦਿਸ਼ *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ਸਦਿਸ਼ S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_ਸਦਿਸ਼ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ਜੇ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ਜੇ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_mad_f32_f16_aa_rx2(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ y, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ x0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ x1,
                                          ਅਚਲ __fp16 * ਸੀਮਿਤ s0, ਅਚਲ __fp16 * ਸੀਮਿਤ s1, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) x0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) x1;

    HVX_ਸਦਿਸ਼_Pair * ਸੀਮਿਤ vy_p  = (HVX_ਸਦਿਸ਼_Pair *) y;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vy        = (HVX_ਸਦਿਸ਼ *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_ਸਦਿਸ਼ S0 = hvx_vec_splat_f16(*s0);
    HVX_ਸਦਿਸ਼ S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_ਸਦਿਸ਼ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ਜੇ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ਜੇ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_mad_f32_f16_aa_vec(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ y, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ x, HVX_ਸਦਿਸ਼ S0, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) x;

    HVX_ਸਦਿਸ਼_Pair * ਸੀਮਿਤ vy_p = (HVX_ਸਦਿਸ਼_Pair *) y;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vy = (HVX_ਸਦਿਸ਼ *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_ਸਦਿਸ਼ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ਜੇ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ਜੇ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_mad_f32_f16_aa_rx2_vec(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ y, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ x0, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ x1,
                                          HVX_ਸਦਿਸ਼ S0, HVX_ਸਦਿਸ਼ S1, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) x0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vx1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) x1;

    HVX_ਸਦਿਸ਼_Pair * ਸੀਮਿਤ vy_p  = (HVX_ਸਦਿਸ਼_Pair *) y;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vy        = (HVX_ਸਦਿਸ਼ *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    uint32_t i = 0;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_ਸਦਿਸ਼ xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        ਜੇ (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        ਜੇ (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_scale_vec_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ uint32_t n, HVX_ਸਦਿਸ਼ vs) {
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((size_t) src % 128 == 0);

    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc = (ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst       = (HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ) dst;

    ਅਚਲ uint32_t nvec = n / VLEN_FP32;
    ਅਚਲ uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #ਹਦਾਇਤ unroll(4)
    ਲਈ (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    ਜੇ (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

#ਜੇ_ਅੰਤ /* HVX_FA_KERNELS_H */
