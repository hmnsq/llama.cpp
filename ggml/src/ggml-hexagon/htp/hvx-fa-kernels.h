#که_نه_تعریف_شوی HVX_FA_KERNELS_H
#تعریف_کړه HVX_FA_KERNELS_H

#ضمن <assert.h>
#ضمن <math.h>
#ضمن "hvx-utils.h"

// Little inner kernels for HVX

#إذا __HVX_ARCH__ < 79
#تعریف_کړه HVX_OP_جمع_کړه_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#تعریف_کړه HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#تعریف_کړه HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#وإلا
#تعریف_کړه HVX_OP_جمع_کړه_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#تعریف_کړه HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#تعریف_کړه HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#د_که_پای

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
ساكن __attribute__((ناکارول_شوی_2)) __attribute__((noinline)) فراغ hvx_vec_f32_منزل_f16_a(فراغ *ptr, HVX_ویکټور v0, HVX_ویکټور v1)
{
    *(HVX_ویکټور *) ptr = hvx_vec_f32_منزل_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
ساكن مضمن فراغ hvx_dot_f16_f16_aa(عائم * محدود r, ثابت فراغ * محدود x, ثابت فراغ * محدود y, بینښه صحيح n, عائم s) {
    ثابت HVX_ویکټور * محدود vx = (ثابت HVX_ویکټور * محدود) x; // fp16
    ثابت HVX_ویکټور * محدود vy = (ثابت HVX_ویکټور * محدود) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ویکټور_Pair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #لارښوونه unroll(4)
    لكل (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    إذا (nloe) {
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ویکټور y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_ویکټور x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_ویکټور rsum = HVX_OP_جمع_کړه_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_مجموعه_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

ساكن مضمن HVX_ویکټور hvx_dot_f16_f16_aa_rx4(ثابت فراغ * محدود y,
                                                ثابت uint8_t * محدود x,
                                                ثابت size_t stride_x,
                                                ثابت size_t nvec,
                                                ثابت size_t nloe) {
    ثابت HVX_ویکټور * محدود vx0 = (ثابت HVX_ویکټور * محدود) x;                   // fp16
    ثابت HVX_ویکټور * محدود vx1 = (ثابت HVX_ویکټور * محدود) (x + stride_x);      // fp16
    ثابت HVX_ویکټور * محدود vx2 = (ثابت HVX_ویکټور * محدود) (x + stride_x * 2);  // fp16
    ثابت HVX_ویکټور * محدود vx3 = (ثابت HVX_ویکټور * محدود) (x + stride_x * 3);  // fp16
    ثابت HVX_ویکټور * محدود vy  = (ثابت HVX_ویکټور * محدود) y;                   // fp16

    HVX_ویکټور_Pair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_ویکټور_Pair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_ویکټور_Pair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_ویکټور_Pair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    لكل (i = 0; i < nvec; i++) {
        HVX_ویکټور y_hf  = vy[i];
        HVX_ویکټور x0_hf = vx0[i];
        HVX_ویکټور x1_hf = vx1[i];
        HVX_ویکټور x2_hf = vx2[i];
        HVX_ویکټور x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    إذا (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_ویکټور_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_ویکټور     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_ویکټور     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_ویکټور     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_ویکټور     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_ویکټور     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_ویکټور rsum0 = HVX_OP_جمع_کړه_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_ویکټور rsum1 = HVX_OP_جمع_کړه_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_ویکټور rsum2 = HVX_OP_جمع_کړه_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_ویکټور rsum3 = HVX_OP_جمع_کړه_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_ویکټور_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    أرجع hvx_vec_reduce_مجموعه_f32x4(rsum0123);
}

ساكن مضمن HVX_ویکټور hvx_dot_f16_f16_aa_rx32(ثابت فراغ * محدود y,
                                                 ثابت uint8_t * محدود x,
                                                 ثابت size_t stride_x,
                                                 ثابت size_t n,
                                                 عائم        s) {

    ثابت size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    ثابت size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ویکټور   sums = Q6_V_vzero();
    ثابت size_t stride_x_4 = stride_x * 4;
    لكل (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_ویکټور     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_ویکټور_Pred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    أرجع HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// MAD: y (F32) += x (F16) * s (F16)
ساكن مضمن فراغ hvx_mad_f32_f16_aa(عائم * محدود y, ثابت فراغ * محدود x, ثابت __fp16 * محدود s, uint32_t n) {
    ثابت HVX_ویکټور * محدود vx0 = (ثابت HVX_ویکټور *) x;

    HVX_ویکټور_Pair * محدود vy_p = (HVX_ویکټور_Pair *) y;
    HVX_ویکټور * محدود vy = (HVX_ویکټور *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_ویکټور S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #لارښوونه unroll(2)
    لكل (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    إذا (nloe) {
        HVX_ویکټور_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_ویکټور xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        إذا (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        إذا (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
ساكن مضمن فراغ hvx_mad_f32_f16_aa_rx2(عائم * محدود y, ثابت فراغ * محدود x0, ثابت فراغ * محدود x1,
                                          ثابت __fp16 * محدود s0, ثابت __fp16 * محدود s1, uint32_t n) {
    ثابت HVX_ویکټور * محدود vx0 = (ثابت HVX_ویکټور *) x0;
    ثابت HVX_ویکټور * محدود vx1 = (ثابت HVX_ویکټور *) x1;

    HVX_ویکټور_Pair * محدود vy_p  = (HVX_ویکټور_Pair *) y;
    HVX_ویکټور * محدود vy        = (HVX_ویکټور *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_ویکټور S0 = hvx_vec_splat_f16(*s0);
    HVX_ویکټور S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #لارښوونه unroll(2)
    لكل (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    إذا (nloe) {
        HVX_ویکټور_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_ویکټور xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        إذا (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        إذا (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}
ساكن مضمن فراغ hvx_mad_f32_f16_aa_vec(عائم * محدود y, ثابت فراغ * محدود x, HVX_ویکټور S0, uint32_t n) {
    ثابت HVX_ویکټور * محدود vx0 = (ثابت HVX_ویکټور *) x;

    HVX_ویکټور_Pair * محدود vy_p = (HVX_ویکټور_Pair *) y;
    HVX_ویکټور * محدود vy = (HVX_ویکټور *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    uint32_t i = 0;

    #لارښوونه unroll(2)
    لكل (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    إذا (nloe) {
        HVX_ویکټور_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_ویکټور xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        إذا (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        إذا (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

ساكن مضمن فراغ hvx_mad_f32_f16_aa_rx2_vec(عائم * محدود y, ثابت فراغ * محدود x0, ثابت فراغ * محدود x1,
                                          HVX_ویکټور S0, HVX_ویکټور S1, uint32_t n) {
    ثابت HVX_ویکټور * محدود vx0 = (ثابت HVX_ویکټور *) x0;
    ثابت HVX_ویکټور * محدود vx1 = (ثابت HVX_ویکټور *) x1;

    HVX_ویکټور_Pair * محدود vy_p  = (HVX_ویکټور_Pair *) y;
    HVX_ویکټور * محدود vy        = (HVX_ویکټور *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    uint32_t i = 0;

    #لارښوونه unroll(2)
    لكل (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    إذا (nloe) {
        HVX_ویکټور_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_ویکټور xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        إذا (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        إذا (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

ساكن مضمن فراغ hvx_scale_vec_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت uint32_t n, HVX_ویکټور vs) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);

    ثابت HVX_ویکټور * محدود vsrc = (ثابت HVX_ویکټور * محدود) src;
    HVX_ویکټور * محدود vdst       = (HVX_ویکټور * محدود) dst;

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #لارښوونه unroll(4)
    لكل (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    إذا (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * حجم(عائم), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

#د_که_پای /* HVX_FA_KERNELS_H */
