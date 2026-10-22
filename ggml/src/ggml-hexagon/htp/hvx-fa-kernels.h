#جے_بنا_تعریف HVX_FA_KERNELS_H
#تعریف_کرو HVX_FA_KERNELS_H

#ضمن <assert.h>
#ضمن <math.h>
#ضمن "hvx-utils.h"

// Little inner kernels for HVX

#إذا __HVX_ARCH__ < 79
#تعریف_کرو HVX_OP_جوڑو_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#تعریف_کرو HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#تعریف_کرو HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#وإلا
#تعریف_کرو HVX_OP_جوڑو_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#تعریف_کرو HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#تعریف_کرو HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#جے_انت

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
ساكن __attribute__((بن_ورتیا_2)) __attribute__((noinline)) فراغ hvx_vec_f32_منزل_f16_a(فراغ *ptr, HVX_سمتیہ v0, HVX_سمتیہ v1)
{
    *(HVX_سمتیہ *) ptr = hvx_vec_f32_منزل_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
ساكن مضمن فراغ hvx_dot_f16_f16_aa(عائم * محدود r, ثابت فراغ * محدود x, ثابت فراغ * محدود y, بنانشانی صحيح n, عائم s) {
    ثابت HVX_سمتیہ * محدود vx = (ثابت HVX_سمتیہ * محدود) x; // fp16
    ثابت HVX_سمتیہ * محدود vy = (ثابت HVX_سمتیہ * محدود) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_سمتیہ_Pair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #ہدایت unroll(4)
    لكل (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    إذا (nloe) {
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_سمتیہ y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_سمتیہ x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_سمتیہ rsum = HVX_OP_جوڑو_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_جوڑ_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

ساكن مضمن HVX_سمتیہ hvx_dot_f16_f16_aa_rx4(ثابت فراغ * محدود y,
                                                ثابت uint8_t * محدود x,
                                                ثابت size_t stride_x,
                                                ثابت size_t nvec,
                                                ثابت size_t nloe) {
    ثابت HVX_سمتیہ * محدود vx0 = (ثابت HVX_سمتیہ * محدود) x;                   // fp16
    ثابت HVX_سمتیہ * محدود vx1 = (ثابت HVX_سمتیہ * محدود) (x + stride_x);      // fp16
    ثابت HVX_سمتیہ * محدود vx2 = (ثابت HVX_سمتیہ * محدود) (x + stride_x * 2);  // fp16
    ثابت HVX_سمتیہ * محدود vx3 = (ثابت HVX_سمتیہ * محدود) (x + stride_x * 3);  // fp16
    ثابت HVX_سمتیہ * محدود vy  = (ثابت HVX_سمتیہ * محدود) y;                   // fp16

    HVX_سمتیہ_Pair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_سمتیہ_Pair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_سمتیہ_Pair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_سمتیہ_Pair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    لكل (i = 0; i < nvec; i++) {
        HVX_سمتیہ y_hf  = vy[i];
        HVX_سمتیہ x0_hf = vx0[i];
        HVX_سمتیہ x1_hf = vx1[i];
        HVX_سمتیہ x2_hf = vx2[i];
        HVX_سمتیہ x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    إذا (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_سمتیہ_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_سمتیہ     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_سمتیہ     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_سمتیہ     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_سمتیہ     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_سمتیہ     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_سمتیہ rsum0 = HVX_OP_جوڑو_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_سمتیہ rsum1 = HVX_OP_جوڑو_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_سمتیہ rsum2 = HVX_OP_جوڑو_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_سمتیہ rsum3 = HVX_OP_جوڑو_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_سمتیہ_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    أرجع hvx_vec_reduce_جوڑ_f32x4(rsum0123);
}

ساكن مضمن HVX_سمتیہ hvx_dot_f16_f16_aa_rx32(ثابت فراغ * محدود y,
                                                 ثابت uint8_t * محدود x,
                                                 ثابت size_t stride_x,
                                                 ثابت size_t n,
                                                 عائم        s) {

    ثابت size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    ثابت size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_سمتیہ   sums = Q6_V_vzero();
    ثابت size_t stride_x_4 = stride_x * 4;
    لكل (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_سمتیہ     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_سمتیہ_Pred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    أرجع HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// MAD: y (F32) += x (F16) * s (F16)
ساكن مضمن فراغ hvx_mad_f32_f16_aa(عائم * محدود y, ثابت فراغ * محدود x, ثابت __fp16 * محدود s, uint32_t n) {
    ثابت HVX_سمتیہ * محدود vx0 = (ثابت HVX_سمتیہ *) x;

    HVX_سمتیہ_Pair * محدود vy_p = (HVX_سمتیہ_Pair *) y;
    HVX_سمتیہ * محدود vy = (HVX_سمتیہ *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_سمتیہ S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #ہدایت unroll(2)
    لكل (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    إذا (nloe) {
        HVX_سمتیہ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_سمتیہ xy = Q6_V_lo_W(xy_p);
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
    ثابت HVX_سمتیہ * محدود vx0 = (ثابت HVX_سمتیہ *) x0;
    ثابت HVX_سمتیہ * محدود vx1 = (ثابت HVX_سمتیہ *) x1;

    HVX_سمتیہ_Pair * محدود vy_p  = (HVX_سمتیہ_Pair *) y;
    HVX_سمتیہ * محدود vy        = (HVX_سمتیہ *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_سمتیہ S0 = hvx_vec_splat_f16(*s0);
    HVX_سمتیہ S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #ہدایت unroll(2)
    لكل (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    إذا (nloe) {
        HVX_سمتیہ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_سمتیہ xy = Q6_V_lo_W(xy_p);
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
ساكن مضمن فراغ hvx_mad_f32_f16_aa_vec(عائم * محدود y, ثابت فراغ * محدود x, HVX_سمتیہ S0, uint32_t n) {
    ثابت HVX_سمتیہ * محدود vx0 = (ثابت HVX_سمتیہ *) x;

    HVX_سمتیہ_Pair * محدود vy_p = (HVX_سمتیہ_Pair *) y;
    HVX_سمتیہ * محدود vy = (HVX_سمتیہ *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    uint32_t i = 0;

    #ہدایت unroll(2)
    لكل (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    إذا (nloe) {
        HVX_سمتیہ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_سمتیہ xy = Q6_V_lo_W(xy_p);
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
                                          HVX_سمتیہ S0, HVX_سمتیہ S1, uint32_t n) {
    ثابت HVX_سمتیہ * محدود vx0 = (ثابت HVX_سمتیہ *) x0;
    ثابت HVX_سمتیہ * محدود vx1 = (ثابت HVX_سمتیہ *) x1;

    HVX_سمتیہ_Pair * محدود vy_p  = (HVX_سمتیہ_Pair *) y;
    HVX_سمتیہ * محدود vy        = (HVX_سمتیہ *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    uint32_t i = 0;

    #ہدایت unroll(2)
    لكل (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    إذا (nloe) {
        HVX_سمتیہ_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_سمتیہ xy = Q6_V_lo_W(xy_p);
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

ساكن مضمن فراغ hvx_scale_vec_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت uint32_t n, HVX_سمتیہ vs) {
    أكد((size_t) dst % 128 == 0);
    أكد((size_t) src % 128 == 0);

    ثابت HVX_سمتیہ * محدود vsrc = (ثابت HVX_سمتیہ * محدود) src;
    HVX_سمتیہ * محدود vdst       = (HVX_سمتیہ * محدود) dst;

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #ہدایت unroll(4)
    لكل (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    إذا (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * حجم(عائم), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

#جے_انت /* HVX_FA_KERNELS_H */
