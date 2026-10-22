#ifndef HVX_FA_KERNELS_H
#define HVX_FA_KERNELS_H

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ "hvx-utils.h"

// Little inner kernels for HVX

#எனில் __HVX_ARCH__ < 79
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#இல்லையெனில்
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
நிலையான __attribute__((unused)) __attribute__((noinline)) வெறுமை hvx_vec_f32_to_f16_a(வெறுமை *ptr, HVX_Vector v0, HVX_Vector v1)
{
    *(HVX_Vector *) ptr = hvx_vec_f32_to_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
நிலையான உள்வரி வெறுமை hvx_dot_f16_f16_aa(தசம * restrict r, மாறிலி வெறுமை * restrict x, மாறிலி வெறுமை * restrict y, unsigned முழு n, தசம s) {
    மாறிலி HVX_Vector * restrict vx = (மாறிலி HVX_Vector * restrict) x; // fp16
    மாறிலி HVX_Vector * restrict vy = (மாறிலி HVX_Vector * restrict) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_VectorPair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #pragma unroll(4)
    க்கு (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    எனில் (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_Vector rsum = HVX_OP_ADD_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

நிலையான உள்வரி HVX_Vector hvx_dot_f16_f16_aa_rx4(மாறிலி வெறுமை * restrict y,
                                                மாறிலி uint8_t * restrict x,
                                                மாறிலி size_t stride_x,
                                                மாறிலி size_t nvec,
                                                மாறிலி size_t nloe) {
    மாறிலி HVX_Vector * restrict vx0 = (மாறிலி HVX_Vector * restrict) x;                   // fp16
    மாறிலி HVX_Vector * restrict vx1 = (மாறிலி HVX_Vector * restrict) (x + stride_x);      // fp16
    மாறிலி HVX_Vector * restrict vx2 = (மாறிலி HVX_Vector * restrict) (x + stride_x * 2);  // fp16
    மாறிலி HVX_Vector * restrict vx3 = (மாறிலி HVX_Vector * restrict) (x + stride_x * 3);  // fp16
    மாறிலி HVX_Vector * restrict vy  = (மாறிலி HVX_Vector * restrict) y;                   // fp16

    HVX_VectorPair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    க்கு (i = 0; i < nvec; i++) {
        HVX_Vector y_hf  = vy[i];
        HVX_Vector x0_hf = vx0[i];
        HVX_Vector x1_hf = vx1[i];
        HVX_Vector x2_hf = vx2[i];
        HVX_Vector x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    எனில் (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_Vector     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_Vector     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_Vector     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_Vector rsum0 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_Vector rsum1 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_Vector rsum2 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_Vector rsum3 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_Vector_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    திரும்பு hvx_vec_reduce_sum_f32x4(rsum0123);
}

நிலையான உள்வரி HVX_Vector hvx_dot_f16_f16_aa_rx32(மாறிலி வெறுமை * restrict y,
                                                 மாறிலி uint8_t * restrict x,
                                                 மாறிலி size_t stride_x,
                                                 மாறிலி size_t n,
                                                 தசம        s) {

    மாறிலி size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    மாறிலி size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector   sums = Q6_V_vzero();
    மாறிலி size_t stride_x_4 = stride_x * 4;
    க்கு (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_Vector     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_VectorPred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    திரும்பு HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// MAD: y (F32) += x (F16) * s (F16)
நிலையான உள்வரி வெறுமை hvx_mad_f32_f16_aa(தசம * restrict y, மாறிலி வெறுமை * restrict x, மாறிலி __fp16 * restrict s, uint32_t n) {
    மாறிலி HVX_Vector * restrict vx0 = (மாறிலி HVX_Vector *) x;

    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #pragma unroll(2)
    க்கு (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    எனில் (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        எனில் (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        எனில் (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
நிலையான உள்வரி வெறுமை hvx_mad_f32_f16_aa_rx2(தசம * restrict y, மாறிலி வெறுமை * restrict x0, மாறிலி வெறுமை * restrict x1,
                                          மாறிலி __fp16 * restrict s0, மாறிலி __fp16 * restrict s1, uint32_t n) {
    மாறிலி HVX_Vector * restrict vx0 = (மாறிலி HVX_Vector *) x0;
    மாறிலி HVX_Vector * restrict vx1 = (மாறிலி HVX_Vector *) x1;

    HVX_VectorPair * restrict vy_p  = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy        = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s0);
    HVX_Vector S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #pragma unroll(2)
    க்கு (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    எனில் (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        எனில் (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        எனில் (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}
நிலையான உள்வரி வெறுமை hvx_mad_f32_f16_aa_vec(தசம * restrict y, மாறிலி வெறுமை * restrict x, HVX_Vector S0, uint32_t n) {
    மாறிலி HVX_Vector * restrict vx0 = (மாறிலி HVX_Vector *) x;

    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    uint32_t i = 0;

    #pragma unroll(2)
    க்கு (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    எனில் (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        எனில் (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        எனில் (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

நிலையான உள்வரி வெறுமை hvx_mad_f32_f16_aa_rx2_vec(தசம * restrict y, மாறிலி வெறுமை * restrict x0, மாறிலி வெறுமை * restrict x1,
                                          HVX_Vector S0, HVX_Vector S1, uint32_t n) {
    மாறிலி HVX_Vector * restrict vx0 = (மாறிலி HVX_Vector *) x0;
    மாறிலி HVX_Vector * restrict vx1 = (மாறிலி HVX_Vector *) x1;

    HVX_VectorPair * restrict vy_p  = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy        = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    uint32_t i = 0;

    #pragma unroll(2)
    க்கு (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    எனில் (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        எனில் (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        எனில் (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

நிலையான உள்வரி வெறுமை hvx_scale_vec_f32_aa(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, மாறிலி uint32_t n, HVX_Vector vs) {
    ப்ரதிஜ்ஞா((size_t) dst % 128 == 0);
    ப்ரதிஜ்ஞா((size_t) src % 128 == 0);

    மாறிலி HVX_Vector * restrict vsrc = (மாறிலி HVX_Vector * restrict) src;
    HVX_Vector * restrict vdst       = (HVX_Vector * restrict) dst;

    மாறிலி uint32_t nvec = n / VLEN_FP32;
    மாறிலி uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #pragma unroll(4)
    க்கு (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    எனில் (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * அளவு(தசம), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

#endif /* HVX_FA_KERNELS_H */
