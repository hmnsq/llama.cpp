#यदि_अपरिभाषितम् HVX_FA_KERNELS_H
#परिभाषय HVX_FA_KERNELS_H

#अन्तर्भाव <assert.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव "hvx-utils.h"

// Little inner kernels for HVX

#यदि __HVX_ARCH__ < 79
#परिभाषय HVX_OP_सङ्कलय_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#परिभाषय HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#परिभाषय HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#अन्यथा
#परिभाषय HVX_OP_सङ्कलय_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#परिभाषय HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#परिभाषय HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#यदि_अन्तः

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
स्थिर __attribute__((अप्रयुक्तम्_2)) __attribute__((noinline)) शून्य hvx_vec_f32_गन्तव्यम्_f16_a(शून्य *ptr, HVX_सदिशम् v0, HVX_सदिशम् v1)
{
    *(HVX_सदिशम् *) ptr = hvx_vec_f32_गन्तव्यम्_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
स्थिर अन्तःस्थ शून्य hvx_dot_f16_f16_aa(भिन्न * सीमित r, ध्रुव शून्य * सीमित x, ध्रुव शून्य * सीमित y, अचिह्नित पूर्णाङ्क n, भिन्न s) {
    ध्रुव HVX_सदिशम् * सीमित vx = (ध्रुव HVX_सदिशम् * सीमित) x; // fp16
    ध्रुव HVX_सदिशम् * सीमित vy = (ध्रुव HVX_सदिशम् * सीमित) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_सदिशम्_Pair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #निर्देशः unroll(4)
    प्रति (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_सदिशम् y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_सदिशम् x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_सदिशम् rsum = HVX_OP_सङ्कलय_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_योग_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_dot_f16_f16_aa_rx4(ध्रुव शून्य * सीमित y,
                                                ध्रुव uint8_t * सीमित x,
                                                ध्रुव size_t stride_x,
                                                ध्रुव size_t nvec,
                                                ध्रुव size_t nloe) {
    ध्रुव HVX_सदिशम् * सीमित vx0 = (ध्रुव HVX_सदिशम् * सीमित) x;                   // fp16
    ध्रुव HVX_सदिशम् * सीमित vx1 = (ध्रुव HVX_सदिशम् * सीमित) (x + stride_x);      // fp16
    ध्रुव HVX_सदिशम् * सीमित vx2 = (ध्रुव HVX_सदिशम् * सीमित) (x + stride_x * 2);  // fp16
    ध्रुव HVX_सदिशम् * सीमित vx3 = (ध्रुव HVX_सदिशम् * सीमित) (x + stride_x * 3);  // fp16
    ध्रुव HVX_सदिशम् * सीमित vy  = (ध्रुव HVX_सदिशम् * सीमित) y;                   // fp16

    HVX_सदिशम्_Pair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_सदिशम्_Pair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_सदिशम्_Pair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_सदिशम्_Pair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    प्रति (i = 0; i < nvec; i++) {
        HVX_सदिशम् y_hf  = vy[i];
        HVX_सदिशम् x0_hf = vx0[i];
        HVX_सदिशम् x1_hf = vx1[i];
        HVX_सदिशम् x2_hf = vx2[i];
        HVX_सदिशम् x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    यदि (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_सदिशम्     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_सदिशम्     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_सदिशम्     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_सदिशम्     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_सदिशम्     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_सदिशम् rsum0 = HVX_OP_सङ्कलय_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_सदिशम् rsum1 = HVX_OP_सङ्कलय_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_सदिशम् rsum2 = HVX_OP_सङ्कलय_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_सदिशम् rsum3 = HVX_OP_सङ्कलय_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_सदिशम्_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    प्रत्यागम hvx_vec_reduce_योग_f32x4(rsum0123);
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_dot_f16_f16_aa_rx32(ध्रुव शून्य * सीमित y,
                                                 ध्रुव uint8_t * सीमित x,
                                                 ध्रुव size_t stride_x,
                                                 ध्रुव size_t n,
                                                 भिन्न        s) {

    ध्रुव size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    ध्रुव size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_सदिशम्   sums = Q6_V_vzero();
    ध्रुव size_t stride_x_4 = stride_x * 4;
    प्रति (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_सदिशम्     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_सदिशम्_Pred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    प्रत्यागम HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// MAD: y (F32) += x (F16) * s (F16)
स्थिर अन्तःस्थ शून्य hvx_mad_f32_f16_aa(भिन्न * सीमित y, ध्रुव शून्य * सीमित x, ध्रुव __fp16 * सीमित s, uint32_t n) {
    ध्रुव HVX_सदिशम् * सीमित vx0 = (ध्रुव HVX_सदिशम् *) x;

    HVX_सदिशम्_Pair * सीमित vy_p = (HVX_सदिशम्_Pair *) y;
    HVX_सदिशम् * सीमित vy = (HVX_सदिशम् *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_सदिशम् S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_सदिशम् xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        यदि (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        यदि (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
स्थिर अन्तःस्थ शून्य hvx_mad_f32_f16_aa_rx2(भिन्न * सीमित y, ध्रुव शून्य * सीमित x0, ध्रुव शून्य * सीमित x1,
                                          ध्रुव __fp16 * सीमित s0, ध्रुव __fp16 * सीमित s1, uint32_t n) {
    ध्रुव HVX_सदिशम् * सीमित vx0 = (ध्रुव HVX_सदिशम् *) x0;
    ध्रुव HVX_सदिशम् * सीमित vx1 = (ध्रुव HVX_सदिशम् *) x1;

    HVX_सदिशम्_Pair * सीमित vy_p  = (HVX_सदिशम्_Pair *) y;
    HVX_सदिशम् * सीमित vy        = (HVX_सदिशम् *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_सदिशम् S0 = hvx_vec_splat_f16(*s0);
    HVX_सदिशम् S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_सदिशम् xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        यदि (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        यदि (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}
स्थिर अन्तःस्थ शून्य hvx_mad_f32_f16_aa_vec(भिन्न * सीमित y, ध्रुव शून्य * सीमित x, HVX_सदिशम् S0, uint32_t n) {
    ध्रुव HVX_सदिशम् * सीमित vx0 = (ध्रुव HVX_सदिशम् *) x;

    HVX_सदिशम्_Pair * सीमित vy_p = (HVX_सदिशम्_Pair *) y;
    HVX_सदिशम् * सीमित vy = (HVX_सदिशम् *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    uint32_t i = 0;

    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_सदिशम् xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        यदि (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        यदि (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

स्थिर अन्तःस्थ शून्य hvx_mad_f32_f16_aa_rx2_vec(भिन्न * सीमित y, ध्रुव शून्य * सीमित x0, ध्रुव शून्य * सीमित x1,
                                          HVX_सदिशम् S0, HVX_सदिशम् S1, uint32_t n) {
    ध्रुव HVX_सदिशम् * सीमित vx0 = (ध्रुव HVX_सदिशम् *) x0;
    ध्रुव HVX_सदिशम् * सीमित vx1 = (ध्रुव HVX_सदिशम् *) x1;

    HVX_सदिशम्_Pair * सीमित vy_p  = (HVX_सदिशम्_Pair *) y;
    HVX_सदिशम् * सीमित vy        = (HVX_सदिशम् *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    uint32_t i = 0;

    #निर्देशः unroll(2)
    प्रति (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    यदि (nloe) {
        HVX_सदिशम्_Pair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_सदिशम् xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        यदि (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        यदि (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

स्थिर अन्तःस्थ शून्य hvx_scale_vec_f32_aa(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव uint32_t n, HVX_सदिशम् vs) {
    प्रतिज्ञा((size_t) dst % 128 == 0);
    प्रतिज्ञा((size_t) src % 128 == 0);

    ध्रुव HVX_सदिशम् * सीमित vsrc = (ध्रुव HVX_सदिशम् * सीमित) src;
    HVX_सदिशम् * सीमित vdst       = (HVX_सदिशम् * सीमित) dst;

    ध्रुव uint32_t nvec = n / VLEN_FP32;
    ध्रुव uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #निर्देशः unroll(4)
    प्रति (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    यदि (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * परिमाण(भिन्न), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

#यदि_अन्तः /* HVX_FA_KERNELS_H */
