#ifndef HVX_MM_KERNELS_FLOAT_H
#define HVX_MM_KERNELS_FLOAT_H

#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

// Float activation copy/quantization kernels (DDR -> VTCM)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ quantize_f32_f32_kernel(
    ಧ್ರುವ uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (ಶೂನ್ಯ) tmp_data;
    ಧ್ರುವ size_t src_row_size = ne0 * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಪ್ರತಿ (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f32_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ quantize_f32_f16_kernel(
    ಧ್ರುವ uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (ಶೂನ್ಯ) tmp_data;
    ಧ್ರುವ size_t src_row_size = ne0 * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಪ್ರತಿ (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f16_f32_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ quantize_f16_f16_kernel(
    ಧ್ರುವ uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_stride,
    size_t dst_stride
) {
    (ಶೂನ್ಯ) tmp_data;
    ಧ್ರುವ size_t src_row_size = ne0 * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಪ್ರತಿ (uint32_t i = 0; i < nrows; ++i) {
        hex_l2fetch(src_data, src_row_size, src_stride, 2);
        hvx_copy_f16_au(dst_data, src_data, ne0);

        dst_data += dst_stride;
        src_data += src_stride;
    }
}

// Float dot product kernels (HVX)

#ಯದಿ __HVX_ARCH__ < 79
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#ಅನ್ಯಥಾ
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ flat_vec_dot_f32_f32_aa_1x1(ಧ್ರುವ uint32_t n, ಭಿನ್ನ * restrict s, ಧ್ರುವ ಶೂನ್ಯ * restrict vx, ಧ್ರುವ ಶೂನ್ಯ * restrict vy) {
    ಧ್ರುವ HVX_Vector * restrict x = (ಧ್ರುವ HVX_Vector *) vx;
    ಧ್ರುವ HVX_Vector * restrict y = (ಧ್ರುವ HVX_Vector *) vy;

    uint32_t nvec = n / VLEN_FP32; // num full fp32 hvx vectors
    uint32_t nloe = n % VLEN_FP32; // leftover elements

    HVX_Vector rsum = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(4)
    ಪ್ರತಿ (i = 0; i < nvec; i++) {
        HVX_Vector prod = HVX_OP_MUL_F32(x[i], y[i]);
        rsum = HVX_OP_ADD_F32(rsum, prod);
    }

    ಯದಿ (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector x_sf = Q6_V_vand_QV(bmask, x[i]);
        HVX_Vector y_sf = Q6_V_vand_QV(bmask, y[i]);
        HVX_Vector prod = HVX_OP_MUL_F32(x_sf, y_sf);
        rsum = HVX_OP_ADD_F32(rsum, prod);
    }

    *s = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(rsum));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ flat_vec_dot_f32_f32_aa_2x1(ಧ್ರುವ uint32_t n, ಭಿನ್ನ * restrict s0,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vx0, ಧ್ರುವ ಶೂನ್ಯ * restrict vx1,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vy0) {
    ಧ್ರುವ HVX_Vector * restrict x0 = (ಧ್ರುವ HVX_Vector *) vx0;
    ಧ್ರುವ HVX_Vector * restrict x1 = (ಧ್ರುವ HVX_Vector *) vx1;
    ಧ್ರುವ HVX_Vector * restrict y  = (ಧ್ರುವ HVX_Vector *) vy0;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_Vector rsum0 = Q6_V_vzero();
    HVX_Vector rsum1 = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    ಪ್ರತಿ (i = 0; i < nvec; i++) {
        HVX_Vector y_sf = y[i];
        HVX_Vector prod0 = HVX_OP_MUL_F32(x0[i], y_sf);
        HVX_Vector prod1 = HVX_OP_MUL_F32(x1[i], y_sf);
        rsum0 = HVX_OP_ADD_F32(rsum0, prod0);
        rsum1 = HVX_OP_ADD_F32(rsum1, prod1);
    }

    ಯದಿ (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector y_sf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_Vector x0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector x1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector prod0 = HVX_OP_MUL_F32(x0_sf, y_sf);
        HVX_Vector prod1 = HVX_OP_MUL_F32(x1_sf, y_sf);
        rsum0 = HVX_OP_ADD_F32(rsum0, prod0);
        rsum1 = HVX_OP_ADD_F32(rsum1, prod1);
    }

    HVX_Vector rsum = hvx_vec_reduce_sum_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ flat_vec_dot_f32_f32_aa_2x2(ಧ್ರುವ uint32_t n, ಭಿನ್ನ * restrict s0, ಭಿನ್ನ * restrict s1,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vx0, ಧ್ರುವ ಶೂನ್ಯ * restrict vx1,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vy0, ಧ್ರುವ ಶೂನ್ಯ * restrict vy1) {
    ಧ್ರುವ HVX_Vector * restrict x0 = (ಧ್ರುವ HVX_Vector *) vx0;
    ಧ್ರುವ HVX_Vector * restrict x1 = (ಧ್ರುವ HVX_Vector *) vx1;
    ಧ್ರುವ HVX_Vector * restrict y0 = (ಧ್ರುವ HVX_Vector *) vy0;
    ಧ್ರುವ HVX_Vector * restrict y1 = (ಧ್ರುವ HVX_Vector *) vy1;

    uint32_t nvec = n / VLEN_FP32;
    uint32_t nloe = n % VLEN_FP32;

    HVX_Vector r0_c0_sum = Q6_V_vzero();
    HVX_Vector r0_c1_sum = Q6_V_vzero();
    HVX_Vector r1_c0_sum = Q6_V_vzero();
    HVX_Vector r1_c1_sum = Q6_V_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    ಪ್ರತಿ (i = 0; i < nvec; i++) {
        HVX_Vector r0_sf = x0[i];
        HVX_Vector r1_sf = x1[i];
        HVX_Vector c0_sf = y0[i];
        HVX_Vector c1_sf = y1[i];

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    ಯದಿ (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);

        HVX_Vector r0_sf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector r1_sf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector c0_sf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_Vector c1_sf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(r0_sf, c0_sf));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(r0_sf, c1_sf));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(r1_sf, c0_sf));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(r1_sf, c1_sf));
    }

    // Reduce and store results
    HVX_Vector r0_r1_c0_sum = hvx_vec_reduce_sum_f32x2(r0_c0_sum, r1_c0_sum);
    HVX_Vector r0_r1_c1_sum = hvx_vec_reduce_sum_f32x2(r0_c1_sum, r1_c1_sum);

    hvx_vec_store_u(s0, 8, r0_r1_c0_sum);
    hvx_vec_store_u(s1, 8, r0_r1_c1_sum);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ flat_vec_dot_bf16_f32_aa_1x1(ಧ್ರುವ uint32_t n, ಭಿನ್ನ * restrict s, ಧ್ರುವ ಶೂನ್ಯ * restrict vx, ಧ್ರುವ ಶೂನ್ಯ * restrict vy) {
    ಧ್ರುವ HVX_Vector * restrict x = (ಧ್ರುವ HVX_Vector *) vx;
    ಧ್ರುವ HVX_Vector * restrict y = (ಧ್ರುವ HVX_Vector *) vy;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_Vector rsum0 = Q6_V_vzero();
    HVX_Vector rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #pragma unroll(2)
    ಪ್ರತಿ (i = 0; i < nvec; i++) {
        HVX_Vector v_bf16 = x[i];
        HVX_VectorPair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_Vector w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_Vector w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y[2 * i + 0]));
        rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y[2 * i + 1]));
    }

    ಯದಿ (rem) {
        HVX_VectorPred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_Vector v_bf16 = Q6_V_vand_QV(bmask, x[i]);
        HVX_VectorPair p = Q6_Ww_vunpack_Vh(v_bf16);
        HVX_Vector w0_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p), 16);
        HVX_Vector w1_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p), 16);

        ಯದಿ (rem <= 32) {
            HVX_VectorPred a_mask = Q6_Q_vsetq2_R(rem * 4);
            HVX_Vector y0 = Q6_V_vand_QV(a_mask, y[2 * i + 0]);
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));
        } ಅನ್ಯಥಾ {
            HVX_Vector y0 = y[2 * i + 0];
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w0_f32, y0));

            HVX_VectorPred a_mask = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_Vector y1 = Q6_V_vand_QV(a_mask, y[2 * i + 1]);
            rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w1_f32, y1));
        }
    }

    HVX_Vector rsum = HVX_OP_ADD_F32(rsum0, rsum1);
    *s = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(rsum));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ flat_vec_dot_bf16_f32_aa_2x1(ಧ್ರುವ uint32_t n, ಭಿನ್ನ * restrict s0,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vx0, ಧ್ರುವ ಶೂನ್ಯ * restrict vx1,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vy0) {
    ಧ್ರುವ HVX_Vector * restrict x0 = (ಧ್ರುವ HVX_Vector *) vx0;
    ಧ್ರುವ HVX_Vector * restrict x1 = (ಧ್ರುವ HVX_Vector *) vx1;
    ಧ್ರುವ HVX_Vector * restrict y  = (ಧ್ರುವ HVX_Vector *) vy0;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_Vector rsum0 = Q6_V_vzero();
    HVX_Vector rsum1 = Q6_V_vzero();

    uint32_t i = 0;
    #pragma unroll(2)
    ಪ್ರತಿ (i = 0; i < nvec; i++) {
        HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_Vector w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_Vector w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_Vector w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_Vector w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_Vector y0 = y[2 * i + 0];
        HVX_Vector y1 = y[2 * i + 1];

        rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
        rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
        rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
    }

    ಯದಿ (rem) {
        HVX_VectorPred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_Vector w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_Vector w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_Vector w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_Vector w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        ಯದಿ (rem <= 32) {
            HVX_VectorPred a_mask = Q6_Q_vsetq2_R(rem * 4);
            HVX_Vector y0 = Q6_V_vand_QV(a_mask, y[2 * i + 0]);
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));
        } ಅನ್ಯಥಾ {
            HVX_Vector y0 = y[2 * i + 0];
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w00_f32, y0));
            rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w10_f32, y0));

            HVX_VectorPred a_mask = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_Vector y1 = Q6_V_vand_QV(a_mask, y[2 * i + 1]);
            rsum0 = HVX_OP_ADD_F32(rsum0, HVX_OP_MUL_F32(w01_f32, y1));
            rsum1 = HVX_OP_ADD_F32(rsum1, HVX_OP_MUL_F32(w11_f32, y1));
        }
    }

    HVX_Vector rsum = hvx_vec_reduce_sum_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ flat_vec_dot_bf16_f32_aa_2x2(ಧ್ರುವ uint32_t n, ಭಿನ್ನ * restrict s0, ಭಿನ್ನ * restrict s1,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vx0, ಧ್ರುವ ಶೂನ್ಯ * restrict vx1,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vy0, ಧ್ರುವ ಶೂನ್ಯ * restrict vy1) {
    ಧ್ರುವ HVX_Vector * restrict x0 = (ಧ್ರುವ HVX_Vector *) vx0;
    ಧ್ರುವ HVX_Vector * restrict x1 = (ಧ್ರುವ HVX_Vector *) vx1;
    ಧ್ರುವ HVX_Vector * restrict y0 = (ಧ್ರುವ HVX_Vector *) vy0;
    ಧ್ರುವ HVX_Vector * restrict y1 = (ಧ್ರುವ HVX_Vector *) vy1;

    uint32_t nvec = n / 64;
    uint32_t rem  = n % 64;

    HVX_Vector r0_c0_sum = Q6_V_vzero();
    HVX_Vector r0_c1_sum = Q6_V_vzero();
    HVX_Vector r1_c0_sum = Q6_V_vzero();
    HVX_Vector r1_c1_sum = Q6_V_vzero();

    uint32_t i = 0;
    #pragma unroll(2)
    ಪ್ರತಿ (i = 0; i < nvec; i++) {
        HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(x0[i]);
        HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(x1[i]);
        HVX_Vector w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_Vector w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_Vector w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_Vector w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        HVX_Vector c0_0 = y0[2 * i + 0];
        HVX_Vector c0_1 = y0[2 * i + 1];
        HVX_Vector c1_0 = y1[2 * i + 0];
        HVX_Vector c1_1 = y1[2 * i + 1];

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w00_f32, c0_0));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w00_f32, c1_0));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w10_f32, c0_0));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w10_f32, c1_0));

        r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w01_f32, c0_1));
        r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w01_f32, c1_1));
        r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w11_f32, c0_1));
        r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w11_f32, c1_1));
    }

    ಯದಿ (rem) {
        HVX_VectorPred bmask = Q6_Q_vsetq2_R(rem * 2);
        HVX_VectorPair p0 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x0[i]));
        HVX_VectorPair p1 = Q6_Ww_vunpack_Vh(Q6_V_vand_QV(bmask, x1[i]));
        HVX_Vector w00_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p0), 16);
        HVX_Vector w01_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p0), 16);
        HVX_Vector w10_f32 = Q6_Vw_vasl_VwR(Q6_V_lo_W(p1), 16);
        HVX_Vector w11_f32 = Q6_Vw_vasl_VwR(Q6_V_hi_W(p1), 16);

        ಯದಿ (rem <= 32) {
            HVX_VectorPred a_mask = Q6_Q_vsetq2_R(rem * 4);
            HVX_Vector c0_0 = Q6_V_vand_QV(a_mask, y0[2 * i + 0]);
            HVX_Vector c1_0 = Q6_V_vand_QV(a_mask, y1[2 * i + 0]);
            r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w10_f32, c1_0));
        } ಅನ್ಯಥಾ {
            HVX_Vector c0_0 = y0[2 * i + 0];
            HVX_Vector c1_0 = y1[2 * i + 0];
            r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w00_f32, c0_0));
            r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w00_f32, c1_0));
            r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w10_f32, c0_0));
            r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w10_f32, c1_0));

            HVX_VectorPred a_mask = Q6_Q_vsetq2_R((rem - 32) * 4);
            HVX_Vector c0_1 = Q6_V_vand_QV(a_mask, y0[2 * i + 1]);
            HVX_Vector c1_1 = Q6_V_vand_QV(a_mask, y1[2 * i + 1]);
            r0_c0_sum = HVX_OP_ADD_F32(r0_c0_sum, HVX_OP_MUL_F32(w01_f32, c0_1));
            r0_c1_sum = HVX_OP_ADD_F32(r0_c1_sum, HVX_OP_MUL_F32(w01_f32, c1_1));
            r1_c0_sum = HVX_OP_ADD_F32(r1_c0_sum, HVX_OP_MUL_F32(w11_f32, c0_1));
            r1_c1_sum = HVX_OP_ADD_F32(r1_c1_sum, HVX_OP_MUL_F32(w11_f32, c1_1));
        }
    }

    HVX_Vector r0_r1_c0_sum = hvx_vec_reduce_sum_f32x2(r0_c0_sum, r1_c0_sum);
    HVX_Vector r0_r1_c1_sum = hvx_vec_reduce_sum_f32x2(r0_c1_sum, r1_c1_sum);

    hvx_vec_store_u(s0, 8, r0_r1_c0_sum);
    hvx_vec_store_u(s1, 8, r0_r1_c1_sum);
}

#undef HVX_OP_ADD_F32
#undef HVX_OP_MUL_F32

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ flat_vec_dot_f16_f16_aa_1x1(ಧ್ರುವ uint32_t n, ಭಿನ್ನ * restrict s, ಧ್ರುವ ಶೂನ್ಯ * restrict vx, ಧ್ರುವ ಶೂನ್ಯ * restrict vy) {
    ಧ್ರುವ HVX_Vector * restrict x = (ಧ್ರುವ HVX_Vector *) vx;
    ಧ್ರುವ HVX_Vector * restrict y = (ಧ್ರುವ HVX_Vector *) vy;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_VectorPair rsum_p = Q6_W_vzero();

    uint32_t i = 0;

    #pragma unroll(4)
    ಪ್ರತಿ (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x[i], y[i]);
    }

    ಯದಿ (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, x[i]);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, y[i]);
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_Vector rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));
    hvx_vec_store_u(s, 4, hvx_vec_reduce_sum_f32(rsum));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ flat_vec_dot_f16_f16_aa_2x1(ಧ್ರುವ uint32_t n, ಭಿನ್ನ * restrict s0,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vx0, ಧ್ರುವ ಶೂನ್ಯ * restrict vx1,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vy0) {
    ಧ್ರುವ HVX_Vector * restrict x0 = (ಧ್ರುವ HVX_Vector *) vx0;
    ಧ್ರುವ HVX_Vector * restrict x1 = (ಧ್ರುವ HVX_Vector *) vx1;
    ಧ್ರುವ HVX_Vector * restrict y  = (ಧ್ರುವ HVX_Vector *) vy0;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    HVX_VectorPair rsum0_p = Q6_W_vzero();
    HVX_VectorPair rsum1_p = Q6_W_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    ಪ್ರತಿ (i = 0; i < nvec; i++) {
        HVX_Vector y_hf = y[i];
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0[i], y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1[i], y_hf);
    }

    ಯದಿ (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_hf  = Q6_V_vand_QV(bmask, y[i]);
        HVX_Vector x0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector x1_hf = Q6_V_vand_QV(bmask, x1[i]);
        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
    }

    HVX_Vector rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p)));
    HVX_Vector rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p)));
    HVX_Vector rsum  = hvx_vec_reduce_sum_f32x2(rsum0, rsum1);
    hvx_vec_store_u(s0, 8, rsum);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ flat_vec_dot_f16_f16_aa_2x2(ಧ್ರುವ uint32_t n, ಭಿನ್ನ * restrict s0, ಭಿನ್ನ * restrict s1,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vx0, ಧ್ರುವ ಶೂನ್ಯ * restrict vx1,
                                ಧ್ರುವ ಶೂನ್ಯ * restrict vy0, ಧ್ರುವ ಶೂನ್ಯ * restrict vy1) {
    ಧ್ರುವ HVX_Vector * restrict x0 = (ಧ್ರುವ HVX_Vector *) vx0;
    ಧ್ರುವ HVX_Vector * restrict x1 = (ಧ್ರುವ HVX_Vector *) vx1;
    ಧ್ರುವ HVX_Vector * restrict y0 = (ಧ್ರುವ HVX_Vector *) vy0;
    ಧ್ರುವ HVX_Vector * restrict y1 = (ಧ್ರುವ HVX_Vector *) vy1;

    uint32_t nvec = n / VLEN_FP16;
    uint32_t nloe = n % VLEN_FP16;

    // Row sums (sf) - 4 accumulators for 2x2 tile
    HVX_VectorPair r0_c0_sum_p = Q6_W_vzero();
    HVX_VectorPair r0_c1_sum_p = Q6_W_vzero();
    HVX_VectorPair r1_c0_sum_p = Q6_W_vzero();
    HVX_VectorPair r1_c1_sum_p = Q6_W_vzero();

    uint32_t i = 0;

    #pragma unroll(2)
    ಪ್ರತಿ (i = 0; i < nvec; i++) {
        HVX_Vector r0_hf = x0[i];
        HVX_Vector r1_hf = x1[i];
        HVX_Vector c0_hf = y0[i];
        HVX_Vector c1_hf = y1[i];

        // Compute 4 dot products: r0xc0, r0xc1, r1xc0, r1xc1
        r0_c0_sum_p = hvx_vec_mpyacc_f32_f16(r0_c0_sum_p, r0_hf, c0_hf);
        r0_c1_sum_p = hvx_vec_mpyacc_f32_f16(r0_c1_sum_p, r0_hf, c1_hf);
        r1_c0_sum_p = hvx_vec_mpyacc_f32_f16(r1_c0_sum_p, r1_hf, c0_hf);
        r1_c1_sum_p = hvx_vec_mpyacc_f32_f16(r1_c1_sum_p, r1_hf, c1_hf);
    }

    ಯದಿ (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);

        HVX_Vector r0_hf = Q6_V_vand_QV(bmask, x0[i]);
        HVX_Vector r1_hf = Q6_V_vand_QV(bmask, x1[i]);
        HVX_Vector c0_hf = Q6_V_vand_QV(bmask, y0[i]);
        HVX_Vector c1_hf = Q6_V_vand_QV(bmask, y1[i]);

        r0_c0_sum_p = hvx_vec_mpyacc_f32_f16(r0_c0_sum_p, r0_hf, c0_hf);
        r0_c1_sum_p = hvx_vec_mpyacc_f32_f16(r0_c1_sum_p, r0_hf, c1_hf);
        r1_c0_sum_p = hvx_vec_mpyacc_f32_f16(r1_c0_sum_p, r1_hf, c0_hf);
        r1_c1_sum_p = hvx_vec_mpyacc_f32_f16(r1_c1_sum_p, r1_hf, c1_hf);
    }

    HVX_Vector r0_c0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c0_sum_p), Q6_V_hi_W(r0_c0_sum_p)));
    HVX_Vector r0_c1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r0_c1_sum_p), Q6_V_hi_W(r0_c1_sum_p)));
    HVX_Vector r1_c0_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c0_sum_p), Q6_V_hi_W(r1_c0_sum_p)));
    HVX_Vector r1_c1_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(r1_c1_sum_p), Q6_V_hi_W(r1_c1_sum_p)));

    // Reduce and store results
    HVX_Vector r0_r1_c0_sum = hvx_vec_reduce_sum_f32x2(r0_c0_sum, r1_c0_sum);
    HVX_Vector r0_r1_c1_sum = hvx_vec_reduce_sum_f32x2(r0_c1_sum, r1_c1_sum);

    hvx_vec_store_u(&s0[0], 8, r0_r1_c0_sum);  // row0,col0 row1,col0
    hvx_vec_store_u(&s1[0], 8, r0_r1_c1_sum);  // row0,col1 row1,col1
}



ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_tensor_add_f32_grid(
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * restrict dst,
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * restrict src2,
    uint32_t start_row,
    uint32_t end_row,
    uint32_t start_col,
    uint32_t end_col,
    ಧ್ರುವ ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು * div_ne11_12,
    ಧ್ರುವ ಸಂರಚನಾ fastdiv_ಮೌಲ್ಯಗಳು * div_ne11
) {
    ಯದಿ (start_row >= end_row || start_col >= end_col) ಪ್ರತ್ಯಾಗಮ;
    ಧ್ರುವ uint32_t nb1 = dst->nb[1]; // row stride in bytes

    ಧ್ರುವ uint32_t ne11 = dst->ne[1];
    ಧ್ರುವ uint32_t ne12 = dst->ne[2];
    ಧ್ರುವ uint32_t ne11_12 = ne11 * ne12;

    ಧ್ರುವ bool is_broadcast1 = (src2->ne[1] == 1);
    ಧ್ರುವ bool is_broadcast2 = (src2->ne[2] == 1);
    ಧ್ರುವ bool is_broadcast3 = (src2->ne[3] == 1);

    ಪ್ರತಿ (uint32_t r = start_row; r < end_row; r++) {
        ಭಿನ್ನ * dst_row = (ಭಿನ್ನ *) ((uint8_t *) dst->data + (size_t) r * nb1);

        uint32_t i13 = fastdiv(r, div_ne11_12);
        uint32_t i12 = fastdiv(r - i13 * ne11_12, div_ne11);
        uint32_t i11 = r - i13 * ne11_12 - i12 * ne11;

        uint32_t i23 = is_broadcast3 ? 0 : i13;
        uint32_t i22 = is_broadcast2 ? 0 : i12;
        uint32_t i21 = is_broadcast1 ? 0 : i11;

        ಧ್ರುವ ಭಿನ್ನ * src2_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ uint8_t *) src2->data +
                                  (size_t) i21 * src2->nb[1] + (size_t) i22 * src2->nb[2] + (size_t) i23 * src2->nb[3]);

        ಭಿನ್ನ * dst_ptr = &dst_row[start_col];
        ಧ್ರುವ ಭಿನ್ನ * src2_ptr = &src2_row[start_col];
        ಪೂರ್ಣಾಙ್ಕ remaining = end_col - start_col;
        ಯಾವತ್ (remaining >= 32) {
            HVX_Vector v_out = hvx_vmemu(dst_ptr);
            HVX_Vector v_z   = hvx_vmemu(src2_ptr);
            hvx_vmemu(dst_ptr) = hvx_vec_add_f32_f32(v_out, v_z);
            dst_ptr += 32;
            src2_ptr += 32;
            remaining -= 32;
        }
        ಯದಿ (remaining > 0) {
            HVX_Vector v_out = hvx_vmemu(dst_ptr);
            HVX_Vector v_z   = hvx_vmemu(src2_ptr);
            hvx_vec_store_u(dst_ptr, remaining * ಪರಿಮಾಣ(ಭಿನ್ನ), hvx_vec_add_f32_f32(v_out, v_z));
        }
    }
}

#endif // HVX_MM_KERNELS_FLOAT_H
