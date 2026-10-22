#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>

#অন্তর্ভাব <math.h>
#অন্তর্ভাব <string.h>

#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "hvx-utils.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "htp-vtcm.h"

#define htp_act_preamble                                 \
    ধ্রুবক কাঠামো htp_tensor * src0 = actx->octx->src[0]; \
    ধ্রুবক কাঠামো htp_tensor * src1 = actx->octx->src[1]; \
    ধ্রুবক কাঠামো htp_tensor * dst  = actx->octx->dst;    \
                                                         \
    ধ্রুবক uint32_t ne00 = src0->ne[0];                   \
    ধ্রুবক uint32_t ne01 = src0->ne[1];                   \
    ধ্রুবক uint32_t ne02 = src0->ne[2];                   \
    ধ্রুবক uint32_t ne03 = src0->ne[3];                   \
                                                         \
    ধ্রুবক uint32_t nb00 = src0->nb[0];                   \
    ধ্রুবক uint32_t nb01 = src0->nb[1];                   \
    ধ্রুবক uint32_t nb02 = src0->nb[2];                   \
    ধ্রুবক uint32_t nb03 = src0->nb[3];                   \
                                                         \
    ধ্রুবক uint32_t ne10 = src1 ? src1->ne[0] : 0;        \
    ধ্রুবক uint32_t ne11 = src1 ? src1->ne[1] : 0;        \
    ধ্রুবক uint32_t ne12 = src1 ? src1->ne[2] : 0;        \
    ধ্রুবক uint32_t ne13 = src1 ? src1->ne[3] : 0;        \
                                                         \
    ধ্রুবক uint32_t nb10 = src1 ? src1->nb[0] : 0;        \
    ধ্রুবক uint32_t nb11 = src1 ? src1->nb[1] : 0;        \
    ধ্রুবক uint32_t nb12 = src1 ? src1->nb[2] : 0;        \
    ধ্রুবক uint32_t nb13 = src1 ? src1->nb[3] : 0;        \
                                                         \
    ধ্রুবক uint32_t ne0 = dst->ne[0];                     \
    ধ্রুবক uint32_t ne1 = dst->ne[1];                     \
    ধ্রুবক uint32_t ne2 = dst->ne[2];                     \
    ধ্রুবক uint32_t ne3 = dst->ne[3];                     \
                                                         \
    ধ্রুবক uint32_t nb0 = dst->nb[0];                     \
    ধ্রুবক uint32_t nb1 = dst->nb[1];                     \
    ধ্রুবক uint32_t nb2 = dst->nb[2];                     \
    ধ্রুবক uint32_t nb3 = dst->nb[3];

কাঠামো htp_act_context;

প্রকারলেখ শূন্য (*glu_compute_f32_fn_t)(ধ্রুবক দশমিক * restrict src0,
                                     ধ্রুবক দশমিক * restrict src1,
                                     দশমিক * restrict dst,
                                     ধ্রুবক uint32_t অঙ্ক_rows,
                                     ধ্রুবক কাঠামো htp_act_context * actx);

প্রকারলেখ শূন্য (*glu_compute_f16_fn_t)(ধ্রুবক _Float16 * restrict src0,
                                     ধ্রুবক _Float16 * restrict src1,
                                     _Float16 * restrict dst,
                                     ধ্রুবক uint32_t অঙ্ক_rows,
                                     ধ্রুবক কাঠামো htp_act_context * actx);

কাঠামো htp_act_context {
    কাঠামো htp_ops_context * octx;

    শূন্য *                   compute;
    ধ্রুবক অক্ষর *             op_str;

    // Precomputed values
    dma_addr_t               তথ্য_src0;
    dma_addr_t               তথ্য_src1;
    dma_addr_t               তথ্য_dst;

    size_t                   src0_row_আকার;
    size_t                   src1_row_আকার;
    size_t                   dst_row_আকার;

    size_t                   src0_row_stride;
    size_t                   src1_row_stride;

    size_t                   src0_row_আকার_aligned;
    size_t                   src1_row_আকার_aligned;
    size_t                   dst_row_আকার_aligned;

    size_t                   src0_spad_half_আকার;
    size_t                   src1_spad_half_আকার;
    size_t                   dst_spad_half_আকার;

    uint32_t                 block;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_per_thread;
    uint32_t                 row_start;
    পূর্ণসংখ্যা                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_আকার_per_thread;
    size_t                   vtcm_src1_আকার_per_thread;
    size_t                   vtcm_dst_আকার_per_thread;
};

কাঠামো htp_act_vtcm_layout {
    size_t যোগ_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t src1_bytes_per_thread;
    size_t dst_bytes_per_thread;

    uint32_t vtcm_row_per_thread;
};

স্থির অন্তঃরেখ শূন্য htp_act_vtcm_layout_build(কাঠামো htp_act_vtcm_layout * L,
                                             size_t                       src0_row_আকার_aligned,
                                             size_t                       src1_row_আকার_aligned,
                                             size_t                       dst_row_আকার_aligned,
                                             uint32_t                     n_threads,
                                             size_t                       vtcm_আকার) {
    ধ্রুবক size_t   spad_আকার_per_row   = src0_row_আকার_aligned + src1_row_আকার_aligned + dst_row_আকার_aligned;
    ধ্রুবক uint32_t vtcm_row_per_thread = (uint32_t) (vtcm_আকার / (n_threads * spad_আকার_per_row));

    L->vtcm_row_per_thread = vtcm_row_per_thread;

    L->src0_bytes_per_thread = src0_row_আকার_aligned * vtcm_row_per_thread;
    L->src1_bytes_per_thread = src1_row_আকার_aligned * vtcm_row_per_thread;
    L->dst_bytes_per_thread  = dst_row_আকার_aligned * vtcm_row_per_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_bytes_per_thread * n_threads;
    L->off_dst  = L->off_src1 + L->src1_bytes_per_thread * n_threads;

    L->যোগ_bytes = L->off_dst + L->dst_bytes_per_thread * n_threads;
}

#define htp_glu_op_preamble                                            \
    ধ্রুবক size_t src0_row_আকার_aligned = actx->src0_row_আকার_aligned;  \
    ধ্রুবক size_t src1_row_আকার_aligned = actx->src1_row_আকার_aligned;  \
    ধ্রুবক size_t dst_row_আকার_aligned  = actx->dst_row_আকার_aligned;   \
    ধ্রুবক পূর্ণসংখ্যা    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
স্থির শূন্য swiglu_f32(ধ্রুবক দশমিক * restrict src0,
                                                 ধ্রুবক দশমিক * restrict src1,
                                                 দশমিক * restrict dst,
                                                 ধ্রুবক uint32_t অঙ্ক_rows,
                                                 ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
স্থির শূন্য swiglu_oai_f32(ধ্রুবক দশমিক * restrict src0,
                                                     ধ্রুবক দশমিক * restrict src1,
                                                     দশমিক * restrict dst,
                                                     ধ্রুবক uint32_t অঙ্ক_rows,
                                                     ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;
    ধ্রুবক দশমিক alpha = ((ধ্রুবক দশমিক *) (actx->octx->op_params))[2];
    ধ্রুবক দশমিক limit = ((ধ্রুবক দশমিক *) (actx->octx->op_params))[3];

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, limit, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -limit, limit, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_add_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f32(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f32_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

স্থির শূন্য swiglu_clamp_f32(ধ্রুবক দশমিক * restrict src0,
                                                       ধ্রুবক দশমিক * restrict src1,
                                                       দশমিক * restrict dst,
                                                       ধ্রুবক uint32_t                 অঙ্ক_rows,
                                                       ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;
    ধ্রুবক দশমিক limit = ((ধ্রুবক দশমিক *) (actx->octx->op_params))[3];

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, limit, nc);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -limit, limit, nc);
        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

স্থির ধ্রুবক দশমিক GELU_COEF_A     = 0.044715f;
স্থির ধ্রুবক দশমিক SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

স্থির অন্তঃরেখ HVX_Vector hvx_vec_fast_sigmoid_f32_2it(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_Vector two_sf = hvx_vec_splat_f32(2.0f);
    HVX_Vector i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_Vector r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_Vector res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    ফেরত Q6_Vsf_equals_Vqf32(res);
}

স্থির অন্তঃরেখ HVX_Vector hvx_vec_fast_sigmoid_f32_guard_2it(HVX_Vector v,
                                                            HVX_Vector one,
                                                            HVX_Vector max_exp,
                                                            HVX_Vector min_exp) {
    ধ্রুবক HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ধ্রুবক HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_f32_2it(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    ফেরত Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

স্থির অন্তঃরেখ শূন্য hvx_geglu_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src0, ধ্রুবক uint8_t * restrict src1, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst  % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src0 % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src1 % 128 == 0);

    HVX_Vector * restrict vdst        = (HVX_Vector *) dst;
    ধ্রুবক HVX_Vector * restrict vsrc0 = (ধ্রুবক HVX_Vector *) src0;
    ধ্রুবক HVX_Vector * restrict vsrc1 = (ধ্রুবক HVX_Vector *) src1;

    ধ্রুবক uint32_t epv  = 128 / আকার(দশমিক);
    ধ্রুবক uint32_t nvec = n / epv;
    ধ্রুবক uint32_t nloe = n % epv;

    ধ্রুবক দশমিক GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    ধ্রুবক HVX_Vector v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    ধ্রুবক HVX_Vector v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    ধ্রুবক HVX_Vector v_one               = hvx_vec_splat_f32(1.0f);
    ধ্রুবক HVX_Vector v_max_exp           = hvx_vec_splat_f32(87.0f);
    ধ্রুবক HVX_Vector v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    জন্য (; i < nvec; i++) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];

        HVX_Vector x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_Vector coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_add_f32_f32(coef, v_sqrt_2_pi);
        HVX_Vector inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner = inner + inner
        HVX_Vector y2 = hvx_vec_add_f32_f32(inner, inner);

        // Fast sigmoid approximation (2 iterations)
        HVX_Vector sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_Vector gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    যদি (nloe) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];

        HVX_Vector x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_Vector coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_add_f32_f32(coef, v_sqrt_2_pi);
        HVX_Vector inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_Vector y2 = hvx_vec_add_f32_f32(inner, inner);

        HVX_Vector sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_Vector gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        HVX_Vector res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((শূন্য *) &vdst[i], nloe * আকার(দশমিক), res);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_geglu_quick_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src0, ধ্রুবক uint8_t * restrict src1, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst  % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src0 % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src1 % 128 == 0);

    HVX_Vector * restrict vdst        = (HVX_Vector *) dst;
    ধ্রুবক HVX_Vector * restrict vsrc0 = (ধ্রুবক HVX_Vector *) src0;
    ধ্রুবক HVX_Vector * restrict vsrc1 = (ধ্রুবক HVX_Vector *) src1;

    ধ্রুবক uint32_t epv  = 128 / আকার(দশমিক);
    ধ্রুবক uint32_t nvec = n / epv;
    ধ্রুবক uint32_t nloe = n % epv;

    ধ্রুবক HVX_Vector v_scale    = hvx_vec_splat_f32(1.702f);
    ধ্রুবক HVX_Vector v_one      = hvx_vec_splat_f32(1.0f);
    ধ্রুবক HVX_Vector v_max_exp  = hvx_vec_splat_f32(87.0f);
    ধ্রুবক HVX_Vector v_min_exp  = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    জন্য (; i < nvec; i++) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];
        HVX_Vector scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_Vector sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
    }

    যদি (nloe) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];
        HVX_Vector scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_Vector sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        HVX_Vector ফলাফল = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
        hvx_vec_store_a((শূন্য *) &vdst[i], nloe * আকার(দশমিক), ফলাফল);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_geglu_erf_f32_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src0, ধ্রুবক uint8_t * restrict src1, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst  % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src0 % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src1 % 128 == 0);

    HVX_Vector * restrict vdst        = (HVX_Vector *) dst;
    ধ্রুবক HVX_Vector * restrict vsrc0 = (ধ্রুবক HVX_Vector *) src0;
    ধ্রুবক HVX_Vector * restrict vsrc1 = (ধ্রুবক HVX_Vector *) src1;

    ধ্রুবক uint32_t epv  = 128 / আকার(দশমিক);
    ধ্রুবক uint32_t nvec = n / epv;
    ধ্রুবক uint32_t nloe = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    জন্য (; i < nvec; i++) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
    }

    যদি (nloe) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];
        HVX_Vector ফলাফল = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
        hvx_vec_store_a((শূন্য *) &vdst[i], nloe * আকার(দশমিক), ফলাফল);
    }
}

স্থির অন্তঃরেখ শূন্য hvx_geglu_f16_aa(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src0, ধ্রুবক uint8_t * restrict src1, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst  % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src0 % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src1 % 128 == 0);

    HVX_Vector * restrict vdst        = (HVX_Vector *) dst;
    ধ্রুবক HVX_Vector * restrict vsrc0 = (ধ্রুবক HVX_Vector *) src0;
    ধ্রুবক HVX_Vector * restrict vsrc1 = (ধ্রুবক HVX_Vector *) src1;

    ধ্রুবক uint32_t epv  = 128 / আকার(_Float16);
    ধ্রুবক uint32_t nvec = n / epv;
    ধ্রুবক uint32_t nloe = n % epv;

    ধ্রুবক HVX_Vector v_coef_a    = hvx_vec_splat_f16((_Float16) GELU_COEF_A);
    ধ্রুবক HVX_Vector v_sqrt_2_pi = hvx_vec_splat_f16((_Float16) SQRT_2_OVER_PI);
    ধ্রুবক HVX_Vector v_one       = hvx_vec_splat_f16((_Float16) 1.0f);
    ধ্রুবক HVX_Vector v_half      = hvx_vec_splat_f16((_Float16) 0.5f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    জন্য (; i < nvec; i++) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];

        // gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))
        HVX_Vector x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_Vector coef  = hvx_vec_add_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_Vector inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_Vector t = hvx_vec_add_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_Vector gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        vdst[i] = hvx_vec_mul_f16_f16(gelu_x, g);
    }

    যদি (nloe) {
        HVX_Vector x = vsrc0[i];
        HVX_Vector g = vsrc1[i];

        HVX_Vector x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_Vector coef  = hvx_vec_add_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_Vector inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_Vector t = hvx_vec_add_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_Vector gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        HVX_Vector res    = hvx_vec_mul_f16_f16(gelu_x, g);
        hvx_vec_store_a((শূন্য *) &vdst[i], nloe * আকার(_Float16), res);
    }
}

// geglu(x, g) = gelu(x) * g
স্থির শূন্য geglu_f32(ধ্রুবক দশমিক * restrict src0,
                                                ধ্রুবক দশমিক * restrict src1,
                                                দশমিক * restrict dst,
                                                ধ্রুবক uint32_t অঙ্ক_rows,
                                                ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_quick(x, g) = x * sigmoid(1.702 * x) * g
স্থির শূন্য geglu_quick_f32(ধ্রুবক দশমিক * restrict src0,
                            ধ্রুবক দশমিক * restrict src1,
                            দশমিক * restrict dst,
                            ধ্রুবক uint32_t অঙ্ক_rows,
                            ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        hvx_geglu_quick_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_erf(x, g) = gelu_erf(x) * g
স্থির শূন্য geglu_erf_f32(ধ্রুবক দশমিক * restrict src0,
                          ধ্রুবক দশমিক * restrict src1,
                          দশমিক * restrict dst,
                          ধ্রুবক uint32_t অঙ্ক_rows,
                          ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        hvx_geglu_erf_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// swiglu(x) = x1 * sigmoid(x0)
স্থির শূন্য swiglu_f16(ধ্রুবক _Float16 * restrict src0,
                       ধ্রুবক _Float16 * restrict src1,
                       _Float16 * restrict dst,
                       ধ্রুবক uint32_t অঙ্ক_rows,
                       ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
স্থির শূন্য swiglu_oai_f16(ধ্রুবক _Float16 * restrict src0,
                           ধ্রুবক _Float16 * restrict src1,
                           _Float16 * restrict dst,
                           ধ্রুবক uint32_t অঙ্ক_rows,
                           ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;
    ধ্রুবক _Float16 alpha = (_Float16) ((ধ্রুবক দশমিক *) (actx->octx->op_params))[2];
    ধ্রুবক _Float16 limit = (_Float16) ((ধ্রুবক দশমিক *) (actx->octx->op_params))[3];

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, limit, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -limit, limit, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_add_scalar_f16((uint8_t *) src1_ptr, src1_ptr, (_Float16) 1.0f, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f16(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f16_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

স্থির শূন্য swiglu_clamp_f16(ধ্রুবক _Float16 * restrict src0,
                             ধ্রুবক _Float16 * restrict src1,
                             _Float16 * restrict dst,
                             ধ্রুবক uint32_t                 অঙ্ক_rows,
                             ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;
    ধ্রুবক _Float16 limit = (_Float16) ((ধ্রুবক দশমিক *) (actx->octx->op_params))[3];

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, limit, nc);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -limit, limit, nc);
        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// geglu(x, g) = gelu(x) * g, gelu via tanh approximation
স্থির শূন্য geglu_f16(ধ্রুবক _Float16 * restrict src0,
                      ধ্রুবক _Float16 * restrict src1,
                      _Float16 * restrict dst,
                      ধ্রুবক uint32_t অঙ্ক_rows,
                      ধ্রুবক কাঠামো htp_act_context * actx) {
    htp_glu_op_preamble;

    জন্য (uint32_t ib = 0; ib < অঙ্ক_rows; ib++) {
        ধ্রুবক uint8_t * restrict src0_ptr = (ধ্রুবক uint8_t *) src0 + (ib * src0_row_আকার_aligned);
        ধ্রুবক uint8_t * restrict src1_ptr = (ধ্রুবক uint8_t *) src1 + (ib * src1_row_আকার_aligned);
        uint8_t * restrict dst_ptr        = (uint8_t *) dst + (ib * dst_row_আকার_aligned);

        hvx_geglu_f16_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#define DEFINE_GLU_PER_THREAD(TYPE, SUFFIX)                                                                            \
    স্থির শূন্য glu_##SUFFIX##_per_thread(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {                           \
        কাঠামো htp_act_context * actx = (কাঠামো htp_act_context *) তথ্য;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        কাঠামো htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : ফাঁকা;                          \
                                                                                                                       \
        size_t src0_row_আকার = actx->src0_row_আকার;                                                                    \
        size_t src1_row_আকার = actx->src1_row_আকার;                                                                    \
        size_t dst_row_আকার  = actx->dst_row_আকার;                                                                     \
                                                                                                                       \
        size_t src0_row_stride = actx->src0_row_stride;                                                                \
        size_t src1_row_stride = actx->src1_row_stride;                                                                \
                                                                                                                       \
        ধ্রুবক uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        ধ্রুবক uint32_t src0_nrows_per_thread = actx->src0_nrows_per_thread;                                            \
                                                                                                                       \
        ধ্রুবক uint32_t src0_start_row = actx->row_start + src0_nrows_per_thread * ith;                                 \
        ধ্রুবক uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, actx->row_start + src0_nrows);     \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        যদি (src0_start_row >= src0_end_row) {                                                                          \
            ফেরত;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        ধ্রুবক dma_addr_t তথ্য_src0 = actx->তথ্য_src0;                                                                  \
        ধ্রুবক dma_addr_t তথ্য_src1 = actx->তথ্য_src1;                                                                  \
        ধ্রুবক dma_addr_t তথ্য_dst  = actx->তথ্য_dst;                                                                   \
                                                                                                                       \
        ধ্রুবক size_t src0_row_আকার_aligned = actx->src0_row_আকার_aligned;                                              \
        ধ্রুবক size_t src1_row_আকার_aligned = actx->src1_row_আকার_aligned;                                              \
        ধ্রুবক size_t dst_row_আকার_aligned  = actx->dst_row_আকার_aligned;                                               \
                                                                                                                       \
        uint8_t * restrict src0_spad_তথ্য = actx->vtcm_src0 + (ith * actx->vtcm_src0_আকার_per_thread);                 \
        uint8_t * restrict src1_spad_তথ্য = actx->vtcm_src1 + (ith * actx->vtcm_src1_আকার_per_thread);                 \
        uint8_t * restrict dst_spad_তথ্য  = actx->vtcm_dst  + (ith * actx->vtcm_dst_আকার_per_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_আকার = actx->src0_spad_half_আকার;                                                        \
        size_t src1_spad_half_আকার = actx->src1_spad_half_আকার;                                                        \
        size_t dst_spad_half_আকার  = actx->dst_spad_half_আকার;                                                         \
                                                                                                                       \
        ধ্রুবক পূর্ণসংখ্যা BLOCK = actx->block;                                                                                 \
        যদি (BLOCK == 0) {                                                                                              \
            FARF(ERROR, "%s : VTCM reservation %zu is too small, needed %zu\n",                                        \
                 actx->op_str, actx->vtcm_src0_আকার_per_thread, src0_row_আকার_aligned);                                \
            ফেরত;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_queue * dma_q = actx->octx->ctx->dma[ith];                                                                 \
        glu_compute_##SUFFIX##_fn_t compute = (glu_compute_##SUFFIX##_fn_t) actx->compute;                             \
                                                                                                                       \
        জন্য (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; ir += BLOCK, spad_idx++) { \
            ধ্রুবক uint32_t block_আকার_2 = MIN(BLOCK, src0_end_row - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_queue_ঠেলো(dma_q,                                                                                      \
                           dma_make_তথ্য(তথ্য_dst, dst_spad_তথ্য + (spad_idx * dst_spad_half_আকার)),                   \
                           dst_row_আকার, dst_row_আকার_aligned, dst_row_আকার, 0);                                       \
                                                                                                                       \
            dma_queue_ঠেলো(dma_q,                                                                                      \
                dma_make_তথ্য(src0_spad_তথ্য + (spad_idx * src0_spad_half_আকার), তথ্য_src0 + (ir * src0_row_stride)),  \
                src0_row_আকার_aligned, src0_row_stride, src0_row_আকার, block_আকার_2);                                    \
                                                                                                                       \
            dma_queue_ঠেলো(dma_q,                                                                                      \
                dma_make_তথ্য(src1_spad_তথ্য + (spad_idx * src1_spad_half_আকার), তথ্য_src1 + (ir * src1_row_stride)),  \
                src1_row_আকার_aligned, src1_row_stride, src1_row_আকার, block_আকার_2);                                    \
        }                                                                                                              \
                                                                                                                       \
        জন্য (uint32_t ir = src0_start_row; ir < src0_end_row; ir += BLOCK) {                                           \
            ধ্রুবক uint32_t block_আকার_2 = MIN(BLOCK, src0_end_row - ir);                                                 \
                                                                                                                       \
            TYPE * dst_spad  = (TYPE *) dma_queue_বের_করো(dma_q).src;                                                      \
            TYPE * src0_spad = (TYPE *) dma_queue_বের_করো(dma_q).dst;                                                      \
            TYPE * src1_spad = (TYPE *) dma_queue_বের_করো(dma_q).dst;                                                      \
                                                                                                                       \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            compute(src0_spad, src1_spad, dst_spad, block_আকার_2, actx);                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(তথ্য_dst + (ir * dst_row_আকার), dst_spad),                             \
                           dst_row_আকার, dst_row_আকার_aligned, dst_row_আকার, block_আকার_2);                              \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            ধ্রুবক uint32_t pref_block = (ir + BLOCK * 2);                                                              \
            যদি (pref_block < src0_end_row) {                                                                           \
                ধ্রুবক uint32_t pref_block_আকার = MIN(BLOCK, src0_end_row - pref_block);                                \
                dma_queue_ঠেলো(dma_q, dma_make_তথ্য(src0_spad, তথ্য_src0 + (pref_block * src0_row_stride)),            \
                               src0_row_আকার_aligned, src0_row_stride, src0_row_আকার, pref_block_আকার);                \
                dma_queue_ঠেলো(dma_q, dma_make_তথ্য(src1_spad, তথ্য_src1 + (pref_block * src1_row_stride)),            \
                               src1_row_আকার_aligned, src1_row_stride, src1_row_আকার, pref_block_আকার);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_queue_flush(dma_q);                                                                                        \
    }

DEFINE_GLU_PER_THREAD(দশমিক, f32)
DEFINE_GLU_PER_THREAD(_Float16, f16)

স্থির পূর্ণসংখ্যা execute_op_activations_f32(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    যদি ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    glu_compute_f32_fn_t compute_fn = ফাঁকা;
    ধ্রুবক অক্ষর *         op_type    = ফাঁকা;

    নির্বাচন (octx->op) {
        ক্ষেত্র HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f32;
            op_type    = "swiglu-f32";
            বিরতি;

        ক্ষেত্র HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f32;
            op_type    = "swiglu-oai-f32";
            বিরতি;

        ক্ষেত্র HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f32;
            op_type    = "swiglu-clamp-f32";
            বিরতি;

        ক্ষেত্র HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f32;
            op_type    = "geglu-f32";
            বিরতি;

        ক্ষেত্র HTP_OP_GLU_GEGLU_QUICK:
            compute_fn = geglu_quick_f32;
            op_type    = "geglu-quick-f32";
            বিরতি;

        ক্ষেত্র HTP_OP_GLU_GEGLU_ERF:
            compute_fn = geglu_erf_f32;
            op_type    = "geglu-erf-f32";
            বিরতি;

        পূর্বনির্ধারিত:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুবক size_t dst_row_আকার = dst->ne[0] * SIZEOF_FP32;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, আকার(দশমিক), (uint32_t) dst_row_আকার, &rows_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ধ্রুবক size_t nc_bytes        = dst_row_আকার;
    ধ্রুবক size_t src0_row_আকার   = nc_bytes;
    ধ্রুবক size_t src1_row_আকার   = nc_bytes;
    ধ্রুবক size_t src0_row_stride = src0->nb[1];
    ধ্রুবক size_t src1_row_stride = src1 ? src1->nb[1] : src0->nb[1];

    ধ্রুবক size_t src0_row_আকার_aligned = hex_round_up(src0_row_আকার, VLEN);
    ধ্রুবক size_t src1_row_আকার_aligned = hex_round_up(src1_row_আকার, VLEN);
    ধ্রুবক size_t dst_row_আকার_aligned  = hex_round_up(dst_row_আকার, VLEN);

    কাঠামো htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_row_আকার_aligned, src1_row_আকার_aligned, dst_row_আকার_aligned, n_threads,
                              octx->ctx->vtcm_আকার);

    // Make sure the reserved vtcm size is sufficient
    যদি (L.vtcm_row_per_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_আকার,
             (src0_row_আকার_aligned + src1_row_আকার_aligned + dst_row_আকার_aligned) * n_threads);
        ফেরত HTP_STATUS_VTCM_TOO_SMALL;
    }

    যদি (src1) {
        FARF(উপরের_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_bytes_per_thread * n_threads,
             L.src1_bytes_per_thread * n_threads, L.dst_bytes_per_thread * n_threads);
    } অন্যথা {
        FARF(উপরের_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_bytes_per_thread * n_threads, L.src1_bytes_per_thread * n_threads, L.dst_bytes_per_thread * n_threads);
    }

    // Prepare context
    কাঠামো htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (শূন্য *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    actx.src0_row_আকার = src0_row_আকার;
    actx.src1_row_আকার = src1_row_আকার;
    actx.dst_row_আকার  = dst_row_আকার;

    actx.src0_row_আকার_aligned = src0_row_আকার_aligned;
    actx.src1_row_আকার_aligned = src1_row_আকার_aligned;
    actx.dst_row_আকার_aligned  = dst_row_আকার_aligned;

    actx.src0_row_stride = src0_row_stride;
    actx.src1_row_stride = src1_row_stride;

    uint8_t * ধ্রুবক base = (uint8_t *) octx->ctx->vtcm_base;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    actx.vtcm_src0_আকার_per_thread = L.src0_bytes_per_thread;
    actx.vtcm_src1_আকার_per_thread = L.src1_bytes_per_thread;
    actx.vtcm_dst_আকার_per_thread  = L.dst_bytes_per_thread;

    actx.src0_spad_half_আকার = L.src0_bytes_per_thread / 2;
    actx.src1_spad_half_আকার = L.src1_bytes_per_thread / 2;
    actx.dst_spad_half_আকার  = L.dst_bytes_per_thread / 2;

    actx.block = actx.src0_spad_half_আকার / actx.src0_row_আকার_aligned;
    actx.src0_nrows = nrows;
    actx.row_start  = row_start;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t তথ্য_src0 = src0->তথ্য;
    dma_addr_t তথ্য_src1 = src1 ? src1->তথ্য : 0;

    যদি (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU ||
                  octx->op == HTP_OP_GLU_GEGLU_QUICK ||
                  octx->op == HTP_OP_GLU_GEGLU_ERF)) {
         ধ্রুবক int32_t swapped = octx->op_params[1];
         তথ্য_src1 = তথ্য_src0;
         actx.src1_row_আকার = actx.src0_row_আকার;

         size_t nc_in_bytes = actx.nc * SIZEOF_FP32;
         যদি (swapped) {
             তথ্য_src0 += nc_in_bytes;
         } অন্যথা {
             তথ্য_src1 += nc_in_bytes;
         }
    }

    actx.তথ্য_src0 = তথ্য_src0;
    actx.তথ্য_src1 = তথ্য_src1;
    actx.তথ্য_dst  = dst->তথ্য;

    work_queue_run(octx->ctx->work_queue, (worker_callback_t)glu_f32_per_thread, &actx, n_threads);
    ফেরত HTP_STATUS_OK;
}

স্থির পূর্ণসংখ্যা execute_op_activations_f16(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    যদি ((dst->ne[0] * SIZEOF_FP16) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    glu_compute_f16_fn_t compute_fn = ফাঁকা;
    ধ্রুবক অক্ষর *         op_type    = ফাঁকা;

    নির্বাচন (octx->op) {
        ক্ষেত্র HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f16;
            op_type    = "swiglu-f16";
            বিরতি;

        ক্ষেত্র HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f16;
            op_type    = "swiglu-oai-f16";
            বিরতি;

        ক্ষেত্র HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f16;
            op_type    = "swiglu-clamp-f16";
            বিরতি;

        ক্ষেত্র HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f16;
            op_type    = "geglu-f16";
            বিরতি;

        পূর্বনির্ধারিত:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুবক size_t dst_row_আকার = dst->ne[0] * SIZEOF_FP16;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, আকার(_Float16), (uint32_t) dst_row_আকার, &rows_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ধ্রুবক size_t nc_bytes        = dst_row_আকার;
    ধ্রুবক size_t src0_row_আকার   = nc_bytes;
    ধ্রুবক size_t src1_row_আকার   = nc_bytes;
    ধ্রুবক size_t src0_row_stride = src0->nb[1];
    ধ্রুবক size_t src1_row_stride = src1 ? src1->nb[1] : src0->nb[1];

    ধ্রুবক size_t src0_row_আকার_aligned = hex_round_up(src0_row_আকার, VLEN);
    ধ্রুবক size_t src1_row_আকার_aligned = hex_round_up(src1_row_আকার, VLEN);
    ধ্রুবক size_t dst_row_আকার_aligned  = hex_round_up(dst_row_আকার, VLEN);

    কাঠামো htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_row_আকার_aligned, src1_row_আকার_aligned, dst_row_আকার_aligned, n_threads,
                              octx->ctx->vtcm_আকার);

    // Make sure the reserved vtcm size is sufficient
    যদি (L.vtcm_row_per_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_আকার,
             (src0_row_আকার_aligned + src1_row_আকার_aligned + dst_row_আকার_aligned) * n_threads);
        ফেরত HTP_STATUS_VTCM_TOO_SMALL;
    }

    যদি (src1) {
        FARF(উপরের_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_bytes_per_thread * n_threads,
             L.src1_bytes_per_thread * n_threads, L.dst_bytes_per_thread * n_threads);
    } অন্যথা {
        FARF(উপরের_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_bytes_per_thread * n_threads, L.src1_bytes_per_thread * n_threads, L.dst_bytes_per_thread * n_threads);
    }

    // Prepare context
    কাঠামো htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (শূন্য *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    actx.src0_row_আকার = src0_row_আকার;
    actx.src1_row_আকার = src1_row_আকার;
    actx.dst_row_আকার  = dst_row_আকার;

    actx.src0_row_আকার_aligned = src0_row_আকার_aligned;
    actx.src1_row_আকার_aligned = src1_row_আকার_aligned;
    actx.dst_row_আকার_aligned  = dst_row_আকার_aligned;

    actx.src0_row_stride = src0_row_stride;
    actx.src1_row_stride = src1_row_stride;

    uint8_t * ধ্রুবক base = (uint8_t *) octx->ctx->vtcm_base;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    actx.vtcm_src0_আকার_per_thread = L.src0_bytes_per_thread;
    actx.vtcm_src1_আকার_per_thread = L.src1_bytes_per_thread;
    actx.vtcm_dst_আকার_per_thread  = L.dst_bytes_per_thread;

    actx.src0_spad_half_আকার = L.src0_bytes_per_thread / 2;
    actx.src1_spad_half_আকার = L.src1_bytes_per_thread / 2;
    actx.dst_spad_half_আকার  = L.dst_bytes_per_thread / 2;

    actx.block = actx.src0_spad_half_আকার / actx.src0_row_আকার_aligned;
    actx.src0_nrows = nrows;
    actx.row_start  = row_start;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t তথ্য_src0 = src0->তথ্য;
    dma_addr_t তথ্য_src1 = src1 ? src1->তথ্য : 0;

    যদি (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU)) {
         ধ্রুবক int32_t swapped = octx->op_params[1];
         তথ্য_src1 = তথ্য_src0;
         actx.src1_row_আকার = actx.src0_row_আকার;

         size_t nc_in_bytes = actx.nc * SIZEOF_FP16;
         যদি (swapped) {
             তথ্য_src0 += nc_in_bytes;
         } অন্যথা {
             তথ্য_src1 += nc_in_bytes;
         }
    }

    actx.তথ্য_src0 = তথ্য_src0;
    actx.তথ্য_src1 = তথ্য_src1;
    actx.তথ্য_dst  = dst->তথ্য;

    work_queue_run(octx->ctx->work_queue, (worker_callback_t)glu_f16_per_thread, &actx, n_threads);
    ফেরত HTP_STATUS_OK;
}


পূর্ণসংখ্যা op_activations(কাঠামো htp_ops_context * octx) {
    নির্বাচন (octx->src[0]->type) {
        ক্ষেত্র HTP_TYPE_F32:
            ফেরত execute_op_activations_f32(octx);

        ক্ষেত্র HTP_TYPE_F16:
            ফেরত execute_op_activations_f16(octx);

        পূর্বনির্ধারিত:
            ফেরত HTP_STATUS_NO_SUPPORT;
    }
}
