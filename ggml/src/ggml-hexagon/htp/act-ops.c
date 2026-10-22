#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "hex-common.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"

#تعریف_کن htp_act_preamble                                 \
    ثابت بنية htp_tensor * src0 = actx->octx->src[0]; \
    ثابت بنية htp_tensor * src1 = actx->octx->src[1]; \
    ثابت بنية htp_tensor * dst  = actx->octx->dst;    \
                                                         \
    ثابت uint32_t ne00 = src0->ne[0];                   \
    ثابت uint32_t ne01 = src0->ne[1];                   \
    ثابت uint32_t ne02 = src0->ne[2];                   \
    ثابت uint32_t ne03 = src0->ne[3];                   \
                                                         \
    ثابت uint32_t nb00 = src0->nb[0];                   \
    ثابت uint32_t nb01 = src0->nb[1];                   \
    ثابت uint32_t nb02 = src0->nb[2];                   \
    ثابت uint32_t nb03 = src0->nb[3];                   \
                                                         \
    ثابت uint32_t ne10 = src1 ? src1->ne[0] : 0;        \
    ثابت uint32_t ne11 = src1 ? src1->ne[1] : 0;        \
    ثابت uint32_t ne12 = src1 ? src1->ne[2] : 0;        \
    ثابت uint32_t ne13 = src1 ? src1->ne[3] : 0;        \
                                                         \
    ثابت uint32_t nb10 = src1 ? src1->nb[0] : 0;        \
    ثابت uint32_t nb11 = src1 ? src1->nb[1] : 0;        \
    ثابت uint32_t nb12 = src1 ? src1->nb[2] : 0;        \
    ثابت uint32_t nb13 = src1 ? src1->nb[3] : 0;        \
                                                         \
    ثابت uint32_t ne0 = dst->ne[0];                     \
    ثابت uint32_t ne1 = dst->ne[1];                     \
    ثابت uint32_t ne2 = dst->ne[2];                     \
    ثابت uint32_t ne3 = dst->ne[3];                     \
                                                         \
    ثابت uint32_t nb0 = dst->nb[0];                     \
    ثابت uint32_t nb1 = dst->nb[1];                     \
    ثابت uint32_t nb2 = dst->nb[2];                     \
    ثابت uint32_t nb3 = dst->nb[3];

بنية htp_act_context;

تعريفنوع فراغ (*glu_compute_f32_fn_t)(ثابت عائم * محدود src0,
                                     ثابت عائم * محدود src1,
                                     عائم * محدود dst,
                                     ثابت uint32_t رقم_ردیفها_2,
                                     ثابت بنية htp_act_context * actx);

تعريفنوع فراغ (*glu_compute_f16_fn_t)(ثابت _Float16 * محدود src0,
                                     ثابت _Float16 * محدود src1,
                                     _Float16 * محدود dst,
                                     ثابت uint32_t رقم_ردیفها_2,
                                     ثابت بنية htp_act_context * actx);

بنية htp_act_context {
    بنية htp_ops_context * octx;

    فراغ *                   compute;
    ثابت حرف *             op_str;

    // Precomputed values
    dma_addr_t               داده_src0;
    dma_addr_t               داده_src1;
    dma_addr_t               داده_dst;

    size_t                   src0_ردیف_اندازه;
    size_t                   src1_ردیف_اندازه;
    size_t                   dst_ردیف_اندازه;

    size_t                   src0_ردیف_stride;
    size_t                   src1_ردیف_stride;

    size_t                   src0_ردیف_اندازه_aligned;
    size_t                   src1_ردیف_اندازه_aligned;
    size_t                   dst_ردیف_اندازه_aligned;

    size_t                   src0_spad_half_اندازه;
    size_t                   src1_spad_half_اندازه;
    size_t                   dst_spad_half_اندازه;

    uint32_t                 بلوک;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_در_هر_thread;
    uint32_t                 ردیف_شروع;
    صحيح                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_اندازه_در_هر_thread;
    size_t                   vtcm_src1_اندازه_در_هر_thread;
    size_t                   vtcm_dst_اندازه_در_هر_thread;
};

بنية htp_act_vtcm_layout {
    size_t مجموع_بایتها;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_بایتها_در_هر_thread;
    size_t src1_بایتها_در_هر_thread;
    size_t dst_بایتها_در_هر_thread;

    uint32_t vtcm_ردیف_در_هر_thread;
};

ساكن مضمن فراغ htp_act_vtcm_layout_build(بنية htp_act_vtcm_layout * L,
                                             size_t                       src0_ردیف_اندازه_aligned,
                                             size_t                       src1_ردیف_اندازه_aligned,
                                             size_t                       dst_ردیف_اندازه_aligned,
                                             uint32_t                     n_نخها,
                                             size_t                       vtcm_اندازه) {
    ثابت size_t   spad_اندازه_در_هر_ردیف   = src0_ردیف_اندازه_aligned + src1_ردیف_اندازه_aligned + dst_ردیف_اندازه_aligned;
    ثابت uint32_t vtcm_ردیف_در_هر_thread = (uint32_t) (vtcm_اندازه / (n_نخها * spad_اندازه_در_هر_ردیف));

    L->vtcm_ردیف_در_هر_thread = vtcm_ردیف_در_هر_thread;

    L->src0_بایتها_در_هر_thread = src0_ردیف_اندازه_aligned * vtcm_ردیف_در_هر_thread;
    L->src1_بایتها_در_هر_thread = src1_ردیف_اندازه_aligned * vtcm_ردیف_در_هر_thread;
    L->dst_بایتها_در_هر_thread  = dst_ردیف_اندازه_aligned * vtcm_ردیف_در_هر_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_بایتها_در_هر_thread * n_نخها;
    L->off_dst  = L->off_src1 + L->src1_بایتها_در_هر_thread * n_نخها;

    L->مجموع_بایتها = L->off_dst + L->dst_بایتها_در_هر_thread * n_نخها;
}

#تعریف_کن htp_glu_op_preamble                                            \
    ثابت size_t src0_ردیف_اندازه_aligned = actx->src0_ردیف_اندازه_aligned;  \
    ثابت size_t src1_ردیف_اندازه_aligned = actx->src1_ردیف_اندازه_aligned;  \
    ثابت size_t dst_ردیف_اندازه_aligned  = actx->dst_ردیف_اندازه_aligned;   \
    ثابت صحيح    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
ساكن فراغ swiglu_f32(ثابت عائم * محدود src0,
                                                 ثابت عائم * محدود src1,
                                                 عائم * محدود dst,
                                                 ثابت uint32_t رقم_ردیفها_2,
                                                 ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساكن فراغ swiglu_oai_f32(ثابت عائم * محدود src0,
                                                     ثابت عائم * محدود src1,
                                                     عائم * محدود dst,
                                                     ثابت uint32_t رقم_ردیفها_2,
                                                     ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت عائم alpha = ((ثابت عائم *) (actx->octx->op_params))[2];
    ثابت عائم حد = ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد, حد, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جمع_کن_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f32(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f32_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ساكن فراغ swiglu_clamp_f32(ثابت عائم * محدود src0,
                                                       ثابت عائم * محدود src1,
                                                       عائم * محدود dst,
                                                       ثابت uint32_t                 رقم_ردیفها_2,
                                                       ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت عائم حد = ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد, nc);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد, حد, nc);
        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ساكن ثابت عائم GELU_COEF_A     = 0.044715f;
ساكن ثابت عائم SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

ساكن مضمن HVX_وکتور hvx_vec_fast_sigmoid_f32_2it(HVX_وکتور v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_وکتور درون_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_وکتور x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(درون_int));
    HVX_وکتور xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_وکتور v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_وکتور v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_وکتور v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, درون_int, 24);

    HVX_وکتور v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_وکتور v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_وکتور two_sf = hvx_vec_splat_f32(2.0f);
    HVX_وکتور i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_وکتور r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_وکتور res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    أرجع Q6_Vsf_equals_Vqf32(res);
}

ساكن مضمن HVX_وکتور hvx_vec_fast_sigmoid_f32_guard_2it(HVX_وکتور v,
                                                            HVX_وکتور one,
                                                            HVX_وکتور max_exp,
                                                            HVX_وکتور min_exp) {
    ثابت HVX_وکتور_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ثابت HVX_وکتور_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_وکتور بیرون = hvx_vec_fast_sigmoid_f32_2it(v);
    بیرون            = Q6_V_vmux_QVV(pred_max, بیرون, one);
    أرجع Q6_V_vmux_QVV(pred_min, بیرون, Q6_V_vzero());
}

ساكن مضمن فراغ hvx_geglu_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بیعلامت دراز) dst  % 128 == 0);
    أكد((بیعلامت دراز) src0 % 128 == 0);
    أكد((بیعلامت دراز) src1 % 128 == 0);

    HVX_وکتور * محدود vdst        = (HVX_وکتور *) dst;
    ثابت HVX_وکتور * محدود vsrc0 = (ثابت HVX_وکتور *) src0;
    ثابت HVX_وکتور * محدود vsrc1 = (ثابت HVX_وکتور *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت عائم GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    ثابت HVX_وکتور v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    ثابت HVX_وکتور v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    ثابت HVX_وکتور v_one               = hvx_vec_splat_f32(1.0f);
    ثابت HVX_وکتور v_max_exp           = hvx_vec_splat_f32(87.0f);
    ثابت HVX_وکتور v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_وکتور x = vsrc0[i];
        HVX_وکتور g = vsrc1[i];

        HVX_وکتور x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_وکتور coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جمع_کن_f32_f32(coef, v_sqrt_2_pi);
        HVX_وکتور inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner = inner + inner
        HVX_وکتور y2 = hvx_vec_جمع_کن_f32_f32(inner, inner);

        // Fast sigmoid approximation (2 iterations)
        HVX_وکتور sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_وکتور gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    إذا (nloe) {
        HVX_وکتور x = vsrc0[i];
        HVX_وکتور g = vsrc1[i];

        HVX_وکتور x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_وکتور coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جمع_کن_f32_f32(coef, v_sqrt_2_pi);
        HVX_وکتور inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_وکتور y2 = hvx_vec_جمع_کن_f32_f32(inner, inner);

        HVX_وکتور sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_وکتور gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        HVX_وکتور res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), res);
    }
}

ساكن مضمن فراغ hvx_geglu_سریع_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بیعلامت دراز) dst  % 128 == 0);
    أكد((بیعلامت دراز) src0 % 128 == 0);
    أكد((بیعلامت دراز) src1 % 128 == 0);

    HVX_وکتور * محدود vdst        = (HVX_وکتور *) dst;
    ثابت HVX_وکتور * محدود vsrc0 = (ثابت HVX_وکتور *) src0;
    ثابت HVX_وکتور * محدود vsrc1 = (ثابت HVX_وکتور *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_وکتور v_scale    = hvx_vec_splat_f32(1.702f);
    ثابت HVX_وکتور v_one      = hvx_vec_splat_f32(1.0f);
    ثابت HVX_وکتور v_max_exp  = hvx_vec_splat_f32(87.0f);
    ثابت HVX_وکتور v_min_exp  = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_وکتور x = vsrc0[i];
        HVX_وکتور g = vsrc1[i];
        HVX_وکتور scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_وکتور sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
    }

    إذا (nloe) {
        HVX_وکتور x = vsrc0[i];
        HVX_وکتور g = vsrc1[i];
        HVX_وکتور scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_وکتور sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        HVX_وکتور نتیجه = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), نتیجه);
    }
}

ساكن مضمن فراغ hvx_geglu_erf_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بیعلامت دراز) dst  % 128 == 0);
    أكد((بیعلامت دراز) src0 % 128 == 0);
    أكد((بیعلامت دراز) src1 % 128 == 0);

    HVX_وکتور * محدود vdst        = (HVX_وکتور *) dst;
    ثابت HVX_وکتور * محدود vsrc0 = (ثابت HVX_وکتور *) src0;
    ثابت HVX_وکتور * محدود vsrc1 = (ثابت HVX_وکتور *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_وکتور x = vsrc0[i];
        HVX_وکتور g = vsrc1[i];
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
    }

    إذا (nloe) {
        HVX_وکتور x = vsrc0[i];
        HVX_وکتور g = vsrc1[i];
        HVX_وکتور نتیجه = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), نتیجه);
    }
}

ساكن مضمن فراغ hvx_geglu_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بیعلامت دراز) dst  % 128 == 0);
    أكد((بیعلامت دراز) src0 % 128 == 0);
    أكد((بیعلامت دراز) src1 % 128 == 0);

    HVX_وکتور * محدود vdst        = (HVX_وکتور *) dst;
    ثابت HVX_وکتور * محدود vsrc0 = (ثابت HVX_وکتور *) src0;
    ثابت HVX_وکتور * محدود vsrc1 = (ثابت HVX_وکتور *) src1;

    ثابت uint32_t epv  = 128 / حجم(_Float16);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_وکتور v_coef_a    = hvx_vec_splat_f16((_Float16) GELU_COEF_A);
    ثابت HVX_وکتور v_sqrt_2_pi = hvx_vec_splat_f16((_Float16) SQRT_2_OVER_PI);
    ثابت HVX_وکتور v_one       = hvx_vec_splat_f16((_Float16) 1.0f);
    ثابت HVX_وکتور v_half      = hvx_vec_splat_f16((_Float16) 0.5f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_وکتور x = vsrc0[i];
        HVX_وکتور g = vsrc1[i];

        // gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))
        HVX_وکتور x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_وکتور coef  = hvx_vec_جمع_کن_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_وکتور inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_وکتور t = hvx_vec_جمع_کن_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_وکتور gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        vdst[i] = hvx_vec_mul_f16_f16(gelu_x, g);
    }

    إذا (nloe) {
        HVX_وکتور x = vsrc0[i];
        HVX_وکتور g = vsrc1[i];

        HVX_وکتور x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_وکتور coef  = hvx_vec_جمع_کن_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_وکتور inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_وکتور t = hvx_vec_جمع_کن_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_وکتور gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        HVX_وکتور res    = hvx_vec_mul_f16_f16(gelu_x, g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(_Float16), res);
    }
}

// geglu(x, g) = gelu(x) * g
ساكن فراغ geglu_f32(ثابت عائم * محدود src0,
                                                ثابت عائم * محدود src1,
                                                عائم * محدود dst,
                                                ثابت uint32_t رقم_ردیفها_2,
                                                ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_quick(x, g) = x * sigmoid(1.702 * x) * g
ساكن فراغ geglu_سریع_f32(ثابت عائم * محدود src0,
                            ثابت عائم * محدود src1,
                            عائم * محدود dst,
                            ثابت uint32_t رقم_ردیفها_2,
                            ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        hvx_geglu_سریع_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_erf(x, g) = gelu_erf(x) * g
ساكن فراغ geglu_erf_f32(ثابت عائم * محدود src0,
                          ثابت عائم * محدود src1,
                          عائم * محدود dst,
                          ثابت uint32_t رقم_ردیفها_2,
                          ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        hvx_geglu_erf_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// swiglu(x) = x1 * sigmoid(x0)
ساكن فراغ swiglu_f16(ثابت _Float16 * محدود src0,
                       ثابت _Float16 * محدود src1,
                       _Float16 * محدود dst,
                       ثابت uint32_t رقم_ردیفها_2,
                       ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساكن فراغ swiglu_oai_f16(ثابت _Float16 * محدود src0,
                           ثابت _Float16 * محدود src1,
                           _Float16 * محدود dst,
                           ثابت uint32_t رقم_ردیفها_2,
                           ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 alpha = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[2];
    ثابت _Float16 حد = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, حد, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -حد, حد, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جمع_کن_scalar_f16((uint8_t *) src1_ptr, src1_ptr, (_Float16) 1.0f, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f16(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f16_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ساكن فراغ swiglu_clamp_f16(ثابت _Float16 * محدود src0,
                             ثابت _Float16 * محدود src1,
                             _Float16 * محدود dst,
                             ثابت uint32_t                 رقم_ردیفها_2,
                             ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 حد = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, حد, nc);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -حد, حد, nc);
        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// geglu(x, g) = gelu(x) * g, gelu via tanh approximation
ساكن فراغ geglu_f16(ثابت _Float16 * محدود src0,
                      ثابت _Float16 * محدود src1,
                      _Float16 * محدود dst,
                      ثابت uint32_t رقم_ردیفها_2,
                      ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < رقم_ردیفها_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_ردیف_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_ردیف_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_ردیف_اندازه_aligned);

        hvx_geglu_f16_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#تعریف_کن DEFINE_GLU_در_هر_THREAD(TYPE, SUFFIX)                                                                            \
    ساكن فراغ glu_##SUFFIX##_per_thread(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {                           \
        بنية htp_act_context * actx = (بنية htp_act_context *) داده;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        بنية htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : عدم;                          \
                                                                                                                       \
        size_t src0_ردیف_اندازه = actx->src0_ردیف_اندازه;                                                                    \
        size_t src1_ردیف_اندازه = actx->src1_ردیف_اندازه;                                                                    \
        size_t dst_ردیف_اندازه  = actx->dst_ردیف_اندازه;                                                                     \
                                                                                                                       \
        size_t src0_ردیف_stride = actx->src0_ردیف_stride;                                                                \
        size_t src1_ردیف_stride = actx->src1_ردیف_stride;                                                                \
                                                                                                                       \
        ثابت uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        ثابت uint32_t src0_nrows_در_هر_thread = actx->src0_nrows_در_هر_thread;                                            \
                                                                                                                       \
        ثابت uint32_t src0_شروع_ردیف = actx->ردیف_شروع + src0_nrows_در_هر_thread * ith;                                 \
        ثابت uint32_t src0_انتها_ردیف   = MIN(src0_شروع_ردیف + src0_nrows_در_هر_thread, actx->ردیف_شروع + src0_nrows);     \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        إذا (src0_شروع_ردیف >= src0_انتها_ردیف) {                                                                          \
            أرجع;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        ثابت dma_addr_t داده_src0 = actx->داده_src0;                                                                  \
        ثابت dma_addr_t داده_src1 = actx->داده_src1;                                                                  \
        ثابت dma_addr_t داده_dst  = actx->داده_dst;                                                                   \
                                                                                                                       \
        ثابت size_t src0_ردیف_اندازه_aligned = actx->src0_ردیف_اندازه_aligned;                                              \
        ثابت size_t src1_ردیف_اندازه_aligned = actx->src1_ردیف_اندازه_aligned;                                              \
        ثابت size_t dst_ردیف_اندازه_aligned  = actx->dst_ردیف_اندازه_aligned;                                               \
                                                                                                                       \
        uint8_t * محدود src0_spad_داده = actx->vtcm_src0 + (ith * actx->vtcm_src0_اندازه_در_هر_thread);                 \
        uint8_t * محدود src1_spad_داده = actx->vtcm_src1 + (ith * actx->vtcm_src1_اندازه_در_هر_thread);                 \
        uint8_t * محدود dst_spad_داده  = actx->vtcm_dst  + (ith * actx->vtcm_dst_اندازه_در_هر_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_اندازه = actx->src0_spad_half_اندازه;                                                        \
        size_t src1_spad_half_اندازه = actx->src1_spad_half_اندازه;                                                        \
        size_t dst_spad_half_اندازه  = actx->dst_spad_half_اندازه;                                                         \
                                                                                                                       \
        ثابت صحيح بلوک_2 = actx->بلوک;                                                                                 \
        إذا (بلوک_2 == 0) {                                                                                              \
            FARF(ERROR, "%s : VTCM reservation %zu is too small, needed %zu\n",                                        \
                 actx->op_str, actx->vtcm_src0_اندازه_در_هر_thread, src0_ردیف_اندازه_aligned);                                \
            أرجع;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_صف * dma_q = actx->octx->ctx->dma[ith];                                                                 \
        glu_compute_##SUFFIX##_fn_t compute = (glu_compute_##SUFFIX##_fn_t) actx->compute;                             \
                                                                                                                       \
        لكل (uint32_t ir = src0_شروع_ردیف, spad_idx = 0; ir < src0_انتها_ردیف && spad_idx < 2; ir += بلوک_2, spad_idx++) { \
            ثابت uint32_t بلوک_اندازه_3 = MIN(بلوک_2, src0_انتها_ردیف - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_صف_افزودن(dma_q,                                                                                      \
                           dma_make_داده(داده_dst, dst_spad_داده + (spad_idx * dst_spad_half_اندازه)),                   \
                           dst_ردیف_اندازه, dst_ردیف_اندازه_aligned, dst_ردیف_اندازه, 0);                                       \
                                                                                                                       \
            dma_صف_افزودن(dma_q,                                                                                      \
                dma_make_داده(src0_spad_داده + (spad_idx * src0_spad_half_اندازه), داده_src0 + (ir * src0_ردیف_stride)),  \
                src0_ردیف_اندازه_aligned, src0_ردیف_stride, src0_ردیف_اندازه, بلوک_اندازه_3);                                    \
                                                                                                                       \
            dma_صف_افزودن(dma_q,                                                                                      \
                dma_make_داده(src1_spad_داده + (spad_idx * src1_spad_half_اندازه), داده_src1 + (ir * src1_ردیف_stride)),  \
                src1_ردیف_اندازه_aligned, src1_ردیف_stride, src1_ردیف_اندازه, بلوک_اندازه_3);                                    \
        }                                                                                                              \
                                                                                                                       \
        لكل (uint32_t ir = src0_شروع_ردیف; ir < src0_انتها_ردیف; ir += بلوک_2) {                                           \
            ثابت uint32_t بلوک_اندازه_3 = MIN(بلوک_2, src0_انتها_ردیف - ir);                                                 \
                                                                                                                       \
            TYPE * dst_spad  = (TYPE *) dma_صف_برداشت(dma_q).src;                                                      \
            TYPE * src0_spad = (TYPE *) dma_صف_برداشت(dma_q).dst;                                                      \
            TYPE * src1_spad = (TYPE *) dma_صف_برداشت(dma_q).dst;                                                      \
                                                                                                                       \
            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            compute(src0_spad, src1_spad, dst_spad, بلوک_اندازه_3, actx);                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_صف_افزودن(dma_q, dma_make_داده(داده_dst + (ir * dst_ردیف_اندازه), dst_spad),                             \
                           dst_ردیف_اندازه, dst_ردیف_اندازه_aligned, dst_ردیف_اندازه, بلوک_اندازه_3);                              \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            ثابت uint32_t pref_بلوک = (ir + بلوک_2 * 2);                                                              \
            إذا (pref_بلوک < src0_انتها_ردیف) {                                                                           \
                ثابت uint32_t pref_بلوک_اندازه = MIN(بلوک_2, src0_انتها_ردیف - pref_بلوک);                                \
                dma_صف_افزودن(dma_q, dma_make_داده(src0_spad, داده_src0 + (pref_بلوک * src0_ردیف_stride)),            \
                               src0_ردیف_اندازه_aligned, src0_ردیف_stride, src0_ردیف_اندازه, pref_بلوک_اندازه);                \
                dma_صف_افزودن(dma_q, dma_make_داده(src1_spad, داده_src1 + (pref_بلوک * src1_ردیف_stride)),            \
                               src1_ردیف_اندازه_aligned, src1_ردیف_stride, src1_ردیف_اندازه, pref_بلوک_اندازه);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_صف_flush(dma_q);                                                                                        \
    }

DEFINE_GLU_در_هر_THREAD(عائم, f32)
DEFINE_GLU_در_هر_THREAD(_Float16, f16)

ساكن صحيح execute_op_activations_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    glu_compute_f32_fn_t compute_fn = عدم;
    ثابت حرف *         op_type    = عدم;

    بدل (octx->op) {
        حالة HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f32;
            op_type    = "swiglu-f32";
            اكسر;

        حالة HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f32;
            op_type    = "swiglu-oai-f32";
            اكسر;

        حالة HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f32;
            op_type    = "swiglu-clamp-f32";
            اكسر;

        حالة HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f32;
            op_type    = "geglu-f32";
            اكسر;

        حالة HTP_OP_GLU_GEGLU_سریع:
            compute_fn = geglu_سریع_f32;
            op_type    = "geglu-quick-f32";
            اكسر;

        حالة HTP_OP_GLU_GEGLU_ERF:
            compute_fn = geglu_erf_f32;
            op_type    = "geglu-erf-f32";
            اكسر;

        افتراضي:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_ردیف_اندازه = dst->ne[0] * SIZEOF_FP32;

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمار > 1) {
        uint32_t ردیفها_در_هر_chunk = 0;
        htp_tensor_mdev_ردیفها_در_هر_chunk(dst, حجم(عائم), (uint32_t) dst_ردیف_اندازه, &ردیفها_در_هر_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(src0_nrows, ردیفها_در_هر_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بایتها        = dst_ردیف_اندازه;
    ثابت size_t src0_ردیف_اندازه   = nc_بایتها;
    ثابت size_t src1_ردیف_اندازه   = nc_بایتها;
    ثابت size_t src0_ردیف_stride = src0->nb[1];
    ثابت size_t src1_ردیف_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_ردیف_اندازه_aligned = hex_دور_به_بالا(src0_ردیف_اندازه, VLEN);
    ثابت size_t src1_ردیف_اندازه_aligned = hex_دور_به_بالا(src1_ردیف_اندازه, VLEN);
    ثابت size_t dst_ردیف_اندازه_aligned  = hex_دور_به_بالا(dst_ردیف_اندازه, VLEN);

    بنية htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_ردیف_اندازه_aligned, src1_ردیف_اندازه_aligned, dst_ردیف_اندازه_aligned, n_نخها,
                              octx->ctx->vtcm_اندازه);

    // Make sure the reserved vtcm size is sufficient
    إذا (L.vtcm_ردیف_در_هر_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_اندازه,
             (src0_ردیف_اندازه_aligned + src1_ردیف_اندازه_aligned + dst_ردیف_اندازه_aligned) * n_نخها);
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    إذا (src1) {
        FARF(بالا_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بایتها_در_هر_thread * n_نخها,
             L.src1_بایتها_در_هر_thread * n_نخها, L.dst_بایتها_در_هر_thread * n_نخها);
    } وإلا {
        FARF(بالا_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بایتها_در_هر_thread * n_نخها, L.src1_بایتها_در_هر_thread * n_نخها, L.dst_بایتها_در_هر_thread * n_نخها);
    }

    // Prepare context
    بنية htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (فراغ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_در_هر_thread = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div);

    actx.src0_ردیف_اندازه = src0_ردیف_اندازه;
    actx.src1_ردیف_اندازه = src1_ردیف_اندازه;
    actx.dst_ردیف_اندازه  = dst_ردیف_اندازه;

    actx.src0_ردیف_اندازه_aligned = src0_ردیف_اندازه_aligned;
    actx.src1_ردیف_اندازه_aligned = src1_ردیف_اندازه_aligned;
    actx.dst_ردیف_اندازه_aligned  = dst_ردیف_اندازه_aligned;

    actx.src0_ردیف_stride = src0_ردیف_stride;
    actx.src1_ردیف_stride = src1_ردیف_stride;

    uint8_t * ثابت پایه = (uint8_t *) octx->ctx->vtcm_پایه;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_dst);

    actx.vtcm_src0_اندازه_در_هر_thread = L.src0_بایتها_در_هر_thread;
    actx.vtcm_src1_اندازه_در_هر_thread = L.src1_بایتها_در_هر_thread;
    actx.vtcm_dst_اندازه_در_هر_thread  = L.dst_بایتها_در_هر_thread;

    actx.src0_spad_half_اندازه = L.src0_بایتها_در_هر_thread / 2;
    actx.src1_spad_half_اندازه = L.src1_بایتها_در_هر_thread / 2;
    actx.dst_spad_half_اندازه  = L.dst_بایتها_در_هر_thread / 2;

    actx.بلوک = actx.src0_spad_half_اندازه / actx.src0_ردیف_اندازه_aligned;
    actx.src0_nrows = nrows;
    actx.ردیف_شروع  = ردیف_شروع;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t داده_src0 = src0->داده;
    dma_addr_t داده_src1 = src1 ? src1->داده : 0;

    إذا (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU ||
                  octx->op == HTP_OP_GLU_GEGLU_سریع ||
                  octx->op == HTP_OP_GLU_GEGLU_ERF)) {
         ثابت int32_t swapped = octx->op_params[1];
         داده_src1 = داده_src0;
         actx.src1_ردیف_اندازه = actx.src0_ردیف_اندازه;

         size_t nc_درون_بایتها = actx.nc * SIZEOF_FP32;
         إذا (swapped) {
             داده_src0 += nc_درون_بایتها;
         } وإلا {
             داده_src1 += nc_درون_بایتها;
         }
    }

    actx.داده_src0 = داده_src0;
    actx.داده_src1 = داده_src1;
    actx.داده_dst  = dst->داده;

    work_صف_پیاپی(octx->ctx->work_صف, (کارگر_callback_t)glu_f32_در_هر_thread, &actx, n_نخها);
    أرجع HTP_وضعیت_OK;
}

ساكن صحيح execute_op_activations_f16(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا ((dst->ne[0] * SIZEOF_FP16) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    glu_compute_f16_fn_t compute_fn = عدم;
    ثابت حرف *         op_type    = عدم;

    بدل (octx->op) {
        حالة HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f16;
            op_type    = "swiglu-f16";
            اكسر;

        حالة HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f16;
            op_type    = "swiglu-oai-f16";
            اكسر;

        حالة HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f16;
            op_type    = "swiglu-clamp-f16";
            اكسر;

        حالة HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f16;
            op_type    = "geglu-f16";
            اكسر;

        افتراضي:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_ردیف_اندازه = dst->ne[0] * SIZEOF_FP16;

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمار > 1) {
        uint32_t ردیفها_در_هر_chunk = 0;
        htp_tensor_mdev_ردیفها_در_هر_chunk(dst, حجم(_Float16), (uint32_t) dst_ردیف_اندازه, &ردیفها_در_هر_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(src0_nrows, ردیفها_در_هر_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بایتها        = dst_ردیف_اندازه;
    ثابت size_t src0_ردیف_اندازه   = nc_بایتها;
    ثابت size_t src1_ردیف_اندازه   = nc_بایتها;
    ثابت size_t src0_ردیف_stride = src0->nb[1];
    ثابت size_t src1_ردیف_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_ردیف_اندازه_aligned = hex_دور_به_بالا(src0_ردیف_اندازه, VLEN);
    ثابت size_t src1_ردیف_اندازه_aligned = hex_دور_به_بالا(src1_ردیف_اندازه, VLEN);
    ثابت size_t dst_ردیف_اندازه_aligned  = hex_دور_به_بالا(dst_ردیف_اندازه, VLEN);

    بنية htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_ردیف_اندازه_aligned, src1_ردیف_اندازه_aligned, dst_ردیف_اندازه_aligned, n_نخها,
                              octx->ctx->vtcm_اندازه);

    // Make sure the reserved vtcm size is sufficient
    إذا (L.vtcm_ردیف_در_هر_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_اندازه,
             (src0_ردیف_اندازه_aligned + src1_ردیف_اندازه_aligned + dst_ردیف_اندازه_aligned) * n_نخها);
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    إذا (src1) {
        FARF(بالا_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بایتها_در_هر_thread * n_نخها,
             L.src1_بایتها_در_هر_thread * n_نخها, L.dst_بایتها_در_هر_thread * n_نخها);
    } وإلا {
        FARF(بالا_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بایتها_در_هر_thread * n_نخها, L.src1_بایتها_در_هر_thread * n_نخها, L.dst_بایتها_در_هر_thread * n_نخها);
    }

    // Prepare context
    بنية htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (فراغ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_در_هر_thread = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div);

    actx.src0_ردیف_اندازه = src0_ردیف_اندازه;
    actx.src1_ردیف_اندازه = src1_ردیف_اندازه;
    actx.dst_ردیف_اندازه  = dst_ردیف_اندازه;

    actx.src0_ردیف_اندازه_aligned = src0_ردیف_اندازه_aligned;
    actx.src1_ردیف_اندازه_aligned = src1_ردیف_اندازه_aligned;
    actx.dst_ردیف_اندازه_aligned  = dst_ردیف_اندازه_aligned;

    actx.src0_ردیف_stride = src0_ردیف_stride;
    actx.src1_ردیف_stride = src1_ردیف_stride;

    uint8_t * ثابت پایه = (uint8_t *) octx->ctx->vtcm_پایه;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_dst);

    actx.vtcm_src0_اندازه_در_هر_thread = L.src0_بایتها_در_هر_thread;
    actx.vtcm_src1_اندازه_در_هر_thread = L.src1_بایتها_در_هر_thread;
    actx.vtcm_dst_اندازه_در_هر_thread  = L.dst_بایتها_در_هر_thread;

    actx.src0_spad_half_اندازه = L.src0_بایتها_در_هر_thread / 2;
    actx.src1_spad_half_اندازه = L.src1_بایتها_در_هر_thread / 2;
    actx.dst_spad_half_اندازه  = L.dst_بایتها_در_هر_thread / 2;

    actx.بلوک = actx.src0_spad_half_اندازه / actx.src0_ردیف_اندازه_aligned;
    actx.src0_nrows = nrows;
    actx.ردیف_شروع  = ردیف_شروع;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t داده_src0 = src0->داده;
    dma_addr_t داده_src1 = src1 ? src1->داده : 0;

    إذا (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU)) {
         ثابت int32_t swapped = octx->op_params[1];
         داده_src1 = داده_src0;
         actx.src1_ردیف_اندازه = actx.src0_ردیف_اندازه;

         size_t nc_درون_بایتها = actx.nc * SIZEOF_FP16;
         إذا (swapped) {
             داده_src0 += nc_درون_بایتها;
         } وإلا {
             داده_src1 += nc_درون_بایتها;
         }
    }

    actx.داده_src0 = داده_src0;
    actx.داده_src1 = داده_src1;
    actx.داده_dst  = dst->داده;

    work_صف_پیاپی(octx->ctx->work_صف, (کارگر_callback_t)glu_f16_در_هر_thread, &actx, n_نخها);
    أرجع HTP_وضعیت_OK;
}


صحيح op_activations(بنية htp_ops_context * octx) {
    بدل (octx->src[0]->type) {
        حالة HTP_TYPE_F32:
            أرجع execute_op_activations_f32(octx);

        حالة HTP_TYPE_F16:
            أرجع execute_op_activations_f16(octx);

        افتراضي:
            أرجع HTP_وضعیت_NO_SUPPORT;
    }
}
