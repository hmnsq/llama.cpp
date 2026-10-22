#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "hex-common.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"

#تعریف_کړه htp_act_preamble                                 \
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
                                     ثابت uint32_t عدد_لیکې_2,
                                     ثابت بنية htp_act_context * actx);

تعريفنوع فراغ (*glu_compute_f16_fn_t)(ثابت _Float16 * محدود src0,
                                     ثابت _Float16 * محدود src1,
                                     _Float16 * محدود dst,
                                     ثابت uint32_t عدد_لیکې_2,
                                     ثابت بنية htp_act_context * actx);

بنية htp_act_context {
    بنية htp_ops_context * octx;

    فراغ *                   compute;
    ثابت حرف *             op_str;

    // Precomputed values
    dma_addr_t               معلومات_src0;
    dma_addr_t               معلومات_src1;
    dma_addr_t               معلومات_dst;

    size_t                   src0_لیکه_اندازه;
    size_t                   src1_لیکه_اندازه;
    size_t                   dst_لیکه_اندازه;

    size_t                   src0_لیکه_stride;
    size_t                   src1_لیکه_stride;

    size_t                   src0_لیکه_اندازه_aligned;
    size_t                   src1_لیکه_اندازه_aligned;
    size_t                   dst_لیکه_اندازه_aligned;

    size_t                   src0_spad_half_اندازه;
    size_t                   src1_spad_half_اندازه;
    size_t                   dst_spad_half_اندازه;

    uint32_t                 بلاک;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_فی_thread;
    uint32_t                 لیکه_پیل;
    صحيح                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_اندازه_فی_thread;
    size_t                   vtcm_src1_اندازه_فی_thread;
    size_t                   vtcm_dst_اندازه_فی_thread;
};

بنية htp_act_vtcm_layout {
    size_t ټول_بایټونه;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_بایټونه_فی_thread;
    size_t src1_بایټونه_فی_thread;
    size_t dst_بایټونه_فی_thread;

    uint32_t vtcm_لیکه_فی_thread;
};

ساكن مضمن فراغ htp_act_vtcm_layout_build(بنية htp_act_vtcm_layout * L,
                                             size_t                       src0_لیکه_اندازه_aligned,
                                             size_t                       src1_لیکه_اندازه_aligned,
                                             size_t                       dst_لیکه_اندازه_aligned,
                                             uint32_t                     n_تارونه,
                                             size_t                       vtcm_اندازه) {
    ثابت size_t   spad_اندازه_فی_لیکه   = src0_لیکه_اندازه_aligned + src1_لیکه_اندازه_aligned + dst_لیکه_اندازه_aligned;
    ثابت uint32_t vtcm_لیکه_فی_thread = (uint32_t) (vtcm_اندازه / (n_تارونه * spad_اندازه_فی_لیکه));

    L->vtcm_لیکه_فی_thread = vtcm_لیکه_فی_thread;

    L->src0_بایټونه_فی_thread = src0_لیکه_اندازه_aligned * vtcm_لیکه_فی_thread;
    L->src1_بایټونه_فی_thread = src1_لیکه_اندازه_aligned * vtcm_لیکه_فی_thread;
    L->dst_بایټونه_فی_thread  = dst_لیکه_اندازه_aligned * vtcm_لیکه_فی_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_بایټونه_فی_thread * n_تارونه;
    L->off_dst  = L->off_src1 + L->src1_بایټونه_فی_thread * n_تارونه;

    L->ټول_بایټونه = L->off_dst + L->dst_بایټونه_فی_thread * n_تارونه;
}

#تعریف_کړه htp_glu_op_preamble                                            \
    ثابت size_t src0_لیکه_اندازه_aligned = actx->src0_لیکه_اندازه_aligned;  \
    ثابت size_t src1_لیکه_اندازه_aligned = actx->src1_لیکه_اندازه_aligned;  \
    ثابت size_t dst_لیکه_اندازه_aligned  = actx->dst_لیکه_اندازه_aligned;   \
    ثابت صحيح    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
ساكن فراغ swiglu_f32(ثابت عائم * محدود src0,
                                                 ثابت عائم * محدود src1,
                                                 عائم * محدود dst,
                                                 ثابت uint32_t عدد_لیکې_2,
                                                 ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساكن فراغ swiglu_oai_f32(ثابت عائم * محدود src0,
                                                     ثابت عائم * محدود src1,
                                                     عائم * محدود dst,
                                                     ثابت uint32_t عدد_لیکې_2,
                                                     ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت عائم alpha = ((ثابت عائم *) (actx->octx->op_params))[2];
    ثابت عائم حد = ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد, حد, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جمع_کړه_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
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
                                                       ثابت uint32_t                 عدد_لیکې_2,
                                                       ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت عائم حد = ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد, nc);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد, حد, nc);
        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ساكن ثابت عائم GELU_COEF_A     = 0.044715f;
ساكن ثابت عائم SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

ساكن مضمن HVX_ویکټور hvx_vec_fast_sigmoid_f32_2it(HVX_ویکټور v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_ویکټور دننه_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_ویکټور x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(دننه_int));
    HVX_ویکټور xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_ویکټور v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_ویکټور v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, دننه_int, 24);

    HVX_ویکټور v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_ویکټور v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_ویکټور two_sf = hvx_vec_splat_f32(2.0f);
    HVX_ویکټور i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_ویکټور r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_ویکټور res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    أرجع Q6_Vsf_equals_Vqf32(res);
}

ساكن مضمن HVX_ویکټور hvx_vec_fast_sigmoid_f32_guard_2it(HVX_ویکټور v,
                                                            HVX_ویکټور one,
                                                            HVX_ویکټور max_exp,
                                                            HVX_ویکټور min_exp) {
    ثابت HVX_ویکټور_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ثابت HVX_ویکټور_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_ویکټور بهر = hvx_vec_fast_sigmoid_f32_2it(v);
    بهر            = Q6_V_vmux_QVV(pred_max, بهر, one);
    أرجع Q6_V_vmux_QVV(pred_min, بهر, Q6_V_vzero());
}

ساكن مضمن فراغ hvx_geglu_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بینښه اوږد) dst  % 128 == 0);
    أكد((بینښه اوږد) src0 % 128 == 0);
    أكد((بینښه اوږد) src1 % 128 == 0);

    HVX_ویکټور * محدود vdst        = (HVX_ویکټور *) dst;
    ثابت HVX_ویکټور * محدود vsrc0 = (ثابت HVX_ویکټور *) src0;
    ثابت HVX_ویکټور * محدود vsrc1 = (ثابت HVX_ویکټور *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت عائم GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    ثابت HVX_ویکټور v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    ثابت HVX_ویکټور v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    ثابت HVX_ویکټور v_one               = hvx_vec_splat_f32(1.0f);
    ثابت HVX_ویکټور v_max_exp           = hvx_vec_splat_f32(87.0f);
    ثابت HVX_ویکټور v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ویکټور x = vsrc0[i];
        HVX_ویکټور g = vsrc1[i];

        HVX_ویکټور x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_ویکټور coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جمع_کړه_f32_f32(coef, v_sqrt_2_pi);
        HVX_ویکټور inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner = inner + inner
        HVX_ویکټور y2 = hvx_vec_جمع_کړه_f32_f32(inner, inner);

        // Fast sigmoid approximation (2 iterations)
        HVX_ویکټور sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_ویکټور gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    إذا (nloe) {
        HVX_ویکټور x = vsrc0[i];
        HVX_ویکټور g = vsrc1[i];

        HVX_ویکټور x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_ویکټور coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جمع_کړه_f32_f32(coef, v_sqrt_2_pi);
        HVX_ویکټور inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_ویکټور y2 = hvx_vec_جمع_کړه_f32_f32(inner, inner);

        HVX_ویکټور sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_ویکټور gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        HVX_ویکټور res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), res);
    }
}

ساكن مضمن فراغ hvx_geglu_چټک_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بینښه اوږد) dst  % 128 == 0);
    أكد((بینښه اوږد) src0 % 128 == 0);
    أكد((بینښه اوږد) src1 % 128 == 0);

    HVX_ویکټور * محدود vdst        = (HVX_ویکټور *) dst;
    ثابت HVX_ویکټور * محدود vsrc0 = (ثابت HVX_ویکټور *) src0;
    ثابت HVX_ویکټور * محدود vsrc1 = (ثابت HVX_ویکټور *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_ویکټور v_scale    = hvx_vec_splat_f32(1.702f);
    ثابت HVX_ویکټور v_one      = hvx_vec_splat_f32(1.0f);
    ثابت HVX_ویکټور v_max_exp  = hvx_vec_splat_f32(87.0f);
    ثابت HVX_ویکټور v_min_exp  = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ویکټور x = vsrc0[i];
        HVX_ویکټور g = vsrc1[i];
        HVX_ویکټور scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_ویکټور sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
    }

    إذا (nloe) {
        HVX_ویکټور x = vsrc0[i];
        HVX_ویکټور g = vsrc1[i];
        HVX_ویکټور scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_ویکټور sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        HVX_ویکټور پایله = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), پایله);
    }
}

ساكن مضمن فراغ hvx_geglu_erf_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بینښه اوږد) dst  % 128 == 0);
    أكد((بینښه اوږد) src0 % 128 == 0);
    أكد((بینښه اوږد) src1 % 128 == 0);

    HVX_ویکټور * محدود vdst        = (HVX_ویکټور *) dst;
    ثابت HVX_ویکټور * محدود vsrc0 = (ثابت HVX_ویکټور *) src0;
    ثابت HVX_ویکټور * محدود vsrc1 = (ثابت HVX_ویکټور *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ویکټور x = vsrc0[i];
        HVX_ویکټور g = vsrc1[i];
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
    }

    إذا (nloe) {
        HVX_ویکټور x = vsrc0[i];
        HVX_ویکټور g = vsrc1[i];
        HVX_ویکټور پایله = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), پایله);
    }
}

ساكن مضمن فراغ hvx_geglu_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بینښه اوږد) dst  % 128 == 0);
    أكد((بینښه اوږد) src0 % 128 == 0);
    أكد((بینښه اوږد) src1 % 128 == 0);

    HVX_ویکټور * محدود vdst        = (HVX_ویکټور *) dst;
    ثابت HVX_ویکټور * محدود vsrc0 = (ثابت HVX_ویکټور *) src0;
    ثابت HVX_ویکټور * محدود vsrc1 = (ثابت HVX_ویکټور *) src1;

    ثابت uint32_t epv  = 128 / حجم(_Float16);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_ویکټور v_coef_a    = hvx_vec_splat_f16((_Float16) GELU_COEF_A);
    ثابت HVX_ویکټور v_sqrt_2_pi = hvx_vec_splat_f16((_Float16) SQRT_2_OVER_PI);
    ثابت HVX_ویکټور v_one       = hvx_vec_splat_f16((_Float16) 1.0f);
    ثابت HVX_ویکټور v_half      = hvx_vec_splat_f16((_Float16) 0.5f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ویکټور x = vsrc0[i];
        HVX_ویکټور g = vsrc1[i];

        // gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))
        HVX_ویکټور x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_ویکټور coef  = hvx_vec_جمع_کړه_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_ویکټور inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_ویکټور t = hvx_vec_جمع_کړه_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_ویکټور gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        vdst[i] = hvx_vec_mul_f16_f16(gelu_x, g);
    }

    إذا (nloe) {
        HVX_ویکټور x = vsrc0[i];
        HVX_ویکټور g = vsrc1[i];

        HVX_ویکټور x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_ویکټور coef  = hvx_vec_جمع_کړه_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_ویکټور inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_ویکټور t = hvx_vec_جمع_کړه_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_ویکټور gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        HVX_ویکټور res    = hvx_vec_mul_f16_f16(gelu_x, g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(_Float16), res);
    }
}

// geglu(x, g) = gelu(x) * g
ساكن فراغ geglu_f32(ثابت عائم * محدود src0,
                                                ثابت عائم * محدود src1,
                                                عائم * محدود dst,
                                                ثابت uint32_t عدد_لیکې_2,
                                                ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_quick(x, g) = x * sigmoid(1.702 * x) * g
ساكن فراغ geglu_چټک_f32(ثابت عائم * محدود src0,
                            ثابت عائم * محدود src1,
                            عائم * محدود dst,
                            ثابت uint32_t عدد_لیکې_2,
                            ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

        hvx_geglu_چټک_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_erf(x, g) = gelu_erf(x) * g
ساكن فراغ geglu_erf_f32(ثابت عائم * محدود src0,
                          ثابت عائم * محدود src1,
                          عائم * محدود dst,
                          ثابت uint32_t عدد_لیکې_2,
                          ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

        hvx_geglu_erf_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// swiglu(x) = x1 * sigmoid(x0)
ساكن فراغ swiglu_f16(ثابت _Float16 * محدود src0,
                       ثابت _Float16 * محدود src1,
                       _Float16 * محدود dst,
                       ثابت uint32_t عدد_لیکې_2,
                       ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساكن فراغ swiglu_oai_f16(ثابت _Float16 * محدود src0,
                           ثابت _Float16 * محدود src1,
                           _Float16 * محدود dst,
                           ثابت uint32_t عدد_لیکې_2,
                           ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 alpha = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[2];
    ثابت _Float16 حد = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, حد, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -حد, حد, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جمع_کړه_scalar_f16((uint8_t *) src1_ptr, src1_ptr, (_Float16) 1.0f, nc);
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
                             ثابت uint32_t                 عدد_لیکې_2,
                             ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 حد = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

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
                      ثابت uint32_t عدد_لیکې_2,
                      ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_لیکې_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_لیکه_اندازه_aligned);

        hvx_geglu_f16_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#تعریف_کړه DEFINE_GLU_فی_THREAD(TYPE, SUFFIX)                                                                            \
    ساكن فراغ glu_##SUFFIX##_per_thread(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {                           \
        بنية htp_act_context * actx = (بنية htp_act_context *) معلومات;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        بنية htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : عدم;                          \
                                                                                                                       \
        size_t src0_لیکه_اندازه = actx->src0_لیکه_اندازه;                                                                    \
        size_t src1_لیکه_اندازه = actx->src1_لیکه_اندازه;                                                                    \
        size_t dst_لیکه_اندازه  = actx->dst_لیکه_اندازه;                                                                     \
                                                                                                                       \
        size_t src0_لیکه_stride = actx->src0_لیکه_stride;                                                                \
        size_t src1_لیکه_stride = actx->src1_لیکه_stride;                                                                \
                                                                                                                       \
        ثابت uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        ثابت uint32_t src0_nrows_فی_thread = actx->src0_nrows_فی_thread;                                            \
                                                                                                                       \
        ثابت uint32_t src0_پیل_لیکه = actx->لیکه_پیل + src0_nrows_فی_thread * ith;                                 \
        ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, actx->لیکه_پیل + src0_nrows);     \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {                                                                          \
            أرجع;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        ثابت dma_addr_t معلومات_src0 = actx->معلومات_src0;                                                                  \
        ثابت dma_addr_t معلومات_src1 = actx->معلومات_src1;                                                                  \
        ثابت dma_addr_t معلومات_dst  = actx->معلومات_dst;                                                                   \
                                                                                                                       \
        ثابت size_t src0_لیکه_اندازه_aligned = actx->src0_لیکه_اندازه_aligned;                                              \
        ثابت size_t src1_لیکه_اندازه_aligned = actx->src1_لیکه_اندازه_aligned;                                              \
        ثابت size_t dst_لیکه_اندازه_aligned  = actx->dst_لیکه_اندازه_aligned;                                               \
                                                                                                                       \
        uint8_t * محدود src0_spad_معلومات = actx->vtcm_src0 + (ith * actx->vtcm_src0_اندازه_فی_thread);                 \
        uint8_t * محدود src1_spad_معلومات = actx->vtcm_src1 + (ith * actx->vtcm_src1_اندازه_فی_thread);                 \
        uint8_t * محدود dst_spad_معلومات  = actx->vtcm_dst  + (ith * actx->vtcm_dst_اندازه_فی_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_اندازه = actx->src0_spad_half_اندازه;                                                        \
        size_t src1_spad_half_اندازه = actx->src1_spad_half_اندازه;                                                        \
        size_t dst_spad_half_اندازه  = actx->dst_spad_half_اندازه;                                                         \
                                                                                                                       \
        ثابت صحيح بلاک_2 = actx->بلاک;                                                                                 \
        إذا (بلاک_2 == 0) {                                                                                              \
            FARF(ERROR, "%s : VTCM reservation %zu is too small, needed %zu\n",                                        \
                 actx->op_str, actx->vtcm_src0_اندازه_فی_thread, src0_لیکه_اندازه_aligned);                                \
            أرجع;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_کتار * dma_q = actx->octx->ctx->dma[ith];                                                                 \
        glu_compute_##SUFFIX##_fn_t compute = (glu_compute_##SUFFIX##_fn_t) actx->compute;                             \
                                                                                                                       \
        لكل (uint32_t ir = src0_پیل_لیکه, spad_idx = 0; ir < src0_پای_ټکی_لیکه && spad_idx < 2; ir += بلاک_2, spad_idx++) { \
            ثابت uint32_t بلاک_اندازه_3 = MIN(بلاک_2, src0_پای_ټکی_لیکه - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_کتار_ټېل(dma_q,                                                                                      \
                           dma_make_معلومات(معلومات_dst, dst_spad_معلومات + (spad_idx * dst_spad_half_اندازه)),                   \
                           dst_لیکه_اندازه, dst_لیکه_اندازه_aligned, dst_لیکه_اندازه, 0);                                       \
                                                                                                                       \
            dma_کتار_ټېل(dma_q,                                                                                      \
                dma_make_معلومات(src0_spad_معلومات + (spad_idx * src0_spad_half_اندازه), معلومات_src0 + (ir * src0_لیکه_stride)),  \
                src0_لیکه_اندازه_aligned, src0_لیکه_stride, src0_لیکه_اندازه, بلاک_اندازه_3);                                    \
                                                                                                                       \
            dma_کتار_ټېل(dma_q,                                                                                      \
                dma_make_معلومات(src1_spad_معلومات + (spad_idx * src1_spad_half_اندازه), معلومات_src1 + (ir * src1_لیکه_stride)),  \
                src1_لیکه_اندازه_aligned, src1_لیکه_stride, src1_لیکه_اندازه, بلاک_اندازه_3);                                    \
        }                                                                                                              \
                                                                                                                       \
        لكل (uint32_t ir = src0_پیل_لیکه; ir < src0_پای_ټکی_لیکه; ir += بلاک_2) {                                           \
            ثابت uint32_t بلاک_اندازه_3 = MIN(بلاک_2, src0_پای_ټکی_لیکه - ir);                                                 \
                                                                                                                       \
            TYPE * dst_spad  = (TYPE *) dma_کتار_راوباسه(dma_q).src;                                                      \
            TYPE * src0_spad = (TYPE *) dma_کتار_راوباسه(dma_q).dst;                                                      \
            TYPE * src1_spad = (TYPE *) dma_کتار_راوباسه(dma_q).dst;                                                      \
                                                                                                                       \
            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            compute(src0_spad, src1_spad, dst_spad, بلاک_اندازه_3, actx);                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_کتار_ټېل(dma_q, dma_make_معلومات(معلومات_dst + (ir * dst_لیکه_اندازه), dst_spad),                             \
                           dst_لیکه_اندازه, dst_لیکه_اندازه_aligned, dst_لیکه_اندازه, بلاک_اندازه_3);                              \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            ثابت uint32_t pref_بلاک = (ir + بلاک_2 * 2);                                                              \
            إذا (pref_بلاک < src0_پای_ټکی_لیکه) {                                                                           \
                ثابت uint32_t pref_بلاک_اندازه = MIN(بلاک_2, src0_پای_ټکی_لیکه - pref_بلاک);                                \
                dma_کتار_ټېل(dma_q, dma_make_معلومات(src0_spad, معلومات_src0 + (pref_بلاک * src0_لیکه_stride)),            \
                               src0_لیکه_اندازه_aligned, src0_لیکه_stride, src0_لیکه_اندازه, pref_بلاک_اندازه);                \
                dma_کتار_ټېل(dma_q, dma_make_معلومات(src1_spad, معلومات_src1 + (pref_بلاک * src1_لیکه_stride)),            \
                               src1_لیکه_اندازه_aligned, src1_لیکه_stride, src1_لیکه_اندازه, pref_بلاک_اندازه);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_کتار_flush(dma_q);                                                                                        \
    }

DEFINE_GLU_فی_THREAD(عائم, f32)
DEFINE_GLU_فی_THREAD(_Float16, f16)

ساكن صحيح execute_op_activations_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        أرجع HTP_حالت_NO_SUPPORT;
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

        حالة HTP_OP_GLU_GEGLU_چټک:
            compute_fn = geglu_چټک_f32;
            op_type    = "geglu-quick-f32";
            اكسر;

        حالة HTP_OP_GLU_GEGLU_ERF:
            compute_fn = geglu_erf_f32;
            op_type    = "geglu-erf-f32";
            اكسر;

        افتراضي:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_لیکه_اندازه = dst->ne[0] * SIZEOF_FP32;

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t لیکې_فی_chunk = 0;
        htp_tensor_mdev_لیکې_فی_chunk(dst, حجم(عائم), (uint32_t) dst_لیکه_اندازه, &لیکې_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(src0_nrows, لیکې_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بایټونه        = dst_لیکه_اندازه;
    ثابت size_t src0_لیکه_اندازه   = nc_بایټونه;
    ثابت size_t src1_لیکه_اندازه   = nc_بایټونه;
    ثابت size_t src0_لیکه_stride = src0->nb[1];
    ثابت size_t src1_لیکه_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_لیکه_اندازه_aligned = hex_پړاو_پورته(src0_لیکه_اندازه, VLEN);
    ثابت size_t src1_لیکه_اندازه_aligned = hex_پړاو_پورته(src1_لیکه_اندازه, VLEN);
    ثابت size_t dst_لیکه_اندازه_aligned  = hex_پړاو_پورته(dst_لیکه_اندازه, VLEN);

    بنية htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_لیکه_اندازه_aligned, src1_لیکه_اندازه_aligned, dst_لیکه_اندازه_aligned, n_تارونه,
                              octx->ctx->vtcm_اندازه);

    // Make sure the reserved vtcm size is sufficient
    إذا (L.vtcm_لیکه_فی_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_اندازه,
             (src0_لیکه_اندازه_aligned + src1_لیکه_اندازه_aligned + dst_لیکه_اندازه_aligned) * n_تارونه);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    إذا (src1) {
        FARF(لوړ_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بایټونه_فی_thread * n_تارونه,
             L.src1_بایټونه_فی_thread * n_تارونه, L.dst_بایټونه_فی_thread * n_تارونه);
    } وإلا {
        FARF(لوړ_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بایټونه_فی_thread * n_تارونه, L.src1_بایټونه_فی_thread * n_تارونه, L.dst_بایټونه_فی_thread * n_تارونه);
    }

    // Prepare context
    بنية htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (فراغ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_فی_thread = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div);

    actx.src0_لیکه_اندازه = src0_لیکه_اندازه;
    actx.src1_لیکه_اندازه = src1_لیکه_اندازه;
    actx.dst_لیکه_اندازه  = dst_لیکه_اندازه;

    actx.src0_لیکه_اندازه_aligned = src0_لیکه_اندازه_aligned;
    actx.src1_لیکه_اندازه_aligned = src1_لیکه_اندازه_aligned;
    actx.dst_لیکه_اندازه_aligned  = dst_لیکه_اندازه_aligned;

    actx.src0_لیکه_stride = src0_لیکه_stride;
    actx.src1_لیکه_stride = src1_لیکه_stride;

    uint8_t * ثابت بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_dst);

    actx.vtcm_src0_اندازه_فی_thread = L.src0_بایټونه_فی_thread;
    actx.vtcm_src1_اندازه_فی_thread = L.src1_بایټونه_فی_thread;
    actx.vtcm_dst_اندازه_فی_thread  = L.dst_بایټونه_فی_thread;

    actx.src0_spad_half_اندازه = L.src0_بایټونه_فی_thread / 2;
    actx.src1_spad_half_اندازه = L.src1_بایټونه_فی_thread / 2;
    actx.dst_spad_half_اندازه  = L.dst_بایټونه_فی_thread / 2;

    actx.بلاک = actx.src0_spad_half_اندازه / actx.src0_لیکه_اندازه_aligned;
    actx.src0_nrows = nrows;
    actx.لیکه_پیل  = لیکه_پیل;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t معلومات_src0 = src0->معلومات;
    dma_addr_t معلومات_src1 = src1 ? src1->معلومات : 0;

    إذا (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU ||
                  octx->op == HTP_OP_GLU_GEGLU_چټک ||
                  octx->op == HTP_OP_GLU_GEGLU_ERF)) {
         ثابت int32_t swapped = octx->op_params[1];
         معلومات_src1 = معلومات_src0;
         actx.src1_لیکه_اندازه = actx.src0_لیکه_اندازه;

         size_t nc_دننه_بایټونه = actx.nc * SIZEOF_FP32;
         إذا (swapped) {
             معلومات_src0 += nc_دننه_بایټونه;
         } وإلا {
             معلومات_src1 += nc_دننه_بایټونه;
         }
    }

    actx.معلومات_src0 = معلومات_src0;
    actx.معلومات_src1 = معلومات_src1;
    actx.معلومات_dst  = dst->معلومات;

    work_کتار_پرلپسې(octx->ctx->work_کتار, (کارګر_callback_t)glu_f32_فی_thread, &actx, n_تارونه);
    أرجع HTP_حالت_OK;
}

ساكن صحيح execute_op_activations_f16(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا ((dst->ne[0] * SIZEOF_FP16) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        أرجع HTP_حالت_NO_SUPPORT;
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
            أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_لیکه_اندازه = dst->ne[0] * SIZEOF_FP16;

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t لیکې_فی_chunk = 0;
        htp_tensor_mdev_لیکې_فی_chunk(dst, حجم(_Float16), (uint32_t) dst_لیکه_اندازه, &لیکې_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(src0_nrows, لیکې_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بایټونه        = dst_لیکه_اندازه;
    ثابت size_t src0_لیکه_اندازه   = nc_بایټونه;
    ثابت size_t src1_لیکه_اندازه   = nc_بایټونه;
    ثابت size_t src0_لیکه_stride = src0->nb[1];
    ثابت size_t src1_لیکه_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_لیکه_اندازه_aligned = hex_پړاو_پورته(src0_لیکه_اندازه, VLEN);
    ثابت size_t src1_لیکه_اندازه_aligned = hex_پړاو_پورته(src1_لیکه_اندازه, VLEN);
    ثابت size_t dst_لیکه_اندازه_aligned  = hex_پړاو_پورته(dst_لیکه_اندازه, VLEN);

    بنية htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_لیکه_اندازه_aligned, src1_لیکه_اندازه_aligned, dst_لیکه_اندازه_aligned, n_تارونه,
                              octx->ctx->vtcm_اندازه);

    // Make sure the reserved vtcm size is sufficient
    إذا (L.vtcm_لیکه_فی_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_اندازه,
             (src0_لیکه_اندازه_aligned + src1_لیکه_اندازه_aligned + dst_لیکه_اندازه_aligned) * n_تارونه);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    إذا (src1) {
        FARF(لوړ_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بایټونه_فی_thread * n_تارونه,
             L.src1_بایټونه_فی_thread * n_تارونه, L.dst_بایټونه_فی_thread * n_تارونه);
    } وإلا {
        FARF(لوړ_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بایټونه_فی_thread * n_تارونه, L.src1_بایټونه_فی_thread * n_تارونه, L.dst_بایټونه_فی_thread * n_تارونه);
    }

    // Prepare context
    بنية htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (فراغ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_فی_thread = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div);

    actx.src0_لیکه_اندازه = src0_لیکه_اندازه;
    actx.src1_لیکه_اندازه = src1_لیکه_اندازه;
    actx.dst_لیکه_اندازه  = dst_لیکه_اندازه;

    actx.src0_لیکه_اندازه_aligned = src0_لیکه_اندازه_aligned;
    actx.src1_لیکه_اندازه_aligned = src1_لیکه_اندازه_aligned;
    actx.dst_لیکه_اندازه_aligned  = dst_لیکه_اندازه_aligned;

    actx.src0_لیکه_stride = src0_لیکه_stride;
    actx.src1_لیکه_stride = src1_لیکه_stride;

    uint8_t * ثابت بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, بنسټ, L.off_dst);

    actx.vtcm_src0_اندازه_فی_thread = L.src0_بایټونه_فی_thread;
    actx.vtcm_src1_اندازه_فی_thread = L.src1_بایټونه_فی_thread;
    actx.vtcm_dst_اندازه_فی_thread  = L.dst_بایټونه_فی_thread;

    actx.src0_spad_half_اندازه = L.src0_بایټونه_فی_thread / 2;
    actx.src1_spad_half_اندازه = L.src1_بایټونه_فی_thread / 2;
    actx.dst_spad_half_اندازه  = L.dst_بایټونه_فی_thread / 2;

    actx.بلاک = actx.src0_spad_half_اندازه / actx.src0_لیکه_اندازه_aligned;
    actx.src0_nrows = nrows;
    actx.لیکه_پیل  = لیکه_پیل;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t معلومات_src0 = src0->معلومات;
    dma_addr_t معلومات_src1 = src1 ? src1->معلومات : 0;

    إذا (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU)) {
         ثابت int32_t swapped = octx->op_params[1];
         معلومات_src1 = معلومات_src0;
         actx.src1_لیکه_اندازه = actx.src0_لیکه_اندازه;

         size_t nc_دننه_بایټونه = actx.nc * SIZEOF_FP16;
         إذا (swapped) {
             معلومات_src0 += nc_دننه_بایټونه;
         } وإلا {
             معلومات_src1 += nc_دننه_بایټونه;
         }
    }

    actx.معلومات_src0 = معلومات_src0;
    actx.معلومات_src1 = معلومات_src1;
    actx.معلومات_dst  = dst->معلومات;

    work_کتار_پرلپسې(octx->ctx->work_کتار, (کارګر_callback_t)glu_f16_فی_thread, &actx, n_تارونه);
    أرجع HTP_حالت_OK;
}


صحيح op_activations(بنية htp_ops_context * octx) {
    بدل (octx->src[0]->type) {
        حالة HTP_TYPE_F32:
            أرجع execute_op_activations_f32(octx);

        حالة HTP_TYPE_F16:
            أرجع execute_op_activations_f16(octx);

        افتراضي:
            أرجع HTP_حالت_NO_SUPPORT;
    }
}
