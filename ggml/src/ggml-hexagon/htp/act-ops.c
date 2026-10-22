#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "hex-common.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"

#وصف_ڪريو htp_act_preamble                                 \
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
                                     ثابت uint32_t انگ_صفون_2,
                                     ثابت بنية htp_act_context * actx);

تعريفنوع فراغ (*glu_compute_f16_fn_t)(ثابت _Float16 * محدود src0,
                                     ثابت _Float16 * محدود src1,
                                     _Float16 * محدود dst,
                                     ثابت uint32_t انگ_صفون_2,
                                     ثابت بنية htp_act_context * actx);

بنية htp_act_context {
    بنية htp_ops_context * octx;

    فراغ *                   compute;
    ثابت حرف *             op_str;

    // Precomputed values
    dma_addr_t               معلومات_src0;
    dma_addr_t               معلومات_src1;
    dma_addr_t               معلومات_dst;

    size_t                   src0_صف_سائيز;
    size_t                   src1_صف_سائيز;
    size_t                   dst_صف_سائيز;

    size_t                   src0_صف_stride;
    size_t                   src1_صف_stride;

    size_t                   src0_صف_سائيز_aligned;
    size_t                   src1_صف_سائيز_aligned;
    size_t                   dst_صف_سائيز_aligned;

    size_t                   src0_spad_half_سائيز;
    size_t                   src1_spad_half_سائيز;
    size_t                   dst_spad_half_سائيز;

    uint32_t                 بلاڪ;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_في_thread;
    uint32_t                 صف_شروعات;
    صحيح                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_سائيز_في_thread;
    size_t                   vtcm_src1_سائيز_في_thread;
    size_t                   vtcm_dst_سائيز_في_thread;
};

بنية htp_act_vtcm_layout {
    size_t ڪل_بائيٽون;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_بائيٽون_في_thread;
    size_t src1_بائيٽون_في_thread;
    size_t dst_بائيٽون_في_thread;

    uint32_t vtcm_صف_في_thread;
};

ساكن مضمن فراغ htp_act_vtcm_layout_build(بنية htp_act_vtcm_layout * L,
                                             size_t                       src0_صف_سائيز_aligned,
                                             size_t                       src1_صف_سائيز_aligned,
                                             size_t                       dst_صف_سائيز_aligned,
                                             uint32_t                     n_ڌاڳا,
                                             size_t                       vtcm_سائيز) {
    ثابت size_t   spad_سائيز_في_صف   = src0_صف_سائيز_aligned + src1_صف_سائيز_aligned + dst_صف_سائيز_aligned;
    ثابت uint32_t vtcm_صف_في_thread = (uint32_t) (vtcm_سائيز / (n_ڌاڳا * spad_سائيز_في_صف));

    L->vtcm_صف_في_thread = vtcm_صف_في_thread;

    L->src0_بائيٽون_في_thread = src0_صف_سائيز_aligned * vtcm_صف_في_thread;
    L->src1_بائيٽون_في_thread = src1_صف_سائيز_aligned * vtcm_صف_في_thread;
    L->dst_بائيٽون_في_thread  = dst_صف_سائيز_aligned * vtcm_صف_في_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_بائيٽون_في_thread * n_ڌاڳا;
    L->off_dst  = L->off_src1 + L->src1_بائيٽون_في_thread * n_ڌاڳا;

    L->ڪل_بائيٽون = L->off_dst + L->dst_بائيٽون_في_thread * n_ڌاڳا;
}

#وصف_ڪريو htp_glu_op_preamble                                            \
    ثابت size_t src0_صف_سائيز_aligned = actx->src0_صف_سائيز_aligned;  \
    ثابت size_t src1_صف_سائيز_aligned = actx->src1_صف_سائيز_aligned;  \
    ثابت size_t dst_صف_سائيز_aligned  = actx->dst_صف_سائيز_aligned;   \
    ثابت صحيح    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
ساكن فراغ swiglu_f32(ثابت عائم * محدود src0,
                                                 ثابت عائم * محدود src1,
                                                 عائم * محدود dst,
                                                 ثابت uint32_t انگ_صفون_2,
                                                 ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساكن فراغ swiglu_oai_f32(ثابت عائم * محدود src0,
                                                     ثابت عائم * محدود src1,
                                                     عائم * محدود dst,
                                                     ثابت uint32_t انگ_صفون_2,
                                                     ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت عائم alpha = ((ثابت عائم *) (actx->octx->op_params))[2];
    ثابت عائم حد_بندي = ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد_بندي, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد_بندي, حد_بندي, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جوڙيو_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
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
                                                       ثابت uint32_t                 انگ_صفون_2,
                                                       ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت عائم حد_بندي = ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد_بندي, nc);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد_بندي, حد_بندي, nc);
        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ساكن ثابت عائم GELU_COEF_A     = 0.044715f;
ساكن ثابت عائم SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

ساكن مضمن HVX_ويڪٽر hvx_vec_fast_sigmoid_f32_2it(HVX_ويڪٽر v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_ويڪٽر اندريون_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_ويڪٽر x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(اندريون_int));
    HVX_ويڪٽر xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_ويڪٽر v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_ويڪٽر v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_ويڪٽر v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, اندريون_int, 24);

    HVX_ويڪٽر v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_ويڪٽر v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_ويڪٽر two_sf = hvx_vec_splat_f32(2.0f);
    HVX_ويڪٽر i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_ويڪٽر r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_ويڪٽر res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    أرجع Q6_Vsf_equals_Vqf32(res);
}

ساكن مضمن HVX_ويڪٽر hvx_vec_fast_sigmoid_f32_guard_2it(HVX_ويڪٽر v,
                                                            HVX_ويڪٽر one,
                                                            HVX_ويڪٽر max_exp,
                                                            HVX_ويڪٽر min_exp) {
    ثابت HVX_ويڪٽر_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ثابت HVX_ويڪٽر_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_ويڪٽر ٻاهريون = hvx_vec_fast_sigmoid_f32_2it(v);
    ٻاهريون            = Q6_V_vmux_QVV(pred_max, ٻاهريون, one);
    أرجع Q6_V_vmux_QVV(pred_min, ٻاهريون, Q6_V_vzero());
}

ساكن مضمن فراغ hvx_geglu_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بنانشان ڊگهو) dst  % 128 == 0);
    أكد((بنانشان ڊگهو) src0 % 128 == 0);
    أكد((بنانشان ڊگهو) src1 % 128 == 0);

    HVX_ويڪٽر * محدود vdst        = (HVX_ويڪٽر *) dst;
    ثابت HVX_ويڪٽر * محدود vsrc0 = (ثابت HVX_ويڪٽر *) src0;
    ثابت HVX_ويڪٽر * محدود vsrc1 = (ثابت HVX_ويڪٽر *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت عائم GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    ثابت HVX_ويڪٽر v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    ثابت HVX_ويڪٽر v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    ثابت HVX_ويڪٽر v_one               = hvx_vec_splat_f32(1.0f);
    ثابت HVX_ويڪٽر v_max_exp           = hvx_vec_splat_f32(87.0f);
    ثابت HVX_ويڪٽر v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ويڪٽر x = vsrc0[i];
        HVX_ويڪٽر g = vsrc1[i];

        HVX_ويڪٽر x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_ويڪٽر coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جوڙيو_f32_f32(coef, v_sqrt_2_pi);
        HVX_ويڪٽر inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner = inner + inner
        HVX_ويڪٽر y2 = hvx_vec_جوڙيو_f32_f32(inner, inner);

        // Fast sigmoid approximation (2 iterations)
        HVX_ويڪٽر sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_ويڪٽر gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    إذا (nloe) {
        HVX_ويڪٽر x = vsrc0[i];
        HVX_ويڪٽر g = vsrc1[i];

        HVX_ويڪٽر x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_ويڪٽر coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جوڙيو_f32_f32(coef, v_sqrt_2_pi);
        HVX_ويڪٽر inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_ويڪٽر y2 = hvx_vec_جوڙيو_f32_f32(inner, inner);

        HVX_ويڪٽر sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_ويڪٽر gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        HVX_ويڪٽر res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), res);
    }
}

ساكن مضمن فراغ hvx_geglu_تيز_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بنانشان ڊگهو) dst  % 128 == 0);
    أكد((بنانشان ڊگهو) src0 % 128 == 0);
    أكد((بنانشان ڊگهو) src1 % 128 == 0);

    HVX_ويڪٽر * محدود vdst        = (HVX_ويڪٽر *) dst;
    ثابت HVX_ويڪٽر * محدود vsrc0 = (ثابت HVX_ويڪٽر *) src0;
    ثابت HVX_ويڪٽر * محدود vsrc1 = (ثابت HVX_ويڪٽر *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_ويڪٽر v_scale    = hvx_vec_splat_f32(1.702f);
    ثابت HVX_ويڪٽر v_one      = hvx_vec_splat_f32(1.0f);
    ثابت HVX_ويڪٽر v_max_exp  = hvx_vec_splat_f32(87.0f);
    ثابت HVX_ويڪٽر v_min_exp  = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ويڪٽر x = vsrc0[i];
        HVX_ويڪٽر g = vsrc1[i];
        HVX_ويڪٽر scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_ويڪٽر sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
    }

    إذا (nloe) {
        HVX_ويڪٽر x = vsrc0[i];
        HVX_ويڪٽر g = vsrc1[i];
        HVX_ويڪٽر scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_ويڪٽر sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        HVX_ويڪٽر نتيجو = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), نتيجو);
    }
}

ساكن مضمن فراغ hvx_geglu_erf_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بنانشان ڊگهو) dst  % 128 == 0);
    أكد((بنانشان ڊگهو) src0 % 128 == 0);
    أكد((بنانشان ڊگهو) src1 % 128 == 0);

    HVX_ويڪٽر * محدود vdst        = (HVX_ويڪٽر *) dst;
    ثابت HVX_ويڪٽر * محدود vsrc0 = (ثابت HVX_ويڪٽر *) src0;
    ثابت HVX_ويڪٽر * محدود vsrc1 = (ثابت HVX_ويڪٽر *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ويڪٽر x = vsrc0[i];
        HVX_ويڪٽر g = vsrc1[i];
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
    }

    إذا (nloe) {
        HVX_ويڪٽر x = vsrc0[i];
        HVX_ويڪٽر g = vsrc1[i];
        HVX_ويڪٽر نتيجو = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), نتيجو);
    }
}

ساكن مضمن فراغ hvx_geglu_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بنانشان ڊگهو) dst  % 128 == 0);
    أكد((بنانشان ڊگهو) src0 % 128 == 0);
    أكد((بنانشان ڊگهو) src1 % 128 == 0);

    HVX_ويڪٽر * محدود vdst        = (HVX_ويڪٽر *) dst;
    ثابت HVX_ويڪٽر * محدود vsrc0 = (ثابت HVX_ويڪٽر *) src0;
    ثابت HVX_ويڪٽر * محدود vsrc1 = (ثابت HVX_ويڪٽر *) src1;

    ثابت uint32_t epv  = 128 / حجم(_Float16);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_ويڪٽر v_coef_a    = hvx_vec_splat_f16((_Float16) GELU_COEF_A);
    ثابت HVX_ويڪٽر v_sqrt_2_pi = hvx_vec_splat_f16((_Float16) SQRT_2_OVER_PI);
    ثابت HVX_ويڪٽر v_one       = hvx_vec_splat_f16((_Float16) 1.0f);
    ثابت HVX_ويڪٽر v_half      = hvx_vec_splat_f16((_Float16) 0.5f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_ويڪٽر x = vsrc0[i];
        HVX_ويڪٽر g = vsrc1[i];

        // gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))
        HVX_ويڪٽر x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_ويڪٽر coef  = hvx_vec_جوڙيو_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_ويڪٽر inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_ويڪٽر t = hvx_vec_جوڙيو_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_ويڪٽر gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        vdst[i] = hvx_vec_mul_f16_f16(gelu_x, g);
    }

    إذا (nloe) {
        HVX_ويڪٽر x = vsrc0[i];
        HVX_ويڪٽر g = vsrc1[i];

        HVX_ويڪٽر x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_ويڪٽر coef  = hvx_vec_جوڙيو_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_ويڪٽر inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_ويڪٽر t = hvx_vec_جوڙيو_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_ويڪٽر gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        HVX_ويڪٽر res    = hvx_vec_mul_f16_f16(gelu_x, g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(_Float16), res);
    }
}

// geglu(x, g) = gelu(x) * g
ساكن فراغ geglu_f32(ثابت عائم * محدود src0,
                                                ثابت عائم * محدود src1,
                                                عائم * محدود dst,
                                                ثابت uint32_t انگ_صفون_2,
                                                ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_quick(x, g) = x * sigmoid(1.702 * x) * g
ساكن فراغ geglu_تيز_f32(ثابت عائم * محدود src0,
                            ثابت عائم * محدود src1,
                            عائم * محدود dst,
                            ثابت uint32_t انگ_صفون_2,
                            ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        hvx_geglu_تيز_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_erf(x, g) = gelu_erf(x) * g
ساكن فراغ geglu_erf_f32(ثابت عائم * محدود src0,
                          ثابت عائم * محدود src1,
                          عائم * محدود dst,
                          ثابت uint32_t انگ_صفون_2,
                          ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        hvx_geglu_erf_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// swiglu(x) = x1 * sigmoid(x0)
ساكن فراغ swiglu_f16(ثابت _Float16 * محدود src0,
                       ثابت _Float16 * محدود src1,
                       _Float16 * محدود dst,
                       ثابت uint32_t انگ_صفون_2,
                       ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساكن فراغ swiglu_oai_f16(ثابت _Float16 * محدود src0,
                           ثابت _Float16 * محدود src1,
                           _Float16 * محدود dst,
                           ثابت uint32_t انگ_صفون_2,
                           ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 alpha = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[2];
    ثابت _Float16 حد_بندي = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, حد_بندي, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -حد_بندي, حد_بندي, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جوڙيو_scalar_f16((uint8_t *) src1_ptr, src1_ptr, (_Float16) 1.0f, nc);
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
                             ثابت uint32_t                 انگ_صفون_2,
                             ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 حد_بندي = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, حد_بندي, nc);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -حد_بندي, حد_بندي, nc);
        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// geglu(x, g) = gelu(x) * g, gelu via tanh approximation
ساكن فراغ geglu_f16(ثابت _Float16 * محدود src0,
                      ثابت _Float16 * محدود src1,
                      _Float16 * محدود dst,
                      ثابت uint32_t انگ_صفون_2,
                      ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < انگ_صفون_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_سائيز_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_سائيز_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_سائيز_aligned);

        hvx_geglu_f16_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#وصف_ڪريو DEFINE_GLU_في_THREAD(TYPE, SUFFIX)                                                                            \
    ساكن فراغ glu_##SUFFIX##_per_thread(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {                           \
        بنية htp_act_context * actx = (بنية htp_act_context *) معلومات;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        بنية htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : عدم;                          \
                                                                                                                       \
        size_t src0_صف_سائيز = actx->src0_صف_سائيز;                                                                    \
        size_t src1_صف_سائيز = actx->src1_صف_سائيز;                                                                    \
        size_t dst_صف_سائيز  = actx->dst_صف_سائيز;                                                                     \
                                                                                                                       \
        size_t src0_صف_stride = actx->src0_صف_stride;                                                                \
        size_t src1_صف_stride = actx->src1_صف_stride;                                                                \
                                                                                                                       \
        ثابت uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        ثابت uint32_t src0_nrows_في_thread = actx->src0_nrows_في_thread;                                            \
                                                                                                                       \
        ثابت uint32_t src0_شروعات_صف = actx->صف_شروعات + src0_nrows_في_thread * ith;                                 \
        ثابت uint32_t src0_پڇاڙي_صف   = MIN(src0_شروعات_صف + src0_nrows_في_thread, actx->صف_شروعات + src0_nrows);     \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        إذا (src0_شروعات_صف >= src0_پڇاڙي_صف) {                                                                          \
            أرجع;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        ثابت dma_addr_t معلومات_src0 = actx->معلومات_src0;                                                                  \
        ثابت dma_addr_t معلومات_src1 = actx->معلومات_src1;                                                                  \
        ثابت dma_addr_t معلومات_dst  = actx->معلومات_dst;                                                                   \
                                                                                                                       \
        ثابت size_t src0_صف_سائيز_aligned = actx->src0_صف_سائيز_aligned;                                              \
        ثابت size_t src1_صف_سائيز_aligned = actx->src1_صف_سائيز_aligned;                                              \
        ثابت size_t dst_صف_سائيز_aligned  = actx->dst_صف_سائيز_aligned;                                               \
                                                                                                                       \
        uint8_t * محدود src0_spad_معلومات = actx->vtcm_src0 + (ith * actx->vtcm_src0_سائيز_في_thread);                 \
        uint8_t * محدود src1_spad_معلومات = actx->vtcm_src1 + (ith * actx->vtcm_src1_سائيز_في_thread);                 \
        uint8_t * محدود dst_spad_معلومات  = actx->vtcm_dst  + (ith * actx->vtcm_dst_سائيز_في_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_سائيز = actx->src0_spad_half_سائيز;                                                        \
        size_t src1_spad_half_سائيز = actx->src1_spad_half_سائيز;                                                        \
        size_t dst_spad_half_سائيز  = actx->dst_spad_half_سائيز;                                                         \
                                                                                                                       \
        ثابت صحيح بلاڪ_2 = actx->بلاڪ;                                                                                 \
        إذا (بلاڪ_2 == 0) {                                                                                              \
            FARF(ERROR, "%s : VTCM reservation %zu is too small, needed %zu\n",                                        \
                 actx->op_str, actx->vtcm_src0_سائيز_في_thread, src0_صف_سائيز_aligned);                                \
            أرجع;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_قطار * dma_q = actx->octx->ctx->dma[ith];                                                                 \
        glu_compute_##SUFFIX##_fn_t compute = (glu_compute_##SUFFIX##_fn_t) actx->compute;                             \
                                                                                                                       \
        لكل (uint32_t ir = src0_شروعات_صف, spad_idx = 0; ir < src0_پڇاڙي_صف && spad_idx < 2; ir += بلاڪ_2, spad_idx++) { \
            ثابت uint32_t بلاڪ_سائيز_3 = MIN(بلاڪ_2, src0_پڇاڙي_صف - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_قطار_ڌڪو(dma_q,                                                                                      \
                           dma_make_معلومات(معلومات_dst, dst_spad_معلومات + (spad_idx * dst_spad_half_سائيز)),                   \
                           dst_صف_سائيز, dst_صف_سائيز_aligned, dst_صف_سائيز, 0);                                       \
                                                                                                                       \
            dma_قطار_ڌڪو(dma_q,                                                                                      \
                dma_make_معلومات(src0_spad_معلومات + (spad_idx * src0_spad_half_سائيز), معلومات_src0 + (ir * src0_صف_stride)),  \
                src0_صف_سائيز_aligned, src0_صف_stride, src0_صف_سائيز, بلاڪ_سائيز_3);                                    \
                                                                                                                       \
            dma_قطار_ڌڪو(dma_q,                                                                                      \
                dma_make_معلومات(src1_spad_معلومات + (spad_idx * src1_spad_half_سائيز), معلومات_src1 + (ir * src1_صف_stride)),  \
                src1_صف_سائيز_aligned, src1_صف_stride, src1_صف_سائيز, بلاڪ_سائيز_3);                                    \
        }                                                                                                              \
                                                                                                                       \
        لكل (uint32_t ir = src0_شروعات_صف; ir < src0_پڇاڙي_صف; ir += بلاڪ_2) {                                           \
            ثابت uint32_t بلاڪ_سائيز_3 = MIN(بلاڪ_2, src0_پڇاڙي_صف - ir);                                                 \
                                                                                                                       \
            TYPE * dst_spad  = (TYPE *) dma_قطار_ڪڍو(dma_q).src;                                                      \
            TYPE * src0_spad = (TYPE *) dma_قطار_ڪڍو(dma_q).dst;                                                      \
            TYPE * src1_spad = (TYPE *) dma_قطار_ڪڍو(dma_q).dst;                                                      \
                                                                                                                       \
            htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            compute(src0_spad, src1_spad, dst_spad, بلاڪ_سائيز_3, actx);                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(معلومات_dst + (ir * dst_صف_سائيز), dst_spad),                             \
                           dst_صف_سائيز, dst_صف_سائيز_aligned, dst_صف_سائيز, بلاڪ_سائيز_3);                              \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            ثابت uint32_t pref_بلاڪ = (ir + بلاڪ_2 * 2);                                                              \
            إذا (pref_بلاڪ < src0_پڇاڙي_صف) {                                                                           \
                ثابت uint32_t pref_بلاڪ_سائيز = MIN(بلاڪ_2, src0_پڇاڙي_صف - pref_بلاڪ);                                \
                dma_قطار_ڌڪو(dma_q, dma_make_معلومات(src0_spad, معلومات_src0 + (pref_بلاڪ * src0_صف_stride)),            \
                               src0_صف_سائيز_aligned, src0_صف_stride, src0_صف_سائيز, pref_بلاڪ_سائيز);                \
                dma_قطار_ڌڪو(dma_q, dma_make_معلومات(src1_spad, معلومات_src1 + (pref_بلاڪ * src1_صف_stride)),            \
                               src1_صف_سائيز_aligned, src1_صف_stride, src1_صف_سائيز, pref_بلاڪ_سائيز);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_قطار_flush(dma_q);                                                                                        \
    }

DEFINE_GLU_في_THREAD(عائم, f32)
DEFINE_GLU_في_THREAD(_Float16, f16)

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

        حالة HTP_OP_GLU_GEGLU_تيز:
            compute_fn = geglu_تيز_f32;
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
    ثابت size_t dst_صف_سائيز = dst->ne[0] * SIZEOF_FP32;

    uint32_t صف_شروعات = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        uint32_t صفون_في_chunk = 0;
        htp_tensor_mdev_صفون_في_chunk(dst, حجم(عائم), (uint32_t) dst_صف_سائيز, &صفون_في_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(src0_nrows, صفون_في_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بائيٽون        = dst_صف_سائيز;
    ثابت size_t src0_صف_سائيز   = nc_بائيٽون;
    ثابت size_t src1_صف_سائيز   = nc_بائيٽون;
    ثابت size_t src0_صف_stride = src0->nb[1];
    ثابت size_t src1_صف_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_صف_سائيز_aligned = hex_ڦيرو_مٿي(src0_صف_سائيز, VLEN);
    ثابت size_t src1_صف_سائيز_aligned = hex_ڦيرو_مٿي(src1_صف_سائيز, VLEN);
    ثابت size_t dst_صف_سائيز_aligned  = hex_ڦيرو_مٿي(dst_صف_سائيز, VLEN);

    بنية htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_صف_سائيز_aligned, src1_صف_سائيز_aligned, dst_صف_سائيز_aligned, n_ڌاڳا,
                              octx->ctx->vtcm_سائيز);

    // Make sure the reserved vtcm size is sufficient
    إذا (L.vtcm_صف_في_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_سائيز,
             (src0_صف_سائيز_aligned + src1_صف_سائيز_aligned + dst_صف_سائيز_aligned) * n_ڌاڳا);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    إذا (src1) {
        FARF(مٿاهون_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بائيٽون_في_thread * n_ڌاڳا,
             L.src1_بائيٽون_في_thread * n_ڌاڳا, L.dst_بائيٽون_في_thread * n_ڌاڳا);
    } وإلا {
        FARF(مٿاهون_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بائيٽون_في_thread * n_ڌاڳا, L.src1_بائيٽون_في_thread * n_ڌاڳا, L.dst_بائيٽون_في_thread * n_ڌاڳا);
    }

    // Prepare context
    بنية htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (فراغ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_في_thread = fastdiv(nrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);

    actx.src0_صف_سائيز = src0_صف_سائيز;
    actx.src1_صف_سائيز = src1_صف_سائيز;
    actx.dst_صف_سائيز  = dst_صف_سائيز;

    actx.src0_صف_سائيز_aligned = src0_صف_سائيز_aligned;
    actx.src1_صف_سائيز_aligned = src1_صف_سائيز_aligned;
    actx.dst_صف_سائيز_aligned  = dst_صف_سائيز_aligned;

    actx.src0_صف_stride = src0_صف_stride;
    actx.src1_صف_stride = src1_صف_stride;

    uint8_t * ثابت بنياد = (uint8_t *) octx->ctx->vtcm_بنياد;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_dst);

    actx.vtcm_src0_سائيز_في_thread = L.src0_بائيٽون_في_thread;
    actx.vtcm_src1_سائيز_في_thread = L.src1_بائيٽون_في_thread;
    actx.vtcm_dst_سائيز_في_thread  = L.dst_بائيٽون_في_thread;

    actx.src0_spad_half_سائيز = L.src0_بائيٽون_في_thread / 2;
    actx.src1_spad_half_سائيز = L.src1_بائيٽون_في_thread / 2;
    actx.dst_spad_half_سائيز  = L.dst_بائيٽون_في_thread / 2;

    actx.بلاڪ = actx.src0_spad_half_سائيز / actx.src0_صف_سائيز_aligned;
    actx.src0_nrows = nrows;
    actx.صف_شروعات  = صف_شروعات;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t معلومات_src0 = src0->معلومات;
    dma_addr_t معلومات_src1 = src1 ? src1->معلومات : 0;

    إذا (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU ||
                  octx->op == HTP_OP_GLU_GEGLU_تيز ||
                  octx->op == HTP_OP_GLU_GEGLU_ERF)) {
         ثابت int32_t swapped = octx->op_params[1];
         معلومات_src1 = معلومات_src0;
         actx.src1_صف_سائيز = actx.src0_صف_سائيز;

         size_t nc_اندريون_بائيٽون = actx.nc * SIZEOF_FP32;
         إذا (swapped) {
             معلومات_src0 += nc_اندريون_بائيٽون;
         } وإلا {
             معلومات_src1 += nc_اندريون_بائيٽون;
         }
    }

    actx.معلومات_src0 = معلومات_src0;
    actx.معلومات_src1 = معلومات_src1;
    actx.معلومات_dst  = dst->معلومات;

    work_قطار_لاڳيتو(octx->ctx->work_قطار, (ڪم_ڪندڙ_callback_t)glu_f32_في_thread, &actx, n_ڌاڳا);
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
    ثابت size_t dst_صف_سائيز = dst->ne[0] * SIZEOF_FP16;

    uint32_t صف_شروعات = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        uint32_t صفون_في_chunk = 0;
        htp_tensor_mdev_صفون_في_chunk(dst, حجم(_Float16), (uint32_t) dst_صف_سائيز, &صفون_في_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(src0_nrows, صفون_في_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بائيٽون        = dst_صف_سائيز;
    ثابت size_t src0_صف_سائيز   = nc_بائيٽون;
    ثابت size_t src1_صف_سائيز   = nc_بائيٽون;
    ثابت size_t src0_صف_stride = src0->nb[1];
    ثابت size_t src1_صف_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_صف_سائيز_aligned = hex_ڦيرو_مٿي(src0_صف_سائيز, VLEN);
    ثابت size_t src1_صف_سائيز_aligned = hex_ڦيرو_مٿي(src1_صف_سائيز, VLEN);
    ثابت size_t dst_صف_سائيز_aligned  = hex_ڦيرو_مٿي(dst_صف_سائيز, VLEN);

    بنية htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_صف_سائيز_aligned, src1_صف_سائيز_aligned, dst_صف_سائيز_aligned, n_ڌاڳا,
                              octx->ctx->vtcm_سائيز);

    // Make sure the reserved vtcm size is sufficient
    إذا (L.vtcm_صف_في_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_سائيز,
             (src0_صف_سائيز_aligned + src1_صف_سائيز_aligned + dst_صف_سائيز_aligned) * n_ڌاڳا);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    إذا (src1) {
        FARF(مٿاهون_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بائيٽون_في_thread * n_ڌاڳا,
             L.src1_بائيٽون_في_thread * n_ڌاڳا, L.dst_بائيٽون_في_thread * n_ڌاڳا);
    } وإلا {
        FARF(مٿاهون_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بائيٽون_في_thread * n_ڌاڳا, L.src1_بائيٽون_في_thread * n_ڌاڳا, L.dst_بائيٽون_في_thread * n_ڌاڳا);
    }

    // Prepare context
    بنية htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (فراغ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_في_thread = fastdiv(nrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);

    actx.src0_صف_سائيز = src0_صف_سائيز;
    actx.src1_صف_سائيز = src1_صف_سائيز;
    actx.dst_صف_سائيز  = dst_صف_سائيز;

    actx.src0_صف_سائيز_aligned = src0_صف_سائيز_aligned;
    actx.src1_صف_سائيز_aligned = src1_صف_سائيز_aligned;
    actx.dst_صف_سائيز_aligned  = dst_صف_سائيز_aligned;

    actx.src0_صف_stride = src0_صف_stride;
    actx.src1_صف_stride = src1_صف_stride;

    uint8_t * ثابت بنياد = (uint8_t *) octx->ctx->vtcm_بنياد;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_dst);

    actx.vtcm_src0_سائيز_في_thread = L.src0_بائيٽون_في_thread;
    actx.vtcm_src1_سائيز_في_thread = L.src1_بائيٽون_في_thread;
    actx.vtcm_dst_سائيز_في_thread  = L.dst_بائيٽون_في_thread;

    actx.src0_spad_half_سائيز = L.src0_بائيٽون_في_thread / 2;
    actx.src1_spad_half_سائيز = L.src1_بائيٽون_في_thread / 2;
    actx.dst_spad_half_سائيز  = L.dst_بائيٽون_في_thread / 2;

    actx.بلاڪ = actx.src0_spad_half_سائيز / actx.src0_صف_سائيز_aligned;
    actx.src0_nrows = nrows;
    actx.صف_شروعات  = صف_شروعات;

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
         actx.src1_صف_سائيز = actx.src0_صف_سائيز;

         size_t nc_اندريون_بائيٽون = actx.nc * SIZEOF_FP16;
         إذا (swapped) {
             معلومات_src0 += nc_اندريون_بائيٽون;
         } وإلا {
             معلومات_src1 += nc_اندريون_بائيٽون;
         }
    }

    actx.معلومات_src0 = معلومات_src0;
    actx.معلومات_src1 = معلومات_src1;
    actx.معلومات_dst  = dst->معلومات;

    work_قطار_لاڳيتو(octx->ctx->work_قطار, (ڪم_ڪندڙ_callback_t)glu_f16_في_thread, &actx, n_ڌاڳا);
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
