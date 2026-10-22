#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hvx-utils.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "htp-vtcm.h"

#నిర్వచించు htp_act_preamble                                 \
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = actx->octx->src[0]; \
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = actx->octx->src[1]; \
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = actx->octx->dst;    \
                                                         \
    స్థిరరాశి uint32_t ne00 = src0->ne[0];                   \
    స్థిరరాశి uint32_t ne01 = src0->ne[1];                   \
    స్థిరరాశి uint32_t ne02 = src0->ne[2];                   \
    స్థిరరాశి uint32_t ne03 = src0->ne[3];                   \
                                                         \
    స్థిరరాశి uint32_t nb00 = src0->nb[0];                   \
    స్థిరరాశి uint32_t nb01 = src0->nb[1];                   \
    స్థిరరాశి uint32_t nb02 = src0->nb[2];                   \
    స్థిరరాశి uint32_t nb03 = src0->nb[3];                   \
                                                         \
    స్థిరరాశి uint32_t ne10 = src1 ? src1->ne[0] : 0;        \
    స్థిరరాశి uint32_t ne11 = src1 ? src1->ne[1] : 0;        \
    స్థిరరాశి uint32_t ne12 = src1 ? src1->ne[2] : 0;        \
    స్థిరరాశి uint32_t ne13 = src1 ? src1->ne[3] : 0;        \
                                                         \
    స్థిరరాశి uint32_t nb10 = src1 ? src1->nb[0] : 0;        \
    స్థిరరాశి uint32_t nb11 = src1 ? src1->nb[1] : 0;        \
    స్థిరరాశి uint32_t nb12 = src1 ? src1->nb[2] : 0;        \
    స్థిరరాశి uint32_t nb13 = src1 ? src1->nb[3] : 0;        \
                                                         \
    స్థిరరాశి uint32_t ne0 = dst->ne[0];                     \
    స్థిరరాశి uint32_t ne1 = dst->ne[1];                     \
    స్థిరరాశి uint32_t ne2 = dst->ne[2];                     \
    స్థిరరాశి uint32_t ne3 = dst->ne[3];                     \
                                                         \
    స్థిరరాశి uint32_t nb0 = dst->nb[0];                     \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];                     \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];                     \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

నిర్మాణం htp_act_context;

రకనిర్వచనం శూన్య (*glu_compute_f32_fn_t)(స్థిరరాశి దశాంశం * పరిమిత src0,
                                     స్థిరరాశి దశాంశం * పరిమిత src1,
                                     దశాంశం * పరిమిత dst,
                                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                                     స్థిరరాశి నిర్మాణం htp_act_context * actx);

రకనిర్వచనం శూన్య (*glu_compute_f16_fn_t)(స్థిరరాశి _Float16 * పరిమిత src0,
                                     స్థిరరాశి _Float16 * పరిమిత src1,
                                     _Float16 * పరిమిత dst,
                                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                                     స్థిరరాశి నిర్మాణం htp_act_context * actx);

నిర్మాణం htp_act_context {
    నిర్మాణం htp_ops_context * octx;

    శూన్య *                   compute;
    స్థిరరాశి అక్షరం *             op_str;

    // Precomputed values
    dma_addr_t               దత్తాంశం_src0;
    dma_addr_t               దత్తాంశం_src1;
    dma_addr_t               దత్తాంశం_dst;

    size_t                   src0_అడ్డువరుస_కొలత;
    size_t                   src1_అడ్డువరుస_కొలత;
    size_t                   dst_అడ్డువరుస_కొలత;

    size_t                   src0_అడ్డువరుస_stride;
    size_t                   src1_అడ్డువరుస_stride;

    size_t                   src0_అడ్డువరుస_కొలత_aligned;
    size_t                   src1_అడ్డువరుస_కొలత_aligned;
    size_t                   dst_అడ్డువరుస_కొలత_aligned;

    size_t                   src0_spad_half_కొలత;
    size_t                   src1_spad_half_కొలత;
    size_t                   dst_spad_half_కొలత;

    uint32_t                 ఖండం;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_చొప్పున_thread;
    uint32_t                 అడ్డువరుస_ప్రారంభం;
    పూర్ణాంకం                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_కొలత_చొప్పున_thread;
    size_t                   vtcm_src1_కొలత_చొప్పున_thread;
    size_t                   vtcm_dst_కొలత_చొప్పున_thread;
};

నిర్మాణం htp_act_vtcm_layout {
    size_t మొత్తం_బైట్లు;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_బైట్లు_చొప్పున_thread;
    size_t src1_బైట్లు_చొప్పున_thread;
    size_t dst_బైట్లు_చొప్పున_thread;

    uint32_t vtcm_అడ్డువరుస_చొప్పున_thread;
};

స్థిర అంతఃవరుస శూన్య htp_act_vtcm_layout_build(నిర్మాణం htp_act_vtcm_layout * L,
                                             size_t                       src0_అడ్డువరుస_కొలత_aligned,
                                             size_t                       src1_అడ్డువరుస_కొలత_aligned,
                                             size_t                       dst_అడ్డువరుస_కొలత_aligned,
                                             uint32_t                     n_దారాలు,
                                             size_t                       vtcm_కొలత) {
    స్థిరరాశి size_t   spad_కొలత_చొప్పున_అడ్డువరుస   = src0_అడ్డువరుస_కొలత_aligned + src1_అడ్డువరుస_కొలత_aligned + dst_అడ్డువరుస_కొలత_aligned;
    స్థిరరాశి uint32_t vtcm_అడ్డువరుస_చొప్పున_thread = (uint32_t) (vtcm_కొలత / (n_దారాలు * spad_కొలత_చొప్పున_అడ్డువరుస));

    L->vtcm_అడ్డువరుస_చొప్పున_thread = vtcm_అడ్డువరుస_చొప్పున_thread;

    L->src0_బైట్లు_చొప్పున_thread = src0_అడ్డువరుస_కొలత_aligned * vtcm_అడ్డువరుస_చొప్పున_thread;
    L->src1_బైట్లు_చొప్పున_thread = src1_అడ్డువరుస_కొలత_aligned * vtcm_అడ్డువరుస_చొప్పున_thread;
    L->dst_బైట్లు_చొప్పున_thread  = dst_అడ్డువరుస_కొలత_aligned * vtcm_అడ్డువరుస_చొప్పున_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_బైట్లు_చొప్పున_thread * n_దారాలు;
    L->off_dst  = L->off_src1 + L->src1_బైట్లు_చొప్పున_thread * n_దారాలు;

    L->మొత్తం_బైట్లు = L->off_dst + L->dst_బైట్లు_చొప్పున_thread * n_దారాలు;
}

#నిర్వచించు htp_glu_op_preamble                                            \
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = actx->src0_అడ్డువరుస_కొలత_aligned;  \
    స్థిరరాశి size_t src1_అడ్డువరుస_కొలత_aligned = actx->src1_అడ్డువరుస_కొలత_aligned;  \
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = actx->dst_అడ్డువరుస_కొలత_aligned;   \
    స్థిరరాశి పూర్ణాంకం    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
స్థిర శూన్య swiglu_f32(స్థిరరాశి దశాంశం * పరిమిత src0,
                                                 స్థిరరాశి దశాంశం * పరిమిత src1,
                                                 దశాంశం * పరిమిత dst,
                                                 స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                                                 స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
స్థిర శూన్య swiglu_oai_f32(స్థిరరాశి దశాంశం * పరిమిత src0,
                                                     స్థిరరాశి దశాంశం * పరిమిత src1,
                                                     దశాంశం * పరిమిత dst,
                                                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                                                     స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;
    స్థిరరాశి దశాంశం alpha = ((స్థిరరాశి దశాంశం *) (actx->octx->op_params))[2];
    స్థిరరాశి దశాంశం హద్దు = ((స్థిరరాశి దశాంశం *) (actx->octx->op_params))[3];

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, హద్దు, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -హద్దు, హద్దు, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_కూడు_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f32(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f32_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

స్థిర శూన్య swiglu_clamp_f32(స్థిరరాశి దశాంశం * పరిమిత src0,
                                                       స్థిరరాశి దశాంశం * పరిమిత src1,
                                                       దశాంశం * పరిమిత dst,
                                                       స్థిరరాశి uint32_t                 అంకె_అడ్డువరుసలు_2,
                                                       స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;
    స్థిరరాశి దశాంశం హద్దు = ((స్థిరరాశి దశాంశం *) (actx->octx->op_params))[3];

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, హద్దు, nc);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -హద్దు, హద్దు, nc);
        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

స్థిర స్థిరరాశి దశాంశం GELU_COEF_A     = 0.044715f;
స్థిర స్థిరరాశి దశాంశం SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_fast_sigmoid_f32_2it(HVX_సదిశ v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_సదిశ లోపల_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_సదిశ x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(లోపల_int));
    HVX_సదిశ xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_సదిశ v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_సదిశ v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, లోపల_int, 24);

    HVX_సదిశ v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_సదిశ v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_సదిశ two_sf = hvx_vec_splat_f32(2.0f);
    HVX_సదిశ i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_సదిశ r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_సదిశ res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    తిరిగి Q6_Vsf_equals_Vqf32(res);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_vec_fast_sigmoid_f32_guard_2it(HVX_సదిశ v,
                                                            HVX_సదిశ one,
                                                            HVX_సదిశ max_exp,
                                                            HVX_సదిశ min_exp) {
    స్థిరరాశి HVX_సదిశ_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    స్థిరరాశి HVX_సదిశ_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_సదిశ బహిర్గతం = hvx_vec_fast_sigmoid_f32_2it(v);
    బహిర్గతం            = Q6_V_vmux_QVV(pred_max, బహిర్గతం, one);
    తిరిగి Q6_V_vmux_QVV(pred_min, బహిర్గతం, Q6_V_vzero());
}

స్థిర అంతఃవరుస శూన్య hvx_geglu_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst  % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src0 % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src1 % 128 == 0);

    HVX_సదిశ * పరిమిత vdst        = (HVX_సదిశ *) dst;
    స్థిరరాశి HVX_సదిశ * పరిమిత vsrc0 = (స్థిరరాశి HVX_సదిశ *) src0;
    స్థిరరాశి HVX_సదిశ * పరిమిత vsrc1 = (స్థిరరాశి HVX_సదిశ *) src1;

    స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;

    స్థిరరాశి దశాంశం GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    స్థిరరాశి HVX_సదిశ v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    స్థిరరాశి HVX_సదిశ v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    స్థిరరాశి HVX_సదిశ v_one               = hvx_vec_splat_f32(1.0f);
    స్థిరరాశి HVX_సదిశ v_max_exp           = hvx_vec_splat_f32(87.0f);
    స్థిరరాశి HVX_సదిశ v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        HVX_సదిశ x = vsrc0[i];
        HVX_సదిశ g = vsrc1[i];

        HVX_సదిశ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_సదిశ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_కూడు_f32_f32(coef, v_sqrt_2_pi);
        HVX_సదిశ inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner = inner + inner
        HVX_సదిశ y2 = hvx_vec_కూడు_f32_f32(inner, inner);

        // Fast sigmoid approximation (2 iterations)
        HVX_సదిశ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_సదిశ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ x = vsrc0[i];
        HVX_సదిశ g = vsrc1[i];

        HVX_సదిశ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_సదిశ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_కూడు_f32_f32(coef, v_sqrt_2_pi);
        HVX_సదిశ inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_సదిశ y2 = hvx_vec_కూడు_f32_f32(inner, inner);

        HVX_సదిశ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_సదిశ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        HVX_సదిశ res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * పరిమాణం(దశాంశం), res);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_geglu_శీఘ్ర_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst  % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src0 % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src1 % 128 == 0);

    HVX_సదిశ * పరిమిత vdst        = (HVX_సదిశ *) dst;
    స్థిరరాశి HVX_సదిశ * పరిమిత vsrc0 = (స్థిరరాశి HVX_సదిశ *) src0;
    స్థిరరాశి HVX_సదిశ * పరిమిత vsrc1 = (స్థిరరాశి HVX_సదిశ *) src1;

    స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;

    స్థిరరాశి HVX_సదిశ v_scale    = hvx_vec_splat_f32(1.702f);
    స్థిరరాశి HVX_సదిశ v_one      = hvx_vec_splat_f32(1.0f);
    స్థిరరాశి HVX_సదిశ v_max_exp  = hvx_vec_splat_f32(87.0f);
    స్థిరరాశి HVX_సదిశ v_min_exp  = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        HVX_సదిశ x = vsrc0[i];
        HVX_సదిశ g = vsrc1[i];
        HVX_సదిశ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_సదిశ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ x = vsrc0[i];
        HVX_సదిశ g = vsrc1[i];
        HVX_సదిశ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_సదిశ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        HVX_సదిశ ఫలితం = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * పరిమాణం(దశాంశం), ఫలితం);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_geglu_erf_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst  % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src0 % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src1 % 128 == 0);

    HVX_సదిశ * పరిమిత vdst        = (HVX_సదిశ *) dst;
    స్థిరరాశి HVX_సదిశ * పరిమిత vsrc0 = (స్థిరరాశి HVX_సదిశ *) src0;
    స్థిరరాశి HVX_సదిశ * పరిమిత vsrc1 = (స్థిరరాశి HVX_సదిశ *) src1;

    స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        HVX_సదిశ x = vsrc0[i];
        HVX_సదిశ g = vsrc1[i];
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ x = vsrc0[i];
        HVX_సదిశ g = vsrc1[i];
        HVX_సదిశ ఫలితం = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * పరిమాణం(దశాంశం), ఫలితం);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_geglu_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src0, స్థిరరాశి uint8_t * పరిమిత src1, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst  % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src0 % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src1 % 128 == 0);

    HVX_సదిశ * పరిమిత vdst        = (HVX_సదిశ *) dst;
    స్థిరరాశి HVX_సదిశ * పరిమిత vsrc0 = (స్థిరరాశి HVX_సదిశ *) src0;
    స్థిరరాశి HVX_సదిశ * పరిమిత vsrc1 = (స్థిరరాశి HVX_సదిశ *) src1;

    స్థిరరాశి uint32_t epv  = 128 / పరిమాణం(_Float16);
    స్థిరరాశి uint32_t nvec = n / epv;
    స్థిరరాశి uint32_t nloe = n % epv;

    స్థిరరాశి HVX_సదిశ v_coef_a    = hvx_vec_splat_f16((_Float16) GELU_COEF_A);
    స్థిరరాశి HVX_సదిశ v_sqrt_2_pi = hvx_vec_splat_f16((_Float16) SQRT_2_OVER_PI);
    స్థిరరాశి HVX_సదిశ v_one       = hvx_vec_splat_f16((_Float16) 1.0f);
    స్థిరరాశి HVX_సదిశ v_half      = hvx_vec_splat_f16((_Float16) 0.5f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        HVX_సదిశ x = vsrc0[i];
        HVX_సదిశ g = vsrc1[i];

        // gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))
        HVX_సదిశ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_సదిశ coef  = hvx_vec_కూడు_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_సదిశ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_సదిశ t = hvx_vec_కూడు_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_సదిశ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        vdst[i] = hvx_vec_mul_f16_f16(gelu_x, g);
    }

    ఒకవేళ (nloe) {
        HVX_సదిశ x = vsrc0[i];
        HVX_సదిశ g = vsrc1[i];

        HVX_సదిశ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_సదిశ coef  = hvx_vec_కూడు_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_సదిశ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_సదిశ t = hvx_vec_కూడు_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_సదిశ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        HVX_సదిశ res    = hvx_vec_mul_f16_f16(gelu_x, g);
        hvx_vec_store_a((శూన్య *) &vdst[i], nloe * పరిమాణం(_Float16), res);
    }
}

// geglu(x, g) = gelu(x) * g
స్థిర శూన్య geglu_f32(స్థిరరాశి దశాంశం * పరిమిత src0,
                                                స్థిరరాశి దశాంశం * పరిమిత src1,
                                                దశాంశం * పరిమిత dst,
                                                స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                                                స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_quick(x, g) = x * sigmoid(1.702 * x) * g
స్థిర శూన్య geglu_శీఘ్ర_f32(స్థిరరాశి దశాంశం * పరిమిత src0,
                            స్థిరరాశి దశాంశం * పరిమిత src1,
                            దశాంశం * పరిమిత dst,
                            స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                            స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        hvx_geglu_శీఘ్ర_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_erf(x, g) = gelu_erf(x) * g
స్థిర శూన్య geglu_erf_f32(స్థిరరాశి దశాంశం * పరిమిత src0,
                          స్థిరరాశి దశాంశం * పరిమిత src1,
                          దశాంశం * పరిమిత dst,
                          స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                          స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        hvx_geglu_erf_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// swiglu(x) = x1 * sigmoid(x0)
స్థిర శూన్య swiglu_f16(స్థిరరాశి _Float16 * పరిమిత src0,
                       స్థిరరాశి _Float16 * పరిమిత src1,
                       _Float16 * పరిమిత dst,
                       స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                       స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
స్థిర శూన్య swiglu_oai_f16(స్థిరరాశి _Float16 * పరిమిత src0,
                           స్థిరరాశి _Float16 * పరిమిత src1,
                           _Float16 * పరిమిత dst,
                           స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                           స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;
    స్థిరరాశి _Float16 alpha = (_Float16) ((స్థిరరాశి దశాంశం *) (actx->octx->op_params))[2];
    స్థిరరాశి _Float16 హద్దు = (_Float16) ((స్థిరరాశి దశాంశం *) (actx->octx->op_params))[3];

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, హద్దు, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -హద్దు, హద్దు, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_కూడు_scalar_f16((uint8_t *) src1_ptr, src1_ptr, (_Float16) 1.0f, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f16(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f16_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

స్థిర శూన్య swiglu_clamp_f16(స్థిరరాశి _Float16 * పరిమిత src0,
                             స్థిరరాశి _Float16 * పరిమిత src1,
                             _Float16 * పరిమిత dst,
                             స్థిరరాశి uint32_t                 అంకె_అడ్డువరుసలు_2,
                             స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;
    స్థిరరాశి _Float16 హద్దు = (_Float16) ((స్థిరరాశి దశాంశం *) (actx->octx->op_params))[3];

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, హద్దు, nc);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -హద్దు, హద్దు, nc);
        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// geglu(x, g) = gelu(x) * g, gelu via tanh approximation
స్థిర శూన్య geglu_f16(స్థిరరాశి _Float16 * పరిమిత src0,
                      స్థిరరాశి _Float16 * పరిమిత src1,
                      _Float16 * పరిమిత dst,
                      స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                      స్థిరరాశి నిర్మాణం htp_act_context * actx) {
    htp_glu_op_preamble;

    కొరకు (uint32_t ib = 0; ib < అంకె_అడ్డువరుసలు_2; ib++) {
        స్థిరరాశి uint8_t * పరిమిత src0_ptr = (స్థిరరాశి uint8_t *) src0 + (ib * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత src1_ptr = (స్థిరరాశి uint8_t *) src1 + (ib * src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_ptr        = (uint8_t *) dst + (ib * dst_అడ్డువరుస_కొలత_aligned);

        hvx_geglu_f16_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#నిర్వచించు DEFINE_GLU_చొప్పున_THREAD(TYPE, SUFFIX)                                                                            \
    స్థిర శూన్య glu_##SUFFIX##_per_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {                           \
        నిర్మాణం htp_act_context * actx = (నిర్మాణం htp_act_context *) దత్తాంశం;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        నిర్మాణం htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : ఖాళీ;                          \
                                                                                                                       \
        size_t src0_అడ్డువరుస_కొలత = actx->src0_అడ్డువరుస_కొలత;                                                                    \
        size_t src1_అడ్డువరుస_కొలత = actx->src1_అడ్డువరుస_కొలత;                                                                    \
        size_t dst_అడ్డువరుస_కొలత  = actx->dst_అడ్డువరుస_కొలత;                                                                     \
                                                                                                                       \
        size_t src0_అడ్డువరుస_stride = actx->src0_అడ్డువరుస_stride;                                                                \
        size_t src1_అడ్డువరుస_stride = actx->src1_అడ్డువరుస_stride;                                                                \
                                                                                                                       \
        స్థిరరాశి uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        స్థిరరాశి uint32_t src0_nrows_చొప్పున_thread = actx->src0_nrows_చొప్పున_thread;                                            \
                                                                                                                       \
        స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = actx->అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;                                 \
        స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, actx->అడ్డువరుస_ప్రారంభం + src0_nrows);     \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {                                                                          \
            తిరిగి;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        స్థిరరాశి dma_addr_t దత్తాంశం_src0 = actx->దత్తాంశం_src0;                                                                  \
        స్థిరరాశి dma_addr_t దత్తాంశం_src1 = actx->దత్తాంశం_src1;                                                                  \
        స్థిరరాశి dma_addr_t దత్తాంశం_dst  = actx->దత్తాంశం_dst;                                                                   \
                                                                                                                       \
        స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = actx->src0_అడ్డువరుస_కొలత_aligned;                                              \
        స్థిరరాశి size_t src1_అడ్డువరుస_కొలత_aligned = actx->src1_అడ్డువరుస_కొలత_aligned;                                              \
        స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = actx->dst_అడ్డువరుస_కొలత_aligned;                                               \
                                                                                                                       \
        uint8_t * పరిమిత src0_spad_దత్తాంశం = actx->vtcm_src0 + (ith * actx->vtcm_src0_కొలత_చొప్పున_thread);                 \
        uint8_t * పరిమిత src1_spad_దత్తాంశం = actx->vtcm_src1 + (ith * actx->vtcm_src1_కొలత_చొప్పున_thread);                 \
        uint8_t * పరిమిత dst_spad_దత్తాంశం  = actx->vtcm_dst  + (ith * actx->vtcm_dst_కొలత_చొప్పున_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_కొలత = actx->src0_spad_half_కొలత;                                                        \
        size_t src1_spad_half_కొలత = actx->src1_spad_half_కొలత;                                                        \
        size_t dst_spad_half_కొలత  = actx->dst_spad_half_కొలత;                                                         \
                                                                                                                       \
        స్థిరరాశి పూర్ణాంకం ఖండం_2 = actx->ఖండం;                                                                                 \
        ఒకవేళ (ఖండం_2 == 0) {                                                                                              \
            FARF(ERROR, "%s : VTCM reservation %zu is too small, needed %zu\n",                                        \
                 actx->op_str, actx->vtcm_src0_కొలత_చొప్పున_thread, src0_అడ్డువరుస_కొలత_aligned);                                \
            తిరిగి;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_వరుస * dma_q = actx->octx->ctx->dma[ith];                                                                 \
        glu_compute_##SUFFIX##_fn_t compute = (glu_compute_##SUFFIX##_fn_t) actx->compute;                             \
                                                                                                                       \
        కొరకు (uint32_t ir = src0_ప్రారంభం_అడ్డువరుస, spad_idx = 0; ir < src0_చివర_అడ్డువరుస && spad_idx < 2; ir += ఖండం_2, spad_idx++) { \
            స్థిరరాశి uint32_t ఖండం_కొలత_3 = MIN(ఖండం_2, src0_చివర_అడ్డువరుస - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_వరుస_నెట్టు(dma_q,                                                                                      \
                           dma_make_దత్తాంశం(దత్తాంశం_dst, dst_spad_దత్తాంశం + (spad_idx * dst_spad_half_కొలత)),                   \
                           dst_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 0);                                       \
                                                                                                                       \
            dma_వరుస_నెట్టు(dma_q,                                                                                      \
                dma_make_దత్తాంశం(src0_spad_దత్తాంశం + (spad_idx * src0_spad_half_కొలత), దత్తాంశం_src0 + (ir * src0_అడ్డువరుస_stride)),  \
                src0_అడ్డువరుస_కొలత_aligned, src0_అడ్డువరుస_stride, src0_అడ్డువరుస_కొలత, ఖండం_కొలత_3);                                    \
                                                                                                                       \
            dma_వరుస_నెట్టు(dma_q,                                                                                      \
                dma_make_దత్తాంశం(src1_spad_దత్తాంశం + (spad_idx * src1_spad_half_కొలత), దత్తాంశం_src1 + (ir * src1_అడ్డువరుస_stride)),  \
                src1_అడ్డువరుస_కొలత_aligned, src1_అడ్డువరుస_stride, src1_అడ్డువరుస_కొలత, ఖండం_కొలత_3);                                    \
        }                                                                                                              \
                                                                                                                       \
        కొరకు (uint32_t ir = src0_ప్రారంభం_అడ్డువరుస; ir < src0_చివర_అడ్డువరుస; ir += ఖండం_2) {                                           \
            స్థిరరాశి uint32_t ఖండం_కొలత_3 = MIN(ఖండం_2, src0_చివర_అడ్డువరుస - ir);                                                 \
                                                                                                                       \
            TYPE * dst_spad  = (TYPE *) dma_వరుస_తీయి(dma_q).src;                                                      \
            TYPE * src0_spad = (TYPE *) dma_వరుస_తీయి(dma_q).dst;                                                      \
            TYPE * src1_spad = (TYPE *) dma_వరుస_తీయి(dma_q).dst;                                                      \
                                                                                                                       \
            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            compute(src0_spad, src1_spad, dst_spad, ఖండం_కొలత_3, actx);                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(దత్తాంశం_dst + (ir * dst_అడ్డువరుస_కొలత), dst_spad),                             \
                           dst_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, ఖండం_కొలత_3);                              \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            స్థిరరాశి uint32_t pref_ఖండం = (ir + ఖండం_2 * 2);                                                              \
            ఒకవేళ (pref_ఖండం < src0_చివర_అడ్డువరుస) {                                                                           \
                స్థిరరాశి uint32_t pref_ఖండం_కొలత = MIN(ఖండం_2, src0_చివర_అడ్డువరుస - pref_ఖండం);                                \
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(src0_spad, దత్తాంశం_src0 + (pref_ఖండం * src0_అడ్డువరుస_stride)),            \
                               src0_అడ్డువరుస_కొలత_aligned, src0_అడ్డువరుస_stride, src0_అడ్డువరుస_కొలత, pref_ఖండం_కొలత);                \
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(src1_spad, దత్తాంశం_src1 + (pref_ఖండం * src1_అడ్డువరుస_stride)),            \
                               src1_అడ్డువరుస_కొలత_aligned, src1_అడ్డువరుస_stride, src1_అడ్డువరుస_కొలత, pref_ఖండం_కొలత);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_వరుస_flush(dma_q);                                                                                        \
    }

DEFINE_GLU_చొప్పున_THREAD(దశాంశం, f32)
DEFINE_GLU_చొప్పున_THREAD(_Float16, f16)

స్థిర పూర్ణాంకం execute_op_activations_f32(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఒకవేళ ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    glu_compute_f32_fn_t compute_fn = ఖాళీ;
    స్థిరరాశి అక్షరం *         op_type    = ఖాళీ;

    ఎంపిక (octx->op) {
        స్థితి HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f32;
            op_type    = "swiglu-f32";
            ఆపు;

        స్థితి HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f32;
            op_type    = "swiglu-oai-f32";
            ఆపు;

        స్థితి HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f32;
            op_type    = "swiglu-clamp-f32";
            ఆపు;

        స్థితి HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f32;
            op_type    = "geglu-f32";
            ఆపు;

        స్థితి HTP_OP_GLU_GEGLU_శీఘ్ర:
            compute_fn = geglu_శీఘ్ర_f32;
            op_type    = "geglu-quick-f32";
            ఆపు;

        స్థితి HTP_OP_GLU_GEGLU_ERF:
            compute_fn = geglu_erf_f32;
            op_type    = "geglu-erf-f32";
            ఆపు;

        అప్రమేయ:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత = dst->ne[0] * SIZEOF_FP32;

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = src0_nrows;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, పరిమాణం(దశాంశం), (uint32_t) dst_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(src0_nrows, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    స్థిరరాశి size_t nc_బైట్లు        = dst_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత   = nc_బైట్లు;
    స్థిరరాశి size_t src1_అడ్డువరుస_కొలత   = nc_బైట్లు;
    స్థిరరాశి size_t src0_అడ్డువరుస_stride = src0->nb[1];
    స్థిరరాశి size_t src1_అడ్డువరుస_stride = src1 ? src1->nb[1] : src0->nb[1];

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src0_అడ్డువరుస_కొలత, VLEN);
    స్థిరరాశి size_t src1_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src1_అడ్డువరుస_కొలత, VLEN);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = hex_విడత_పైకి(dst_అడ్డువరుస_కొలత, VLEN);

    నిర్మాణం htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_అడ్డువరుస_కొలత_aligned, src1_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత_aligned, n_దారాలు,
                              octx->ctx->vtcm_కొలత);

    // Make sure the reserved vtcm size is sufficient
    ఒకవేళ (L.vtcm_అడ్డువరుస_చొప్పున_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_కొలత,
             (src0_అడ్డువరుస_కొలత_aligned + src1_అడ్డువరుస_కొలత_aligned + dst_అడ్డువరుస_కొలత_aligned) * n_దారాలు);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    ఒకవేళ (src1) {
        FARF(ఎగువ_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_బైట్లు_చొప్పున_thread * n_దారాలు,
             L.src1_బైట్లు_చొప్పున_thread * n_దారాలు, L.dst_బైట్లు_చొప్పున_thread * n_దారాలు);
    } లేకపోతే {
        FARF(ఎగువ_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_బైట్లు_చొప్పున_thread * n_దారాలు, L.src1_బైట్లు_చొప్పున_thread * n_దారాలు, L.dst_బైట్లు_చొప్పున_thread * n_దారాలు);
    }

    // Prepare context
    నిర్మాణం htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (శూన్య *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div);

    actx.src0_అడ్డువరుస_కొలత = src0_అడ్డువరుస_కొలత;
    actx.src1_అడ్డువరుస_కొలత = src1_అడ్డువరుస_కొలత;
    actx.dst_అడ్డువరుస_కొలత  = dst_అడ్డువరుస_కొలత;

    actx.src0_అడ్డువరుస_కొలత_aligned = src0_అడ్డువరుస_కొలత_aligned;
    actx.src1_అడ్డువరుస_కొలత_aligned = src1_అడ్డువరుస_కొలత_aligned;
    actx.dst_అడ్డువరుస_కొలత_aligned  = dst_అడ్డువరుస_కొలత_aligned;

    actx.src0_అడ్డువరుస_stride = src0_అడ్డువరుస_stride;
    actx.src1_అడ్డువరుస_stride = src1_అడ్డువరుస_stride;

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_dst);

    actx.vtcm_src0_కొలత_చొప్పున_thread = L.src0_బైట్లు_చొప్పున_thread;
    actx.vtcm_src1_కొలత_చొప్పున_thread = L.src1_బైట్లు_చొప్పున_thread;
    actx.vtcm_dst_కొలత_చొప్పున_thread  = L.dst_బైట్లు_చొప్పున_thread;

    actx.src0_spad_half_కొలత = L.src0_బైట్లు_చొప్పున_thread / 2;
    actx.src1_spad_half_కొలత = L.src1_బైట్లు_చొప్పున_thread / 2;
    actx.dst_spad_half_కొలత  = L.dst_బైట్లు_చొప్పున_thread / 2;

    actx.ఖండం = actx.src0_spad_half_కొలత / actx.src0_అడ్డువరుస_కొలత_aligned;
    actx.src0_nrows = nrows;
    actx.అడ్డువరుస_ప్రారంభం  = అడ్డువరుస_ప్రారంభం;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t దత్తాంశం_src0 = src0->దత్తాంశం;
    dma_addr_t దత్తాంశం_src1 = src1 ? src1->దత్తాంశం : 0;

    ఒకవేళ (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU ||
                  octx->op == HTP_OP_GLU_GEGLU_శీఘ్ర ||
                  octx->op == HTP_OP_GLU_GEGLU_ERF)) {
         స్థిరరాశి int32_t swapped = octx->op_params[1];
         దత్తాంశం_src1 = దత్తాంశం_src0;
         actx.src1_అడ్డువరుస_కొలత = actx.src0_అడ్డువరుస_కొలత;

         size_t nc_లోపల_బైట్లు = actx.nc * SIZEOF_FP32;
         ఒకవేళ (swapped) {
             దత్తాంశం_src0 += nc_లోపల_బైట్లు;
         } లేకపోతే {
             దత్తాంశం_src1 += nc_లోపల_బైట్లు;
         }
    }

    actx.దత్తాంశం_src0 = దత్తాంశం_src0;
    actx.దత్తాంశం_src1 = దత్తాంశం_src1;
    actx.దత్తాంశం_dst  = dst->దత్తాంశం;

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, (కార్మికుడు_callback_t)glu_f32_చొప్పున_thread, &actx, n_దారాలు);
    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం execute_op_activations_f16(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఒకవేళ ((dst->ne[0] * SIZEOF_FP16) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    glu_compute_f16_fn_t compute_fn = ఖాళీ;
    స్థిరరాశి అక్షరం *         op_type    = ఖాళీ;

    ఎంపిక (octx->op) {
        స్థితి HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f16;
            op_type    = "swiglu-f16";
            ఆపు;

        స్థితి HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f16;
            op_type    = "swiglu-oai-f16";
            ఆపు;

        స్థితి HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f16;
            op_type    = "swiglu-clamp-f16";
            ఆపు;

        స్థితి HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f16;
            op_type    = "geglu-f16";
            ఆపు;

        అప్రమేయ:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత = dst->ne[0] * SIZEOF_FP16;

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = src0_nrows;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, పరిమాణం(_Float16), (uint32_t) dst_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(src0_nrows, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    స్థిరరాశి size_t nc_బైట్లు        = dst_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత   = nc_బైట్లు;
    స్థిరరాశి size_t src1_అడ్డువరుస_కొలత   = nc_బైట్లు;
    స్థిరరాశి size_t src0_అడ్డువరుస_stride = src0->nb[1];
    స్థిరరాశి size_t src1_అడ్డువరుస_stride = src1 ? src1->nb[1] : src0->nb[1];

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src0_అడ్డువరుస_కొలత, VLEN);
    స్థిరరాశి size_t src1_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src1_అడ్డువరుస_కొలత, VLEN);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = hex_విడత_పైకి(dst_అడ్డువరుస_కొలత, VLEN);

    నిర్మాణం htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_అడ్డువరుస_కొలత_aligned, src1_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత_aligned, n_దారాలు,
                              octx->ctx->vtcm_కొలత);

    // Make sure the reserved vtcm size is sufficient
    ఒకవేళ (L.vtcm_అడ్డువరుస_చొప్పున_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_కొలత,
             (src0_అడ్డువరుస_కొలత_aligned + src1_అడ్డువరుస_కొలత_aligned + dst_అడ్డువరుస_కొలత_aligned) * n_దారాలు);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    ఒకవేళ (src1) {
        FARF(ఎగువ_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_బైట్లు_చొప్పున_thread * n_దారాలు,
             L.src1_బైట్లు_చొప్పున_thread * n_దారాలు, L.dst_బైట్లు_చొప్పున_thread * n_దారాలు);
    } లేకపోతే {
        FARF(ఎగువ_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_బైట్లు_చొప్పున_thread * n_దారాలు, L.src1_బైట్లు_చొప్పున_thread * n_దారాలు, L.dst_బైట్లు_చొప్పున_thread * n_దారాలు);
    }

    // Prepare context
    నిర్మాణం htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (శూన్య *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div);

    actx.src0_అడ్డువరుస_కొలత = src0_అడ్డువరుస_కొలత;
    actx.src1_అడ్డువరుస_కొలత = src1_అడ్డువరుస_కొలత;
    actx.dst_అడ్డువరుస_కొలత  = dst_అడ్డువరుస_కొలత;

    actx.src0_అడ్డువరుస_కొలత_aligned = src0_అడ్డువరుస_కొలత_aligned;
    actx.src1_అడ్డువరుస_కొలత_aligned = src1_అడ్డువరుస_కొలత_aligned;
    actx.dst_అడ్డువరుస_కొలత_aligned  = dst_అడ్డువరుస_కొలత_aligned;

    actx.src0_అడ్డువరుస_stride = src0_అడ్డువరుస_stride;
    actx.src1_అడ్డువరుస_stride = src1_అడ్డువరుస_stride;

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_dst);

    actx.vtcm_src0_కొలత_చొప్పున_thread = L.src0_బైట్లు_చొప్పున_thread;
    actx.vtcm_src1_కొలత_చొప్పున_thread = L.src1_బైట్లు_చొప్పున_thread;
    actx.vtcm_dst_కొలత_చొప్పున_thread  = L.dst_బైట్లు_చొప్పున_thread;

    actx.src0_spad_half_కొలత = L.src0_బైట్లు_చొప్పున_thread / 2;
    actx.src1_spad_half_కొలత = L.src1_బైట్లు_చొప్పున_thread / 2;
    actx.dst_spad_half_కొలత  = L.dst_బైట్లు_చొప్పున_thread / 2;

    actx.ఖండం = actx.src0_spad_half_కొలత / actx.src0_అడ్డువరుస_కొలత_aligned;
    actx.src0_nrows = nrows;
    actx.అడ్డువరుస_ప్రారంభం  = అడ్డువరుస_ప్రారంభం;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t దత్తాంశం_src0 = src0->దత్తాంశం;
    dma_addr_t దత్తాంశం_src1 = src1 ? src1->దత్తాంశం : 0;

    ఒకవేళ (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU)) {
         స్థిరరాశి int32_t swapped = octx->op_params[1];
         దత్తాంశం_src1 = దత్తాంశం_src0;
         actx.src1_అడ్డువరుస_కొలత = actx.src0_అడ్డువరుస_కొలత;

         size_t nc_లోపల_బైట్లు = actx.nc * SIZEOF_FP16;
         ఒకవేళ (swapped) {
             దత్తాంశం_src0 += nc_లోపల_బైట్లు;
         } లేకపోతే {
             దత్తాంశం_src1 += nc_లోపల_బైట్లు;
         }
    }

    actx.దత్తాంశం_src0 = దత్తాంశం_src0;
    actx.దత్తాంశం_src1 = దత్తాంశం_src1;
    actx.దత్తాంశం_dst  = dst->దత్తాంశం;

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, (కార్మికుడు_callback_t)glu_f16_చొప్పున_thread, &actx, n_దారాలు);
    తిరిగి HTP_స్థితి_OK;
}


పూర్ణాంకం op_activations(నిర్మాణం htp_ops_context * octx) {
    ఎంపిక (octx->src[0]->type) {
        స్థితి HTP_TYPE_F32:
            తిరిగి execute_op_activations_f32(octx);

        స్థితి HTP_TYPE_F16:
            తిరిగి execute_op_activations_f16(octx);

        అప్రమేయ:
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
}
