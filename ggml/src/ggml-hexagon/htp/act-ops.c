#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "hex-common.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"

#تعریف_کرو htp_act_preamble                                 \
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
                                     ثابت uint32_t عدد_صفاں_2,
                                     ثابت بنية htp_act_context * actx);

تعريفنوع فراغ (*glu_compute_f16_fn_t)(ثابت _Float16 * محدود src0,
                                     ثابت _Float16 * محدود src1,
                                     _Float16 * محدود dst,
                                     ثابت uint32_t عدد_صفاں_2,
                                     ثابت بنية htp_act_context * actx);

بنية htp_act_context {
    بنية htp_ops_context * octx;

    فراغ *                   compute;
    ثابت حرف *             op_str;

    // Precomputed values
    dma_addr_t               انکڑے_src0;
    dma_addr_t               انکڑے_src1;
    dma_addr_t               انکڑے_dst;

    size_t                   src0_صف_ناپ;
    size_t                   src1_صف_ناپ;
    size_t                   dst_صف_ناپ;

    size_t                   src0_صف_stride;
    size_t                   src1_صف_stride;

    size_t                   src0_صف_ناپ_aligned;
    size_t                   src1_صف_ناپ_aligned;
    size_t                   dst_صف_ناپ_aligned;

    size_t                   src0_spad_half_ناپ;
    size_t                   src1_spad_half_ناپ;
    size_t                   dst_spad_half_ناپ;

    uint32_t                 بلاک;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_فی_thread;
    uint32_t                 صف_شروع;
    صحيح                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_ناپ_فی_thread;
    size_t                   vtcm_src1_ناپ_فی_thread;
    size_t                   vtcm_dst_ناپ_فی_thread;
};

بنية htp_act_vtcm_layout {
    size_t کل_بائٹاں;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_بائٹاں_فی_thread;
    size_t src1_بائٹاں_فی_thread;
    size_t dst_بائٹاں_فی_thread;

    uint32_t vtcm_صف_فی_thread;
};

ساكن مضمن فراغ htp_act_vtcm_layout_build(بنية htp_act_vtcm_layout * L,
                                             size_t                       src0_صف_ناپ_aligned,
                                             size_t                       src1_صف_ناپ_aligned,
                                             size_t                       dst_صف_ناپ_aligned,
                                             uint32_t                     n_دھاگے,
                                             size_t                       vtcm_ناپ) {
    ثابت size_t   spad_ناپ_فی_صف   = src0_صف_ناپ_aligned + src1_صف_ناپ_aligned + dst_صف_ناپ_aligned;
    ثابت uint32_t vtcm_صف_فی_thread = (uint32_t) (vtcm_ناپ / (n_دھاگے * spad_ناپ_فی_صف));

    L->vtcm_صف_فی_thread = vtcm_صف_فی_thread;

    L->src0_بائٹاں_فی_thread = src0_صف_ناپ_aligned * vtcm_صف_فی_thread;
    L->src1_بائٹاں_فی_thread = src1_صف_ناپ_aligned * vtcm_صف_فی_thread;
    L->dst_بائٹاں_فی_thread  = dst_صف_ناپ_aligned * vtcm_صف_فی_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_بائٹاں_فی_thread * n_دھاگے;
    L->off_dst  = L->off_src1 + L->src1_بائٹاں_فی_thread * n_دھاگے;

    L->کل_بائٹاں = L->off_dst + L->dst_بائٹاں_فی_thread * n_دھاگے;
}

#تعریف_کرو htp_glu_op_preamble                                            \
    ثابت size_t src0_صف_ناپ_aligned = actx->src0_صف_ناپ_aligned;  \
    ثابت size_t src1_صف_ناپ_aligned = actx->src1_صف_ناپ_aligned;  \
    ثابت size_t dst_صف_ناپ_aligned  = actx->dst_صف_ناپ_aligned;   \
    ثابت صحيح    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
ساكن فراغ swiglu_f32(ثابت عائم * محدود src0,
                                                 ثابت عائم * محدود src1,
                                                 عائم * محدود dst,
                                                 ثابت uint32_t عدد_صفاں_2,
                                                 ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساكن فراغ swiglu_oai_f32(ثابت عائم * محدود src0,
                                                     ثابت عائم * محدود src1,
                                                     عائم * محدود dst,
                                                     ثابت uint32_t عدد_صفاں_2,
                                                     ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت عائم alpha = ((ثابت عائم *) (actx->octx->op_params))[2];
    ثابت عائم حد_بندی = ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد_بندی, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد_بندی, حد_بندی, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جوڑو_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
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
                                                       ثابت uint32_t                 عدد_صفاں_2,
                                                       ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت عائم حد_بندی = ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, حد_بندی, nc);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -حد_بندی, حد_بندی, nc);
        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ساكن ثابت عائم GELU_COEF_A     = 0.044715f;
ساكن ثابت عائم SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

ساكن مضمن HVX_سمتیہ hvx_vec_fast_sigmoid_f32_2it(HVX_سمتیہ v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_سمتیہ وچ_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_سمتیہ x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(وچ_int));
    HVX_سمتیہ xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_سمتیہ v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_سمتیہ v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, وچ_int, 24);

    HVX_سمتیہ v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_سمتیہ v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_سمتیہ two_sf = hvx_vec_splat_f32(2.0f);
    HVX_سمتیہ i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_سمتیہ r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_سمتیہ res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    أرجع Q6_Vsf_equals_Vqf32(res);
}

ساكن مضمن HVX_سمتیہ hvx_vec_fast_sigmoid_f32_guard_2it(HVX_سمتیہ v,
                                                            HVX_سمتیہ one,
                                                            HVX_سمتیہ max_exp,
                                                            HVX_سمتیہ min_exp) {
    ثابت HVX_سمتیہ_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ثابت HVX_سمتیہ_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_سمتیہ باہرلا = hvx_vec_fast_sigmoid_f32_2it(v);
    باہرلا            = Q6_V_vmux_QVV(pred_max, باہرلا, one);
    أرجع Q6_V_vmux_QVV(pred_min, باہرلا, Q6_V_vzero());
}

ساكن مضمن فراغ hvx_geglu_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بنانشانی لما) dst  % 128 == 0);
    أكد((بنانشانی لما) src0 % 128 == 0);
    أكد((بنانشانی لما) src1 % 128 == 0);

    HVX_سمتیہ * محدود vdst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود vsrc0 = (ثابت HVX_سمتیہ *) src0;
    ثابت HVX_سمتیہ * محدود vsrc1 = (ثابت HVX_سمتیہ *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت عائم GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    ثابت HVX_سمتیہ v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    ثابت HVX_سمتیہ v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    ثابت HVX_سمتیہ v_one               = hvx_vec_splat_f32(1.0f);
    ثابت HVX_سمتیہ v_max_exp           = hvx_vec_splat_f32(87.0f);
    ثابت HVX_سمتیہ v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];

        HVX_سمتیہ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_سمتیہ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جوڑو_f32_f32(coef, v_sqrt_2_pi);
        HVX_سمتیہ inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner = inner + inner
        HVX_سمتیہ y2 = hvx_vec_جوڑو_f32_f32(inner, inner);

        // Fast sigmoid approximation (2 iterations)
        HVX_سمتیہ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_سمتیہ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    إذا (nloe) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];

        HVX_سمتیہ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_سمتیہ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_جوڑو_f32_f32(coef, v_sqrt_2_pi);
        HVX_سمتیہ inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_سمتیہ y2 = hvx_vec_جوڑو_f32_f32(inner, inner);

        HVX_سمتیہ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_سمتیہ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        HVX_سمتیہ res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), res);
    }
}

ساكن مضمن فراغ hvx_geglu_تیز_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بنانشانی لما) dst  % 128 == 0);
    أكد((بنانشانی لما) src0 % 128 == 0);
    أكد((بنانشانی لما) src1 % 128 == 0);

    HVX_سمتیہ * محدود vdst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود vsrc0 = (ثابت HVX_سمتیہ *) src0;
    ثابت HVX_سمتیہ * محدود vsrc1 = (ثابت HVX_سمتیہ *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_سمتیہ v_scale    = hvx_vec_splat_f32(1.702f);
    ثابت HVX_سمتیہ v_one      = hvx_vec_splat_f32(1.0f);
    ثابت HVX_سمتیہ v_max_exp  = hvx_vec_splat_f32(87.0f);
    ثابت HVX_سمتیہ v_min_exp  = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];
        HVX_سمتیہ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_سمتیہ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
    }

    إذا (nloe) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];
        HVX_سمتیہ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_سمتیہ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        HVX_سمتیہ نتیجہ = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), نتیجہ);
    }
}

ساكن مضمن فراغ hvx_geglu_erf_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بنانشانی لما) dst  % 128 == 0);
    أكد((بنانشانی لما) src0 % 128 == 0);
    أكد((بنانشانی لما) src1 % 128 == 0);

    HVX_سمتیہ * محدود vdst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود vsrc0 = (ثابت HVX_سمتیہ *) src0;
    ثابت HVX_سمتیہ * محدود vsrc1 = (ثابت HVX_سمتیہ *) src1;

    ثابت uint32_t epv  = 128 / حجم(عائم);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
    }

    إذا (nloe) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];
        HVX_سمتیہ نتیجہ = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(عائم), نتیجہ);
    }
}

ساكن مضمن فراغ hvx_geglu_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src0, ثابت uint8_t * محدود src1, uint32_t n) {
    أكد((بنانشانی لما) dst  % 128 == 0);
    أكد((بنانشانی لما) src0 % 128 == 0);
    أكد((بنانشانی لما) src1 % 128 == 0);

    HVX_سمتیہ * محدود vdst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود vsrc0 = (ثابت HVX_سمتیہ *) src0;
    ثابت HVX_سمتیہ * محدود vsrc1 = (ثابت HVX_سمتیہ *) src1;

    ثابت uint32_t epv  = 128 / حجم(_Float16);
    ثابت uint32_t nvec = n / epv;
    ثابت uint32_t nloe = n % epv;

    ثابت HVX_سمتیہ v_coef_a    = hvx_vec_splat_f16((_Float16) GELU_COEF_A);
    ثابت HVX_سمتیہ v_sqrt_2_pi = hvx_vec_splat_f16((_Float16) SQRT_2_OVER_PI);
    ثابت HVX_سمتیہ v_one       = hvx_vec_splat_f16((_Float16) 1.0f);
    ثابت HVX_سمتیہ v_half      = hvx_vec_splat_f16((_Float16) 0.5f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];

        // gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))
        HVX_سمتیہ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_سمتیہ coef  = hvx_vec_جوڑو_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_سمتیہ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_سمتیہ t = hvx_vec_جوڑو_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_سمتیہ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        vdst[i] = hvx_vec_mul_f16_f16(gelu_x, g);
    }

    إذا (nloe) {
        HVX_سمتیہ x = vsrc0[i];
        HVX_سمتیہ g = vsrc1[i];

        HVX_سمتیہ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_سمتیہ coef  = hvx_vec_جوڑو_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_سمتیہ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_سمتیہ t = hvx_vec_جوڑو_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_سمتیہ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        HVX_سمتیہ res    = hvx_vec_mul_f16_f16(gelu_x, g);
        hvx_vec_store_a((فراغ *) &vdst[i], nloe * حجم(_Float16), res);
    }
}

// geglu(x, g) = gelu(x) * g
ساكن فراغ geglu_f32(ثابت عائم * محدود src0,
                                                ثابت عائم * محدود src1,
                                                عائم * محدود dst,
                                                ثابت uint32_t عدد_صفاں_2,
                                                ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_quick(x, g) = x * sigmoid(1.702 * x) * g
ساكن فراغ geglu_تیز_f32(ثابت عائم * محدود src0,
                            ثابت عائم * محدود src1,
                            عائم * محدود dst,
                            ثابت uint32_t عدد_صفاں_2,
                            ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        hvx_geglu_تیز_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_erf(x, g) = gelu_erf(x) * g
ساكن فراغ geglu_erf_f32(ثابت عائم * محدود src0,
                          ثابت عائم * محدود src1,
                          عائم * محدود dst,
                          ثابت uint32_t عدد_صفاں_2,
                          ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        hvx_geglu_erf_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// swiglu(x) = x1 * sigmoid(x0)
ساكن فراغ swiglu_f16(ثابت _Float16 * محدود src0,
                       ثابت _Float16 * محدود src1,
                       _Float16 * محدود dst,
                       ثابت uint32_t عدد_صفاں_2,
                       ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ساكن فراغ swiglu_oai_f16(ثابت _Float16 * محدود src0,
                           ثابت _Float16 * محدود src1,
                           _Float16 * محدود dst,
                           ثابت uint32_t عدد_صفاں_2,
                           ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 alpha = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[2];
    ثابت _Float16 حد_بندی = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, حد_بندی, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -حد_بندی, حد_بندی, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_جوڑو_scalar_f16((uint8_t *) src1_ptr, src1_ptr, (_Float16) 1.0f, nc);
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
                             ثابت uint32_t                 عدد_صفاں_2,
                             ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;
    ثابت _Float16 حد_بندی = (_Float16) ((ثابت عائم *) (actx->octx->op_params))[3];

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, حد_بندی, nc);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -حد_بندی, حد_بندی, nc);
        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// geglu(x, g) = gelu(x) * g, gelu via tanh approximation
ساكن فراغ geglu_f16(ثابت _Float16 * محدود src0,
                      ثابت _Float16 * محدود src1,
                      _Float16 * محدود dst,
                      ثابت uint32_t عدد_صفاں_2,
                      ثابت بنية htp_act_context * actx) {
    htp_glu_op_preamble;

    لكل (uint32_t ib = 0; ib < عدد_صفاں_2; ib++) {
        ثابت uint8_t * محدود src0_ptr = (ثابت uint8_t *) src0 + (ib * src0_صف_ناپ_aligned);
        ثابت uint8_t * محدود src1_ptr = (ثابت uint8_t *) src1 + (ib * src1_صف_ناپ_aligned);
        uint8_t * محدود dst_ptr        = (uint8_t *) dst + (ib * dst_صف_ناپ_aligned);

        hvx_geglu_f16_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#تعریف_کرو DEFINE_GLU_فی_THREAD(TYPE, SUFFIX)                                                                            \
    ساكن فراغ glu_##SUFFIX##_per_thread(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {                           \
        بنية htp_act_context * actx = (بنية htp_act_context *) انکڑے;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        بنية htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : عدم;                          \
                                                                                                                       \
        size_t src0_صف_ناپ = actx->src0_صف_ناپ;                                                                    \
        size_t src1_صف_ناپ = actx->src1_صف_ناپ;                                                                    \
        size_t dst_صف_ناپ  = actx->dst_صف_ناپ;                                                                     \
                                                                                                                       \
        size_t src0_صف_stride = actx->src0_صف_stride;                                                                \
        size_t src1_صف_stride = actx->src1_صف_stride;                                                                \
                                                                                                                       \
        ثابت uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        ثابت uint32_t src0_nrows_فی_thread = actx->src0_nrows_فی_thread;                                            \
                                                                                                                       \
        ثابت uint32_t src0_شروع_صف = actx->صف_شروع + src0_nrows_فی_thread * ith;                                 \
        ثابت uint32_t src0_سرا_صف   = MIN(src0_شروع_صف + src0_nrows_فی_thread, actx->صف_شروع + src0_nrows);     \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        إذا (src0_شروع_صف >= src0_سرا_صف) {                                                                          \
            أرجع;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        ثابت dma_addr_t انکڑے_src0 = actx->انکڑے_src0;                                                                  \
        ثابت dma_addr_t انکڑے_src1 = actx->انکڑے_src1;                                                                  \
        ثابت dma_addr_t انکڑے_dst  = actx->انکڑے_dst;                                                                   \
                                                                                                                       \
        ثابت size_t src0_صف_ناپ_aligned = actx->src0_صف_ناپ_aligned;                                              \
        ثابت size_t src1_صف_ناپ_aligned = actx->src1_صف_ناپ_aligned;                                              \
        ثابت size_t dst_صف_ناپ_aligned  = actx->dst_صف_ناپ_aligned;                                               \
                                                                                                                       \
        uint8_t * محدود src0_spad_انکڑے = actx->vtcm_src0 + (ith * actx->vtcm_src0_ناپ_فی_thread);                 \
        uint8_t * محدود src1_spad_انکڑے = actx->vtcm_src1 + (ith * actx->vtcm_src1_ناپ_فی_thread);                 \
        uint8_t * محدود dst_spad_انکڑے  = actx->vtcm_dst  + (ith * actx->vtcm_dst_ناپ_فی_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_ناپ = actx->src0_spad_half_ناپ;                                                        \
        size_t src1_spad_half_ناپ = actx->src1_spad_half_ناپ;                                                        \
        size_t dst_spad_half_ناپ  = actx->dst_spad_half_ناپ;                                                         \
                                                                                                                       \
        ثابت صحيح بلاک_2 = actx->بلاک;                                                                                 \
        إذا (بلاک_2 == 0) {                                                                                              \
            FARF(ERROR, "%s : VTCM reservation %zu is too small, needed %zu\n",                                        \
                 actx->op_str, actx->vtcm_src0_ناپ_فی_thread, src0_صف_ناپ_aligned);                                \
            أرجع;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_قطار * dma_q = actx->octx->ctx->dma[ith];                                                                 \
        glu_compute_##SUFFIX##_fn_t compute = (glu_compute_##SUFFIX##_fn_t) actx->compute;                             \
                                                                                                                       \
        لكل (uint32_t ir = src0_شروع_صف, spad_idx = 0; ir < src0_سرا_صف && spad_idx < 2; ir += بلاک_2, spad_idx++) { \
            ثابت uint32_t بلاک_ناپ_3 = MIN(بلاک_2, src0_سرا_صف - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_قطار_ڈالو(dma_q,                                                                                      \
                           dma_make_انکڑے(انکڑے_dst, dst_spad_انکڑے + (spad_idx * dst_spad_half_ناپ)),                   \
                           dst_صف_ناپ, dst_صف_ناپ_aligned, dst_صف_ناپ, 0);                                       \
                                                                                                                       \
            dma_قطار_ڈالو(dma_q,                                                                                      \
                dma_make_انکڑے(src0_spad_انکڑے + (spad_idx * src0_spad_half_ناپ), انکڑے_src0 + (ir * src0_صف_stride)),  \
                src0_صف_ناپ_aligned, src0_صف_stride, src0_صف_ناپ, بلاک_ناپ_3);                                    \
                                                                                                                       \
            dma_قطار_ڈالو(dma_q,                                                                                      \
                dma_make_انکڑے(src1_spad_انکڑے + (spad_idx * src1_spad_half_ناپ), انکڑے_src1 + (ir * src1_صف_stride)),  \
                src1_صف_ناپ_aligned, src1_صف_stride, src1_صف_ناپ, بلاک_ناپ_3);                                    \
        }                                                                                                              \
                                                                                                                       \
        لكل (uint32_t ir = src0_شروع_صف; ir < src0_سرا_صف; ir += بلاک_2) {                                           \
            ثابت uint32_t بلاک_ناپ_3 = MIN(بلاک_2, src0_سرا_صف - ir);                                                 \
                                                                                                                       \
            TYPE * dst_spad  = (TYPE *) dma_قطار_نکالو(dma_q).src;                                                      \
            TYPE * src0_spad = (TYPE *) dma_قطار_نکالو(dma_q).dst;                                                      \
            TYPE * src1_spad = (TYPE *) dma_قطار_نکالو(dma_q).dst;                                                      \
                                                                                                                       \
            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            compute(src0_spad, src1_spad, dst_spad, بلاک_ناپ_3, actx);                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(انکڑے_dst + (ir * dst_صف_ناپ), dst_spad),                             \
                           dst_صف_ناپ, dst_صف_ناپ_aligned, dst_صف_ناپ, بلاک_ناپ_3);                              \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            ثابت uint32_t pref_بلاک = (ir + بلاک_2 * 2);                                                              \
            إذا (pref_بلاک < src0_سرا_صف) {                                                                           \
                ثابت uint32_t pref_بلاک_ناپ = MIN(بلاک_2, src0_سرا_صف - pref_بلاک);                                \
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(src0_spad, انکڑے_src0 + (pref_بلاک * src0_صف_stride)),            \
                               src0_صف_ناپ_aligned, src0_صف_stride, src0_صف_ناپ, pref_بلاک_ناپ);                \
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(src1_spad, انکڑے_src1 + (pref_بلاک * src1_صف_stride)),            \
                               src1_صف_ناپ_aligned, src1_صف_stride, src1_صف_ناپ, pref_بلاک_ناپ);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_قطار_flush(dma_q);                                                                                        \
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

        حالة HTP_OP_GLU_GEGLU_تیز:
            compute_fn = geglu_تیز_f32;
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
    ثابت size_t dst_صف_ناپ = dst->ne[0] * SIZEOF_FP32;

    uint32_t صف_شروع = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفاں_فی_chunk = 0;
        htp_tensor_mdev_صفاں_فی_chunk(dst, حجم(عائم), (uint32_t) dst_صف_ناپ, &صفاں_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(src0_nrows, صفاں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        nrows     = range.گنتی;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بائٹاں        = dst_صف_ناپ;
    ثابت size_t src0_صف_ناپ   = nc_بائٹاں;
    ثابت size_t src1_صف_ناپ   = nc_بائٹاں;
    ثابت size_t src0_صف_stride = src0->nb[1];
    ثابت size_t src1_صف_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_صف_ناپ_aligned = hex_گیڑا_اتے(src0_صف_ناپ, VLEN);
    ثابت size_t src1_صف_ناپ_aligned = hex_گیڑا_اتے(src1_صف_ناپ, VLEN);
    ثابت size_t dst_صف_ناپ_aligned  = hex_گیڑا_اتے(dst_صف_ناپ, VLEN);

    بنية htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_صف_ناپ_aligned, src1_صف_ناپ_aligned, dst_صف_ناپ_aligned, n_دھاگے,
                              octx->ctx->vtcm_ناپ);

    // Make sure the reserved vtcm size is sufficient
    إذا (L.vtcm_صف_فی_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_ناپ,
             (src0_صف_ناپ_aligned + src1_صف_ناپ_aligned + dst_صف_ناپ_aligned) * n_دھاگے);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    إذا (src1) {
        FARF(بلند_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بائٹاں_فی_thread * n_دھاگے,
             L.src1_بائٹاں_فی_thread * n_دھاگے, L.dst_بائٹاں_فی_thread * n_دھاگے);
    } وإلا {
        FARF(بلند_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بائٹاں_فی_thread * n_دھاگے, L.src1_بائٹاں_فی_thread * n_دھاگے, L.dst_بائٹاں_فی_thread * n_دھاگے);
    }

    // Prepare context
    بنية htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (فراغ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);

    actx.src0_صف_ناپ = src0_صف_ناپ;
    actx.src1_صف_ناپ = src1_صف_ناپ;
    actx.dst_صف_ناپ  = dst_صف_ناپ;

    actx.src0_صف_ناپ_aligned = src0_صف_ناپ_aligned;
    actx.src1_صف_ناپ_aligned = src1_صف_ناپ_aligned;
    actx.dst_صف_ناپ_aligned  = dst_صف_ناپ_aligned;

    actx.src0_صف_stride = src0_صف_stride;
    actx.src1_صف_stride = src1_صف_stride;

    uint8_t * ثابت ادھار = (uint8_t *) octx->ctx->vtcm_ادھار;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_dst);

    actx.vtcm_src0_ناپ_فی_thread = L.src0_بائٹاں_فی_thread;
    actx.vtcm_src1_ناپ_فی_thread = L.src1_بائٹاں_فی_thread;
    actx.vtcm_dst_ناپ_فی_thread  = L.dst_بائٹاں_فی_thread;

    actx.src0_spad_half_ناپ = L.src0_بائٹاں_فی_thread / 2;
    actx.src1_spad_half_ناپ = L.src1_بائٹاں_فی_thread / 2;
    actx.dst_spad_half_ناپ  = L.dst_بائٹاں_فی_thread / 2;

    actx.بلاک = actx.src0_spad_half_ناپ / actx.src0_صف_ناپ_aligned;
    actx.src0_nrows = nrows;
    actx.صف_شروع  = صف_شروع;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t انکڑے_src0 = src0->انکڑے;
    dma_addr_t انکڑے_src1 = src1 ? src1->انکڑے : 0;

    إذا (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU ||
                  octx->op == HTP_OP_GLU_GEGLU_تیز ||
                  octx->op == HTP_OP_GLU_GEGLU_ERF)) {
         ثابت int32_t swapped = octx->op_params[1];
         انکڑے_src1 = انکڑے_src0;
         actx.src1_صف_ناپ = actx.src0_صف_ناپ;

         size_t nc_وچ_بائٹاں = actx.nc * SIZEOF_FP32;
         إذا (swapped) {
             انکڑے_src0 += nc_وچ_بائٹاں;
         } وإلا {
             انکڑے_src1 += nc_وچ_بائٹاں;
         }
    }

    actx.انکڑے_src0 = انکڑے_src0;
    actx.انکڑے_src1 = انکڑے_src1;
    actx.انکڑے_dst  = dst->انکڑے;

    work_قطار_لگاتار(octx->ctx->work_قطار, (کامہ_callback_t)glu_f32_فی_thread, &actx, n_دھاگے);
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
    ثابت size_t dst_صف_ناپ = dst->ne[0] * SIZEOF_FP16;

    uint32_t صف_شروع = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفاں_فی_chunk = 0;
        htp_tensor_mdev_صفاں_فی_chunk(dst, حجم(_Float16), (uint32_t) dst_صف_ناپ, &صفاں_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(src0_nrows, صفاں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        nrows     = range.گنتی;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ثابت size_t nc_بائٹاں        = dst_صف_ناپ;
    ثابت size_t src0_صف_ناپ   = nc_بائٹاں;
    ثابت size_t src1_صف_ناپ   = nc_بائٹاں;
    ثابت size_t src0_صف_stride = src0->nb[1];
    ثابت size_t src1_صف_stride = src1 ? src1->nb[1] : src0->nb[1];

    ثابت size_t src0_صف_ناپ_aligned = hex_گیڑا_اتے(src0_صف_ناپ, VLEN);
    ثابت size_t src1_صف_ناپ_aligned = hex_گیڑا_اتے(src1_صف_ناپ, VLEN);
    ثابت size_t dst_صف_ناپ_aligned  = hex_گیڑا_اتے(dst_صف_ناپ, VLEN);

    بنية htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_صف_ناپ_aligned, src1_صف_ناپ_aligned, dst_صف_ناپ_aligned, n_دھاگے,
                              octx->ctx->vtcm_ناپ);

    // Make sure the reserved vtcm size is sufficient
    إذا (L.vtcm_صف_فی_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_ناپ,
             (src0_صف_ناپ_aligned + src1_صف_ناپ_aligned + dst_صف_ناپ_aligned) * n_دھاگے);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    إذا (src1) {
        FARF(بلند_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_بائٹاں_فی_thread * n_دھاگے,
             L.src1_بائٹاں_فی_thread * n_دھاگے, L.dst_بائٹاں_فی_thread * n_دھاگے);
    } وإلا {
        FARF(بلند_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_بائٹاں_فی_thread * n_دھاگے, L.src1_بائٹاں_فی_thread * n_دھاگے, L.dst_بائٹاں_فی_thread * n_دھاگے);
    }

    // Prepare context
    بنية htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (فراغ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);

    actx.src0_صف_ناپ = src0_صف_ناپ;
    actx.src1_صف_ناپ = src1_صف_ناپ;
    actx.dst_صف_ناپ  = dst_صف_ناپ;

    actx.src0_صف_ناپ_aligned = src0_صف_ناپ_aligned;
    actx.src1_صف_ناپ_aligned = src1_صف_ناپ_aligned;
    actx.dst_صف_ناپ_aligned  = dst_صف_ناپ_aligned;

    actx.src0_صف_stride = src0_صف_stride;
    actx.src1_صف_stride = src1_صف_stride;

    uint8_t * ثابت ادھار = (uint8_t *) octx->ctx->vtcm_ادھار;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, ادھار, L.off_dst);

    actx.vtcm_src0_ناپ_فی_thread = L.src0_بائٹاں_فی_thread;
    actx.vtcm_src1_ناپ_فی_thread = L.src1_بائٹاں_فی_thread;
    actx.vtcm_dst_ناپ_فی_thread  = L.dst_بائٹاں_فی_thread;

    actx.src0_spad_half_ناپ = L.src0_بائٹاں_فی_thread / 2;
    actx.src1_spad_half_ناپ = L.src1_بائٹاں_فی_thread / 2;
    actx.dst_spad_half_ناپ  = L.dst_بائٹاں_فی_thread / 2;

    actx.بلاک = actx.src0_spad_half_ناپ / actx.src0_صف_ناپ_aligned;
    actx.src0_nrows = nrows;
    actx.صف_شروع  = صف_شروع;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t انکڑے_src0 = src0->انکڑے;
    dma_addr_t انکڑے_src1 = src1 ? src1->انکڑے : 0;

    إذا (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU)) {
         ثابت int32_t swapped = octx->op_params[1];
         انکڑے_src1 = انکڑے_src0;
         actx.src1_صف_ناپ = actx.src0_صف_ناپ;

         size_t nc_وچ_بائٹاں = actx.nc * SIZEOF_FP16;
         إذا (swapped) {
             انکڑے_src0 += nc_وچ_بائٹاں;
         } وإلا {
             انکڑے_src1 += nc_وچ_بائٹاں;
         }
    }

    actx.انکڑے_src0 = انکڑے_src0;
    actx.انکڑے_src1 = انکڑے_src1;
    actx.انکڑے_dst  = dst->انکڑے;

    work_قطار_لگاتار(octx->ctx->work_قطار, (کامہ_callback_t)glu_f16_فی_thread, &actx, n_دھاگے);
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
