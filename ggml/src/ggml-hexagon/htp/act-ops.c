#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_act_preamble                                 \
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = actx->octx->src[0]; \
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = actx->octx->src[1]; \
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = actx->octx->dst;    \
                                                         \
    ਅਚਲ uint32_t ne00 = src0->ne[0];                   \
    ਅਚਲ uint32_t ne01 = src0->ne[1];                   \
    ਅਚਲ uint32_t ne02 = src0->ne[2];                   \
    ਅਚਲ uint32_t ne03 = src0->ne[3];                   \
                                                         \
    ਅਚਲ uint32_t nb00 = src0->nb[0];                   \
    ਅਚਲ uint32_t nb01 = src0->nb[1];                   \
    ਅਚਲ uint32_t nb02 = src0->nb[2];                   \
    ਅਚਲ uint32_t nb03 = src0->nb[3];                   \
                                                         \
    ਅਚਲ uint32_t ne10 = src1 ? src1->ne[0] : 0;        \
    ਅਚਲ uint32_t ne11 = src1 ? src1->ne[1] : 0;        \
    ਅਚਲ uint32_t ne12 = src1 ? src1->ne[2] : 0;        \
    ਅਚਲ uint32_t ne13 = src1 ? src1->ne[3] : 0;        \
                                                         \
    ਅਚਲ uint32_t nb10 = src1 ? src1->nb[0] : 0;        \
    ਅਚਲ uint32_t nb11 = src1 ? src1->nb[1] : 0;        \
    ਅਚਲ uint32_t nb12 = src1 ? src1->nb[2] : 0;        \
    ਅਚਲ uint32_t nb13 = src1 ? src1->nb[3] : 0;        \
                                                         \
    ਅਚਲ uint32_t ne0 = dst->ne[0];                     \
    ਅਚਲ uint32_t ne1 = dst->ne[1];                     \
    ਅਚਲ uint32_t ne2 = dst->ne[2];                     \
    ਅਚਲ uint32_t ne3 = dst->ne[3];                     \
                                                         \
    ਅਚਲ uint32_t nb0 = dst->nb[0];                     \
    ਅਚਲ uint32_t nb1 = dst->nb[1];                     \
    ਅਚਲ uint32_t nb2 = dst->nb[2];                     \
    ਅਚਲ uint32_t nb3 = dst->nb[3];

ਬਣਤਰ htp_act_context;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*glu_compute_f32_fn_t)(ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src0,
                                     ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src1,
                                     ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst,
                                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                                     ਅਚਲ ਬਣਤਰ htp_act_context * actx);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*glu_compute_f16_fn_t)(ਅਚਲ _Float16 * ਸੀਮਿਤ src0,
                                     ਅਚਲ _Float16 * ਸੀਮਿਤ src1,
                                     _Float16 * ਸੀਮਿਤ dst,
                                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                                     ਅਚਲ ਬਣਤਰ htp_act_context * actx);

ਬਣਤਰ htp_act_context {
    ਬਣਤਰ htp_ops_context * octx;

    ਖਾਲੀ *                   compute;
    ਅਚਲ ਅੱਖਰ *             op_str;

    // Precomputed values
    dma_addr_t               ਅੰਕੜੇ_src0;
    dma_addr_t               ਅੰਕੜੇ_src1;
    dma_addr_t               ਅੰਕੜੇ_dst;

    size_t                   src0_ਪੰਕਤੀ_ਮਾਪ;
    size_t                   src1_ਪੰਕਤੀ_ਮਾਪ;
    size_t                   dst_ਪੰਕਤੀ_ਮਾਪ;

    size_t                   src0_ਪੰਕਤੀ_stride;
    size_t                   src1_ਪੰਕਤੀ_stride;

    size_t                   src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t                   src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t                   dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    size_t                   src0_spad_half_ਮਾਪ;
    size_t                   src1_spad_half_ਮਾਪ;
    size_t                   dst_spad_half_ਮਾਪ;

    uint32_t                 ਖੰਡ;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_ਪ੍ਰਤੀ_thread;
    uint32_t                 ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ਪੂਰਨ                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    size_t                   vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    size_t                   vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;
};

ਬਣਤਰ htp_act_vtcm_layout {
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    uint32_t vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_act_vtcm_layout_build(ਬਣਤਰ htp_act_vtcm_layout * L,
                                             size_t                       src0_ਪੰਕਤੀ_ਮਾਪ_aligned,
                                             size_t                       src1_ਪੰਕਤੀ_ਮਾਪ_aligned,
                                             size_t                       dst_ਪੰਕਤੀ_ਮਾਪ_aligned,
                                             uint32_t                     n_ਧਾਗੇ,
                                             size_t                       vtcm_ਮਾਪ) {
    ਅਚਲ size_t   spad_ਮਾਪ_ਪ੍ਰਤੀ_ਪੰਕਤੀ   = src0_ਪੰਕਤੀ_ਮਾਪ_aligned + src1_ਪੰਕਤੀ_ਮਾਪ_aligned + dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
    ਅਚਲ uint32_t vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread = (uint32_t) (vtcm_ਮਾਪ / (n_ਧਾਗੇ * spad_ਮਾਪ_ਪ੍ਰਤੀ_ਪੰਕਤੀ));

    L->vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread = vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread;

    L->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = src0_ਪੰਕਤੀ_ਮਾਪ_aligned * vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread;
    L->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = src1_ਪੰਕਤੀ_ਮਾਪ_aligned * vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread;
    L->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned * vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
    L->off_dst  = L->off_src1 + L->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;

    L->ਕੁੱਲ_ਬਾਈਟਾਂ = L->off_dst + L->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_glu_op_preamble                                            \
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = actx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned;  \
    ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = actx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned;  \
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = actx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;   \
    ਅਚਲ ਪੂਰਨ    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
ਸਥਿਰ ਖਾਲੀ swiglu_f32(ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src0,
                                                 ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src1,
                                                 ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst,
                                                 ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                                                 ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ਸਥਿਰ ਖਾਲੀ swiglu_oai_f32(ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src0,
                                                     ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src1,
                                                     ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst,
                                                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                                                     ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = ((ਅਚਲ ਦਸ਼ਮਲਵ *) (actx->octx->op_params))[2];
    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = ((ਅਚਲ ਦਸ਼ਮਲਵ *) (actx->octx->op_params))[3];

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, ਹੱਦਬੰਦੀ, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_ਜੋੜੋ_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f32(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f32_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ਸਥਿਰ ਖਾਲੀ swiglu_clamp_f32(ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src0,
                                                       ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src1,
                                                       ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst,
                                                       ਅਚਲ uint32_t                 ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                                                       ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;
    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = ((ਅਚਲ ਦਸ਼ਮਲਵ *) (actx->octx->op_params))[3];

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, ਹੱਦਬੰਦੀ, nc);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ, nc);
        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ GELU_COEF_A     = 0.044715f;
ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_fast_sigmoid_f32_2it(HVX_ਸਦਿਸ਼ v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_ਸਦਿਸ਼ ਵਿੱਚ_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_ਸਦਿਸ਼ x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(ਵਿੱਚ_int));
    HVX_ਸਦਿਸ਼ xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_ਸਦਿਸ਼ v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_ਸਦਿਸ਼ v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, ਵਿੱਚ_int, 24);

    HVX_ਸਦਿਸ਼ v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_ਸਦਿਸ਼ v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_ਸਦਿਸ਼ two_sf = hvx_vec_splat_f32(2.0f);
    HVX_ਸਦਿਸ਼ i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_ਸਦਿਸ਼ r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_ਸਦਿਸ਼ res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    ਵਾਪਸ Q6_Vsf_equals_Vqf32(res);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_vec_fast_sigmoid_f32_guard_2it(HVX_ਸਦਿਸ਼ v,
                                                            HVX_ਸਦਿਸ਼ one,
                                                            HVX_ਸਦਿਸ਼ max_exp,
                                                            HVX_ਸਦਿਸ਼ min_exp) {
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ = hvx_vec_fast_sigmoid_f32_2it(v);
    ਬਾਹਰਲਾ            = Q6_V_vmux_QVV(pred_max, ਬਾਹਰਲਾ, one);
    ਵਾਪਸ Q6_V_vmux_QVV(pred_min, ਬਾਹਰਲਾ, Q6_V_vzero());
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_geglu_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst  % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src0 % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src1 % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst        = (HVX_ਸਦਿਸ਼ *) dst;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src1;

    ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;

    ਅਚਲ ਦਸ਼ਮਲਵ GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    ਅਚਲ HVX_ਸਦਿਸ਼ v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_one               = hvx_vec_splat_f32(1.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_max_exp           = hvx_vec_splat_f32(87.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ x = vsrc0[i];
        HVX_ਸਦਿਸ਼ g = vsrc1[i];

        HVX_ਸਦਿਸ਼ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_ਸਦਿਸ਼ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_ਜੋੜੋ_f32_f32(coef, v_sqrt_2_pi);
        HVX_ਸਦਿਸ਼ inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner = inner + inner
        HVX_ਸਦਿਸ਼ y2 = hvx_vec_ਜੋੜੋ_f32_f32(inner, inner);

        // Fast sigmoid approximation (2 iterations)
        HVX_ਸਦਿਸ਼ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_ਸਦਿਸ਼ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ x = vsrc0[i];
        HVX_ਸਦਿਸ਼ g = vsrc1[i];

        HVX_ਸਦਿਸ਼ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_ਸਦਿਸ਼ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_ਜੋੜੋ_f32_f32(coef, v_sqrt_2_pi);
        HVX_ਸਦਿਸ਼ inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_ਸਦਿਸ਼ y2 = hvx_vec_ਜੋੜੋ_f32_f32(inner, inner);

        HVX_ਸਦਿਸ਼ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_ਸਦਿਸ਼ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        HVX_ਸਦਿਸ਼ res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), res);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_geglu_ਤੇਜ਼_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst  % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src0 % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src1 % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst        = (HVX_ਸਦਿਸ਼ *) dst;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src1;

    ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;

    ਅਚਲ HVX_ਸਦਿਸ਼ v_scale    = hvx_vec_splat_f32(1.702f);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_one      = hvx_vec_splat_f32(1.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_max_exp  = hvx_vec_splat_f32(87.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_min_exp  = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ x = vsrc0[i];
        HVX_ਸਦਿਸ਼ g = vsrc1[i];
        HVX_ਸਦਿਸ਼ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_ਸਦਿਸ਼ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ x = vsrc0[i];
        HVX_ਸਦਿਸ਼ g = vsrc1[i];
        HVX_ਸਦਿਸ਼ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_ਸਦਿਸ਼ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਨਤੀਜਾ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_geglu_erf_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst  % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src0 % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src1 % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst        = (HVX_ਸਦਿਸ਼ *) dst;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src1;

    ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ x = vsrc0[i];
        HVX_ਸਦਿਸ਼ g = vsrc1[i];
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ x = vsrc0[i];
        HVX_ਸਦਿਸ਼ g = vsrc1[i];
        HVX_ਸਦਿਸ਼ ਨਤੀਜਾ = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਨਤੀਜਾ);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_geglu_f16_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src0, ਅਚਲ uint8_t * ਸੀਮਿਤ src1, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst  % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src0 % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src1 % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst        = (HVX_ਸਦਿਸ਼ *) dst;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src0;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src1;

    ਅਚਲ uint32_t epv  = 128 / ਆਕਾਰ(_Float16);
    ਅਚਲ uint32_t nvec = n / epv;
    ਅਚਲ uint32_t nloe = n % epv;

    ਅਚਲ HVX_ਸਦਿਸ਼ v_coef_a    = hvx_vec_splat_f16((_Float16) GELU_COEF_A);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_sqrt_2_pi = hvx_vec_splat_f16((_Float16) SQRT_2_OVER_PI);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_one       = hvx_vec_splat_f16((_Float16) 1.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ v_half      = hvx_vec_splat_f16((_Float16) 0.5f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ x = vsrc0[i];
        HVX_ਸਦਿਸ਼ g = vsrc1[i];

        // gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))
        HVX_ਸਦਿਸ਼ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_ਸਦਿਸ਼ coef  = hvx_vec_ਜੋੜੋ_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_ਸਦਿਸ਼ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_ਸਦਿਸ਼ t = hvx_vec_ਜੋੜੋ_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_ਸਦਿਸ਼ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        vdst[i] = hvx_vec_mul_f16_f16(gelu_x, g);
    }

    ਜੇ (nloe) {
        HVX_ਸਦਿਸ਼ x = vsrc0[i];
        HVX_ਸਦਿਸ਼ g = vsrc1[i];

        HVX_ਸਦਿਸ਼ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_ਸਦਿਸ਼ coef  = hvx_vec_ਜੋੜੋ_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_ਸਦਿਸ਼ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_ਸਦਿਸ਼ t = hvx_vec_ਜੋੜੋ_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_ਸਦਿਸ਼ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        HVX_ਸਦਿਸ਼ res    = hvx_vec_mul_f16_f16(gelu_x, g);
        hvx_vec_store_a((ਖਾਲੀ *) &vdst[i], nloe * ਆਕਾਰ(_Float16), res);
    }
}

// geglu(x, g) = gelu(x) * g
ਸਥਿਰ ਖਾਲੀ geglu_f32(ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src0,
                                                ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src1,
                                                ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst,
                                                ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                                                ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_quick(x, g) = x * sigmoid(1.702 * x) * g
ਸਥਿਰ ਖਾਲੀ geglu_ਤੇਜ਼_f32(ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src0,
                            ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src1,
                            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst,
                            ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                            ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_geglu_ਤੇਜ਼_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_erf(x, g) = gelu_erf(x) * g
ਸਥਿਰ ਖਾਲੀ geglu_erf_f32(ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src0,
                          ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src1,
                          ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst,
                          ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                          ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_geglu_erf_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// swiglu(x) = x1 * sigmoid(x0)
ਸਥਿਰ ਖਾਲੀ swiglu_f16(ਅਚਲ _Float16 * ਸੀਮਿਤ src0,
                       ਅਚਲ _Float16 * ਸੀਮਿਤ src1,
                       _Float16 * ਸੀਮਿਤ dst,
                       ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                       ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ਸਥਿਰ ਖਾਲੀ swiglu_oai_f16(ਅਚਲ _Float16 * ਸੀਮਿਤ src0,
                           ਅਚਲ _Float16 * ਸੀਮਿਤ src1,
                           _Float16 * ਸੀਮਿਤ dst,
                           ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                           ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;
    ਅਚਲ _Float16 alpha = (_Float16) ((ਅਚਲ ਦਸ਼ਮਲਵ *) (actx->octx->op_params))[2];
    ਅਚਲ _Float16 ਹੱਦਬੰਦੀ = (_Float16) ((ਅਚਲ ਦਸ਼ਮਲਵ *) (actx->octx->op_params))[3];

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, ਹੱਦਬੰਦੀ, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_ਜੋੜੋ_scalar_f16((uint8_t *) src1_ptr, src1_ptr, (_Float16) 1.0f, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f16(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f16_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ਸਥਿਰ ਖਾਲੀ swiglu_clamp_f16(ਅਚਲ _Float16 * ਸੀਮਿਤ src0,
                             ਅਚਲ _Float16 * ਸੀਮਿਤ src1,
                             _Float16 * ਸੀਮਿਤ dst,
                             ਅਚਲ uint32_t                 ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                             ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;
    ਅਚਲ _Float16 ਹੱਦਬੰਦੀ = (_Float16) ((ਅਚਲ ਦਸ਼ਮਲਵ *) (actx->octx->op_params))[3];

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, ਹੱਦਬੰਦੀ, nc);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ, nc);
        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// geglu(x, g) = gelu(x) * g, gelu via tanh approximation
ਸਥਿਰ ਖਾਲੀ geglu_f16(ਅਚਲ _Float16 * ਸੀਮਿਤ src0,
                      ਅਚਲ _Float16 * ਸੀਮਿਤ src1,
                      _Float16 * ਸੀਮਿਤ dst,
                      ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                      ਅਚਲ ਬਣਤਰ htp_act_context * actx) {
    htp_glu_op_preamble;

    ਲਈ (uint32_t ib = 0; ib < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ib++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src0_ptr = (ਅਚਲ uint8_t *) src0 + (ib * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ src1_ptr = (ਅਚਲ uint8_t *) src1 + (ib * src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ptr        = (uint8_t *) dst + (ib * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_geglu_f16_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_GLU_ਪ੍ਰਤੀ_THREAD(TYPE, SUFFIX)                                                                            \
    ਸਥਿਰ ਖਾਲੀ glu_##SUFFIX##_per_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {                           \
        ਬਣਤਰ htp_act_context * actx = (ਬਣਤਰ htp_act_context *) ਅੰਕੜੇ;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        ਬਣਤਰ htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : ਨਿਰਾ;                          \
                                                                                                                       \
        size_t src0_ਪੰਕਤੀ_ਮਾਪ = actx->src0_ਪੰਕਤੀ_ਮਾਪ;                                                                    \
        size_t src1_ਪੰਕਤੀ_ਮਾਪ = actx->src1_ਪੰਕਤੀ_ਮਾਪ;                                                                    \
        size_t dst_ਪੰਕਤੀ_ਮਾਪ  = actx->dst_ਪੰਕਤੀ_ਮਾਪ;                                                                     \
                                                                                                                       \
        size_t src0_ਪੰਕਤੀ_stride = actx->src0_ਪੰਕਤੀ_stride;                                                                \
        size_t src1_ਪੰਕਤੀ_stride = actx->src1_ਪੰਕਤੀ_stride;                                                                \
                                                                                                                       \
        ਅਚਲ uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        ਅਚਲ uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = actx->src0_nrows_ਪ੍ਰਤੀ_thread;                                            \
                                                                                                                       \
        ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = actx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;                                 \
        ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, actx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows);     \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {                                                                          \
            ਵਾਪਸ;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        ਅਚਲ dma_addr_t ਅੰਕੜੇ_src0 = actx->ਅੰਕੜੇ_src0;                                                                  \
        ਅਚਲ dma_addr_t ਅੰਕੜੇ_src1 = actx->ਅੰਕੜੇ_src1;                                                                  \
        ਅਚਲ dma_addr_t ਅੰਕੜੇ_dst  = actx->ਅੰਕੜੇ_dst;                                                                   \
                                                                                                                       \
        ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = actx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned;                                              \
        ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = actx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned;                                              \
        ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = actx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;                                               \
                                                                                                                       \
        uint8_t * ਸੀਮਿਤ src0_spad_ਅੰਕੜੇ = actx->vtcm_src0 + (ith * actx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread);                 \
        uint8_t * ਸੀਮਿਤ src1_spad_ਅੰਕੜੇ = actx->vtcm_src1 + (ith * actx->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread);                 \
        uint8_t * ਸੀਮਿਤ dst_spad_ਅੰਕੜੇ  = actx->vtcm_dst  + (ith * actx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_ਮਾਪ = actx->src0_spad_half_ਮਾਪ;                                                        \
        size_t src1_spad_half_ਮਾਪ = actx->src1_spad_half_ਮਾਪ;                                                        \
        size_t dst_spad_half_ਮਾਪ  = actx->dst_spad_half_ਮਾਪ;                                                         \
                                                                                                                       \
        ਅਚਲ ਪੂਰਨ ਖੰਡ_2 = actx->ਖੰਡ;                                                                                 \
        ਜੇ (ਖੰਡ_2 == 0) {                                                                                              \
            FARF(ERROR, "%s : VTCM reservation %zu is too small, needed %zu\n",                                        \
                 actx->op_str, actx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread, src0_ਪੰਕਤੀ_ਮਾਪ_aligned);                                \
            ਵਾਪਸ;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_ਕਤਾਰ * dma_q = actx->octx->ctx->dma[ith];                                                                 \
        glu_compute_##SUFFIX##_fn_t compute = (glu_compute_##SUFFIX##_fn_t) actx->compute;                             \
                                                                                                                       \
        ਲਈ (uint32_t ir = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, spad_idx = 0; ir < src0_ਸਿਰਾ_ਪੰਕਤੀ && spad_idx < 2; ir += ਖੰਡ_2, spad_idx++) { \
            ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = MIN(ਖੰਡ_2, src0_ਸਿਰਾ_ਪੰਕਤੀ - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                                      \
                           dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst, dst_spad_ਅੰਕੜੇ + (spad_idx * dst_spad_half_ਮਾਪ)),                   \
                           dst_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 0);                                       \
                                                                                                                       \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                                      \
                dma_make_ਅੰਕੜੇ(src0_spad_ਅੰਕੜੇ + (spad_idx * src0_spad_half_ਮਾਪ), ਅੰਕੜੇ_src0 + (ir * src0_ਪੰਕਤੀ_stride)),  \
                src0_ਪੰਕਤੀ_ਮਾਪ_aligned, src0_ਪੰਕਤੀ_stride, src0_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);                                    \
                                                                                                                       \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                                      \
                dma_make_ਅੰਕੜੇ(src1_spad_ਅੰਕੜੇ + (spad_idx * src1_spad_half_ਮਾਪ), ਅੰਕੜੇ_src1 + (ir * src1_ਪੰਕਤੀ_stride)),  \
                src1_ਪੰਕਤੀ_ਮਾਪ_aligned, src1_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);                                    \
        }                                                                                                              \
                                                                                                                       \
        ਲਈ (uint32_t ir = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < src0_ਸਿਰਾ_ਪੰਕਤੀ; ir += ਖੰਡ_2) {                                           \
            ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = MIN(ਖੰਡ_2, src0_ਸਿਰਾ_ਪੰਕਤੀ - ir);                                                 \
                                                                                                                       \
            TYPE * dst_spad  = (TYPE *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;                                                      \
            TYPE * src0_spad = (TYPE *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;                                                      \
            TYPE * src1_spad = (TYPE *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;                                                      \
                                                                                                                       \
            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            compute(src0_spad, src1_spad, dst_spad, ਖੰਡ_ਮਾਪ_3, actx);                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ), dst_spad),                             \
                           dst_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);                              \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            ਅਚਲ uint32_t pref_ਖੰਡ = (ir + ਖੰਡ_2 * 2);                                                              \
            ਜੇ (pref_ਖੰਡ < src0_ਸਿਰਾ_ਪੰਕਤੀ) {                                                                           \
                ਅਚਲ uint32_t pref_ਖੰਡ_ਮਾਪ = MIN(ਖੰਡ_2, src0_ਸਿਰਾ_ਪੰਕਤੀ - pref_ਖੰਡ);                                \
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(src0_spad, ਅੰਕੜੇ_src0 + (pref_ਖੰਡ * src0_ਪੰਕਤੀ_stride)),            \
                               src0_ਪੰਕਤੀ_ਮਾਪ_aligned, src0_ਪੰਕਤੀ_stride, src0_ਪੰਕਤੀ_ਮਾਪ, pref_ਖੰਡ_ਮਾਪ);                \
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(src1_spad, ਅੰਕੜੇ_src1 + (pref_ਖੰਡ * src1_ਪੰਕਤੀ_stride)),            \
                               src1_ਪੰਕਤੀ_ਮਾਪ_aligned, src1_ਪੰਕਤੀ_stride, src1_ਪੰਕਤੀ_ਮਾਪ, pref_ਖੰਡ_ਮਾਪ);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_ਕਤਾਰ_flush(dma_q);                                                                                        \
    }

DEFINE_GLU_ਪ੍ਰਤੀ_THREAD(ਦਸ਼ਮਲਵ, f32)
DEFINE_GLU_ਪ੍ਰਤੀ_THREAD(_Float16, f16)

ਸਥਿਰ ਪੂਰਨ execute_op_activations_f32(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਜੇ ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    glu_compute_f32_fn_t compute_fn = ਨਿਰਾ;
    ਅਚਲ ਅੱਖਰ *         op_type    = ਨਿਰਾ;

    ਚੋਣ (octx->op) {
        ਹਾਲਤ HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f32;
            op_type    = "swiglu-f32";
            ਰੋਕੋ;

        ਹਾਲਤ HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f32;
            op_type    = "swiglu-oai-f32";
            ਰੋਕੋ;

        ਹਾਲਤ HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f32;
            op_type    = "swiglu-clamp-f32";
            ਰੋਕੋ;

        ਹਾਲਤ HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f32;
            op_type    = "geglu-f32";
            ਰੋਕੋ;

        ਹਾਲਤ HTP_OP_GLU_GEGLU_ਤੇਜ਼:
            compute_fn = geglu_ਤੇਜ਼_f32;
            op_type    = "geglu-quick-f32";
            ਰੋਕੋ;

        ਹਾਲਤ HTP_OP_GLU_GEGLU_ERF:
            compute_fn = geglu_erf_f32;
            op_type    = "geglu-erf-f32";
            ਰੋਕੋ;

        ਮੂਲ:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ = dst->ne[0] * SIZEOF_FP32;

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = src0_nrows;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ), (uint32_t) dst_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(src0_nrows, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ਅਚਲ size_t nc_ਬਾਈਟਾਂ        = dst_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ   = nc_ਬਾਈਟਾਂ;
    ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ   = nc_ਬਾਈਟਾਂ;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_stride = src0->nb[1];
    ਅਚਲ size_t src1_ਪੰਕਤੀ_stride = src1 ? src1->nb[1] : src0->nb[1];

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਮਾਪ, VLEN);
    ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src1_ਪੰਕਤੀ_ਮਾਪ, VLEN);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = hex_ਗੇੜਾ_ਉੱਪਰ(dst_ਪੰਕਤੀ_ਮਾਪ, VLEN);

    ਬਣਤਰ htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_ਪੰਕਤੀ_ਮਾਪ_aligned, src1_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, n_ਧਾਗੇ,
                              octx->ctx->vtcm_ਮਾਪ);

    // Make sure the reserved vtcm size is sufficient
    ਜੇ (L.vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_ਮਾਪ,
             (src0_ਪੰਕਤੀ_ਮਾਪ_aligned + src1_ਪੰਕਤੀ_ਮਾਪ_aligned + dst_ਪੰਕਤੀ_ਮਾਪ_aligned) * n_ਧਾਗੇ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    ਜੇ (src1) {
        FARF(ਉੱਚਾ_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ,
             L.src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ, L.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ);
    } ਨਹੀਂਤਾਂ {
        FARF(ਉੱਚਾ_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ, L.src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ, L.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ);
    }

    // Prepare context
    ਬਣਤਰ htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (ਖਾਲੀ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);

    actx.src0_ਪੰਕਤੀ_ਮਾਪ = src0_ਪੰਕਤੀ_ਮਾਪ;
    actx.src1_ਪੰਕਤੀ_ਮਾਪ = src1_ਪੰਕਤੀ_ਮਾਪ;
    actx.dst_ਪੰਕਤੀ_ਮਾਪ  = dst_ਪੰਕਤੀ_ਮਾਪ;

    actx.src0_ਪੰਕਤੀ_ਮਾਪ_aligned = src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    actx.src1_ਪੰਕਤੀ_ਮਾਪ_aligned = src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    actx.dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    actx.src0_ਪੰਕਤੀ_stride = src0_ਪੰਕਤੀ_stride;
    actx.src1_ਪੰਕਤੀ_stride = src1_ਪੰਕਤੀ_stride;

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_dst);

    actx.vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = L.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    actx.vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread = L.src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    actx.vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = L.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    actx.src0_spad_half_ਮਾਪ = L.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / 2;
    actx.src1_spad_half_ਮਾਪ = L.src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / 2;
    actx.dst_spad_half_ਮਾਪ  = L.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / 2;

    actx.ਖੰਡ = actx.src0_spad_half_ਮਾਪ / actx.src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    actx.src0_nrows = nrows;
    actx.ਪੰਕਤੀ_ਸ਼ੁਰੂ  = ਪੰਕਤੀ_ਸ਼ੁਰੂ;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t ਅੰਕੜੇ_src0 = src0->ਅੰਕੜੇ;
    dma_addr_t ਅੰਕੜੇ_src1 = src1 ? src1->ਅੰਕੜੇ : 0;

    ਜੇ (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU ||
                  octx->op == HTP_OP_GLU_GEGLU_ਤੇਜ਼ ||
                  octx->op == HTP_OP_GLU_GEGLU_ERF)) {
         ਅਚਲ int32_t swapped = octx->op_params[1];
         ਅੰਕੜੇ_src1 = ਅੰਕੜੇ_src0;
         actx.src1_ਪੰਕਤੀ_ਮਾਪ = actx.src0_ਪੰਕਤੀ_ਮਾਪ;

         size_t nc_ਵਿੱਚ_ਬਾਈਟਾਂ = actx.nc * SIZEOF_FP32;
         ਜੇ (swapped) {
             ਅੰਕੜੇ_src0 += nc_ਵਿੱਚ_ਬਾਈਟਾਂ;
         } ਨਹੀਂਤਾਂ {
             ਅੰਕੜੇ_src1 += nc_ਵਿੱਚ_ਬਾਈਟਾਂ;
         }
    }

    actx.ਅੰਕੜੇ_src0 = ਅੰਕੜੇ_src0;
    actx.ਅੰਕੜੇ_src1 = ਅੰਕੜੇ_src1;
    actx.ਅੰਕੜੇ_dst  = dst->ਅੰਕੜੇ;

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, (ਕਾਮਾ_callback_t)glu_f32_ਪ੍ਰਤੀ_thread, &actx, n_ਧਾਗੇ);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ execute_op_activations_f16(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਜੇ ((dst->ne[0] * SIZEOF_FP16) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    glu_compute_f16_fn_t compute_fn = ਨਿਰਾ;
    ਅਚਲ ਅੱਖਰ *         op_type    = ਨਿਰਾ;

    ਚੋਣ (octx->op) {
        ਹਾਲਤ HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f16;
            op_type    = "swiglu-f16";
            ਰੋਕੋ;

        ਹਾਲਤ HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f16;
            op_type    = "swiglu-oai-f16";
            ਰੋਕੋ;

        ਹਾਲਤ HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f16;
            op_type    = "swiglu-clamp-f16";
            ਰੋਕੋ;

        ਹਾਲਤ HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f16;
            op_type    = "geglu-f16";
            ਰੋਕੋ;

        ਮੂਲ:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ = dst->ne[0] * SIZEOF_FP16;

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = src0_nrows;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, ਆਕਾਰ(_Float16), (uint32_t) dst_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(src0_nrows, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ਅਚਲ size_t nc_ਬਾਈਟਾਂ        = dst_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ   = nc_ਬਾਈਟਾਂ;
    ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ   = nc_ਬਾਈਟਾਂ;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_stride = src0->nb[1];
    ਅਚਲ size_t src1_ਪੰਕਤੀ_stride = src1 ? src1->nb[1] : src0->nb[1];

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਮਾਪ, VLEN);
    ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src1_ਪੰਕਤੀ_ਮਾਪ, VLEN);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = hex_ਗੇੜਾ_ਉੱਪਰ(dst_ਪੰਕਤੀ_ਮਾਪ, VLEN);

    ਬਣਤਰ htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_ਪੰਕਤੀ_ਮਾਪ_aligned, src1_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, n_ਧਾਗੇ,
                              octx->ctx->vtcm_ਮਾਪ);

    // Make sure the reserved vtcm size is sufficient
    ਜੇ (L.vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_ਮਾਪ,
             (src0_ਪੰਕਤੀ_ਮਾਪ_aligned + src1_ਪੰਕਤੀ_ਮਾਪ_aligned + dst_ਪੰਕਤੀ_ਮਾਪ_aligned) * n_ਧਾਗੇ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    ਜੇ (src1) {
        FARF(ਉੱਚਾ_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ,
             L.src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ, L.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ);
    } ਨਹੀਂਤਾਂ {
        FARF(ਉੱਚਾ_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ, L.src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ, L.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ);
    }

    // Prepare context
    ਬਣਤਰ htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (ਖਾਲੀ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);

    actx.src0_ਪੰਕਤੀ_ਮਾਪ = src0_ਪੰਕਤੀ_ਮਾਪ;
    actx.src1_ਪੰਕਤੀ_ਮਾਪ = src1_ਪੰਕਤੀ_ਮਾਪ;
    actx.dst_ਪੰਕਤੀ_ਮਾਪ  = dst_ਪੰਕਤੀ_ਮਾਪ;

    actx.src0_ਪੰਕਤੀ_ਮਾਪ_aligned = src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    actx.src1_ਪੰਕਤੀ_ਮਾਪ_aligned = src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    actx.dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    actx.src0_ਪੰਕਤੀ_stride = src0_ਪੰਕਤੀ_stride;
    actx.src1_ਪੰਕਤੀ_stride = src1_ਪੰਕਤੀ_stride;

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_dst);

    actx.vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = L.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    actx.vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread = L.src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    actx.vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = L.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;

    actx.src0_spad_half_ਮਾਪ = L.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / 2;
    actx.src1_spad_half_ਮਾਪ = L.src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / 2;
    actx.dst_spad_half_ਮਾਪ  = L.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread / 2;

    actx.ਖੰਡ = actx.src0_spad_half_ਮਾਪ / actx.src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    actx.src0_nrows = nrows;
    actx.ਪੰਕਤੀ_ਸ਼ੁਰੂ  = ਪੰਕਤੀ_ਸ਼ੁਰੂ;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t ਅੰਕੜੇ_src0 = src0->ਅੰਕੜੇ;
    dma_addr_t ਅੰਕੜੇ_src1 = src1 ? src1->ਅੰਕੜੇ : 0;

    ਜੇ (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU)) {
         ਅਚਲ int32_t swapped = octx->op_params[1];
         ਅੰਕੜੇ_src1 = ਅੰਕੜੇ_src0;
         actx.src1_ਪੰਕਤੀ_ਮਾਪ = actx.src0_ਪੰਕਤੀ_ਮਾਪ;

         size_t nc_ਵਿੱਚ_ਬਾਈਟਾਂ = actx.nc * SIZEOF_FP16;
         ਜੇ (swapped) {
             ਅੰਕੜੇ_src0 += nc_ਵਿੱਚ_ਬਾਈਟਾਂ;
         } ਨਹੀਂਤਾਂ {
             ਅੰਕੜੇ_src1 += nc_ਵਿੱਚ_ਬਾਈਟਾਂ;
         }
    }

    actx.ਅੰਕੜੇ_src0 = ਅੰਕੜੇ_src0;
    actx.ਅੰਕੜੇ_src1 = ਅੰਕੜੇ_src1;
    actx.ਅੰਕੜੇ_dst  = dst->ਅੰਕੜੇ;

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, (ਕਾਮਾ_callback_t)glu_f16_ਪ੍ਰਤੀ_thread, &actx, n_ਧਾਗੇ);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}


ਪੂਰਨ op_activations(ਬਣਤਰ htp_ops_context * octx) {
    ਚੋਣ (octx->src[0]->type) {
        ਹਾਲਤ HTP_TYPE_F32:
            ਵਾਪਸ execute_op_activations_f32(octx);

        ਹਾਲਤ HTP_TYPE_F16:
            ਵਾਪਸ execute_op_activations_f16(octx);

        ਮੂਲ:
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
}
