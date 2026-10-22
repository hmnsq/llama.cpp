#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"

#ವ್ಯಾಖ್ಯಾನಿಸು htp_act_preamble                                 \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = actx->octx->src[0]; \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = actx->octx->src[1]; \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = actx->octx->dst;    \
                                                         \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];                   \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1];                   \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];                   \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];                   \
                                                         \
    ಸ್ಥಿರಾಂಕ uint32_t nb00 = src0->nb[0];                   \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1];                   \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];                   \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];                   \
                                                         \
    ಸ್ಥಿರಾಂಕ uint32_t ne10 = src1 ? src1->ne[0] : 0;        \
    ಸ್ಥಿರಾಂಕ uint32_t ne11 = src1 ? src1->ne[1] : 0;        \
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = src1 ? src1->ne[2] : 0;        \
    ಸ್ಥಿರಾಂಕ uint32_t ne13 = src1 ? src1->ne[3] : 0;        \
                                                         \
    ಸ್ಥಿರಾಂಕ uint32_t nb10 = src1 ? src1->nb[0] : 0;        \
    ಸ್ಥಿರಾಂಕ uint32_t nb11 = src1 ? src1->nb[1] : 0;        \
    ಸ್ಥಿರಾಂಕ uint32_t nb12 = src1 ? src1->nb[2] : 0;        \
    ಸ್ಥಿರಾಂಕ uint32_t nb13 = src1 ? src1->nb[3] : 0;        \
                                                         \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];                     \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];                     \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];                     \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];                     \
                                                         \
    ಸ್ಥಿರಾಂಕ uint32_t nb0 = dst->nb[0];                     \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];                     \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];                     \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

ರಚನೆ htp_act_context;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*glu_compute_f32_fn_t)(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src0,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src1,
                                     ದಶಮಾಂಶ * ಸೀಮಿತ dst,
                                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*glu_compute_f16_fn_t)(ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src0,
                                     ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src1,
                                     _Float16 * ಸೀಮಿತ dst,
                                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx);

ರಚನೆ htp_act_context {
    ರಚನೆ htp_ops_context * octx;

    ಶೂನ್ಯ *                   compute;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *             op_str;

    // Precomputed values
    dma_addr_t               ದತ್ತಾಂಶ_src0;
    dma_addr_t               ದತ್ತಾಂಶ_src1;
    dma_addr_t               ದತ್ತಾಂಶ_dst;

    size_t                   src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t                   src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t                   dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

    size_t                   src0_ಅಡ್ಡಸಾಲು_stride;
    size_t                   src1_ಅಡ್ಡಸಾಲು_stride;

    size_t                   src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t                   src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t                   dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    size_t                   src0_spad_half_ಗಾತ್ರ;
    size_t                   src1_spad_half_ಗಾತ್ರ;
    size_t                   dst_spad_half_ಗಾತ್ರ;

    uint32_t                 ಖಂಡ;
    uint32_t                 src0_nrows;
    uint32_t                 src0_nrows_ಪ್ರತಿ_thread;
    uint32_t                 ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ಪೂರ್ಣಾಂಕ                      nc;

    uint8_t *                vtcm_src0;
    uint8_t *                vtcm_src1;
    uint8_t *                vtcm_dst;

    size_t                   vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    size_t                   vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    size_t                   vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;
};

ರಚನೆ htp_act_vtcm_layout {
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    uint32_t vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_act_vtcm_layout_build(ರಚನೆ htp_act_vtcm_layout * L,
                                             size_t                       src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
                                             size_t                       src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
                                             size_t                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
                                             uint32_t                     n_ಎಳೆಗಳು,
                                             size_t                       vtcm_ಗಾತ್ರ) {
    ಸ್ಥಿರಾಂಕ size_t   spad_ಗಾತ್ರ_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು   = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    ಸ್ಥಿರಾಂಕ uint32_t vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread = (uint32_t) (vtcm_ಗಾತ್ರ / (n_ಎಳೆಗಳು * spad_ಗಾತ್ರ_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು));

    L->vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread = vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread;

    L->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread;
    L->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread;
    L->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread;

    L->off_src0 = 0;
    L->off_src1 = L->off_src0 + L->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
    L->off_dst  = L->off_src1 + L->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;

    L->ಒಟ್ಟು_ಬೈಟುಗಳು = L->off_dst + L->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
}

#ವ್ಯಾಖ್ಯಾನಿಸು htp_glu_op_preamble                                            \
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = actx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;  \
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = actx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;  \
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = actx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;   \
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    nc                    = actx->nc;

// swiglu(x) = x1 * sigmoid(x0)
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src0,
                                                 ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src1,
                                                 ದಶಮಾಂಶ * ಸೀಮಿತ dst,
                                                 ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                                                 ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_oai_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src0,
                                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src1,
                                                     ದಶಮಾಂಶ * ಸೀಮಿತ dst,
                                                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                                                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (actx->octx->op_params))[2];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಎಲ್ಲೆ = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (actx->octx->op_params))[3];

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, ಎಲ್ಲೆ, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -ಎಲ್ಲೆ, ಎಲ್ಲೆ, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_ಕೂಡಿಸು_scalar_f32((uint8_t *) src1_ptr, src1_ptr, 1.0, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f32(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f32_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ swiglu_clamp_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src0,
                                                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src1,
                                                       ದಶಮಾಂಶ * ಸೀಮಿತ dst,
                                                       ಸ್ಥಿರಾಂಕ uint32_t                 ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                                                       ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಎಲ್ಲೆ = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (actx->octx->op_params))[3];

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_min_scalar_f32((uint8_t *) src0_ptr, src0_ptr, ಎಲ್ಲೆ, nc);
        hvx_clamp_scalar_f32((uint8_t *) src1_ptr, src1_ptr, -ಎಲ್ಲೆ, ಎಲ್ಲೆ, nc);
        hvx_sigmoid_f32_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f32_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ GELU_COEF_A     = 0.044715f;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_fast_sigmoid_f32_2it(HVX_ಸದಿಶ v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_ಸದಿಶ ಒಳ_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_ಸದಿಶ x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(ಒಳ_int));
    HVX_ಸದಿಶ xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_ಸದಿಶ v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_ಸದಿಶ v3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    v3            = Q6_Vw_vaslacc_VwVwR(v3, ಒಳ_int, 24);

    HVX_ಸದಿಶ v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_ಸದಿಶ v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    // Newton-Raphson with 2 iterations
    HVX_ಸದಿಶ two_sf = hvx_vec_splat_f32(2.0f);
    HVX_ಸದಿಶ i_sf   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EEEEBB3), v5);
    HVX_ಸದಿಶ r_qf   = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v5)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v5))));
    HVX_ಸದಿಶ res = Q6_Vsf_equals_Vqf32(r_qf);

    res = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(res);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_vec_fast_sigmoid_f32_guard_2it(HVX_ಸದಿಶ v,
                                                            HVX_ಸದಿಶ one,
                                                            HVX_ಸದಿಶ max_exp,
                                                            HVX_ಸದಿಶ min_exp) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_ಸದಿಶ ಹೊರಹೋಗು = hvx_vec_fast_sigmoid_f32_2it(v);
    ಹೊರಹೋಗು            = Q6_V_vmux_QVV(pred_max, ಹೊರಹೋಗು, one);
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(pred_min, ಹೊರಹೋಗು, Q6_V_vzero());
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_geglu_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst  % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src0 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src1 % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst        = (HVX_ಸದಿಶ *) dst;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vsrc0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vsrc1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src1;

    ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ GELU_COEF_A_TIMES_SQRT = GELU_COEF_A * SQRT_2_OVER_PI;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_coef_a_times_sqrt = hvx_vec_splat_f32(GELU_COEF_A_TIMES_SQRT);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_sqrt_2_pi         = hvx_vec_splat_f32(SQRT_2_OVER_PI);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_one               = hvx_vec_splat_f32(1.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_max_exp           = hvx_vec_splat_f32(87.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_min_exp           = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        HVX_ಸದಿಶ x = vsrc0[i];
        HVX_ಸದಿಶ g = vsrc1[i];

        HVX_ಸದಿಶ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_ಸದಿಶ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_ಕೂಡಿಸು_f32_f32(coef, v_sqrt_2_pi);
        HVX_ಸದಿಶ inner = hvx_vec_mul_f32_f32(x, coef);

        // y2 = 2 * inner = inner + inner
        HVX_ಸದಿಶ y2 = hvx_vec_ಕೂಡಿಸು_f32_f32(inner, inner);

        // Fast sigmoid approximation (2 iterations)
        HVX_ಸದಿಶ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_ಸದಿಶ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        vdst[i] = hvx_vec_mul_f32_f32(gelu_x, g);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ x = vsrc0[i];
        HVX_ಸದಿಶ g = vsrc1[i];

        HVX_ಸದಿಶ x2 = hvx_vec_mul_f32_f32(x, x);
        HVX_ಸದಿಶ coef = hvx_vec_mul_f32_f32(x2, v_coef_a_times_sqrt);
        coef = hvx_vec_ಕೂಡಿಸು_f32_f32(coef, v_sqrt_2_pi);
        HVX_ಸದಿಶ inner = hvx_vec_mul_f32_f32(x, coef);

        HVX_ಸದಿಶ y2 = hvx_vec_ಕೂಡಿಸು_f32_f32(inner, inner);

        HVX_ಸದಿಶ sig2y = hvx_vec_fast_sigmoid_f32_guard_2it(y2, v_one, v_max_exp, v_min_exp);

        HVX_ಸದಿಶ gelu_x = hvx_vec_mul_f32_f32(x, sig2y);
        HVX_ಸದಿಶ res = hvx_vec_mul_f32_f32(gelu_x, g);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), res);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_geglu_ಶೀಘ್ರ_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst  % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src0 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src1 % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst        = (HVX_ಸದಿಶ *) dst;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vsrc0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vsrc1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src1;

    ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scale    = hvx_vec_splat_f32(1.702f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_one      = hvx_vec_splat_f32(1.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_max_exp  = hvx_vec_splat_f32(87.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_min_exp  = hvx_vec_splat_f32(-87.0f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        HVX_ಸದಿಶ x = vsrc0[i];
        HVX_ಸದಿಶ g = vsrc1[i];
        HVX_ಸದಿಶ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_ಸದಿಶ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ x = vsrc0[i];
        HVX_ಸದಿಶ g = vsrc1[i];
        HVX_ಸದಿಶ scaled_x = hvx_vec_mul_f32_f32(x, v_scale);
        HVX_ಸದಿಶ sigmoid_x = hvx_vec_fast_sigmoid_f32_guard_2it(scaled_x, v_one, v_max_exp, v_min_exp);
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, sigmoid_x), g);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), ಫಲಿತಾಂಶ);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_geglu_erf_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst  % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src0 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src1 % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst        = (HVX_ಸದಿಶ *) dst;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vsrc0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vsrc1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src1;

    ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        HVX_ಸದಿಶ x = vsrc0[i];
        HVX_ಸದಿಶ g = vsrc1[i];
        vdst[i] = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ x = vsrc0[i];
        HVX_ಸದಿಶ g = vsrc1[i];
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = hvx_vec_mul_f32_f32(hvx_vec_gelu_erf_f32(x), g);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), ಫಲಿತಾಂಶ);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_geglu_f16_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst  % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src0 % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src1 % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst        = (HVX_ಸದಿಶ *) dst;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vsrc0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src0;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ vsrc1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src1;

    ಸ್ಥಿರಾಂಕ uint32_t epv  = 128 / ಗಾತ್ರ(_Float16);
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / epv;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % epv;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_coef_a    = hvx_vec_splat_f16((_Float16) GELU_COEF_A);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_sqrt_2_pi = hvx_vec_splat_f16((_Float16) SQRT_2_OVER_PI);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_one       = hvx_vec_splat_f16((_Float16) 1.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_half      = hvx_vec_splat_f16((_Float16) 0.5f);

    uint32_t i = 0;

    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        HVX_ಸದಿಶ x = vsrc0[i];
        HVX_ಸದಿಶ g = vsrc1[i];

        // gelu(x) = 0.5*x*(1 + tanh(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))
        HVX_ಸದಿಶ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_ಸದಿಶ coef  = hvx_vec_ಕೂಡಿಸು_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_ಸದಿಶ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_ಸದಿಶ t = hvx_vec_ಕೂಡಿಸು_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_ಸದಿಶ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        vdst[i] = hvx_vec_mul_f16_f16(gelu_x, g);
    }

    ಆದರೆ (nloe) {
        HVX_ಸದಿಶ x = vsrc0[i];
        HVX_ಸದಿಶ g = vsrc1[i];

        HVX_ಸದಿಶ x2    = hvx_vec_mul_f16_f16(x, x);
        HVX_ಸದಿಶ coef  = hvx_vec_ಕೂಡಿಸು_f16_f16(hvx_vec_mul_f16_f16(x2, v_coef_a), v_one);
        HVX_ಸದಿಶ inner = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, coef), v_sqrt_2_pi);

        HVX_ಸದಿಶ t = hvx_vec_ಕೂಡಿಸು_f16_f16(hvx_vec_tanh_f16(inner), v_one);

        HVX_ಸದಿಶ gelu_x = hvx_vec_mul_f16_f16(hvx_vec_mul_f16_f16(x, t), v_half);
        HVX_ಸದಿಶ res    = hvx_vec_mul_f16_f16(gelu_x, g);
        hvx_vec_store_a((ಶೂನ್ಯ *) &vdst[i], nloe * ಗಾತ್ರ(_Float16), res);
    }
}

// geglu(x, g) = gelu(x) * g
ಸ್ಥಿರ ಶೂನ್ಯ geglu_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src0,
                                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src1,
                                                ದಶಮಾಂಶ * ಸೀಮಿತ dst,
                                                ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                                                ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_geglu_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_quick(x, g) = x * sigmoid(1.702 * x) * g
ಸ್ಥಿರ ಶೂನ್ಯ geglu_ಶೀಘ್ರ_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src0,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src1,
                            ದಶಮಾಂಶ * ಸೀಮಿತ dst,
                            ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_geglu_ಶೀಘ್ರ_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// geglu_erf(x, g) = gelu_erf(x) * g
ಸ್ಥಿರ ಶೂನ್ಯ geglu_erf_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src0,
                          ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src1,
                          ದಶಮಾಂಶ * ಸೀಮಿತ dst,
                          ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                          ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_geglu_erf_f32_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

// swiglu(x) = x1 * sigmoid(x0)
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_f16(ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src0,
                       ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src1,
                       _Float16 * ಸೀಮಿತ dst,
                       ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                       ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// out = x * sigmoid(alpha * x) * (clamp(y, -limit, limit) + 1.f)
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_oai_f16(ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src0,
                           ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src1,
                           _Float16 * ಸೀಮಿತ dst,
                           ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                           ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;
    ಸ್ಥಿರಾಂಕ _Float16 alpha = (_Float16) ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (actx->octx->op_params))[2];
    ಸ್ಥಿರಾಂಕ _Float16 ಎಲ್ಲೆ = (_Float16) ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (actx->octx->op_params))[3];

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        // x (src0_ptr) = std::min(src0_p[k], limit);
        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, ಎಲ್ಲೆ, nc);
        // y1 (src1_ptr) = std::clamp(src1_p[k], -limit, limit);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -ಎಲ್ಲೆ, ಎಲ್ಲೆ, nc);
        // y (src1_ptr) = y1 + 1.f
        hvx_ಕೂಡಿಸು_scalar_f16((uint8_t *) src1_ptr, src1_ptr, (_Float16) 1.0f, nc);
        // x1 (dst_ptr) = alpha * x
        hvx_mul_scalar_f16(dst_ptr, src0_ptr, alpha, nc);
        // x2 (dst_ptr) = sigmoid(x1) = 1/(1+exp(-x1))
        hvx_sigmoid_f16_aa(dst_ptr, dst_ptr, nc);
        // out = x * sigmoid(alpha * x) * (y + 1.f)
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ swiglu_clamp_f16(ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src0,
                             ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src1,
                             _Float16 * ಸೀಮಿತ dst,
                             ಸ್ಥಿರಾಂಕ uint32_t                 ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                             ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;
    ಸ್ಥಿರಾಂಕ _Float16 ಎಲ್ಲೆ = (_Float16) ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (actx->octx->op_params))[3];

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_min_scalar_f16((uint8_t *) src0_ptr, src0_ptr, ಎಲ್ಲೆ, nc);
        hvx_clamp_scalar_f16((uint8_t *) src1_ptr, src1_ptr, -ಎಲ್ಲೆ, ಎಲ್ಲೆ, nc);
        hvx_sigmoid_f16_aa(dst_ptr, src0_ptr, nc);
        hvx_mul_mul_f16_aa(dst_ptr, src0_ptr, dst_ptr, src1_ptr, nc);
    }
}

// geglu(x, g) = gelu(x) * g, gelu via tanh approximation
ಸ್ಥಿರ ಶೂನ್ಯ geglu_f16(ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src0,
                      ಸ್ಥಿರಾಂಕ _Float16 * ಸೀಮಿತ src1,
                      _Float16 * ಸೀಮಿತ dst,
                      ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                      ಸ್ಥಿರಾಂಕ ರಚನೆ htp_act_context * actx) {
    htp_glu_op_preamble;

    ಗಾಗಿ (uint32_t ib = 0; ib < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ib++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src0 + (ib * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1_ptr = (ಸ್ಥಿರಾಂಕ uint8_t *) src1 + (ib * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ptr        = (uint8_t *) dst + (ib * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_geglu_f16_aa(dst_ptr, src0_ptr, src1_ptr, nc);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_GLU_ಪ್ರತಿ_THREAD(TYPE, SUFFIX)                                                                            \
    ಸ್ಥಿರ ಶೂನ್ಯ glu_##SUFFIX##_per_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {                           \
        ರಚನೆ htp_act_context * actx = (ರಚನೆ htp_act_context *) ದತ್ತಾಂಶ;                                               \
        htp_act_preamble;                                                                                              \
                                                                                                                       \
        ರಚನೆ htp_thread_trace * tr = actx->octx->ctx ? &actx->octx->ctx->trace[ith] : ಖಾಲಿ;                          \
                                                                                                                       \
        size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = actx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;                                                                    \
        size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = actx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;                                                                    \
        size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = actx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;                                                                     \
                                                                                                                       \
        size_t src0_ಅಡ್ಡಸಾಲು_stride = actx->src0_ಅಡ್ಡಸಾಲು_stride;                                                                \
        size_t src1_ಅಡ್ಡಸಾಲು_stride = actx->src1_ಅಡ್ಡಸಾಲು_stride;                                                                \
                                                                                                                       \
        ಸ್ಥಿರಾಂಕ uint32_t src0_nrows            = actx->src0_nrows;                                                       \
        ಸ್ಥಿರಾಂಕ uint32_t src0_nrows_ಪ್ರತಿ_thread = actx->src0_nrows_ಪ್ರತಿ_thread;                                            \
                                                                                                                       \
        ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = actx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;                                 \
        ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, actx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows);     \
                                                                                                                       \
        /* no work for this thread */                                                                                  \
        ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {                                                                          \
            ಹಿಂತಿರುಗಿ;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src0 = actx->ದತ್ತಾಂಶ_src0;                                                                  \
        ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src1 = actx->ದತ್ತಾಂಶ_src1;                                                                  \
        ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_dst  = actx->ದತ್ತಾಂಶ_dst;                                                                   \
                                                                                                                       \
        ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = actx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;                                              \
        ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = actx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;                                              \
        ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = actx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;                                               \
                                                                                                                       \
        uint8_t * ಸೀಮಿತ src0_spad_ದತ್ತಾಂಶ = actx->vtcm_src0 + (ith * actx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread);                 \
        uint8_t * ಸೀಮಿತ src1_spad_ದತ್ತಾಂಶ = actx->vtcm_src1 + (ith * actx->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread);                 \
        uint8_t * ಸೀಮಿತ dst_spad_ದತ್ತಾಂಶ  = actx->vtcm_dst  + (ith * actx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread);                  \
                                                                                                                       \
        size_t src0_spad_half_ಗಾತ್ರ = actx->src0_spad_half_ಗಾತ್ರ;                                                        \
        size_t src1_spad_half_ಗಾತ್ರ = actx->src1_spad_half_ಗಾತ್ರ;                                                        \
        size_t dst_spad_half_ಗಾತ್ರ  = actx->dst_spad_half_ಗಾತ್ರ;                                                         \
                                                                                                                       \
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_2 = actx->ಖಂಡ;                                                                                 \
        ಆದರೆ (ಖಂಡ_2 == 0) {                                                                                              \
            FARF(ERROR, "%s : VTCM reservation %zu is too small, needed %zu\n",                                        \
                 actx->op_str, actx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);                                \
            ಹಿಂತಿರುಗಿ;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        dma_ಸರತಿ * dma_q = actx->octx->ctx->dma[ith];                                                                 \
        glu_compute_##SUFFIX##_fn_t compute = (glu_compute_##SUFFIX##_fn_t) actx->compute;                             \
                                                                                                                       \
        ಗಾಗಿ (uint32_t ir = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, spad_idx = 0; ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು && spad_idx < 2; ir += ಖಂಡ_2, spad_idx++) { \
            ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = MIN(ಖಂಡ_2, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - ir);                                                 \
                                                                                                                       \
            /* Dummy DMA transation for sequencing (interleaving dst,src,dst,...) */                                   \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                                      \
                           dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst, dst_spad_ದತ್ತಾಂಶ + (spad_idx * dst_spad_half_ಗಾತ್ರ)),                   \
                           dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);                                       \
                                                                                                                       \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                                      \
                dma_make_ದತ್ತಾಂಶ(src0_spad_ದತ್ತಾಂಶ + (spad_idx * src0_spad_half_ಗಾತ್ರ), ದತ್ತಾಂಶ_src0 + (ir * src0_ಅಡ್ಡಸಾಲು_stride)),  \
                src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);                                    \
                                                                                                                       \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                                      \
                dma_make_ದತ್ತಾಂಶ(src1_spad_ದತ್ತಾಂಶ + (spad_idx * src1_spad_half_ಗಾತ್ರ), ದತ್ತಾಂಶ_src1 + (ir * src1_ಅಡ್ಡಸಾಲು_stride)),  \
                src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src1_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);                                    \
        }                                                                                                              \
                                                                                                                       \
        ಗಾಗಿ (uint32_t ir = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು; ir += ಖಂಡ_2) {                                           \
            ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = MIN(ಖಂಡ_2, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - ir);                                                 \
                                                                                                                       \
            TYPE * dst_spad  = (TYPE *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;                                                      \
            TYPE * src0_spad = (TYPE *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;                                                      \
            TYPE * src1_spad = (TYPE *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;                                                      \
                                                                                                                       \
            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                     \
            compute(src0_spad, src1_spad, dst_spad, ಖಂಡ_ಗಾತ್ರ_3, actx);                                                 \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);                                                      \
                                                                                                                       \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ), dst_spad),                             \
                           dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);                              \
                                                                                                                       \
            /* prefetch N+2 loop iteration if any */                                                                   \
            ಸ್ಥಿರಾಂಕ uint32_t pref_ಖಂಡ = (ir + ಖಂಡ_2 * 2);                                                              \
            ಆದರೆ (pref_ಖಂಡ < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {                                                                           \
                ಸ್ಥಿರಾಂಕ uint32_t pref_ಖಂಡ_ಗಾತ್ರ = MIN(ಖಂಡ_2, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - pref_ಖಂಡ);                                \
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(src0_spad, ದತ್ತಾಂಶ_src0 + (pref_ಖಂಡ * src0_ಅಡ್ಡಸಾಲು_stride)),            \
                               src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು_stride, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, pref_ಖಂಡ_ಗಾತ್ರ);                \
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(src1_spad, ದತ್ತಾಂಶ_src1 + (pref_ಖಂಡ * src1_ಅಡ್ಡಸಾಲು_stride)),            \
                               src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src1_ಅಡ್ಡಸಾಲು_stride, src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, pref_ಖಂಡ_ಗಾತ್ರ);                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        dma_ಸರತಿ_flush(dma_q);                                                                                        \
    }

DEFINE_GLU_ಪ್ರತಿ_THREAD(ದಶಮಾಂಶ, f32)
DEFINE_GLU_ಪ್ರತಿ_THREAD(_Float16, f16)

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ execute_op_activations_f32(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆದರೆ ((dst->ne[0] * SIZEOF_FP32) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    glu_compute_f32_fn_t compute_fn = ಖಾಲಿ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *         op_type    = ಖಾಲಿ;

    ಆಯ್ಕೆ (octx->op) {
        ಸ್ಥಿತಿ HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f32;
            op_type    = "swiglu-f32";
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f32;
            op_type    = "swiglu-oai-f32";
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f32;
            op_type    = "swiglu-clamp-f32";
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f32;
            op_type    = "geglu-f32";
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_OP_GLU_GEGLU_ಶೀಘ್ರ:
            compute_fn = geglu_ಶೀಘ್ರ_f32;
            op_type    = "geglu-quick-f32";
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_OP_GLU_GEGLU_ERF:
            compute_fn = geglu_erf_f32;
            op_type    = "geglu-erf-f32";
            ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->ne[0] * SIZEOF_FP32;

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = src0_nrows;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, ಗಾತ್ರ(ದಶಮಾಂಶ), (uint32_t) dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(src0_nrows, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ಸ್ಥಿರಾಂಕ size_t nc_ಬೈಟುಗಳು        = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ   = nc_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ   = nc_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_stride = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_stride = src1 ? src1->nb[1] : src0->nb[1];

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = hex_ಸುತ್ತು_ಮೇಲೆ(dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);

    ರಚನೆ htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, n_ಎಳೆಗಳು,
                              octx->ctx->vtcm_ಗಾತ್ರ);

    // Make sure the reserved vtcm size is sufficient
    ಆದರೆ (L.vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_ಗಾತ್ರ,
             (src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned) * n_ಎಳೆಗಳು);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    ಆದರೆ (src1) {
        FARF(ಮೇಲಿನ_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು,
             L.src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು, L.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        FARF(ಮೇಲಿನ_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು, L.src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು, L.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು);
    }

    // Prepare context
    ರಚನೆ htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (ಶೂನ್ಯ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);

    actx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    actx.src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    actx.dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

    actx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    actx.src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    actx.dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    actx.src0_ಅಡ್ಡಸಾಲು_stride = src0_ಅಡ್ಡಸಾಲು_stride;
    actx.src1_ಅಡ್ಡಸಾಲು_stride = src1_ಅಡ್ಡಸಾಲು_stride;

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_dst);

    actx.vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = L.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    actx.vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread = L.src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    actx.vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = L.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    actx.src0_spad_half_ಗಾತ್ರ = L.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / 2;
    actx.src1_spad_half_ಗಾತ್ರ = L.src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / 2;
    actx.dst_spad_half_ಗಾತ್ರ  = L.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / 2;

    actx.ಖಂಡ = actx.src0_spad_half_ಗಾತ್ರ / actx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    actx.src0_nrows = nrows;
    actx.ಅಡ್ಡಸಾಲು_ಆರಂಭ  = ಅಡ್ಡಸಾಲು_ಆರಂಭ;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t ದತ್ತಾಂಶ_src0 = src0->ದತ್ತಾಂಶ;
    dma_addr_t ದತ್ತಾಂಶ_src1 = src1 ? src1->ದತ್ತಾಂಶ : 0;

    ಆದರೆ (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU ||
                  octx->op == HTP_OP_GLU_GEGLU_ಶೀಘ್ರ ||
                  octx->op == HTP_OP_GLU_GEGLU_ERF)) {
         ಸ್ಥಿರಾಂಕ int32_t swapped = octx->op_params[1];
         ದತ್ತಾಂಶ_src1 = ದತ್ತಾಂಶ_src0;
         actx.src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = actx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

         size_t nc_ಒಳ_ಬೈಟುಗಳು = actx.nc * SIZEOF_FP32;
         ಆದರೆ (swapped) {
             ದತ್ತಾಂಶ_src0 += nc_ಒಳ_ಬೈಟುಗಳು;
         } ಇಲ್ಲದಿದ್ದರೆ {
             ದತ್ತಾಂಶ_src1 += nc_ಒಳ_ಬೈಟುಗಳು;
         }
    }

    actx.ದತ್ತಾಂಶ_src0 = ದತ್ತಾಂಶ_src0;
    actx.ದತ್ತಾಂಶ_src1 = ದತ್ತಾಂಶ_src1;
    actx.ದತ್ತಾಂಶ_dst  = dst->ದತ್ತಾಂಶ;

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, (ಕೆಲಸಗಾರ_callback_t)glu_f32_ಪ್ರತಿ_thread, &actx, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ execute_op_activations_f16(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆದರೆ ((dst->ne[0] * SIZEOF_FP16) != dst->nb[1]) {
        FARF(ERROR, "Non-contiguous dst is not supported at this time \n");
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    glu_compute_f16_fn_t compute_fn = ಖಾಲಿ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *         op_type    = ಖಾಲಿ;

    ಆಯ್ಕೆ (octx->op) {
        ಸ್ಥಿತಿ HTP_OP_GLU_SWIGLU:
            compute_fn = swiglu_f16;
            op_type    = "swiglu-f16";
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_OP_GLU_SWIGLU_OAI:
            compute_fn = swiglu_oai_f16;
            op_type    = "swiglu-oai-f16";
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_OP_GLU_SWIGLU_CLAMP:
            compute_fn = swiglu_clamp_f16;
            op_type    = "swiglu-clamp-f16";
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_OP_GLU_GEGLU:
            compute_fn = geglu_f16;
            op_type    = "geglu-f16";
            ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            FARF(ERROR, "Unsupported activations Op %u\n", octx->op);
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->ne[0] * SIZEOF_FP16;

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = src0_nrows;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, ಗಾತ್ರ(_Float16), (uint32_t) dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(src0_nrows, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    // row_size   = bytes of useful data per row (what the kernel touches / what DMA copies).
    // row_stride = bytes between successive rows in DDR (may exceed row_size for non-contig src).
    ಸ್ಥಿರಾಂಕ size_t nc_ಬೈಟುಗಳು        = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ   = nc_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ   = nc_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_stride = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_stride = src1 ? src1->nb[1] : src0->nb[1];

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = hex_ಸುತ್ತು_ಮೇಲೆ(dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);

    ರಚನೆ htp_act_vtcm_layout L;
    htp_act_vtcm_layout_build(&L, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, n_ಎಳೆಗಳು,
                              octx->ctx->vtcm_ಗಾತ್ರ);

    // Make sure the reserved vtcm size is sufficient
    ಆದರೆ (L.vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread == 0) {
        FARF(ERROR, "act-%s : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n", op_type, octx->ctx->vtcm_ಗಾತ್ರ,
             (src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned) * n_ಎಳೆಗಳು);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    ಆದರೆ (src1) {
        FARF(ಮೇಲಿನ_2, "%s: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n",
             op_type, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2],
             src1->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], L.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು,
             L.src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು, L.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        FARF(ಮೇಲಿನ_2, "%s: %ux%ux%ux%u -> %ux%ux%ux%u : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu\n", op_type,
             src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
             L.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು, L.src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು, L.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು);
    }

    // Prepare context
    ರಚನೆ htp_act_context actx;
    actx.octx    = octx;
    actx.compute = (ಶೂನ್ಯ *) compute_fn;
    actx.op_str  = op_type;

    actx.src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);

    actx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    actx.src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    actx.dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

    actx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    actx.src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    actx.dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    actx.src0_ಅಡ್ಡಸಾಲು_stride = src0_ಅಡ್ಡಸಾಲು_stride;
    actx.src1_ಅಡ್ಡಸಾಲು_stride = src1_ಅಡ್ಡಸಾಲು_stride;

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;
    actx.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_src0);
    actx.vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_src1);
    actx.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_dst);

    actx.vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = L.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    actx.vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread = L.src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    actx.vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = L.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    actx.src0_spad_half_ಗಾತ್ರ = L.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / 2;
    actx.src1_spad_half_ಗಾತ್ರ = L.src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / 2;
    actx.dst_spad_half_ಗಾತ್ರ  = L.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / 2;

    actx.ಖಂಡ = actx.src0_spad_half_ಗಾತ್ರ / actx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    actx.src0_nrows = nrows;
    actx.ಅಡ್ಡಸಾಲು_ಆರಂಭ  = ಅಡ್ಡಸಾಲು_ಆರಂಭ;

    actx.nc = dst->ne[0];

    // Addresses and GLU logic
    dma_addr_t ದತ್ತಾಂಶ_src0 = src0->ದತ್ತಾಂಶ;
    dma_addr_t ದತ್ತಾಂಶ_src1 = src1 ? src1->ದತ್ತಾಂಶ : 0;

    ಆದರೆ (!src1 && (octx->op == HTP_OP_GLU_SWIGLU ||
                  octx->op == HTP_OP_GLU_SWIGLU_OAI ||
                  octx->op == HTP_OP_GLU_SWIGLU_CLAMP ||
                  octx->op == HTP_OP_GLU_GEGLU)) {
         ಸ್ಥಿರಾಂಕ int32_t swapped = octx->op_params[1];
         ದತ್ತಾಂಶ_src1 = ದತ್ತಾಂಶ_src0;
         actx.src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = actx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

         size_t nc_ಒಳ_ಬೈಟುಗಳು = actx.nc * SIZEOF_FP16;
         ಆದರೆ (swapped) {
             ದತ್ತಾಂಶ_src0 += nc_ಒಳ_ಬೈಟುಗಳು;
         } ಇಲ್ಲದಿದ್ದರೆ {
             ದತ್ತಾಂಶ_src1 += nc_ಒಳ_ಬೈಟುಗಳು;
         }
    }

    actx.ದತ್ತಾಂಶ_src0 = ದತ್ತಾಂಶ_src0;
    actx.ದತ್ತಾಂಶ_src1 = ದತ್ತಾಂಶ_src1;
    actx.ದತ್ತಾಂಶ_dst  = dst->ದತ್ತಾಂಶ;

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, (ಕೆಲಸಗಾರ_callback_t)glu_f16_ಪ್ರತಿ_thread, &actx, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}


ಪೂರ್ಣಾಂಕ op_activations(ರಚನೆ htp_ops_context * octx) {
    ಆಯ್ಕೆ (octx->src[0]->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:
            ಹಿಂತಿರುಗಿ execute_op_activations_f32(octx);

        ಸ್ಥಿತಿ HTP_TYPE_F16:
            ಹಿಂತಿರುಗಿ execute_op_activations_f16(octx);

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
}
