#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>

#समावेश <math.h>
#समावेश <string.h>

#समावेश "dma-queue.h"
#समावेश "work-queue.h"
#समावेश "hvx-utils.h"
#समावेश "hex-fastdiv.h"
#समावेश "hex-common.h"
#समावेश "hex-profile.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "htp-vtcm.h"
#समावेश "htp/softmax-ops.h"
#समावेश "hvx-flash-attn.h"

काष्ठा htp_softmax_context {
    काष्ठा htp_ops_context * octx;
    स्थिर काष्ठा htp_softmax_kernel_params * kparams;

    व्योम *   compute;

    dma_addr_t तथ्य_src0;
    dma_addr_t तथ्य_src1;
    dma_addr_t तथ्य_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_आकार_per_thread;
    uint32_t  vtcm_src1_आकार_per_thread;
    uint32_t  vtcm_dst_आकार_per_thread;

    uint32_t  src0_spad_half_आकार;
    uint32_t  src1_spad_half_आकार;
    uint32_t  dst_spad_half_आकार;

    uint32_t  src0_row_आकार_aligned;
    uint32_t  src1_row_आकार_aligned;
    uint32_t  dst_row_आकार_aligned;

    bool     use_f16;

    uint32_t n_शीर्ष;
    uint32_t n_शीर्ष_log2;

    भग्न    scale;
    भग्न    max_bias;
    भग्न    m0;
    भग्न    m1;

    काष्ठा fastdiv_मूल्यावली div_ne01;
    काष्ठा fastdiv_मूल्यावली div_ne02;
    काष्ठा fastdiv_मूल्यावली div_ne12;
    काष्ठा fastdiv_मूल्यावली div_ne13;

    uint32_t src0_nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;

    भग्न    slopes[512] __attribute__((aligned(128)));
};

प्रकार व्योम (*softmax_compute_fn_t)(
    व्योम * restrict dst,
    स्थिर व्योम * restrict src0,
    स्थिर व्योम * restrict mask,
    uint32_t ne00,
    भग्न scale,
    भग्न slope
);

अटल व्योम hvx_fast_softmax_prep_f16(स्थिर uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      स्थिर पूर्णांक अंक_elems,
                                      भग्न     scale,
                                      स्थिर uint8_t * restrict mask,
                                      भग्न slope) {
    स्थिर HVX_Vector * restrict v_src  = (स्थिर HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    स्थिर HVX_Vector * restrict v_mask = (स्थिर HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    स्थिर पूर्णांक nvec_64 = अंक_elems / VLEN_FP16;
    स्थिर पूर्णांक nloe_64 = अंक_elems % VLEN_FP16;

    #आशय unroll(2)
    क्रम (पूर्णांक i = 0; i < nvec_64; i++) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_mask[i]);
        HVX_Vector m0 = Q6_V_lo_W(p);
        HVX_Vector m1 = Q6_V_hi_W(p);

        HVX_Vector s0 = v_src[2 * i];
        HVX_Vector s1 = v_src[2 * i + 1];

        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));
        HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    अगर (nloe_64 > 0) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_mask[nvec_64]);
        HVX_Vector m0 = Q6_V_lo_W(p);

        HVX_Vector s0 = v_src[2 * nvec_64];
        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));

        अगर (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * माप(भग्न), Q6_Vsf_equals_Vqf32(v0));
        } अन्यथा {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_Vector m1 = Q6_V_hi_W(p);
            HVX_Vector s1 = v_src[2 * nvec_64 + 1];
            HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * माप(भग्न), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

अटल व्योम hvx_fast_softmax_prep_f32(स्थिर uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      स्थिर पूर्णांक अंक_elems,
                                      भग्न     scale,
                                      स्थिर uint8_t * restrict mask,
                                      भग्न slope) {
    स्थिर HVX_Vector * restrict v_src  = (स्थिर HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    स्थिर HVX_Vector * restrict v_mask = (स्थिर HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    स्थिर पूर्णांक nvec = अंक_elems / VLEN_FP32;
    स्थिर पूर्णांक nloe = अंक_elems % VLEN_FP32;

    #आशय unroll(4)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v3 = v_mask[i];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    अगर (nloe > 0) {
        HVX_Vector v1 = v_src[nvec];
        HVX_Vector v3 = v_mask[nvec];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * माप(भग्न), Q6_Vsf_equals_Vqf32(v5));
    }
}

अटल व्योम hvx_fast_softmax_f32(स्थिर uint8_t * restrict src, uint8_t * restrict dst, स्थिर पूर्णांक अंक_elems) {
    स्थिर HVX_Vector * restrict v_src = (स्थिर HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    स्थिर पूर्णांक nvec = अंक_elems / VLEN_FP32;
    स्थिर पूर्णांक nloe = अंक_elems % VLEN_FP32;

    HVX_Vector max_vec = hvx_vec_splat_f32(((स्थिर भग्न *) src)[0]);

    #आशय unroll(2)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    अगर (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * माप(भग्न));
        HVX_Vector neg_inf    = hvx_vec_splat_f32(-अनन्त);
        HVX_Vector v_पुच्छ     = Q6_V_vmux_QVV(q_mask, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_पुच्छ);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_Vector योग_vec = Q6_V_vsplat_R(0x00000000);

    #आशय unroll(2)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        योग_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(योग_vec), v3);

        v_dst[i] = v3;
    }

    अगर (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * माप(भग्न));
        HVX_Vector v1     = v_src[nvec];
        HVX_Vector v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_Vector v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        HVX_Vector v3_pad = Q6_V_vmux_QVV(q_mask, v3, Q6_V_vzero());

        योग_vec     = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(योग_vec), v3_pad);
        v_dst[nvec] = v3_pad;
    }

    योग_vec = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_vec));

    HVX_VectorPred pos_योग   = Q6_Q_vcmp_gt_VwVw(योग_vec, Q6_V_vzero());
    HVX_Vector     v4        = hvx_vec_inverse_f32(योग_vec);
    HVX_Vector     scale_vec = Q6_V_vmux_QVV(pos_योग, v4, hvx_vec_splat_f32(1.0f));

    #आशय unroll(2)
    क्रम (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_dst[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    अगर (nloe > 0) {
        HVX_Vector v1 = v_dst[nvec];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * माप(भग्न), Q6_Vsf_equals_Vqf32(v2));
    }
}

अटल व्योम compute_fast_softmax_f32_nomask(
    व्योम * restrict dst,
    स्थिर व्योम * restrict src0,
    स्थिर व्योम * restrict mask,
    uint32_t ne00,
    भग्न scale,
    भग्न slope
) {
    (व्योम) mask;
    (व्योम) slope;
    hvx_scale_f32((uint8_t *) dst, (स्थिर uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((स्थिर uint8_t *) dst, (uint8_t *) dst, ne00);
}

अटल व्योम compute_fast_softmax_f32_mask_f32(
    व्योम * restrict dst,
    स्थिर व्योम * restrict src0,
    स्थिर व्योम * restrict mask,
    uint32_t ne00,
    भग्न scale,
    भग्न slope
) {
    hvx_fast_softmax_prep_f32((स्थिर uint8_t *) src0, (uint8_t *) dst, ne00, scale, (स्थिर uint8_t *) mask, slope);
    hvx_fast_softmax_f32((स्थिर uint8_t *) dst, (uint8_t *) dst, ne00);
}

अटल व्योम compute_fast_softmax_f32_mask_f16(
    व्योम * restrict dst,
    स्थिर व्योम * restrict src0,
    स्थिर व्योम * restrict mask,
    uint32_t ne00,
    भग्न scale,
    भग्न slope
) {
    hvx_fast_softmax_prep_f16((स्थिर uint8_t *) src0, (uint8_t *) dst, ne00, scale, (स्थिर uint8_t *) mask, slope);
    hvx_fast_softmax_f32((स्थिर uint8_t *) dst, (uint8_t *) dst, ne00);
}

अटल स्थिर softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_गिनती] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_MASK_F32] = compute_fast_softmax_f32_mask_f32,
    [HTP_SOFTMAX_KERNEL_MASK_F16] = compute_fast_softmax_f32_mask_f16,
};

अटल व्योम softmax_thread_dma(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    स्थिर काष्ठा htp_softmax_context * smctx = (स्थिर काष्ठा htp_softmax_context *) तथ्य;
    काष्ठा htp_ops_context * octx = smctx->octx;
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर uint32_t src0_nrows            = smctx->nrows;
    स्थिर uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    स्थिर uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    स्थिर dma_addr_t तथ्य_src0 = smctx->तथ्य_src0;
    स्थिर dma_addr_t तथ्य_dst  = smctx->तथ्य_dst;

    स्थिर माप_प्रकार src0_row_आकार = src0->ne[0] * माप(भग्न);
    स्थिर माप_प्रकार dst_row_आकार  = src0->ne[0] * माप(भग्न);

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_आकार_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_आकार_per_thread);

    स्थिर माप_प्रकार src0_vtcm_half = smctx->src0_spad_half_आकार;
    स्थिर माप_प्रकार dst_vtcm_half  = smctx->dst_spad_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];

    क्रम (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = तथ्य_dst  + r * dst_row_आकार;
        dma_addr_t cur_src0 = तथ्य_src0 + r * src0_row_आकार;
        व्योम * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        व्योम * s_spad = src0_vtcm_base + idx * src0_vtcm_half;

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(cur_dst, d_spad),
                       dst_row_आकार, smctx->dst_row_आकार_aligned, dst_row_आकार, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s_spad, cur_src0),
                       smctx->src0_row_आकार_aligned, src0_row_आकार, src0_row_आकार, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    स्थिर uint32_t ne00 = src0->ne[0];

    क्रम (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        व्योम * d_spad = (व्योम *) dma_queue_निकालो(dma_q).src;
        व्योम * s_spad = (व्योम *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, शून्य, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = तथ्य_dst + r * dst_row_आकार;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(cur_dst, d_spad),
                       dst_row_आकार, smctx->dst_row_आकार_aligned, dst_row_आकार, 1);

        स्थिर uint32_t अग्र_r = r + 2;
        अगर (अग्र_r < src0_end_row) {
            dma_addr_t अग्र_src0 = तथ्य_src0 + अग्र_r * src0_row_आकार;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s_spad, अग्र_src0),
                           smctx->src0_row_आकार_aligned, src0_row_आकार, src0_row_आकार, 1);
        }
    }

    dma_queue_flush(dma_q);
}

अटल व्योम softmax_thread_mask_dma(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    स्थिर काष्ठा htp_softmax_context * smctx = (स्थिर काष्ठा htp_softmax_context *) तथ्य;
    काष्ठा htp_ops_context * octx = smctx->octx;
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर uint32_t src0_nrows            = smctx->nrows;
    स्थिर uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    स्थिर uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    स्थिर dma_addr_t तथ्य_src0 = smctx->तथ्य_src0;
    स्थिर dma_addr_t तथ्य_src1 = smctx->तथ्य_src1;
    स्थिर dma_addr_t तथ्य_dst  = smctx->तथ्य_dst;

    स्थिर माप_प्रकार src0_row_आकार = src0->ne[0] * माप(भग्न);
    स्थिर माप_प्रकार dst_row_आकार  = src0->ne[0] * माप(भग्न);
    स्थिर माप_प्रकार mask_row_आकार = smctx->use_f16 ? (src1->ne[0] * माप(__fp16)) : (src1->ne[0] * माप(भग्न));

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_आकार_per_thread);
    uint8_t * src1_vtcm_base = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_आकार_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_आकार_per_thread);

    स्थिर माप_प्रकार src0_vtcm_half = smctx->src0_spad_half_आकार;
    स्थिर माप_प्रकार src1_vtcm_half = smctx->src1_spad_half_आकार;
    स्थिर माप_प्रकार dst_vtcm_half  = smctx->dst_spad_half_आकार;

    स्थिर uint32_t nb11 = src1->nb[1];
    स्थिर uint32_t nb12 = src1->nb[2];
    स्थिर uint32_t nb13 = src1->nb[3];

    स्थिर uint32_t ne00 = src0->ne[0];
    स्थिर uint32_t ne01 = src0->ne[1];
    स्थिर uint32_t ne02 = src0->ne[2];
    स्थिर uint32_t ne03 = src0->ne[3];
    स्थिर uint32_t ne12 = src1->ne[2];
    स्थिर uint32_t ne13 = src1->ne[3];

    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne01 = &smctx->div_ne01;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne02 = &smctx->div_ne02;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne12 = &smctx->div_ne12;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne13 = &smctx->div_ne13;

    dma_queue * dma_q = octx->ctx->dma[ith];

    क्रम (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = तथ्य_dst  + r * dst_row_आकार;
        dma_addr_t cur_src0 = तथ्य_src0 + r * src0_row_आकार;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = तथ्य_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        व्योम * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        व्योम * s_spad = src0_vtcm_base + idx * src0_vtcm_half;
        व्योम * m_spad = src1_vtcm_base + idx * src1_vtcm_half;

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(cur_dst, d_spad),
                       dst_row_आकार, smctx->dst_row_आकार_aligned, dst_row_आकार, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s_spad, cur_src0),
                       smctx->src0_row_आकार_aligned, src0_row_आकार, src0_row_आकार, 1);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(m_spad, cur_src1),
                       smctx->src1_row_आकार_aligned, mask_row_आकार, mask_row_आकार, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    स्थिर bool has_bias = smctx->max_bias > 0.0f;
    uint32_t पूर्व_i2 = (uint32_t)-1;
    भग्न slope = 1.0f;

    क्रम (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        व्योम * d_spad = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).src;
        व्योम * s_spad = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).dst;
        व्योम * m_spad = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).dst;

        अगर (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            अगर (i2 != पूर्व_i2) {
                slope = smctx->slopes[i2];
                पूर्व_i2 = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, slope);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = तथ्य_dst + r * dst_row_आकार;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(cur_dst, d_spad),
                       dst_row_आकार, smctx->dst_row_आकार_aligned, dst_row_आकार, 1);

        स्थिर uint32_t अग्र_r = r + 2;
        अगर (अग्र_r < src0_end_row) {
            dma_addr_t अग्र_src0 = तथ्य_src0 + अग्र_r * src0_row_आकार;

            uint32_t ni1 = fastmodulo(अग्र_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(अग्र_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t अग्र_src1 = तथ्य_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s_spad, अग्र_src0),
                           smctx->src0_row_आकार_aligned, src0_row_आकार, src0_row_आकार, 1);
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(m_spad, अग्र_src1),
                           smctx->src1_row_आकार_aligned, mask_row_आकार, mask_row_आकार, 1);
        }
    }

    dma_queue_flush(dma_q);
}

अटल पूर्णांक execute_op_softmax_f32(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    स्थिर अक्षर * op_type = "softmax-f32";

    स्थिर काष्ठा htp_softmax_kernel_params * kparams =
        (स्थिर काष्ठा htp_softmax_kernel_params *) octx->kernel_params;

    अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    अगर (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_गिनती) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    अगर (octx->ctx->vtcm_आकार < (माप_प्रकार) kparams->vtcm_आकार) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_आकार, kparams->vtcm_आकार);
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    स्थिर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर माप_प्रकार elem_आकार    = माप(भग्न);
    स्थिर माप_प्रकार dst_row_आकार = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_आकार, (uint32_t) dst_row_आकार, &rows_per_chunk);
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
        अगर (nrows < octx->n_threads) {
            htp_ops_context_set_n_threads(octx, nrows ? nrows : 1);
        }
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t n_threads = octx->n_threads;
    uint8_t * स्थिर vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    स्थिर uint32_t off_src0 = 0;
    स्थिर uint32_t off_dst  = off_src0 + kparams->vtcm_src0_आकार_per_thread * kparams->n_threads;
    स्थिर uint32_t off_src1 = off_dst  + kparams->vtcm_dst_आकार_per_thread  * kparams->n_threads;

    काष्ठा htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (व्योम *) softmax_kernels[kparams->kernel_id],

        .तथ्य_src0             = src0->तथ्य,
        .तथ्य_src1             = kparams->use_src1 ? octx->src[1]->तथ्य : 0,
        .तथ्य_dst              = dst->तथ्य,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_base, off_src1, kparams->use_src1),

        .vtcm_src0_आकार_per_thread = kparams->vtcm_src0_आकार_per_thread,
        .vtcm_src1_आकार_per_thread = kparams->vtcm_src1_आकार_per_thread,
        .vtcm_dst_आकार_per_thread  = kparams->vtcm_dst_आकार_per_thread,

        .src0_spad_half_आकार   = kparams->src0_spad_half_आकार,
        .src1_spad_half_आकार   = kparams->src1_spad_half_आकार,
        .dst_spad_half_आकार    = kparams->dst_spad_half_आकार,

        .src0_row_आकार_aligned = kparams->src0_row_आकार_aligned,
        .src1_row_आकार_aligned = kparams->src1_row_आकार_aligned,
        .dst_row_आकार_aligned  = kparams->dst_row_आकार_aligned,

        .use_f16               = kparams->use_f16 != 0,

        .n_शीर्ष                = kparams->n_शीर्ष,
        .n_शीर्ष_log2           = kparams->n_शीर्ष_log2,

        .scale                 = kparams->scale,
        .max_bias              = kparams->max_bias,
        .m0                    = kparams->m0,
        .m1                    = kparams->m1,

        .div_ne01              = kparams->div_ne01,
        .div_ne02              = kparams->div_ne02,
        .div_ne12              = kparams->div_ne12,
        .div_ne13              = kparams->div_ne13,

        .src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .row_start             = row_start,
        .nrows                 = nrows,
    };

    अगर (kparams->max_bias > 0.0f && kparams->use_src1) {
        अगर (kparams->n_शीर्ष > 512) {
            वापस HTP_STATUS_INVAL_PARAMS;
        }
        क्रम (uint32_t h = 0; h < kparams->n_शीर्ष; h += 32) {
            HVX_Vector v_slopes = hvx_alibi_slopes(h, 1, kparams->n_शीर्ष_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_queue_func_t task_func = kparams->use_src1 ? softmax_thread_mask_dma : softmax_thread_dma;
    work_queue_run(octx->ctx->work_queue, task_func, &smctx, n_threads);

    वापस HTP_STATUS_OK;
}

पूर्णांक op_softmax(काष्ठा htp_ops_context * octx) {
    पूर्णांक err = HTP_STATUS_OK;

    चयन (octx->src[0]->type) {
        हाल HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            अवरोध;

        शेष:
            err = HTP_STATUS_NO_SUPPORT;
            अवरोध;
    }

    वापस err;
}
