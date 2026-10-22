#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "work-queue.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "htp-vtcm.h"
#अन्तर्भाव "htp/softmax-ops.h"
#अन्तर्भाव "hvx-flash-attn.h"

संरचना htp_softmax_context {
    संरचना htp_ops_context * octx;
    स्थिरांक संरचना htp_softmax_kernel_params * kparams;

    शून्य *   compute;

    dma_addr_t data_src0;
    dma_addr_t data_src1;
    dma_addr_t data_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_size_per_thread;
    uint32_t  vtcm_src1_size_per_thread;
    uint32_t  vtcm_dst_size_per_thread;

    uint32_t  src0_spad_half_size;
    uint32_t  src1_spad_half_size;
    uint32_t  dst_spad_half_size;

    uint32_t  src0_row_size_aligned;
    uint32_t  src1_row_size_aligned;
    uint32_t  dst_row_size_aligned;

    bool     use_f16;

    uint32_t n_head;
    uint32_t n_head_log2;

    दशांश    scale;
    दशांश    max_bias;
    दशांश    m0;
    दशांश    m1;

    संरचना fastdiv_मूल्ये div_ne01;
    संरचना fastdiv_मूल्ये div_ne02;
    संरचना fastdiv_मूल्ये div_ne12;
    संरचना fastdiv_मूल्ये div_ne13;

    uint32_t src0_nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;

    दशांश    slopes[512] __attribute__((aligned(128)));
};

प्रकारव्याख्या शून्य (*softmax_compute_fn_t)(
    शून्य * restrict dst,
    स्थिरांक शून्य * restrict src0,
    स्थिरांक शून्य * restrict mask,
    uint32_t ne00,
    दशांश scale,
    दशांश slope
);

स्थिर शून्य hvx_fast_softmax_prep_f16(स्थिरांक uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      स्थिरांक पूर्णांक अंक_elems,
                                      दशांश     scale,
                                      स्थिरांक uint8_t * restrict mask,
                                      दशांश slope) {
    स्थिरांक HVX_Vector * restrict v_src  = (स्थिरांक HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    स्थिरांक HVX_Vector * restrict v_mask = (स्थिरांक HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    स्थिरांक पूर्णांक nvec_64 = अंक_elems / VLEN_FP16;
    स्थिरांक पूर्णांक nloe_64 = अंक_elems % VLEN_FP16;

    #pragma unroll(2)
    साठी (पूर्णांक i = 0; i < nvec_64; i++) {
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

    जर (nloe_64 > 0) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_mask[nvec_64]);
        HVX_Vector m0 = Q6_V_lo_W(p);

        HVX_Vector s0 = v_src[2 * nvec_64];
        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));

        जर (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * आकार(दशांश), Q6_Vsf_equals_Vqf32(v0));
        } अन्यथा {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_Vector m1 = Q6_V_hi_W(p);
            HVX_Vector s1 = v_src[2 * nvec_64 + 1];
            HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * आकार(दशांश), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

स्थिर शून्य hvx_fast_softmax_prep_f32(स्थिरांक uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      स्थिरांक पूर्णांक अंक_elems,
                                      दशांश     scale,
                                      स्थिरांक uint8_t * restrict mask,
                                      दशांश slope) {
    स्थिरांक HVX_Vector * restrict v_src  = (स्थिरांक HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    स्थिरांक HVX_Vector * restrict v_mask = (स्थिरांक HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    स्थिरांक पूर्णांक nvec = अंक_elems / VLEN_FP32;
    स्थिरांक पूर्णांक nloe = अंक_elems % VLEN_FP32;

    #pragma unroll(4)
    साठी (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v3 = v_mask[i];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    जर (nloe > 0) {
        HVX_Vector v1 = v_src[nvec];
        HVX_Vector v3 = v_mask[nvec];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * आकार(दशांश), Q6_Vsf_equals_Vqf32(v5));
    }
}

स्थिर शून्य hvx_fast_softmax_f32(स्थिरांक uint8_t * restrict src, uint8_t * restrict dst, स्थिरांक पूर्णांक अंक_elems) {
    स्थिरांक HVX_Vector * restrict v_src = (स्थिरांक HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    स्थिरांक पूर्णांक nvec = अंक_elems / VLEN_FP32;
    स्थिरांक पूर्णांक nloe = अंक_elems % VLEN_FP32;

    HVX_Vector max_vec = hvx_vec_splat_f32(((स्थिरांक दशांश *) src)[0]);

    #pragma unroll(2)
    साठी (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    जर (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * आकार(दशांश));
        HVX_Vector neg_inf    = hvx_vec_splat_f32(-अनन्त);
        HVX_Vector v_tail     = Q6_V_vmux_QVV(q_mask, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_tail);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_Vector sum_vec = Q6_V_vsplat_R(0x00000000);

    #pragma unroll(2)
    साठी (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3);

        v_dst[i] = v3;
    }

    जर (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * आकार(दशांश));
        HVX_Vector v1     = v_src[nvec];
        HVX_Vector v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_Vector v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        HVX_Vector v3_pad = Q6_V_vmux_QVV(q_mask, v3, Q6_V_vzero());

        sum_vec     = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3_pad);
        v_dst[nvec] = v3_pad;
    }

    sum_vec = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_vec));

    HVX_VectorPred pos_sum   = Q6_Q_vcmp_gt_VwVw(sum_vec, Q6_V_vzero());
    HVX_Vector     v4        = hvx_vec_inverse_f32(sum_vec);
    HVX_Vector     scale_vec = Q6_V_vmux_QVV(pos_sum, v4, hvx_vec_splat_f32(1.0f));

    #pragma unroll(2)
    साठी (पूर्णांक i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_dst[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    जर (nloe > 0) {
        HVX_Vector v1 = v_dst[nvec];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * आकार(दशांश), Q6_Vsf_equals_Vqf32(v2));
    }
}

स्थिर शून्य compute_fast_softmax_f32_nomask(
    शून्य * restrict dst,
    स्थिरांक शून्य * restrict src0,
    स्थिरांक शून्य * restrict mask,
    uint32_t ne00,
    दशांश scale,
    दशांश slope
) {
    (शून्य) mask;
    (शून्य) slope;
    hvx_scale_f32((uint8_t *) dst, (स्थिरांक uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((स्थिरांक uint8_t *) dst, (uint8_t *) dst, ne00);
}

स्थिर शून्य compute_fast_softmax_f32_mask_f32(
    शून्य * restrict dst,
    स्थिरांक शून्य * restrict src0,
    स्थिरांक शून्य * restrict mask,
    uint32_t ne00,
    दशांश scale,
    दशांश slope
) {
    hvx_fast_softmax_prep_f32((स्थिरांक uint8_t *) src0, (uint8_t *) dst, ne00, scale, (स्थिरांक uint8_t *) mask, slope);
    hvx_fast_softmax_f32((स्थिरांक uint8_t *) dst, (uint8_t *) dst, ne00);
}

स्थिर शून्य compute_fast_softmax_f32_mask_f16(
    शून्य * restrict dst,
    स्थिरांक शून्य * restrict src0,
    स्थिरांक शून्य * restrict mask,
    uint32_t ne00,
    दशांश scale,
    दशांश slope
) {
    hvx_fast_softmax_prep_f16((स्थिरांक uint8_t *) src0, (uint8_t *) dst, ne00, scale, (स्थिरांक uint8_t *) mask, slope);
    hvx_fast_softmax_f32((स्थिरांक uint8_t *) dst, (uint8_t *) dst, ne00);
}

स्थिर स्थिरांक softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_मोजणी] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_MASK_F32] = compute_fast_softmax_f32_mask_f32,
    [HTP_SOFTMAX_KERNEL_MASK_F16] = compute_fast_softmax_f32_mask_f16,
};

स्थिर शून्य softmax_thread_dma(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य * data) {
    (शून्य) nth;
    स्थिरांक संरचना htp_softmax_context * smctx = (स्थिरांक संरचना htp_softmax_context *) data;
    संरचना htp_ops_context * octx = smctx->octx;
    स्थिरांक संरचना htp_tensor * src0 = octx->src[0];
    स्थिरांक संरचना htp_tensor * dst  = octx->dst;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिरांक uint32_t src0_nrows            = smctx->nrows;
    स्थिरांक uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    स्थिरांक uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    स्थिरांक uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    जर (src0_start_row >= src0_end_row) {
        परत;
    }

    स्थिरांक dma_addr_t data_src0 = smctx->data_src0;
    स्थिरांक dma_addr_t data_dst  = smctx->data_dst;

    स्थिरांक माप_प्रकार src0_row_size = src0->ne[0] * आकार(दशांश);
    स्थिरांक माप_प्रकार dst_row_size  = src0->ne[0] * आकार(दशांश);

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    स्थिरांक माप_प्रकार src0_vtcm_half = smctx->src0_spad_half_size;
    स्थिरांक माप_प्रकार dst_vtcm_half  = smctx->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    साठी (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;
        शून्य * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        शून्य * s_spad = src0_vtcm_base + idx * src0_vtcm_half;

        dma_queue_ढकला(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_ढकला(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    स्थिरांक uint32_t ne00 = src0->ne[0];

    साठी (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        शून्य * d_spad = (शून्य *) dma_queue_काढा(dma_q).src;
        शून्य * s_spad = (शून्य *) dma_queue_काढा(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, रिक्त, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_ढकला(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        स्थिरांक uint32_t next_r = r + 2;
        जर (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;
            dma_queue_ढकला(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

स्थिर शून्य softmax_thread_mask_dma(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य * data) {
    (शून्य) nth;
    स्थिरांक संरचना htp_softmax_context * smctx = (स्थिरांक संरचना htp_softmax_context *) data;
    संरचना htp_ops_context * octx = smctx->octx;
    स्थिरांक संरचना htp_tensor * src0 = octx->src[0];
    स्थिरांक संरचना htp_tensor * src1 = octx->src[1];
    स्थिरांक संरचना htp_tensor * dst  = octx->dst;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिरांक uint32_t src0_nrows            = smctx->nrows;
    स्थिरांक uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    स्थिरांक uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    स्थिरांक uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    जर (src0_start_row >= src0_end_row) {
        परत;
    }

    स्थिरांक dma_addr_t data_src0 = smctx->data_src0;
    स्थिरांक dma_addr_t data_src1 = smctx->data_src1;
    स्थिरांक dma_addr_t data_dst  = smctx->data_dst;

    स्थिरांक माप_प्रकार src0_row_size = src0->ne[0] * आकार(दशांश);
    स्थिरांक माप_प्रकार dst_row_size  = src0->ne[0] * आकार(दशांश);
    स्थिरांक माप_प्रकार mask_row_size = smctx->use_f16 ? (src1->ne[0] * आकार(__fp16)) : (src1->ne[0] * आकार(दशांश));

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_base = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    स्थिरांक माप_प्रकार src0_vtcm_half = smctx->src0_spad_half_size;
    स्थिरांक माप_प्रकार src1_vtcm_half = smctx->src1_spad_half_size;
    स्थिरांक माप_प्रकार dst_vtcm_half  = smctx->dst_spad_half_size;

    स्थिरांक uint32_t nb11 = src1->nb[1];
    स्थिरांक uint32_t nb12 = src1->nb[2];
    स्थिरांक uint32_t nb13 = src1->nb[3];

    स्थिरांक uint32_t ne00 = src0->ne[0];
    स्थिरांक uint32_t ne01 = src0->ne[1];
    स्थिरांक uint32_t ne02 = src0->ne[2];
    स्थिरांक uint32_t ne03 = src0->ne[3];
    स्थिरांक uint32_t ne12 = src1->ne[2];
    स्थिरांक uint32_t ne13 = src1->ne[3];

    स्थिरांक संरचना fastdiv_मूल्ये * div_ne01 = &smctx->div_ne01;
    स्थिरांक संरचना fastdiv_मूल्ये * div_ne02 = &smctx->div_ne02;
    स्थिरांक संरचना fastdiv_मूल्ये * div_ne12 = &smctx->div_ne12;
    स्थिरांक संरचना fastdiv_मूल्ये * div_ne13 = &smctx->div_ne13;

    dma_queue * dma_q = octx->ctx->dma[ith];

    साठी (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = data_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        शून्य * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        शून्य * s_spad = src0_vtcm_base + idx * src0_vtcm_half;
        शून्य * m_spad = src1_vtcm_base + idx * src1_vtcm_half;

        dma_queue_ढकला(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_ढकला(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        dma_queue_ढकला(dma_q, dma_make_data(m_spad, cur_src1),
                       smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    स्थिरांक bool has_bias = smctx->max_bias > 0.0f;
    uint32_t prev_i2 = (uint32_t)-1;
    दशांश slope = 1.0f;

    साठी (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        शून्य * d_spad = (शून्य *) (uintptr_t) dma_queue_काढा(dma_q).src;
        शून्य * s_spad = (शून्य *) (uintptr_t) dma_queue_काढा(dma_q).dst;
        शून्य * m_spad = (शून्य *) (uintptr_t) dma_queue_काढा(dma_q).dst;

        जर (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            जर (i2 != prev_i2) {
                slope = smctx->slopes[i2];
                prev_i2 = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, slope);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_ढकला(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        स्थिरांक uint32_t next_r = r + 2;
        जर (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;

            uint32_t ni1 = fastmodulo(next_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(next_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t next_src1 = data_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_queue_ढकला(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
            dma_queue_ढकला(dma_q, dma_make_data(m_spad, next_src1),
                           smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

स्थिर पूर्णांक execute_op_softmax_f32(संरचना htp_ops_context * octx) {
    स्थिरांक संरचना htp_tensor * src0 = octx->src[0];
    स्थिरांक संरचना htp_tensor * dst  = octx->dst;

    स्थिरांक अक्षर * op_type = "softmax-f32";

    स्थिरांक संरचना htp_softmax_kernel_params * kparams =
        (स्थिरांक संरचना htp_softmax_kernel_params *) octx->kernel_params;

    जर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        परत HTP_STATUS_INVAL_PARAMS;
    }

    जर (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_मोजणी) {
        परत HTP_STATUS_INVAL_PARAMS;
    }

    जर (octx->ctx->vtcm_size < (माप_प्रकार) kparams->vtcm_size) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_size, kparams->vtcm_size);
        परत HTP_STATUS_VTCM_TOO_SMALL;
    }

    स्थिरांक uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिरांक माप_प्रकार elem_size    = आकार(दशांश);
    स्थिरांक माप_प्रकार dst_row_size = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    जर (octx->ctx->mdev.मोजणी > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        स्थिरांक संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.मोजणी, &octx->ctx->mdev.मोजणी_div);
        row_start = range.start;
        nrows     = range.मोजणी;
        जर (nrows < octx->n_threads) {
            htp_ops_context_set_n_threads(octx, nrows ? nrows : 1);
        }
    }

    जर (nrows == 0) {
        परत HTP_STATUS_OK;
    }

    स्थिरांक uint32_t n_threads = octx->n_threads;
    uint8_t * स्थिरांक vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    स्थिरांक uint32_t off_src0 = 0;
    स्थिरांक uint32_t off_dst  = off_src0 + kparams->vtcm_src0_size_per_thread * kparams->n_threads;
    स्थिरांक uint32_t off_src1 = off_dst  + kparams->vtcm_dst_size_per_thread  * kparams->n_threads;

    संरचना htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (शून्य *) softmax_kernels[kparams->kernel_id],

        .data_src0             = src0->data,
        .data_src1             = kparams->use_src1 ? octx->src[1]->data : 0,
        .data_dst              = dst->data,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_base, off_src1, kparams->use_src1),

        .vtcm_src0_size_per_thread = kparams->vtcm_src0_size_per_thread,
        .vtcm_src1_size_per_thread = kparams->vtcm_src1_size_per_thread,
        .vtcm_dst_size_per_thread  = kparams->vtcm_dst_size_per_thread,

        .src0_spad_half_size   = kparams->src0_spad_half_size,
        .src1_spad_half_size   = kparams->src1_spad_half_size,
        .dst_spad_half_size    = kparams->dst_spad_half_size,

        .src0_row_size_aligned = kparams->src0_row_size_aligned,
        .src1_row_size_aligned = kparams->src1_row_size_aligned,
        .dst_row_size_aligned  = kparams->dst_row_size_aligned,

        .use_f16               = kparams->use_f16 != 0,

        .n_head                = kparams->n_head,
        .n_head_log2           = kparams->n_head_log2,

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

    जर (kparams->max_bias > 0.0f && kparams->use_src1) {
        जर (kparams->n_head > 512) {
            परत HTP_STATUS_INVAL_PARAMS;
        }
        साठी (uint32_t h = 0; h < kparams->n_head; h += 32) {
            HVX_Vector v_slopes = hvx_alibi_slopes(h, 1, kparams->n_head_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_queue_func_t task_func = kparams->use_src1 ? softmax_thread_mask_dma : softmax_thread_dma;
    work_queue_run(octx->ctx->work_queue, task_func, &smctx, n_threads);

    परत HTP_STATUS_OK;
}

पूर्णांक op_softmax(संरचना htp_ops_context * octx) {
    पूर्णांक err = HTP_STATUS_OK;

    निवड (octx->src[0]->type) {
        स्थिती HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            थांबा;

        पूर्वनिर्धारित:
            err = HTP_STATUS_NO_SUPPORT;
            थांबा;
    }

    परत err;
}
