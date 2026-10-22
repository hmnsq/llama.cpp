#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <HAP_perf.h>

#அந்தர்பாவ <math.h>
#அந்தர்பாவ <string.h>

#அந்தர்பாவ "dma-queue.h"
#அந்தர்பாவ "work-queue.h"
#அந்தர்பாவ "hvx-utils.h"
#அந்தர்பாவ "hex-fastdiv.h"
#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-profile.h"

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"
#அந்தர்பாவ "htp-vtcm.h"
#அந்தர்பாவ "htp/softmax-ops.h"
#அந்தர்பாவ "hvx-flash-attn.h"

கட்டமைப்பு htp_softmax_context {
    கட்டமைப்பு htp_ops_context * octx;
    மாறிலி கட்டமைப்பு htp_softmax_kernel_params * kparams;

    வெறுமை *   compute;

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

    தசம    scale;
    தசம    max_bias;
    தசம    m0;
    தசம    m1;

    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne01;
    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne02;
    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne12;
    கட்டமைப்பு fastdiv_மதிப்புகள் div_ne13;

    uint32_t src0_nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;

    தசம    slopes[512] __attribute__((aligned(128)));
};

வகைவரை வெறுமை (*softmax_compute_fn_t)(
    வெறுமை * restrict dst,
    மாறிலி வெறுமை * restrict src0,
    மாறிலி வெறுமை * restrict mask,
    uint32_t ne00,
    தசம scale,
    தசம slope
);

நிலையான வெறுமை hvx_fast_softmax_prep_f16(மாறிலி uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      மாறிலி முழு இலக்கம்_elems,
                                      தசம     scale,
                                      மாறிலி uint8_t * restrict mask,
                                      தசம slope) {
    மாறிலி HVX_Vector * restrict v_src  = (மாறிலி HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    மாறிலி HVX_Vector * restrict v_mask = (மாறிலி HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    மாறிலி முழு nvec_64 = இலக்கம்_elems / VLEN_FP16;
    மாறிலி முழு nloe_64 = இலக்கம்_elems % VLEN_FP16;

    #pragma unroll(2)
    க்கு (முழு i = 0; i < nvec_64; i++) {
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

    எனில் (nloe_64 > 0) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_mask[nvec_64]);
        HVX_Vector m0 = Q6_V_lo_W(p);

        HVX_Vector s0 = v_src[2 * nvec_64];
        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));

        எனில் (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * அளவு(தசம), Q6_Vsf_equals_Vqf32(v0));
        } இல்லையெனில் {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_Vector m1 = Q6_V_hi_W(p);
            HVX_Vector s1 = v_src[2 * nvec_64 + 1];
            HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * அளவு(தசம), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

நிலையான வெறுமை hvx_fast_softmax_prep_f32(மாறிலி uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      மாறிலி முழு இலக்கம்_elems,
                                      தசம     scale,
                                      மாறிலி uint8_t * restrict mask,
                                      தசம slope) {
    மாறிலி HVX_Vector * restrict v_src  = (மாறிலி HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    மாறிலி HVX_Vector * restrict v_mask = (மாறிலி HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    மாறிலி முழு nvec = இலக்கம்_elems / VLEN_FP32;
    மாறிலி முழு nloe = இலக்கம்_elems % VLEN_FP32;

    #pragma unroll(4)
    க்கு (முழு i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v3 = v_mask[i];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    எனில் (nloe > 0) {
        HVX_Vector v1 = v_src[nvec];
        HVX_Vector v3 = v_mask[nvec];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * அளவு(தசம), Q6_Vsf_equals_Vqf32(v5));
    }
}

நிலையான வெறுமை hvx_fast_softmax_f32(மாறிலி uint8_t * restrict src, uint8_t * restrict dst, மாறிலி முழு இலக்கம்_elems) {
    மாறிலி HVX_Vector * restrict v_src = (மாறிலி HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    மாறிலி முழு nvec = இலக்கம்_elems / VLEN_FP32;
    மாறிலி முழு nloe = இலக்கம்_elems % VLEN_FP32;

    HVX_Vector max_vec = hvx_vec_splat_f32(((மாறிலி தசம *) src)[0]);

    #pragma unroll(2)
    க்கு (முழு i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    எனில் (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * அளவு(தசம));
        HVX_Vector neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_Vector v_tail     = Q6_V_vmux_QVV(q_mask, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_tail);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_Vector sum_vec = Q6_V_vsplat_R(0x00000000);

    #pragma unroll(2)
    க்கு (முழு i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3);

        v_dst[i] = v3;
    }

    எனில் (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * அளவு(தசம));
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
    க்கு (முழு i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_dst[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    எனில் (nloe > 0) {
        HVX_Vector v1 = v_dst[nvec];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * அளவு(தசம), Q6_Vsf_equals_Vqf32(v2));
    }
}

நிலையான வெறுமை compute_fast_softmax_f32_nomask(
    வெறுமை * restrict dst,
    மாறிலி வெறுமை * restrict src0,
    மாறிலி வெறுமை * restrict mask,
    uint32_t ne00,
    தசம scale,
    தசம slope
) {
    (வெறுமை) mask;
    (வெறுமை) slope;
    hvx_scale_f32((uint8_t *) dst, (மாறிலி uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((மாறிலி uint8_t *) dst, (uint8_t *) dst, ne00);
}

நிலையான வெறுமை compute_fast_softmax_f32_mask_f32(
    வெறுமை * restrict dst,
    மாறிலி வெறுமை * restrict src0,
    மாறிலி வெறுமை * restrict mask,
    uint32_t ne00,
    தசம scale,
    தசம slope
) {
    hvx_fast_softmax_prep_f32((மாறிலி uint8_t *) src0, (uint8_t *) dst, ne00, scale, (மாறிலி uint8_t *) mask, slope);
    hvx_fast_softmax_f32((மாறிலி uint8_t *) dst, (uint8_t *) dst, ne00);
}

நிலையான வெறுமை compute_fast_softmax_f32_mask_f16(
    வெறுமை * restrict dst,
    மாறிலி வெறுமை * restrict src0,
    மாறிலி வெறுமை * restrict mask,
    uint32_t ne00,
    தசம scale,
    தசம slope
) {
    hvx_fast_softmax_prep_f16((மாறிலி uint8_t *) src0, (uint8_t *) dst, ne00, scale, (மாறிலி uint8_t *) mask, slope);
    hvx_fast_softmax_f32((மாறிலி uint8_t *) dst, (uint8_t *) dst, ne00);
}

நிலையான மாறிலி softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_மொத்தம்] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_MASK_F32] = compute_fast_softmax_f32_mask_f32,
    [HTP_SOFTMAX_KERNEL_MASK_F16] = compute_fast_softmax_f32_mask_f16,
};

நிலையான வெறுமை softmax_thread_dma(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    (வெறுமை) nth;
    மாறிலி கட்டமைப்பு htp_softmax_context * smctx = (மாறிலி கட்டமைப்பு htp_softmax_context *) data;
    கட்டமைப்பு htp_ops_context * octx = smctx->octx;
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;
    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];

    மாறிலி uint32_t src0_nrows            = smctx->nrows;
    மாறிலி uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    மாறிலி uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    மாறிலி uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    எனில் (src0_start_row >= src0_end_row) {
        திரும்பு;
    }

    மாறிலி dma_addr_t data_src0 = smctx->data_src0;
    மாறிலி dma_addr_t data_dst  = smctx->data_dst;

    மாறிலி size_t src0_row_size = src0->ne[0] * அளவு(தசம);
    மாறிலி size_t dst_row_size  = src0->ne[0] * அளவு(தசம);

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    மாறிலி size_t src0_vtcm_half = smctx->src0_spad_half_size;
    மாறிலி size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    க்கு (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;
        வெறுமை * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        வெறுமை * s_spad = src0_vtcm_base + idx * src0_vtcm_half;

        dma_queue_தள்ளு(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_தள்ளு(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    மாறிலி uint32_t ne00 = src0->ne[0];

    க்கு (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        வெறுமை * d_spad = (வெறுமை *) dma_queue_எடு(dma_q).src;
        வெறுமை * s_spad = (வெறுமை *) dma_queue_எடு(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, வெற்று, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_தள்ளு(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        மாறிலி uint32_t next_r = r + 2;
        எனில் (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;
            dma_queue_தள்ளு(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

நிலையான வெறுமை softmax_thread_mask_dma(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    (வெறுமை) nth;
    மாறிலி கட்டமைப்பு htp_softmax_context * smctx = (மாறிலி கட்டமைப்பு htp_softmax_context *) data;
    கட்டமைப்பு htp_ops_context * octx = smctx->octx;
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;
    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];

    மாறிலி uint32_t src0_nrows            = smctx->nrows;
    மாறிலி uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    மாறிலி uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    மாறிலி uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    எனில் (src0_start_row >= src0_end_row) {
        திரும்பு;
    }

    மாறிலி dma_addr_t data_src0 = smctx->data_src0;
    மாறிலி dma_addr_t data_src1 = smctx->data_src1;
    மாறிலி dma_addr_t data_dst  = smctx->data_dst;

    மாறிலி size_t src0_row_size = src0->ne[0] * அளவு(தசம);
    மாறிலி size_t dst_row_size  = src0->ne[0] * அளவு(தசம);
    மாறிலி size_t mask_row_size = smctx->use_f16 ? (src1->ne[0] * அளவு(__fp16)) : (src1->ne[0] * அளவு(தசம));

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_base = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    மாறிலி size_t src0_vtcm_half = smctx->src0_spad_half_size;
    மாறிலி size_t src1_vtcm_half = smctx->src1_spad_half_size;
    மாறிலி size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    மாறிலி uint32_t nb11 = src1->nb[1];
    மாறிலி uint32_t nb12 = src1->nb[2];
    மாறிலி uint32_t nb13 = src1->nb[3];

    மாறிலி uint32_t ne00 = src0->ne[0];
    மாறிலி uint32_t ne01 = src0->ne[1];
    மாறிலி uint32_t ne02 = src0->ne[2];
    மாறிலி uint32_t ne03 = src0->ne[3];
    மாறிலி uint32_t ne12 = src1->ne[2];
    மாறிலி uint32_t ne13 = src1->ne[3];

    மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div_ne01 = &smctx->div_ne01;
    மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div_ne02 = &smctx->div_ne02;
    மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div_ne12 = &smctx->div_ne12;
    மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div_ne13 = &smctx->div_ne13;

    dma_queue * dma_q = octx->ctx->dma[ith];

    க்கு (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = data_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        வெறுமை * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        வெறுமை * s_spad = src0_vtcm_base + idx * src0_vtcm_half;
        வெறுமை * m_spad = src1_vtcm_base + idx * src1_vtcm_half;

        dma_queue_தள்ளு(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_தள்ளு(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        dma_queue_தள்ளு(dma_q, dma_make_data(m_spad, cur_src1),
                       smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    மாறிலி bool has_bias = smctx->max_bias > 0.0f;
    uint32_t prev_i2 = (uint32_t)-1;
    தசம slope = 1.0f;

    க்கு (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        வெறுமை * d_spad = (வெறுமை *) (uintptr_t) dma_queue_எடு(dma_q).src;
        வெறுமை * s_spad = (வெறுமை *) (uintptr_t) dma_queue_எடு(dma_q).dst;
        வெறுமை * m_spad = (வெறுமை *) (uintptr_t) dma_queue_எடு(dma_q).dst;

        எனில் (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            எனில் (i2 != prev_i2) {
                slope = smctx->slopes[i2];
                prev_i2 = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, slope);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_தள்ளு(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        மாறிலி uint32_t next_r = r + 2;
        எனில் (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;

            uint32_t ni1 = fastmodulo(next_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(next_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t next_src1 = data_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_queue_தள்ளு(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
            dma_queue_தள்ளு(dma_q, dma_make_data(m_spad, next_src1),
                           smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

நிலையான முழு execute_op_softmax_f32(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    மாறிலி எழுத்து * op_type = "softmax-f32";

    மாறிலி கட்டமைப்பு htp_softmax_kernel_params * kparams =
        (மாறிலி கட்டமைப்பு htp_softmax_kernel_params *) octx->kernel_params;

    எனில் (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }

    எனில் (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_மொத்தம்) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }

    எனில் (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_size, kparams->vtcm_size);
        திரும்பு HTP_STATUS_VTCM_TOO_SMALL;
    }

    மாறிலி uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி size_t elem_size    = அளவு(தசம);
    மாறிலி size_t dst_row_size = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        row_start = range.start;
        nrows     = range.மொத்தம்;
        எனில் (nrows < octx->n_threads) {
            htp_ops_context_set_n_threads(octx, nrows ? nrows : 1);
        }
    }

    எனில் (nrows == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    மாறிலி uint32_t n_threads = octx->n_threads;
    uint8_t * மாறிலி vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    மாறிலி uint32_t off_src0 = 0;
    மாறிலி uint32_t off_dst  = off_src0 + kparams->vtcm_src0_size_per_thread * kparams->n_threads;
    மாறிலி uint32_t off_src1 = off_dst  + kparams->vtcm_dst_size_per_thread  * kparams->n_threads;

    கட்டமைப்பு htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (வெறுமை *) softmax_kernels[kparams->kernel_id],

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

    எனில் (kparams->max_bias > 0.0f && kparams->use_src1) {
        எனில் (kparams->n_head > 512) {
            திரும்பு HTP_STATUS_INVAL_PARAMS;
        }
        க்கு (uint32_t h = 0; h < kparams->n_head; h += 32) {
            HVX_Vector v_slopes = hvx_alibi_slopes(h, 1, kparams->n_head_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_queue_func_t task_func = kparams->use_src1 ? softmax_thread_mask_dma : softmax_thread_dma;
    work_queue_run(octx->ctx->work_queue, task_func, &smctx, n_threads);

    திரும்பு HTP_STATUS_OK;
}

முழு op_softmax(கட்டமைப்பு htp_ops_context * octx) {
    முழு err = HTP_STATUS_OK;

    தேர்வு (octx->src[0]->type) {
        நிலை HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            நிறுத்து;

        இயல்பு:
            err = HTP_STATUS_NO_SUPPORT;
            நிறுத்து;
    }

    திரும்பு err;
}
