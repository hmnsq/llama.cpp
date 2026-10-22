#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "work-queue.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"
#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "htp-vtcm.h"
#ଅନ୍ତର୍ଭାଵ "htp/softmax-ops.h"
#ଅନ୍ତର୍ଭାଵ "hvx-flash-attn.h"

ସଂରଚନା htp_softmax_context {
    ସଂରଚନା htp_ops_context * octx;
    ଅଚଳ ସଂରଚନା htp_softmax_kernel_params * kparams;

    ଶୂନ୍ୟ *   compute;

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

    ଦଶମିକ    scale;
    ଦଶମିକ    max_bias;
    ଦଶମିକ    m0;
    ଦଶମିକ    m1;

    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne01;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne02;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne12;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne13;

    uint32_t src0_nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;

    ଦଶମିକ    slopes[512] __attribute__((aligned(128)));
};

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ (*softmax_compute_fn_t)(
    ଶୂନ୍ୟ * restrict dst,
    ଅଚଳ ଶୂନ୍ୟ * restrict src0,
    ଅଚଳ ଶୂନ୍ୟ * restrict mask,
    uint32_t ne00,
    ଦଶମିକ scale,
    ଦଶମିକ slope
);

ସ୍ଥିର ଶୂନ୍ୟ hvx_fast_softmax_prep_f16(ଅଚଳ uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elems,
                                      ଦଶମିକ     scale,
                                      ଅଚଳ uint8_t * restrict mask,
                                      ଦଶମିକ slope) {
    ଅଚଳ HVX_Vector * restrict v_src  = (ଅଚଳ HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    ଅଚଳ HVX_Vector * restrict v_mask = (ଅଚଳ HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nvec_64 = ଅଙ୍କ_elems / VLEN_FP16;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nloe_64 = ଅଙ୍କ_elems % VLEN_FP16;

    #pragma unroll(2)
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nvec_64; i++) {
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

    ଯଦି (nloe_64 > 0) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_mask[nvec_64]);
        HVX_Vector m0 = Q6_V_lo_W(p);

        HVX_Vector s0 = v_src[2 * nvec_64];
        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));

        ଯଦି (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * ଆକାର(ଦଶମିକ), Q6_Vsf_equals_Vqf32(v0));
        } ଅନ୍ୟଥା {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_Vector m1 = Q6_V_hi_W(p);
            HVX_Vector s1 = v_src[2 * nvec_64 + 1];
            HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * ଆକାର(ଦଶମିକ), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_fast_softmax_prep_f32(ଅଚଳ uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elems,
                                      ଦଶମିକ     scale,
                                      ଅଚଳ uint8_t * restrict mask,
                                      ଦଶମିକ slope) {
    ଅଚଳ HVX_Vector * restrict v_src  = (ଅଚଳ HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    ଅଚଳ HVX_Vector * restrict v_mask = (ଅଚଳ HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nvec = ଅଙ୍କ_elems / VLEN_FP32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nloe = ଅଙ୍କ_elems % VLEN_FP32;

    #pragma unroll(4)
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v3 = v_mask[i];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    ଯଦି (nloe > 0) {
        HVX_Vector v1 = v_src[nvec];
        HVX_Vector v3 = v_mask[nvec];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * ଆକାର(ଦଶମିକ), Q6_Vsf_equals_Vqf32(v5));
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_fast_softmax_f32(ଅଚଳ uint8_t * restrict src, uint8_t * restrict dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elems) {
    ଅଚଳ HVX_Vector * restrict v_src = (ଅଚଳ HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nvec = ଅଙ୍କ_elems / VLEN_FP32;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nloe = ଅଙ୍କ_elems % VLEN_FP32;

    HVX_Vector max_vec = hvx_vec_splat_f32(((ଅଚଳ ଦଶମିକ *) src)[0]);

    #pragma unroll(2)
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    ଯଦି (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * ଆକାର(ଦଶମିକ));
        HVX_Vector neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_Vector v_tail     = Q6_V_vmux_QVV(q_mask, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_tail);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_Vector sum_vec = Q6_V_vsplat_R(0x00000000);

    #pragma unroll(2)
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3);

        v_dst[i] = v3;
    }

    ଯଦି (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * ଆକାର(ଦଶମିକ));
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
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_dst[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    ଯଦି (nloe > 0) {
        HVX_Vector v1 = v_dst[nvec];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * ଆକାର(ଦଶମିକ), Q6_Vsf_equals_Vqf32(v2));
    }
}

ସ୍ଥିର ଶୂନ୍ୟ compute_fast_softmax_f32_nomask(
    ଶୂନ୍ୟ * restrict dst,
    ଅଚଳ ଶୂନ୍ୟ * restrict src0,
    ଅଚଳ ଶୂନ୍ୟ * restrict mask,
    uint32_t ne00,
    ଦଶମିକ scale,
    ଦଶମିକ slope
) {
    (ଶୂନ୍ୟ) mask;
    (ଶୂନ୍ୟ) slope;
    hvx_scale_f32((uint8_t *) dst, (ଅଚଳ uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((ଅଚଳ uint8_t *) dst, (uint8_t *) dst, ne00);
}

ସ୍ଥିର ଶୂନ୍ୟ compute_fast_softmax_f32_mask_f32(
    ଶୂନ୍ୟ * restrict dst,
    ଅଚଳ ଶୂନ୍ୟ * restrict src0,
    ଅଚଳ ଶୂନ୍ୟ * restrict mask,
    uint32_t ne00,
    ଦଶମିକ scale,
    ଦଶମିକ slope
) {
    hvx_fast_softmax_prep_f32((ଅଚଳ uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ଅଚଳ uint8_t *) mask, slope);
    hvx_fast_softmax_f32((ଅଚଳ uint8_t *) dst, (uint8_t *) dst, ne00);
}

ସ୍ଥିର ଶୂନ୍ୟ compute_fast_softmax_f32_mask_f16(
    ଶୂନ୍ୟ * restrict dst,
    ଅଚଳ ଶୂନ୍ୟ * restrict src0,
    ଅଚଳ ଶୂନ୍ୟ * restrict mask,
    uint32_t ne00,
    ଦଶମିକ scale,
    ଦଶମିକ slope
) {
    hvx_fast_softmax_prep_f16((ଅଚଳ uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ଅଚଳ uint8_t *) mask, slope);
    hvx_fast_softmax_f32((ଅଚଳ uint8_t *) dst, (uint8_t *) dst, ne00);
}

ସ୍ଥିର ଅଚଳ softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_ଗଣନା] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_MASK_F32] = compute_fast_softmax_f32_mask_f32,
    [HTP_SOFTMAX_KERNEL_MASK_F16] = compute_fast_softmax_f32_mask_f16,
};

ସ୍ଥିର ଶୂନ୍ୟ softmax_thread_dma(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    (ଶୂନ୍ୟ) nth;
    ଅଚଳ ସଂରଚନା htp_softmax_context * smctx = (ଅଚଳ ସଂରଚନା htp_softmax_context *) data;
    ସଂରଚନା htp_ops_context * octx = smctx->octx;
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ uint32_t src0_nrows            = smctx->nrows;
    ଅଚଳ uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    ଅଚଳ uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    ଯଦି (src0_start_row >= src0_end_row) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ dma_addr_t data_src0 = smctx->data_src0;
    ଅଚଳ dma_addr_t data_dst  = smctx->data_dst;

    ଅଚଳ size_t src0_row_size = src0->ne[0] * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t dst_row_size  = src0->ne[0] * ଆକାର(ଦଶମିକ);

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    ଅଚଳ size_t src0_vtcm_half = smctx->src0_spad_half_size;
    ଅଚଳ size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ପାଇଁ (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;
        ଶୂନ୍ୟ * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        ଶୂନ୍ୟ * s_spad = src0_vtcm_base + idx * src0_vtcm_half;

        dma_queue_ଠେଲ(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_ଠେଲ(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ଅଚଳ uint32_t ne00 = src0->ne[0];

    ପାଇଁ (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        ଶୂନ୍ୟ * d_spad = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).src;
        ଶୂନ୍ୟ * s_spad = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, ଖାଲି, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_ଠେଲ(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        ଅଚଳ uint32_t next_r = r + 2;
        ଯଦି (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;
            dma_queue_ଠେଲ(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

ସ୍ଥିର ଶୂନ୍ୟ softmax_thread_mask_dma(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    (ଶୂନ୍ୟ) nth;
    ଅଚଳ ସଂରଚନା htp_softmax_context * smctx = (ଅଚଳ ସଂରଚନା htp_softmax_context *) data;
    ସଂରଚନା htp_ops_context * octx = smctx->octx;
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ uint32_t src0_nrows            = smctx->nrows;
    ଅଚଳ uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    ଅଚଳ uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    ଯଦି (src0_start_row >= src0_end_row) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ dma_addr_t data_src0 = smctx->data_src0;
    ଅଚଳ dma_addr_t data_src1 = smctx->data_src1;
    ଅଚଳ dma_addr_t data_dst  = smctx->data_dst;

    ଅଚଳ size_t src0_row_size = src0->ne[0] * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t dst_row_size  = src0->ne[0] * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t mask_row_size = smctx->use_f16 ? (src1->ne[0] * ଆକାର(__fp16)) : (src1->ne[0] * ଆକାର(ଦଶମିକ));

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_base = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    ଅଚଳ size_t src0_vtcm_half = smctx->src0_spad_half_size;
    ଅଚଳ size_t src1_vtcm_half = smctx->src1_spad_half_size;
    ଅଚଳ size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    ଅଚଳ uint32_t nb11 = src1->nb[1];
    ଅଚଳ uint32_t nb12 = src1->nb[2];
    ଅଚଳ uint32_t nb13 = src1->nb[3];

    ଅଚଳ uint32_t ne00 = src0->ne[0];
    ଅଚଳ uint32_t ne01 = src0->ne[1];
    ଅଚଳ uint32_t ne02 = src0->ne[2];
    ଅଚଳ uint32_t ne03 = src0->ne[3];
    ଅଚଳ uint32_t ne12 = src1->ne[2];
    ଅଚଳ uint32_t ne13 = src1->ne[3];

    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * div_ne01 = &smctx->div_ne01;
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * div_ne02 = &smctx->div_ne02;
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * div_ne12 = &smctx->div_ne12;
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * div_ne13 = &smctx->div_ne13;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ପାଇଁ (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = data_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        ଶୂନ୍ୟ * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        ଶୂନ୍ୟ * s_spad = src0_vtcm_base + idx * src0_vtcm_half;
        ଶୂନ୍ୟ * m_spad = src1_vtcm_base + idx * src1_vtcm_half;

        dma_queue_ଠେଲ(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_ଠେଲ(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        dma_queue_ଠେଲ(dma_q, dma_make_data(m_spad, cur_src1),
                       smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ଅଚଳ bool has_bias = smctx->max_bias > 0.0f;
    uint32_t prev_i2 = (uint32_t)-1;
    ଦଶମିକ slope = 1.0f;

    ପାଇଁ (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        ଶୂନ୍ୟ * d_spad = (ଶୂନ୍ୟ *) (uintptr_t) dma_queue_କାଢ଼(dma_q).src;
        ଶୂନ୍ୟ * s_spad = (ଶୂନ୍ୟ *) (uintptr_t) dma_queue_କାଢ଼(dma_q).dst;
        ଶୂନ୍ୟ * m_spad = (ଶୂନ୍ୟ *) (uintptr_t) dma_queue_କାଢ଼(dma_q).dst;

        ଯଦି (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            ଯଦି (i2 != prev_i2) {
                slope = smctx->slopes[i2];
                prev_i2 = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, slope);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_ଠେଲ(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        ଅଚଳ uint32_t next_r = r + 2;
        ଯଦି (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;

            uint32_t ni1 = fastmodulo(next_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(next_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t next_src1 = data_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_queue_ଠେଲ(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
            dma_queue_ଠେଲ(dma_q, dma_make_data(m_spad, next_src1),
                           smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା execute_op_softmax_f32(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ଅଚଳ ଅକ୍ଷର * op_type = "softmax-f32";

    ଅଚଳ ସଂରଚନା htp_softmax_kernel_params * kparams =
        (ଅଚଳ ସଂରଚନା htp_softmax_kernel_params *) octx->kernel_params;

    ଯଦି (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    ଯଦି (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_ଗଣନା) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    ଯଦି (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_size, kparams->vtcm_size);
        ଫେରନ୍ତୁ HTP_STATUS_VTCM_TOO_SMALL;
    }

    ଅଚଳ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ size_t elem_size    = ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t dst_row_size = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
        ଯଦି (nrows < octx->n_threads) {
            htp_ops_context_set_n_threads(octx, nrows ? nrows : 1);
        }
    }

    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads = octx->n_threads;
    uint8_t * ଅଚଳ vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    ଅଚଳ uint32_t off_src0 = 0;
    ଅଚଳ uint32_t off_dst  = off_src0 + kparams->vtcm_src0_size_per_thread * kparams->n_threads;
    ଅଚଳ uint32_t off_src1 = off_dst  + kparams->vtcm_dst_size_per_thread  * kparams->n_threads;

    ସଂରଚନା htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (ଶୂନ୍ୟ *) softmax_kernels[kparams->kernel_id],

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

    ଯଦି (kparams->max_bias > 0.0f && kparams->use_src1) {
        ଯଦି (kparams->n_head > 512) {
            ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
        }
        ପାଇଁ (uint32_t h = 0; h < kparams->n_head; h += 32) {
            HVX_Vector v_slopes = hvx_alibi_slopes(h, 1, kparams->n_head_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_queue_func_t task_func = kparams->use_src1 ? softmax_thread_mask_dma : softmax_thread_dma;
    work_queue_run(octx->ctx->work_queue, task_func, &smctx, n_threads);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_softmax(ସଂରଚନା htp_ops_context * octx) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = HTP_STATUS_OK;

    ବାଛ (octx->src[0]->type) {
        ସ୍ଥିତି HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            ଭାଙ୍ଗ;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            err = HTP_STATUS_NO_SUPPORT;
            ଭାଙ୍ଗ;
    }

    ଫେରନ୍ତୁ err;
}
