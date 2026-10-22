#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <HAP_perf.h>

#അന്തര്ഭാവ <math.h>
#അന്തര്ഭാവ <string.h>

#അന്തര്ഭാവ "dma-queue.h"
#അന്തര്ഭാവ "work-queue.h"
#അന്തര്ഭാവ "hvx-utils.h"
#അന്തര്ഭാവ "hex-fastdiv.h"
#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-profile.h"

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-tensor.h"
#അന്തര്ഭാവ "htp-vtcm.h"
#അന്തര്ഭാവ "htp/softmax-ops.h"
#അന്തര്ഭാവ "hvx-flash-attn.h"

ഘടന htp_softmax_context {
    ഘടന htp_ops_context * octx;
    സ്ഥിരം ഘടന htp_softmax_kernel_params * kparams;

    ശൂന്യം *   compute;

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

    ദശാംശം    scale;
    ദശാംശം    max_bias;
    ദശാംശം    m0;
    ദശാംശം    m1;

    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne01;
    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne02;
    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne12;
    ഘടന fastdiv_മൂല്യങ്ങൾ div_ne13;

    uint32_t src0_nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;

    ദശാംശം    slopes[512] __attribute__((aligned(128)));
};

തരനിർവചനം ശൂന്യം (*softmax_compute_fn_t)(
    ശൂന്യം * restrict dst,
    സ്ഥിരം ശൂന്യം * restrict src0,
    സ്ഥിരം ശൂന്യം * restrict mask,
    uint32_t ne00,
    ദശാംശം scale,
    ദശാംശം slope
);

സ്ഥിതം ശൂന്യം hvx_fast_softmax_prep_f16(സ്ഥിരം uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_elems,
                                      ദശാംശം     scale,
                                      സ്ഥിരം uint8_t * restrict mask,
                                      ദശാംശം slope) {
    സ്ഥിരം HVX_Vector * restrict v_src  = (സ്ഥിരം HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    സ്ഥിരം HVX_Vector * restrict v_mask = (സ്ഥിരം HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nvec_64 = അക്കം_elems / VLEN_FP16;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nloe_64 = അക്കം_elems % VLEN_FP16;

    #pragma unroll(2)
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nvec_64; i++) {
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

    എങ്കിൽ (nloe_64 > 0) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_mask[nvec_64]);
        HVX_Vector m0 = Q6_V_lo_W(p);

        HVX_Vector s0 = v_src[2 * nvec_64];
        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));

        എങ്കിൽ (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * വലുപ്പം(ദശാംശം), Q6_Vsf_equals_Vqf32(v0));
        } അല്ലെങ്കിൽ {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_Vector m1 = Q6_V_hi_W(p);
            HVX_Vector s1 = v_src[2 * nvec_64 + 1];
            HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * വലുപ്പം(ദശാംശം), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

സ്ഥിതം ശൂന്യം hvx_fast_softmax_prep_f32(സ്ഥിരം uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_elems,
                                      ദശാംശം     scale,
                                      സ്ഥിരം uint8_t * restrict mask,
                                      ദശാംശം slope) {
    സ്ഥിരം HVX_Vector * restrict v_src  = (സ്ഥിരം HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    സ്ഥിരം HVX_Vector * restrict v_mask = (സ്ഥിരം HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nvec = അക്കം_elems / VLEN_FP32;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nloe = അക്കം_elems % VLEN_FP32;

    #pragma unroll(4)
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v3 = v_mask[i];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    എങ്കിൽ (nloe > 0) {
        HVX_Vector v1 = v_src[nvec];
        HVX_Vector v3 = v_mask[nvec];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * വലുപ്പം(ദശാംശം), Q6_Vsf_equals_Vqf32(v5));
    }
}

സ്ഥിതം ശൂന്യം hvx_fast_softmax_f32(സ്ഥിരം uint8_t * restrict src, uint8_t * restrict dst, സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_elems) {
    സ്ഥിരം HVX_Vector * restrict v_src = (സ്ഥിരം HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nvec = അക്കം_elems / VLEN_FP32;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nloe = അക്കം_elems % VLEN_FP32;

    HVX_Vector max_vec = hvx_vec_splat_f32(((സ്ഥിരം ദശാംശം *) src)[0]);

    #pragma unroll(2)
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    എങ്കിൽ (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * വലുപ്പം(ദശാംശം));
        HVX_Vector neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_Vector v_tail     = Q6_V_vmux_QVV(q_mask, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_tail);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_Vector sum_vec = Q6_V_vsplat_R(0x00000000);

    #pragma unroll(2)
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3);

        v_dst[i] = v3;
    }

    എങ്കിൽ (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * വലുപ്പം(ദശാംശം));
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_dst[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    എങ്കിൽ (nloe > 0) {
        HVX_Vector v1 = v_dst[nvec];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * വലുപ്പം(ദശാംശം), Q6_Vsf_equals_Vqf32(v2));
    }
}

സ്ഥിതം ശൂന്യം compute_fast_softmax_f32_nomask(
    ശൂന്യം * restrict dst,
    സ്ഥിരം ശൂന്യം * restrict src0,
    സ്ഥിരം ശൂന്യം * restrict mask,
    uint32_t ne00,
    ദശാംശം scale,
    ദശാംശം slope
) {
    (ശൂന്യം) mask;
    (ശൂന്യം) slope;
    hvx_scale_f32((uint8_t *) dst, (സ്ഥിരം uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((സ്ഥിരം uint8_t *) dst, (uint8_t *) dst, ne00);
}

സ്ഥിതം ശൂന്യം compute_fast_softmax_f32_mask_f32(
    ശൂന്യം * restrict dst,
    സ്ഥിരം ശൂന്യം * restrict src0,
    സ്ഥിരം ശൂന്യം * restrict mask,
    uint32_t ne00,
    ദശാംശം scale,
    ദശാംശം slope
) {
    hvx_fast_softmax_prep_f32((സ്ഥിരം uint8_t *) src0, (uint8_t *) dst, ne00, scale, (സ്ഥിരം uint8_t *) mask, slope);
    hvx_fast_softmax_f32((സ്ഥിരം uint8_t *) dst, (uint8_t *) dst, ne00);
}

സ്ഥിതം ശൂന്യം compute_fast_softmax_f32_mask_f16(
    ശൂന്യം * restrict dst,
    സ്ഥിരം ശൂന്യം * restrict src0,
    സ്ഥിരം ശൂന്യം * restrict mask,
    uint32_t ne00,
    ദശാംശം scale,
    ദശാംശം slope
) {
    hvx_fast_softmax_prep_f16((സ്ഥിരം uint8_t *) src0, (uint8_t *) dst, ne00, scale, (സ്ഥിരം uint8_t *) mask, slope);
    hvx_fast_softmax_f32((സ്ഥിരം uint8_t *) dst, (uint8_t *) dst, ne00);
}

സ്ഥിതം സ്ഥിരം softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_എണ്ണം] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_MASK_F32] = compute_fast_softmax_f32_mask_f32,
    [HTP_SOFTMAX_KERNEL_MASK_F16] = compute_fast_softmax_f32_mask_f16,
};

സ്ഥിതം ശൂന്യം softmax_thread_dma(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    (ശൂന്യം) nth;
    സ്ഥിരം ഘടന htp_softmax_context * smctx = (സ്ഥിരം ഘടന htp_softmax_context *) data;
    ഘടന htp_ops_context * octx = smctx->octx;
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;
    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];

    സ്ഥിരം uint32_t src0_nrows            = smctx->nrows;
    സ്ഥിരം uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    സ്ഥിരം uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    സ്ഥിരം uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    എങ്കിൽ (src0_start_row >= src0_end_row) {
        തിരികെ;
    }

    സ്ഥിരം dma_addr_t data_src0 = smctx->data_src0;
    സ്ഥിരം dma_addr_t data_dst  = smctx->data_dst;

    സ്ഥിരം size_t src0_row_size = src0->ne[0] * വലുപ്പം(ദശാംശം);
    സ്ഥിരം size_t dst_row_size  = src0->ne[0] * വലുപ്പം(ദശാംശം);

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    സ്ഥിരം size_t src0_vtcm_half = smctx->src0_spad_half_size;
    സ്ഥിരം size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    വേണ്ടി (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;
        ശൂന്യം * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        ശൂന്യം * s_spad = src0_vtcm_base + idx * src0_vtcm_half;

        dma_queue_തള്ളുക(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    സ്ഥിരം uint32_t ne00 = src0->ne[0];

    വേണ്ടി (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        ശൂന്യം * d_spad = (ശൂന്യം *) dma_queue_എടുക്കുക(dma_q).src;
        ശൂന്യം * s_spad = (ശൂന്യം *) dma_queue_എടുക്കുക(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, ശൂന്യ, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_തള്ളുക(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        സ്ഥിരം uint32_t next_r = r + 2;
        എങ്കിൽ (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;
            dma_queue_തള്ളുക(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

സ്ഥിതം ശൂന്യം softmax_thread_mask_dma(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    (ശൂന്യം) nth;
    സ്ഥിരം ഘടന htp_softmax_context * smctx = (സ്ഥിരം ഘടന htp_softmax_context *) data;
    ഘടന htp_ops_context * octx = smctx->octx;
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * src1 = octx->src[1];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;
    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];

    സ്ഥിരം uint32_t src0_nrows            = smctx->nrows;
    സ്ഥിരം uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    സ്ഥിരം uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    സ്ഥിരം uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    എങ്കിൽ (src0_start_row >= src0_end_row) {
        തിരികെ;
    }

    സ്ഥിരം dma_addr_t data_src0 = smctx->data_src0;
    സ്ഥിരം dma_addr_t data_src1 = smctx->data_src1;
    സ്ഥിരം dma_addr_t data_dst  = smctx->data_dst;

    സ്ഥിരം size_t src0_row_size = src0->ne[0] * വലുപ്പം(ദശാംശം);
    സ്ഥിരം size_t dst_row_size  = src0->ne[0] * വലുപ്പം(ദശാംശം);
    സ്ഥിരം size_t mask_row_size = smctx->use_f16 ? (src1->ne[0] * വലുപ്പം(__fp16)) : (src1->ne[0] * വലുപ്പം(ദശാംശം));

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_base = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_size_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_size_per_thread);

    സ്ഥിരം size_t src0_vtcm_half = smctx->src0_spad_half_size;
    സ്ഥിരം size_t src1_vtcm_half = smctx->src1_spad_half_size;
    സ്ഥിരം size_t dst_vtcm_half  = smctx->dst_spad_half_size;

    സ്ഥിരം uint32_t nb11 = src1->nb[1];
    സ്ഥിരം uint32_t nb12 = src1->nb[2];
    സ്ഥിരം uint32_t nb13 = src1->nb[3];

    സ്ഥിരം uint32_t ne00 = src0->ne[0];
    സ്ഥിരം uint32_t ne01 = src0->ne[1];
    സ്ഥിരം uint32_t ne02 = src0->ne[2];
    സ്ഥിരം uint32_t ne03 = src0->ne[3];
    സ്ഥിരം uint32_t ne12 = src1->ne[2];
    സ്ഥിരം uint32_t ne13 = src1->ne[3];

    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_ne01 = &smctx->div_ne01;
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_ne02 = &smctx->div_ne02;
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_ne12 = &smctx->div_ne12;
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_ne13 = &smctx->div_ne13;

    dma_queue * dma_q = octx->ctx->dma[ith];

    വേണ്ടി (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = data_dst  + r * dst_row_size;
        dma_addr_t cur_src0 = data_src0 + r * src0_row_size;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = data_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        ശൂന്യം * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        ശൂന്യം * s_spad = src0_vtcm_base + idx * src0_vtcm_half;
        ശൂന്യം * m_spad = src1_vtcm_base + idx * src1_vtcm_half;

        dma_queue_തള്ളുക(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 0);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s_spad, cur_src0),
                       smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
        dma_queue_തള്ളുക(dma_q, dma_make_data(m_spad, cur_src1),
                       smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    സ്ഥിരം bool has_bias = smctx->max_bias > 0.0f;
    uint32_t prev_i2 = (uint32_t)-1;
    ദശാംശം slope = 1.0f;

    വേണ്ടി (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        ശൂന്യം * d_spad = (ശൂന്യം *) (uintptr_t) dma_queue_എടുക്കുക(dma_q).src;
        ശൂന്യം * s_spad = (ശൂന്യം *) (uintptr_t) dma_queue_എടുക്കുക(dma_q).dst;
        ശൂന്യം * m_spad = (ശൂന്യം *) (uintptr_t) dma_queue_എടുക്കുക(dma_q).dst;

        എങ്കിൽ (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            എങ്കിൽ (i2 != prev_i2) {
                slope = smctx->slopes[i2];
                prev_i2 = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, slope);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = data_dst + r * dst_row_size;
        dma_queue_തള്ളുക(dma_q, dma_make_data(cur_dst, d_spad),
                       dst_row_size, smctx->dst_row_size_aligned, dst_row_size, 1);

        സ്ഥിരം uint32_t next_r = r + 2;
        എങ്കിൽ (next_r < src0_end_row) {
            dma_addr_t next_src0 = data_src0 + next_r * src0_row_size;

            uint32_t ni1 = fastmodulo(next_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(next_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t next_src1 = data_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_queue_തള്ളുക(dma_q, dma_make_data(s_spad, next_src0),
                           smctx->src0_row_size_aligned, src0_row_size, src0_row_size, 1);
            dma_queue_തള്ളുക(dma_q, dma_make_data(m_spad, next_src1),
                           smctx->src1_row_size_aligned, mask_row_size, mask_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ execute_op_softmax_f32(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;

    സ്ഥിരം അക്ഷരം * op_type = "softmax-f32";

    സ്ഥിരം ഘടന htp_softmax_kernel_params * kparams =
        (സ്ഥിരം ഘടന htp_softmax_kernel_params *) octx->kernel_params;

    എങ്കിൽ (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }

    എങ്കിൽ (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_എണ്ണം) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }

    എങ്കിൽ (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_size, kparams->vtcm_size);
        തിരികെ HTP_STATUS_VTCM_TOO_SMALL;
    }

    സ്ഥിരം uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    സ്ഥിരം size_t elem_size    = വലുപ്പം(ദശാംശം);
    സ്ഥിരം size_t dst_row_size = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        row_start = range.start;
        nrows     = range.എണ്ണം;
        എങ്കിൽ (nrows < octx->n_threads) {
            htp_ops_context_set_n_threads(octx, nrows ? nrows : 1);
        }
    }

    എങ്കിൽ (nrows == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t n_threads = octx->n_threads;
    uint8_t * സ്ഥിരം vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    സ്ഥിരം uint32_t off_src0 = 0;
    സ്ഥിരം uint32_t off_dst  = off_src0 + kparams->vtcm_src0_size_per_thread * kparams->n_threads;
    സ്ഥിരം uint32_t off_src1 = off_dst  + kparams->vtcm_dst_size_per_thread  * kparams->n_threads;

    ഘടന htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (ശൂന്യം *) softmax_kernels[kparams->kernel_id],

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

    എങ്കിൽ (kparams->max_bias > 0.0f && kparams->use_src1) {
        എങ്കിൽ (kparams->n_head > 512) {
            തിരികെ HTP_STATUS_INVAL_PARAMS;
        }
        വേണ്ടി (uint32_t h = 0; h < kparams->n_head; h += 32) {
            HVX_Vector v_slopes = hvx_alibi_slopes(h, 1, kparams->n_head_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_queue_func_t task_func = kparams->use_src1 ? softmax_thread_mask_dma : softmax_thread_dma;
    work_queue_run(octx->ctx->work_queue, task_func, &smctx, n_threads);

    തിരികെ HTP_STATUS_OK;
}

പൂർണ്ണസംഖ്യ op_softmax(ഘടന htp_ops_context * octx) {
    പൂർണ്ണസംഖ്യ err = HTP_STATUS_OK;

    തിരഞ്ഞെടുപ്പ് (octx->src[0]->type) {
        അവസ്ഥ HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            നിർത്തുക;

        സ്ഥിരസ്ഥിതി:
            err = HTP_STATUS_NO_SUPPORT;
            നിർത്തുക;
    }

    തിരികെ err;
}
