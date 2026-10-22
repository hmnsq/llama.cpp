#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#অন্তর্ভাব <math.h>
#অন্তর্ভাব <string.h>

#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "work-queue.h"
#অন্তর্ভাব "hvx-utils.h"
#অন্তর্ভাব "hex-fastdiv.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "htp-vtcm.h"
#অন্তর্ভাব "htp/softmax-ops.h"
#অন্তর্ভাব "hvx-flash-attn.h"

কাঠামো htp_softmax_context {
    কাঠামো htp_ops_context * octx;
    ধ্রুবক কাঠামো htp_softmax_kernel_params * kparams;

    শূন্য *   compute;

    dma_addr_t তথ্য_src0;
    dma_addr_t তথ্য_src1;
    dma_addr_t তথ্য_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_আকার_per_thread;
    uint32_t  vtcm_src1_আকার_per_thread;
    uint32_t  vtcm_dst_আকার_per_thread;

    uint32_t  src0_spad_half_আকার;
    uint32_t  src1_spad_half_আকার;
    uint32_t  dst_spad_half_আকার;

    uint32_t  src0_row_আকার_aligned;
    uint32_t  src1_row_আকার_aligned;
    uint32_t  dst_row_আকার_aligned;

    bool     use_f16;

    uint32_t n_শীর্ষ;
    uint32_t n_শীর্ষ_log2;

    দশমিক    scale;
    দশমিক    max_bias;
    দশমিক    m0;
    দশমিক    m1;

    কাঠামো fastdiv_মানসমূহ div_ne01;
    কাঠামো fastdiv_মানসমূহ div_ne02;
    কাঠামো fastdiv_মানসমূহ div_ne12;
    কাঠামো fastdiv_মানসমূহ div_ne13;

    uint32_t src0_nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;

    দশমিক    slopes[512] __attribute__((aligned(128)));
};

প্রকারলেখ শূন্য (*softmax_compute_fn_t)(
    শূন্য * restrict dst,
    ধ্রুবক শূন্য * restrict src0,
    ধ্রুবক শূন্য * restrict mask,
    uint32_t ne00,
    দশমিক scale,
    দশমিক slope
);

স্থির শূন্য hvx_fast_softmax_prep_f16(ধ্রুবক uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems,
                                      দশমিক     scale,
                                      ধ্রুবক uint8_t * restrict mask,
                                      দশমিক slope) {
    ধ্রুবক HVX_Vector * restrict v_src  = (ধ্রুবক HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    ধ্রুবক HVX_Vector * restrict v_mask = (ধ্রুবক HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    ধ্রুবক পূর্ণসংখ্যা nvec_64 = অঙ্ক_elems / VLEN_FP16;
    ধ্রুবক পূর্ণসংখ্যা nloe_64 = অঙ্ক_elems % VLEN_FP16;

    #pragma unroll(2)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec_64; i++) {
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

    যদি (nloe_64 > 0) {
        HVX_VectorPair p = hvx_vec_f16_to_f32(v_mask[nvec_64]);
        HVX_Vector m0 = Q6_V_lo_W(p);

        HVX_Vector s0 = v_src[2 * nvec_64];
        HVX_Vector v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, slope_vec));

        যদি (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * আকার(দশমিক), Q6_Vsf_equals_Vqf32(v0));
        } অন্যথা {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_Vector m1 = Q6_V_hi_W(p);
            HVX_Vector s1 = v_src[2 * nvec_64 + 1];
            HVX_Vector v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, slope_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * আকার(দশমিক), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

স্থির শূন্য hvx_fast_softmax_prep_f32(ধ্রুবক uint8_t * restrict src,
                                      uint8_t * restrict dst,
                                      ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems,
                                      দশমিক     scale,
                                      ধ্রুবক uint8_t * restrict mask,
                                      দশমিক slope) {
    ধ্রুবক HVX_Vector * restrict v_src  = (ধ্রুবক HVX_Vector *) src;
    HVX_Vector * restrict v_dst        = (HVX_Vector *) dst;
    ধ্রুবক HVX_Vector * restrict v_mask = (ধ্রুবক HVX_Vector *) mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);

    ধ্রুবক পূর্ণসংখ্যা nvec = অঙ্ক_elems / VLEN_FP32;
    ধ্রুবক পূর্ণসংখ্যা nloe = অঙ্ক_elems % VLEN_FP32;

    #pragma unroll(4)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v3 = v_mask[i];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    যদি (nloe > 0) {
        HVX_Vector v1 = v_src[nvec];
        HVX_Vector v3 = v_mask[nvec];

        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * আকার(দশমিক), Q6_Vsf_equals_Vqf32(v5));
    }
}

স্থির শূন্য hvx_fast_softmax_f32(ধ্রুবক uint8_t * restrict src, uint8_t * restrict dst, ধ্রুবক পূর্ণসংখ্যা অঙ্ক_elems) {
    ধ্রুবক HVX_Vector * restrict v_src = (ধ্রুবক HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    ধ্রুবক পূর্ণসংখ্যা nvec = অঙ্ক_elems / VLEN_FP32;
    ধ্রুবক পূর্ণসংখ্যা nloe = অঙ্ক_elems % VLEN_FP32;

    HVX_Vector max_vec = hvx_vec_splat_f32(((ধ্রুবক দশমিক *) src)[0]);

    #pragma unroll(2)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    যদি (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * আকার(দশমিক));
        HVX_Vector neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_Vector v_লেজ     = Q6_V_vmux_QVV(q_mask, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_লেজ);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_Vector যোগ_vec = Q6_V_vsplat_R(0x00000000);

    #pragma unroll(2)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        যোগ_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(যোগ_vec), v3);

        v_dst[i] = v3;
    }

    যদি (nloe > 0) {
        HVX_VectorPred q_mask = Q6_Q_vsetq_R(nloe * আকার(দশমিক));
        HVX_Vector v1     = v_src[nvec];
        HVX_Vector v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_Vector v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        HVX_Vector v3_pad = Q6_V_vmux_QVV(q_mask, v3, Q6_V_vzero());

        যোগ_vec     = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(যোগ_vec), v3_pad);
        v_dst[nvec] = v3_pad;
    }

    যোগ_vec = hvx_vec_reduce_যোগ_f32(Q6_Vsf_equals_Vqf32(যোগ_vec));

    HVX_VectorPred pos_যোগ   = Q6_Q_vcmp_gt_VwVw(যোগ_vec, Q6_V_vzero());
    HVX_Vector     v4        = hvx_vec_inverse_f32(যোগ_vec);
    HVX_Vector     scale_vec = Q6_V_vmux_QVV(pos_যোগ, v4, hvx_vec_splat_f32(1.0f));

    #pragma unroll(2)
    জন্য (পূর্ণসংখ্যা i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_dst[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    যদি (nloe > 0) {
        HVX_Vector v1 = v_dst[nvec];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * আকার(দশমিক), Q6_Vsf_equals_Vqf32(v2));
    }
}

স্থির শূন্য compute_fast_softmax_f32_nomask(
    শূন্য * restrict dst,
    ধ্রুবক শূন্য * restrict src0,
    ধ্রুবক শূন্য * restrict mask,
    uint32_t ne00,
    দশমিক scale,
    দশমিক slope
) {
    (শূন্য) mask;
    (শূন্য) slope;
    hvx_scale_f32((uint8_t *) dst, (ধ্রুবক uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((ধ্রুবক uint8_t *) dst, (uint8_t *) dst, ne00);
}

স্থির শূন্য compute_fast_softmax_f32_mask_f32(
    শূন্য * restrict dst,
    ধ্রুবক শূন্য * restrict src0,
    ধ্রুবক শূন্য * restrict mask,
    uint32_t ne00,
    দশমিক scale,
    দশমিক slope
) {
    hvx_fast_softmax_prep_f32((ধ্রুবক uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ধ্রুবক uint8_t *) mask, slope);
    hvx_fast_softmax_f32((ধ্রুবক uint8_t *) dst, (uint8_t *) dst, ne00);
}

স্থির শূন্য compute_fast_softmax_f32_mask_f16(
    শূন্য * restrict dst,
    ধ্রুবক শূন্য * restrict src0,
    ধ্রুবক শূন্য * restrict mask,
    uint32_t ne00,
    দশমিক scale,
    দশমিক slope
) {
    hvx_fast_softmax_prep_f16((ধ্রুবক uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ধ্রুবক uint8_t *) mask, slope);
    hvx_fast_softmax_f32((ধ্রুবক uint8_t *) dst, (uint8_t *) dst, ne00);
}

স্থির ধ্রুবক softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_গুনতি] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_MASK_F32] = compute_fast_softmax_f32_mask_f32,
    [HTP_SOFTMAX_KERNEL_MASK_F16] = compute_fast_softmax_f32_mask_f16,
};

স্থির শূন্য softmax_thread_dma(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    (শূন্য) nth;
    ধ্রুবক কাঠামো htp_softmax_context * smctx = (ধ্রুবক কাঠামো htp_softmax_context *) তথ্য;
    কাঠামো htp_ops_context * octx = smctx->octx;
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;
    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুবক uint32_t src0_nrows            = smctx->nrows;
    ধ্রুবক uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    ধ্রুবক uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    ধ্রুবক uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    যদি (src0_start_row >= src0_end_row) {
        ফেরত;
    }

    ধ্রুবক dma_addr_t তথ্য_src0 = smctx->তথ্য_src0;
    ধ্রুবক dma_addr_t তথ্য_dst  = smctx->তথ্য_dst;

    ধ্রুবক size_t src0_row_আকার = src0->ne[0] * আকার(দশমিক);
    ধ্রুবক size_t dst_row_আকার  = src0->ne[0] * আকার(দশমিক);

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_আকার_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_আকার_per_thread);

    ধ্রুবক size_t src0_vtcm_half = smctx->src0_spad_half_আকার;
    ধ্রুবক size_t dst_vtcm_half  = smctx->dst_spad_half_আকার;

    dma_queue * dma_q = octx->ctx->dma[ith];

    জন্য (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = তথ্য_dst  + r * dst_row_আকার;
        dma_addr_t cur_src0 = তথ্য_src0 + r * src0_row_আকার;
        শূন্য * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        শূন্য * s_spad = src0_vtcm_base + idx * src0_vtcm_half;

        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(cur_dst, d_spad),
                       dst_row_আকার, smctx->dst_row_আকার_aligned, dst_row_আকার, 0);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s_spad, cur_src0),
                       smctx->src0_row_আকার_aligned, src0_row_আকার, src0_row_আকার, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ধ্রুবক uint32_t ne00 = src0->ne[0];

    জন্য (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        শূন্য * d_spad = (শূন্য *) dma_queue_বের_করো(dma_q).src;
        শূন্য * s_spad = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, ফাঁকা, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = তথ্য_dst + r * dst_row_আকার;
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(cur_dst, d_spad),
                       dst_row_আকার, smctx->dst_row_আকার_aligned, dst_row_আকার, 1);

        ধ্রুবক uint32_t পরবর্তী_r = r + 2;
        যদি (পরবর্তী_r < src0_end_row) {
            dma_addr_t পরবর্তী_src0 = তথ্য_src0 + পরবর্তী_r * src0_row_আকার;
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s_spad, পরবর্তী_src0),
                           smctx->src0_row_আকার_aligned, src0_row_আকার, src0_row_আকার, 1);
        }
    }

    dma_queue_flush(dma_q);
}

স্থির শূন্য softmax_thread_mask_dma(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    (শূন্য) nth;
    ধ্রুবক কাঠামো htp_softmax_context * smctx = (ধ্রুবক কাঠামো htp_softmax_context *) তথ্য;
    কাঠামো htp_ops_context * octx = smctx->octx;
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;
    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুবক uint32_t src0_nrows            = smctx->nrows;
    ধ্রুবক uint32_t src0_nrows_per_thread = smctx->src0_nrows_per_thread;

    ধ্রুবক uint32_t src0_start_row = smctx->row_start + src0_nrows_per_thread * ith;
    ধ্রুবক uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, smctx->row_start + src0_nrows);

    যদি (src0_start_row >= src0_end_row) {
        ফেরত;
    }

    ধ্রুবক dma_addr_t তথ্য_src0 = smctx->তথ্য_src0;
    ধ্রুবক dma_addr_t তথ্য_src1 = smctx->তথ্য_src1;
    ধ্রুবক dma_addr_t তথ্য_dst  = smctx->তথ্য_dst;

    ধ্রুবক size_t src0_row_আকার = src0->ne[0] * আকার(দশমিক);
    ধ্রুবক size_t dst_row_আকার  = src0->ne[0] * আকার(দশমিক);
    ধ্রুবক size_t mask_row_আকার = smctx->use_f16 ? (src1->ne[0] * আকার(__fp16)) : (src1->ne[0] * আকার(দশমিক));

    uint8_t * src0_vtcm_base = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_আকার_per_thread);
    uint8_t * src1_vtcm_base = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_আকার_per_thread);
    uint8_t * dst_vtcm_base  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_আকার_per_thread);

    ধ্রুবক size_t src0_vtcm_half = smctx->src0_spad_half_আকার;
    ধ্রুবক size_t src1_vtcm_half = smctx->src1_spad_half_আকার;
    ধ্রুবক size_t dst_vtcm_half  = smctx->dst_spad_half_আকার;

    ধ্রুবক uint32_t nb11 = src1->nb[1];
    ধ্রুবক uint32_t nb12 = src1->nb[2];
    ধ্রুবক uint32_t nb13 = src1->nb[3];

    ধ্রুবক uint32_t ne00 = src0->ne[0];
    ধ্রুবক uint32_t ne01 = src0->ne[1];
    ধ্রুবক uint32_t ne02 = src0->ne[2];
    ধ্রুবক uint32_t ne03 = src0->ne[3];
    ধ্রুবক uint32_t ne12 = src1->ne[2];
    ধ্রুবক uint32_t ne13 = src1->ne[3];

    ধ্রুবক কাঠামো fastdiv_মানসমূহ * div_ne01 = &smctx->div_ne01;
    ধ্রুবক কাঠামো fastdiv_মানসমূহ * div_ne02 = &smctx->div_ne02;
    ধ্রুবক কাঠামো fastdiv_মানসমূহ * div_ne12 = &smctx->div_ne12;
    ধ্রুবক কাঠামো fastdiv_মানসমূহ * div_ne13 = &smctx->div_ne13;

    dma_queue * dma_q = octx->ctx->dma[ith];

    জন্য (uint32_t r = src0_start_row, idx = 0; r < src0_end_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = তথ্য_dst  + r * dst_row_আকার;
        dma_addr_t cur_src0 = তথ্য_src0 + r * src0_row_আকার;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = তথ্য_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        শূন্য * d_spad = dst_vtcm_base  + idx * dst_vtcm_half;
        শূন্য * s_spad = src0_vtcm_base + idx * src0_vtcm_half;
        শূন্য * m_spad = src1_vtcm_base + idx * src1_vtcm_half;

        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(cur_dst, d_spad),
                       dst_row_আকার, smctx->dst_row_আকার_aligned, dst_row_আকার, 0);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s_spad, cur_src0),
                       smctx->src0_row_আকার_aligned, src0_row_আকার, src0_row_আকার, 1);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(m_spad, cur_src1),
                       smctx->src1_row_আকার_aligned, mask_row_আকার, mask_row_আকার, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ধ্রুবক bool has_bias = smctx->max_bias > 0.0f;
    uint32_t পূর্ব_i2 = (uint32_t)-1;
    দশমিক slope = 1.0f;

    জন্য (uint32_t r = src0_start_row; r < src0_end_row; ++r) {
        শূন্য * d_spad = (শূন্য *) (uintptr_t) dma_queue_বের_করো(dma_q).src;
        শূন্য * s_spad = (শূন্য *) (uintptr_t) dma_queue_বের_করো(dma_q).dst;
        শূন্য * m_spad = (শূন্য *) (uintptr_t) dma_queue_বের_করো(dma_q).dst;

        যদি (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            যদি (i2 != পূর্ব_i2) {
                slope = smctx->slopes[i2];
                পূর্ব_i2 = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, slope);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = তথ্য_dst + r * dst_row_আকার;
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(cur_dst, d_spad),
                       dst_row_আকার, smctx->dst_row_আকার_aligned, dst_row_আকার, 1);

        ধ্রুবক uint32_t পরবর্তী_r = r + 2;
        যদি (পরবর্তী_r < src0_end_row) {
            dma_addr_t পরবর্তী_src0 = তথ্য_src0 + পরবর্তী_r * src0_row_আকার;

            uint32_t ni1 = fastmodulo(পরবর্তী_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(পরবর্তী_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t পরবর্তী_src1 = তথ্য_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s_spad, পরবর্তী_src0),
                           smctx->src0_row_আকার_aligned, src0_row_আকার, src0_row_আকার, 1);
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(m_spad, পরবর্তী_src1),
                           smctx->src1_row_আকার_aligned, mask_row_আকার, mask_row_আকার, 1);
        }
    }

    dma_queue_flush(dma_q);
}

স্থির পূর্ণসংখ্যা execute_op_softmax_f32(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    ধ্রুবক অক্ষর * op_type = "softmax-f32";

    ধ্রুবক কাঠামো htp_softmax_kernel_params * kparams =
        (ধ্রুবক কাঠামো htp_softmax_kernel_params *) octx->kernel_params;

    যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    যদি (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_গুনতি) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    যদি (octx->ctx->vtcm_আকার < (size_t) kparams->vtcm_আকার) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_আকার, kparams->vtcm_আকার);
        ফেরত HTP_STATUS_VTCM_TOO_SMALL;
    }

    ধ্রুবক uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুবক size_t elem_আকার    = আকার(দশমিক);
    ধ্রুবক size_t dst_row_আকার = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_আকার, (uint32_t) dst_row_আকার, &rows_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
        যদি (nrows < octx->n_threads) {
            htp_ops_context_set_n_threads(octx, nrows ? nrows : 1);
        }
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;
    uint8_t * ধ্রুবক vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    ধ্রুবক uint32_t off_src0 = 0;
    ধ্রুবক uint32_t off_dst  = off_src0 + kparams->vtcm_src0_আকার_per_thread * kparams->n_threads;
    ধ্রুবক uint32_t off_src1 = off_dst  + kparams->vtcm_dst_আকার_per_thread  * kparams->n_threads;

    কাঠামো htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (শূন্য *) softmax_kernels[kparams->kernel_id],

        .তথ্য_src0             = src0->তথ্য,
        .তথ্য_src1             = kparams->use_src1 ? octx->src[1]->তথ্য : 0,
        .তথ্য_dst              = dst->তথ্য,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_base, off_src1, kparams->use_src1),

        .vtcm_src0_আকার_per_thread = kparams->vtcm_src0_আকার_per_thread,
        .vtcm_src1_আকার_per_thread = kparams->vtcm_src1_আকার_per_thread,
        .vtcm_dst_আকার_per_thread  = kparams->vtcm_dst_আকার_per_thread,

        .src0_spad_half_আকার   = kparams->src0_spad_half_আকার,
        .src1_spad_half_আকার   = kparams->src1_spad_half_আকার,
        .dst_spad_half_আকার    = kparams->dst_spad_half_আকার,

        .src0_row_আকার_aligned = kparams->src0_row_আকার_aligned,
        .src1_row_আকার_aligned = kparams->src1_row_আকার_aligned,
        .dst_row_আকার_aligned  = kparams->dst_row_আকার_aligned,

        .use_f16               = kparams->use_f16 != 0,

        .n_শীর্ষ                = kparams->n_শীর্ষ,
        .n_শীর্ষ_log2           = kparams->n_শীর্ষ_log2,

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

    যদি (kparams->max_bias > 0.0f && kparams->use_src1) {
        যদি (kparams->n_শীর্ষ > 512) {
            ফেরত HTP_STATUS_INVAL_PARAMS;
        }
        জন্য (uint32_t h = 0; h < kparams->n_শীর্ষ; h += 32) {
            HVX_Vector v_slopes = hvx_alibi_slopes(h, 1, kparams->n_শীর্ষ_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_queue_func_t task_func = kparams->use_src1 ? softmax_thread_mask_dma : softmax_thread_dma;
    work_queue_run(octx->ctx->work_queue, task_func, &smctx, n_threads);

    ফেরত HTP_STATUS_OK;
}

পূর্ণসংখ্যা op_softmax(কাঠামো htp_ops_context * octx) {
    পূর্ণসংখ্যা err = HTP_STATUS_OK;

    নির্বাচন (octx->src[0]->type) {
        ক্ষেত্র HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            বিরতি;

        পূর্বনির্ধারিত:
            err = HTP_STATUS_NO_SUPPORT;
            বিরতি;
    }

    ফেরত err;
}
