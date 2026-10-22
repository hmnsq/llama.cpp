#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "hex-common.h"
#समावेश "hex-profile.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "hvx-types.h"
#समावेश "hvx-utils.h"
#समावेश "dma-queue.h"

#घोषणा htp_cumsum_tensors_preamble                         \
    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0]; \
    स्थिर काष्ठा htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    स्थिर uint32_t ne00 = src0->ne[0];                      \
    स्थिर uint32_t ne01 = src0->ne[1];                      \
    स्थिर uint32_t ne02 = src0->ne[2];                      \
    स्थिर uint32_t ne03 = src0->ne[3];                      \
                                                            \
    स्थिर uint32_t ne0 = dst->ne[0];                        \
    स्थिर uint32_t ne1 = dst->ne[1];                        \
    स्थिर uint32_t ne2 = dst->ne[2];                        \
    स्थिर uint32_t ne3 = dst->ne[3];                        \
                                                            \
    स्थिर uint32_t nb00 = src0->nb[0];                      \
    स्थिर uint32_t nb01 = src0->nb[1];                      \
    स्थिर uint32_t nb02 = src0->nb[2];                      \
    स्थिर uint32_t nb03 = src0->nb[3];                      \
                                                            \
    स्थिर uint32_t nb0 = dst->nb[0];                        \
    स्थिर uint32_t nb1 = dst->nb[1];                        \
    स्थिर uint32_t nb2 = dst->nb[2];                        \
    स्थिर uint32_t nb3 = dst->nb[3];

काष्ठा htp_cumsum_context {
    काष्ठा htp_ops_context * octx;
    माप_प्रकार          src_row_आकार;
    माप_प्रकार          dst_row_आकार;
    माप_प्रकार          src_row_आकार_aligned;
    माप_प्रकार          dst_row_आकार_aligned;
    uint32_t        rows_per_thread;
    uint32_t        योग_rows;
    uint32_t        row_start;
};

#घोषणा htp_cumsum_preamble                                                \
    काष्ठा htp_cumsum_context * cctx = (काष्ठा htp_cumsum_context *) तथ्य; \
    काष्ठा htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#अगर __HVX_ARCH__ > 75
अटल अंतरभूत HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    वापस Q6_Vsf_vadd_VsfVsf(a, b);
}
#अन्यथा
अटल अंतरभूत HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    वापस Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#पूर्ण_अगर  // __HVX_ARCH__ > 75

अटल अंतरभूत HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    स्थिर HVX_Vector zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_in);

    वापस v;
}

अटल अंतरभूत HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    वापस hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

अटल अंतरभूत व्योम hvx_cumsum_row_f32(स्थिर भग्न * restrict src, भग्न * restrict dst, uint32_t n) {
    स्थिर uint32_t nvec = n / VLEN_FP32;
    स्थिर uint32_t nloe = n % VLEN_FP32;

    HVX_Vector carry = Q6_V_vsplat_R(0);

    क्रम (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = *((स्थिर HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        carry = hvx_splat_last_f32(v);
    }

    अगर (nloe) {
        भग्न acc = hvx_vec_get_f32(carry);
        स्थिर भग्न * src_पुच्छ = src + nvec * VLEN_FP32;
        भग्न       * dst_पुच्छ = dst + nvec * VLEN_FP32;
        क्रम (uint32_t i = 0; i < nloe; i++) {
            acc        += src_पुच्छ[i];
            dst_पुच्छ[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

अटल व्योम cumsum_thread_f32_dma(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_cumsum_preamble;

    स्थिर uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    स्थिर uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->योग_rows);

    अगर (ir0 >= ir1) {
        वापस;
    }

    स्थिर माप_प्रकार src_row_आकार         = cctx->src_row_आकार;
    स्थिर माप_प्रकार dst_row_आकार         = cctx->dst_row_आकार;
    स्थिर माप_प्रकार src_row_आकार_aligned = cctx->src_row_आकार_aligned;
    स्थिर माप_प्रकार dst_row_आकार_aligned = cctx->dst_row_आकार_aligned;

    स्थिर dma_addr_t src_तथ्य = src0->तथ्य;
    स्थिर dma_addr_t dst_तथ्य = dst->तथ्य;

    uint8_t * src_spad = octx->src0_spad.तथ्य + (ith * src_row_आकार_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.तथ्य  + (ith * dst_row_आकार_aligned * 2);

    क्रम (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(dst_तथ्य, dst_spad + (spad_idx * dst_row_आकार_aligned)),
                       dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 0);

        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(src_spad + (spad_idx * src_row_आकार_aligned),
                                     src_तथ्य + (ir * src_row_आकार)),
                       src_row_आकार_aligned, src_row_आकार, src_row_आकार, 1);
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    क्रम (uint32_t ir = ir0; ir < ir1; ir++) {
        भग्न * dst_spad_row = (भग्न *) dma_queue_निकालो(dma_q).src;
        भग्न * src_spad_row = (भग्न *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_row_f32(src_spad_row, dst_spad_row, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(dst_तथ्य + (ir * dst_row_आकार), dst_spad_row),
                       dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 1);

        स्थिर uint32_t अग्र_row = ir + 2;
        अगर (अग्र_row < ir1) {
            dma_queue_धकेलो(dma_q,
                           dma_make_तथ्य(src_spad_row, src_तथ्य + (अग्र_row * src_row_आकार)),
                           src_row_आकार_aligned, src_row_आकार, src_row_आकार, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(ऊपरी_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

अटल व्योम cumsum_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_cumsum_preamble;

    स्थिर uint8_t * src_तथ्य = (स्थिर uint8_t *) src0->तथ्य;
    uint8_t *       dst_तथ्य = (uint8_t *) dst->तथ्य;

    स्थिर uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    स्थिर uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->योग_rows);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    क्रम (uint32_t ir = ir0; ir < ir1; ir++) {
        स्थिर भग्न * restrict src_row = (स्थिर भग्न *) (src_तथ्य + ir * cctx->src_row_आकार);
        भग्न       * restrict dst_row = (भग्न *)       (dst_तथ्य + ir * cctx->dst_row_आकार);
        hvx_cumsum_row_f32(src_row, dst_row, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(ऊपरी_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

पूर्णांक op_cumsum_f32(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    स्थिर uint32_t योग_rows      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर माप_प्रकार dst_तथ्य_row_आकार = dst->ne[0] * माप(भग्न);

    uint32_t row_start = 0;
    uint32_t nrows     = योग_rows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, माप(भग्न), (uint32_t) dst_तथ्य_row_आकार, &rows_per_chunk);
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t n_threads = octx->n_threads;

    स्थिर माप_प्रकार src_row_आकार         = src0->nb[1];
    स्थिर माप_प्रकार dst_row_आकार         = dst->nb[1];
    स्थिर माप_प्रकार src_row_आकार_aligned = hex_round_up(src_row_आकार, VLEN);
    स्थिर माप_प्रकार dst_row_आकार_aligned = hex_round_up(dst_row_आकार, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    स्थिर माप_प्रकार spad_per_thread = 2 * (src_row_आकार_aligned + dst_row_आकार_aligned);

    octx->src0_spad.आकार_per_thread = src_row_आकार_aligned * 2;
    octx->dst_spad.आकार_per_thread  = dst_row_आकार_aligned * 2;

    octx->src0_spad.आकार  = n_threads * octx->src0_spad.आकार_per_thread;
    octx->dst_spad.आकार   = n_threads * octx->dst_spad.आकार_per_thread;

    octx->src0_spad.तथ्य  = octx->ctx->vtcm_base;                        octx->src0_spad.src = शून्य;
    octx->dst_spad.तथ्य   = octx->src0_spad.तथ्य + octx->src0_spad.आकार; octx->dst_spad.src  = शून्य;

    काष्ठा htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_row_आकार         = src_row_आकार,
        .dst_row_आकार         = dst_row_आकार,
        .src_row_आकार_aligned = src_row_आकार_aligned,
        .dst_row_आकार_aligned = dst_row_आकार_aligned,
        .rows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .योग_rows           = nrows,
        .row_start            = row_start,
    };

    अगर (octx->ctx->vtcm_आकार < spad_per_thread * n_threads) {
        अगर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            वापस HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32, &cctx, n_threads);
    } अन्यथा {
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32_dma, &cctx, n_threads);
    }

    वापस HTP_STATUS_OK;
}

पूर्णांक op_cumsum(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * dst = octx->dst;

    चयन (dst->type) {
        हाल HTP_TYPE_F32:
            वापस op_cumsum_f32(octx);
        शेष:
            वापस HTP_STATUS_NO_SUPPORT;
    }
}
