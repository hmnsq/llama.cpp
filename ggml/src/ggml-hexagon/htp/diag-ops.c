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
#समावेश "hex-utils.h"
#समावेश "hvx-copy.h"
#समावेश "dma-queue.h"

#घोषणा htp_diag_tensors_preamble                           \
    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0]; \
    स्थिर काष्ठा htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    स्थिर uint32_t ne02 = src0->ne[2];                      \
                                                            \
    स्थिर uint32_t ne0 = dst->ne[0];                        \
    स्थिर uint32_t ne1 = dst->ne[1];                        \
                                                            \
    स्थिर uint32_t nb02 = src0->nb[2];                      \
    स्थिर uint32_t nb03 = src0->nb[3];                      \
                                                            \
    स्थिर uint32_t nb1 = dst->nb[1];                        \
    स्थिर uint32_t nb2 = dst->nb[2];                        \
    स्थिर uint32_t nb3 = dst->nb[3];

काष्ठा htp_diag_context {
    काष्ठा htp_ops_context * octx;
    माप_प्रकार          src_batch_आकार;
    माप_प्रकार          dst_row_आकार;
    माप_प्रकार          src_batch_आकार_aligned;
    माप_प्रकार          dst_row_आकार_aligned;
    uint32_t        batches_per_thread;
    uint32_t        योग_batches;
    uint32_t        batch_start;
};

#घोषणा htp_diag_preamble                                              \
    काष्ठा htp_diag_context * dctx = (काष्ठा htp_diag_context *) तथ्य; \
    काष्ठा htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

अटल अंतरभूत व्योम hvx_diag_row_f32(स्थिर भग्न * restrict src, भग्न * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

अटल व्योम diag_thread_f32_dma(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    स्थिर uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    स्थिर uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->योग_batches);

    अगर (ib0 >= ib1) {
        वापस;
    }

    स्थिर माप_प्रकार src_batch_आकार         = dctx->src_batch_आकार;
    स्थिर माप_प्रकार dst_row_आकार           = dctx->dst_row_आकार;
    स्थिर माप_प्रकार src_batch_आकार_aligned = dctx->src_batch_आकार_aligned;
    स्थिर माप_प्रकार dst_row_आकार_aligned   = dctx->dst_row_आकार_aligned;

    स्थिर dma_addr_t src_तथ्य = src0->तथ्य;
    स्थिर dma_addr_t dst_तथ्य = dst->तथ्य;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.तथ्य + (ith * src_batch_आकार_aligned);
    uint8_t * dst_spad = octx->dst_spad.तथ्य  + (ith * dst_row_आकार_aligned);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    क्रम (uint32_t ib = ib0; ib < ib1; ib++) {
        स्थिर uint32_t i3 = ib / ne02;
        स्थिर uint32_t i2 = ib % ne02;

        स्थिर dma_addr_t src_batch = src_तथ्य + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_धकेलो(dma_q,
                       dma_make_तथ्य(src_spad, src_batch),
                       src_batch_आकार_aligned, src_batch_आकार, src_batch_आकार, 1);
        dma_queue_flush(dma_q);

        स्थिर भग्न * src_spad_f32 = (स्थिर भग्न *) src_spad;
        भग्न       * dst_spad_f32 = (भग्न *) dst_spad;

        क्रम (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            स्थिर dma_addr_t dst_row = dst_तथ्य + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_धकेलो(dma_q,
                           dma_make_तथ्य(dst_row, dst_spad),
                           dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 1);
            dma_queue_flush(dma_q);
        }
    }

    FARF(ऊपरी_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

अटल व्योम diag_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_diag_preamble;

    स्थिर uint8_t * src_तथ्य = (स्थिर uint8_t *) src0->तथ्य;
    uint8_t *       dst_तथ्य = (uint8_t *) dst->तथ्य;

    स्थिर uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    स्थिर uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->योग_batches);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    क्रम (uint32_t ib = ib0; ib < ib1; ib++) {
        स्थिर uint32_t i3 = ib / ne02;
        स्थिर uint32_t i2 = ib % ne02;

        स्थिर भग्न * restrict src_batch = (स्थिर भग्न *)(src_तथ्य + i3 * nb03 + i2 * nb02);

        क्रम (uint32_t i1 = 0; i1 < ne1; i1++) {
            भग्न * restrict dst_row = (भग्न *)(dst_तथ्य + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(ऊपरी_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

पूर्णांक op_diag_f32(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    स्थिर uint32_t योग_batches = src0->ne[2] * src0->ne[3];
    स्थिर माप_प्रकार dst_batch_आकार  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = योग_batches;

    अगर (octx->ctx->mdev.गिनती > 1) {
        bool can_split = htp_tensor_mdev_तथ्य_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == माप(भग्न)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        अगर (can_split) {
            अगर (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) == 0)) {
                batches_per_chunk = 1;
            } अन्यथा अगर (dst->nb[2] == dst_batch_आकार &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_आकार > 0) ? (HEX_L2_पंक्ति_आकार / hex_gcd_u32(dst_batch_आकार, HEX_L2_पंक्ति_आकार)) : 1;
            } अन्यथा {
                can_split = false;
            }
        }

        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        batch_start = range.start;
        nbatches    = range.गिनती;
    }

    अगर (nbatches == 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t n_threads     = octx->n_threads;

    स्थिर माप_प्रकार src_batch_आकार         = src0->ne[0] * माप(भग्न);
    स्थिर माप_प्रकार dst_row_आकार           = dst->ne[0] * माप(भग्न);
    स्थिर माप_प्रकार src_batch_आकार_aligned = hex_round_up(src_batch_आकार, VLEN);
    स्थिर माप_प्रकार dst_row_आकार_aligned   = hex_round_up(dst_row_आकार, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    स्थिर माप_प्रकार spad_per_thread = src_batch_आकार_aligned + dst_row_आकार_aligned;

    octx->src0_spad.आकार_per_thread = src_batch_आकार_aligned;
    octx->dst_spad.आकार_per_thread  = dst_row_आकार_aligned;

    octx->src0_spad.आकार = n_threads * octx->src0_spad.आकार_per_thread;
    octx->dst_spad.आकार  = n_threads * octx->dst_spad.आकार_per_thread;

    octx->src0_spad.तथ्य = octx->ctx->vtcm_base;                        octx->src0_spad.src = शून्य;
    octx->dst_spad.तथ्य  = octx->src0_spad.तथ्य + octx->src0_spad.आकार; octx->dst_spad.src  = शून्य;

    काष्ठा htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_आकार         = src_batch_आकार,
        .dst_row_आकार           = dst_row_आकार,
        .src_batch_आकार_aligned = src_batch_आकार_aligned,
        .dst_row_आकार_aligned   = dst_row_आकार_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .योग_batches          = nbatches,
        .batch_start            = batch_start,
    };

    अगर (octx->ctx->vtcm_आकार < spad_per_thread * n_threads) {
        अगर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            वापस HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } अन्यथा {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    वापस HTP_STATUS_OK;
}

पूर्णांक op_diag(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * dst = octx->dst;

    पूर्णांक err = HTP_STATUS_OK;

    चयन (dst->type) {
        हाल HTP_TYPE_F32:
            err = op_diag_f32(octx);
            अवरोध;
        शेष:
            err = HTP_STATUS_NO_SUPPORT;
            अवरोध;
    }

    वापस err;
}
