#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "hvx-types.h"
#ଅନ୍ତର୍ଭାଵ "hex-utils.h"
#ଅନ୍ତର୍ଭାଵ "hvx-copy.h"
#ଅନ୍ତର୍ଭାଵ "dma-queue.h"

#define htp_diag_tensors_preamble                           \
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0]; \
    ଅଚଳ ସଂରଚନା htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ଅଚଳ uint32_t ne02 = src0->ne[2];                      \
                                                            \
    ଅଚଳ uint32_t ne0 = dst->ne[0];                        \
    ଅଚଳ uint32_t ne1 = dst->ne[1];                        \
                                                            \
    ଅଚଳ uint32_t nb02 = src0->nb[2];                      \
    ଅଚଳ uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ଅଚଳ uint32_t nb1 = dst->nb[1];                        \
    ଅଚଳ uint32_t nb2 = dst->nb[2];                        \
    ଅଚଳ uint32_t nb3 = dst->nb[3];

ସଂରଚନା htp_diag_context {
    ସଂରଚନା htp_ops_context * octx;
    size_t          src_batch_size;
    size_t          dst_row_size;
    size_t          src_batch_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        batches_per_thread;
    uint32_t        total_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    ସଂରଚନା htp_diag_context * dctx = (ସଂରଚନା htp_diag_context *) data; \
    ସଂରଚନା htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_diag_row_f32(ଅଚଳ ଦଶମିକ * restrict src, ଦଶମିକ * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

ସ୍ଥିର ଶୂନ୍ୟ diag_thread_f32_dma(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    ଅଚଳ uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ଅଚଳ uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    ଯଦି (ib0 >= ib1) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ size_t src_batch_size         = dctx->src_batch_size;
    ଅଚଳ size_t dst_row_size           = dctx->dst_row_size;
    ଅଚଳ size_t src_batch_size_aligned = dctx->src_batch_size_aligned;
    ଅଚଳ size_t dst_row_size_aligned   = dctx->dst_row_size_aligned;

    ଅଚଳ dma_addr_t src_data = src0->data;
    ଅଚଳ dma_addr_t dst_data = dst->data;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ପାଇଁ (uint32_t ib = ib0; ib < ib1; ib++) {
        ଅଚଳ uint32_t i3 = ib / ne02;
        ଅଚଳ uint32_t i2 = ib % ne02;

        ଅଚଳ dma_addr_t src_batch = src_data + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_ଠେଲ(dma_q,
                       dma_make_data(src_spad, src_batch),
                       src_batch_size_aligned, src_batch_size, src_batch_size, 1);
        dma_queue_flush(dma_q);

        ଅଚଳ ଦଶମିକ * src_spad_f32 = (ଅଚଳ ଦଶମିକ *) src_spad;
        ଦଶମିକ       * dst_spad_f32 = (ଦଶମିକ *) dst_spad;

        ପାଇଁ (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ଅଚଳ dma_addr_t dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_ଠେଲ(dma_q,
                           dma_make_data(dst_row, dst_spad),
                           dst_row_size, dst_row_size_aligned, dst_row_size, 1);
            dma_queue_flush(dma_q);
        }
    }

    FARF(ଉପର_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ସ୍ଥିର ଶୂନ୍ୟ diag_thread_f32(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_diag_preamble;

    ଅଚଳ uint8_t * src_data = (ଅଚଳ uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    ଅଚଳ uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ଅଚଳ uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    ପାଇଁ (uint32_t ib = ib0; ib < ib1; ib++) {
        ଅଚଳ uint32_t i3 = ib / ne02;
        ଅଚଳ uint32_t i2 = ib % ne02;

        ଅଚଳ ଦଶମିକ * restrict src_batch = (ଅଚଳ ଦଶମିକ *)(src_data + i3 * nb03 + i2 * nb02);

        ପାଇଁ (uint32_t i1 = 0; i1 < ne1; i1++) {
            ଦଶମିକ * restrict dst_row = (ଦଶମିକ *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(ଉପର_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_diag_f32(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ଅଚଳ uint32_t total_batches = src0->ne[2] * src0->ne[3];
    ଅଚଳ size_t dst_batch_size  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == ଆକାର(ଦଶମିକ)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        ଯଦି (can_split) {
            ଯଦି (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) == 0)) {
                batches_per_chunk = 1;
            } ଅନ୍ୟଥା ଯଦି (dst->nb[2] == dst_batch_size &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_ଧାଡ଼ି_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_ଧାଡ଼ି_SIZE)) : 1;
            } ଅନ୍ୟଥା {
                can_split = false;
            }
        }

        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        batch_start = range.start;
        nbatches    = range.ଗଣନା_2;
    }

    ଯଦି (nbatches == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads     = octx->n_threads;

    ଅଚଳ size_t src_batch_size         = src0->ne[0] * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t dst_row_size           = dst->ne[0] * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
    ଅଚଳ size_t dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ଅଚଳ size_t spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;

    octx->src0_spad.size_per_thread = src_batch_size_aligned;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = ଖାଲି;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = ଖାଲି;

    ସଂରଚନା htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_size         = src_batch_size,
        .dst_row_size           = dst_row_size,
        .src_batch_size_aligned = src_batch_size_aligned,
        .dst_row_size_aligned   = dst_row_size_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .total_batches          = nbatches,
        .batch_start            = batch_start,
    };

    ଯଦି (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        ଯଦି (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } ଅନ୍ୟଥା {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_diag(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * dst = octx->dst;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = HTP_STATUS_OK;

    ବାଛ (dst->type) {
        ସ୍ଥିତି HTP_TYPE_F32:
            err = op_diag_f32(octx);
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            err = HTP_STATUS_NO_SUPPORT;
            ଭାଙ୍ଗ;
    }

    ଫେରନ୍ତୁ err;
}
