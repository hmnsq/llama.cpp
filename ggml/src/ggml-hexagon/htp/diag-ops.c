#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "hex-profile.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp-tensor.h"
#અન્તર્ભાવ "hvx-types.h"
#અન્તર્ભાવ "hex-utils.h"
#અન્તર્ભાવ "hvx-copy.h"
#અન્તર્ભાવ "dma-queue.h"

#define htp_diag_tensors_preamble                           \
    અચલ સંરચના htp_tensor * restrict src0 = octx->src[0]; \
    અચલ સંરચના htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    અચલ uint32_t ne02 = src0->ne[2];                      \
                                                            \
    અચલ uint32_t ne0 = dst->ne[0];                        \
    અચલ uint32_t ne1 = dst->ne[1];                        \
                                                            \
    અચલ uint32_t nb02 = src0->nb[2];                      \
    અચલ uint32_t nb03 = src0->nb[3];                      \
                                                            \
    અચલ uint32_t nb1 = dst->nb[1];                        \
    અચલ uint32_t nb2 = dst->nb[2];                        \
    અચલ uint32_t nb3 = dst->nb[3];

સંરચના htp_diag_context {
    સંરચના htp_ops_context * octx;
    size_t          src_batch_size;
    size_t          dst_row_size;
    size_t          src_batch_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        batches_per_thread;
    uint32_t        total_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    સંરચના htp_diag_context * dctx = (સંરચના htp_diag_context *) data; \
    સંરચના htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

સ્થિર અંતઃરેખ શૂન્ય hvx_diag_row_f32(અચલ દશાંશ * restrict src, દશાંશ * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

સ્થિર શૂન્ય diag_thread_f32_dma(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    અચલ uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    અચલ uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    જો (ib0 >= ib1) {
        પાછા;
    }

    અચલ size_t src_batch_size         = dctx->src_batch_size;
    અચલ size_t dst_row_size           = dctx->dst_row_size;
    અચલ size_t src_batch_size_aligned = dctx->src_batch_size_aligned;
    અચલ size_t dst_row_size_aligned   = dctx->dst_row_size_aligned;

    અચલ dma_addr_t src_data = src0->data;
    અચલ dma_addr_t dst_data = dst->data;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);

    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];

    માટે (uint32_t ib = ib0; ib < ib1; ib++) {
        અચલ uint32_t i3 = ib / ne02;
        અચલ uint32_t i2 = ib % ne02;

        અચલ dma_addr_t src_batch = src_data + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_ધકેલો(dma_q,
                       dma_make_data(src_spad, src_batch),
                       src_batch_size_aligned, src_batch_size, src_batch_size, 1);
        dma_queue_flush(dma_q);

        અચલ દશાંશ * src_spad_f32 = (અચલ દશાંશ *) src_spad;
        દશાંશ       * dst_spad_f32 = (દશાંશ *) dst_spad;

        માટે (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            અચલ dma_addr_t dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_ધકેલો(dma_q,
                           dma_make_data(dst_row, dst_spad),
                           dst_row_size, dst_row_size_aligned, dst_row_size, 1);
            dma_queue_flush(dma_q);
        }
    }

    FARF(ઉપલું_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

સ્થિર શૂન્ય diag_thread_f32(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    htp_diag_preamble;

    અચલ uint8_t * src_data = (અચલ uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    અચલ uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    અચલ uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    માટે (uint32_t ib = ib0; ib < ib1; ib++) {
        અચલ uint32_t i3 = ib / ne02;
        અચલ uint32_t i2 = ib % ne02;

        અચલ દશાંશ * restrict src_batch = (અચલ દશાંશ *)(src_data + i3 * nb03 + i2 * nb02);

        માટે (uint32_t i1 = 0; i1 < ne1; i1++) {
            દશાંશ * restrict dst_row = (દશાંશ *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(ઉપલું_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

પૂર્ણાંક op_diag_f32(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_tensor * src0 = octx->src[0];
    અચલ સંરચના htp_tensor * dst  = octx->dst;

    અચલ uint32_t total_batches = src0->ne[2] * src0->ne[3];
    અચલ size_t dst_batch_size  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    જો (octx->ctx->mdev.ગણતરી > 1) {
        bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == કદ(દશાંશ)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        જો (can_split) {
            જો (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) == 0)) {
                batches_per_chunk = 1;
            } નહીંતર જો (dst->nb[2] == dst_batch_size &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_લીટી_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_લીટી_SIZE)) : 1;
            } નહીંતર {
                can_split = false;
            }
        }

        અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
        batch_start = range.start;
        nbatches    = range.ગણતરી;
    }

    જો (nbatches == 0) {
        પાછા HTP_STATUS_OK;
    }

    અચલ uint32_t n_threads     = octx->n_threads;

    અચલ size_t src_batch_size         = src0->ne[0] * કદ(દશાંશ);
    અચલ size_t dst_row_size           = dst->ne[0] * કદ(દશાંશ);
    અચલ size_t src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
    અચલ size_t dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    અચલ size_t spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;

    octx->src0_spad.size_per_thread = src_batch_size_aligned;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = ખાલી;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = ખાલી;

    સંરચના htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_size         = src_batch_size,
        .dst_row_size           = dst_row_size,
        .src_batch_size_aligned = src_batch_size_aligned,
        .dst_row_size_aligned   = dst_row_size_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .total_batches          = nbatches,
        .batch_start            = batch_start,
    };

    જો (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        જો (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            પાછા HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } નહીંતર {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    પાછા HTP_STATUS_OK;
}

પૂર્ણાંક op_diag(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_tensor * dst = octx->dst;

    પૂર્ણાંક err = HTP_STATUS_OK;

    પસંદગી (dst->type) {
        સ્થિતિ HTP_TYPE_F32:
            err = op_diag_f32(octx);
            તોડો;
        મૂળભૂત:
            err = HTP_STATUS_NO_SUPPORT;
            તોડો;
    }

    પાછા err;
}
