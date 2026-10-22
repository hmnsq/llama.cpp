#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"
#அந்தர்பாவ "hvx-types.h"
#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "hvx-copy.h"
#அந்தர்பாவ "dma-queue.h"

#define htp_diag_tensors_preamble                           \
    மாறிலி கட்டமைப்பு htp_tensor * restrict src0 = octx->src[0]; \
    மாறிலி கட்டமைப்பு htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    மாறிலி uint32_t ne02 = src0->ne[2];                      \
                                                            \
    மாறிலி uint32_t ne0 = dst->ne[0];                        \
    மாறிலி uint32_t ne1 = dst->ne[1];                        \
                                                            \
    மாறிலி uint32_t nb02 = src0->nb[2];                      \
    மாறிலி uint32_t nb03 = src0->nb[3];                      \
                                                            \
    மாறிலி uint32_t nb1 = dst->nb[1];                        \
    மாறிலி uint32_t nb2 = dst->nb[2];                        \
    மாறிலி uint32_t nb3 = dst->nb[3];

கட்டமைப்பு htp_diag_context {
    கட்டமைப்பு htp_ops_context * octx;
    size_t          src_batch_size;
    size_t          dst_row_size;
    size_t          src_batch_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        batches_per_thread;
    uint32_t        total_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    கட்டமைப்பு htp_diag_context * dctx = (கட்டமைப்பு htp_diag_context *) data; \
    கட்டமைப்பு htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

நிலையான உள்வரி வெறுமை hvx_diag_row_f32(மாறிலி தசம * restrict src, தசம * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

நிலையான வெறுமை diag_thread_f32_dma(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    மாறிலி uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    மாறிலி uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    எனில் (ib0 >= ib1) {
        திரும்பு;
    }

    மாறிலி size_t src_batch_size         = dctx->src_batch_size;
    மாறிலி size_t dst_row_size           = dctx->dst_row_size;
    மாறிலி size_t src_batch_size_aligned = dctx->src_batch_size_aligned;
    மாறிலி size_t dst_row_size_aligned   = dctx->dst_row_size_aligned;

    மாறிலி dma_addr_t src_data = src0->data;
    மாறிலி dma_addr_t dst_data = dst->data;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];

    க்கு (uint32_t ib = ib0; ib < ib1; ib++) {
        மாறிலி uint32_t i3 = ib / ne02;
        மாறிலி uint32_t i2 = ib % ne02;

        மாறிலி dma_addr_t src_batch = src_data + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_தள்ளு(dma_q,
                       dma_make_data(src_spad, src_batch),
                       src_batch_size_aligned, src_batch_size, src_batch_size, 1);
        dma_queue_flush(dma_q);

        மாறிலி தசம * src_spad_f32 = (மாறிலி தசம *) src_spad;
        தசம       * dst_spad_f32 = (தசம *) dst_spad;

        க்கு (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            மாறிலி dma_addr_t dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_தள்ளு(dma_q,
                           dma_make_data(dst_row, dst_spad),
                           dst_row_size, dst_row_size_aligned, dst_row_size, 1);
            dma_queue_flush(dma_q);
        }
    }

    FARF(உயர்வு_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

நிலையான வெறுமை diag_thread_f32(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    htp_diag_preamble;

    மாறிலி uint8_t * src_data = (மாறிலி uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    மாறிலி uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    மாறிலி uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    க்கு (uint32_t ib = ib0; ib < ib1; ib++) {
        மாறிலி uint32_t i3 = ib / ne02;
        மாறிலி uint32_t i2 = ib % ne02;

        மாறிலி தசம * restrict src_batch = (மாறிலி தசம *)(src_data + i3 * nb03 + i2 * nb02);

        க்கு (uint32_t i1 = 0; i1 < ne1; i1++) {
            தசம * restrict dst_row = (தசம *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(உயர்வு_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

முழு op_diag_f32(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    மாறிலி uint32_t total_batches = src0->ne[2] * src0->ne[3];
    மாறிலி size_t dst_batch_size  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == அளவு(தசம)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        எனில் (can_split) {
            எனில் (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) == 0)) {
                batches_per_chunk = 1;
            } இல்லையெனில் எனில் (dst->nb[2] == dst_batch_size &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_வரி_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_வரி_SIZE)) : 1;
            } இல்லையெனில் {
                can_split = false;
            }
        }

        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        batch_start = range.start;
        nbatches    = range.மொத்தம்;
    }

    எனில் (nbatches == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    மாறிலி uint32_t n_threads     = octx->n_threads;

    மாறிலி size_t src_batch_size         = src0->ne[0] * அளவு(தசம);
    மாறிலி size_t dst_row_size           = dst->ne[0] * அளவு(தசம);
    மாறிலி size_t src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
    மாறிலி size_t dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    மாறிலி size_t spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;

    octx->src0_spad.size_per_thread = src_batch_size_aligned;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = வெற்று;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = வெற்று;

    கட்டமைப்பு htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_size         = src_batch_size,
        .dst_row_size           = dst_row_size,
        .src_batch_size_aligned = src_batch_size_aligned,
        .dst_row_size_aligned   = dst_row_size_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .total_batches          = nbatches,
        .batch_start            = batch_start,
    };

    எனில் (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        எனில் (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            திரும்பு HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } இல்லையெனில் {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    திரும்பு HTP_STATUS_OK;
}

முழு op_diag(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_tensor * dst = octx->dst;

    முழு err = HTP_STATUS_OK;

    தேர்வு (dst->type) {
        நிலை HTP_TYPE_F32:
            err = op_diag_f32(octx);
            நிறுத்து;
        இயல்பு:
            err = HTP_STATUS_NO_SUPPORT;
            நிறுத்து;
    }

    திரும்பு err;
}
