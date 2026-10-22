#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hvx-types.h"
#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-copy.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"

#define htp_diag_tensors_preamble                           \
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * restrict src0 = octx->src[0]; \
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ಧ್ರುವ uint32_t ne02 = src0->ne[2];                      \
                                                            \
    ಧ್ರುವ uint32_t ne0 = dst->ne[0];                        \
    ಧ್ರುವ uint32_t ne1 = dst->ne[1];                        \
                                                            \
    ಧ್ರುವ uint32_t nb02 = src0->nb[2];                      \
    ಧ್ರುವ uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ಧ್ರುವ uint32_t nb1 = dst->nb[1];                        \
    ಧ್ರುವ uint32_t nb2 = dst->nb[2];                        \
    ಧ್ರುವ uint32_t nb3 = dst->nb[3];

ಸಂರಚನಾ htp_diag_context {
    ಸಂರಚನಾ htp_ops_context * octx;
    size_t          src_batch_size;
    size_t          dst_row_size;
    size_t          src_batch_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        batches_per_thread;
    uint32_t        total_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    ಸಂರಚನಾ htp_diag_context * dctx = (ಸಂರಚನಾ htp_diag_context *) data; \
    ಸಂರಚನಾ htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hvx_diag_row_f32(ಧ್ರುವ ಭಿನ್ನ * restrict src, ಭಿನ್ನ * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ diag_thread_f32_dma(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    ಧ್ರುವ uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ಧ್ರುವ uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    ಯದಿ (ib0 >= ib1) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ size_t src_batch_size         = dctx->src_batch_size;
    ಧ್ರುವ size_t dst_row_size           = dctx->dst_row_size;
    ಧ್ರುವ size_t src_batch_size_aligned = dctx->src_batch_size_aligned;
    ಧ್ರುವ size_t dst_row_size_aligned   = dctx->dst_row_size_aligned;

    ಧ್ರುವ dma_addr_t src_data = src0->data;
    ಧ್ರುವ dma_addr_t dst_data = dst->data;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);

    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಪ್ರತಿ (uint32_t ib = ib0; ib < ib1; ib++) {
        ಧ್ರುವ uint32_t i3 = ib / ne02;
        ಧ್ರುವ uint32_t i2 = ib % ne02;

        ಧ್ರುವ dma_addr_t src_batch = src_data + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_ತಳ್ಳು(dma_q,
                       dma_make_data(src_spad, src_batch),
                       src_batch_size_aligned, src_batch_size, src_batch_size, 1);
        dma_queue_flush(dma_q);

        ಧ್ರುವ ಭಿನ್ನ * src_spad_f32 = (ಧ್ರುವ ಭಿನ್ನ *) src_spad;
        ಭಿನ್ನ       * dst_spad_f32 = (ಭಿನ್ನ *) dst_spad;

        ಪ್ರತಿ (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ಧ್ರುವ dma_addr_t dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_ತಳ್ಳು(dma_q,
                           dma_make_data(dst_row, dst_spad),
                           dst_row_size, dst_row_size_aligned, dst_row_size, 1);
            dma_queue_flush(dma_q);
        }
    }

    FARF(ಮೇಲಿನ_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ diag_thread_f32(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    htp_diag_preamble;

    ಧ್ರುವ uint8_t * src_data = (ಧ್ರುವ uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    ಧ್ರುವ uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ಧ್ರುವ uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    ಪ್ರತಿ (uint32_t ib = ib0; ib < ib1; ib++) {
        ಧ್ರುವ uint32_t i3 = ib / ne02;
        ಧ್ರುವ uint32_t i2 = ib % ne02;

        ಧ್ರುವ ಭಿನ್ನ * restrict src_batch = (ಧ್ರುವ ಭಿನ್ನ *)(src_data + i3 * nb03 + i2 * nb02);

        ಪ್ರತಿ (uint32_t i1 = 0; i1 < ne1; i1++) {
            ಭಿನ್ನ * restrict dst_row = (ಭಿನ್ನ *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(ಮೇಲಿನ_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

ಪೂರ್ಣಾಙ್ಕ op_diag_f32(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    ಧ್ರುವ uint32_t total_batches = src0->ne[2] * src0->ne[3];
    ಧ್ರುವ size_t dst_batch_size  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == ಪರಿಮಾಣ(ಭಿನ್ನ)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        ಯದಿ (can_split) {
            ಯದಿ (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) == 0)) {
                batches_per_chunk = 1;
            } ಅನ್ಯಥಾ ಯದಿ (dst->nb[2] == dst_batch_size &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_ಸಾಲು_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_ಸಾಲು_SIZE)) : 1;
            } ಅನ್ಯಥಾ {
                can_split = false;
            }
        }

        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        batch_start = range.start;
        nbatches    = range.ಎಣಿಕೆ;
    }

    ಯದಿ (nbatches == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಧ್ರುವ uint32_t n_threads     = octx->n_threads;

    ಧ್ರುವ size_t src_batch_size         = src0->ne[0] * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ size_t dst_row_size           = dst->ne[0] * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ size_t src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
    ಧ್ರುವ size_t dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ಧ್ರುವ size_t spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;

    octx->src0_spad.size_per_thread = src_batch_size_aligned;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = ರಿಕ್ತ;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = ರಿಕ್ತ;

    ಸಂರಚನಾ htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_size         = src_batch_size,
        .dst_row_size           = dst_row_size,
        .src_batch_size_aligned = src_batch_size_aligned,
        .dst_row_size_aligned   = dst_row_size_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .total_batches          = nbatches,
        .batch_start            = batch_start,
    };

    ಯದಿ (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        ಯದಿ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } ಅನ್ಯಥಾ {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಪೂರ್ಣಾಙ್ಕ op_diag(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst = octx->dst;

    ಪೂರ್ಣಾಙ್ಕ err = HTP_STATUS_OK;

    ವಿಕಲ್ಪ (dst->type) {
        ಪಕ್ಷ HTP_TYPE_F32:
            err = op_diag_f32(octx);
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            err = HTP_STATUS_NO_SUPPORT;
            ಭಙ್ಗ;
    }

    ಪ್ರತ್ಯಾಗಮ err;
}
