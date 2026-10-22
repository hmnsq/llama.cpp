#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "hvx-types.h"
#অন্তর্ভাব "hex-utils.h"
#অন্তর্ভাব "hvx-copy.h"
#অন্তর্ভাব "dma-queue.h"

#define htp_diag_tensors_preamble                           \
    ধ্রুবক কাঠামো htp_tensor * restrict src0 = octx->src[0]; \
    ধ্রুবক কাঠামো htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ধ্রুবক uint32_t ne02 = src0->ne[2];                      \
                                                            \
    ধ্রুবক uint32_t ne0 = dst->ne[0];                        \
    ধ্রুবক uint32_t ne1 = dst->ne[1];                        \
                                                            \
    ধ্রুবক uint32_t nb02 = src0->nb[2];                      \
    ধ্রুবক uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ধ্রুবক uint32_t nb1 = dst->nb[1];                        \
    ধ্রুবক uint32_t nb2 = dst->nb[2];                        \
    ধ্রুবক uint32_t nb3 = dst->nb[3];

কাঠামো htp_diag_context {
    কাঠামো htp_ops_context * octx;
    size_t          src_batch_আকার;
    size_t          dst_row_আকার;
    size_t          src_batch_আকার_aligned;
    size_t          dst_row_আকার_aligned;
    uint32_t        batches_per_thread;
    uint32_t        যোগ_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    কাঠামো htp_diag_context * dctx = (কাঠামো htp_diag_context *) তথ্য; \
    কাঠামো htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

স্থির অন্তঃরেখ শূন্য hvx_diag_row_f32(ধ্রুবক দশমিক * restrict src, দশমিক * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

স্থির শূন্য diag_thread_f32_dma(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    ধ্রুবক uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ধ্রুবক uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->যোগ_batches);

    যদি (ib0 >= ib1) {
        ফেরত;
    }

    ধ্রুবক size_t src_batch_আকার         = dctx->src_batch_আকার;
    ধ্রুবক size_t dst_row_আকার           = dctx->dst_row_আকার;
    ধ্রুবক size_t src_batch_আকার_aligned = dctx->src_batch_আকার_aligned;
    ধ্রুবক size_t dst_row_আকার_aligned   = dctx->dst_row_আকার_aligned;

    ধ্রুবক dma_addr_t src_তথ্য = src0->তথ্য;
    ধ্রুবক dma_addr_t dst_তথ্য = dst->তথ্য;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.তথ্য + (ith * src_batch_আকার_aligned);
    uint8_t * dst_spad = octx->dst_spad.তথ্য  + (ith * dst_row_আকার_aligned);

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];

    জন্য (uint32_t ib = ib0; ib < ib1; ib++) {
        ধ্রুবক uint32_t i3 = ib / ne02;
        ধ্রুবক uint32_t i2 = ib % ne02;

        ধ্রুবক dma_addr_t src_batch = src_তথ্য + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(src_spad, src_batch),
                       src_batch_আকার_aligned, src_batch_আকার, src_batch_আকার, 1);
        dma_queue_flush(dma_q);

        ধ্রুবক দশমিক * src_spad_f32 = (ধ্রুবক দশমিক *) src_spad;
        দশমিক       * dst_spad_f32 = (দশমিক *) dst_spad;

        জন্য (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ধ্রুবক dma_addr_t dst_row = dst_তথ্য + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_ঠেলো(dma_q,
                           dma_make_তথ্য(dst_row, dst_spad),
                           dst_row_আকার, dst_row_আকার_aligned, dst_row_আকার, 1);
            dma_queue_flush(dma_q);
        }
    }

    FARF(উপরের_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

স্থির শূন্য diag_thread_f32(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    htp_diag_preamble;

    ধ্রুবক uint8_t * src_তথ্য = (ধ্রুবক uint8_t *) src0->তথ্য;
    uint8_t *       dst_তথ্য = (uint8_t *) dst->তথ্য;

    ধ্রুবক uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ধ্রুবক uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->যোগ_batches);

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    জন্য (uint32_t ib = ib0; ib < ib1; ib++) {
        ধ্রুবক uint32_t i3 = ib / ne02;
        ধ্রুবক uint32_t i2 = ib % ne02;

        ধ্রুবক দশমিক * restrict src_batch = (ধ্রুবক দশমিক *)(src_তথ্য + i3 * nb03 + i2 * nb02);

        জন্য (uint32_t i1 = 0; i1 < ne1; i1++) {
            দশমিক * restrict dst_row = (দশমিক *)(dst_তথ্য + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(উপরের_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

পূর্ণসংখ্যা op_diag_f32(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    ধ্রুবক uint32_t যোগ_batches = src0->ne[2] * src0->ne[3];
    ধ্রুবক size_t dst_batch_আকার  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = যোগ_batches;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        bool can_split = htp_tensor_mdev_তথ্য_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == আকার(দশমিক)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        যদি (can_split) {
            যদি (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_সারি_আকার - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_সারি_আকার - 1)) == 0)) {
                batches_per_chunk = 1;
            } অন্যথা যদি (dst->nb[2] == dst_batch_আকার &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_আকার > 0) ? (HEX_L2_সারি_আকার / hex_gcd_u32(dst_batch_আকার, HEX_L2_সারি_আকার)) : 1;
            } অন্যথা {
                can_split = false;
            }
        }

        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(যোগ_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        batch_start = range.start;
        nbatches    = range.গুনতি;
    }

    যদি (nbatches == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads     = octx->n_threads;

    ধ্রুবক size_t src_batch_আকার         = src0->ne[0] * আকার(দশমিক);
    ধ্রুবক size_t dst_row_আকার           = dst->ne[0] * আকার(দশমিক);
    ধ্রুবক size_t src_batch_আকার_aligned = hex_round_up(src_batch_আকার, VLEN);
    ধ্রুবক size_t dst_row_আকার_aligned   = hex_round_up(dst_row_আকার, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ধ্রুবক size_t spad_per_thread = src_batch_আকার_aligned + dst_row_আকার_aligned;

    octx->src0_spad.আকার_per_thread = src_batch_আকার_aligned;
    octx->dst_spad.আকার_per_thread  = dst_row_আকার_aligned;

    octx->src0_spad.আকার_2 = n_threads * octx->src0_spad.আকার_per_thread;
    octx->dst_spad.আকার_2  = n_threads * octx->dst_spad.আকার_per_thread;

    octx->src0_spad.তথ্য = octx->ctx->vtcm_base;                        octx->src0_spad.src = ফাঁকা;
    octx->dst_spad.তথ্য  = octx->src0_spad.তথ্য + octx->src0_spad.আকার_2; octx->dst_spad.src  = ফাঁকা;

    কাঠামো htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_আকার         = src_batch_আকার,
        .dst_row_আকার           = dst_row_আকার,
        .src_batch_আকার_aligned = src_batch_আকার_aligned,
        .dst_row_আকার_aligned   = dst_row_আকার_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .যোগ_batches          = nbatches,
        .batch_start            = batch_start,
    };

    যদি (octx->ctx->vtcm_আকার < spad_per_thread * n_threads) {
        যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ফেরত HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } অন্যথা {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    ফেরত HTP_STATUS_OK;
}

পূর্ণসংখ্যা op_diag(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * dst = octx->dst;

    পূর্ণসংখ্যা err = HTP_STATUS_OK;

    নির্বাচন (dst->type) {
        ক্ষেত্র HTP_TYPE_F32:
            err = op_diag_f32(octx);
            বিরতি;
        পূর্বনির্ধারিত:
            err = HTP_STATUS_NO_SUPPORT;
            বিরতি;
    }

    ফেরত err;
}
