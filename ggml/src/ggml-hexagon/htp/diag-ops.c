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
    ধ্রুব সংরচনা htp_tensor * restrict src0 = octx->src[0]; \
    ধ্রুব সংরচনা htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ধ্রুব uint32_t ne02 = src0->ne[2];                      \
                                                            \
    ধ্রুব uint32_t ne0 = dst->ne[0];                        \
    ধ্রুব uint32_t ne1 = dst->ne[1];                        \
                                                            \
    ধ্রুব uint32_t nb02 = src0->nb[2];                      \
    ধ্রুব uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ধ্রুব uint32_t nb1 = dst->nb[1];                        \
    ধ্রুব uint32_t nb2 = dst->nb[2];                        \
    ধ্রুব uint32_t nb3 = dst->nb[3];

সংরচনা htp_diag_context {
    সংরচনা htp_ops_context * octx;
    size_t          src_batch_size;
    size_t          dst_row_size;
    size_t          src_batch_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        batches_per_thread;
    uint32_t        total_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    সংরচনা htp_diag_context * dctx = (সংরচনা htp_diag_context *) data; \
    সংরচনা htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

স্থির অন্তঃস্থ শূন্য hvx_diag_row_f32(ধ্রুব ভিন্ন * restrict src, ভিন্ন * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

স্থির শূন্য diag_thread_f32_dma(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    ধ্রুব uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ধ্রুব uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    যদি (ib0 >= ib1) {
        প্রত্যাগম;
    }

    ধ্রুব size_t src_batch_size         = dctx->src_batch_size;
    ধ্রুব size_t dst_row_size           = dctx->dst_row_size;
    ধ্রুব size_t src_batch_size_aligned = dctx->src_batch_size_aligned;
    ধ্রুব size_t dst_row_size_aligned   = dctx->dst_row_size_aligned;

    ধ্রুব dma_addr_t src_data = src0->data;
    ধ্রুব dma_addr_t dst_data = dst->data;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    প্রতি (uint32_t ib = ib0; ib < ib1; ib++) {
        ধ্রুব uint32_t i3 = ib / ne02;
        ধ্রুব uint32_t i2 = ib % ne02;

        ধ্রুব dma_addr_t src_batch = src_data + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_ঠেলো(dma_q,
                       dma_make_data(src_spad, src_batch),
                       src_batch_size_aligned, src_batch_size, src_batch_size, 1);
        dma_queue_flush(dma_q);

        ধ্রুব ভিন্ন * src_spad_f32 = (ধ্রুব ভিন্ন *) src_spad;
        ভিন্ন       * dst_spad_f32 = (ভিন্ন *) dst_spad;

        প্রতি (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ধ্রুব dma_addr_t dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_ঠেলো(dma_q,
                           dma_make_data(dst_row, dst_spad),
                           dst_row_size, dst_row_size_aligned, dst_row_size, 1);
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

স্থির শূন্য diag_thread_f32(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_diag_preamble;

    ধ্রুব uint8_t * src_data = (ধ্রুব uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    ধ্রুব uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ধ্রুব uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    প্রতি (uint32_t ib = ib0; ib < ib1; ib++) {
        ধ্রুব uint32_t i3 = ib / ne02;
        ধ্রুব uint32_t i2 = ib % ne02;

        ধ্রুব ভিন্ন * restrict src_batch = (ধ্রুব ভিন্ন *)(src_data + i3 * nb03 + i2 * nb02);

        প্রতি (uint32_t i1 = 0; i1 < ne1; i1++) {
            ভিন্ন * restrict dst_row = (ভিন্ন *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(উপরের_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

পূর্ণাঙ্ক op_diag_f32(সংরচনা htp_ops_context * octx) {
    ধ্রুব সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুব সংরচনা htp_tensor * dst  = octx->dst;

    ধ্রুব uint32_t total_batches = src0->ne[2] * src0->ne[3];
    ধ্রুব size_t dst_batch_size  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == পরিমাণ(ভিন্ন)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        যদি (can_split) {
            যদি (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) == 0)) {
                batches_per_chunk = 1;
            } অন্যথা যদি (dst->nb[2] == dst_batch_size &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_সারি_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_সারি_SIZE)) : 1;
            } অন্যথা {
                can_split = false;
            }
        }

        ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        batch_start = range.start;
        nbatches    = range.গুনতি;
    }

    যদি (nbatches == 0) {
        প্রত্যাগম HTP_STATUS_OK;
    }

    ধ্রুব uint32_t n_threads     = octx->n_threads;

    ধ্রুব size_t src_batch_size         = src0->ne[0] * পরিমাণ(ভিন্ন);
    ধ্রুব size_t dst_row_size           = dst->ne[0] * পরিমাণ(ভিন্ন);
    ধ্রুব size_t src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
    ধ্রুব size_t dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ধ্রুব size_t spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;

    octx->src0_spad.size_per_thread = src_batch_size_aligned;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = রিক্ত;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = রিক্ত;

    সংরচনা htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_size         = src_batch_size,
        .dst_row_size           = dst_row_size,
        .src_batch_size_aligned = src_batch_size_aligned,
        .dst_row_size_aligned   = dst_row_size_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .total_batches          = nbatches,
        .batch_start            = batch_start,
    };

    যদি (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            প্রত্যাগম HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } অন্যথা {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    প্রত্যাগম HTP_STATUS_OK;
}

পূর্ণাঙ্ক op_diag(সংরচনা htp_ops_context * octx) {
    ধ্রুব সংরচনা htp_tensor * dst = octx->dst;

    পূর্ণাঙ্ক err = HTP_STATUS_OK;

    বিকল্প (dst->type) {
        পক্ষ HTP_TYPE_F32:
            err = op_diag_f32(octx);
            ভঙ্গ;
        স্বভাব:
            err = HTP_STATUS_NO_SUPPORT;
            ভঙ্গ;
    }

    প্রত্যাগম err;
}
