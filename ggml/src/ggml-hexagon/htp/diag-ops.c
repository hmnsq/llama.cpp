#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#define GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hex-utils.h"
#ضمن "hvx-copy.h"
#ضمن "dma-queue.h"

#define htp_diag_tensors_preamble                           \
    ثابت بنية htp_tensor * restrict src0 = octx->src[0]; \
    ثابت بنية htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ثابت uint32_t ne02 = src0->ne[2];                      \
                                                            \
    ثابت uint32_t ne0 = dst->ne[0];                        \
    ثابت uint32_t ne1 = dst->ne[1];                        \
                                                            \
    ثابت uint32_t nb02 = src0->nb[2];                      \
    ثابت uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ثابت uint32_t nb1 = dst->nb[1];                        \
    ثابت uint32_t nb2 = dst->nb[2];                        \
    ثابت uint32_t nb3 = dst->nb[3];

بنية htp_diag_context {
    بنية htp_ops_context * octx;
    size_t          src_batch_size;
    size_t          dst_row_size;
    size_t          src_batch_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        batches_per_thread;
    uint32_t        total_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    بنية htp_diag_context * dctx = (بنية htp_diag_context *) data; \
    بنية htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

ساكن مضمن فراغ hvx_diag_row_f32(ثابت عائم * restrict src, عائم * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

ساكن فراغ diag_thread_f32_dma(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    إذا (ib0 >= ib1) {
        أرجع;
    }

    ثابت size_t src_batch_size         = dctx->src_batch_size;
    ثابت size_t dst_row_size           = dctx->dst_row_size;
    ثابت size_t src_batch_size_aligned = dctx->src_batch_size_aligned;
    ثابت size_t dst_row_size_aligned   = dctx->dst_row_size_aligned;

    ثابت dma_addr_t src_data = src0->data;
    ثابت dma_addr_t dst_data = dst->data;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت dma_addr_t src_batch = src_data + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_ڈالو(dma_q,
                       dma_make_data(src_spad, src_batch),
                       src_batch_size_aligned, src_batch_size, src_batch_size, 1);
        dma_queue_flush(dma_q);

        ثابت عائم * src_spad_f32 = (ثابت عائم *) src_spad;
        عائم       * dst_spad_f32 = (عائم *) dst_spad;

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ثابت dma_addr_t dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_ڈالو(dma_q,
                           dma_make_data(dst_row, dst_spad),
                           dst_row_size, dst_row_size_aligned, dst_row_size, 1);
            dma_queue_flush(dma_q);
        }
    }

    FARF(بلند_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ساكن فراغ diag_thread_f32(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_diag_preamble;

    ثابت uint8_t * src_data = (ثابت uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    ثابت uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت عائم * restrict src_batch = (ثابت عائم *)(src_data + i3 * nb03 + i2 * nb02);

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            عائم * restrict dst_row = (عائم *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(بلند_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحيح op_diag_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t total_batches = src0->ne[2] * src0->ne[3];
    ثابت size_t dst_batch_size  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    إذا (octx->ctx->mdev.گنتی > 1) {
        bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == حجم(عائم)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        إذا (can_split) {
            إذا (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_سطر_SIZE - 1)) == 0)) {
                batches_per_chunk = 1;
            } وإلا إذا (dst->nb[2] == dst_batch_size &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_سطر_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_سطر_SIZE)) : 1;
            } وإلا {
                can_split = false;
            }
        }

        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        batch_start = range.start;
        nbatches    = range.گنتی;
    }

    إذا (nbatches == 0) {
        أرجع HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads     = octx->n_threads;

    ثابت size_t src_batch_size         = src0->ne[0] * حجم(عائم);
    ثابت size_t dst_row_size           = dst->ne[0] * حجم(عائم);
    ثابت size_t src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
    ثابت size_t dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ثابت size_t spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;

    octx->src0_spad.size_per_thread = src_batch_size_aligned;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = عدم;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = عدم;

    بنية htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_size         = src_batch_size,
        .dst_row_size           = dst_row_size,
        .src_batch_size_aligned = src_batch_size_aligned,
        .dst_row_size_aligned   = dst_row_size_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .total_batches          = nbatches,
        .batch_start            = batch_start,
    };

    إذا (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            أرجع HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } وإلا {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    أرجع HTP_STATUS_OK;
}

صحيح op_diag(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * dst = octx->dst;

    صحيح err = HTP_STATUS_OK;

    بدل (dst->type) {
        حالة HTP_TYPE_F32:
            err = op_diag_f32(octx);
            اكسر;
        افتراضي:
            err = HTP_STATUS_NO_SUPPORT;
            اكسر;
    }

    أرجع err;
}
