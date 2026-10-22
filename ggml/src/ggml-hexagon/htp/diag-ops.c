#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-types.h"
#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "hvx-copy.h"
#अन्तर्भाव "dma-queue.h"

#define htp_diag_tensors_preamble                           \
    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0]; \
    ध्रुव संरचना htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ध्रुव uint32_t ne02 = src0->ne[2];                      \
                                                            \
    ध्रुव uint32_t ne0 = dst->ne[0];                        \
    ध्रुव uint32_t ne1 = dst->ne[1];                        \
                                                            \
    ध्रुव uint32_t nb02 = src0->nb[2];                      \
    ध्रुव uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ध्रुव uint32_t nb1 = dst->nb[1];                        \
    ध्रुव uint32_t nb2 = dst->nb[2];                        \
    ध्रुव uint32_t nb3 = dst->nb[3];

संरचना htp_diag_context {
    संरचना htp_ops_context * octx;
    माप_प्रकार          src_batch_size;
    माप_प्रकार          dst_row_size;
    माप_प्रकार          src_batch_size_aligned;
    माप_प्रकार          dst_row_size_aligned;
    uint32_t        batches_per_thread;
    uint32_t        total_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    संरचना htp_diag_context * dctx = (संरचना htp_diag_context *) data; \
    संरचना htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

स्थिर अन्तःस्थ शून्य hvx_diag_row_f32(ध्रुव भिन्न * restrict src, भिन्न * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

स्थिर शून्य diag_thread_f32_dma(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    ध्रुव uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ध्रुव uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    यदि (ib0 >= ib1) {
        प्रत्यागम;
    }

    ध्रुव माप_प्रकार src_batch_size         = dctx->src_batch_size;
    ध्रुव माप_प्रकार dst_row_size           = dctx->dst_row_size;
    ध्रुव माप_प्रकार src_batch_size_aligned = dctx->src_batch_size_aligned;
    ध्रुव माप_प्रकार dst_row_size_aligned   = dctx->dst_row_size_aligned;

    ध्रुव dma_addr_t src_data = src0->data;
    ध्रुव dma_addr_t dst_data = dst->data;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    प्रति (uint32_t ib = ib0; ib < ib1; ib++) {
        ध्रुव uint32_t i3 = ib / ne02;
        ध्रुव uint32_t i2 = ib % ne02;

        ध्रुव dma_addr_t src_batch = src_data + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_धकेलो(dma_q,
                       dma_make_data(src_spad, src_batch),
                       src_batch_size_aligned, src_batch_size, src_batch_size, 1);
        dma_queue_flush(dma_q);

        ध्रुव भिन्न * src_spad_f32 = (ध्रुव भिन्न *) src_spad;
        भिन्न       * dst_spad_f32 = (भिन्न *) dst_spad;

        प्रति (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ध्रुव dma_addr_t dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_धकेलो(dma_q,
                           dma_make_data(dst_row, dst_spad),
                           dst_row_size, dst_row_size_aligned, dst_row_size, 1);
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

स्थिर शून्य diag_thread_f32(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_diag_preamble;

    ध्रुव uint8_t * src_data = (ध्रुव uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    ध्रुव uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    ध्रुव uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    प्रति (uint32_t ib = ib0; ib < ib1; ib++) {
        ध्रुव uint32_t i3 = ib / ne02;
        ध्रुव uint32_t i2 = ib % ne02;

        ध्रुव भिन्न * restrict src_batch = (ध्रुव भिन्न *)(src_data + i3 * nb03 + i2 * nb02);

        प्रति (uint32_t i1 = 0; i1 < ne1; i1++) {
            भिन्न * restrict dst_row = (भिन्न *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(ऊपरी_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

पूर्णाङ्क op_diag_f32(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    ध्रुव uint32_t total_batches = src0->ne[2] * src0->ne[3];
    ध्रुव माप_प्रकार dst_batch_size  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    यदि (octx->ctx->mdev.गिनती > 1) {
        bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == परिमाण(भिन्न)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        यदि (can_split) {
            यदि (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_पंक्ति_SIZE - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_पंक्ति_SIZE - 1)) == 0)) {
                batches_per_chunk = 1;
            } अन्यथा यदि (dst->nb[2] == dst_batch_size &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_पंक्ति_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_पंक्ति_SIZE)) : 1;
            } अन्यथा {
                can_split = false;
            }
        }

        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        batch_start = range.start;
        nbatches    = range.गिनती;
    }

    यदि (nbatches == 0) {
        प्रत्यागम HTP_STATUS_OK;
    }

    ध्रुव uint32_t n_threads     = octx->n_threads;

    ध्रुव माप_प्रकार src_batch_size         = src0->ne[0] * परिमाण(भिन्न);
    ध्रुव माप_प्रकार dst_row_size           = dst->ne[0] * परिमाण(भिन्न);
    ध्रुव माप_प्रकार src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
    ध्रुव माप_प्रकार dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ध्रुव माप_प्रकार spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;

    octx->src0_spad.size_per_thread = src_batch_size_aligned;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = रिक्त;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = रिक्त;

    संरचना htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_size         = src_batch_size,
        .dst_row_size           = dst_row_size,
        .src_batch_size_aligned = src_batch_size_aligned,
        .dst_row_size_aligned   = dst_row_size_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .total_batches          = nbatches,
        .batch_start            = batch_start,
    };

    यदि (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            प्रत्यागम HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } अन्यथा {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    प्रत्यागम HTP_STATUS_OK;
}

पूर्णाङ्क op_diag(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * dst = octx->dst;

    पूर्णाङ्क err = HTP_STATUS_OK;

    विकल्प (dst->type) {
        पक्ष HTP_TYPE_F32:
            err = op_diag_f32(octx);
            भङ्ग;
        स्वभाव:
            err = HTP_STATUS_NO_SUPPORT;
            भङ्ग;
    }

    प्रत्यागम err;
}
