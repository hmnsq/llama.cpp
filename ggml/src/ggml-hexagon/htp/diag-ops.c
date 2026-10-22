#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#परिभाषय GGML_COMMON_DECL_C
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

#परिभाषय htp_diag_tensors_preamble                           \
    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0]; \
    ध्रुव संरचना htp_tensor * सीमित dst  = octx->dst;    \
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
    size_t          src_batch_आकार;
    size_t          dst_row_आकार;
    size_t          src_batch_आकार_aligned;
    size_t          dst_row_आकार_aligned;
    uint32_t        batches_प्रत्येकम्_thread;
    uint32_t        योग_batches;
    uint32_t        batch_start;
};

#परिभाषय htp_diag_preamble                                              \
    संरचना htp_diag_context * dctx = (संरचना htp_diag_context *) दत्त; \
    संरचना htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

स्थिर अन्तःस्थ शून्य hvx_diag_row_f32(ध्रुव भिन्न * सीमित src, भिन्न * सीमित dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

स्थिर शून्य diag_thread_f32_dma(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    ध्रुव uint32_t ib0 = dctx->batch_start + dctx->batches_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t ib1 = MIN(ib0 + dctx->batches_प्रत्येकम्_thread, dctx->batch_start + dctx->योग_batches);

    यदि (ib0 >= ib1) {
        प्रत्यागम;
    }

    ध्रुव size_t src_batch_आकार         = dctx->src_batch_आकार;
    ध्रुव size_t dst_row_आकार           = dctx->dst_row_आकार;
    ध्रुव size_t src_batch_आकार_aligned = dctx->src_batch_आकार_aligned;
    ध्रुव size_t dst_row_आकार_aligned   = dctx->dst_row_आकार_aligned;

    ध्रुव dma_addr_t src_दत्त = src0->दत्त;
    ध्रुव dma_addr_t dst_दत्त = dst->दत्त;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.दत्त + (ith * src_batch_आकार_aligned);
    uint8_t * dst_spad = octx->dst_spad.दत्त  + (ith * dst_row_आकार_aligned);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    प्रति (uint32_t ib = ib0; ib < ib1; ib++) {
        ध्रुव uint32_t i3 = ib / ne02;
        ध्रुव uint32_t i2 = ib % ne02;

        ध्रुव dma_addr_t src_batch = src_दत्त + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_प्रक्षिप(dma_q,
                       dma_make_दत्त(src_spad, src_batch),
                       src_batch_आकार_aligned, src_batch_आकार, src_batch_आकार, 1);
        dma_queue_flush(dma_q);

        ध्रुव भिन्न * src_spad_f32 = (ध्रुव भिन्न *) src_spad;
        भिन्न       * dst_spad_f32 = (भिन्न *) dst_spad;

        प्रति (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ध्रुव dma_addr_t dst_row = dst_दत्त + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_प्रक्षिप(dma_q,
                           dma_make_दत्त(dst_row, dst_spad),
                           dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 1);
            dma_queue_flush(dma_q);
        }
    }

    FARF(उच्चः_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

स्थिर शून्य diag_thread_f32(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_diag_preamble;

    ध्रुव uint8_t * src_दत्त = (ध्रुव uint8_t *) src0->दत्त;
    uint8_t *       dst_दत्त = (uint8_t *) dst->दत्त;

    ध्रुव uint32_t ib0 = dctx->batch_start + dctx->batches_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t ib1 = MIN(ib0 + dctx->batches_प्रत्येकम्_thread, dctx->batch_start + dctx->योग_batches);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    प्रति (uint32_t ib = ib0; ib < ib1; ib++) {
        ध्रुव uint32_t i3 = ib / ne02;
        ध्रुव uint32_t i2 = ib % ne02;

        ध्रुव भिन्न * सीमित src_batch = (ध्रुव भिन्न *)(src_दत्त + i3 * nb03 + i2 * nb02);

        प्रति (uint32_t i1 = 0; i1 < ne1; i1++) {
            भिन्न * सीमित dst_row = (भिन्न *)(dst_दत्त + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(उच्चः_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

पूर्णाङ्क op_diag_f32(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    ध्रुव uint32_t योग_batches = src0->ne[2] * src0->ne[3];
    ध्रुव size_t dst_batch_आकार  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = योग_batches;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        bool can_विभज = htp_tensor_mdev_दत्त_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == परिमाण(भिन्न)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_प्रत्येकम्_chunk = 1;
        यदि (can_विभज) {
            यदि (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) == 0)) {
                batches_प्रत्येकम्_chunk = 1;
            } अन्यथा यदि (dst->nb[2] == dst_batch_आकार &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_प्रत्येकम्_chunk = (dst_batch_आकार > 0) ? (HEX_L2_पङ्क्तिः_आकार / hex_gcd_u32(dst_batch_आकार, HEX_L2_पङ्क्तिः_आकार)) : 1;
            } अन्यथा {
                can_विभज = false;
            }
        }

        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_batches, can_विभज ? batches_प्रत्येकम्_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        batch_start = range.start;
        nbatches    = range.गणनम्;
    }

    यदि (nbatches == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः     = octx->n_तन्तवः;

    ध्रुव size_t src_batch_आकार         = src0->ne[0] * परिमाण(भिन्न);
    ध्रुव size_t dst_row_आकार           = dst->ne[0] * परिमाण(भिन्न);
    ध्रुव size_t src_batch_आकार_aligned = hex_आवर्तनम्_ऊर्ध्वम्(src_batch_आकार, VLEN);
    ध्रुव size_t dst_row_आकार_aligned   = hex_आवर्तनम्_ऊर्ध्वम्(dst_row_आकार, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ध्रुव size_t spad_प्रत्येकम्_thread = src_batch_आकार_aligned + dst_row_आकार_aligned;

    octx->src0_spad.आकार_प्रत्येकम्_thread = src_batch_आकार_aligned;
    octx->dst_spad.आकार_प्रत्येकम्_thread  = dst_row_आकार_aligned;

    octx->src0_spad.आकार = n_तन्तवः * octx->src0_spad.आकार_प्रत्येकम्_thread;
    octx->dst_spad.आकार  = n_तन्तवः * octx->dst_spad.आकार_प्रत्येकम्_thread;

    octx->src0_spad.दत्त = octx->ctx->vtcm_आधारः;                        octx->src0_spad.src = रिक्त;
    octx->dst_spad.दत्त  = octx->src0_spad.दत्त + octx->src0_spad.आकार; octx->dst_spad.src  = रिक्त;

    संरचना htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_आकार         = src_batch_आकार,
        .dst_row_आकार           = dst_row_आकार,
        .src_batch_आकार_aligned = src_batch_आकार_aligned,
        .dst_row_आकार_aligned   = dst_row_आकार_aligned,
        .batches_प्रत्येकम्_thread     = fastdiv(nbatches + n_तन्तवः - 1, &octx->n_तन्तवः_div),
        .योग_batches          = nbatches,
        .batch_start            = batch_start,
    };

    यदि (octx->ctx->vtcm_आकार < spad_प्रत्येकम्_thread * n_तन्तवः) {
        यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
        }
        work_queue_सततम्(octx->ctx->work_queue, diag_thread_f32, &dctx, n_तन्तवः);
    } अन्यथा {
        work_queue_सततम्(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_तन्तवः);
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_diag(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * dst = octx->dst;

    पूर्णाङ्क err = HTP_स्थितिः_OK;

    विकल्प (dst->type) {
        पक्ष HTP_TYPE_F32:
            err = op_diag_f32(octx);
            भङ्ग;
        स्वभाव:
            err = HTP_स्थितिः_NO_SUPPORT;
            भङ्ग;
    }

    प्रत्यागम err;
}
