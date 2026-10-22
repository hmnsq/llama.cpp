#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "hvx-types.h"
#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "hvx-copy.h"
#అన్తర్భావ "dma-queue.h"

#నిర్వచించు htp_diag_tensors_preamble                           \
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0]; \
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst  = octx->dst;    \
                                                            \
    స్థిరరాశి uint32_t ne02 = src0->ne[2];                      \
                                                            \
    స్థిరరాశి uint32_t ne0 = dst->ne[0];                        \
    స్థిరరాశి uint32_t ne1 = dst->ne[1];                        \
                                                            \
    స్థిరరాశి uint32_t nb02 = src0->nb[2];                      \
    స్థిరరాశి uint32_t nb03 = src0->nb[3];                      \
                                                            \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];                        \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];                        \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

నిర్మాణం htp_diag_context {
    నిర్మాణం htp_ops_context * octx;
    size_t          src_batch_కొలత;
    size_t          dst_అడ్డువరుస_కొలత;
    size_t          src_batch_కొలత_aligned;
    size_t          dst_అడ్డువరుస_కొలత_aligned;
    uint32_t        batches_చొప్పున_thread;
    uint32_t        మొత్తం_batches;
    uint32_t        batch_ప్రారంభం;
};

#నిర్వచించు htp_diag_preamble                                              \
    నిర్మాణం htp_diag_context * dctx = (నిర్మాణం htp_diag_context *) దత్తాంశం; \
    నిర్మాణం htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

స్థిర అంతఃవరుస శూన్య hvx_diag_అడ్డువరుస_f32(స్థిరరాశి దశాంశం * పరిమిత src, దశాంశం * పరిమిత dst,
                                    uint32_t అడ్డువరుస_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[అడ్డువరుస_idx] = src[అడ్డువరుస_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

స్థిర శూన్య diag_thread_f32_dma(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_diag_preamble;
    dma_వరుస * dma_q = octx->ctx->dma[ith];

    స్థిరరాశి uint32_t ib0 = dctx->batch_ప్రారంభం + dctx->batches_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t ib1 = MIN(ib0 + dctx->batches_చొప్పున_thread, dctx->batch_ప్రారంభం + dctx->మొత్తం_batches);

    ఒకవేళ (ib0 >= ib1) {
        తిరిగి;
    }

    స్థిరరాశి size_t src_batch_కొలత         = dctx->src_batch_కొలత;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత           = dctx->dst_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t src_batch_కొలత_aligned = dctx->src_batch_కొలత_aligned;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned   = dctx->dst_అడ్డువరుస_కొలత_aligned;

    స్థిరరాశి dma_addr_t src_దత్తాంశం = src0->దత్తాంశం;
    స్థిరరాశి dma_addr_t dst_దత్తాంశం = dst->దత్తాంశం;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.దత్తాంశం + (ith * src_batch_కొలత_aligned);
    uint8_t * dst_spad = octx->dst_spad.దత్తాంశం  + (ith * dst_అడ్డువరుస_కొలత_aligned);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    కొరకు (uint32_t ib = ib0; ib < ib1; ib++) {
        స్థిరరాశి uint32_t i3 = ib / ne02;
        స్థిరరాశి uint32_t i2 = ib % ne02;

        స్థిరరాశి dma_addr_t src_batch = src_దత్తాంశం + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(src_spad, src_batch),
                       src_batch_కొలత_aligned, src_batch_కొలత, src_batch_కొలత, 1);
        dma_వరుస_flush(dma_q);

        స్థిరరాశి దశాంశం * src_spad_f32 = (స్థిరరాశి దశాంశం *) src_spad;
        దశాంశం       * dst_spad_f32 = (దశాంశం *) dst_spad;

        కొరకు (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_అడ్డువరుస_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            స్థిరరాశి dma_addr_t dst_అడ్డువరుస = dst_దత్తాంశం + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_వరుస_నెట్టు(dma_q,
                           dma_make_దత్తాంశం(dst_అడ్డువరుస, dst_spad),
                           dst_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 1);
            dma_వరుస_flush(dma_q);
        }
    }

    FARF(ఎగువ_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

స్థిర శూన్య diag_thread_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_diag_preamble;

    స్థిరరాశి uint8_t * src_దత్తాంశం = (స్థిరరాశి uint8_t *) src0->దత్తాంశం;
    uint8_t *       dst_దత్తాంశం = (uint8_t *) dst->దత్తాంశం;

    స్థిరరాశి uint32_t ib0 = dctx->batch_ప్రారంభం + dctx->batches_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t ib1 = MIN(ib0 + dctx->batches_చొప్పున_thread, dctx->batch_ప్రారంభం + dctx->మొత్తం_batches);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    కొరకు (uint32_t ib = ib0; ib < ib1; ib++) {
        స్థిరరాశి uint32_t i3 = ib / ne02;
        స్థిరరాశి uint32_t i2 = ib % ne02;

        స్థిరరాశి దశాంశం * పరిమిత src_batch = (స్థిరరాశి దశాంశం *)(src_దత్తాంశం + i3 * nb03 + i2 * nb02);

        కొరకు (uint32_t i1 = 0; i1 < ne1; i1++) {
            దశాంశం * పరిమిత dst_అడ్డువరుస = (దశాంశం *)(dst_దత్తాంశం + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_అడ్డువరుస_f32(src_batch, dst_అడ్డువరుస, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(ఎగువ_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

పూర్ణాంకం op_diag_f32(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    స్థిరరాశి uint32_t మొత్తం_batches = src0->ne[2] * src0->ne[3];
    స్థిరరాశి size_t dst_batch_కొలత  = dst->ne[1] * dst->nb[1];

    uint32_t batch_ప్రారంభం = 0;
    uint32_t nbatches    = మొత్తం_batches;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        bool can_విభజించు = htp_tensor_mdev_దత్తాంశం_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == పరిమాణం(దశాంశం)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_చొప్పున_chunk = 1;
        ఒకవేళ (can_విభజించు) {
            ఒకవేళ (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) == 0)) {
                batches_చొప్పున_chunk = 1;
            } లేకపోతే ఒకవేళ (dst->nb[2] == dst_batch_కొలత &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_చొప్పున_chunk = (dst_batch_కొలత > 0) ? (HEX_L2_పంక్తి_కొలత / hex_గరిష్ఠ_సామాన్య_భాజకం_u32(dst_batch_కొలత, HEX_L2_పంక్తి_కొలత)) : 1;
            } లేకపోతే {
                can_విభజించు = false;
            }
        }

        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_batches, can_విభజించు ? batches_చొప్పున_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        batch_ప్రారంభం = range.ప్రారంభం;
        nbatches    = range.లెక్క;
    }

    ఒకవేళ (nbatches == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు     = octx->n_దారాలు;

    స్థిరరాశి size_t src_batch_కొలత         = src0->ne[0] * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత           = dst->ne[0] * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t src_batch_కొలత_aligned = hex_విడత_పైకి(src_batch_కొలత, VLEN);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned   = hex_విడత_పైకి(dst_అడ్డువరుస_కొలత, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    స్థిరరాశి size_t spad_చొప్పున_thread = src_batch_కొలత_aligned + dst_అడ్డువరుస_కొలత_aligned;

    octx->src0_spad.కొలత_చొప్పున_thread = src_batch_కొలత_aligned;
    octx->dst_spad.కొలత_చొప్పున_thread  = dst_అడ్డువరుస_కొలత_aligned;

    octx->src0_spad.కొలత = n_దారాలు * octx->src0_spad.కొలత_చొప్పున_thread;
    octx->dst_spad.కొలత  = n_దారాలు * octx->dst_spad.కొలత_చొప్పున_thread;

    octx->src0_spad.దత్తాంశం = octx->ctx->vtcm_పునాది;                        octx->src0_spad.src = ఖాళీ;
    octx->dst_spad.దత్తాంశం  = octx->src0_spad.దత్తాంశం + octx->src0_spad.కొలత; octx->dst_spad.src  = ఖాళీ;

    నిర్మాణం htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_కొలత         = src_batch_కొలత,
        .dst_అడ్డువరుస_కొలత           = dst_అడ్డువరుస_కొలత,
        .src_batch_కొలత_aligned = src_batch_కొలత_aligned,
        .dst_అడ్డువరుస_కొలత_aligned   = dst_అడ్డువరుస_కొలత_aligned,
        .batches_చొప్పున_thread     = fastdiv(nbatches + n_దారాలు - 1, &octx->n_దారాలు_div),
        .మొత్తం_batches          = nbatches,
        .batch_ప్రారంభం            = batch_ప్రారంభం,
    };

    ఒకవేళ (octx->ctx->vtcm_కొలత < spad_చొప్పున_thread * n_దారాలు) {
        ఒకవేళ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            తిరిగి HTP_స్థితి_NO_SUPPORT;
        }
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, diag_thread_f32, &dctx, n_దారాలు);
    } లేకపోతే {
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, diag_thread_f32_dma, &dctx, n_దారాలు);
    }

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_diag(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;

    పూర్ణాంకం err = HTP_స్థితి_OK;

    ఎంపిక (dst->type) {
        స్థితి HTP_TYPE_F32:
            err = op_diag_f32(octx);
            ఆపు;
        అప్రమేయ:
            err = HTP_స్థితి_NO_SUPPORT;
            ఆపు;
    }

    తిరిగి err;
}
