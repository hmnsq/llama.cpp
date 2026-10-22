#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
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
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "dma-queue.h"

#నిర్వచించు htp_cumsum_tensors_preamble                         \
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0]; \
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst  = octx->dst;    \
                                                            \
    స్థిరరాశి uint32_t ne00 = src0->ne[0];                      \
    స్థిరరాశి uint32_t ne01 = src0->ne[1];                      \
    స్థిరరాశి uint32_t ne02 = src0->ne[2];                      \
    స్థిరరాశి uint32_t ne03 = src0->ne[3];                      \
                                                            \
    స్థిరరాశి uint32_t ne0 = dst->ne[0];                        \
    స్థిరరాశి uint32_t ne1 = dst->ne[1];                        \
    స్థిరరాశి uint32_t ne2 = dst->ne[2];                        \
    స్థిరరాశి uint32_t ne3 = dst->ne[3];                        \
                                                            \
    స్థిరరాశి uint32_t nb00 = src0->nb[0];                      \
    స్థిరరాశి uint32_t nb01 = src0->nb[1];                      \
    స్థిరరాశి uint32_t nb02 = src0->nb[2];                      \
    స్థిరరాశి uint32_t nb03 = src0->nb[3];                      \
                                                            \
    స్థిరరాశి uint32_t nb0 = dst->nb[0];                        \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];                        \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];                        \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

నిర్మాణం htp_cumsum_context {
    నిర్మాణం htp_ops_context * octx;
    size_t          src_అడ్డువరుస_కొలత;
    size_t          dst_అడ్డువరుస_కొలత;
    size_t          src_అడ్డువరుస_కొలత_aligned;
    size_t          dst_అడ్డువరుస_కొలత_aligned;
    uint32_t        అడ్డువరుసలు_చొప్పున_thread;
    uint32_t        మొత్తం_అడ్డువరుసలు;
    uint32_t        అడ్డువరుస_ప్రారంభం;
};

#నిర్వచించు htp_cumsum_preamble                                                \
    నిర్మాణం htp_cumsum_context * cctx = (నిర్మాణం htp_cumsum_context *) దత్తాంశం; \
    నిర్మాణం htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_వరుస * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#ఒకవేళ __HVX_ARCH__ > 75
స్థిర అంతఃవరుస HVX_సదిశ hvx_cumsum_vadd(HVX_సదిశ a, HVX_సదిశ b) {
    తిరిగి Q6_Vsf_vadd_VsfVsf(a, b);
}
#లేకపోతే
స్థిర అంతఃవరుస HVX_సదిశ hvx_cumsum_vadd(HVX_సదిశ a, HVX_సదిశ b) {
    తిరిగి Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#అయితే_ముగింపు  // __HVX_ARCH__ > 75

స్థిర అంతఃవరుస HVX_సదిశ hvx_ఉపసర్గ_పరిశీలన_f32(HVX_సదిశ v, HVX_సదిశ వాహకం_లోపల) {
    స్థిరరాశి HVX_సదిశ zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, వాహకం_లోపల);

    తిరిగి v;
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_splat_చివరి_f32(HVX_సదిశ v) {
    తిరిగి hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

స్థిర అంతఃవరుస శూన్య hvx_cumsum_అడ్డువరుస_f32(స్థిరరాశి దశాంశం * పరిమిత src, దశాంశం * పరిమిత dst, uint32_t n) {
    స్థిరరాశి uint32_t nvec = n / VLEN_FP32;
    స్థిరరాశి uint32_t nloe = n % VLEN_FP32;

    HVX_సదిశ వాహకం = Q6_V_vsplat_R(0);

    కొరకు (uint32_t i = 0; i < nvec; i++) {
        HVX_సదిశ v = *((స్థిరరాశి HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_ఉపసర్గ_పరిశీలన_f32(v, వాహకం);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        వాహకం = hvx_splat_చివరి_f32(v);
    }

    ఒకవేళ (nloe) {
        దశాంశం acc = hvx_vec_get_f32(వాహకం);
        స్థిరరాశి దశాంశం * src_తోక = src + nvec * VLEN_FP32;
        దశాంశం       * dst_తోక = dst + nvec * VLEN_FP32;
        కొరకు (uint32_t i = 0; i < nloe; i++) {
            acc        += src_తోక[i];
            dst_తోక[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

స్థిర శూన్య cumsum_thread_f32_dma(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_cumsum_preamble;

    స్థిరరాశి uint32_t ir0 = cctx->అడ్డువరుస_ప్రారంభం + cctx->అడ్డువరుసలు_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + cctx->అడ్డువరుసలు_చొప్పున_thread, cctx->అడ్డువరుస_ప్రారంభం + cctx->మొత్తం_అడ్డువరుసలు);

    ఒకవేళ (ir0 >= ir1) {
        తిరిగి;
    }

    స్థిరరాశి size_t src_అడ్డువరుస_కొలత         = cctx->src_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత         = cctx->dst_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t src_అడ్డువరుస_కొలత_aligned = cctx->src_అడ్డువరుస_కొలత_aligned;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned = cctx->dst_అడ్డువరుస_కొలత_aligned;

    స్థిరరాశి dma_addr_t src_దత్తాంశం = src0->దత్తాంశం;
    స్థిరరాశి dma_addr_t dst_దత్తాంశం = dst->దత్తాంశం;

    uint8_t * src_spad = octx->src0_spad.దత్తాంశం + (ith * src_అడ్డువరుస_కొలత_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.దత్తాంశం  + (ith * dst_అడ్డువరుస_కొలత_aligned * 2);

    కొరకు (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(dst_దత్తాంశం, dst_spad + (spad_idx * dst_అడ్డువరుస_కొలత_aligned)),
                       dst_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 0);

        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(src_spad + (spad_idx * src_అడ్డువరుస_కొలత_aligned),
                                     src_దత్తాంశం + (ir * src_అడ్డువరుస_కొలత)),
                       src_అడ్డువరుస_కొలత_aligned, src_అడ్డువరుస_కొలత, src_అడ్డువరుస_కొలత, 1);
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    కొరకు (uint32_t ir = ir0; ir < ir1; ir++) {
        దశాంశం * dst_spad_అడ్డువరుస = (దశాంశం *) dma_వరుస_తీయి(dma_q).src;
        దశాంశం * src_spad_అడ్డువరుస = (దశాంశం *) dma_వరుస_తీయి(dma_q).dst;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_అడ్డువరుస_f32(src_spad_అడ్డువరుస, dst_spad_అడ్డువరుస, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_వరుస_నెట్టు(dma_q,
                       dma_make_దత్తాంశం(dst_దత్తాంశం + (ir * dst_అడ్డువరుస_కొలత), dst_spad_అడ్డువరుస),
                       dst_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 1);

        స్థిరరాశి uint32_t తదుపరి_అడ్డువరుస = ir + 2;
        ఒకవేళ (తదుపరి_అడ్డువరుస < ir1) {
            dma_వరుస_నెట్టు(dma_q,
                           dma_make_దత్తాంశం(src_spad_అడ్డువరుస, src_దత్తాంశం + (తదుపరి_అడ్డువరుస * src_అడ్డువరుస_కొలత)),
                           src_అడ్డువరుస_కొలత_aligned, src_అడ్డువరుస_కొలత, src_అడ్డువరుస_కొలత, 1);
        }
    }

    dma_వరుస_flush(dma_q);

    FARF(ఎగువ_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

స్థిర శూన్య cumsum_thread_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_cumsum_preamble;

    స్థిరరాశి uint8_t * src_దత్తాంశం = (స్థిరరాశి uint8_t *) src0->దత్తాంశం;
    uint8_t *       dst_దత్తాంశం = (uint8_t *) dst->దత్తాంశం;

    స్థిరరాశి uint32_t ir0 = cctx->అడ్డువరుస_ప్రారంభం + cctx->అడ్డువరుసలు_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + cctx->అడ్డువరుసలు_చొప్పున_thread, cctx->అడ్డువరుస_ప్రారంభం + cctx->మొత్తం_అడ్డువరుసలు);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    కొరకు (uint32_t ir = ir0; ir < ir1; ir++) {
        స్థిరరాశి దశాంశం * పరిమిత src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) (src_దత్తాంశం + ir * cctx->src_అడ్డువరుస_కొలత);
        దశాంశం       * పరిమిత dst_అడ్డువరుస = (దశాంశం *)       (dst_దత్తాంశం + ir * cctx->dst_అడ్డువరుస_కొలత);
        hvx_cumsum_అడ్డువరుస_f32(src_అడ్డువరుస, dst_అడ్డువరుస, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(ఎగువ_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

పూర్ణాంకం op_cumsum_f32(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి size_t dst_దత్తాంశం_అడ్డువరుస_కొలత = dst->ne[0] * పరిమాణం(దశాంశం);

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = మొత్తం_అడ్డువరుసలు;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, పరిమాణం(దశాంశం), (uint32_t) dst_దత్తాంశం_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_అడ్డువరుసలు, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    స్థిరరాశి size_t src_అడ్డువరుస_కొలత         = src0->nb[1];
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత         = dst->nb[1];
    స్థిరరాశి size_t src_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src_అడ్డువరుస_కొలత, VLEN);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(dst_అడ్డువరుస_కొలత, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    స్థిరరాశి size_t spad_చొప్పున_thread = 2 * (src_అడ్డువరుస_కొలత_aligned + dst_అడ్డువరుస_కొలత_aligned);

    octx->src0_spad.కొలత_చొప్పున_thread = src_అడ్డువరుస_కొలత_aligned * 2;
    octx->dst_spad.కొలత_చొప్పున_thread  = dst_అడ్డువరుస_కొలత_aligned * 2;

    octx->src0_spad.కొలత  = n_దారాలు * octx->src0_spad.కొలత_చొప్పున_thread;
    octx->dst_spad.కొలత   = n_దారాలు * octx->dst_spad.కొలత_చొప్పున_thread;

    octx->src0_spad.దత్తాంశం  = octx->ctx->vtcm_పునాది;                        octx->src0_spad.src = ఖాళీ;
    octx->dst_spad.దత్తాంశం   = octx->src0_spad.దత్తాంశం + octx->src0_spad.కొలత; octx->dst_spad.src  = ఖాళీ;

    నిర్మాణం htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_అడ్డువరుస_కొలత         = src_అడ్డువరుస_కొలత,
        .dst_అడ్డువరుస_కొలత         = dst_అడ్డువరుస_కొలత,
        .src_అడ్డువరుస_కొలత_aligned = src_అడ్డువరుస_కొలత_aligned,
        .dst_అడ్డువరుస_కొలత_aligned = dst_అడ్డువరుస_కొలత_aligned,
        .అడ్డువరుసలు_చొప్పున_thread      = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div),
        .మొత్తం_అడ్డువరుసలు           = nrows,
        .అడ్డువరుస_ప్రారంభం            = అడ్డువరుస_ప్రారంభం,
    };

    ఒకవేళ (octx->ctx->vtcm_కొలత < spad_చొప్పున_thread * n_దారాలు) {
        ఒకవేళ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            తిరిగి HTP_స్థితి_NO_SUPPORT;
        }
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, cumsum_thread_f32, &cctx, n_దారాలు);
    } లేకపోతే {
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, cumsum_thread_f32_dma, &cctx, n_దారాలు);
    }

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_cumsum(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;

    ఎంపిక (dst->type) {
        స్థితి HTP_TYPE_F32:
            తిరిగి op_cumsum_f32(octx);
        అప్రమేయ:
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
}
