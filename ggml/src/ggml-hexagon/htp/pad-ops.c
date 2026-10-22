#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#అన్తర్భావ <string.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hvx-utils.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"

/* Circular wrap: maps any integer x into [0, n) */
స్థిర అంతఃవరుస uint32_t wrap_around(int32_t x, uint32_t n) {
    తిరిగి (uint32_t)(((x % (int32_t)n) + (int32_t)n) % (int32_t)n);
}

/* Decompose a flat dst row index into (i1, i2, i3) */
స్థిర అంతఃవరుస శూన్య pad_decompose_అడ్డువరుస(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     uint32_t *i1, uint32_t *i2, uint32_t *i3) {
    *i1 = ir % ne1;
    *i2 = (ir / ne1) % ne2;
    *i3 = ir / (ne1 * ne2);
}

/* Return non-zero if row (i1,i2,i3) falls in the non-padded interior */
స్థిర అంతఃవరుస పూర్ణాంకం pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                   int32_t lp1, int32_t rp1, uint32_t ne1,
                                   int32_t lp2, int32_t rp2, uint32_t ne2,
                                   int32_t lp3, int32_t rp3, uint32_t ne3) {
    తిరిగి ((int32_t)i1 >= lp1 && (int32_t)i1 < (int32_t)ne1 - rp1) &&
           ((int32_t)i2 >= lp2 && (int32_t)i2 < (int32_t)ne2 - rp2) &&
           ((int32_t)i3 >= lp3 && (int32_t)i3 < (int32_t)ne3 - rp3);
}

/* Compute the DDR src row pointer for a zero-pad interior row */
స్థిర అంతఃవరుస స్థిరరాశి uint8_t * pad_src_అడ్డువరుస_ptr(స్థిరరాశి నిర్మాణం htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    తిరిగి (స్థిరరాశి uint8_t *) src->దత్తాంశం
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

స్థిర అంతఃవరుస dma_addr_t pad_src_అడ్డువరుస_దత్తాంశం(స్థిరరాశి నిర్మాణం htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    తిరిగి src->దత్తాంశం
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
స్థిర అంతఃవరుస స్థిరరాశి uint8_t * pad_circ_src_అడ్డువరుస_ptr(స్థిరరాశి నిర్మాణం htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    తిరిగి (స్థిరరాశి uint8_t *) src->దత్తాంశం
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

స్థిర అంతఃవరుస dma_addr_t pad_circ_src_అడ్డువరుస_దత్తాంశం(స్థిరరాశి నిర్మాణం htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    తిరిగి src->దత్తాంశం
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

నిర్మాణం htp_pad_context {
    నిర్మాణం htp_ops_context * octx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_చొప్పున_thread;
    uint32_t మొత్తం_dst_అడ్డువరుసలు;
    uint32_t అడ్డువరుస_ప్రారంభం;

    size_t   type_కొలత;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_అడ్డువరుస_కొలత;
    size_t   src_అడ్డువరుస_కొలత_aligned;
    size_t   dst_అడ్డువరుస_కొలత;
    size_t   dst_అడ్డువరుస_కొలత_aligned;
};

#నిర్వచించు htp_pad_preamble                                                             \
    స్థిరరాశి నిర్మాణం htp_tensor * src = octx->src[0];                                    \
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;                                       \
                                                                                     \
    స్థిరరాశి uint32_t ne00 = src->ne[0];                                                \
    స్థిరరాశి uint32_t nb00 = src->nb[0];                                                \
                                                                                     \
    స్థిరరాశి uint32_t ne0 = dst->ne[0];                                                 \
    స్థిరరాశి uint32_t ne1 = dst->ne[1];                                                 \
    స్థిరరాశి uint32_t ne2 = dst->ne[2];                                                 \
    స్థిరరాశి uint32_t ne3 = dst->ne[3];                                                 \
                                                                                     \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];                                                 \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];                                                 \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];                                                 \
                                                                                     \
    స్థిరరాశి int32_t lp0 = pctx->lp0, rp0 = pctx->rp0;                                  \
    స్థిరరాశి int32_t lp1 = pctx->lp1, rp1 = pctx->rp1;                                  \
    స్థిరరాశి int32_t lp2 = pctx->lp2, rp2 = pctx->rp2;                                  \
    స్థిరరాశి int32_t lp3 = pctx->lp3, rp3 = pctx->rp3;                                  \
                                                                                     \
    స్థిరరాశి size_t type_కొలత = pctx->type_కొలత;                                        \
                                                                                     \
    స్థిరరాశి uint32_t అడ్డువరుస_ప్రారంభం = pctx->అడ్డువరుస_ప్రారంభం + pctx->nrows_చొప్పున_thread * ith;       \
    స్థిరరాశి uint32_t అడ్డువరుస_చివర   = MIN(అడ్డువరుస_ప్రారంభం + pctx->nrows_చొప్పున_thread, pctx->అడ్డువరుస_ప్రారంభం + pctx->మొత్తం_dst_అడ్డువరుసలు);


#నిర్వచించు htp_pad_dma_preamble                                                                \
    స్థిరరాశి size_t src_అడ్డువరుస_కొలత         = pctx->src_అడ్డువరుస_కొలత;                                 \
    స్థిరరాశి size_t src_అడ్డువరుస_కొలత_aligned = pctx->src_అడ్డువరుస_కొలత_aligned;                         \
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత         = pctx->dst_అడ్డువరుస_కొలత;                                 \
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned = pctx->dst_అడ్డువరుస_కొలత_aligned;                         \
                                                                                            \
    uint8_t * src_spad_పునాది = octx->src0_spad.దత్తాంశం + ith * octx->src0_spad.కొలత_చొప్పున_thread; \
    uint8_t * dst_spad_పునాది = octx->dst_spad.దత్తాంశం  + ith * octx->dst_spad.కొలత_చొప్పున_thread;  \
                                                                                            \
    dma_వరుస * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

స్థిర శూన్య pad_పని_చొప్పున_thread_hvx(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    స్థిరరాశి నిర్మాణం htp_pad_context * pctx = (స్థిరరాశి నిర్మాణం htp_pad_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, అడ్డువరుస_ప్రారంభం);

    కొరకు (uint32_t dst_అడ్డువరుస = అడ్డువరుస_ప్రారంభం; dst_అడ్డువరుస < అడ్డువరుస_చివర; dst_అడ్డువరుస++) {
        uint32_t i1, i2, i3;
        pad_decompose_అడ్డువరుస(dst_అడ్డువరుస, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->దత్తాంశం + i1 * nb1 + i2 * nb2 + i3 * nb3;

        స్థిరరాశి పూర్ణాంకం interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ఒకవేళ (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } లేకపోతే {
            స్థిరరాశి uint8_t * src_ptr = pad_src_అడ్డువరుస_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            ఒకవేళ (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_అడ్డువరుస_ప్రారంభం = dst_ptr + (size_t)lp0 * type_కొలత;
            ఒకవేళ (nb00 == type_కొలత) {
                hvx_copy_f32_uu(dst_అడ్డువరుస_ప్రారంభం, src_ptr, ne00);
            } లేకపోతే {
                కొరకు (uint32_t i = 0; i < ne00; i++) {
                    జ్ఞాపకం_నకలు(dst_అడ్డువరుస_ప్రారంభం + i * type_కొలత,
                           src_ptr + (size_t)i * nb00,
                           type_కొలత);
                }
            }

            ఒకవేళ (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_కొలత, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, అడ్డువరుస_ప్రారంభం);

    FARF(ఎగువ_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         అడ్డువరుస_ప్రారంభం, అడ్డువరుస_చివర);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

స్థిర శూన్య pad_పని_చొప్పున_thread_hvx_dma(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    స్థిరరాశి నిర్మాణం htp_pad_context * pctx = (స్థిరరాశి నిర్మాణం htp_pad_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    కొరకు (uint32_t ir = అడ్డువరుస_ప్రారంభం, spad_idx = 0; ir < అడ్డువరుస_చివర && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_పునాది + spad_idx * src_అడ్డువరుస_కొలత_aligned;
        uint8_t * dst_spad_cur = dst_spad_పునాది + spad_idx * dst_అడ్డువరుస_కొలత_aligned;

        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(dst->దత్తాంశం, dst_spad_cur),
            dst_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 0);

        uint32_t i1, i2, i3;
        pad_decompose_అడ్డువరుస(ir, ne1, ne2, &i1, &i2, &i3);
        స్థిరరాశి పూర్ణాంకం interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        స్థిరరాశి dma_addr_t src_దత్తాంశం = interior
            ? pad_src_అడ్డువరుస_దత్తాంశం(src, i1, i2, i3, lp1, lp2, lp3) : src->దత్తాంశం;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(src_spad_cur, src_దత్తాంశం),
            src_అడ్డువరుస_కొలత_aligned, src_అడ్డువరుస_కొలత, src_అడ్డువరుస_కొలత, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    కొరకు (uint32_t ir = అడ్డువరుస_ప్రారంభం; ir < అడ్డువరుస_చివర; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_అడ్డువరుస(ir, ne1, ne2, &i1, &i2, &i3);

        స్థిరరాశి dma_addr_t dst_దత్తాంశం = dst->దత్తాంశం + i1 * nb1 + i2 * nb2 + i3 * nb3;

        స్థిరరాశి పూర్ణాంకం interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ఒకవేళ (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } లేకపోతే {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_కొలత;

            ఒకవేళ ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } లేకపోతే {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(dst_దత్తాంశం, dst_spad_cur),
            dst_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 1);

        స్థిరరాశి uint32_t తదుపరి_అడ్డువరుస = ir + 2;
        ఒకవేళ (తదుపరి_అడ్డువరుస < అడ్డువరుస_చివర) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_అడ్డువరుస(తదుపరి_అడ్డువరుస, ne1, ne2, &ni1, &ni2, &ni3);
            స్థిరరాశి పూర్ణాంకం తదుపరి_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            స్థిరరాశి dma_addr_t తదుపరి_src_దత్తాంశం = తదుపరి_interior
                ? pad_src_అడ్డువరుస_దత్తాంశం(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->దత్తాంశం;

            dma_వరుస_నెట్టు(dma_q,
                dma_make_దత్తాంశం(src_spad_cur, తదుపరి_src_దత్తాంశం),
                src_అడ్డువరుస_కొలత_aligned, src_అడ్డువరుస_కొలత, src_అడ్డువరుస_కొలత, తదుపరి_interior ? 1 : 0);
        }
    }

    dma_వరుస_flush(dma_q);

    FARF(ఎగువ_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         అడ్డువరుస_ప్రారంభం, అడ్డువరుస_చివర);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

స్థిర శూన్య pad_పని_చొప్పున_thread_hvx_circular(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    స్థిరరాశి నిర్మాణం htp_pad_context * pctx = (స్థిరరాశి నిర్మాణం htp_pad_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, అడ్డువరుస_ప్రారంభం);

    కొరకు (uint32_t dst_అడ్డువరుస = అడ్డువరుస_ప్రారంభం; dst_అడ్డువరుస < అడ్డువరుస_చివర; dst_అడ్డువరుస++) {
        uint32_t i1, i2, i3;
        pad_decompose_అడ్డువరుస(dst_అడ్డువరుస, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->దత్తాంశం + i1 * nb1 + i2 * nb2 + i3 * nb3;
        స్థిరరాశి uint8_t * src_అడ్డువరుస = pad_circ_src_అడ్డువరుస_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        ఒకవేళ (nb00 == type_కొలత) {

            ఒకవేళ (lp0 > 0) {
                ఒకవేళ ((uint32_t)lp0 < 32) {
                    జ్ఞాపకం_నకలు(dst_ptr,
                           src_అడ్డువరుస + (size_t)(ne00 - (uint32_t)lp0) * type_కొలత,
                           (size_t)lp0 * type_కొలత);
                } లేకపోతే {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_అడ్డువరుస + (size_t)(ne00 - (uint32_t)lp0) * type_కొలత,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_కొలత, src_అడ్డువరుస, ne00);
            ఒకవేళ (rp0 > 0) {
                ఒకవేళ ((uint32_t)rp0 < 32) {
                    జ్ఞాపకం_నకలు(dst_ptr + ((size_t)lp0 + ne00) * type_కొలత,
                           src_అడ్డువరుస,
                           (size_t)rp0 * type_కొలత);
                } లేకపోతే {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_కొలత,
                                    src_అడ్డువరుస,
                                    (uint32_t)rp0);
                }
            }
        } లేకపోతే {
            కొరకు (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(దశాంశం *)(dst_ptr + i * type_కొలత) =
                    *(స్థిరరాశి దశాంశం *)(src_అడ్డువరుస + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            కొరకు (uint32_t i = 0; i < ne00; i++) {
                *(దశాంశం *)(dst_ptr + ((size_t)lp0 + i) * type_కొలత) =
                    *(స్థిరరాశి దశాంశం *)(src_అడ్డువరుస + (size_t)i * nb00);
            }
            కొరకు (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(దశాంశం *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_కొలత) =
                    *(స్థిరరాశి దశాంశం *)(src_అడ్డువరుస + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, అడ్డువరుస_ప్రారంభం);

    FARF(ఎగువ_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         అడ్డువరుస_ప్రారంభం, అడ్డువరుస_చివర);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

స్థిర శూన్య pad_పని_చొప్పున_thread_hvx_circular_dma(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    స్థిరరాశి నిర్మాణం htp_pad_context * pctx = (స్థిరరాశి నిర్మాణం htp_pad_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    కొరకు (uint32_t ir = అడ్డువరుస_ప్రారంభం, spad_idx = 0; ir < అడ్డువరుస_చివర && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_పునాది + spad_idx * src_అడ్డువరుస_కొలత_aligned;
        uint8_t * dst_spad_cur = dst_spad_పునాది + spad_idx * dst_అడ్డువరుస_కొలత_aligned;

        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(dst->దత్తాంశం, dst_spad_cur),
            dst_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_అడ్డువరుస(ir, ne1, ne2, &pi1, &pi2, &pi3);
        స్థిరరాశి dma_addr_t src_దత్తాంశం = pad_circ_src_అడ్డువరుస_దత్తాంశం(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(src_spad_cur, src_దత్తాంశం),
            src_అడ్డువరుస_కొలత_aligned, src_అడ్డువరుస_కొలత, src_అడ్డువరుస_కొలత, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    కొరకు (uint32_t ir = అడ్డువరుస_ప్రారంభం; ir < అడ్డువరుస_చివర; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_వరుస_తీయి(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_అడ్డువరుస(ir, ne1, ne2, &i1, &i2, &i3);
        స్థిరరాశి dma_addr_t dst_దత్తాంశం = dst->దత్తాంశం + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ఒకవేళ (lp0 > 0) {
            uint8_t * dst_ఎడమ       = dst_spad_cur;
            స్థిరరాశి uint8_t * src_ఎడమ = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_కొలత;
            ఒకవేళ ((uint32_t)lp0 < 32) {
                జ్ఞాపకం_నకలు(dst_ఎడమ, src_ఎడమ, (size_t)lp0 * type_కొలత);
            } లేకపోతే {
                hvx_copy_f32_uu(dst_ఎడమ, src_ఎడమ, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_కొలత;
            ఒకవేళ ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } లేకపోతే {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        ఒకవేళ (rp0 > 0) {
            uint8_t * dst_కుడి = dst_spad_cur + ((size_t)lp0 + ne00) * type_కొలత;
            ఒకవేళ ((uint32_t)rp0 < 32) {
                జ్ఞాపకం_నకలు(dst_కుడి, src_spad_cur, (size_t)rp0 * type_కొలత);
            } లేకపోతే {
                ఒకవేళ ((uintptr_t)dst_కుడి % VLEN == 0) {
                    hvx_copy_f32_aa(dst_కుడి, src_spad_cur, (uint32_t)rp0);
                } లేకపోతే {
                    hvx_copy_f32_ua(dst_కుడి, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(dst_దత్తాంశం, dst_spad_cur),
            dst_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 1);

        స్థిరరాశి uint32_t తదుపరి_అడ్డువరుస = ir + 2;
        ఒకవేళ (తదుపరి_అడ్డువరుస < అడ్డువరుస_చివర) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_అడ్డువరుస(తదుపరి_అడ్డువరుస, ne1, ne2, &nri1, &nri2, &nri3);
            స్థిరరాశి dma_addr_t తదుపరి_src_దత్తాంశం = pad_circ_src_అడ్డువరుస_దత్తాంశం(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_వరుస_నెట్టు(dma_q,
                dma_make_దత్తాంశం(src_spad_cur, తదుపరి_src_దత్తాంశం),
                src_అడ్డువరుస_కొలత_aligned, src_అడ్డువరుస_కొలత, src_అడ్డువరుస_కొలత, 1);
        }
    }

    dma_వరుస_flush(dma_q);

    FARF(ఎగువ_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         అడ్డువరుస_ప్రారంభం, అడ్డువరుస_చివర);
}

పూర్ణాంకం op_pad(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_కొలత;
    ఎంపిక (src0->type) {
        స్థితి HTP_TYPE_F32: type_కొలత = 4; ఆపు;
        అప్రమేయ:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి int32_t lp0 = octx->op_params[0];
    స్థిరరాశి int32_t rp0 = octx->op_params[1];
    స్థిరరాశి int32_t lp1 = octx->op_params[2];
    స్థిరరాశి int32_t rp1 = octx->op_params[3];
    స్థిరరాశి int32_t lp2 = octx->op_params[4];
    స్థిరరాశి int32_t rp2 = octx->op_params[5];
    స్థిరరాశి int32_t lp3 = octx->op_params[6];
    స్థిరరాశి int32_t rp3 = octx->op_params[7];
    స్థిరరాశి int32_t circular = octx->op_params[8];

    స్థిరరాశి uint32_t ne0  = dst->ne[0];
    స్థిరరాశి uint32_t ne00 = src0->ne[0];

    స్థిరరాశి uint32_t మొత్తం_dst_అడ్డువరుసలు = dst->ne[1] * dst->ne[2] * dst->ne[3];
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత     = (size_t)ne0 * type_కొలత;

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = మొత్తం_dst_అడ్డువరుసలు;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, type_కొలత, (uint32_t) dst_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_dst_అడ్డువరుసలు, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    స్థిరరాశి size_t src_అడ్డువరుస_కొలత         = (size_t)ne00 * type_కొలత;
    స్థిరరాశి size_t src_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src_అడ్డువరుస_కొలత, VLEN);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(dst_అడ్డువరుస_కొలత, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    స్థిరరాశి size_t vtcm_needed = (size_t)n_దారాలు * 2 * (src_అడ్డువరుస_కొలత_aligned + dst_అడ్డువరుస_కొలత_aligned);

    స్థిరరాశి పూర్ణాంకం use_dma = (src0->nb[0] == (uint32_t)type_కొలత) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_కొలత >= vtcm_needed);

    ఒకవేళ (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (use_dma) {
        octx->src0_spad.కొలత_చొప్పున_thread = 2 * src_అడ్డువరుస_కొలత_aligned;
        octx->dst_spad.కొలత_చొప్పున_thread  = 2 * dst_అడ్డువరుస_కొలత_aligned;
        octx->src0_spad.కొలత = n_దారాలు * octx->src0_spad.కొలత_చొప్పున_thread;
        octx->dst_spad.కొలత  = n_దారాలు * octx->dst_spad.కొలత_చొప్పున_thread;
        octx->src0_spad.దత్తాంశం = octx->ctx->vtcm_పునాది;
        octx->dst_spad.దత్తాంశం  = octx->src0_spad.దత్తాంశం + octx->src0_spad.కొలత;
        octx->src0_spad.src  = ఖాళీ;
        octx->dst_spad.src   = ఖాళీ;
    }

    నిర్మాణం htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div),
        .మొత్తం_dst_అడ్డువరుసలు   = nrows,
        .అడ్డువరుస_ప్రారంభం        = అడ్డువరుస_ప్రారంభం,
        .type_కొలత        = type_కొలత,
        .src_అడ్డువరుస_కొలత         = src_అడ్డువరుస_కొలత,
        .src_అడ్డువరుస_కొలత_aligned = src_అడ్డువరుస_కొలత_aligned,
        .dst_అడ్డువరుస_కొలత         = dst_అడ్డువరుస_కొలత,
        .dst_అడ్డువరుస_కొలత_aligned = dst_అడ్డువరుస_కొలత_aligned,
    };

    FARF(ఎగువ_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    ఒకవేళ      (circular && use_dma) { work_వరుస_నిరంతరం(octx->ctx->work_వరుస, pad_పని_చొప్పున_thread_hvx_circular_dma, &pctx, n_దారాలు); }
    లేకపోతే ఒకవేళ (circular)            { work_వరుస_నిరంతరం(octx->ctx->work_వరుస, pad_పని_చొప్పున_thread_hvx_circular,     &pctx, n_దారాలు); }
    లేకపోతే ఒకవేళ (use_dma)             { work_వరుస_నిరంతరం(octx->ctx->work_వరుస, pad_పని_చొప్పున_thread_hvx_dma,          &pctx, n_దారాలు); }
    లేకపోతే                          { work_వరుస_నిరంతరం(octx->ctx->work_వరుస, pad_పని_చొప్పున_thread_hvx,              &pctx, n_దారాలు); }

    తిరిగి HTP_స్థితి_OK;
}
