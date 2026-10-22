#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <hexagon_protos.h>
#అన్తర్భావ <hexagon_types.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "hex-common.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "htp-vtcm.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "im2col-ops.h"

నిర్మాణం htp_im2col_context {
    నిర్మాణం htp_ops_context * octx;

    uint32_t pe_అడ్డువరుస_పునాది;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_అడ్డువరుసలు_చొప్పున_thread;                   // output rows per worker
    uint32_t pe_src_అడ్డువరుస_బైట్లు;                     // one output row's source, rounded 256
    uint32_t pe_dst_అడ్డువరుస_బైట్లు;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_కొలత_చొప్పున_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_కొలత_చొప్పున_thread;              // 2 * pe_dst_row_bytes

    uint32_t pe_owb;                               // output-col block size
    uint32_t pe_wb;                                // staged source window width
    uint32_t pe_nbpr;                              // blocks per row

    // Precomputed dimensions for worker threads
    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;

    // Per-thread initial coordinates (removes division from thread prologue)
    uint32_t thread_లోపల[HTP_MAX_NTHREADS];
    uint32_t thread_iod[HTP_MAX_NTHREADS];
    uint32_t thread_ioh[HTP_MAX_NTHREADS];

    శూన్య * compute;
    uint32_t dst_elem;
};

రకనిర్వచనం శూన్య (*im2col_blocked_compute_fn_t)(
    స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,
    శూన్య * పరిమిత dstb, స్థిరరాశి దశాంశం * పరిమిత srcb,
    దశాంశం * పరిమిత scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

రకనిర్వచనం శూన్య (*im2col_patchembed_compute_fn_t)(
    స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,
    శూన్య * పరిమిత dstb, స్థిరరాశి దశాంశం * పరిమిత srcb, uint32_t ioh);

రకనిర్వచనం శూన్య (*im2col_3d_blocked_compute_fn_t)(
    స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,
    శూన్య * పరిమిత dstb, స్థిరరాశి దశాంశం * పరిమిత srcb,
    దశాంశం * పరిమిత scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

రకనిర్వచనం శూన్య (*im2col_3d_patchembed_compute_fn_t)(
    స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,
    శూన్య * పరిమిత dstb, స్థిరరాశి దశాంశం * పరిమిత srcb);

నిర్మాణం im2col_cursor {
    uint32_t r;
    uint32_t లోపల;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

స్థిర అంతఃవరుస శూన్య im2col_అడుగు_అడ్డువరుస(నిర్మాణం im2col_cursor * c, స్థిరరాశి నిర్మాణం htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ఒకవేళ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->లోపల++;
    }
}

స్థిర అంతఃవరుస శూన్య im2col_అడుగు_ఖండం(నిర్మాణం im2col_cursor * c, స్థిరరాశి నిర్మాణం htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ఒకవేళ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ఒకవేళ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->లోపల++;
        }
    }
}

స్థిర శూన్య im2col_blocked_dma_stage(నిర్మాణం htp_im2col_context * ictx, dma_వరుస * dma_q,
                                     దశాంశం * srcb, స్థిరరాశి నిర్మాణం im2col_cursor * c) {
    స్థిరరాశి uint32_t లోపల = c->లోపల, ioh = c->ioh, c0 = c->c0;
    స్థిరరాశి నిర్మాణం htp_ops_context * octx = ictx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    స్థిరరాశి int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    స్థిరరాశి int32_t d1 = octx->op_params[5];
    స్థిరరాశి bool is_2D = (octx->op_params[6] == 1);
    స్థిరరాశి uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    స్థిరరాశి uint32_t KH = ictx->KH;
    స్థిరరాశి uint32_t Wb = ictx->pe_wb;
    స్థిరరాశి dma_addr_t src_దత్తాంశం = src1->దత్తాంశం;
    స్థిరరాశి int32_t win0 = (int32_t) c0 * s0 - p0;
    స్థిరరాశి size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    స్థిరరాశి size_t batch_స్థానభ్రంశం        = is_2D ? (size_t) లోపల * src1->nb[3] : (size_t) లోపల * src1->nb[2];

    కొరకు (uint32_t ikh = 0; ikh < KH; ikh++) {
        స్థిరరాశి int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        ఒకవేళ (iih < 0 || iih >= (int32_t) IH) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(srcb, 0), 0, 0, 0, 0);
            కొనసాగు;
        }
        స్థిరరాశి int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        ఒకవేళ (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        ఒకవేళ (hi <= lo) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(srcb, 0), 0, 0, 0, 0);
            కొనసాగు;
        }
        స్థిరరాశి uint32_t cpw = (uint32_t) (hi - lo);
        దశాంశం * vdst = srcb + (size_t) ikh * Wb + lo;
        స్థిరరాశి dma_addr_t vsrc = src_దత్తాంశం + batch_స్థానభ్రంశం + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vdst, vsrc),
                       (size_t) KH * Wb * పరిమాణం(దశాంశం), src_channel_stride,
                       cpw * పరిమాణం(దశాంశం), IC);
    }
}

#నిర్వచించు IM2_నిలువు_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
స్థిర శూన్య FNAME(                                                                               \
    స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,                                                      \
    శూన్య * పరిమిత dstb_raw, స్థిరరాశి దశాంశం * పరిమిత srcb,                                       \
    దశాంశం * పరిమిత scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * పరిమిత dstb = (DST_CTYPE *) dstb_raw;                                          \
    స్థిరరాశి నిర్మాణం htp_ops_context * octx = ictx->octx;                                            \
    స్థిరరాశి int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    స్థిరరాశి int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    స్థిరరాశి int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    స్థిరరాశి uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    స్థిరరాశి uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    స్థిరరాశి uint32_t Wb = ictx->pe_wb;                                                             \
    స్థిరరాశి uint32_t patch_stride = ictx->patch_stride;                                            \
    స్థిరరాశి int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    కొరకు (uint32_t j = 0; j < nb; j++) {                                                          \
        స్థిరరాశి uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        స్థిరరాశి int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        కొరకు (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            స్థిరరాశి int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            స్థిరరాశి పూర్ణాంకం     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            కొరకు (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * బహిర్గతం_నిరంతరం = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                ఒకవేళ (!okh) {                                                                      \
                    SPLAT_FN(బహిర్గతం_నిరంతరం, 0, KW);                                                    \
                    కొనసాగు;                                                                    \
                }                                                                                \
                స్థిరరాశి దశాంశం * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                ఒకవేళ (d0 == 1) {                                                                   \
                    స్థిరరాశి int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    ఒకవేళ (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    ఒకవేళ (hi <= lo) {                                                              \
                        SPLAT_FN(బహిర్గతం_నిరంతరం, 0, KW);                                                \
                    } లేకపోతే {                                                                     \
                        ఒకవేళ (lo > 0) {                                                            \
                            SPLAT_FN(బహిర్గతం_నిరంతరం, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (బహిర్గతం_నిరంతరం + lo),                                      \
                                (స్థిరరాశి uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        ఒకవేళ (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(బహిర్గతం_నిరంతరం + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    కొనసాగు;                                                                    \
                }                                                                                \
                కొరకు (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    స్థిరరాశి uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    కొరకు (uint32_t c = 0; c < chunk; c++) {                                       \
                        స్థిరరాశి int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (బహిర్గతం_నిరంతరం + ikw), (స్థిరరాశి uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2_నిలువు_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_నిలువు_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, దశాంశం,    hvx_copy_f32_uu,     hvx_splat_f32_u)

స్థిర శూన్య im2col_blocked_dma_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం htp_im2col_context * ictx      = (నిర్మాణం htp_im2col_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context *    octx      = ictx->octx;
    నిర్మాణం htp_thread_trace * పరిమిత tr = &octx->ctx->trace[ith];
    స్థిరరాశి uint32_t OW                     = ictx->OW;
    స్థిరరాశి uint32_t KH                     = ictx->KH;
    స్థిరరాశి uint32_t owb                    = ictx->pe_owb;
    స్థిరరాశి uint32_t nbpr                   = ictx->pe_nbpr;
    స్థిరరాశి uint32_t patch_stride           = ictx->patch_stride;
    స్థిరరాశి uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    స్థిరరాశి dma_addr_t dst_దత్తాంశం             = octx->dst->దత్తాంశం;
    dma_వరుస *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_పునాది              = ictx->pe_vtcm_src + ith * ictx->pe_src_కొలత_చొప్పున_thread;
    uint8_t *      dstb_పునాది              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_కొలత_చొప్పున_thread;
    దశాంశం *        srcb2[2]               = { (దశాంశం *) srcb_పునాది, (దశాంశం *) (srcb_పునాది + ictx->pe_src_అడ్డువరుస_బైట్లు) };
    శూన్య *         dstb2[2]               = { dstb_పునాది, dstb_పునాది + ictx->pe_dst_అడ్డువరుస_బైట్లు };
    దశాంశం *        scratch                = (దశాంశం *) (srcb_పునాది + 2 * ictx->pe_src_అడ్డువరుస_బైట్లు);
    స్థిరరాశి uint32_t అడ్డువరుస_చివర_max            = ictx->pe_అడ్డువరుస_పునాది + ictx->pe_nrows;
    స్థిరరాశి uint32_t అడ్డువరుస_ప్రారంభం              = ictx->pe_అడ్డువరుస_పునాది + ictx->pe_అడ్డువరుసలు_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t అడ్డువరుస_చివర                = MIN(అడ్డువరుస_ప్రారంభం + ictx->pe_అడ్డువరుసలు_చొప్పున_thread, అడ్డువరుస_చివర_max);
    ఒకవేళ (అడ్డువరుస_ప్రారంభం >= అడ్డువరుస_చివర) {
        తిరిగి;
    }
    స్థిరరాశి uint32_t nrows_local  = అడ్డువరుస_చివర - అడ్డువరుస_ప్రారంభం;
    స్థిరరాశి uint32_t మొత్తం_ఖండాలు = nrows_local * nbpr;
    ఒకవేళ (మొత్తం_ఖండాలు == 0) {
        తిరిగి;
    }
    నిర్మాణం im2col_cursor cur = {
        .r   = అడ్డువరుస_ప్రారంభం,
        .లోపల  = ictx->thread_లోపల[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    నిర్మాణం im2col_cursor stage = cur;
    కొరకు (uint32_t i = 0; i < మొత్తం_ఖండాలు && i < 2; i++) {
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_దత్తాంశం, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_అడుగు_ఖండం(&stage, ictx);
    }
    కొరకు (uint32_t bi = 0; bi < మొత్తం_ఖండాలు; bi++) {
        స్థిరరాశి uint32_t buf = bi & 1u;
        దశాంశం * srcb = srcb2[buf];
        శూన్య *  dstb = dstb2[buf];
        స్థిరరాశి uint32_t nb = MIN(owb, OW - cur.c0);
        dma_వరుస_తీయి(dma_q);
        కొరకు (uint32_t k = 0; k < KH; k++) {
            dma_వరుస_తీయి(dma_q);
        }
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        స్థిరరాశి dma_addr_t ddr = dst_దత్తాంశం + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ఒకవేళ (bi + 2 < మొత్తం_ఖండాలు) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_అడుగు_ఖండం(&stage, ictx);
        }
        im2col_అడుగు_ఖండం(&cur, ictx);
    }
    dma_వరుస_flush(dma_q);
}

స్థిర శూన్య im2col_patchembed_dma_stage(నిర్మాణం htp_im2col_context * ictx, dma_వరుస * dma_q,
                                        దశాంశం * srcb, స్థిరరాశి నిర్మాణం im2col_cursor * c) {
    స్థిరరాశి నిర్మాణం htp_ops_context * octx = ictx->octx;
    స్థిరరాశి uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    స్థిరరాశి uint32_t KH = ictx->KH;
    స్థిరరాశి uint32_t లోపల = c->లోపల, ioh = c->ioh;
    స్థిరరాశి size_t src_channel_stride  = (size_t) IH * IW * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t vtcm_channel_stride = (size_t) KH * IW * పరిమాణం(దశాంశం);
    స్థిరరాశి dma_addr_t src_దత్తాంశం        = octx->src[1]->దత్తాంశం;
    స్థిరరాశి dma_addr_t vsrc            = src_దత్తాంశం + (size_t) (((లోపల * IC) * IH + ioh * KH) * IW) * పరిమాణం(దశాంశం);

    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

స్థిర శూన్య im2col_patchembed_compute_f16(
    స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,
    శూన్య * పరిమిత dstb_raw, స్థిరరాశి దశాంశం * పరిమిత srcb, uint32_t ioh) {
    uint16_t * పరిమిత dstb = (uint16_t *) dstb_raw;
    స్థిరరాశి uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    స్థిరరాశి uint32_t KH = ictx->KH, KW = ictx->KW;
    స్థిరరాశి uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    కొరకు (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        కొరకు (uint32_t ikh = 0; ikh < KH; ikh++) {
            స్థిరరాశి int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            కొరకు (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * బహిర్గతం_నిరంతరం = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ఒకవేళ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(బహిర్గతం_నిరంతరం, 0, KW);
                    కొనసాగు;
                }
                స్థిరరాశి దశాంశం * src_నిరంతరం = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) బహిర్గతం_నిరంతరం, (స్థిరరాశి uint8_t *) src_నిరంతరం, KW);
            }
        }
    }
}

స్థిర శూన్య im2col_patchembed_compute_f32(
    స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,
    శూన్య * పరిమిత dstb_raw, స్థిరరాశి దశాంశం * పరిమిత srcb, uint32_t ioh) {
    దశాంశం * పరిమిత dstb = (దశాంశం *) dstb_raw;
    స్థిరరాశి uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    స్థిరరాశి uint32_t KH = ictx->KH, KW = ictx->KW;
    స్థిరరాశి uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    కొరకు (uint32_t iow = 0; iow < OW; iow++) {
        దశాంశం * dst_patch = dstb + (size_t) iow * patch_stride;
        కొరకు (uint32_t ikh = 0; ikh < KH; ikh++) {
            స్థిరరాశి int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            కొరకు (uint32_t iic = 0; iic < IC; iic++) {
                దశాంశం * బహిర్గతం_నిరంతరం = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ఒకవేళ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(బహిర్గతం_నిరంతరం, 0, KW);
                    కొనసాగు;
                }
                స్థిరరాశి దశాంశం * src_నిరంతరం = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) బహిర్గతం_నిరంతరం, (స్థిరరాశి uint8_t *) src_నిరంతరం, KW);
            }
        }
    }
}

స్థిర శూన్య im2col_patchembed_dma_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం htp_im2col_context * ictx        = (నిర్మాణం htp_im2col_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context *    octx        = ictx->octx;
    నిర్మాణం htp_thread_trace * పరిమిత tr   = &octx->ctx->trace[ith];
    స్థిరరాశి uint32_t OW                       = ictx->OW;
    స్థిరరాశి uint32_t patch_stride             = ictx->patch_stride;
    స్థిరరాశి uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    స్థిరరాశి dma_addr_t dst_దత్తాంశం               = octx->dst->దత్తాంశం;
    dma_వరుస *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_పునాది                = ictx->pe_vtcm_src + ith * ictx->pe_src_కొలత_చొప్పున_thread;
    uint8_t *      dstb_పునాది                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_కొలత_చొప్పున_thread;
    దశాంశం *        srcb2[2]                 = { (దశాంశం *) srcb_పునాది, (దశాంశం *) (srcb_పునాది + ictx->pe_src_అడ్డువరుస_బైట్లు) };
    శూన్య *         dstb2[2]                 = { dstb_పునాది, dstb_పునాది + ictx->pe_dst_అడ్డువరుస_బైట్లు };
    స్థిరరాశి uint32_t అడ్డువరుస_చివర_max              = ictx->pe_అడ్డువరుస_పునాది + ictx->pe_nrows;
    స్థిరరాశి uint32_t అడ్డువరుస_ప్రారంభం                = ictx->pe_అడ్డువరుస_పునాది + ictx->pe_అడ్డువరుసలు_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t అడ్డువరుస_చివర                  = MIN(అడ్డువరుస_ప్రారంభం + ictx->pe_అడ్డువరుసలు_చొప్పున_thread, అడ్డువరుస_చివర_max);
    ఒకవేళ (అడ్డువరుస_ప్రారంభం >= అడ్డువరుస_చివర) {
        తిరిగి;
    }
    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = అడ్డువరుస_చివర - అడ్డువరుస_ప్రారంభం;
    ఒకవేళ (మొత్తం_అడ్డువరుసలు == 0) {
        తిరిగి;
    }
    నిర్మాణం im2col_cursor cur = {
        .r   = అడ్డువరుస_ప్రారంభం,
        .లోపల  = ictx->thread_లోపల[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    నిర్మాణం im2col_cursor stage = cur;
    కొరకు (uint32_t i = 0; i < మొత్తం_అడ్డువరుసలు && i < 2; i++) {
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_దత్తాంశం, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_అడుగు_అడ్డువరుస(&stage, ictx);
    }
    కొరకు (uint32_t ri = 0; ri < మొత్తం_అడ్డువరుసలు; ri++) {
        స్థిరరాశి uint32_t buf = ri & 1u;
        దశాంశం * srcb = srcb2[buf];
        శూన్య *  dstb = dstb2[buf];
        dma_వరుస_తీయి(dma_q);
        dma_వరుస_తీయి(dma_q);
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        స్థిరరాశి dma_addr_t ddr_అడ్డువరుస = dst_దత్తాంశం + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(ddr_అడ్డువరుస, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ఒకవేళ (ri + 2 < మొత్తం_అడ్డువరుసలు) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_అడుగు_అడ్డువరుస(&stage, ictx);
        }
        im2col_అడుగు_అడ్డువరుస(&cur, ictx);
    }
    dma_వరుస_flush(dma_q);
}

స్థిర అంతఃవరుస శూన్య im2col_3d_అడుగు_అడ్డువరుస(నిర్మాణం im2col_cursor * c, స్థిరరాశి నిర్మాణం htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ఒకవేళ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        ఒకవేళ (c->iod >= ictx->OD) {
            c->iod = 0;
            c->లోపల++;
        }
    }
}

స్థిర అంతఃవరుస శూన్య im2col_3d_అడుగు_ఖండం(నిర్మాణం im2col_cursor * c, స్థిరరాశి నిర్మాణం htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ఒకవేళ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ఒకవేళ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            ఒకవేళ (c->iod >= ictx->OD) {
                c->iod = 0;
                c->లోపల++;
            }
        }
    }
}

స్థిర శూన్య im2col_3d_blocked_dma_stage(నిర్మాణం htp_im2col_context * ictx, dma_వరుస * dma_q,
                                        దశాంశం * srcb, స్థిరరాశి నిర్మాణం im2col_cursor * c) {
    స్థిరరాశి uint32_t లోపల = c->లోపల, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    స్థిరరాశి నిర్మాణం htp_ops_context * octx = ictx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    స్థిరరాశి int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    స్థిరరాశి int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    స్థిరరాశి uint32_t IC = ictx->IC;
    స్థిరరాశి uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    స్థిరరాశి uint32_t KD = ictx->KD, KH = ictx->KH;
    స్థిరరాశి uint32_t Wb = ictx->pe_wb;
    స్థిరరాశి int32_t win0 = (int32_t) c0 * s0 - p0;
    స్థిరరాశి dma_addr_t src_దత్తాంశం = src1->దత్తాంశం;
    స్థిరరాశి size_t src_channel_stride = src1->nb[3];
    స్థిరరాశి size_t vtcm_channel_stride = (size_t) KD * KH * Wb * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t batch_స్థానభ్రంశం        = (size_t) (లోపల * IC) * src1->nb[3];

    కొరకు (uint32_t ikd = 0; ikd < KD; ikd++) {
        స్థిరరాశి int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        స్థిరరాశి bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        కొరకు (uint32_t ikh = 0; ikh < KH; ikh++) {
            స్థిరరాశి int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            ఒకవేళ (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(srcb, 0), 0, 0, 0, 0);
                కొనసాగు;
            }
            స్థిరరాశి int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            ఒకవేళ (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            ఒకవేళ (hi <= lo) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(srcb, 0), 0, 0, 0, 0);
                కొనసాగు;
            }
            దశాంశం * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            స్థిరరాశి dma_addr_t vsrc = src_దత్తాంశం + batch_స్థానభ్రంశం +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * పరిమాణం(దశాంశం), IC);
        }
    }
}

#నిర్వచించు IM2_నిలువు_3_D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
స్థిర శూన్య FNAME(స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,                                            \
                  శూన్య * పరిమిత dstb_raw, స్థిరరాశి దశాంశం * పరిమిత srcb,                             \
                  దశాంశం * పరిమిత scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * పరిమిత dstb = (DST_CTYPE *) dstb_raw;                                              \
    స్థిరరాశి నిర్మాణం htp_ops_context * octx = ictx->octx;                                                \
    స్థిరరాశి int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    స్థిరరాశి int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    స్థిరరాశి int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    స్థిరరాశి uint32_t IC = ictx->IC;                                                                    \
    స్థిరరాశి uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    స్థిరరాశి uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    స్థిరరాశి uint32_t patch_stride = ictx->patch_stride;                                                \
    స్థిరరాశి uint32_t Wb = ictx->pe_wb;                                                                 \
    స్థిరరాశి int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    కొరకు (uint32_t j = 0; j < nb; j++) {                                                              \
        స్థిరరాశి uint32_t iow = c0 + j;                                                                 \
        స్థిరరాశి int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        కొరకు (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            కొరకు (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                స్థిరరాశి int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                కొరకు (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    స్థిరరాశి int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * బహిర్గతం_నిరంతరం = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    ఒకవేళ (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(బహిర్గతం_నిరంతరం, 0, KW);                                                    \
                        కొనసాగు;                                                                    \
                    }                                                                                \
                    స్థిరరాశి దశాంశం * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    ఒకవేళ (d0 == 1) {                                                                   \
                        స్థిరరాశి int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        ఒకవేళ (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        ఒకవేళ (hi <= lo) {                                                              \
                            SPLAT_FN(బహిర్గతం_నిరంతరం, 0, KW);                                                \
                        } లేకపోతే {                                                                     \
                            ఒకవేళ (lo > 0) {                                                            \
                                SPLAT_FN(బహిర్గతం_నిరంతరం, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (బహిర్గతం_నిరంతరం + lo),                                      \
                                    (స్థిరరాశి uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            ఒకవేళ (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(బహిర్గతం_నిరంతరం + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        కొనసాగు;                                                                    \
                    }                                                                                \
                    కొరకు (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        స్థిరరాశి uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        కొరకు (uint32_t c = 0; c < chunk; c++) {                                       \
                            స్థిరరాశి int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (బహిర్గతం_నిరంతరం + ikw), (స్థిరరాశి uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2_నిలువు_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_నిలువు_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, దశాంశం,    hvx_copy_f32_uu,     hvx_splat_f32_u)

స్థిర శూన్య im2col_3d_blocked_dma_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం htp_im2col_context * ictx        = (నిర్మాణం htp_im2col_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context *    octx        = ictx->octx;
    నిర్మాణం htp_thread_trace * పరిమిత tr   = &octx->ctx->trace[ith];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst  = octx->dst;
    స్థిరరాశి uint32_t OW           = ictx->OW;
    స్థిరరాశి uint32_t KD = ictx->KD, KH = ictx->KH;
    స్థిరరాశి uint32_t kd_kh        = KD * KH;
    స్థిరరాశి uint32_t patch_stride = ictx->patch_stride;
    స్థిరరాశి uint32_t owb          = ictx->pe_owb;
    స్థిరరాశి uint32_t nbpr         = ictx->pe_nbpr;
    స్థిరరాశి uint32_t dst_elem     = ictx->dst_elem;
    స్థిరరాశి dma_addr_t dst_దత్తాంశం   = dst->దత్తాంశం;
    dma_వరుస * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_పునాది         = ictx->pe_vtcm_src + ith * ictx->pe_src_కొలత_చొప్పున_thread;
    uint8_t * dstb_పునాది         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_కొలత_చొప్పున_thread;
    దశాంశం * srcb2[2]            = { (దశాంశం *) srcb_పునాది, (దశాంశం *) (srcb_పునాది + ictx->pe_src_అడ్డువరుస_బైట్లు) };
    శూన్య * dstb2[2]             = { dstb_పునాది, dstb_పునాది + ictx->pe_dst_అడ్డువరుస_బైట్లు };
    దశాంశం * scratch             = (దశాంశం *) (srcb_పునాది + 2 * ictx->pe_src_అడ్డువరుస_బైట్లు);
    స్థిరరాశి uint32_t అడ్డువరుస_చివర_max  = ictx->pe_అడ్డువరుస_పునాది + ictx->pe_nrows;
    స్థిరరాశి uint32_t అడ్డువరుస_ప్రారంభం    = ictx->pe_అడ్డువరుస_పునాది + ictx->pe_అడ్డువరుసలు_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t అడ్డువరుస_చివర      = MIN(అడ్డువరుస_ప్రారంభం + ictx->pe_అడ్డువరుసలు_చొప్పున_thread, అడ్డువరుస_చివర_max);
    ఒకవేళ (అడ్డువరుస_ప్రారంభం >= అడ్డువరుస_చివర) {
        తిరిగి;
    }
    స్థిరరాశి uint32_t nrows_local  = అడ్డువరుస_చివర - అడ్డువరుస_ప్రారంభం;
    స్థిరరాశి uint32_t మొత్తం_ఖండాలు = nrows_local * nbpr;
    ఒకవేళ (మొత్తం_ఖండాలు == 0) {
        తిరిగి;
    }
    నిర్మాణం im2col_cursor cur = {
        .r   = అడ్డువరుస_ప్రారంభం,
        .లోపల  = ictx->thread_లోపల[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    నిర్మాణం im2col_cursor stage = cur;
    కొరకు (uint32_t i = 0; i < మొత్తం_ఖండాలు && i < 2; i++) {
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_దత్తాంశం, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_అడుగు_ఖండం(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    కొరకు (uint32_t bi = 0; bi < మొత్తం_ఖండాలు; bi++) {
        స్థిరరాశి uint32_t buf = bi & 1u;
        దశాంశం * srcb = srcb2[buf];
        శూన్య *  dstb = dstb2[buf];
        స్థిరరాశి uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_వరుస_తీయి(dma_q);
        కొరకు (uint32_t p = 0; p < kd_kh; p++) {
            dma_వరుస_తీయి(dma_q);
        }
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        స్థిరరాశి dma_addr_t ddr = dst_దత్తాంశం + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ఒకవేళ (bi + 2 < మొత్తం_ఖండాలు) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_అడుగు_ఖండం(&stage, ictx);
        }
        im2col_3d_అడుగు_ఖండం(&cur, ictx);
    }
    dma_వరుస_flush(dma_q);
}
స్థిర శూన్య im2col_3d_patchembed_dma_stage(నిర్మాణం htp_im2col_context * ictx, dma_వరుస * dma_q,
                                           దశాంశం * srcb, స్థిరరాశి నిర్మాణం im2col_cursor * c) {
    స్థిరరాశి నిర్మాణం htp_ops_context * octx = ictx->octx;
    స్థిరరాశి uint32_t IC = ictx->IC;
    స్థిరరాశి uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    స్థిరరాశి uint32_t KD = ictx->KD, KH = ictx->KH;
    స్థిరరాశి dma_addr_t src_దత్తాంశం = octx->src[1]->దత్తాంశం;
    స్థిరరాశి uint32_t లోపల = c->లోపల, iod = c->iod, ioh = c->ioh;
    స్థిరరాశి size_t src_channel_stride  = (size_t) ID * IH * IW * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t vtcm_channel_stride = (size_t) KD * KH * IW * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t అడ్డువరుస_కొలత            = (size_t) KH * IW * పరిమాణం(దశాంశం);

    కొరకు (uint32_t ikd = 0; ikd < KD; ikd++) {
        స్థిరరాశి uint32_t iid = iod * KD + ikd;
        దశాంశం * vdst = srcb + (size_t) ikd * KH * IW;
        స్థిరరాశి dma_addr_t vsrc = src_దత్తాంశం + (size_t) (((లోపల * IC) * ID + iid) * IH + ioh * KH) * IW * పరిమాణం(దశాంశం);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, అడ్డువరుస_కొలత, IC);
    }
}
స్థిర శూన్య im2col_3d_patchembed_compute_f16(
    స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,
    శూన్య * పరిమిత dstb_raw, స్థిరరాశి దశాంశం * పరిమిత srcb) {
    uint16_t * పరిమిత dstb = (uint16_t *) dstb_raw;
    స్థిరరాశి uint32_t OW           = ictx->OW;
    స్థిరరాశి uint32_t IC           = ictx->IC;
    స్థిరరాశి uint32_t KD           = ictx->KD;
    స్థిరరాశి uint32_t KH           = ictx->KH;
    స్థిరరాశి uint32_t KW           = ictx->KW;
    స్థిరరాశి uint32_t IW           = ictx->IW;
    స్థిరరాశి uint32_t patch_stride = ictx->patch_stride;

    కొరకు (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        కొరకు (uint32_t iic = 0; iic < IC; iic++) {
            కొరకు (uint32_t ikd = 0; ikd < KD; ikd++) {
                కొరకు (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * బహిర్గతం_నిరంతరం = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    స్థిరరాశి దశాంశం * src_నిరంతరం = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) బహిర్గతం_నిరంతరం, (స్థిరరాశి uint8_t *) src_నిరంతరం, KW);
                }
            }
        }
    }
}

స్థిర శూన్య im2col_3d_patchembed_compute_f32(
    స్థిరరాశి నిర్మాణం htp_im2col_context * ictx,
    శూన్య * పరిమిత dstb_raw, స్థిరరాశి దశాంశం * పరిమిత srcb) {
    దశాంశం * పరిమిత dstb = (దశాంశం *) dstb_raw;
    స్థిరరాశి uint32_t OW           = ictx->OW;
    స్థిరరాశి uint32_t IC           = ictx->IC;
    స్థిరరాశి uint32_t KD           = ictx->KD;
    స్థిరరాశి uint32_t KH           = ictx->KH;
    స్థిరరాశి uint32_t KW           = ictx->KW;
    స్థిరరాశి uint32_t IW           = ictx->IW;
    స్థిరరాశి uint32_t patch_stride = ictx->patch_stride;

    కొరకు (uint32_t iow = 0; iow < OW; iow++) {
        దశాంశం * dst_patch = dstb + (size_t) iow * patch_stride;
        కొరకు (uint32_t iic = 0; iic < IC; iic++) {
            కొరకు (uint32_t ikd = 0; ikd < KD; ikd++) {
                కొరకు (uint32_t ikh = 0; ikh < KH; ikh++) {
                    దశాంశం * బహిర్గతం_నిరంతరం = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    స్థిరరాశి దశాంశం * src_నిరంతరం = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) బహిర్గతం_నిరంతరం, (స్థిరరాశి uint8_t *) src_నిరంతరం, KW);
                }
            }
        }
    }
}

స్థిర శూన్య im2col_3d_patchembed_dma_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం htp_im2col_context * ictx        = (నిర్మాణం htp_im2col_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context *    octx        = ictx->octx;
    నిర్మాణం htp_thread_trace * పరిమిత tr   = &octx->ctx->trace[ith];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst  = octx->dst;
    స్థిరరాశి uint32_t KD           = ictx->KD;
    స్థిరరాశి uint32_t OW           = ictx->OW;
    స్థిరరాశి uint32_t patch_stride = ictx->patch_stride;
    స్థిరరాశి uint32_t dst_elem     = ictx->dst_elem;
    స్థిరరాశి dma_addr_t dst_దత్తాంశం   = dst->దత్తాంశం;
    dma_వరుస * dma_q           = octx->ctx->dma[ith];
    స్థిరరాశి uint32_t అడ్డువరుస_చివర_max  = ictx->pe_అడ్డువరుస_పునాది + ictx->pe_nrows;
    స్థిరరాశి uint32_t అడ్డువరుస_ప్రారంభం    = ictx->pe_అడ్డువరుస_పునాది + ictx->pe_అడ్డువరుసలు_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t అడ్డువరుస_చివర      = MIN(అడ్డువరుస_ప్రారంభం + ictx->pe_అడ్డువరుసలు_చొప్పున_thread, అడ్డువరుస_చివర_max);
    ఒకవేళ (అడ్డువరుస_ప్రారంభం >= అడ్డువరుస_చివర) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t nrows_local  = అడ్డువరుస_చివర - అడ్డువరుస_ప్రారంభం;
    uint8_t * src_పునాది          = ictx->pe_vtcm_src + ith * ictx->pe_src_కొలత_చొప్పున_thread;
    uint8_t * dstb_పునాది         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_కొలత_చొప్పున_thread;
    దశాంశం * srcb2[2]            = { (దశాంశం *) src_పునాది, (దశాంశం *) (src_పునాది + ictx->pe_src_అడ్డువరుస_బైట్లు) };
    శూన్య * dstb2[2]             = { dstb_పునాది, dstb_పునాది + ictx->pe_dst_అడ్డువరుస_బైట్లు };
    నిర్మాణం im2col_cursor cur = {
        .r   = అడ్డువరుస_ప్రారంభం,
        .లోపల  = ictx->thread_లోపల[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    నిర్మాణం im2col_cursor stage = cur;
    కొరకు (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_దత్తాంశం, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_అడుగు_అడ్డువరుస(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    కొరకు (uint32_t idx = 0; idx < nrows_local; idx++) {
        స్థిరరాశి uint32_t buf = idx & 1u;
        దశాంశం * srcb = srcb2[buf];
        శూన్య *  dstb = dstb2[buf];
        dma_వరుస_తీయి(dma_q);
        కొరకు (uint32_t p = 0; p < KD; p++) {
            dma_వరుస_తీయి(dma_q);
        }
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        స్థిరరాశి dma_addr_t ddr_అడ్డువరుస = dst_దత్తాంశం + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(ddr_అడ్డువరుస, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ఒకవేళ (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_అడుగు_అడ్డువరుస(&stage, ictx);
        }
        im2col_3d_అడుగు_అడ్డువరుస(&cur, ictx);
    }
    dma_వరుస_flush(dma_q);
}

పూర్ణాంకం op_im2col(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_im2col_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_im2col_kernel_params *) octx->kernel_params;
    ఒకవేళ (kparams->kernel_type == HTP_IM2_నిలువు_KERNEL_UNSUPPORTED) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;

    uint32_t అడ్డువరుస_పునాది = 0;
    uint32_t nrows    = kparams->మొత్తం_అడ్డువరుసలు;
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి uint32_t అడ్డువరుస_కొలత = dst->nb[2];
        స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_chunk = (అడ్డువరుస_కొలత > 0) ? (HEX_L2_పంక్తి_కొలత / hex_గరిష్ఠ_సామాన్య_భాజకం_u32(అడ్డువరుస_కొలత, HEX_L2_పంక్తి_కొలత)) : 1;
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range =
            htp_tensor_mdev_విభజన(kparams->మొత్తం_అడ్డువరుసలు, htp_tensor_mdev_దత్తాంశం_aligned(dst) ? అడ్డువరుసలు_చొప్పున_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_పునాది = range.ప్రారంభం;
        nrows    = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    నిర్మాణం htp_im2col_context ictx = { 0 };
    uint8_t * స్థిరరాశి పునాది = octx->ctx->vtcm_పునాది;

    స్థిరరాశి uint32_t pth = MIN(octx->n_దారాలు, nrows);
    ictx.octx                   = octx;
    ictx.pe_అడ్డువరుస_పునాది            = అడ్డువరుస_పునాది;
    ictx.pe_nrows               = nrows;
    ictx.pe_అడ్డువరుసలు_చొప్పున_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_అడ్డువరుస_బైట్లు       = kparams->src_అడ్డువరుస_బైట్లు;
    ictx.pe_dst_అడ్డువరుస_బైట్లు       = kparams->dst_అడ్డువరుస_బైట్లు;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, పునాది, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, పునాది, kparams->off_dst);
    ictx.pe_src_కొలత_చొప్పున_thread = kparams->src_బైట్లు_చొప్పున_thread;
    ictx.pe_dst_కొలత_చొప్పున_thread = kparams->dst_బైట్లు_చొప్పున_thread;
    ictx.pe_owb                 = kparams->owb;
    ictx.pe_wb                  = kparams->wb;
    ictx.pe_nbpr                = kparams->nbpr;
    ictx.N                      = kparams->N;
    ictx.IC                     = kparams->IC;
    ictx.ID                     = kparams->ID;
    ictx.IH                     = kparams->IH;
    ictx.IW                     = kparams->IW;
    ictx.KD                     = kparams->KD;
    ictx.KH                     = kparams->KH;
    ictx.KW                     = kparams->KW;
    ictx.OD                     = kparams->OD;
    ictx.OH                     = kparams->OH;
    ictx.OW                     = kparams->OW;
    ictx.patch_stride           = kparams->patch_stride;

    కొరకు (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = అడ్డువరుస_పునాది + ictx.pe_అడ్డువరుసలు_చొప్పున_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_లోపల[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? పరిమాణం(__fp16) : పరిమాణం(దశాంశం);

    ఒకవేళ (kparams->kernel_type == HTP_IM2_నిలువు_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (శూన్య *) im2col_patchembed_compute_f16
                                                 : (శూన్య *) im2col_patchembed_compute_f32;
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, im2col_patchembed_dma_thread, &ictx, pth);
        తిరిగి HTP_స్థితి_OK;
    } లేకపోతే ఒకవేళ (kparams->kernel_type == HTP_IM2_నిలువు_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (శూన్య *) im2col_blocked_compute_f16
                                                 : (శూన్య *) im2col_blocked_compute_f32;
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, im2col_blocked_dma_thread, &ictx, pth);
        తిరిగి HTP_స్థితి_OK;
    }

    తిరిగి HTP_స్థితి_NO_SUPPORT;
}

పూర్ణాంకం op_im2col_3d(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_im2col_kernel_params * kparams =
        (స్థిరరాశి నిర్మాణం htp_im2col_kernel_params *) octx->kernel_params;
    ఒకవేళ (kparams->kernel_type == HTP_IM2_నిలువు_KERNEL_UNSUPPORTED) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;

    uint32_t అడ్డువరుస_పునాది = 0;
    uint32_t nrows    = kparams->మొత్తం_అడ్డువరుసలు;
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి uint32_t అడ్డువరుస_కొలత = dst->nb[2];
        స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_chunk = (అడ్డువరుస_కొలత > 0) ? (HEX_L2_పంక్తి_కొలత / hex_గరిష్ఠ_సామాన్య_భాజకం_u32(అడ్డువరుస_కొలత, HEX_L2_పంక్తి_కొలత)) : 1;
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range =
            htp_tensor_mdev_విభజన(kparams->మొత్తం_అడ్డువరుసలు, htp_tensor_mdev_దత్తాంశం_aligned(dst) ? అడ్డువరుసలు_చొప్పున_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_పునాది = range.ప్రారంభం;
        nrows    = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    నిర్మాణం htp_im2col_context ictx = { 0 };
    uint8_t * స్థిరరాశి పునాది = octx->ctx->vtcm_పునాది;

    స్థిరరాశి uint32_t pth = MIN(octx->n_దారాలు, nrows);
    ictx.octx                   = octx;
    ictx.pe_అడ్డువరుస_పునాది            = అడ్డువరుస_పునాది;
    ictx.pe_nrows               = nrows;
    ictx.pe_అడ్డువరుసలు_చొప్పున_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_అడ్డువరుస_బైట్లు       = kparams->src_అడ్డువరుస_బైట్లు;
    ictx.pe_dst_అడ్డువరుస_బైట్లు       = kparams->dst_అడ్డువరుస_బైట్లు;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, పునాది, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, పునాది, kparams->off_dst);
    ictx.pe_src_కొలత_చొప్పున_thread = kparams->src_బైట్లు_చొప్పున_thread;
    ictx.pe_dst_కొలత_చొప్పున_thread = kparams->dst_బైట్లు_చొప్పున_thread;
    ictx.pe_owb                 = kparams->owb;
    ictx.pe_wb                  = kparams->wb;
    ictx.pe_nbpr                = kparams->nbpr;
    ictx.N                      = kparams->N;
    ictx.IC                     = kparams->IC;
    ictx.ID                     = kparams->ID;
    ictx.IH                     = kparams->IH;
    ictx.IW                     = kparams->IW;
    ictx.KD                     = kparams->KD;
    ictx.KH                     = kparams->KH;
    ictx.KW                     = kparams->KW;
    ictx.OD                     = kparams->OD;
    ictx.OH                     = kparams->OH;
    ictx.OW                     = kparams->OW;
    ictx.patch_stride           = kparams->patch_stride;

    కొరకు (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = అడ్డువరుస_పునాది + ictx.pe_అడ్డువరుసలు_చొప్పున_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_లోపల[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? పరిమాణం(__fp16) : పరిమాణం(దశాంశం);

    ఒకవేళ (kparams->kernel_type == HTP_IM2_నిలువు_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (శూన్య *) im2col_3d_patchembed_compute_f16
                                                 : (శూన్య *) im2col_3d_patchembed_compute_f32;
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, im2col_3d_patchembed_dma_thread, &ictx, pth);
        తిరిగి HTP_స్థితి_OK;
    } లేకపోతే ఒకవేళ (kparams->kernel_type == HTP_IM2_నిలువు_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (శూన్య *) im2col_3d_blocked_compute_f16
                                                 : (శూన్య *) im2col_3d_blocked_compute_f32;
        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, im2col_3d_blocked_dma_thread, &ictx, pth);
        తిరిగి HTP_స్థితి_OK;
    }

    తిరిగి HTP_స్థితి_NO_SUPPORT;
}
