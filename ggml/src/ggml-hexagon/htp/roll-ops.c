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
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"

నిర్మాణం htp_roll_context {
    నిర్మాణం htp_ops_context * octx;

    uint32_t అడ్డువరుస_ప్రారంభం;
    uint32_t nrows;
    uint32_t nrows_చొప్పున_thread;

    నిర్మాణం fastdiv_విలువలు div_ne1;
    నిర్మాణం fastdiv_విలువలు div_ne2_ne1;
};

స్థిర అంతఃవరుస uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    ఒకవేళ (i < 0) {
        తిరిగి (uint32_t) (i + (int32_t) ne);
    }
    ఒకవేళ ((uint32_t) i >= ne) {
        తిరిగి (uint32_t) i - ne;
    }
    తిరిగి (uint32_t) i;
}

#నిర్వచించు htp_roll_preamble                             \
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];    \
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;       \
                                                      \
    స్థిరరాశి uint32_t ne0 = dst->ne[0];                  \
    స్థిరరాశి uint32_t ne1 = dst->ne[1];                  \
    స్థిరరాశి uint32_t ne2 = dst->ne[2];                  \
    స్థిరరాశి uint32_t ne3 = dst->ne[3];                  \
                                                      \
    స్థిరరాశి uint32_t nb01 = src0->nb[1];                \
    స్థిరరాశి uint32_t nb02 = src0->nb[2];                \
    స్థిరరాశి uint32_t nb03 = src0->nb[3];                \
                                                      \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];                  \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];                  \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];                  \
                                                      \
    స్థిరరాశి int32_t s0 = octx->op_params[0];            \
    స్థిరరాశి int32_t s1 = octx->op_params[1];            \
    స్థిరరాశి int32_t s2 = octx->op_params[2];            \
    స్థిరరాశి int32_t s3 = octx->op_params[3];            \
                                                      \
    స్థిరరాశి uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    స్థిరరాశి uint32_t n0      = ne0 - i0_src0;

#నిర్వచించు htp_roll_dma_preamble dma_వరుస * q = octx->ctx->dma[0];

స్థిర అంతఃవరుస శూన్య roll_dma_నెట్టు(dma_వరుస * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    బైట్లు,
                                 uint32_t    nrows) {
    ఒకవేళ (బైట్లు == 0 || nrows == 0) {
        తిరిగి;
    }

    ఒకవేళ (!dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(dst, src), dst_stride, src_stride, బైట్లు, nrows)) {
        dma_వరుస_flush(q);
        dma_వరుస_నెట్టు(q, dma_make_దత్తాంశం(dst, src),
                       dst_stride, src_stride, బైట్లు, nrows);
    }
}

స్థిర అంతఃవరుస శూన్య roll_dma_నెట్టు_అడ్డువరుసలు(dma_వరుస *               q,
                                      స్థిరరాశి నిర్మాణం htp_tensor * dst,
                                      స్థిరరాశి నిర్మాణం htp_tensor * src0,
                                      uint32_t                  dst_అడ్డువరుస,
                                      uint32_t                  src_అడ్డువరుస,
                                      uint32_t                  nrows,
                                      uint32_t                  అడ్డువరుస_కొలత,
                                      uint32_t                  i0_src0) {
    స్థిరరాశి dma_addr_t dst_పునాది = dst->దత్తాంశం + (size_t) dst_అడ్డువరుస * అడ్డువరుస_కొలత;
    స్థిరరాశి dma_addr_t src_పునాది = src0->దత్తాంశం + (size_t) src_అడ్డువరుస * అడ్డువరుస_కొలత;
    స్థిరరాశి uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_నెట్టు(q, dst_పునాది, src_పునాది + (size_t) i0_src0 * పరిమాణం(దశాంశం),
                  అడ్డువరుస_కొలత, అడ్డువరుస_కొలత, n0 * పరిమాణం(దశాంశం), nrows);
    roll_dma_నెట్టు(q, dst_పునాది + (size_t) n0 * పరిమాణం(దశాంశం), src_పునాది,
                  అడ్డువరుస_కొలత, అడ్డువరుస_కొలత, i0_src0 * పరిమాణం(దశాంశం), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
స్థిర అంతఃవరుస శూన్య roll_dma_నెట్టు_range(dma_వరుస * q,
                                       dma_addr_t  dst_అడ్డువరుస,
                                       dma_addr_t  src_అడ్డువరుస,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_నెట్టు(q, dst_అడ్డువరుస, src_అడ్డువరుస + (size_t) i0_src0 * పరిమాణం(దశాంశం),
                  dst_stride, src_stride, n0 * పరిమాణం(దశాంశం), nrows);
    roll_dma_నెట్టు(q, dst_అడ్డువరుస + (size_t) n0 * పరిమాణం(దశాంశం), src_అడ్డువరుస,
                  dst_stride, src_stride, i0_src0 * పరిమాణం(దశాంశం), nrows);
}

స్థిర పూర్ణాంకం roll_dma_f32_contiguous(నిర్మాణం htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    స్థిరరాశి uint32_t అడ్డువరుస_కొలత = ne0 * పరిమాణం(దశాంశం);

    ఒకవేళ (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_నెట్టు_అడ్డువరుసలు(q, dst, src0, 0, 0, ne1 * ne2 * ne3, అడ్డువరుస_కొలత, i0_src0);
        dma_వరుస_flush(q);
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (s1 == 0) {
        స్థిరరాశి uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        కొరకు (uint32_t i3 = 0; i3 < ne3; i3++) {
            స్థిరరాశి uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            స్థిరరాశి uint32_t dst_row0 = i3 * ne2 * ne1;
            స్థిరరాశి uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            స్థిరరాశి uint32_t n2_మొదటి = ne2 - i2_src0;

            roll_dma_నెట్టు_అడ్డువరుసలు(q, dst, src0, dst_row0, src_row0, n2_మొదటి * ne1,
                               అడ్డువరుస_కొలత, i0_src0);
            roll_dma_నెట్టు_అడ్డువరుసలు(q, dst, src0, dst_row0 + n2_మొదటి * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, అడ్డువరుస_కొలత, i0_src0);
        }

        dma_వరుస_flush(q);
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    స్థిరరాశి uint32_t n1_మొదటి = ne1 - i1_src0;

    కొరకు (uint32_t i3 = 0; i3 < ne3; i3++) {
        స్థిరరాశి uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        కొరకు (uint32_t i2 = 0; i2 < ne2; i2++) {
            స్థిరరాశి uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            స్థిరరాశి uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            స్థిరరాశి uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_నెట్టు_అడ్డువరుసలు(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_మొదటి, అడ్డువరుస_కొలత, i0_src0);
            roll_dma_నెట్టు_అడ్డువరుసలు(q, dst, src0, dst_row0 + n1_మొదటి, src_row0,
                               i1_src0, అడ్డువరుస_కొలత, i0_src0);
        }
    }

    dma_వరుస_flush(q);
    తిరిగి HTP_స్థితి_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
స్థిర పూర్ణాంకం roll_dma_f32_strided(నిర్మాణం htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    స్థిరరాశి uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    స్థిరరాశి uint32_t n1_మొదటి = ne1 - i1_src0;

    కొరకు (uint32_t i3 = 0; i3 < ne3; i3++) {
        స్థిరరాశి uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        కొరకు (uint32_t i2 = 0; i2 < ne2; i2++) {
            స్థిరరాశి uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            స్థిరరాశి dma_addr_t dst_row0 = dst->దత్తాంశం  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            స్థిరరాశి dma_addr_t src_row0 = src0->దత్తాంశం + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_నెట్టు_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_మొదటి, i0_src0, n0);
            roll_dma_నెట్టు_range(q, dst_row0 + (size_t) n1_మొదటి * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_వరుస_flush(q);
    తిరిగి HTP_స్థితి_OK;
}

స్థిర శూన్య roll_thread_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_roll_context * rctx = (నిర్మాణం htp_roll_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    స్థిరరాశి uint32_t అడ్డువరుస_ప్రారంభం = rctx->అడ్డువరుస_ప్రారంభం + rctx->nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t అడ్డువరుస_చివర   = MIN(అడ్డువరుస_ప్రారంభం + rctx->nrows_చొప్పున_thread, rctx->అడ్డువరుస_ప్రారంభం + rctx->nrows);
    ఒకవేళ (అడ్డువరుస_ప్రారంభం >= అడ్డువరుస_చివర) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, అడ్డువరుస_ప్రారంభం);

    కొరకు (uint32_t అడ్డువరుస = అడ్డువరుస_ప్రారంభం; అడ్డువరుస < అడ్డువరుస_చివర; అడ్డువరుస++) {
        స్థిరరాశి uint32_t i3  = fastdiv(అడ్డువరుస, &rctx->div_ne2_ne1);
        స్థిరరాశి uint32_t rem = అడ్డువరుస - i3 * ne2 * ne1;
        స్థిరరాశి uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        స్థిరరాశి uint32_t i1  = rem - i2 * ne1;

        స్థిరరాశి uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        స్థిరరాశి uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        స్థిరరాశి uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        స్థిరరాశి uint8_t * src_అడ్డువరుస = (స్థిరరాశి uint8_t *) (uintptr_t) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_అడ్డువరుస = (uint8_t *) (uintptr_t) dst->దత్తాంశం + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_అడ్డువరుస + i0_src0 * పరిమాణం(దశాంశం), n0 * పరిమాణం(దశాంశం), ne0 * పరిమాణం(దశాంశం), 1);
        hvx_copy_uu(dst_అడ్డువరుస, src_అడ్డువరుస + i0_src0 * పరిమాణం(దశాంశం), n0, పరిమాణం(దశాంశం));

        ఒకవేళ (i0_src0 != 0) {
            hex_l2fetch(src_అడ్డువరుస, i0_src0 * పరిమాణం(దశాంశం), ne0 * పరిమాణం(దశాంశం), 1);
            hvx_copy_uu(dst_అడ్డువరుస + n0 * పరిమాణం(దశాంశం), src_అడ్డువరుస, i0_src0, పరిమాణం(దశాంశం));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, అడ్డువరుస_ప్రారంభం);

    FARF(ఎగువ_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         అడ్డువరుస_ప్రారంభం, అడ్డువరుస_చివర, s0, s1, s2, s3);
}

పూర్ణాంకం execute_op_roll_f32(నిర్మాణం htp_ops_context * octx) {
    htp_roll_preamble;

    ఒకవేళ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (src0->nb[0] != పరిమాణం(దశాంశం) || dst->nb[0] != పరిమాణం(దశాంశం)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = ne1 * ne2 * ne3;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత = ne0 * పరిమాణం(దశాంశం);

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = మొత్తం_అడ్డువరుసలు;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, పరిమాణం(దశాంశం), (uint32_t) dst_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_అడ్డువరుసలు, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (octx->ctx->mdev.లెక్క <= 1) {
        ఒకవేళ (htp_tensor_is_contiguous(src0, పరిమాణం(దశాంశం)) && htp_tensor_is_contiguous(dst, పరిమాణం(దశాంశం))) {
            తిరిగి roll_dma_f32_contiguous(octx);
        }
        తిరిగి roll_dma_f32_strided(octx);
    }

    ఒకవేళ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;
    నిర్మాణం htp_roll_context rctx = {
        .octx             = octx,
        .అడ్డువరుస_ప్రారంభం        = అడ్డువరుస_ప్రారంభం,
        .nrows            = nrows,
        .nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div),
        .div_ne1          = init_fastdiv_విలువలు(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_విలువలు(dst->ne[2] * dst->ne[1]),
    };

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, roll_thread_f32, &rctx, n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_roll(నిర్మాణం htp_ops_context * octx) {
    ఎంపిక (octx->src[0]->type) {
        స్థితి HTP_TYPE_F32:
            తిరిగి execute_op_roll_f32(octx);

        అప్రమేయ:
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
}
