#అన్తర్భావ "concat-ops.h"
#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "dma-copy.h"
#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "hexagon_protos.h"
#అన్తర్భావ "hexagon_types.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-fence.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "htp-vtcm.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "hvx_hexagon_protos.h"
#అన్తర్భావ "gather-rows.h"

#అన్తర్భావ <string.h>

నిర్మాణం htp_concat_context {
    నిర్మాణం htp_ops_context * octx;
    uint8_t * spad0_పునాది;
    uint8_t * spad1_పునాది;
    uint32_t  spad0_కొలత_చొప్పున_thread;
    uint32_t  spad1_కొలత_చొప్పున_thread;
    uint32_t  అడ్డువరుస_ప్రారంభం;
    uint32_t  nrows;
    uint32_t  nrows_చొప్పున_thread;
    uint32_t  nplanes;
    నిర్మాణం fastdiv_విలువలు div_ne2;
};

స్థిర అంతఃవరుస dma_addr_t concat_plane_addr(స్థిరరాశి నిర్మాణం htp_tensor * t, uint32_t p, స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne2, uint32_t ne2) {
    స్థిరరాశి uint32_t i3 = fastdiv(p, div_ne2);
    స్థిరరాశి uint32_t i2 = p - i3 * ne2;
    తిరిగి t->దత్తాంశం + i2 * t->nb[2] + i3 * t->nb[3];
}

స్థిర శూన్య concat_2d_f32_transposed(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_concat_context * cctx = (నిర్మాణం htp_concat_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = cctx->octx;

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    స్థిరరాశి uint32_t src0_ne0 = src0->ne[0];
    స్థిరరాశి uint32_t src1_ne0 = src1->ne[0];

    స్థిరరాశి uint32_t అడ్డువరుస_చివర = cctx->అడ్డువరుస_ప్రారంభం + cctx->nrows;
    స్థిరరాశి uint32_t ప్రారంభం_i = cctx->అడ్డువరుస_ప్రారంభం + ith * cctx->nrows_చొప్పున_thread;
    స్థిరరాశి uint32_t చివర_i   = (ప్రారంభం_i + cctx->nrows_చొప్పున_thread < అడ్డువరుస_చివర) ? (ప్రారంభం_i + cctx->nrows_చొప్పున_thread) : అడ్డువరుస_చివర;
    ఒకవేళ (ప్రారంభం_i >= చివర_i) తిరిగి;

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_పునాది = cctx->spad0_పునాది + ith * cctx->spad0_కొలత_చొప్పున_thread;
    uint8_t * spad1_పునాది = cctx->spad1_పునాది + ith * cctx->spad1_కొలత_చొప్పున_thread;

    స్థిరరాశి uint32_t ఖండం_i = 32;
    స్థిరరాశి uint32_t spad1_stride = ఖండం_i * పరిమాణం(దశాంశం);

    స్థిరరాశి HVX_సదిశ offsets = hvx_vec_gather_offsets_w(spad1_stride);
    స్థిరరాశి uint32_t src1_ne0_padded = hex_విడత_పైకి(src1_ne0, 32);
    స్థిరరాశి uint32_t src0_అడ్డువరుస_బైట్లు  = src0_ne0 * పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t src0_అడ్డువరుస_padded = hex_విడత_పైకి(src0_అడ్డువరుస_బైట్లు, VLEN);
    స్థిరరాశి uint32_t src0_pre        = src0_అడ్డువరుస_padded - src0_అడ్డువరుస_బైట్లు;
    స్థిరరాశి uint32_t spad0_అడ్డువరుస_బైట్లు = src0_అడ్డువరుస_padded + src1_ne0_padded * పరిమాణం(దశాంశం);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne2 = &cctx->div_ne2;
    స్థిరరాశి uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = ప్రారంభం_i;

    స్థిరరాశి dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(spad1_పునాది, src1_addr), spad1_stride, src1->nb[0], MIN(చివర_i - i, ఖండం_i) * పరిమాణం(దశాంశం), src1_ne0);

    స్థిరరాశి dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(spad0_పునాది + src0_pre, src0_addr), spad0_అడ్డువరుస_బైట్లు, src0->nb[1], src0_అడ్డువరుస_బైట్లు, MIN(చివర_i - i, ఖండం_i));

    dma_వరుస_తీయి(dma_q); // src1

    అయితే (p < cctx->nplanes) {
        స్థిరరాశి uint32_t current_ఖండం_i = MIN(చివర_i - i, ఖండం_i);

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        కొరకు (uint32_t j = 0; j < src1_ne0; j += 32) {
            స్థిరరాశి uint8_t * src_ptr = spad1_పునాది + j * spad1_stride;
            uint8_t * dst_ptr = spad0_పునాది + src0_అడ్డువరుస_padded + j * పరిమాణం(దశాంశం);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_అడ్డువరుస_బైట్లు, src_ptr, spad1_stride, offsets, current_ఖండం_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_పునాది + src0_అడ్డువరుస_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + ఖండం_i;
        ఒకవేళ (ni >= చివర_i) {
            ni = ప్రారంభం_i;
            np++;
        }
        స్థిరరాశి bool has_తదుపరి = np < cctx->nplanes;
        స్థిరరాశి uint32_t తదుపరి_ఖండం_i = MIN(చివర_i - ni, ఖండం_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ఒకవేళ (has_తదుపరి) {
            స్థిరరాశి dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(spad1_పునాది, nsrc1_addr), spad1_stride, src1->nb[0], తదుపరి_ఖండం_i * పరిమాణం(దశాంశం), src1_ne0);
        }

        dma_వరుస_తీయి(dma_q); // src0

        స్థిరరాశి dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_addr, spad0_పునాది + src0_pre), dst->nb[1], spad0_అడ్డువరుస_బైట్లు, (src0_ne0 + src1_ne0) * పరిమాణం(దశాంశం), current_ఖండం_i);

        ఒకవేళ (has_తదుపరి) {
            dma_వరుస_తీయి(dma_q); // next src1
        }
        dma_వరుస_తీయి(dma_q); // dst

        // spad0 is free after the dst write
        ఒకవేళ (has_తదుపరి) {
            స్థిరరాశి dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(spad0_పునాది + src0_pre, nsrc0_addr), spad0_అడ్డువరుస_బైట్లు, src0->nb[1], src0_అడ్డువరుస_బైట్లు, తదుపరి_ఖండం_i);
        }

        p = np;
        i = ni;
    }
    dma_వరుస_flush(dma_q);
}

స్థిర శూన్య concat_2d_f16_transposed(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_concat_context * cctx = (నిర్మాణం htp_concat_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = cctx->octx;

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    స్థిరరాశి uint32_t src0_ne0 = src0->ne[0];
    స్థిరరాశి uint32_t src1_ne0 = src1->ne[0];

    స్థిరరాశి uint32_t అడ్డువరుస_చివర = cctx->అడ్డువరుస_ప్రారంభం + cctx->nrows;
    స్థిరరాశి uint32_t ప్రారంభం_i = cctx->అడ్డువరుస_ప్రారంభం + ith * cctx->nrows_చొప్పున_thread;
    స్థిరరాశి uint32_t చివర_i   = (ప్రారంభం_i + cctx->nrows_చొప్పున_thread < అడ్డువరుస_చివర) ? (ప్రారంభం_i + cctx->nrows_చొప్పున_thread) : అడ్డువరుస_చివర;
    ఒకవేళ (ప్రారంభం_i >= చివర_i) తిరిగి;

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_పునాది = cctx->spad0_పునాది + ith * cctx->spad0_కొలత_చొప్పున_thread;
    uint8_t * spad1_పునాది = cctx->spad1_పునాది + ith * cctx->spad1_కొలత_చొప్పున_thread;

    స్థిరరాశి uint32_t ఖండం_i = 64;
    స్థిరరాశి uint32_t spad1_stride = ఖండం_i * పరిమాణం(__fp16);

    స్థిరరాశి HVX_సదిశ offsets = hvx_vec_gather_offsets_h(spad1_stride);
    స్థిరరాశి uint32_t src1_ne0_padded = hex_విడత_పైకి(src1_ne0, 64);
    స్థిరరాశి uint32_t src0_అడ్డువరుస_బైట్లు  = src0_ne0 * పరిమాణం(__fp16);
    స్థిరరాశి uint32_t src0_అడ్డువరుస_padded = hex_విడత_పైకి(src0_అడ్డువరుస_బైట్లు, VLEN);
    స్థిరరాశి uint32_t src0_pre        = src0_అడ్డువరుస_padded - src0_అడ్డువరుస_బైట్లు;
    స్థిరరాశి uint32_t spad0_అడ్డువరుస_బైట్లు = src0_అడ్డువరుస_padded + src1_ne0_padded * పరిమాణం(__fp16);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne2 = &cctx->div_ne2;
    స్థిరరాశి uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = ప్రారంభం_i;

    స్థిరరాశి dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(spad1_పునాది, src1_addr), spad1_stride, src1->nb[0], MIN(చివర_i - i, ఖండం_i) * పరిమాణం(__fp16), src1_ne0);

    స్థిరరాశి dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(spad0_పునాది + src0_pre, src0_addr), spad0_అడ్డువరుస_బైట్లు, src0->nb[1], src0_అడ్డువరుస_బైట్లు, MIN(చివర_i - i, ఖండం_i));

    dma_వరుస_తీయి(dma_q); // src1

    అయితే (p < cctx->nplanes) {
        స్థిరరాశి uint32_t current_ఖండం_i = MIN(చివర_i - i, ఖండం_i);

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        కొరకు (uint32_t j = 0; j < src1_ne0; j += 64) {
            స్థిరరాశి uint8_t * src_ptr = spad1_పునాది + j * spad1_stride;
            uint8_t * dst_ptr = spad0_పునాది + src0_అడ్డువరుస_padded + j * పరిమాణం(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_అడ్డువరుస_బైట్లు, src_ptr, spad1_stride, offsets, current_ఖండం_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_పునాది + src0_అడ్డువరుస_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + ఖండం_i;
        ఒకవేళ (ni >= చివర_i) {
            ni = ప్రారంభం_i;
            np++;
        }
        స్థిరరాశి bool has_తదుపరి = np < cctx->nplanes;
        స్థిరరాశి uint32_t తదుపరి_ఖండం_i = MIN(చివర_i - ni, ఖండం_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ఒకవేళ (has_తదుపరి) {
            స్థిరరాశి dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(spad1_పునాది, nsrc1_addr), spad1_stride, src1->nb[0], తదుపరి_ఖండం_i * పరిమాణం(__fp16), src1_ne0);
        }

        dma_వరుస_తీయి(dma_q); // src0

        స్థిరరాశి dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_addr, spad0_పునాది + src0_pre), dst->nb[1], spad0_అడ్డువరుస_బైట్లు, (src0_ne0 + src1_ne0) * పరిమాణం(__fp16), current_ఖండం_i);

        ఒకవేళ (has_తదుపరి) {
            dma_వరుస_తీయి(dma_q); // next src1
        }
        dma_వరుస_తీయి(dma_q); // dst

        // spad0 is free after the dst write
        ఒకవేళ (has_తదుపరి) {
            స్థిరరాశి dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(spad0_పునాది + src0_pre, nsrc0_addr), spad0_అడ్డువరుస_బైట్లు, src0->nb[1], src0_అడ్డువరుస_బైట్లు, తదుపరి_ఖండం_i);
        }

        p = np;
        i = ni;
    }
    dma_వరుస_flush(dma_q);
}

స్థిర పూర్ణాంకం concat_regular(నిర్మాణం htp_ops_context * octx, పూర్ణాంకం dim, uint32_t type_కొలత) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    నిర్మాణం htp_tensor view0 = *dst;
    నిర్మాణం htp_tensor view1 = *dst;
    కొరకు (పూర్ణాంకం d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.దత్తాంశం += src0->ne[dim] * dst->nb[dim];

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = మొత్తం_అడ్డువరుసలు_0;
    uint32_t rstart1 = 0, nrows1 = మొత్తం_అడ్డువరుసలు_1;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range0 = htp_tensor_mdev_విభజన(
            మొత్తం_అడ్డువరుసలు_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        rstart0 = range0.ప్రారంభం;
        nrows0  = range0.లెక్క;

        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range1 = htp_tensor_mdev_విభజన(
            మొత్తం_అడ్డువరుసలు_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        rstart1 = range1.ప్రారంభం;
        nrows1  = range1.లెక్క;
    }

    dma_వరుస * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_కొలత, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_కొలత, rstart1, nrows1);
    dma_వరుస_flush(q);

    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం concat_transposed(నిర్మాణం htp_ops_context * octx, స్థిరరాశి నిర్మాణం htp_concat_kernel_params * kparams, uint32_t type_కొలత) {
    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = dst->ne[1];
    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = మొత్తం_అడ్డువరుసలు;
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_అడ్డువరుసలు, 1, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (kparams->vtcm_కొలత > octx->ctx->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;

    నిర్మాణం htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_పునాది            = vtcm_పునాది;
    cctx.spad1_పునాది            = vtcm_పునాది + n_దారాలు * kparams->u.transposed.spad0_కొలత_చొప్పున_thread;
    cctx.spad0_కొలత_చొప్పున_thread = kparams->u.transposed.spad0_కొలత_చొప్పున_thread;
    cctx.spad1_కొలత_చొప్పున_thread = kparams->u.transposed.spad1_కొలత_చొప్పున_thread;
    cctx.అడ్డువరుస_ప్రారంభం             = అడ్డువరుస_ప్రారంభం;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_విలువలు(dst->ne[2]);
    cctx.nrows_చొప్పున_thread      = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div);

    work_వరుస_func_t కార్మికుడు_func = (type_కొలత == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, కార్మికుడు_func, &cctx, n_దారాలు);
    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_concat(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_concat_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_concat_kernel_params *) octx->kernel_params;
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    పూర్ణాంకం స్థితి_2 = HTP_స్థితి_OK;
    స్థిరరాశి uint32_t type_కొలత = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    ఎంపిక (kparams->kernel_type) {
        స్థితి HTP_CONCAT_KERNEL_REGULAR:
            స్థితి_2 = concat_regular(octx, kparams->dim, type_కొలత);
            ఆపు;

        స్థితి HTP_CONCAT_KERNEL_TRANSPOSED:
            స్థితి_2 = concat_transposed(octx, kparams, type_కొలత);
            ఆపు;

        స్థితి HTP_CONCAT_KERNEL_GATHER_అడ్డువరుసలు:
            స్థితి_2 = htp_gather_అడ్డువరుసలు(octx, src0, src1, &kparams->u.gather, kparams->n_దారాలు);
            ఆపు;

        అప్రమేయ:
            స్థితి_2 = HTP_స్థితి_NO_SUPPORT;
            ఆపు;
    }

    htp_ops_context_సమితి_స్థితి(octx, స్థితి_2);

    తిరిగి octx->స్థితి_2;
}
