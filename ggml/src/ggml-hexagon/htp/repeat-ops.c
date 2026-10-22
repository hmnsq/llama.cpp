#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#అన్తర్భావ <string.h>

#అన్తర్భావ "hvx-utils.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"

నిర్మాణం htp_repeat_context {
    నిర్మాణం htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_చొప్పున_thread;
    uint32_t మొత్తం_dst_అడ్డువరుసలు;  // ne1 * ne2 * ne3
    uint32_t అడ్డువరుస_ప్రారంభం;

    size_t   type_కొలత;
};

స్థిర శూన్య repeat_పని_చొప్పున_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    స్థిరరాశి నిర్మాణం htp_repeat_context * rctx = (స్థిరరాశి నిర్మాణం htp_repeat_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = rctx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;

    స్థిరరాశి uint32_t ne00 = src->ne[0];
    స్థిరరాశి uint32_t ne01 = src->ne[1];
    స్థిరరాశి uint32_t ne02 = src->ne[2];
    స్థిరరాశి uint32_t ne03 = src->ne[3];

    స్థిరరాశి uint32_t nb00 = src->nb[0];
    స్థిరరాశి uint32_t nb01 = src->nb[1];
    స్థిరరాశి uint32_t nb02 = src->nb[2];
    స్థిరరాశి uint32_t nb03 = src->nb[3];

    స్థిరరాశి uint32_t ne0 = dst->ne[0];
    స్థిరరాశి uint32_t ne1 = dst->ne[1];
    స్థిరరాశి uint32_t ne2 = dst->ne[2];
    స్థిరరాశి uint32_t ne3 = dst->ne[3];

    స్థిరరాశి uint32_t nb0 = dst->nb[0];
    స్థిరరాశి uint32_t nb1 = dst->nb[1];
    స్థిరరాశి uint32_t nb2 = dst->nb[2];
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

    స్థిరరాశి uint32_t nr0 = rctx->nr0;
    స్థిరరాశి uint32_t nr1 = rctx->nr1;
    స్థిరరాశి uint32_t nr2 = rctx->nr2;
    స్థిరరాశి uint32_t nr3 = rctx->nr3;

    స్థిరరాశి size_t అడ్డువరుస_బైట్లు = ne00 * rctx->type_కొలత;

    స్థిరరాశి uint32_t అడ్డువరుస_ప్రారంభం = rctx->అడ్డువరుస_ప్రారంభం + rctx->nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t అడ్డువరుస_చివర   = MIN(అడ్డువరుస_ప్రారంభం + rctx->nrows_చొప్పున_thread, rctx->అడ్డువరుస_ప్రారంభం + rctx->మొత్తం_dst_అడ్డువరుసలు);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, అడ్డువరుస_ప్రారంభం);

    కొరకు (uint32_t dst_అడ్డువరుస = అడ్డువరుస_ప్రారంభం; dst_అడ్డువరుస < అడ్డువరుస_చివర; dst_అడ్డువరుస++) {
        // Decompose flat dst row index into (i1, i2, i3)
        స్థిరరాశి uint32_t i1 = dst_అడ్డువరుస % ne1;
        స్థిరరాశి uint32_t i2 = (dst_అడ్డువరుస / ne1) % ne2;
        స్థిరరాశి uint32_t i3 = dst_అడ్డువరుస / (ne1 * ne2);

        // Map to source indices (tiling)
        స్థిరరాశి uint32_t k1 = i1 % ne01;
        స్థిరరాశి uint32_t k2 = i2 % ne02;
        స్థిరరాశి uint32_t k3 = i3 % ne03;

        స్థిరరాశి uint8_t * src_అడ్డువరుస = (స్థిరరాశి uint8_t *) src->దత్తాంశం + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_పునాది      = (uint8_t *) dst->దత్తాంశం + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        కొరకు (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_పునాది + i0 * ne00 * nb0;
            జ్ఞాపకం_నకలు(dst_ptr, src_అడ్డువరుస, అడ్డువరుస_బైట్లు);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, అడ్డువరుస_ప్రారంభం);

    FARF(ఎగువ_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         అడ్డువరుస_ప్రారంభం, అడ్డువరుస_చివర);
}

పూర్ణాంకం op_repeat(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    ఒకవేళ (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    size_t type_కొలత;
    ఎంపిక (src0->type) {
        స్థితి HTP_TYPE_F32: type_కొలత = 4; ఆపు;
        స్థితి HTP_TYPE_F16: type_కొలత = 2; ఆపు;
        అప్రమేయ:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t మొత్తం_dst_అడ్డువరుసలు  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత = dst->ne[0] * type_కొలత;

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

    నిర్మాణం htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div),
        .మొత్తం_dst_అడ్డువరుసలు   = nrows,
        .అడ్డువరుస_ప్రారంభం        = అడ్డువరుస_ప్రారంభం,
        .type_కొలత        = type_కొలత,
    };

    FARF(ఎగువ_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, repeat_పని_చొప్పున_thread, &rctx, n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}
