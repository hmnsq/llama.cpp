#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#అన్తర్భావ <string.h>
#అన్తర్భావ <math.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hvx-utils.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"

#నిర్వచించు కూడిక_అడ్డువరుసలు_preamble                         \
    స్థిరరాశి నిర్మాణం htp_tensor *src0 = octx->src[0]; \
    స్థిరరాశి నిర్మాణం htp_tensor *dst  = octx->dst;    \
                                                  \
    స్థిరరాశి uint32_t ne00 = src0->ne[0];            \
    స్థిరరాశి uint32_t ne01 = src0->ne[1];            \
    స్థిరరాశి uint32_t ne02 = src0->ne[2];            \
    స్థిరరాశి uint32_t ne03 = src0->ne[3];            \
                                                  \
    స్థిరరాశి uint32_t nb00 = src0->nb[0];            \
    స్థిరరాశి uint32_t nb01 = src0->nb[1];            \
    స్థిరరాశి uint32_t nb02 = src0->nb[2];            \
    స్థిరరాశి uint32_t nb03 = src0->nb[3];            \
                                                  \
    స్థిరరాశి uint32_t  ne0 = dst->ne[0];             \
    స్థిరరాశి uint32_t  ne1 = dst->ne[1];             \
    స్థిరరాశి uint32_t  ne2 = dst->ne[2];             \
    స్థిరరాశి uint32_t  ne3 = dst->ne[3];             \
                                                  \
    స్థిరరాశి uint32_t  nb0 = dst->nb[0];             \
    స్థిరరాశి uint32_t  nb1 = dst->nb[1];             \
    స్థిరరాశి uint32_t  nb2 = dst->nb[2];             \
    స్థిరరాశి uint32_t  nb3 = dst->nb[3];             \

నిర్మాణం కూడిక_అడ్డువరుసలు_context {
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి uint8_t * src_దత్తాంశం;
    uint8_t       * dst_దత్తాంశం;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        అడ్డువరుసలు_చొప్పున_thread;
    uint32_t        మొత్తం_అడ్డువరుసలు;
    bool            opt_మార్గం;
};

స్థిర శూన్య కూడిక_అడ్డువరుసలు_thread_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య *దత్తాంశం) {
    స్థిరరాశి నిర్మాణం కూడిక_అడ్డువరుసలు_context * smctx = (స్థిరరాశి నిర్మాణం కూడిక_అడ్డువరుసలు_context *) దత్తాంశం;

    స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_thread = smctx->అడ్డువరుసలు_చొప్పున_thread;
    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు      = smctx->మొత్తం_అడ్డువరుసలు;

    స్థిరరాశి uint32_t ప్రారంభం_అడ్డువరుస = అడ్డువరుసలు_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_అడ్డువరుస   = MIN(ప్రారంభం_అడ్డువరుస + అడ్డువరుసలు_చొప్పున_thread, మొత్తం_అడ్డువరుసలు);

    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= చివర_అడ్డువరుస) {
        తిరిగి;
    }

    స్థిరరాశి size_t   src_stride = smctx->src_stride;
    స్థిరరాశి size_t   dst_stride = smctx->dst_stride;
    స్థిరరాశి uint32_t ne00       = smctx->ne00;
    స్థిరరాశి bool     opt_మార్గం   = smctx->opt_మార్గం;

    స్థిరరాశి దశాంశం * పరిమిత src_th = (స్థిరరాశి దశాంశం *) (smctx->src_దత్తాంశం + (ప్రారంభం_అడ్డువరుస * src_stride));
    దశాంశం       * పరిమిత dst_th = (దశాంశం *)       (smctx->dst_దత్తాంశం + (ప్రారంభం_అడ్డువరుస * dst_stride));

    // Calculate actual number of rows for this thread
    స్థిరరాశి uint32_t n_అడ్డువరుసలు = చివర_అడ్డువరుస - ప్రారంభం_అడ్డువరుస;

    నిర్మాణం htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ప్రారంభం_అడ్డువరుస);

    కొరకు (uint32_t ir = 0; ir < n_అడ్డువరుసలు; ir++) {
        స్థిరరాశి దశాంశం * పరిమిత src_local = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src_th + ir * src_stride);
        దశాంశం       * పరిమిత dst_local = (దశాంశం *)       ((uint8_t *)       dst_th + ir * dst_stride);

        ఒకవేళ (ir + 1 < n_అడ్డువరుసలు) {
            hex_l2fetch((స్థిరరాశి uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        ఒకవేళ (opt_మార్గం) {
            *dst_local = hvx_reduce_కూడిక_f32_a((స్థిరరాశి uint8_t *) src_local, ne00);
        } లేకపోతే {
            *dst_local = hvx_reduce_కూడిక_f32((స్థిరరాశి uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ప్రారంభం_అడ్డువరుస);
}

పూర్ణాంకం op_కూడిక_అడ్డువరుసలు(నిర్మాణం htp_ops_context * octx) {
    కూడిక_అడ్డువరుసలు_preamble;

    ఒకవేళ (octx->src[0]->type != HTP_TYPE_F32) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t src0_nrows      = ne01 * ne02 * ne03;
    స్థిరరాశి size_t dst_దత్తాంశం_అడ్డువరుస_కొలత = dst->ne[0] * పరిమాణం(దశాంశం);

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = src0_nrows;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, పరిమాణం(దశాంశం), (uint32_t) dst_దత్తాంశం_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(src0_nrows, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;
    స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div);

    bool opt_మార్గం = false;
    ఒకవేళ ((0 == hex_is_aligned((శూన్య *) src0->దత్తాంశం, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_మార్గం = true;
    }

    నిర్మాణం కూడిక_అడ్డువరుసలు_context smctx = {
        .octx            = octx,
        .src_దత్తాంశం        = (స్థిరరాశి uint8_t *) src0->దత్తాంశం + అడ్డువరుస_ప్రారంభం * nb01,
        .dst_దత్తాంశం        = (uint8_t *) dst->దత్తాంశం + అడ్డువరుస_ప్రారంభం * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .అడ్డువరుసలు_చొప్పున_thread = అడ్డువరుసలు_చొప్పున_thread,
        .మొత్తం_అడ్డువరుసలు      = nrows,
        .opt_మార్గం        = opt_మార్గం,
    };

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, కూడిక_అడ్డువరుసలు_thread_f32, &smctx, n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}

నిర్మాణం కూడిక_context {
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి దశాంశం            * src_దత్తాంశం;
    దశాంశం                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 మొత్తం_elems;
    uint32_t                 elems_చొప్పున_thread;
};

స్థిర శూన్య కూడిక_thread_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం కూడిక_context * sctx = (నిర్మాణం కూడిక_context *) దత్తాంశం;
    స్థిరరాశి uint32_t ప్రారంభం = sctx->elems_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర   = MIN(ప్రారంభం + sctx->elems_చొప్పున_thread, sctx->మొత్తం_elems);

    ఒకవేళ (ప్రారంభం >= చివర) {
        sctx->partial_sums[ith] = 0.0f;
        తిరిగి;
    }

    స్థిరరాశి uint32_t n = చివర - ప్రారంభం;
    స్థిరరాశి దశాంశం * src = sctx->src_దత్తాంశం + ప్రారంభం;

    నిర్మాణం htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ప్రారంభం);

    hex_l2fetch_ఖండం((స్థిరరాశి శూన్య *) src, n * పరిమాణం(దశాంశం));

    sctx->partial_sums[ith] = hvx_reduce_కూడిక_f32((స్థిరరాశి uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ప్రారంభం);
}

పూర్ణాంకం op_కూడిక(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఒకవేళ (src0->type != HTP_TYPE_F32) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1 && octx->ctx->mdev.idx > 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t మొత్తం_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ఒకవేళ (మొత్తం_elems == 0) {
        ((దశాంశం *) dst->దత్తాంశం)[0] = 0.0f;
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = (మొత్తం_elems >= 1024) ? MIN(octx->n_దారాలు, HTP_MAX_NTHREADS) : 1;
    స్థిరరాశి uint32_t raw_chunk = (మొత్తం_elems + n_దారాలు - 1) / n_దారాలు;
    స్థిరరాశి uint32_t elems_చొప్పున_thread = hex_విడత_పైకి(raw_chunk, 32);

    నిర్మాణం కూడిక_context sctx = {
        .octx             = octx,
        .src_దత్తాంశం         = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం,
        .మొత్తం_elems      = మొత్తం_elems,
        .elems_చొప్పున_thread = elems_చొప్పున_thread,
    };

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, కూడిక_thread_f32, &sctx, n_దారాలు);

    దశాంశం కూడిక = 0.0f;
    కొరకు (uint32_t i = 0; i < n_దారాలు; i++) {
        కూడిక += sctx.partial_sums[i];
    }
    ((దశాంశం *) dst->దత్తాంశం)[0] = కూడిక;

    తిరిగి HTP_స్థితి_OK;
}

స్థిర అంతఃవరుస శూన్య argmax_ముక్క_f32(
    స్థిరరాశి దశాంశం * పరిమిత src,
    uint32_t n,
    uint32_t స్థానభ్రంశం,
    దశాంశం * బహిర్గతం_val,
    int32_t * బహిర్గతం_idx
) {
    hvx_argmax_f32(src, n, స్థానభ్రంశం, బహిర్గతం_val, బహిర్గతం_idx);
}

నిర్మాణం argmax_context {
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి దశాంశం            * src_దత్తాంశం;
    int32_t                * dst_దత్తాంశం;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 అడ్డువరుస_ప్రారంభం;
    uint32_t                 nrows;
    uint32_t                 అడ్డువరుసలు_చొప్పున_thread;
    uint32_t                 elems_చొప్పున_thread;

    దశాంశం                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

స్థిర శూన్య argmax_thread_single_అడ్డువరుస(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం argmax_context * actx = (నిర్మాణం argmax_context *) దత్తాంశం;
    స్థిరరాశి uint32_t ప్రారంభం = actx->elems_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర   = MIN(ప్రారంభం + actx->elems_చొప్పున_thread, actx->ne00);

    ఒకవేళ (ప్రారంభం >= చివర) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        తిరిగి;
    }

    స్థిరరాశి uint32_t n = చివర - ప్రారంభం;
    స్థిరరాశి దశాంశం * src = actx->src_దత్తాంశం + ప్రారంభం;

    నిర్మాణం htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ప్రారంభం);

    hex_l2fetch_ఖండం((స్థిరరాశి శూన్య *) src, n * పరిమాణం(దశాంశం));

    దశాంశం max_val;
    int32_t max_idx;
    argmax_ముక్క_f32(src, n, ప్రారంభం, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ప్రారంభం);
}

స్థిర శూన్య argmax_thread_multi_అడ్డువరుస(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం argmax_context * actx = (నిర్మాణం argmax_context *) దత్తాంశం;
    స్థిరరాశి uint32_t r0 = actx->అడ్డువరుస_ప్రారంభం + actx->అడ్డువరుసలు_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t r1 = MIN(r0 + actx->అడ్డువరుసలు_చొప్పున_thread, actx->అడ్డువరుస_ప్రారంభం + actx->nrows);

    ఒకవేళ (r0 >= r1) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    కొరకు (uint32_t r = r0; r < r1; r++) {
        స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) actx->src_దత్తాంశం + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_దత్తాంశం + r * actx->dst_stride);

        hex_l2fetch_ఖండం((స్థిరరాశి శూన్య *) src_అడ్డువరుస, actx->ne00 * పరిమాణం(దశాంశం));

        దశాంశం max_val;
        int32_t max_idx;
        argmax_ముక్క_f32(src_అడ్డువరుస, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

పూర్ణాంకం op_argmax(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఒకవేళ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t ne00 = src0->ne[0];
    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    ఒకవేళ (ne00 == 0 || src0_nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (src0_nrows == 1) {
        ఒకవేళ (octx->ctx->mdev.లెక్క > 1 && octx->ctx->mdev.idx > 0) {
            తిరిగి HTP_స్థితి_OK;
        }

        ఒకవేళ (ne00 == 1) {
            ((int32_t *) dst->దత్తాంశం)[0] = 0;
            తిరిగి HTP_స్థితి_OK;
        }

        స్థిరరాశి uint32_t n_దారాలు = (ne00 >= 1024) ? MIN(octx->n_దారాలు, HTP_MAX_NTHREADS) : 1;
        స్థిరరాశి uint32_t raw_chunk = (ne00 + n_దారాలు - 1) / n_దారాలు;
        స్థిరరాశి uint32_t elems_చొప్పున_thread = hex_విడత_పైకి(raw_chunk, 32);

        నిర్మాణం argmax_context actx = {
            .octx             = octx,
            .src_దత్తాంశం         = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం,
            .dst_దత్తాంశం         = (int32_t *) dst->దత్తాంశం,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * పరిమాణం(దశాంశం)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) పరిమాణం(int32_t),
            .అడ్డువరుస_ప్రారంభం        = 0,
            .nrows            = 1,
            .అడ్డువరుసలు_చొప్పున_thread  = 1,
            .elems_చొప్పున_thread = elems_చొప్పున_thread,
        };

        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, argmax_thread_single_అడ్డువరుస, &actx, n_దారాలు);

        దశాంశం ఉత్తమ_val = actx.partial_max[0];
        int32_t ఉత్తమ_idx = actx.partial_idx[0];
        కొరకు (uint32_t i = 1; i < n_దారాలు; i++) {
            ఒకవేళ (actx.partial_max[i] > ఉత్తమ_val) {
                ఉత్తమ_val = actx.partial_max[i];
                ఉత్తమ_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->దత్తాంశం)[0] = ఉత్తమ_idx;
        తిరిగి HTP_స్థితి_OK;
    }

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = src0_nrows;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి bool can_విభజించు = htp_tensor_mdev_దత్తాంశం_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, పరిమాణం(int32_t));
        స్థిరరాశి uint32_t elems_చొప్పున_chunk = can_విభజించు ? 32 : 0;
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            src0_nrows, elems_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = MIN(octx->n_దారాలు, nrows);
    స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_thread = (nrows + n_దారాలు - 1) / n_దారాలు;

    నిర్మాణం argmax_context actx = {
        .octx             = octx,
        .src_దత్తాంశం         = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం,
        .dst_దత్తాంశం         = (int32_t *) dst->దత్తాంశం,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * పరిమాణం(దశాంశం)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) పరిమాణం(int32_t),
        .అడ్డువరుస_ప్రారంభం        = అడ్డువరుస_ప్రారంభం,
        .nrows            = nrows,
        .అడ్డువరుసలు_చొప్పున_thread  = అడ్డువరుసలు_చొప్పున_thread,
        .elems_చొప్పున_thread = 0,
    };

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, argmax_thread_multi_అడ్డువరుస, &actx, n_దారాలు);
    తిరిగి HTP_స్థితి_OK;
}
