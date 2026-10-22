#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"

#అన్తర్భావ "hvx-copy.h"
#అన్తర్భావ "hvx-utils.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"

// ggml op_params layout for FILL:
//   op_params[0] (as float) - the scalar fill value

#నిర్వచించు fill_preamble                          \
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst; \
                                               \
    స్థిరరాశి uint32_t ne0 = dst->ne[0];           \
    స్థిరరాశి uint32_t ne1 = dst->ne[1];           \
    స్థిరరాశి uint32_t ne2 = dst->ne[2];           \
    స్థిరరాశి uint32_t ne3 = dst->ne[3];           \
                                               \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];           \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];           \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];           \
                                               \
    స్థిరరాశి uint32_t nr = ne1 * ne2 * ne3;

నిర్మాణం htp_fill_context {
    నిర్మాణం htp_ops_context * octx;
    uint32_t nrows_చొప్పున_thread;
    uint32_t మొత్తం_అడ్డువరుసలు;  // ne1 * ne2 * ne3
    uint32_t అడ్డువరుస_ప్రారంభం;
    bool     opt_మార్గం;
    HVX_సదిశ splat_vec;
    uint32_t   elem_కొలత;
};

స్థిర శూన్య fill_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    స్థిరరాశి నిర్మాణం htp_fill_context * fctx = (స్థిరరాశి నిర్మాణం htp_fill_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context        * octx = fctx->octx;
    fill_preamble;

    // Parallelise over the flat row index spanning ne1*ne2*ne3
    స్థిరరాశి uint32_t ir0 = fctx->అడ్డువరుస_ప్రారంభం + fctx->nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + fctx->nrows_చొప్పున_thread, fctx->అడ్డువరుస_ప్రారంభం + fctx->మొత్తం_అడ్డువరుసలు);

    ఒకవేళ (ir0 >= ir1) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    ఒకవేళ (fctx->opt_మార్గం) {
        // Opt path: tensor is fully contiguous, treat as flat array
        స్థిరరాశి uint32_t elem_ప్రారంభం = ir0 * ne0;
        స్థిరరాశి uint32_t elem_చివర = ir1 * ne0;
        uint8_t * dst_ptr = (uint8_t *) dst->దత్తాంశం + elem_ప్రారంభం * fctx->elem_కొలత;
        hvx_splat_u(dst_ptr, fctx->splat_vec, elem_చివర - elem_ప్రారంభం, fctx->elem_కొలత);
    } లేకపోతే {
        // Non-contiguous path: must respect strides
        కొరకు (uint32_t ir = ir0; ir < ir1; ++ir) {
            స్థిరరాశి uint32_t i1 = ir % ne1;
            స్థిరరాశి uint32_t i2 = (ir / ne1) % ne2;
            స్థిరరాశి uint32_t i3 = ir / (ne1 * ne2);
            uint8_t * dst_ptr = (uint8_t *) dst->దత్తాంశం + i1*nb1 + i2*nb2 + i3*nb3;
            hvx_splat_u(dst_ptr, fctx->splat_vec, ne0, fctx->elem_కొలత);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
    FARF(ఎగువ_2, "fill %u/%u: rows %u:%u\n", ith, nth, ir0, ir1);
}

పూర్ణాంకం op_fill(నిర్మాణం htp_ops_context * octx) {
    fill_preamble;

    ఒకవేళ (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = nr;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి uint32_t అడ్డువరుస_కొలత = nb1;
        స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_chunk = (అడ్డువరుస_కొలత > 0) ? (HEX_L2_పంక్తి_కొలత / hex_గరిష్ఠ_సామాన్య_భాజకం_u32(అడ్డువరుస_కొలత, HEX_L2_పంక్తి_కొలత)) : 1;
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(nr, htp_tensor_mdev_దత్తాంశం_aligned(dst) ? అడ్డువరుసలు_చొప్పున_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    // nr = ne1*ne2*ne3 (flat row count across all outer dims); parallelise over it.
    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    // Optimize if fully contiguous: skip stride arithmetic, treat as flat array
    స్థిరరాశి bool opt_మార్గం = (nb2 == nb1 * ne1) && (nb3 == nb2 * ne2);

    FARF(ఎగువ_2, "fill: (%ux%ux%ux%u) type=%u opt=%d\n",
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type, (పూర్ణాంకం) opt_మార్గం);

    దశాంశం val_f32 = 0.f;
    జ్ఞాపకం_నకలు(&val_f32, &octx->op_params[0], పరిమాణం(దశాంశం));

    నిర్మాణం htp_fill_context fctx = {
        .octx             = octx,
        .nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div),
        .మొత్తం_అడ్డువరుసలు       = nrows,
        .అడ్డువరుస_ప్రారంభం        = అడ్డువరుస_ప్రారంభం,
        .opt_మార్గం         = opt_మార్గం,
    };

    ఎంపిక (dst->type) {
    స్థితి HTP_TYPE_F32:
        fctx.splat_vec = hvx_vec_splat_f32(val_f32);
        fctx.elem_కొలత = పరిమాణం(దశాంశం);
        ఆపు;
    స్థితి HTP_TYPE_F16:
        fctx.splat_vec = hvx_vec_splat_f16((_Float16) val_f32);
        fctx.elem_కొలత = పరిమాణం(_Float16);
        ఆపు;
    అప్రమేయ:
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, fill_thread, &fctx, n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}
