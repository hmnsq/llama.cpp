#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <string.h>

#شامل "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"

ساخت htp_repeat_context {
    ساخت htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_فی_thread;
    uint32_t کل_dst_صفیں;  // ne1 * ne2 * ne3
    uint32_t صف_آغاز;

    size_t   type_سائز;
};

ساکن خالی repeat_کام_فی_thread(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ثابت ساخت htp_repeat_context * rctx = (ثابت ساخت htp_repeat_context *) ڈیٹا;
    ساخت htp_ops_context * octx = rctx->octx;
    ثابت ساخت htp_tensor * src = octx->src[0];
    ثابت ساخت htp_tensor * dst = octx->dst;

    ثابت uint32_t ne00 = src->ne[0];
    ثابت uint32_t ne01 = src->ne[1];
    ثابت uint32_t ne02 = src->ne[2];
    ثابت uint32_t ne03 = src->ne[3];

    ثابت uint32_t nb00 = src->nb[0];
    ثابت uint32_t nb01 = src->nb[1];
    ثابت uint32_t nb02 = src->nb[2];
    ثابت uint32_t nb03 = src->nb[3];

    ثابت uint32_t ne0 = dst->ne[0];
    ثابت uint32_t ne1 = dst->ne[1];
    ثابت uint32_t ne2 = dst->ne[2];
    ثابت uint32_t ne3 = dst->ne[3];

    ثابت uint32_t nb0 = dst->nb[0];
    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    ثابت uint32_t nr0 = rctx->nr0;
    ثابت uint32_t nr1 = rctx->nr1;
    ثابت uint32_t nr2 = rctx->nr2;
    ثابت uint32_t nr3 = rctx->nr3;

    ثابت size_t صف_بائٹیں = ne00 * rctx->type_سائز;

    ثابت uint32_t صف_آغاز = rctx->صف_آغاز + rctx->nrows_فی_thread * ith;
    ثابت uint32_t صف_انتہا   = MIN(صف_آغاز + rctx->nrows_فی_thread, rctx->صف_آغاز + rctx->کل_dst_صفیں);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, صف_آغاز);

    برائے (uint32_t dst_صف = صف_آغاز; dst_صف < صف_انتہا; dst_صف++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ثابت uint32_t i1 = dst_صف % ne1;
        ثابت uint32_t i2 = (dst_صف / ne1) % ne2;
        ثابت uint32_t i3 = dst_صف / (ne1 * ne2);

        // Map to source indices (tiling)
        ثابت uint32_t k1 = i1 % ne01;
        ثابت uint32_t k2 = i2 % ne02;
        ثابت uint32_t k3 = i3 % ne03;

        ثابت uint8_t * src_صف = (ثابت uint8_t *) src->ڈیٹا + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_بنیاد      = (uint8_t *) dst->ڈیٹا + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        برائے (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_بنیاد + i0 * ne00 * nb0;
            حافظہ_نقل(dst_ptr, src_صف, صف_بائٹیں);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_آغاز);

    FARF(بلند_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_آغاز, صف_انتہا);
}

صحیح op_repeat(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    اگر (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        واپس HTP_حالت_INVAL_PARAMS;
    }

    size_t type_سائز;
    انتخاب (src0->type) {
        حالت HTP_TYPE_F32: type_سائز = 4; توڑو;
        حالت HTP_TYPE_F16: type_سائز = 2; توڑو;
        طےشدہ:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t کل_dst_صفیں  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت size_t dst_صف_سائز = dst->ne[0] * type_سائز;

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = کل_dst_صفیں;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, type_سائز, (uint32_t) dst_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_dst_صفیں, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    ساخت htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .کل_dst_صفیں   = nrows,
        .صف_آغاز        = صف_آغاز,
        .type_سائز        = type_سائز,
    };

    FARF(بلند_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_قطار_تسلسل(octx->ctx->work_قطار, repeat_کام_فی_thread, &rctx, n_دھاگے);

    واپس HTP_حالت_OK;
}
