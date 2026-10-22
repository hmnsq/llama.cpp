#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>

#ضمن "hvx-utils.h"

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

بنية htp_repeat_context {
    بنية htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_في_thread;
    uint32_t ڪل_dst_صفون;  // ne1 * ne2 * ne3
    uint32_t صف_شروعات;

    size_t   type_سائيز;
};

ساكن فراغ repeat_ڪم_في_thread(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    ثابت بنية htp_repeat_context * rctx = (ثابت بنية htp_repeat_context *) معلومات;
    بنية htp_ops_context * octx = rctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;

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

    ثابت size_t صف_بائيٽون = ne00 * rctx->type_سائيز;

    ثابت uint32_t صف_شروعات = rctx->صف_شروعات + rctx->nrows_في_thread * ith;
    ثابت uint32_t صف_پڇاڙي   = MIN(صف_شروعات + rctx->nrows_في_thread, rctx->صف_شروعات + rctx->ڪل_dst_صفون);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروعات);

    لكل (uint32_t dst_صف = صف_شروعات; dst_صف < صف_پڇاڙي; dst_صف++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ثابت uint32_t i1 = dst_صف % ne1;
        ثابت uint32_t i2 = (dst_صف / ne1) % ne2;
        ثابت uint32_t i3 = dst_صف / (ne1 * ne2);

        // Map to source indices (tiling)
        ثابت uint32_t k1 = i1 % ne01;
        ثابت uint32_t k2 = i2 % ne02;
        ثابت uint32_t k3 = i3 % ne03;

        ثابت uint8_t * src_صف = (ثابت uint8_t *) src->معلومات + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_بنياد      = (uint8_t *) dst->معلومات + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        لكل (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_بنياد + i0 * ne00 * nb0;
            يادگيري_نقل(dst_ptr, src_صف, صف_بائيٽون);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروعات);

    FARF(مٿاهون_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_شروعات, صف_پڇاڙي);
}

صحيح op_repeat(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    إذا (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    size_t type_سائيز;
    بدل (src0->type) {
        حالة HTP_TYPE_F32: type_سائيز = 4; اكسر;
        حالة HTP_TYPE_F16: type_سائيز = 2; اكسر;
        افتراضي:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t ڪل_dst_صفون  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت size_t dst_صف_سائيز = dst->ne[0] * type_سائيز;

    uint32_t صف_شروعات = 0;
    uint32_t nrows     = ڪل_dst_صفون;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        uint32_t صفون_في_chunk = 0;
        htp_tensor_mdev_صفون_في_chunk(dst, type_سائيز, (uint32_t) dst_صف_سائيز, &صفون_في_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(ڪل_dst_صفون, صفون_في_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

    بنية htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_في_thread = fastdiv(nrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div),
        .ڪل_dst_صفون   = nrows,
        .صف_شروعات        = صف_شروعات,
        .type_سائيز        = type_سائيز,
    };

    FARF(مٿاهون_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_قطار_لاڳيتو(octx->ctx->work_قطار, repeat_ڪم_في_thread, &rctx, n_ڌاڳا);

    أرجع HTP_حالت_OK;
}
