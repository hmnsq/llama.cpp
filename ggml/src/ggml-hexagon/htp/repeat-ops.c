#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>

#ضمن "hvx-utils.h"

#تعریف_کن GGML_COMMON_DECL_C
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

    uint32_t nrows_در_هر_thread;
    uint32_t مجموع_dst_ردیفها;  // ne1 * ne2 * ne3
    uint32_t ردیف_شروع;

    size_t   type_اندازه;
};

ساكن فراغ repeat_کار_در_هر_thread(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    ثابت بنية htp_repeat_context * rctx = (ثابت بنية htp_repeat_context *) داده;
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

    ثابت size_t ردیف_بایتها = ne00 * rctx->type_اندازه;

    ثابت uint32_t ردیف_شروع = rctx->ردیف_شروع + rctx->nrows_در_هر_thread * ith;
    ثابت uint32_t ردیف_انتها   = MIN(ردیف_شروع + rctx->nrows_در_هر_thread, rctx->ردیف_شروع + rctx->مجموع_dst_ردیفها);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ردیف_شروع);

    لكل (uint32_t dst_ردیف = ردیف_شروع; dst_ردیف < ردیف_انتها; dst_ردیف++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ثابت uint32_t i1 = dst_ردیف % ne1;
        ثابت uint32_t i2 = (dst_ردیف / ne1) % ne2;
        ثابت uint32_t i3 = dst_ردیف / (ne1 * ne2);

        // Map to source indices (tiling)
        ثابت uint32_t k1 = i1 % ne01;
        ثابت uint32_t k2 = i2 % ne02;
        ثابت uint32_t k3 = i3 % ne03;

        ثابت uint8_t * src_ردیف = (ثابت uint8_t *) src->داده + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_پایه      = (uint8_t *) dst->داده + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        لكل (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_پایه + i0 * ne00 * nb0;
            کپی_حافظه(dst_ptr, src_ردیف, ردیف_بایتها);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ردیف_شروع);

    FARF(بالا_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ردیف_شروع, ردیف_انتها);
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
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    size_t type_اندازه;
    بدل (src0->type) {
        حالة HTP_TYPE_F32: type_اندازه = 4; اكسر;
        حالة HTP_TYPE_F16: type_اندازه = 2; اكسر;
        افتراضي:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t مجموع_dst_ردیفها  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت size_t dst_ردیف_اندازه = dst->ne[0] * type_اندازه;

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = مجموع_dst_ردیفها;

    إذا (octx->ctx->mdev.شمار > 1) {
        uint32_t ردیفها_در_هر_chunk = 0;
        htp_tensor_mdev_ردیفها_در_هر_chunk(dst, type_اندازه, (uint32_t) dst_ردیف_اندازه, &ردیفها_در_هر_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_dst_ردیفها, ردیفها_در_هر_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;

    بنية htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_در_هر_thread = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div),
        .مجموع_dst_ردیفها   = nrows,
        .ردیف_شروع        = ردیف_شروع,
        .type_اندازه        = type_اندازه,
    };

    FARF(بالا_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_صف_پیاپی(octx->ctx->work_صف, repeat_کار_در_هر_thread, &rctx, n_نخها);

    أرجع HTP_وضعیت_OK;
}
