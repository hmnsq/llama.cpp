#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"

ਬਣਤਰ htp_repeat_context {
    ਬਣਤਰ htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_ਪ੍ਰਤੀ_thread;
    uint32_t ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ;  // ne1 * ne2 * ne3
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;

    size_t   type_ਮਾਪ;
};

ਸਥਿਰ ਖਾਲੀ repeat_ਕੰਮ_ਪ੍ਰਤੀ_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਅਚਲ ਬਣਤਰ htp_repeat_context * rctx = (ਅਚਲ ਬਣਤਰ htp_repeat_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = rctx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;

    ਅਚਲ uint32_t ne00 = src->ne[0];
    ਅਚਲ uint32_t ne01 = src->ne[1];
    ਅਚਲ uint32_t ne02 = src->ne[2];
    ਅਚਲ uint32_t ne03 = src->ne[3];

    ਅਚਲ uint32_t nb00 = src->nb[0];
    ਅਚਲ uint32_t nb01 = src->nb[1];
    ਅਚਲ uint32_t nb02 = src->nb[2];
    ਅਚਲ uint32_t nb03 = src->nb[3];

    ਅਚਲ uint32_t ne0 = dst->ne[0];
    ਅਚਲ uint32_t ne1 = dst->ne[1];
    ਅਚਲ uint32_t ne2 = dst->ne[2];
    ਅਚਲ uint32_t ne3 = dst->ne[3];

    ਅਚਲ uint32_t nb0 = dst->nb[0];
    ਅਚਲ uint32_t nb1 = dst->nb[1];
    ਅਚਲ uint32_t nb2 = dst->nb[2];
    ਅਚਲ uint32_t nb3 = dst->nb[3];

    ਅਚਲ uint32_t nr0 = rctx->nr0;
    ਅਚਲ uint32_t nr1 = rctx->nr1;
    ਅਚਲ uint32_t nr2 = rctx->nr2;
    ਅਚਲ uint32_t nr3 = rctx->nr3;

    ਅਚਲ size_t ਪੰਕਤੀ_ਬਾਈਟਾਂ = ne00 * rctx->type_ਮਾਪ;

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = rctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + rctx->nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ   = MIN(ਪੰਕਤੀ_ਸ਼ੁਰੂ + rctx->nrows_ਪ੍ਰਤੀ_thread, rctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + rctx->ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ਪੰਕਤੀ_ਸ਼ੁਰੂ);

    ਲਈ (uint32_t dst_ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; dst_ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; dst_ਪੰਕਤੀ++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ਅਚਲ uint32_t i1 = dst_ਪੰਕਤੀ % ne1;
        ਅਚਲ uint32_t i2 = (dst_ਪੰਕਤੀ / ne1) % ne2;
        ਅਚਲ uint32_t i3 = dst_ਪੰਕਤੀ / (ne1 * ne2);

        // Map to source indices (tiling)
        ਅਚਲ uint32_t k1 = i1 % ne01;
        ਅਚਲ uint32_t k2 = i2 % ne02;
        ਅਚਲ uint32_t k3 = i3 % ne03;

        ਅਚਲ uint8_t * src_ਪੰਕਤੀ = (ਅਚਲ uint8_t *) src->ਅੰਕੜੇ + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_ਅਧਾਰ      = (uint8_t *) dst->ਅੰਕੜੇ + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        ਲਈ (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_ਅਧਾਰ + i0 * ne00 * nb0;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ptr, src_ਪੰਕਤੀ, ਪੰਕਤੀ_ਬਾਈਟਾਂ);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ਪੰਕਤੀ_ਸ਼ੁਰੂ);

    FARF(ਉੱਚਾ_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ਪੰਕਤੀ_ਸ਼ੁਰੂ, ਪੰਕਤੀ_ਸਿਰਾ);
}

ਪੂਰਨ op_repeat(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    ਜੇ (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    size_t type_ਮਾਪ;
    ਚੋਣ (src0->type) {
        ਹਾਲਤ HTP_TYPE_F32: type_ਮਾਪ = 4; ਰੋਕੋ;
        ਹਾਲਤ HTP_TYPE_F16: type_ਮਾਪ = 2; ਰੋਕੋ;
        ਮੂਲ:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ = dst->ne[0] * type_ਮਾਪ;

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, type_ਮਾਪ, (uint32_t) dst_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    ਬਣਤਰ htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
        .ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ   = nrows,
        .ਪੰਕਤੀ_ਸ਼ੁਰੂ        = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .type_ਮਾਪ        = type_ਮਾਪ,
    };

    FARF(ਉੱਚਾ_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, repeat_ਕੰਮ_ਪ੍ਰਤੀ_thread, &rctx, n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
