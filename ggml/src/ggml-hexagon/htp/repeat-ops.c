#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

ರಚನೆ htp_repeat_context {
    ರಚನೆ htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_ಪ್ರತಿ_thread;
    uint32_t ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು;  // ne1 * ne2 * ne3
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;

    size_t   type_ಗಾತ್ರ;
};

ಸ್ಥಿರ ಶೂನ್ಯ repeat_ಕೆಲಸ_ಪ್ರತಿ_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_repeat_context * rctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_repeat_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = rctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t nb00 = src->nb[0];
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src->nb[3];

    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t nb0 = dst->nb[0];
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

    ಸ್ಥಿರಾಂಕ uint32_t nr0 = rctx->nr0;
    ಸ್ಥಿರಾಂಕ uint32_t nr1 = rctx->nr1;
    ಸ್ಥಿರಾಂಕ uint32_t nr2 = rctx->nr2;
    ಸ್ಥಿರಾಂಕ uint32_t nr3 = rctx->nr3;

    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = ne00 * rctx->type_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = rctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + rctx->nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ   = MIN(ಅಡ್ಡಸಾಲು_ಆರಂಭ + rctx->nrows_ಪ್ರತಿ_thread, rctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + rctx->ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ಅಡ್ಡಸಾಲು_ಆರಂಭ);

    ಗಾಗಿ (uint32_t dst_ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; dst_ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; dst_ಅಡ್ಡಸಾಲು++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ಸ್ಥಿರಾಂಕ uint32_t i1 = dst_ಅಡ್ಡಸಾಲು % ne1;
        ಸ್ಥಿರಾಂಕ uint32_t i2 = (dst_ಅಡ್ಡಸಾಲು / ne1) % ne2;
        ಸ್ಥಿರಾಂಕ uint32_t i3 = dst_ಅಡ್ಡಸಾಲು / (ne1 * ne2);

        // Map to source indices (tiling)
        ಸ್ಥಿರಾಂಕ uint32_t k1 = i1 % ne01;
        ಸ್ಥಿರಾಂಕ uint32_t k2 = i2 % ne02;
        ಸ್ಥಿರಾಂಕ uint32_t k3 = i3 % ne03;

        ಸ್ಥಿರಾಂಕ uint8_t * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) src->ದತ್ತಾಂಶ + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_ತಳ      = (uint8_t *) dst->ದತ್ತಾಂಶ + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        ಗಾಗಿ (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_ತಳ + i0 * ne00 * nb0;
            ಸ್ಮರಣೆ_ನಕಲು(dst_ptr, src_ಅಡ್ಡಸಾಲು, ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ಅಡ್ಡಸಾಲು_ಆರಂಭ);

    FARF(ಮೇಲಿನ_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ಅಡ್ಡಸಾಲು_ಆರಂಭ, ಅಡ್ಡಸಾಲು_ಕೊನೆ);
}

ಪೂರ್ಣಾಂಕ op_repeat(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    ಆದರೆ (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    size_t type_ಗಾತ್ರ;
    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32: type_ಗಾತ್ರ = 4; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_TYPE_F16: type_ಗಾತ್ರ = 2; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->ne[0] * type_ಗಾತ್ರ;

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, type_ಗಾತ್ರ, (uint32_t) dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    ರಚನೆ htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
        .ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು   = nrows,
        .ಅಡ್ಡಸಾಲು_ಆರಂಭ        = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .type_ಗಾತ್ರ        = type_ಗಾತ್ರ,
    };

    FARF(ಮೇಲಿನ_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, repeat_ಕೆಲಸ_ಪ್ರತಿ_thread, &rctx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
