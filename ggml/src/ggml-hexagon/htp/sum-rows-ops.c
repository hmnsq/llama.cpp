#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <math.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

#ವ್ಯಾಖ್ಯಾನಿಸು ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_preamble                         \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor *src0 = octx->src[0]; \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor *dst  = octx->dst;    \
                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];            \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1];            \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];            \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];            \
                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t nb00 = src0->nb[0];            \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1];            \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];            \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];            \
                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t  ne0 = dst->ne[0];             \
    ಸ್ಥಿರಾಂಕ uint32_t  ne1 = dst->ne[1];             \
    ಸ್ಥಿರಾಂಕ uint32_t  ne2 = dst->ne[2];             \
    ಸ್ಥಿರಾಂಕ uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t  nb0 = dst->nb[0];             \
    ಸ್ಥಿರಾಂಕ uint32_t  nb1 = dst->nb[1];             \
    ಸ್ಥಿರಾಂಕ uint32_t  nb2 = dst->nb[2];             \
    ಸ್ಥಿರಾಂಕ uint32_t  nb3 = dst->nb[3];             \

ರಚನೆ ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_context {
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ uint8_t * src_ದತ್ತಾಂಶ;
    uint8_t       * dst_ದತ್ತಾಂಶ;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
    uint32_t        ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    bool            opt_ಮಾರ್ಗ;
};

ಸ್ಥಿರ ಶೂನ್ಯ ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_thread_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ *ದತ್ತಾಂಶ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_context * smctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_context *) ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = smctx->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು      = smctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;

    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(ಆರಂಭ_ಅಡ್ಡಸಾಲು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);

    ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t   src_stride = smctx->src_stride;
    ಸ್ಥಿರಾಂಕ size_t   dst_stride = smctx->dst_stride;
    ಸ್ಥಿರಾಂಕ uint32_t ne00       = smctx->ne00;
    ಸ್ಥಿರಾಂಕ bool     opt_ಮಾರ್ಗ   = smctx->opt_ಮಾರ್ಗ;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src_th = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (smctx->src_ದತ್ತಾಂಶ + (ಆರಂಭ_ಅಡ್ಡಸಾಲು * src_stride));
    ದಶಮಾಂಶ       * ಸೀಮಿತ dst_th = (ದಶಮಾಂಶ *)       (smctx->dst_ದತ್ತಾಂಶ + (ಆರಂಭ_ಅಡ್ಡಸಾಲು * dst_stride));

    // Calculate actual number of rows for this thread
    ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು = ಕೊನೆ_ಅಡ್ಡಸಾಲು - ಆರಂಭ_ಅಡ್ಡಸಾಲು;

    ರಚನೆ htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಆರಂಭ_ಅಡ್ಡಸಾಲು);

    ಗಾಗಿ (uint32_t ir = 0; ir < n_ಅಡ್ಡಸಾಲುಗಳು; ir++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src_th + ir * src_stride);
        ದಶಮಾಂಶ       * ಸೀಮಿತ dst_local = (ದಶಮಾಂಶ *)       ((uint8_t *)       dst_th + ir * dst_stride);

        ಆದರೆ (ir + 1 < n_ಅಡ್ಡಸಾಲುಗಳು) {
            hex_l2fetch((ಸ್ಥಿರಾಂಕ uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        ಆದರೆ (opt_ಮಾರ್ಗ) {
            *dst_local = hvx_reduce_ಮೊತ್ತ_f32_a((ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne00);
        } ಇಲ್ಲದಿದ್ದರೆ {
            *dst_local = hvx_reduce_ಮೊತ್ತ_f32((ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಆರಂಭ_ಅಡ್ಡಸಾಲು);
}

ಪೂರ್ಣಾಂಕ op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ರಚನೆ htp_ops_context * octx) {
    ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_preamble;

    ಆದರೆ (octx->src[0]->type != HTP_TYPE_F32) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ಸ್ಥಿರಾಂಕ size_t dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = src0_nrows;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, ಗಾತ್ರ(ದಶಮಾಂಶ), (uint32_t) dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(src0_nrows, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);

    bool opt_ಮಾರ್ಗ = false;
    ಆದರೆ ((0 == hex_is_aligned((ಶೂನ್ಯ *) src0->ದತ್ತಾಂಶ, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_ಮಾರ್ಗ = true;
    }

    ರಚನೆ ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_context smctx = {
        .octx            = octx,
        .src_ದತ್ತಾಂಶ        = (ಸ್ಥಿರಾಂಕ uint8_t *) src0->ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು_ಆರಂಭ * nb01,
        .dst_ದತ್ತಾಂಶ        = (uint8_t *) dst->ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು_ಆರಂಭ * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread,
        .ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು      = nrows,
        .opt_ಮಾರ್ಗ        = opt_ಮಾರ್ಗ,
    };

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_thread_f32, &smctx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ರಚನೆ ಮೊತ್ತ_context {
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ            * src_ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 ಒಟ್ಟು_elems;
    uint32_t                 elems_ಪ್ರತಿ_thread;
};

ಸ್ಥಿರ ಶೂನ್ಯ ಮೊತ್ತ_thread_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ ಮೊತ್ತ_context * sctx = (ರಚನೆ ಮೊತ್ತ_context *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ = sctx->elems_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ   = MIN(ಆರಂಭ + sctx->elems_ಪ್ರತಿ_thread, sctx->ಒಟ್ಟು_elems);

    ಆದರೆ (ಆರಂಭ >= ಕೊನೆ) {
        sctx->partial_sums[ith] = 0.0f;
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n = ಕೊನೆ - ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src = sctx->src_ದತ್ತಾಂಶ + ಆರಂಭ;

    ರಚನೆ htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಆರಂಭ);

    hex_l2fetch_ಖಂಡ((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) src, n * ಗಾತ್ರ(ದಶಮಾಂಶ));

    sctx->partial_sums[ith] = hvx_reduce_ಮೊತ್ತ_f32((ಸ್ಥಿರಾಂಕ uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಆರಂಭ);
}

ಪೂರ್ಣಾಂಕ op_ಮೊತ್ತ(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆದರೆ (src0->type != HTP_TYPE_F32) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1 && octx->ctx->mdev.idx > 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ಆದರೆ (ಒಟ್ಟು_elems == 0) {
        ((ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ)[0] = 0.0f;
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = (ಒಟ್ಟು_elems >= 1024) ? MIN(octx->n_ಎಳೆಗಳು, HTP_MAX_NTHREADS) : 1;
    ಸ್ಥಿರಾಂಕ uint32_t raw_chunk = (ಒಟ್ಟು_elems + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t elems_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(raw_chunk, 32);

    ರಚನೆ ಮೊತ್ತ_context sctx = {
        .octx             = octx,
        .src_ದತ್ತಾಂಶ         = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ,
        .ಒಟ್ಟು_elems      = ಒಟ್ಟು_elems,
        .elems_ಪ್ರತಿ_thread = elems_ಪ್ರತಿ_thread,
    };

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, ಮೊತ್ತ_thread_f32, &sctx, n_ಎಳೆಗಳು);

    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
    ಗಾಗಿ (uint32_t i = 0; i < n_ಎಳೆಗಳು; i++) {
        ಮೊತ್ತ += sctx.partial_sums[i];
    }
    ((ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ)[0] = ಮೊತ್ತ;

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ argmax_ತುಂಡು_f32(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src,
    uint32_t n,
    uint32_t ಸ್ಥಾನಾಂತರ,
    ದಶಮಾಂಶ * ಹೊರಹೋಗು_val,
    int32_t * ಹೊರಹೋಗು_idx
) {
    hvx_argmax_f32(src, n, ಸ್ಥಾನಾಂತರ, ಹೊರಹೋಗು_val, ಹೊರಹೋಗು_idx);
}

ರಚನೆ argmax_context {
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ            * src_ದತ್ತಾಂಶ;
    int32_t                * dst_ದತ್ತಾಂಶ;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t                 nrows;
    uint32_t                 ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
    uint32_t                 elems_ಪ್ರತಿ_thread;

    ದಶಮಾಂಶ                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ಸ್ಥಿರ ಶೂನ್ಯ argmax_thread_single_ಅಡ್ಡಸಾಲು(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ argmax_context * actx = (ರಚನೆ argmax_context *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ = actx->elems_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ   = MIN(ಆರಂಭ + actx->elems_ಪ್ರತಿ_thread, actx->ne00);

    ಆದರೆ (ಆರಂಭ >= ಕೊನೆ) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n = ಕೊನೆ - ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src = actx->src_ದತ್ತಾಂಶ + ಆರಂಭ;

    ರಚನೆ htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಆರಂಭ);

    hex_l2fetch_ಖಂಡ((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) src, n * ಗಾತ್ರ(ದಶಮಾಂಶ));

    ದಶಮಾಂಶ max_val;
    int32_t max_idx;
    argmax_ತುಂಡು_f32(src, n, ಆರಂಭ, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಆರಂಭ);
}

ಸ್ಥಿರ ಶೂನ್ಯ argmax_thread_multi_ಅಡ್ಡಸಾಲು(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ argmax_context * actx = (ರಚನೆ argmax_context *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ uint32_t r0 = actx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + actx->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t r1 = MIN(r0 + actx->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, actx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + actx->nrows);

    ಆದರೆ (r0 >= r1) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    ಗಾಗಿ (uint32_t r = r0; r < r1; r++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) actx->src_ದತ್ತಾಂಶ + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_ದತ್ತಾಂಶ + r * actx->dst_stride);

        hex_l2fetch_ಖಂಡ((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) src_ಅಡ್ಡಸಾಲು, actx->ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ));

        ದಶಮಾಂಶ max_val;
        int32_t max_idx;
        argmax_ತುಂಡು_f32(src_ಅಡ್ಡಸಾಲು, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

ಪೂರ್ಣಾಂಕ op_argmax(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆದರೆ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    ಆದರೆ (ne00 == 0 || src0_nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (src0_nrows == 1) {
        ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1 && octx->ctx->mdev.idx > 0) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
        }

        ಆದರೆ (ne00 == 1) {
            ((int32_t *) dst->ದತ್ತಾಂಶ)[0] = 0;
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
        }

        ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = (ne00 >= 1024) ? MIN(octx->n_ಎಳೆಗಳು, HTP_MAX_NTHREADS) : 1;
        ಸ್ಥಿರಾಂಕ uint32_t raw_chunk = (ne00 + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ uint32_t elems_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(raw_chunk, 32);

        ರಚನೆ argmax_context actx = {
            .octx             = octx,
            .src_ದತ್ತಾಂಶ         = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ,
            .dst_ದತ್ತಾಂಶ         = (int32_t *) dst->ದತ್ತಾಂಶ,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) ಗಾತ್ರ(int32_t),
            .ಅಡ್ಡಸಾಲು_ಆರಂಭ        = 0,
            .nrows            = 1,
            .ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread  = 1,
            .elems_ಪ್ರತಿ_thread = elems_ಪ್ರತಿ_thread,
        };

        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, argmax_thread_single_ಅಡ್ಡಸಾಲು, &actx, n_ಎಳೆಗಳು);

        ದಶಮಾಂಶ ಉತ್ತಮ_val = actx.partial_max[0];
        int32_t ಉತ್ತಮ_idx = actx.partial_idx[0];
        ಗಾಗಿ (uint32_t i = 1; i < n_ಎಳೆಗಳು; i++) {
            ಆದರೆ (actx.partial_max[i] > ಉತ್ತಮ_val) {
                ಉತ್ತಮ_val = actx.partial_max[i];
                ಉತ್ತಮ_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->ದತ್ತಾಂಶ)[0] = ಉತ್ತಮ_idx;
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = src0_nrows;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, ಗಾತ್ರ(int32_t));
        ಸ್ಥಿರಾಂಕ uint32_t elems_ಪ್ರತಿ_chunk = can_ವಿಭಜಿಸು ? 32 : 0;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            src0_nrows, elems_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = MIN(octx->n_ಎಳೆಗಳು, nrows);
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (nrows + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು;

    ರಚನೆ argmax_context actx = {
        .octx             = octx,
        .src_ದತ್ತಾಂಶ         = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ,
        .dst_ದತ್ತಾಂಶ         = (int32_t *) dst->ದತ್ತಾಂಶ,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) ಗಾತ್ರ(int32_t),
        .ಅಡ್ಡಸಾಲು_ಆರಂಭ        = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .nrows            = nrows,
        .ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread  = ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread,
        .elems_ಪ್ರತಿ_thread = 0,
    };

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, argmax_thread_multi_ಅಡ್ಡಸಾಲು, &actx, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
