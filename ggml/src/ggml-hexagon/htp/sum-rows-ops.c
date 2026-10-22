#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਜੋੜ_ਪੰਕਤੀਆਂ_preamble                         \
    ਅਚਲ ਬਣਤਰ htp_tensor *src0 = octx->src[0]; \
    ਅਚਲ ਬਣਤਰ htp_tensor *dst  = octx->dst;    \
                                                  \
    ਅਚਲ uint32_t ne00 = src0->ne[0];            \
    ਅਚਲ uint32_t ne01 = src0->ne[1];            \
    ਅਚਲ uint32_t ne02 = src0->ne[2];            \
    ਅਚਲ uint32_t ne03 = src0->ne[3];            \
                                                  \
    ਅਚਲ uint32_t nb00 = src0->nb[0];            \
    ਅਚਲ uint32_t nb01 = src0->nb[1];            \
    ਅਚਲ uint32_t nb02 = src0->nb[2];            \
    ਅਚਲ uint32_t nb03 = src0->nb[3];            \
                                                  \
    ਅਚਲ uint32_t  ne0 = dst->ne[0];             \
    ਅਚਲ uint32_t  ne1 = dst->ne[1];             \
    ਅਚਲ uint32_t  ne2 = dst->ne[2];             \
    ਅਚਲ uint32_t  ne3 = dst->ne[3];             \
                                                  \
    ਅਚਲ uint32_t  nb0 = dst->nb[0];             \
    ਅਚਲ uint32_t  nb1 = dst->nb[1];             \
    ਅਚਲ uint32_t  nb2 = dst->nb[2];             \
    ਅਚਲ uint32_t  nb3 = dst->nb[3];             \

ਬਣਤਰ ਜੋੜ_ਪੰਕਤੀਆਂ_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ uint8_t * src_ਅੰਕੜੇ;
    uint8_t       * dst_ਅੰਕੜੇ;
    uint32_t        ne00;
    size_t          src_stride;
    size_t          dst_stride;
    uint32_t        ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    uint32_t        ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    bool            opt_ਰਸਤਾ;
};

ਸਥਿਰ ਖਾਲੀ ਜੋੜ_ਪੰਕਤੀਆਂ_thread_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ *ਅੰਕੜੇ) {
    ਅਚਲ ਬਣਤਰ ਜੋੜ_ਪੰਕਤੀਆਂ_context * smctx = (ਅਚਲ ਬਣਤਰ ਜੋੜ_ਪੰਕਤੀਆਂ_context *) ਅੰਕੜੇ;

    ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = smctx->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ      = smctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ;

    ਅਚਲ uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_ਪੰਕਤੀ   = MIN(ਸ਼ੁਰੂ_ਪੰਕਤੀ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, ਕੁੱਲ_ਪੰਕਤੀਆਂ);

    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਅਚਲ size_t   src_stride = smctx->src_stride;
    ਅਚਲ size_t   dst_stride = smctx->dst_stride;
    ਅਚਲ uint32_t ne00       = smctx->ne00;
    ਅਚਲ bool     opt_ਰਸਤਾ   = smctx->opt_ਰਸਤਾ;

    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src_th = (ਅਚਲ ਦਸ਼ਮਲਵ *) (smctx->src_ਅੰਕੜੇ + (ਸ਼ੁਰੂ_ਪੰਕਤੀ * src_stride));
    ਦਸ਼ਮਲਵ       * ਸੀਮਿਤ dst_th = (ਦਸ਼ਮਲਵ *)       (smctx->dst_ਅੰਕੜੇ + (ਸ਼ੁਰੂ_ਪੰਕਤੀ * dst_stride));

    // Calculate actual number of rows for this thread
    ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ = ਸਿਰਾ_ਪੰਕਤੀ - ਸ਼ੁਰੂ_ਪੰਕਤੀ;

    ਬਣਤਰ htp_thread_trace * tr = &smctx->octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸ਼ੁਰੂ_ਪੰਕਤੀ);

    ਲਈ (uint32_t ir = 0; ir < n_ਪੰਕਤੀਆਂ; ir++) {
        ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src_local = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) src_th + ir * src_stride);
        ਦਸ਼ਮਲਵ       * ਸੀਮਿਤ dst_local = (ਦਸ਼ਮਲਵ *)       ((uint8_t *)       dst_th + ir * dst_stride);

        ਜੇ (ir + 1 < n_ਪੰਕਤੀਆਂ) {
            hex_l2fetch((ਅਚਲ uint8_t *) src_local + src_stride, src_stride, src_stride, 1);
        }

        ਜੇ (opt_ਰਸਤਾ) {
            *dst_local = hvx_reduce_ਜੋੜ_f32_a((ਅਚਲ uint8_t *) src_local, ne00);
        } ਨਹੀਂਤਾਂ {
            *dst_local = hvx_reduce_ਜੋੜ_f32((ਅਚਲ uint8_t *) src_local, ne00);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸ਼ੁਰੂ_ਪੰਕਤੀ);
}

ਪੂਰਨ op_ਜੋੜ_ਪੰਕਤੀਆਂ(ਬਣਤਰ htp_ops_context * octx) {
    ਜੋੜ_ਪੰਕਤੀਆਂ_preamble;

    ਜੇ (octx->src[0]->type != HTP_TYPE_F32) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t src0_nrows      = ne01 * ne02 * ne03;
    ਅਚਲ size_t dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = dst->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = src0_nrows;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ), (uint32_t) dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(src0_nrows, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;
    ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);

    bool opt_ਰਸਤਾ = false;
    ਜੇ ((0 == hex_is_aligned((ਖਾਲੀ *) src0->ਅੰਕੜੇ, VLEN)) && !(nb01 & (VLEN - 1))) {
        opt_ਰਸਤਾ = true;
    }

    ਬਣਤਰ ਜੋੜ_ਪੰਕਤੀਆਂ_context smctx = {
        .octx            = octx,
        .src_ਅੰਕੜੇ        = (ਅਚਲ uint8_t *) src0->ਅੰਕੜੇ + ਪੰਕਤੀ_ਸ਼ੁਰੂ * nb01,
        .dst_ਅੰਕੜੇ        = (uint8_t *) dst->ਅੰਕੜੇ + ਪੰਕਤੀ_ਸ਼ੁਰੂ * nb1,
        .ne00            = ne00,
        .src_stride      = nb01,
        .dst_stride      = nb1,
        .ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread,
        .ਕੁੱਲ_ਪੰਕਤੀਆਂ      = nrows,
        .opt_ਰਸਤਾ        = opt_ਰਸਤਾ,
    };

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, ਜੋੜ_ਪੰਕਤੀਆਂ_thread_f32, &smctx, n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਬਣਤਰ ਜੋੜ_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਦਸ਼ਮਲਵ            * src_ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ                    partial_sums[HTP_MAX_NTHREADS];
    uint32_t                 ਕੁੱਲ_elems;
    uint32_t                 elems_ਪ੍ਰਤੀ_thread;
};

ਸਥਿਰ ਖਾਲੀ ਜੋੜ_thread_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ ਜੋੜ_context * sctx = (ਬਣਤਰ ਜੋੜ_context *) ਅੰਕੜੇ;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ = sctx->elems_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ   = MIN(ਸ਼ੁਰੂ + sctx->elems_ਪ੍ਰਤੀ_thread, sctx->ਕੁੱਲ_elems);

    ਜੇ (ਸ਼ੁਰੂ >= ਸਿਰਾ) {
        sctx->partial_sums[ith] = 0.0f;
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t n = ਸਿਰਾ - ਸ਼ੁਰੂ;
    ਅਚਲ ਦਸ਼ਮਲਵ * src = sctx->src_ਅੰਕੜੇ + ਸ਼ੁਰੂ;

    ਬਣਤਰ htp_thread_trace * tr = &sctx->octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸ਼ੁਰੂ);

    hex_l2fetch_ਖੰਡ((ਅਚਲ ਖਾਲੀ *) src, n * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    sctx->partial_sums[ith] = hvx_reduce_ਜੋੜ_f32((ਅਚਲ uint8_t *) src, n);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸ਼ੁਰੂ);
}

ਪੂਰਨ op_ਜੋੜ(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਜੇ (src0->type != HTP_TYPE_F32) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1 && octx->ctx->mdev.idx > 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ਜੇ (ਕੁੱਲ_elems == 0) {
        ((ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ)[0] = 0.0f;
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = (ਕੁੱਲ_elems >= 1024) ? MIN(octx->n_ਧਾਗੇ, HTP_MAX_NTHREADS) : 1;
    ਅਚਲ uint32_t raw_chunk = (ਕੁੱਲ_elems + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ;
    ਅਚਲ uint32_t elems_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(raw_chunk, 32);

    ਬਣਤਰ ਜੋੜ_context sctx = {
        .octx             = octx,
        .src_ਅੰਕੜੇ         = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ,
        .ਕੁੱਲ_elems      = ਕੁੱਲ_elems,
        .elems_ਪ੍ਰਤੀ_thread = elems_ਪ੍ਰਤੀ_thread,
    };

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, ਜੋੜ_thread_f32, &sctx, n_ਧਾਗੇ);

    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
    ਲਈ (uint32_t i = 0; i < n_ਧਾਗੇ; i++) {
        ਜੋੜ += sctx.partial_sums[i];
    }
    ((ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ)[0] = ਜੋੜ;

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ argmax_ਟੁਕੜਾ_f32(
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src,
    uint32_t n,
    uint32_t ਵਿੱਥ,
    ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ_val,
    int32_t * ਬਾਹਰਲਾ_idx
) {
    hvx_argmax_f32(src, n, ਵਿੱਥ, ਬਾਹਰਲਾ_val, ਬਾਹਰਲਾ_idx);
}

ਬਣਤਰ argmax_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਦਸ਼ਮਲਵ            * src_ਅੰਕੜੇ;
    int32_t                * dst_ਅੰਕੜੇ;
    uint32_t                 ne00;
    uint32_t                 src_stride;
    uint32_t                 dst_stride;
    uint32_t                 ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t                 nrows;
    uint32_t                 ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    uint32_t                 elems_ਪ੍ਰਤੀ_thread;

    ਦਸ਼ਮਲਵ                    partial_max[HTP_MAX_NTHREADS];
    int32_t                  partial_idx[HTP_MAX_NTHREADS];
};

ਸਥਿਰ ਖਾਲੀ argmax_thread_single_ਪੰਕਤੀ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ argmax_context * actx = (ਬਣਤਰ argmax_context *) ਅੰਕੜੇ;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ = actx->elems_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ   = MIN(ਸ਼ੁਰੂ + actx->elems_ਪ੍ਰਤੀ_thread, actx->ne00);

    ਜੇ (ਸ਼ੁਰੂ >= ਸਿਰਾ) {
        actx->partial_max[ith] = -INFINITY;
        actx->partial_idx[ith] = 0;
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t n = ਸਿਰਾ - ਸ਼ੁਰੂ;
    ਅਚਲ ਦਸ਼ਮਲਵ * src = actx->src_ਅੰਕੜੇ + ਸ਼ੁਰੂ;

    ਬਣਤਰ htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸ਼ੁਰੂ);

    hex_l2fetch_ਖੰਡ((ਅਚਲ ਖਾਲੀ *) src, n * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਦਸ਼ਮਲਵ max_val;
    int32_t max_idx;
    argmax_ਟੁਕੜਾ_f32(src, n, ਸ਼ੁਰੂ, &max_val, &max_idx);

    actx->partial_max[ith] = max_val;
    actx->partial_idx[ith] = max_idx;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸ਼ੁਰੂ);
}

ਸਥਿਰ ਖਾਲੀ argmax_thread_multi_ਪੰਕਤੀ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ argmax_context * actx = (ਬਣਤਰ argmax_context *) ਅੰਕੜੇ;
    ਅਚਲ uint32_t r0 = actx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + actx->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t r1 = MIN(r0 + actx->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, actx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + actx->nrows);

    ਜੇ (r0 >= r1) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &actx->octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);

    ਲਈ (uint32_t r = r0; r < r1; r++) {
        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) actx->src_ਅੰਕੜੇ + r * actx->src_stride);
        int32_t * dst_val     = (int32_t *) ((uint8_t *) actx->dst_ਅੰਕੜੇ + r * actx->dst_stride);

        hex_l2fetch_ਖੰਡ((ਅਚਲ ਖਾਲੀ *) src_ਪੰਕਤੀ, actx->ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        ਦਸ਼ਮਲਵ max_val;
        int32_t max_idx;
        argmax_ਟੁਕੜਾ_f32(src_ਪੰਕਤੀ, actx->ne00, 0, &max_val, &max_idx);

        *dst_val = max_idx;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r0);
}

ਪੂਰਨ op_argmax(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਜੇ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_I32) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t ne00 = src0->ne[0];
    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    ਜੇ (ne00 == 0 || src0_nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (src0_nrows == 1) {
        ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1 && octx->ctx->mdev.idx > 0) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
        }

        ਜੇ (ne00 == 1) {
            ((int32_t *) dst->ਅੰਕੜੇ)[0] = 0;
            ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
        }

        ਅਚਲ uint32_t n_ਧਾਗੇ = (ne00 >= 1024) ? MIN(octx->n_ਧਾਗੇ, HTP_MAX_NTHREADS) : 1;
        ਅਚਲ uint32_t raw_chunk = (ne00 + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ;
        ਅਚਲ uint32_t elems_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(raw_chunk, 32);

        ਬਣਤਰ argmax_context actx = {
            .octx             = octx,
            .src_ਅੰਕੜੇ         = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ,
            .dst_ਅੰਕੜੇ         = (int32_t *) dst->ਅੰਕੜੇ,
            .ne00             = ne00,
            .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ)),
            .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) ਆਕਾਰ(int32_t),
            .ਪੰਕਤੀ_ਸ਼ੁਰੂ        = 0,
            .nrows            = 1,
            .ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread  = 1,
            .elems_ਪ੍ਰਤੀ_thread = elems_ਪ੍ਰਤੀ_thread,
        };

        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, argmax_thread_single_ਪੰਕਤੀ, &actx, n_ਧਾਗੇ);

        ਦਸ਼ਮਲਵ ਉੱਤਮ_val = actx.partial_max[0];
        int32_t ਉੱਤਮ_idx = actx.partial_idx[0];
        ਲਈ (uint32_t i = 1; i < n_ਧਾਗੇ; i++) {
            ਜੇ (actx.partial_max[i] > ਉੱਤਮ_val) {
                ਉੱਤਮ_val = actx.partial_max[i];
                ਉੱਤਮ_idx = actx.partial_idx[i];
            }
        }
        ((int32_t *) dst->ਅੰਕੜੇ)[0] = ਉੱਤਮ_idx;
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = src0_nrows;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) &&
                               htp_tensor_is_contiguous(dst, ਆਕਾਰ(int32_t));
        ਅਚਲ uint32_t elems_ਪ੍ਰਤੀ_chunk = can_ਵੰਡੋ ? 32 : 0;
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            src0_nrows, elems_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = MIN(octx->n_ਧਾਗੇ, nrows);
    ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (nrows + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ;

    ਬਣਤਰ argmax_context actx = {
        .octx             = octx,
        .src_ਅੰਕੜੇ         = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ,
        .dst_ਅੰਕੜੇ         = (int32_t *) dst->ਅੰਕੜੇ,
        .ne00             = ne00,
        .src_stride       = src0->nb[1] > 0 ? (uint32_t) src0->nb[1] : (uint32_t) (ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ)),
        .dst_stride       = dst->nb[0] > 0 ? (uint32_t) dst->nb[0] : (uint32_t) ਆਕਾਰ(int32_t),
        .ਪੰਕਤੀ_ਸ਼ੁਰੂ        = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .nrows            = nrows,
        .ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread  = ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread,
        .elems_ਪ੍ਰਤੀ_thread = 0,
    };

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, argmax_thread_multi_ਪੰਕਤੀ, &actx, n_ਧਾਗੇ);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
