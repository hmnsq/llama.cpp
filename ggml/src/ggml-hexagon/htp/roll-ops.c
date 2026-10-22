#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"

ਬਣਤਰ htp_roll_context {
    ਬਣਤਰ htp_ops_context * octx;

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t nrows;
    uint32_t nrows_ਪ੍ਰਤੀ_thread;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne1;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne2_ne1;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    ਜੇ (i < 0) {
        ਵਾਪਸ (uint32_t) (i + (int32_t) ne);
    }
    ਜੇ ((uint32_t) i >= ne) {
        ਵਾਪਸ (uint32_t) i - ne;
    }
    ਵਾਪਸ (uint32_t) i;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_roll_preamble                             \
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];    \
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;       \
                                                      \
    ਅਚਲ uint32_t ne0 = dst->ne[0];                  \
    ਅਚਲ uint32_t ne1 = dst->ne[1];                  \
    ਅਚਲ uint32_t ne2 = dst->ne[2];                  \
    ਅਚਲ uint32_t ne3 = dst->ne[3];                  \
                                                      \
    ਅਚਲ uint32_t nb01 = src0->nb[1];                \
    ਅਚਲ uint32_t nb02 = src0->nb[2];                \
    ਅਚਲ uint32_t nb03 = src0->nb[3];                \
                                                      \
    ਅਚਲ uint32_t nb1 = dst->nb[1];                  \
    ਅਚਲ uint32_t nb2 = dst->nb[2];                  \
    ਅਚਲ uint32_t nb3 = dst->nb[3];                  \
                                                      \
    ਅਚਲ int32_t s0 = octx->op_params[0];            \
    ਅਚਲ int32_t s1 = octx->op_params[1];            \
    ਅਚਲ int32_t s2 = octx->op_params[2];            \
    ਅਚਲ int32_t s3 = octx->op_params[3];            \
                                                      \
    ਅਚਲ uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    ਅਚਲ uint32_t n0      = ne0 - i0_src0;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_roll_dma_preamble dma_ਕਤਾਰ * q = octx->ctx->dma[0];

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ roll_dma_ਧੱਕੋ(dma_ਕਤਾਰ * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    ਬਾਈਟਾਂ,
                                 uint32_t    nrows) {
    ਜੇ (ਬਾਈਟਾਂ == 0 || nrows == 0) {
        ਵਾਪਸ;
    }

    ਜੇ (!dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(dst, src), dst_stride, src_stride, ਬਾਈਟਾਂ, nrows)) {
        dma_ਕਤਾਰ_flush(q);
        dma_ਕਤਾਰ_ਧੱਕੋ(q, dma_make_ਅੰਕੜੇ(dst, src),
                       dst_stride, src_stride, ਬਾਈਟਾਂ, nrows);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ roll_dma_ਧੱਕੋ_ਪੰਕਤੀਆਂ(dma_ਕਤਾਰ *               q,
                                      ਅਚਲ ਬਣਤਰ htp_tensor * dst,
                                      ਅਚਲ ਬਣਤਰ htp_tensor * src0,
                                      uint32_t                  dst_ਪੰਕਤੀ,
                                      uint32_t                  src_ਪੰਕਤੀ,
                                      uint32_t                  nrows,
                                      uint32_t                  ਪੰਕਤੀ_ਮਾਪ,
                                      uint32_t                  i0_src0) {
    ਅਚਲ dma_addr_t dst_ਅਧਾਰ = dst->ਅੰਕੜੇ + (size_t) dst_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ dma_addr_t src_ਅਧਾਰ = src0->ਅੰਕੜੇ + (size_t) src_ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_ਧੱਕੋ(q, dst_ਅਧਾਰ, src_ਅਧਾਰ + (size_t) i0_src0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                  ਪੰਕਤੀ_ਮਾਪ, ਪੰਕਤੀ_ਮਾਪ, n0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), nrows);
    roll_dma_ਧੱਕੋ(q, dst_ਅਧਾਰ + (size_t) n0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), src_ਅਧਾਰ,
                  ਪੰਕਤੀ_ਮਾਪ, ਪੰਕਤੀ_ਮਾਪ, i0_src0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ roll_dma_ਧੱਕੋ_range(dma_ਕਤਾਰ * q,
                                       dma_addr_t  dst_ਪੰਕਤੀ,
                                       dma_addr_t  src_ਪੰਕਤੀ,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_ਧੱਕੋ(q, dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ + (size_t) i0_src0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                  dst_stride, src_stride, n0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), nrows);
    roll_dma_ਧੱਕੋ(q, dst_ਪੰਕਤੀ + (size_t) n0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), src_ਪੰਕਤੀ,
                  dst_stride, src_stride, i0_src0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), nrows);
}

ਸਥਿਰ ਪੂਰਨ roll_dma_f32_contiguous(ਬਣਤਰ htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ = ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਜੇ (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_ਧੱਕੋ_ਪੰਕਤੀਆਂ(q, dst, src0, 0, 0, ne1 * ne2 * ne3, ਪੰਕਤੀ_ਮਾਪ, i0_src0);
        dma_ਕਤਾਰ_flush(q);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (s1 == 0) {
        ਅਚਲ uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        ਲਈ (uint32_t i3 = 0; i3 < ne3; i3++) {
            ਅਚਲ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            ਅਚਲ uint32_t dst_row0 = i3 * ne2 * ne1;
            ਅਚਲ uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            ਅਚਲ uint32_t n2_ਪਹਿਲਾ = ne2 - i2_src0;

            roll_dma_ਧੱਕੋ_ਪੰਕਤੀਆਂ(q, dst, src0, dst_row0, src_row0, n2_ਪਹਿਲਾ * ne1,
                               ਪੰਕਤੀ_ਮਾਪ, i0_src0);
            roll_dma_ਧੱਕੋ_ਪੰਕਤੀਆਂ(q, dst, src0, dst_row0 + n2_ਪਹਿਲਾ * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, ਪੰਕਤੀ_ਮਾਪ, i0_src0);
        }

        dma_ਕਤਾਰ_flush(q);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    ਅਚਲ uint32_t n1_ਪਹਿਲਾ = ne1 - i1_src0;

    ਲਈ (uint32_t i3 = 0; i3 < ne3; i3++) {
        ਅਚਲ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        ਲਈ (uint32_t i2 = 0; i2 < ne2; i2++) {
            ਅਚਲ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            ਅਚਲ uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            ਅਚਲ uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_ਧੱਕੋ_ਪੰਕਤੀਆਂ(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_ਪਹਿਲਾ, ਪੰਕਤੀ_ਮਾਪ, i0_src0);
            roll_dma_ਧੱਕੋ_ਪੰਕਤੀਆਂ(q, dst, src0, dst_row0 + n1_ਪਹਿਲਾ, src_row0,
                               i1_src0, ਪੰਕਤੀ_ਮਾਪ, i0_src0);
        }
    }

    dma_ਕਤਾਰ_flush(q);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
ਸਥਿਰ ਪੂਰਨ roll_dma_f32_strided(ਬਣਤਰ htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ਅਚਲ uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    ਅਚਲ uint32_t n1_ਪਹਿਲਾ = ne1 - i1_src0;

    ਲਈ (uint32_t i3 = 0; i3 < ne3; i3++) {
        ਅਚਲ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        ਲਈ (uint32_t i2 = 0; i2 < ne2; i2++) {
            ਅਚਲ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            ਅਚਲ dma_addr_t dst_row0 = dst->ਅੰਕੜੇ  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            ਅਚਲ dma_addr_t src_row0 = src0->ਅੰਕੜੇ + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_ਧੱਕੋ_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_ਪਹਿਲਾ, i0_src0, n0);
            roll_dma_ਧੱਕੋ_range(q, dst_row0 + (size_t) n1_ਪਹਿਲਾ * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_ਕਤਾਰ_flush(q);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਖਾਲੀ roll_thread_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_roll_context * rctx = (ਬਣਤਰ htp_roll_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = rctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + rctx->nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ   = MIN(ਪੰਕਤੀ_ਸ਼ੁਰੂ + rctx->nrows_ਪ੍ਰਤੀ_thread, rctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + rctx->nrows);
    ਜੇ (ਪੰਕਤੀ_ਸ਼ੁਰੂ >= ਪੰਕਤੀ_ਸਿਰਾ) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ਪੰਕਤੀ_ਸ਼ੁਰੂ);

    ਲਈ (uint32_t ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; ਪੰਕਤੀ++) {
        ਅਚਲ uint32_t i3  = fastdiv(ਪੰਕਤੀ, &rctx->div_ne2_ne1);
        ਅਚਲ uint32_t rem = ਪੰਕਤੀ - i3 * ne2 * ne1;
        ਅਚਲ uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        ਅਚਲ uint32_t i1  = rem - i2 * ne1;

        ਅਚਲ uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        ਅਚਲ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        ਅਚਲ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        ਅਚਲ uint8_t * src_ਪੰਕਤੀ = (ਅਚਲ uint8_t *) (uintptr_t) src0->ਅੰਕੜੇ + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_ਪੰਕਤੀ = (uint8_t *) (uintptr_t) dst->ਅੰਕੜੇ + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_ਪੰਕਤੀ + i0_src0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), n0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 1);
        hvx_copy_uu(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ + i0_src0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), n0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        ਜੇ (i0_src0 != 0) {
            hex_l2fetch(src_ਪੰਕਤੀ, i0_src0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 1);
            hvx_copy_uu(dst_ਪੰਕਤੀ + n0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), src_ਪੰਕਤੀ, i0_src0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ਪੰਕਤੀ_ਸ਼ੁਰੂ);

    FARF(ਉੱਚਾ_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         ਪੰਕਤੀ_ਸ਼ੁਰੂ, ਪੰਕਤੀ_ਸਿਰਾ, s0, s1, s2, s3);
}

ਪੂਰਨ execute_op_roll_f32(ਬਣਤਰ htp_ops_context * octx) {
    htp_roll_preamble;

    ਜੇ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (src0->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || dst->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ne1 * ne2 * ne3;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ = ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = ਕੁੱਲ_ਪੰਕਤੀਆਂ;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ), (uint32_t) dst_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_ਪੰਕਤੀਆਂ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 <= 1) {
        ਜੇ (htp_tensor_is_contiguous(src0, ਆਕਾਰ(ਦਸ਼ਮਲਵ)) && htp_tensor_is_contiguous(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ))) {
            ਵਾਪਸ roll_dma_f32_contiguous(octx);
        }
        ਵਾਪਸ roll_dma_f32_strided(octx);
    }

    ਜੇ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;
    ਬਣਤਰ htp_roll_context rctx = {
        .octx             = octx,
        .ਪੰਕਤੀ_ਸ਼ੁਰੂ        = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .nrows            = nrows,
        .nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
        .div_ne1          = init_fastdiv_ਮੁੱਲਾਂ(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_ਮੁੱਲਾਂ(dst->ne[2] * dst->ne[1]),
    };

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, roll_thread_f32, &rctx, n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_roll(ਬਣਤਰ htp_ops_context * octx) {
    ਚੋਣ (octx->src[0]->type) {
        ਹਾਲਤ HTP_TYPE_F32:
            ਵਾਪਸ execute_op_roll_f32(octx);

        ਮੂਲ:
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
}
