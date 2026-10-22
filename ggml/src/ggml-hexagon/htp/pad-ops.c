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
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"

/* Circular wrap: maps any integer x into [0, n) */
ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t wrap_around(int32_t x, uint32_t n) {
    ਵਾਪਸ (uint32_t)(((x % (int32_t)n) + (int32_t)n) % (int32_t)n);
}

/* Decompose a flat dst row index into (i1, i2, i3) */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ pad_decompose_ਪੰਕਤੀ(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     uint32_t *i1, uint32_t *i2, uint32_t *i3) {
    *i1 = ir % ne1;
    *i2 = (ir / ne1) % ne2;
    *i3 = ir / (ne1 * ne2);
}

/* Return non-zero if row (i1,i2,i3) falls in the non-padded interior */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                   int32_t lp1, int32_t rp1, uint32_t ne1,
                                   int32_t lp2, int32_t rp2, uint32_t ne2,
                                   int32_t lp3, int32_t rp3, uint32_t ne3) {
    ਵਾਪਸ ((int32_t)i1 >= lp1 && (int32_t)i1 < (int32_t)ne1 - rp1) &&
           ((int32_t)i2 >= lp2 && (int32_t)i2 < (int32_t)ne2 - rp2) &&
           ((int32_t)i3 >= lp3 && (int32_t)i3 < (int32_t)ne3 - rp3);
}

/* Compute the DDR src row pointer for a zero-pad interior row */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ uint8_t * pad_src_ਪੰਕਤੀ_ptr(ਅਚਲ ਬਣਤਰ htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    ਵਾਪਸ (ਅਚਲ uint8_t *) src->ਅੰਕੜੇ
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_addr_t pad_src_ਪੰਕਤੀ_ਅੰਕੜੇ(ਅਚਲ ਬਣਤਰ htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    ਵਾਪਸ src->ਅੰਕੜੇ
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ uint8_t * pad_circ_src_ਪੰਕਤੀ_ptr(ਅਚਲ ਬਣਤਰ htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    ਵਾਪਸ (ਅਚਲ uint8_t *) src->ਅੰਕੜੇ
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_addr_t pad_circ_src_ਪੰਕਤੀ_ਅੰਕੜੇ(ਅਚਲ ਬਣਤਰ htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    ਵਾਪਸ src->ਅੰਕੜੇ
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ਬਣਤਰ htp_pad_context {
    ਬਣਤਰ htp_ops_context * octx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_ਪ੍ਰਤੀ_thread;
    uint32_t ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ;
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;

    size_t   type_ਮਾਪ;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_ਪੰਕਤੀ_ਮਾਪ;
    size_t   src_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t   dst_ਪੰਕਤੀ_ਮਾਪ;
    size_t   dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_pad_preamble                                                             \
    ਅਚਲ ਬਣਤਰ htp_tensor * src = octx->src[0];                                    \
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;                                       \
                                                                                     \
    ਅਚਲ uint32_t ne00 = src->ne[0];                                                \
    ਅਚਲ uint32_t nb00 = src->nb[0];                                                \
                                                                                     \
    ਅਚਲ uint32_t ne0 = dst->ne[0];                                                 \
    ਅਚਲ uint32_t ne1 = dst->ne[1];                                                 \
    ਅਚਲ uint32_t ne2 = dst->ne[2];                                                 \
    ਅਚਲ uint32_t ne3 = dst->ne[3];                                                 \
                                                                                     \
    ਅਚਲ uint32_t nb1 = dst->nb[1];                                                 \
    ਅਚਲ uint32_t nb2 = dst->nb[2];                                                 \
    ਅਚਲ uint32_t nb3 = dst->nb[3];                                                 \
                                                                                     \
    ਅਚਲ int32_t lp0 = pctx->lp0, rp0 = pctx->rp0;                                  \
    ਅਚਲ int32_t lp1 = pctx->lp1, rp1 = pctx->rp1;                                  \
    ਅਚਲ int32_t lp2 = pctx->lp2, rp2 = pctx->rp2;                                  \
    ਅਚਲ int32_t lp3 = pctx->lp3, rp3 = pctx->rp3;                                  \
                                                                                     \
    ਅਚਲ size_t type_ਮਾਪ = pctx->type_ਮਾਪ;                                        \
                                                                                     \
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = pctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + pctx->nrows_ਪ੍ਰਤੀ_thread * ith;       \
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ   = MIN(ਪੰਕਤੀ_ਸ਼ੁਰੂ + pctx->nrows_ਪ੍ਰਤੀ_thread, pctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + pctx->ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ);


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_pad_dma_preamble                                                                \
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ         = pctx->src_ਪੰਕਤੀ_ਮਾਪ;                                 \
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ_aligned = pctx->src_ਪੰਕਤੀ_ਮਾਪ_aligned;                         \
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ         = pctx->dst_ਪੰਕਤੀ_ਮਾਪ;                                 \
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned = pctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;                         \
                                                                                            \
    uint8_t * src_spad_ਅਧਾਰ = octx->src0_spad.ਅੰਕੜੇ + ith * octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread; \
    uint8_t * dst_spad_ਅਧਾਰ = octx->dst_spad.ਅੰਕੜੇ  + ith * octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;  \
                                                                                            \
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

ਸਥਿਰ ਖਾਲੀ pad_ਕੰਮ_ਪ੍ਰਤੀ_thread_hvx(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਅਚਲ ਬਣਤਰ htp_pad_context * pctx = (ਅਚਲ ਬਣਤਰ htp_pad_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ਪੰਕਤੀ_ਸ਼ੁਰੂ);

    ਲਈ (uint32_t dst_ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; dst_ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; dst_ਪੰਕਤੀ++) {
        uint32_t i1, i2, i3;
        pad_decompose_ਪੰਕਤੀ(dst_ਪੰਕਤੀ, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->ਅੰਕੜੇ + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ਅਚਲ ਪੂਰਨ interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ਜੇ (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ uint8_t * src_ptr = pad_src_ਪੰਕਤੀ_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            ਜੇ (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_ਪੰਕਤੀ_ਸ਼ੁਰੂ = dst_ptr + (size_t)lp0 * type_ਮਾਪ;
            ਜੇ (nb00 == type_ਮਾਪ) {
                hvx_copy_f32_uu(dst_ਪੰਕਤੀ_ਸ਼ੁਰੂ, src_ptr, ne00);
            } ਨਹੀਂਤਾਂ {
                ਲਈ (uint32_t i = 0; i < ne00; i++) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ਪੰਕਤੀ_ਸ਼ੁਰੂ + i * type_ਮਾਪ,
                           src_ptr + (size_t)i * nb00,
                           type_ਮਾਪ);
                }
            }

            ਜੇ (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_ਮਾਪ, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ਪੰਕਤੀ_ਸ਼ੁਰੂ);

    FARF(ਉੱਚਾ_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ਪੰਕਤੀ_ਸ਼ੁਰੂ, ਪੰਕਤੀ_ਸਿਰਾ);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ਸਥਿਰ ਖਾਲੀ pad_ਕੰਮ_ਪ੍ਰਤੀ_thread_hvx_dma(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਅਚਲ ਬਣਤਰ htp_pad_context * pctx = (ਅਚਲ ਬਣਤਰ htp_pad_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    ਲਈ (uint32_t ir = ਪੰਕਤੀ_ਸ਼ੁਰੂ, spad_idx = 0; ir < ਪੰਕਤੀ_ਸਿਰਾ && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_ਅਧਾਰ + spad_idx * src_ਪੰਕਤੀ_ਮਾਪ_aligned;
        uint8_t * dst_spad_cur = dst_spad_ਅਧਾਰ + spad_idx * dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(dst->ਅੰਕੜੇ, dst_spad_cur),
            dst_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 0);

        uint32_t i1, i2, i3;
        pad_decompose_ਪੰਕਤੀ(ir, ne1, ne2, &i1, &i2, &i3);
        ਅਚਲ ਪੂਰਨ interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ਅਚਲ dma_addr_t src_ਅੰਕੜੇ = interior
            ? pad_src_ਪੰਕਤੀ_ਅੰਕੜੇ(src, i1, i2, i3, lp1, lp2, lp3) : src->ਅੰਕੜੇ;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(src_spad_cur, src_ਅੰਕੜੇ),
            src_ਪੰਕਤੀ_ਮਾਪ_aligned, src_ਪੰਕਤੀ_ਮਾਪ, src_ਪੰਕਤੀ_ਮਾਪ, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਲਈ (uint32_t ir = ਪੰਕਤੀ_ਸ਼ੁਰੂ; ir < ਪੰਕਤੀ_ਸਿਰਾ; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_ਪੰਕਤੀ(ir, ne1, ne2, &i1, &i2, &i3);

        ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ = dst->ਅੰਕੜੇ + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ਅਚਲ ਪੂਰਨ interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ਜੇ (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } ਨਹੀਂਤਾਂ {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_ਮਾਪ;

            ਜੇ ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } ਨਹੀਂਤਾਂ {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, dst_spad_cur),
            dst_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 1);

        ਅਚਲ uint32_t ਅਗਲਾ_ਪੰਕਤੀ = ir + 2;
        ਜੇ (ਅਗਲਾ_ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_ਪੰਕਤੀ(ਅਗਲਾ_ਪੰਕਤੀ, ne1, ne2, &ni1, &ni2, &ni3);
            ਅਚਲ ਪੂਰਨ ਅਗਲਾ_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            ਅਚਲ dma_addr_t ਅਗਲਾ_src_ਅੰਕੜੇ = ਅਗਲਾ_interior
                ? pad_src_ਪੰਕਤੀ_ਅੰਕੜੇ(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->ਅੰਕੜੇ;

            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                dma_make_ਅੰਕੜੇ(src_spad_cur, ਅਗਲਾ_src_ਅੰਕੜੇ),
                src_ਪੰਕਤੀ_ਮਾਪ_aligned, src_ਪੰਕਤੀ_ਮਾਪ, src_ਪੰਕਤੀ_ਮਾਪ, ਅਗਲਾ_interior ? 1 : 0);
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);

    FARF(ਉੱਚਾ_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ਪੰਕਤੀ_ਸ਼ੁਰੂ, ਪੰਕਤੀ_ਸਿਰਾ);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

ਸਥਿਰ ਖਾਲੀ pad_ਕੰਮ_ਪ੍ਰਤੀ_thread_hvx_circular(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਅਚਲ ਬਣਤਰ htp_pad_context * pctx = (ਅਚਲ ਬਣਤਰ htp_pad_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ਪੰਕਤੀ_ਸ਼ੁਰੂ);

    ਲਈ (uint32_t dst_ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; dst_ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; dst_ਪੰਕਤੀ++) {
        uint32_t i1, i2, i3;
        pad_decompose_ਪੰਕਤੀ(dst_ਪੰਕਤੀ, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->ਅੰਕੜੇ + i1 * nb1 + i2 * nb2 + i3 * nb3;
        ਅਚਲ uint8_t * src_ਪੰਕਤੀ = pad_circ_src_ਪੰਕਤੀ_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        ਜੇ (nb00 == type_ਮਾਪ) {

            ਜੇ (lp0 > 0) {
                ਜੇ ((uint32_t)lp0 < 32) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ptr,
                           src_ਪੰਕਤੀ + (size_t)(ne00 - (uint32_t)lp0) * type_ਮਾਪ,
                           (size_t)lp0 * type_ਮਾਪ);
                } ਨਹੀਂਤਾਂ {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_ਪੰਕਤੀ + (size_t)(ne00 - (uint32_t)lp0) * type_ਮਾਪ,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_ਮਾਪ, src_ਪੰਕਤੀ, ne00);
            ਜੇ (rp0 > 0) {
                ਜੇ ((uint32_t)rp0 < 32) {
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ptr + ((size_t)lp0 + ne00) * type_ਮਾਪ,
                           src_ਪੰਕਤੀ,
                           (size_t)rp0 * type_ਮਾਪ);
                } ਨਹੀਂਤਾਂ {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_ਮਾਪ,
                                    src_ਪੰਕਤੀ,
                                    (uint32_t)rp0);
                }
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(ਦਸ਼ਮਲਵ *)(dst_ptr + i * type_ਮਾਪ) =
                    *(ਅਚਲ ਦਸ਼ਮਲਵ *)(src_ਪੰਕਤੀ + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            ਲਈ (uint32_t i = 0; i < ne00; i++) {
                *(ਦਸ਼ਮਲਵ *)(dst_ptr + ((size_t)lp0 + i) * type_ਮਾਪ) =
                    *(ਅਚਲ ਦਸ਼ਮਲਵ *)(src_ਪੰਕਤੀ + (size_t)i * nb00);
            }
            ਲਈ (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(ਦਸ਼ਮਲਵ *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_ਮਾਪ) =
                    *(ਅਚਲ ਦਸ਼ਮਲਵ *)(src_ਪੰਕਤੀ + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ਪੰਕਤੀ_ਸ਼ੁਰੂ);

    FARF(ਉੱਚਾ_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ਪੰਕਤੀ_ਸ਼ੁਰੂ, ਪੰਕਤੀ_ਸਿਰਾ);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ਸਥਿਰ ਖਾਲੀ pad_ਕੰਮ_ਪ੍ਰਤੀ_thread_hvx_circular_dma(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਅਚਲ ਬਣਤਰ htp_pad_context * pctx = (ਅਚਲ ਬਣਤਰ htp_pad_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    ਲਈ (uint32_t ir = ਪੰਕਤੀ_ਸ਼ੁਰੂ, spad_idx = 0; ir < ਪੰਕਤੀ_ਸਿਰਾ && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_ਅਧਾਰ + spad_idx * src_ਪੰਕਤੀ_ਮਾਪ_aligned;
        uint8_t * dst_spad_cur = dst_spad_ਅਧਾਰ + spad_idx * dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(dst->ਅੰਕੜੇ, dst_spad_cur),
            dst_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_ਪੰਕਤੀ(ir, ne1, ne2, &pi1, &pi2, &pi3);
        ਅਚਲ dma_addr_t src_ਅੰਕੜੇ = pad_circ_src_ਪੰਕਤੀ_ਅੰਕੜੇ(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(src_spad_cur, src_ਅੰਕੜੇ),
            src_ਪੰਕਤੀ_ਮਾਪ_aligned, src_ਪੰਕਤੀ_ਮਾਪ, src_ਪੰਕਤੀ_ਮਾਪ, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਲਈ (uint32_t ir = ਪੰਕਤੀ_ਸ਼ੁਰੂ; ir < ਪੰਕਤੀ_ਸਿਰਾ; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_ਪੰਕਤੀ(ir, ne1, ne2, &i1, &i2, &i3);
        ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ = dst->ਅੰਕੜੇ + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ਜੇ (lp0 > 0) {
            uint8_t * dst_ਖੱਬਾ       = dst_spad_cur;
            ਅਚਲ uint8_t * src_ਖੱਬਾ = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_ਮਾਪ;
            ਜੇ ((uint32_t)lp0 < 32) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ਖੱਬਾ, src_ਖੱਬਾ, (size_t)lp0 * type_ਮਾਪ);
            } ਨਹੀਂਤਾਂ {
                hvx_copy_f32_uu(dst_ਖੱਬਾ, src_ਖੱਬਾ, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_ਮਾਪ;
            ਜੇ ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } ਨਹੀਂਤਾਂ {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        ਜੇ (rp0 > 0) {
            uint8_t * dst_ਸੱਜਾ = dst_spad_cur + ((size_t)lp0 + ne00) * type_ਮਾਪ;
            ਜੇ ((uint32_t)rp0 < 32) {
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_ਸੱਜਾ, src_spad_cur, (size_t)rp0 * type_ਮਾਪ);
            } ਨਹੀਂਤਾਂ {
                ਜੇ ((uintptr_t)dst_ਸੱਜਾ % VLEN == 0) {
                    hvx_copy_f32_aa(dst_ਸੱਜਾ, src_spad_cur, (uint32_t)rp0);
                } ਨਹੀਂਤਾਂ {
                    hvx_copy_f32_ua(dst_ਸੱਜਾ, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, dst_spad_cur),
            dst_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 1);

        ਅਚਲ uint32_t ਅਗਲਾ_ਪੰਕਤੀ = ir + 2;
        ਜੇ (ਅਗਲਾ_ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_ਪੰਕਤੀ(ਅਗਲਾ_ਪੰਕਤੀ, ne1, ne2, &nri1, &nri2, &nri3);
            ਅਚਲ dma_addr_t ਅਗਲਾ_src_ਅੰਕੜੇ = pad_circ_src_ਪੰਕਤੀ_ਅੰਕੜੇ(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                dma_make_ਅੰਕੜੇ(src_spad_cur, ਅਗਲਾ_src_ਅੰਕੜੇ),
                src_ਪੰਕਤੀ_ਮਾਪ_aligned, src_ਪੰਕਤੀ_ਮਾਪ, src_ਪੰਕਤੀ_ਮਾਪ, 1);
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);

    FARF(ਉੱਚਾ_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ਪੰਕਤੀ_ਸ਼ੁਰੂ, ਪੰਕਤੀ_ਸਿਰਾ);
}

ਪੂਰਨ op_pad(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_ਮਾਪ;
    ਚੋਣ (src0->type) {
        ਹਾਲਤ HTP_TYPE_F32: type_ਮਾਪ = 4; ਰੋਕੋ;
        ਮੂਲ:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ int32_t lp0 = octx->op_params[0];
    ਅਚਲ int32_t rp0 = octx->op_params[1];
    ਅਚਲ int32_t lp1 = octx->op_params[2];
    ਅਚਲ int32_t rp1 = octx->op_params[3];
    ਅਚਲ int32_t lp2 = octx->op_params[4];
    ਅਚਲ int32_t rp2 = octx->op_params[5];
    ਅਚਲ int32_t lp3 = octx->op_params[6];
    ਅਚਲ int32_t rp3 = octx->op_params[7];
    ਅਚਲ int32_t circular = octx->op_params[8];

    ਅਚਲ uint32_t ne0  = dst->ne[0];
    ਅਚਲ uint32_t ne00 = src0->ne[0];

    ਅਚਲ uint32_t ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ     = (size_t)ne0 * type_ਮਾਪ;

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

    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ         = (size_t)ne00 * type_ਮਾਪ;
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src_ਪੰਕਤੀ_ਮਾਪ, VLEN);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(dst_ਪੰਕਤੀ_ਮਾਪ, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    ਅਚਲ size_t vtcm_needed = (size_t)n_ਧਾਗੇ * 2 * (src_ਪੰਕਤੀ_ਮਾਪ_aligned + dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

    ਅਚਲ ਪੂਰਨ use_dma = (src0->nb[0] == (uint32_t)type_ਮਾਪ) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_ਮਾਪ >= vtcm_needed);

    ਜੇ (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (use_dma) {
        octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread = 2 * src_ਪੰਕਤੀ_ਮਾਪ_aligned;
        octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread  = 2 * dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
        octx->src0_spad.ਮਾਪ = n_ਧਾਗੇ * octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;
        octx->dst_spad.ਮਾਪ  = n_ਧਾਗੇ * octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;
        octx->src0_spad.ਅੰਕੜੇ = octx->ctx->vtcm_ਅਧਾਰ;
        octx->dst_spad.ਅੰਕੜੇ  = octx->src0_spad.ਅੰਕੜੇ + octx->src0_spad.ਮਾਪ;
        octx->src0_spad.src  = ਨਿਰਾ;
        octx->dst_spad.src   = ਨਿਰਾ;
    }

    ਬਣਤਰ htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
        .ਕੁੱਲ_dst_ਪੰਕਤੀਆਂ   = nrows,
        .ਪੰਕਤੀ_ਸ਼ੁਰੂ        = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .type_ਮਾਪ        = type_ਮਾਪ,
        .src_ਪੰਕਤੀ_ਮਾਪ         = src_ਪੰਕਤੀ_ਮਾਪ,
        .src_ਪੰਕਤੀ_ਮਾਪ_aligned = src_ਪੰਕਤੀ_ਮਾਪ_aligned,
        .dst_ਪੰਕਤੀ_ਮਾਪ         = dst_ਪੰਕਤੀ_ਮਾਪ,
        .dst_ਪੰਕਤੀ_ਮਾਪ_aligned = dst_ਪੰਕਤੀ_ਮਾਪ_aligned,
    };

    FARF(ਉੱਚਾ_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    ਜੇ      (circular && use_dma) { work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, pad_ਕੰਮ_ਪ੍ਰਤੀ_thread_hvx_circular_dma, &pctx, n_ਧਾਗੇ); }
    ਨਹੀਂਤਾਂ ਜੇ (circular)            { work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, pad_ਕੰਮ_ਪ੍ਰਤੀ_thread_hvx_circular,     &pctx, n_ਧਾਗੇ); }
    ਨਹੀਂਤਾਂ ਜੇ (use_dma)             { work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, pad_ਕੰਮ_ਪ੍ਰਤੀ_thread_hvx_dma,          &pctx, n_ਧਾਗੇ); }
    ਨਹੀਂਤਾਂ                          { work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, pad_ਕੰਮ_ਪ੍ਰਤੀ_thread_hvx,              &pctx, n_ਧਾਗੇ); }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
