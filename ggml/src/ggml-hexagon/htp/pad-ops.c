#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

/* Circular wrap: maps any integer x into [0, n) */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t wrap_around(int32_t x, uint32_t n) {
    ಹಿಂತಿರುಗಿ (uint32_t)(((x % (int32_t)n) + (int32_t)n) % (int32_t)n);
}

/* Decompose a flat dst row index into (i1, i2, i3) */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ pad_decompose_ಅಡ್ಡಸಾಲು(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     uint32_t *i1, uint32_t *i2, uint32_t *i3) {
    *i1 = ir % ne1;
    *i2 = (ir / ne1) % ne2;
    *i3 = ir / (ne1 * ne2);
}

/* Return non-zero if row (i1,i2,i3) falls in the non-padded interior */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                   int32_t lp1, int32_t rp1, uint32_t ne1,
                                   int32_t lp2, int32_t rp2, uint32_t ne2,
                                   int32_t lp3, int32_t rp3, uint32_t ne3) {
    ಹಿಂತಿರುಗಿ ((int32_t)i1 >= lp1 && (int32_t)i1 < (int32_t)ne1 - rp1) &&
           ((int32_t)i2 >= lp2 && (int32_t)i2 < (int32_t)ne2 - rp2) &&
           ((int32_t)i3 >= lp3 && (int32_t)i3 < (int32_t)ne3 - rp3);
}

/* Compute the DDR src row pointer for a zero-pad interior row */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ uint8_t * pad_src_ಅಡ್ಡಸಾಲು_ptr(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    ಹಿಂತಿರುಗಿ (ಸ್ಥಿರಾಂಕ uint8_t *) src->ದತ್ತಾಂಶ
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_addr_t pad_src_ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    ಹಿಂತಿರುಗಿ src->ದತ್ತಾಂಶ
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ uint8_t * pad_circ_src_ಅಡ್ಡಸಾಲು_ptr(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    ಹಿಂತಿರುಗಿ (ಸ್ಥಿರಾಂಕ uint8_t *) src->ದತ್ತಾಂಶ
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_addr_t pad_circ_src_ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    ಹಿಂತಿರುಗಿ src->ದತ್ತಾಂಶ
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ರಚನೆ htp_pad_context {
    ರಚನೆ htp_ops_context * octx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_ಪ್ರತಿ_thread;
    uint32_t ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;

    size_t   type_ಗಾತ್ರ;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t   src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t   dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t   dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
};

#ವ್ಯಾಖ್ಯಾನಿಸು htp_pad_preamble                                                             \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = octx->src[0];                                    \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;                                       \
                                                                                     \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src->ne[0];                                                \
    ಸ್ಥಿರಾಂಕ uint32_t nb00 = src->nb[0];                                                \
                                                                                     \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];                                                 \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];                                                 \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];                                                 \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];                                                 \
                                                                                     \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];                                                 \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];                                                 \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];                                                 \
                                                                                     \
    ಸ್ಥಿರಾಂಕ int32_t lp0 = pctx->lp0, rp0 = pctx->rp0;                                  \
    ಸ್ಥಿರಾಂಕ int32_t lp1 = pctx->lp1, rp1 = pctx->rp1;                                  \
    ಸ್ಥಿರಾಂಕ int32_t lp2 = pctx->lp2, rp2 = pctx->rp2;                                  \
    ಸ್ಥಿರಾಂಕ int32_t lp3 = pctx->lp3, rp3 = pctx->rp3;                                  \
                                                                                     \
    ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ = pctx->type_ಗಾತ್ರ;                                        \
                                                                                     \
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = pctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + pctx->nrows_ಪ್ರತಿ_thread * ith;       \
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ   = MIN(ಅಡ್ಡಸಾಲು_ಆರಂಭ + pctx->nrows_ಪ್ರತಿ_thread, pctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + pctx->ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು);


#ವ್ಯಾಖ್ಯಾನಿಸು htp_pad_dma_preamble                                                                \
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = pctx->src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;                                 \
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = pctx->src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;                         \
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = pctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;                                 \
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = pctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;                         \
                                                                                            \
    uint8_t * src_spad_ತಳ = octx->src0_spad.ದತ್ತಾಂಶ + ith * octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread; \
    uint8_t * dst_spad_ತಳ = octx->dst_spad.ದತ್ತಾಂಶ  + ith * octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;  \
                                                                                            \
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ pad_ಕೆಲಸ_ಪ್ರತಿ_thread_hvx(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pad_context * pctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pad_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ಅಡ್ಡಸಾಲು_ಆರಂಭ);

    ಗಾಗಿ (uint32_t dst_ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; dst_ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; dst_ಅಡ್ಡಸಾಲು++) {
        uint32_t i1, i2, i3;
        pad_decompose_ಅಡ್ಡಸಾಲು(dst_ಅಡ್ಡಸಾಲು, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->ದತ್ತಾಂಶ + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ಆದರೆ (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ uint8_t * src_ptr = pad_src_ಅಡ್ಡಸಾಲು_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            ಆದರೆ (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_ಅಡ್ಡಸಾಲು_ಆರಂಭ = dst_ptr + (size_t)lp0 * type_ಗಾತ್ರ;
            ಆದರೆ (nb00 == type_ಗಾತ್ರ) {
                hvx_copy_f32_uu(dst_ಅಡ್ಡಸಾಲು_ಆರಂಭ, src_ptr, ne00);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾಗಿ (uint32_t i = 0; i < ne00; i++) {
                    ಸ್ಮರಣೆ_ನಕಲು(dst_ಅಡ್ಡಸಾಲು_ಆರಂಭ + i * type_ಗಾತ್ರ,
                           src_ptr + (size_t)i * nb00,
                           type_ಗಾತ್ರ);
                }
            }

            ಆದರೆ (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_ಗಾತ್ರ, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ಅಡ್ಡಸಾಲು_ಆರಂಭ);

    FARF(ಮೇಲಿನ_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ಅಡ್ಡಸಾಲು_ಆರಂಭ, ಅಡ್ಡಸಾಲು_ಕೊನೆ);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ pad_ಕೆಲಸ_ಪ್ರತಿ_thread_hvx_dma(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pad_context * pctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pad_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    ಗಾಗಿ (uint32_t ir = ಅಡ್ಡಸಾಲು_ಆರಂಭ, spad_idx = 0; ir < ಅಡ್ಡಸಾಲು_ಕೊನೆ && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_ತಳ + spad_idx * src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
        uint8_t * dst_spad_cur = dst_spad_ತಳ + spad_idx * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(dst->ದತ್ತಾಂಶ, dst_spad_cur),
            dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);

        uint32_t i1, i2, i3;
        pad_decompose_ಅಡ್ಡಸಾಲು(ir, ne1, ne2, &i1, &i2, &i3);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ಸ್ಥಿರಾಂಕ dma_addr_t src_ದತ್ತಾಂಶ = interior
            ? pad_src_ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ(src, i1, i2, i3, lp1, lp2, lp3) : src->ದತ್ತಾಂಶ;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(src_spad_cur, src_ದತ್ತಾಂಶ),
            src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಗಾಗಿ (uint32_t ir = ಅಡ್ಡಸಾಲು_ಆರಂಭ; ir < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_ಅಡ್ಡಸಾಲು(ir, ne1, ne2, &i1, &i2, &i3);

        ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ = dst->ದತ್ತಾಂಶ + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ಆದರೆ (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } ಇಲ್ಲದಿದ್ದರೆ {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_ಗಾತ್ರ;

            ಆದರೆ ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } ಇಲ್ಲದಿದ್ದರೆ {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, dst_spad_cur),
            dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಅಡ್ಡಸಾಲು = ir + 2;
        ಆದರೆ (ಮುಂದಿನ_ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_ಅಡ್ಡಸಾಲು(ಮುಂದಿನ_ಅಡ್ಡಸಾಲು, ne1, ne2, &ni1, &ni2, &ni3);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಮುಂದಿನ_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            ಸ್ಥಿರಾಂಕ dma_addr_t ಮುಂದಿನ_src_ದತ್ತಾಂಶ = ಮುಂದಿನ_interior
                ? pad_src_ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->ದತ್ತಾಂಶ;

            dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                dma_make_ದತ್ತಾಂಶ(src_spad_cur, ಮುಂದಿನ_src_ದತ್ತಾಂಶ),
                src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಮುಂದಿನ_interior ? 1 : 0);
        }
    }

    dma_ಸರತಿ_flush(dma_q);

    FARF(ಮೇಲಿನ_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ಅಡ್ಡಸಾಲು_ಆರಂಭ, ಅಡ್ಡಸಾಲು_ಕೊನೆ);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ pad_ಕೆಲಸ_ಪ್ರತಿ_thread_hvx_circular(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pad_context * pctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pad_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ಅಡ್ಡಸಾಲು_ಆರಂಭ);

    ಗಾಗಿ (uint32_t dst_ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; dst_ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; dst_ಅಡ್ಡಸಾಲು++) {
        uint32_t i1, i2, i3;
        pad_decompose_ಅಡ್ಡಸಾಲು(dst_ಅಡ್ಡಸಾಲು, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->ದತ್ತಾಂಶ + i1 * nb1 + i2 * nb2 + i3 * nb3;
        ಸ್ಥಿರಾಂಕ uint8_t * src_ಅಡ್ಡಸಾಲು = pad_circ_src_ಅಡ್ಡಸಾಲು_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        ಆದರೆ (nb00 == type_ಗಾತ್ರ) {

            ಆದರೆ (lp0 > 0) {
                ಆದರೆ ((uint32_t)lp0 < 32) {
                    ಸ್ಮರಣೆ_ನಕಲು(dst_ptr,
                           src_ಅಡ್ಡಸಾಲು + (size_t)(ne00 - (uint32_t)lp0) * type_ಗಾತ್ರ,
                           (size_t)lp0 * type_ಗಾತ್ರ);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_ಅಡ್ಡಸಾಲು + (size_t)(ne00 - (uint32_t)lp0) * type_ಗಾತ್ರ,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_ಗಾತ್ರ, src_ಅಡ್ಡಸಾಲು, ne00);
            ಆದರೆ (rp0 > 0) {
                ಆದರೆ ((uint32_t)rp0 < 32) {
                    ಸ್ಮರಣೆ_ನಕಲು(dst_ptr + ((size_t)lp0 + ne00) * type_ಗಾತ್ರ,
                           src_ಅಡ್ಡಸಾಲು,
                           (size_t)rp0 * type_ಗಾತ್ರ);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_ಗಾತ್ರ,
                                    src_ಅಡ್ಡಸಾಲು,
                                    (uint32_t)rp0);
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(ದಶಮಾಂಶ *)(dst_ptr + i * type_ಗಾತ್ರ) =
                    *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(src_ಅಡ್ಡಸಾಲು + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            ಗಾಗಿ (uint32_t i = 0; i < ne00; i++) {
                *(ದಶಮಾಂಶ *)(dst_ptr + ((size_t)lp0 + i) * type_ಗಾತ್ರ) =
                    *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(src_ಅಡ್ಡಸಾಲು + (size_t)i * nb00);
            }
            ಗಾಗಿ (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(ದಶಮಾಂಶ *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_ಗಾತ್ರ) =
                    *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(src_ಅಡ್ಡಸಾಲು + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ಅಡ್ಡಸಾಲು_ಆರಂಭ);

    FARF(ಮೇಲಿನ_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ಅಡ್ಡಸಾಲು_ಆರಂಭ, ಅಡ್ಡಸಾಲು_ಕೊನೆ);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ pad_ಕೆಲಸ_ಪ್ರತಿ_thread_hvx_circular_dma(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pad_context * pctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_pad_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    ಗಾಗಿ (uint32_t ir = ಅಡ್ಡಸಾಲು_ಆರಂಭ, spad_idx = 0; ir < ಅಡ್ಡಸಾಲು_ಕೊನೆ && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_ತಳ + spad_idx * src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
        uint8_t * dst_spad_cur = dst_spad_ತಳ + spad_idx * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(dst->ದತ್ತಾಂಶ, dst_spad_cur),
            dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_ಅಡ್ಡಸಾಲು(ir, ne1, ne2, &pi1, &pi2, &pi3);
        ಸ್ಥಿರಾಂಕ dma_addr_t src_ದತ್ತಾಂಶ = pad_circ_src_ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(src_spad_cur, src_ದತ್ತಾಂಶ),
            src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಗಾಗಿ (uint32_t ir = ಅಡ್ಡಸಾಲು_ಆರಂಭ; ir < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_ಅಡ್ಡಸಾಲು(ir, ne1, ne2, &i1, &i2, &i3);
        ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ = dst->ದತ್ತಾಂಶ + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ಆದರೆ (lp0 > 0) {
            uint8_t * dst_ಎಡ       = dst_spad_cur;
            ಸ್ಥಿರಾಂಕ uint8_t * src_ಎಡ = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_ಗಾತ್ರ;
            ಆದರೆ ((uint32_t)lp0 < 32) {
                ಸ್ಮರಣೆ_ನಕಲು(dst_ಎಡ, src_ಎಡ, (size_t)lp0 * type_ಗಾತ್ರ);
            } ಇಲ್ಲದಿದ್ದರೆ {
                hvx_copy_f32_uu(dst_ಎಡ, src_ಎಡ, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_ಗಾತ್ರ;
            ಆದರೆ ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } ಇಲ್ಲದಿದ್ದರೆ {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        ಆದರೆ (rp0 > 0) {
            uint8_t * dst_ಬಲ = dst_spad_cur + ((size_t)lp0 + ne00) * type_ಗಾತ್ರ;
            ಆದರೆ ((uint32_t)rp0 < 32) {
                ಸ್ಮರಣೆ_ನಕಲು(dst_ಬಲ, src_spad_cur, (size_t)rp0 * type_ಗಾತ್ರ);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ ((uintptr_t)dst_ಬಲ % VLEN == 0) {
                    hvx_copy_f32_aa(dst_ಬಲ, src_spad_cur, (uint32_t)rp0);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    hvx_copy_f32_ua(dst_ಬಲ, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, dst_spad_cur),
            dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಅಡ್ಡಸಾಲು = ir + 2;
        ಆದರೆ (ಮುಂದಿನ_ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_ಅಡ್ಡಸಾಲು(ಮುಂದಿನ_ಅಡ್ಡಸಾಲು, ne1, ne2, &nri1, &nri2, &nri3);
            ಸ್ಥಿರಾಂಕ dma_addr_t ಮುಂದಿನ_src_ದತ್ತಾಂಶ = pad_circ_src_ಅಡ್ಡಸಾಲು_ದತ್ತಾಂಶ(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                dma_make_ದತ್ತಾಂಶ(src_spad_cur, ಮುಂದಿನ_src_ದತ್ತಾಂಶ),
                src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
        }
    }

    dma_ಸರತಿ_flush(dma_q);

    FARF(ಮೇಲಿನ_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ಅಡ್ಡಸಾಲು_ಆರಂಭ, ಅಡ್ಡಸಾಲು_ಕೊನೆ);
}

ಪೂರ್ಣಾಂಕ op_pad(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_ಗಾತ್ರ;
    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32: type_ಗಾತ್ರ = 4; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ int32_t lp0 = octx->op_params[0];
    ಸ್ಥಿರಾಂಕ int32_t rp0 = octx->op_params[1];
    ಸ್ಥಿರಾಂಕ int32_t lp1 = octx->op_params[2];
    ಸ್ಥಿರಾಂಕ int32_t rp1 = octx->op_params[3];
    ಸ್ಥಿರಾಂಕ int32_t lp2 = octx->op_params[4];
    ಸ್ಥಿರಾಂಕ int32_t rp2 = octx->op_params[5];
    ಸ್ಥಿರಾಂಕ int32_t lp3 = octx->op_params[6];
    ಸ್ಥಿರಾಂಕ int32_t rp3 = octx->op_params[7];
    ಸ್ಥಿರಾಂಕ int32_t circular = octx->op_params[8];

    ಸ್ಥಿರಾಂಕ uint32_t ne0  = dst->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ     = (size_t)ne0 * type_ಗಾತ್ರ;

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

    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = (size_t)ne00 * type_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    ಸ್ಥಿರಾಂಕ size_t vtcm_needed = (size_t)n_ಎಳೆಗಳು * 2 * (src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ use_dma = (src0->nb[0] == (uint32_t)type_ಗಾತ್ರ) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_ಗಾತ್ರ >= vtcm_needed);

    ಆದರೆ (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (use_dma) {
        octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread = 2 * src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
        octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread  = 2 * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
        octx->src0_spad.ಗಾತ್ರ_2 = n_ಎಳೆಗಳು * octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;
        octx->dst_spad.ಗಾತ್ರ_2  = n_ಎಳೆಗಳು * octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;
        octx->src0_spad.ದತ್ತಾಂಶ = octx->ctx->vtcm_ತಳ;
        octx->dst_spad.ದತ್ತಾಂಶ  = octx->src0_spad.ದತ್ತಾಂಶ + octx->src0_spad.ಗಾತ್ರ_2;
        octx->src0_spad.src  = ಖಾಲಿ;
        octx->dst_spad.src   = ಖಾಲಿ;
    }

    ರಚನೆ htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
        .ಒಟ್ಟು_dst_ಅಡ್ಡಸಾಲುಗಳು   = nrows,
        .ಅಡ್ಡಸಾಲು_ಆರಂಭ        = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .type_ಗಾತ್ರ        = type_ಗಾತ್ರ,
        .src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
        .src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
        .dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
        .dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
    };

    FARF(ಮೇಲಿನ_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    ಆದರೆ      (circular && use_dma) { work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, pad_ಕೆಲಸ_ಪ್ರತಿ_thread_hvx_circular_dma, &pctx, n_ಎಳೆಗಳು); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (circular)            { work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, pad_ಕೆಲಸ_ಪ್ರತಿ_thread_hvx_circular,     &pctx, n_ಎಳೆಗಳು); }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_dma)             { work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, pad_ಕೆಲಸ_ಪ್ರತಿ_thread_hvx_dma,          &pctx, n_ಎಳೆಗಳು); }
    ಇಲ್ಲದಿದ್ದರೆ                          { work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, pad_ಕೆಲಸ_ಪ್ರತಿ_thread_hvx,              &pctx, n_ಎಳೆಗಳು); }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
