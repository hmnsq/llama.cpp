#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

/* Circular wrap: maps any integer x into [0, n) */
ساكن مضمن uint32_t wrap_around(int32_t x, uint32_t n) {
    أرجع (uint32_t)(((x % (int32_t)n) + (int32_t)n) % (int32_t)n);
}

/* Decompose a flat dst row index into (i1, i2, i3) */
ساكن مضمن فراغ pad_decompose_صف(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     uint32_t *i1, uint32_t *i2, uint32_t *i3) {
    *i1 = ir % ne1;
    *i2 = (ir / ne1) % ne2;
    *i3 = ir / (ne1 * ne2);
}

/* Return non-zero if row (i1,i2,i3) falls in the non-padded interior */
ساكن مضمن صحيح pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                   int32_t lp1, int32_t rp1, uint32_t ne1,
                                   int32_t lp2, int32_t rp2, uint32_t ne2,
                                   int32_t lp3, int32_t rp3, uint32_t ne3) {
    أرجع ((int32_t)i1 >= lp1 && (int32_t)i1 < (int32_t)ne1 - rp1) &&
           ((int32_t)i2 >= lp2 && (int32_t)i2 < (int32_t)ne2 - rp2) &&
           ((int32_t)i3 >= lp3 && (int32_t)i3 < (int32_t)ne3 - rp3);
}

/* Compute the DDR src row pointer for a zero-pad interior row */
ساكن مضمن ثابت uint8_t * pad_src_صف_ptr(ثابت بنية htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع (ثابت uint8_t *) src->انکڑے
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

ساكن مضمن dma_addr_t pad_src_صف_انکڑے(ثابت بنية htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع src->انکڑے
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
ساكن مضمن ثابت uint8_t * pad_circ_src_صف_ptr(ثابت بنية htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع (ثابت uint8_t *) src->انکڑے
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ساكن مضمن dma_addr_t pad_circ_src_صف_انکڑے(ثابت بنية htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع src->انکڑے
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

بنية htp_pad_context {
    بنية htp_ops_context * octx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_فی_thread;
    uint32_t کل_dst_صفاں;
    uint32_t صف_شروع;

    size_t   type_ناپ;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_صف_ناپ;
    size_t   src_صف_ناپ_aligned;
    size_t   dst_صف_ناپ;
    size_t   dst_صف_ناپ_aligned;
};

#تعریف_کرو htp_pad_preamble                                                             \
    ثابت بنية htp_tensor * src = octx->src[0];                                    \
    ثابت بنية htp_tensor * dst = octx->dst;                                       \
                                                                                     \
    ثابت uint32_t ne00 = src->ne[0];                                                \
    ثابت uint32_t nb00 = src->nb[0];                                                \
                                                                                     \
    ثابت uint32_t ne0 = dst->ne[0];                                                 \
    ثابت uint32_t ne1 = dst->ne[1];                                                 \
    ثابت uint32_t ne2 = dst->ne[2];                                                 \
    ثابت uint32_t ne3 = dst->ne[3];                                                 \
                                                                                     \
    ثابت uint32_t nb1 = dst->nb[1];                                                 \
    ثابت uint32_t nb2 = dst->nb[2];                                                 \
    ثابت uint32_t nb3 = dst->nb[3];                                                 \
                                                                                     \
    ثابت int32_t lp0 = pctx->lp0, rp0 = pctx->rp0;                                  \
    ثابت int32_t lp1 = pctx->lp1, rp1 = pctx->rp1;                                  \
    ثابت int32_t lp2 = pctx->lp2, rp2 = pctx->rp2;                                  \
    ثابت int32_t lp3 = pctx->lp3, rp3 = pctx->rp3;                                  \
                                                                                     \
    ثابت size_t type_ناپ = pctx->type_ناپ;                                        \
                                                                                     \
    ثابت uint32_t صف_شروع = pctx->صف_شروع + pctx->nrows_فی_thread * ith;       \
    ثابت uint32_t صف_سرا   = MIN(صف_شروع + pctx->nrows_فی_thread, pctx->صف_شروع + pctx->کل_dst_صفاں);


#تعریف_کرو htp_pad_dma_preamble                                                                \
    ثابت size_t src_صف_ناپ         = pctx->src_صف_ناپ;                                 \
    ثابت size_t src_صف_ناپ_aligned = pctx->src_صف_ناپ_aligned;                         \
    ثابت size_t dst_صف_ناپ         = pctx->dst_صف_ناپ;                                 \
    ثابت size_t dst_صف_ناپ_aligned = pctx->dst_صف_ناپ_aligned;                         \
                                                                                            \
    uint8_t * src_spad_ادھار = octx->src0_spad.انکڑے + ith * octx->src0_spad.ناپ_فی_thread; \
    uint8_t * dst_spad_ادھار = octx->dst_spad.انکڑے  + ith * octx->dst_spad.ناپ_فی_thread;  \
                                                                                            \
    dma_قطار * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

ساكن فراغ pad_کم_فی_thread_hvx(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) انکڑے;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروع);

    لكل (uint32_t dst_صف = صف_شروع; dst_صف < صف_سرا; dst_صف++) {
        uint32_t i1, i2, i3;
        pad_decompose_صف(dst_صف, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->انکڑے + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ثابت صحيح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        إذا (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } وإلا {
            ثابت uint8_t * src_ptr = pad_src_صف_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            إذا (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_صف_شروع = dst_ptr + (size_t)lp0 * type_ناپ;
            إذا (nb00 == type_ناپ) {
                hvx_copy_f32_uu(dst_صف_شروع, src_ptr, ne00);
            } وإلا {
                لكل (uint32_t i = 0; i < ne00; i++) {
                    حافظہ_نقل(dst_صف_شروع + i * type_ناپ,
                           src_ptr + (size_t)i * nb00,
                           type_ناپ);
                }
            }

            إذا (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_ناپ, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروع);

    FARF(بلند_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_شروع, صف_سرا);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ساكن فراغ pad_کم_فی_thread_hvx_dma(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) انکڑے;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    لكل (uint32_t ir = صف_شروع, spad_idx = 0; ir < صف_سرا && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_ادھار + spad_idx * src_صف_ناپ_aligned;
        uint8_t * dst_spad_cur = dst_spad_ادھار + spad_idx * dst_صف_ناپ_aligned;

        dma_قطار_ڈالو(dma_q,
            dma_make_انکڑے(dst->انکڑے, dst_spad_cur),
            dst_صف_ناپ, dst_صف_ناپ_aligned, dst_صف_ناپ, 0);

        uint32_t i1, i2, i3;
        pad_decompose_صف(ir, ne1, ne2, &i1, &i2, &i3);
        ثابت صحيح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ثابت dma_addr_t src_انکڑے = interior
            ? pad_src_صف_انکڑے(src, i1, i2, i3, lp1, lp2, lp3) : src->انکڑے;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_قطار_ڈالو(dma_q,
            dma_make_انکڑے(src_spad_cur, src_انکڑے),
            src_صف_ناپ_aligned, src_صف_ناپ, src_صف_ناپ, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = صف_شروع; ir < صف_سرا; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_صف(ir, ne1, ne2, &i1, &i2, &i3);

        ثابت dma_addr_t dst_انکڑے = dst->انکڑے + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ثابت صحيح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        إذا (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } وإلا {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_ناپ;

            إذا ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } وإلا {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_قطار_ڈالو(dma_q,
            dma_make_انکڑے(dst_انکڑے, dst_spad_cur),
            dst_صف_ناپ, dst_صف_ناپ_aligned, dst_صف_ناپ, 1);

        ثابت uint32_t اگلا_صف = ir + 2;
        إذا (اگلا_صف < صف_سرا) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_صف(اگلا_صف, ne1, ne2, &ni1, &ni2, &ni3);
            ثابت صحيح اگلا_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            ثابت dma_addr_t اگلا_src_انکڑے = اگلا_interior
                ? pad_src_صف_انکڑے(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->انکڑے;

            dma_قطار_ڈالو(dma_q,
                dma_make_انکڑے(src_spad_cur, اگلا_src_انکڑے),
                src_صف_ناپ_aligned, src_صف_ناپ, src_صف_ناپ, اگلا_interior ? 1 : 0);
        }
    }

    dma_قطار_flush(dma_q);

    FARF(بلند_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_شروع, صف_سرا);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

ساكن فراغ pad_کم_فی_thread_hvx_circular(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) انکڑے;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروع);

    لكل (uint32_t dst_صف = صف_شروع; dst_صف < صف_سرا; dst_صف++) {
        uint32_t i1, i2, i3;
        pad_decompose_صف(dst_صف, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->انکڑے + i1 * nb1 + i2 * nb2 + i3 * nb3;
        ثابت uint8_t * src_صف = pad_circ_src_صف_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        إذا (nb00 == type_ناپ) {

            إذا (lp0 > 0) {
                إذا ((uint32_t)lp0 < 32) {
                    حافظہ_نقل(dst_ptr,
                           src_صف + (size_t)(ne00 - (uint32_t)lp0) * type_ناپ,
                           (size_t)lp0 * type_ناپ);
                } وإلا {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_صف + (size_t)(ne00 - (uint32_t)lp0) * type_ناپ,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_ناپ, src_صف, ne00);
            إذا (rp0 > 0) {
                إذا ((uint32_t)rp0 < 32) {
                    حافظہ_نقل(dst_ptr + ((size_t)lp0 + ne00) * type_ناپ,
                           src_صف,
                           (size_t)rp0 * type_ناپ);
                } وإلا {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_ناپ,
                                    src_صف,
                                    (uint32_t)rp0);
                }
            }
        } وإلا {
            لكل (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(عائم *)(dst_ptr + i * type_ناپ) =
                    *(ثابت عائم *)(src_صف + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            لكل (uint32_t i = 0; i < ne00; i++) {
                *(عائم *)(dst_ptr + ((size_t)lp0 + i) * type_ناپ) =
                    *(ثابت عائم *)(src_صف + (size_t)i * nb00);
            }
            لكل (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(عائم *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_ناپ) =
                    *(ثابت عائم *)(src_صف + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروع);

    FARF(بلند_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_شروع, صف_سرا);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ساكن فراغ pad_کم_فی_thread_hvx_circular_dma(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) انکڑے;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    لكل (uint32_t ir = صف_شروع, spad_idx = 0; ir < صف_سرا && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_ادھار + spad_idx * src_صف_ناپ_aligned;
        uint8_t * dst_spad_cur = dst_spad_ادھار + spad_idx * dst_صف_ناپ_aligned;

        dma_قطار_ڈالو(dma_q,
            dma_make_انکڑے(dst->انکڑے, dst_spad_cur),
            dst_صف_ناپ, dst_صف_ناپ_aligned, dst_صف_ناپ, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_صف(ir, ne1, ne2, &pi1, &pi2, &pi3);
        ثابت dma_addr_t src_انکڑے = pad_circ_src_صف_انکڑے(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_قطار_ڈالو(dma_q,
            dma_make_انکڑے(src_spad_cur, src_انکڑے),
            src_صف_ناپ_aligned, src_صف_ناپ, src_صف_ناپ, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = صف_شروع; ir < صف_سرا; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_صف(ir, ne1, ne2, &i1, &i2, &i3);
        ثابت dma_addr_t dst_انکڑے = dst->انکڑے + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        إذا (lp0 > 0) {
            uint8_t * dst_بایاں       = dst_spad_cur;
            ثابت uint8_t * src_بایاں = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_ناپ;
            إذا ((uint32_t)lp0 < 32) {
                حافظہ_نقل(dst_بایاں, src_بایاں, (size_t)lp0 * type_ناپ);
            } وإلا {
                hvx_copy_f32_uu(dst_بایاں, src_بایاں, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_ناپ;
            إذا ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } وإلا {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        إذا (rp0 > 0) {
            uint8_t * dst_دایاں = dst_spad_cur + ((size_t)lp0 + ne00) * type_ناپ;
            إذا ((uint32_t)rp0 < 32) {
                حافظہ_نقل(dst_دایاں, src_spad_cur, (size_t)rp0 * type_ناپ);
            } وإلا {
                إذا ((uintptr_t)dst_دایاں % VLEN == 0) {
                    hvx_copy_f32_aa(dst_دایاں, src_spad_cur, (uint32_t)rp0);
                } وإلا {
                    hvx_copy_f32_ua(dst_دایاں, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_قطار_ڈالو(dma_q,
            dma_make_انکڑے(dst_انکڑے, dst_spad_cur),
            dst_صف_ناپ, dst_صف_ناپ_aligned, dst_صف_ناپ, 1);

        ثابت uint32_t اگلا_صف = ir + 2;
        إذا (اگلا_صف < صف_سرا) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_صف(اگلا_صف, ne1, ne2, &nri1, &nri2, &nri3);
            ثابت dma_addr_t اگلا_src_انکڑے = pad_circ_src_صف_انکڑے(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_قطار_ڈالو(dma_q,
                dma_make_انکڑے(src_spad_cur, اگلا_src_انکڑے),
                src_صف_ناپ_aligned, src_صف_ناپ, src_صف_ناپ, 1);
        }
    }

    dma_قطار_flush(dma_q);

    FARF(بلند_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_شروع, صف_سرا);
}

صحيح op_pad(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_ناپ;
    بدل (src0->type) {
        حالة HTP_TYPE_F32: type_ناپ = 4; اكسر;
        افتراضي:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت int32_t lp0 = octx->op_params[0];
    ثابت int32_t rp0 = octx->op_params[1];
    ثابت int32_t lp1 = octx->op_params[2];
    ثابت int32_t rp1 = octx->op_params[3];
    ثابت int32_t lp2 = octx->op_params[4];
    ثابت int32_t rp2 = octx->op_params[5];
    ثابت int32_t lp3 = octx->op_params[6];
    ثابت int32_t rp3 = octx->op_params[7];
    ثابت int32_t circular = octx->op_params[8];

    ثابت uint32_t ne0  = dst->ne[0];
    ثابت uint32_t ne00 = src0->ne[0];

    ثابت uint32_t کل_dst_صفاں = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت size_t dst_صف_ناپ     = (size_t)ne0 * type_ناپ;

    uint32_t صف_شروع = 0;
    uint32_t nrows     = کل_dst_صفاں;

    إذا (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفاں_فی_chunk = 0;
        htp_tensor_mdev_صفاں_فی_chunk(dst, type_ناپ, (uint32_t) dst_صف_ناپ, &صفاں_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(کل_dst_صفاں, صفاں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        nrows     = range.گنتی;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    ثابت size_t src_صف_ناپ         = (size_t)ne00 * type_ناپ;
    ثابت size_t src_صف_ناپ_aligned = hex_گیڑا_اتے(src_صف_ناپ, VLEN);
    ثابت size_t dst_صف_ناپ_aligned = hex_گیڑا_اتے(dst_صف_ناپ, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    ثابت size_t vtcm_needed = (size_t)n_دھاگے * 2 * (src_صف_ناپ_aligned + dst_صف_ناپ_aligned);

    ثابت صحيح use_dma = (src0->nb[0] == (uint32_t)type_ناپ) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_ناپ >= vtcm_needed);

    إذا (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (use_dma) {
        octx->src0_spad.ناپ_فی_thread = 2 * src_صف_ناپ_aligned;
        octx->dst_spad.ناپ_فی_thread  = 2 * dst_صف_ناپ_aligned;
        octx->src0_spad.ناپ = n_دھاگے * octx->src0_spad.ناپ_فی_thread;
        octx->dst_spad.ناپ  = n_دھاگے * octx->dst_spad.ناپ_فی_thread;
        octx->src0_spad.انکڑے = octx->ctx->vtcm_ادھار;
        octx->dst_spad.انکڑے  = octx->src0_spad.انکڑے + octx->src0_spad.ناپ;
        octx->src0_spad.src  = عدم;
        octx->dst_spad.src   = عدم;
    }

    بنية htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .کل_dst_صفاں   = nrows,
        .صف_شروع        = صف_شروع,
        .type_ناپ        = type_ناپ,
        .src_صف_ناپ         = src_صف_ناپ,
        .src_صف_ناپ_aligned = src_صف_ناپ_aligned,
        .dst_صف_ناپ         = dst_صف_ناپ,
        .dst_صف_ناپ_aligned = dst_صف_ناپ_aligned,
    };

    FARF(بلند_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    إذا      (circular && use_dma) { work_قطار_لگاتار(octx->ctx->work_قطار, pad_کم_فی_thread_hvx_circular_dma, &pctx, n_دھاگے); }
    وإلا إذا (circular)            { work_قطار_لگاتار(octx->ctx->work_قطار, pad_کم_فی_thread_hvx_circular,     &pctx, n_دھاگے); }
    وإلا إذا (use_dma)             { work_قطار_لگاتار(octx->ctx->work_قطار, pad_کم_فی_thread_hvx_dma,          &pctx, n_دھاگے); }
    وإلا                          { work_قطار_لگاتار(octx->ctx->work_قطار, pad_کم_فی_thread_hvx,              &pctx, n_دھاگے); }

    أرجع HTP_حالت_OK;
}
