#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#وصف_ڪريو GGML_COMMON_DECL_C
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
    أرجع (ثابت uint8_t *) src->معلومات
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

ساكن مضمن dma_addr_t pad_src_صف_معلومات(ثابت بنية htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع src->معلومات
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
ساكن مضمن ثابت uint8_t * pad_circ_src_صف_ptr(ثابت بنية htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع (ثابت uint8_t *) src->معلومات
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ساكن مضمن dma_addr_t pad_circ_src_صف_معلومات(ثابت بنية htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع src->معلومات
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

    uint32_t nrows_في_thread;
    uint32_t ڪل_dst_صفون;
    uint32_t صف_شروعات;

    size_t   type_سائيز;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_صف_سائيز;
    size_t   src_صف_سائيز_aligned;
    size_t   dst_صف_سائيز;
    size_t   dst_صف_سائيز_aligned;
};

#وصف_ڪريو htp_pad_preamble                                                             \
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
    ثابت size_t type_سائيز = pctx->type_سائيز;                                        \
                                                                                     \
    ثابت uint32_t صف_شروعات = pctx->صف_شروعات + pctx->nrows_في_thread * ith;       \
    ثابت uint32_t صف_پڇاڙي   = MIN(صف_شروعات + pctx->nrows_في_thread, pctx->صف_شروعات + pctx->ڪل_dst_صفون);


#وصف_ڪريو htp_pad_dma_preamble                                                                \
    ثابت size_t src_صف_سائيز         = pctx->src_صف_سائيز;                                 \
    ثابت size_t src_صف_سائيز_aligned = pctx->src_صف_سائيز_aligned;                         \
    ثابت size_t dst_صف_سائيز         = pctx->dst_صف_سائيز;                                 \
    ثابت size_t dst_صف_سائيز_aligned = pctx->dst_صف_سائيز_aligned;                         \
                                                                                            \
    uint8_t * src_spad_بنياد = octx->src0_spad.معلومات + ith * octx->src0_spad.سائيز_في_thread; \
    uint8_t * dst_spad_بنياد = octx->dst_spad.معلومات  + ith * octx->dst_spad.سائيز_في_thread;  \
                                                                                            \
    dma_قطار * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

ساكن فراغ pad_ڪم_في_thread_hvx(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) معلومات;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروعات);

    لكل (uint32_t dst_صف = صف_شروعات; dst_صف < صف_پڇاڙي; dst_صف++) {
        uint32_t i1, i2, i3;
        pad_decompose_صف(dst_صف, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->معلومات + i1 * nb1 + i2 * nb2 + i3 * nb3;

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

            uint8_t * dst_صف_شروعات = dst_ptr + (size_t)lp0 * type_سائيز;
            إذا (nb00 == type_سائيز) {
                hvx_copy_f32_uu(dst_صف_شروعات, src_ptr, ne00);
            } وإلا {
                لكل (uint32_t i = 0; i < ne00; i++) {
                    يادگيري_نقل(dst_صف_شروعات + i * type_سائيز,
                           src_ptr + (size_t)i * nb00,
                           type_سائيز);
                }
            }

            إذا (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_سائيز, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروعات);

    FARF(مٿاهون_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_شروعات, صف_پڇاڙي);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ساكن فراغ pad_ڪم_في_thread_hvx_dma(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) معلومات;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    لكل (uint32_t ir = صف_شروعات, spad_idx = 0; ir < صف_پڇاڙي && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_بنياد + spad_idx * src_صف_سائيز_aligned;
        uint8_t * dst_spad_cur = dst_spad_بنياد + spad_idx * dst_صف_سائيز_aligned;

        dma_قطار_ڌڪو(dma_q,
            dma_make_معلومات(dst->معلومات, dst_spad_cur),
            dst_صف_سائيز, dst_صف_سائيز_aligned, dst_صف_سائيز, 0);

        uint32_t i1, i2, i3;
        pad_decompose_صف(ir, ne1, ne2, &i1, &i2, &i3);
        ثابت صحيح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ثابت dma_addr_t src_معلومات = interior
            ? pad_src_صف_معلومات(src, i1, i2, i3, lp1, lp2, lp3) : src->معلومات;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_قطار_ڌڪو(dma_q,
            dma_make_معلومات(src_spad_cur, src_معلومات),
            src_صف_سائيز_aligned, src_صف_سائيز, src_صف_سائيز, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = صف_شروعات; ir < صف_پڇاڙي; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_صف(ir, ne1, ne2, &i1, &i2, &i3);

        ثابت dma_addr_t dst_معلومات = dst->معلومات + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ثابت صحيح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        إذا (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } وإلا {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_سائيز;

            إذا ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } وإلا {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_قطار_ڌڪو(dma_q,
            dma_make_معلومات(dst_معلومات, dst_spad_cur),
            dst_صف_سائيز, dst_صف_سائيز_aligned, dst_صف_سائيز, 1);

        ثابت uint32_t ايندڙ_صف = ir + 2;
        إذا (ايندڙ_صف < صف_پڇاڙي) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_صف(ايندڙ_صف, ne1, ne2, &ni1, &ni2, &ni3);
            ثابت صحيح ايندڙ_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            ثابت dma_addr_t ايندڙ_src_معلومات = ايندڙ_interior
                ? pad_src_صف_معلومات(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->معلومات;

            dma_قطار_ڌڪو(dma_q,
                dma_make_معلومات(src_spad_cur, ايندڙ_src_معلومات),
                src_صف_سائيز_aligned, src_صف_سائيز, src_صف_سائيز, ايندڙ_interior ? 1 : 0);
        }
    }

    dma_قطار_flush(dma_q);

    FARF(مٿاهون_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_شروعات, صف_پڇاڙي);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

ساكن فراغ pad_ڪم_في_thread_hvx_circular(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) معلومات;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروعات);

    لكل (uint32_t dst_صف = صف_شروعات; dst_صف < صف_پڇاڙي; dst_صف++) {
        uint32_t i1, i2, i3;
        pad_decompose_صف(dst_صف, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->معلومات + i1 * nb1 + i2 * nb2 + i3 * nb3;
        ثابت uint8_t * src_صف = pad_circ_src_صف_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        إذا (nb00 == type_سائيز) {

            إذا (lp0 > 0) {
                إذا ((uint32_t)lp0 < 32) {
                    يادگيري_نقل(dst_ptr,
                           src_صف + (size_t)(ne00 - (uint32_t)lp0) * type_سائيز,
                           (size_t)lp0 * type_سائيز);
                } وإلا {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_صف + (size_t)(ne00 - (uint32_t)lp0) * type_سائيز,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_سائيز, src_صف, ne00);
            إذا (rp0 > 0) {
                إذا ((uint32_t)rp0 < 32) {
                    يادگيري_نقل(dst_ptr + ((size_t)lp0 + ne00) * type_سائيز,
                           src_صف,
                           (size_t)rp0 * type_سائيز);
                } وإلا {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_سائيز,
                                    src_صف,
                                    (uint32_t)rp0);
                }
            }
        } وإلا {
            لكل (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(عائم *)(dst_ptr + i * type_سائيز) =
                    *(ثابت عائم *)(src_صف + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            لكل (uint32_t i = 0; i < ne00; i++) {
                *(عائم *)(dst_ptr + ((size_t)lp0 + i) * type_سائيز) =
                    *(ثابت عائم *)(src_صف + (size_t)i * nb00);
            }
            لكل (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(عائم *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_سائيز) =
                    *(ثابت عائم *)(src_صف + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_شروعات);

    FARF(مٿاهون_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_شروعات, صف_پڇاڙي);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ساكن فراغ pad_ڪم_في_thread_hvx_circular_dma(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) معلومات;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    لكل (uint32_t ir = صف_شروعات, spad_idx = 0; ir < صف_پڇاڙي && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_بنياد + spad_idx * src_صف_سائيز_aligned;
        uint8_t * dst_spad_cur = dst_spad_بنياد + spad_idx * dst_صف_سائيز_aligned;

        dma_قطار_ڌڪو(dma_q,
            dma_make_معلومات(dst->معلومات, dst_spad_cur),
            dst_صف_سائيز, dst_صف_سائيز_aligned, dst_صف_سائيز, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_صف(ir, ne1, ne2, &pi1, &pi2, &pi3);
        ثابت dma_addr_t src_معلومات = pad_circ_src_صف_معلومات(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_قطار_ڌڪو(dma_q,
            dma_make_معلومات(src_spad_cur, src_معلومات),
            src_صف_سائيز_aligned, src_صف_سائيز, src_صف_سائيز, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = صف_شروعات; ir < صف_پڇاڙي; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_صف(ir, ne1, ne2, &i1, &i2, &i3);
        ثابت dma_addr_t dst_معلومات = dst->معلومات + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        إذا (lp0 > 0) {
            uint8_t * dst_کاٻو       = dst_spad_cur;
            ثابت uint8_t * src_کاٻو = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_سائيز;
            إذا ((uint32_t)lp0 < 32) {
                يادگيري_نقل(dst_کاٻو, src_کاٻو, (size_t)lp0 * type_سائيز);
            } وإلا {
                hvx_copy_f32_uu(dst_کاٻو, src_کاٻو, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_سائيز;
            إذا ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } وإلا {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        إذا (rp0 > 0) {
            uint8_t * dst_ساڄو = dst_spad_cur + ((size_t)lp0 + ne00) * type_سائيز;
            إذا ((uint32_t)rp0 < 32) {
                يادگيري_نقل(dst_ساڄو, src_spad_cur, (size_t)rp0 * type_سائيز);
            } وإلا {
                إذا ((uintptr_t)dst_ساڄو % VLEN == 0) {
                    hvx_copy_f32_aa(dst_ساڄو, src_spad_cur, (uint32_t)rp0);
                } وإلا {
                    hvx_copy_f32_ua(dst_ساڄو, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_قطار_ڌڪو(dma_q,
            dma_make_معلومات(dst_معلومات, dst_spad_cur),
            dst_صف_سائيز, dst_صف_سائيز_aligned, dst_صف_سائيز, 1);

        ثابت uint32_t ايندڙ_صف = ir + 2;
        إذا (ايندڙ_صف < صف_پڇاڙي) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_صف(ايندڙ_صف, ne1, ne2, &nri1, &nri2, &nri3);
            ثابت dma_addr_t ايندڙ_src_معلومات = pad_circ_src_صف_معلومات(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_قطار_ڌڪو(dma_q,
                dma_make_معلومات(src_spad_cur, ايندڙ_src_معلومات),
                src_صف_سائيز_aligned, src_صف_سائيز, src_صف_سائيز, 1);
        }
    }

    dma_قطار_flush(dma_q);

    FARF(مٿاهون_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_شروعات, صف_پڇاڙي);
}

صحيح op_pad(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_سائيز;
    بدل (src0->type) {
        حالة HTP_TYPE_F32: type_سائيز = 4; اكسر;
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

    ثابت uint32_t ڪل_dst_صفون = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت size_t dst_صف_سائيز     = (size_t)ne0 * type_سائيز;

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

    ثابت size_t src_صف_سائيز         = (size_t)ne00 * type_سائيز;
    ثابت size_t src_صف_سائيز_aligned = hex_ڦيرو_مٿي(src_صف_سائيز, VLEN);
    ثابت size_t dst_صف_سائيز_aligned = hex_ڦيرو_مٿي(dst_صف_سائيز, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    ثابت size_t vtcm_needed = (size_t)n_ڌاڳا * 2 * (src_صف_سائيز_aligned + dst_صف_سائيز_aligned);

    ثابت صحيح use_dma = (src0->nb[0] == (uint32_t)type_سائيز) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_سائيز >= vtcm_needed);

    إذا (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (use_dma) {
        octx->src0_spad.سائيز_في_thread = 2 * src_صف_سائيز_aligned;
        octx->dst_spad.سائيز_في_thread  = 2 * dst_صف_سائيز_aligned;
        octx->src0_spad.سائيز = n_ڌاڳا * octx->src0_spad.سائيز_في_thread;
        octx->dst_spad.سائيز  = n_ڌاڳا * octx->dst_spad.سائيز_في_thread;
        octx->src0_spad.معلومات = octx->ctx->vtcm_بنياد;
        octx->dst_spad.معلومات  = octx->src0_spad.معلومات + octx->src0_spad.سائيز;
        octx->src0_spad.src  = عدم;
        octx->dst_spad.src   = عدم;
    }

    بنية htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_في_thread = fastdiv(nrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div),
        .ڪل_dst_صفون   = nrows,
        .صف_شروعات        = صف_شروعات,
        .type_سائيز        = type_سائيز,
        .src_صف_سائيز         = src_صف_سائيز,
        .src_صف_سائيز_aligned = src_صف_سائيز_aligned,
        .dst_صف_سائيز         = dst_صف_سائيز,
        .dst_صف_سائيز_aligned = dst_صف_سائيز_aligned,
    };

    FARF(مٿاهون_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    إذا      (circular && use_dma) { work_قطار_لاڳيتو(octx->ctx->work_قطار, pad_ڪم_في_thread_hvx_circular_dma, &pctx, n_ڌاڳا); }
    وإلا إذا (circular)            { work_قطار_لاڳيتو(octx->ctx->work_قطار, pad_ڪم_في_thread_hvx_circular,     &pctx, n_ڌاڳا); }
    وإلا إذا (use_dma)             { work_قطار_لاڳيتو(octx->ctx->work_قطار, pad_ڪم_في_thread_hvx_dma,          &pctx, n_ڌاڳا); }
    وإلا                          { work_قطار_لاڳيتو(octx->ctx->work_قطار, pad_ڪم_في_thread_hvx,              &pctx, n_ڌاڳا); }

    أرجع HTP_حالت_OK;
}
