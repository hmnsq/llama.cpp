#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کن GGML_COMMON_DECL_C
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
ساكن مضمن فراغ pad_decompose_ردیف(uint32_t ir, uint32_t ne1, uint32_t ne2,
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
ساكن مضمن ثابت uint8_t * pad_src_ردیف_ptr(ثابت بنية htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع (ثابت uint8_t *) src->داده
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

ساكن مضمن dma_addr_t pad_src_ردیف_داده(ثابت بنية htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع src->داده
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
ساكن مضمن ثابت uint8_t * pad_circ_src_ردیف_ptr(ثابت بنية htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع (ثابت uint8_t *) src->داده
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ساكن مضمن dma_addr_t pad_circ_src_ردیف_داده(ثابت بنية htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    أرجع src->داده
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

    uint32_t nrows_در_هر_thread;
    uint32_t مجموع_dst_ردیفها;
    uint32_t ردیف_شروع;

    size_t   type_اندازه;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_ردیف_اندازه;
    size_t   src_ردیف_اندازه_aligned;
    size_t   dst_ردیف_اندازه;
    size_t   dst_ردیف_اندازه_aligned;
};

#تعریف_کن htp_pad_preamble                                                             \
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
    ثابت size_t type_اندازه = pctx->type_اندازه;                                        \
                                                                                     \
    ثابت uint32_t ردیف_شروع = pctx->ردیف_شروع + pctx->nrows_در_هر_thread * ith;       \
    ثابت uint32_t ردیف_انتها   = MIN(ردیف_شروع + pctx->nrows_در_هر_thread, pctx->ردیف_شروع + pctx->مجموع_dst_ردیفها);


#تعریف_کن htp_pad_dma_preamble                                                                \
    ثابت size_t src_ردیف_اندازه         = pctx->src_ردیف_اندازه;                                 \
    ثابت size_t src_ردیف_اندازه_aligned = pctx->src_ردیف_اندازه_aligned;                         \
    ثابت size_t dst_ردیف_اندازه         = pctx->dst_ردیف_اندازه;                                 \
    ثابت size_t dst_ردیف_اندازه_aligned = pctx->dst_ردیف_اندازه_aligned;                         \
                                                                                            \
    uint8_t * src_spad_پایه = octx->src0_spad.داده + ith * octx->src0_spad.اندازه_در_هر_thread; \
    uint8_t * dst_spad_پایه = octx->dst_spad.داده  + ith * octx->dst_spad.اندازه_در_هر_thread;  \
                                                                                            \
    dma_صف * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

ساكن فراغ pad_کار_در_هر_thread_hvx(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) داده;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ردیف_شروع);

    لكل (uint32_t dst_ردیف = ردیف_شروع; dst_ردیف < ردیف_انتها; dst_ردیف++) {
        uint32_t i1, i2, i3;
        pad_decompose_ردیف(dst_ردیف, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->داده + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ثابت صحيح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        إذا (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } وإلا {
            ثابت uint8_t * src_ptr = pad_src_ردیف_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            إذا (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_ردیف_شروع = dst_ptr + (size_t)lp0 * type_اندازه;
            إذا (nb00 == type_اندازه) {
                hvx_copy_f32_uu(dst_ردیف_شروع, src_ptr, ne00);
            } وإلا {
                لكل (uint32_t i = 0; i < ne00; i++) {
                    کپی_حافظه(dst_ردیف_شروع + i * type_اندازه,
                           src_ptr + (size_t)i * nb00,
                           type_اندازه);
                }
            }

            إذا (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_اندازه, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ردیف_شروع);

    FARF(بالا_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ردیف_شروع, ردیف_انتها);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ساكن فراغ pad_کار_در_هر_thread_hvx_dma(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) داده;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    لكل (uint32_t ir = ردیف_شروع, spad_idx = 0; ir < ردیف_انتها && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_پایه + spad_idx * src_ردیف_اندازه_aligned;
        uint8_t * dst_spad_cur = dst_spad_پایه + spad_idx * dst_ردیف_اندازه_aligned;

        dma_صف_افزودن(dma_q,
            dma_make_داده(dst->داده, dst_spad_cur),
            dst_ردیف_اندازه, dst_ردیف_اندازه_aligned, dst_ردیف_اندازه, 0);

        uint32_t i1, i2, i3;
        pad_decompose_ردیف(ir, ne1, ne2, &i1, &i2, &i3);
        ثابت صحيح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ثابت dma_addr_t src_داده = interior
            ? pad_src_ردیف_داده(src, i1, i2, i3, lp1, lp2, lp3) : src->داده;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_صف_افزودن(dma_q,
            dma_make_داده(src_spad_cur, src_داده),
            src_ردیف_اندازه_aligned, src_ردیف_اندازه, src_ردیف_اندازه, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = ردیف_شروع; ir < ردیف_انتها; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_صف_برداشت(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_صف_برداشت(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_ردیف(ir, ne1, ne2, &i1, &i2, &i3);

        ثابت dma_addr_t dst_داده = dst->داده + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ثابت صحيح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        إذا (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } وإلا {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_اندازه;

            إذا ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } وإلا {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_صف_افزودن(dma_q,
            dma_make_داده(dst_داده, dst_spad_cur),
            dst_ردیف_اندازه, dst_ردیف_اندازه_aligned, dst_ردیف_اندازه, 1);

        ثابت uint32_t بعدی_ردیف = ir + 2;
        إذا (بعدی_ردیف < ردیف_انتها) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_ردیف(بعدی_ردیف, ne1, ne2, &ni1, &ni2, &ni3);
            ثابت صحيح بعدی_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            ثابت dma_addr_t بعدی_src_داده = بعدی_interior
                ? pad_src_ردیف_داده(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->داده;

            dma_صف_افزودن(dma_q,
                dma_make_داده(src_spad_cur, بعدی_src_داده),
                src_ردیف_اندازه_aligned, src_ردیف_اندازه, src_ردیف_اندازه, بعدی_interior ? 1 : 0);
        }
    }

    dma_صف_flush(dma_q);

    FARF(بالا_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ردیف_شروع, ردیف_انتها);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

ساكن فراغ pad_کار_در_هر_thread_hvx_circular(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) داده;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, ردیف_شروع);

    لكل (uint32_t dst_ردیف = ردیف_شروع; dst_ردیف < ردیف_انتها; dst_ردیف++) {
        uint32_t i1, i2, i3;
        pad_decompose_ردیف(dst_ردیف, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->داده + i1 * nb1 + i2 * nb2 + i3 * nb3;
        ثابت uint8_t * src_ردیف = pad_circ_src_ردیف_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        إذا (nb00 == type_اندازه) {

            إذا (lp0 > 0) {
                إذا ((uint32_t)lp0 < 32) {
                    کپی_حافظه(dst_ptr,
                           src_ردیف + (size_t)(ne00 - (uint32_t)lp0) * type_اندازه,
                           (size_t)lp0 * type_اندازه);
                } وإلا {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_ردیف + (size_t)(ne00 - (uint32_t)lp0) * type_اندازه,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_اندازه, src_ردیف, ne00);
            إذا (rp0 > 0) {
                إذا ((uint32_t)rp0 < 32) {
                    کپی_حافظه(dst_ptr + ((size_t)lp0 + ne00) * type_اندازه,
                           src_ردیف,
                           (size_t)rp0 * type_اندازه);
                } وإلا {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_اندازه,
                                    src_ردیف,
                                    (uint32_t)rp0);
                }
            }
        } وإلا {
            لكل (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(عائم *)(dst_ptr + i * type_اندازه) =
                    *(ثابت عائم *)(src_ردیف + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            لكل (uint32_t i = 0; i < ne00; i++) {
                *(عائم *)(dst_ptr + ((size_t)lp0 + i) * type_اندازه) =
                    *(ثابت عائم *)(src_ردیف + (size_t)i * nb00);
            }
            لكل (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(عائم *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_اندازه) =
                    *(ثابت عائم *)(src_ردیف + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ردیف_شروع);

    FARF(بالا_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ردیف_شروع, ردیف_انتها);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ساكن فراغ pad_کار_در_هر_thread_hvx_circular_dma(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    ثابت بنية htp_pad_context * pctx = (ثابت بنية htp_pad_context *) داده;
    بنية htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    لكل (uint32_t ir = ردیف_شروع, spad_idx = 0; ir < ردیف_انتها && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_پایه + spad_idx * src_ردیف_اندازه_aligned;
        uint8_t * dst_spad_cur = dst_spad_پایه + spad_idx * dst_ردیف_اندازه_aligned;

        dma_صف_افزودن(dma_q,
            dma_make_داده(dst->داده, dst_spad_cur),
            dst_ردیف_اندازه, dst_ردیف_اندازه_aligned, dst_ردیف_اندازه, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_ردیف(ir, ne1, ne2, &pi1, &pi2, &pi3);
        ثابت dma_addr_t src_داده = pad_circ_src_ردیف_داده(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_صف_افزودن(dma_q,
            dma_make_داده(src_spad_cur, src_داده),
            src_ردیف_اندازه_aligned, src_ردیف_اندازه, src_ردیف_اندازه, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = ردیف_شروع; ir < ردیف_انتها; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_صف_برداشت(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_صف_برداشت(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_ردیف(ir, ne1, ne2, &i1, &i2, &i3);
        ثابت dma_addr_t dst_داده = dst->داده + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        إذا (lp0 > 0) {
            uint8_t * dst_چپ       = dst_spad_cur;
            ثابت uint8_t * src_چپ = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_اندازه;
            إذا ((uint32_t)lp0 < 32) {
                کپی_حافظه(dst_چپ, src_چپ, (size_t)lp0 * type_اندازه);
            } وإلا {
                hvx_copy_f32_uu(dst_چپ, src_چپ, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_اندازه;
            إذا ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } وإلا {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        إذا (rp0 > 0) {
            uint8_t * dst_راست = dst_spad_cur + ((size_t)lp0 + ne00) * type_اندازه;
            إذا ((uint32_t)rp0 < 32) {
                کپی_حافظه(dst_راست, src_spad_cur, (size_t)rp0 * type_اندازه);
            } وإلا {
                إذا ((uintptr_t)dst_راست % VLEN == 0) {
                    hvx_copy_f32_aa(dst_راست, src_spad_cur, (uint32_t)rp0);
                } وإلا {
                    hvx_copy_f32_ua(dst_راست, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_صف_افزودن(dma_q,
            dma_make_داده(dst_داده, dst_spad_cur),
            dst_ردیف_اندازه, dst_ردیف_اندازه_aligned, dst_ردیف_اندازه, 1);

        ثابت uint32_t بعدی_ردیف = ir + 2;
        إذا (بعدی_ردیف < ردیف_انتها) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_ردیف(بعدی_ردیف, ne1, ne2, &nri1, &nri2, &nri3);
            ثابت dma_addr_t بعدی_src_داده = pad_circ_src_ردیف_داده(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_صف_افزودن(dma_q,
                dma_make_داده(src_spad_cur, بعدی_src_داده),
                src_ردیف_اندازه_aligned, src_ردیف_اندازه, src_ردیف_اندازه, 1);
        }
    }

    dma_صف_flush(dma_q);

    FARF(بالا_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         ردیف_شروع, ردیف_انتها);
}

صحيح op_pad(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_اندازه;
    بدل (src0->type) {
        حالة HTP_TYPE_F32: type_اندازه = 4; اكسر;
        افتراضي:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            أرجع HTP_وضعیت_NO_SUPPORT;
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

    ثابت uint32_t مجموع_dst_ردیفها = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت size_t dst_ردیف_اندازه     = (size_t)ne0 * type_اندازه;

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

    ثابت size_t src_ردیف_اندازه         = (size_t)ne00 * type_اندازه;
    ثابت size_t src_ردیف_اندازه_aligned = hex_دور_به_بالا(src_ردیف_اندازه, VLEN);
    ثابت size_t dst_ردیف_اندازه_aligned = hex_دور_به_بالا(dst_ردیف_اندازه, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    ثابت size_t vtcm_needed = (size_t)n_نخها * 2 * (src_ردیف_اندازه_aligned + dst_ردیف_اندازه_aligned);

    ثابت صحيح use_dma = (src0->nb[0] == (uint32_t)type_اندازه) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_اندازه >= vtcm_needed);

    إذا (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (use_dma) {
        octx->src0_spad.اندازه_در_هر_thread = 2 * src_ردیف_اندازه_aligned;
        octx->dst_spad.اندازه_در_هر_thread  = 2 * dst_ردیف_اندازه_aligned;
        octx->src0_spad.اندازه = n_نخها * octx->src0_spad.اندازه_در_هر_thread;
        octx->dst_spad.اندازه  = n_نخها * octx->dst_spad.اندازه_در_هر_thread;
        octx->src0_spad.داده = octx->ctx->vtcm_پایه;
        octx->dst_spad.داده  = octx->src0_spad.داده + octx->src0_spad.اندازه;
        octx->src0_spad.src  = عدم;
        octx->dst_spad.src   = عدم;
    }

    بنية htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_در_هر_thread = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div),
        .مجموع_dst_ردیفها   = nrows,
        .ردیف_شروع        = ردیف_شروع,
        .type_اندازه        = type_اندازه,
        .src_ردیف_اندازه         = src_ردیف_اندازه,
        .src_ردیف_اندازه_aligned = src_ردیف_اندازه_aligned,
        .dst_ردیف_اندازه         = dst_ردیف_اندازه,
        .dst_ردیف_اندازه_aligned = dst_ردیف_اندازه_aligned,
    };

    FARF(بالا_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    إذا      (circular && use_dma) { work_صف_پیاپی(octx->ctx->work_صف, pad_کار_در_هر_thread_hvx_circular_dma, &pctx, n_نخها); }
    وإلا إذا (circular)            { work_صف_پیاپی(octx->ctx->work_صف, pad_کار_در_هر_thread_hvx_circular,     &pctx, n_نخها); }
    وإلا إذا (use_dma)             { work_صف_پیاپی(octx->ctx->work_صف, pad_کار_در_هر_thread_hvx_dma,          &pctx, n_نخها); }
    وإلا                          { work_صف_پیاپی(octx->ctx->work_صف, pad_کار_در_هر_thread_hvx,              &pctx, n_نخها); }

    أرجع HTP_وضعیت_OK;
}
