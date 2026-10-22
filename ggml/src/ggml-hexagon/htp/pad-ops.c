#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <string.h>

#شامل "dma-queue.h"
#شامل "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"

/* Circular wrap: maps any integer x into [0, n) */
ساکن اندرونی uint32_t wrap_around(int32_t x, uint32_t n) {
    واپس (uint32_t)(((x % (int32_t)n) + (int32_t)n) % (int32_t)n);
}

/* Decompose a flat dst row index into (i1, i2, i3) */
ساکن اندرونی خالی pad_decompose_صف(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     uint32_t *i1, uint32_t *i2, uint32_t *i3) {
    *i1 = ir % ne1;
    *i2 = (ir / ne1) % ne2;
    *i3 = ir / (ne1 * ne2);
}

/* Return non-zero if row (i1,i2,i3) falls in the non-padded interior */
ساکن اندرونی صحیح pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                   int32_t lp1, int32_t rp1, uint32_t ne1,
                                   int32_t lp2, int32_t rp2, uint32_t ne2,
                                   int32_t lp3, int32_t rp3, uint32_t ne3) {
    واپس ((int32_t)i1 >= lp1 && (int32_t)i1 < (int32_t)ne1 - rp1) &&
           ((int32_t)i2 >= lp2 && (int32_t)i2 < (int32_t)ne2 - rp2) &&
           ((int32_t)i3 >= lp3 && (int32_t)i3 < (int32_t)ne3 - rp3);
}

/* Compute the DDR src row pointer for a zero-pad interior row */
ساکن اندرونی ثابت uint8_t * pad_src_صف_ptr(ثابت ساخت htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    واپس (ثابت uint8_t *) src->ڈیٹا
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

ساکن اندرونی dma_addr_t pad_src_صف_ڈیٹا(ثابت ساخت htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    واپس src->ڈیٹا
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
ساکن اندرونی ثابت uint8_t * pad_circ_src_صف_ptr(ثابت ساخت htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    واپس (ثابت uint8_t *) src->ڈیٹا
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ساکن اندرونی dma_addr_t pad_circ_src_صف_ڈیٹا(ثابت ساخت htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    واپس src->ڈیٹا
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ساخت htp_pad_context {
    ساخت htp_ops_context * octx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_فی_thread;
    uint32_t کل_dst_صفیں;
    uint32_t صف_آغاز;

    size_t   type_سائز;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_صف_سائز;
    size_t   src_صف_سائز_aligned;
    size_t   dst_صف_سائز;
    size_t   dst_صف_سائز_aligned;
};

#تعریف_کرو htp_pad_preamble                                                             \
    ثابت ساخت htp_tensor * src = octx->src[0];                                    \
    ثابت ساخت htp_tensor * dst = octx->dst;                                       \
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
    ثابت size_t type_سائز = pctx->type_سائز;                                        \
                                                                                     \
    ثابت uint32_t صف_آغاز = pctx->صف_آغاز + pctx->nrows_فی_thread * ith;       \
    ثابت uint32_t صف_انتہا   = MIN(صف_آغاز + pctx->nrows_فی_thread, pctx->صف_آغاز + pctx->کل_dst_صفیں);


#تعریف_کرو htp_pad_dma_preamble                                                                \
    ثابت size_t src_صف_سائز         = pctx->src_صف_سائز;                                 \
    ثابت size_t src_صف_سائز_aligned = pctx->src_صف_سائز_aligned;                         \
    ثابت size_t dst_صف_سائز         = pctx->dst_صف_سائز;                                 \
    ثابت size_t dst_صف_سائز_aligned = pctx->dst_صف_سائز_aligned;                         \
                                                                                            \
    uint8_t * src_spad_بنیاد = octx->src0_spad.ڈیٹا + ith * octx->src0_spad.سائز_فی_thread; \
    uint8_t * dst_spad_بنیاد = octx->dst_spad.ڈیٹا  + ith * octx->dst_spad.سائز_فی_thread;  \
                                                                                            \
    dma_قطار * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

ساکن خالی pad_کام_فی_thread_hvx(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ثابت ساخت htp_pad_context * pctx = (ثابت ساخت htp_pad_context *) ڈیٹا;
    ساخت htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, صف_آغاز);

    برائے (uint32_t dst_صف = صف_آغاز; dst_صف < صف_انتہا; dst_صف++) {
        uint32_t i1, i2, i3;
        pad_decompose_صف(dst_صف, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->ڈیٹا + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ثابت صحیح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        اگر (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } ورنہ {
            ثابت uint8_t * src_ptr = pad_src_صف_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            اگر (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_صف_آغاز = dst_ptr + (size_t)lp0 * type_سائز;
            اگر (nb00 == type_سائز) {
                hvx_copy_f32_uu(dst_صف_آغاز, src_ptr, ne00);
            } ورنہ {
                برائے (uint32_t i = 0; i < ne00; i++) {
                    حافظہ_نقل(dst_صف_آغاز + i * type_سائز,
                           src_ptr + (size_t)i * nb00,
                           type_سائز);
                }
            }

            اگر (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_سائز, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_آغاز);

    FARF(بلند_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_آغاز, صف_انتہا);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ساکن خالی pad_کام_فی_thread_hvx_dma(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ثابت ساخت htp_pad_context * pctx = (ثابت ساخت htp_pad_context *) ڈیٹا;
    ساخت htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    برائے (uint32_t ir = صف_آغاز, spad_idx = 0; ir < صف_انتہا && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_بنیاد + spad_idx * src_صف_سائز_aligned;
        uint8_t * dst_spad_cur = dst_spad_بنیاد + spad_idx * dst_صف_سائز_aligned;

        dma_قطار_ڈالو(dma_q,
            dma_make_ڈیٹا(dst->ڈیٹا, dst_spad_cur),
            dst_صف_سائز, dst_صف_سائز_aligned, dst_صف_سائز, 0);

        uint32_t i1, i2, i3;
        pad_decompose_صف(ir, ne1, ne2, &i1, &i2, &i3);
        ثابت صحیح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ثابت dma_addr_t src_ڈیٹا = interior
            ? pad_src_صف_ڈیٹا(src, i1, i2, i3, lp1, lp2, lp3) : src->ڈیٹا;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_قطار_ڈالو(dma_q,
            dma_make_ڈیٹا(src_spad_cur, src_ڈیٹا),
            src_صف_سائز_aligned, src_صف_سائز, src_صف_سائز, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    برائے (uint32_t ir = صف_آغاز; ir < صف_انتہا; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_صف(ir, ne1, ne2, &i1, &i2, &i3);

        ثابت dma_addr_t dst_ڈیٹا = dst->ڈیٹا + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ثابت صحیح interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        اگر (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } ورنہ {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_سائز;

            اگر ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } ورنہ {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_قطار_ڈالو(dma_q,
            dma_make_ڈیٹا(dst_ڈیٹا, dst_spad_cur),
            dst_صف_سائز, dst_صف_سائز_aligned, dst_صف_سائز, 1);

        ثابت uint32_t اگلا_صف = ir + 2;
        اگر (اگلا_صف < صف_انتہا) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_صف(اگلا_صف, ne1, ne2, &ni1, &ni2, &ni3);
            ثابت صحیح اگلا_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            ثابت dma_addr_t اگلا_src_ڈیٹا = اگلا_interior
                ? pad_src_صف_ڈیٹا(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->ڈیٹا;

            dma_قطار_ڈالو(dma_q,
                dma_make_ڈیٹا(src_spad_cur, اگلا_src_ڈیٹا),
                src_صف_سائز_aligned, src_صف_سائز, src_صف_سائز, اگلا_interior ? 1 : 0);
        }
    }

    dma_قطار_flush(dma_q);

    FARF(بلند_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_آغاز, صف_انتہا);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

ساکن خالی pad_کام_فی_thread_hvx_circular(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ثابت ساخت htp_pad_context * pctx = (ثابت ساخت htp_pad_context *) ڈیٹا;
    ساخت htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, صف_آغاز);

    برائے (uint32_t dst_صف = صف_آغاز; dst_صف < صف_انتہا; dst_صف++) {
        uint32_t i1, i2, i3;
        pad_decompose_صف(dst_صف, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->ڈیٹا + i1 * nb1 + i2 * nb2 + i3 * nb3;
        ثابت uint8_t * src_صف = pad_circ_src_صف_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        اگر (nb00 == type_سائز) {

            اگر (lp0 > 0) {
                اگر ((uint32_t)lp0 < 32) {
                    حافظہ_نقل(dst_ptr,
                           src_صف + (size_t)(ne00 - (uint32_t)lp0) * type_سائز,
                           (size_t)lp0 * type_سائز);
                } ورنہ {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_صف + (size_t)(ne00 - (uint32_t)lp0) * type_سائز,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_سائز, src_صف, ne00);
            اگر (rp0 > 0) {
                اگر ((uint32_t)rp0 < 32) {
                    حافظہ_نقل(dst_ptr + ((size_t)lp0 + ne00) * type_سائز,
                           src_صف,
                           (size_t)rp0 * type_سائز);
                } ورنہ {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_سائز,
                                    src_صف,
                                    (uint32_t)rp0);
                }
            }
        } ورنہ {
            برائے (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(اعشاری *)(dst_ptr + i * type_سائز) =
                    *(ثابت اعشاری *)(src_صف + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            برائے (uint32_t i = 0; i < ne00; i++) {
                *(اعشاری *)(dst_ptr + ((size_t)lp0 + i) * type_سائز) =
                    *(ثابت اعشاری *)(src_صف + (size_t)i * nb00);
            }
            برائے (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(اعشاری *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_سائز) =
                    *(ثابت اعشاری *)(src_صف + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, صف_آغاز);

    FARF(بلند_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_آغاز, صف_انتہا);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ساکن خالی pad_کام_فی_thread_hvx_circular_dma(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ثابت ساخت htp_pad_context * pctx = (ثابت ساخت htp_pad_context *) ڈیٹا;
    ساخت htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    برائے (uint32_t ir = صف_آغاز, spad_idx = 0; ir < صف_انتہا && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_بنیاد + spad_idx * src_صف_سائز_aligned;
        uint8_t * dst_spad_cur = dst_spad_بنیاد + spad_idx * dst_صف_سائز_aligned;

        dma_قطار_ڈالو(dma_q,
            dma_make_ڈیٹا(dst->ڈیٹا, dst_spad_cur),
            dst_صف_سائز, dst_صف_سائز_aligned, dst_صف_سائز, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_صف(ir, ne1, ne2, &pi1, &pi2, &pi3);
        ثابت dma_addr_t src_ڈیٹا = pad_circ_src_صف_ڈیٹا(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_قطار_ڈالو(dma_q,
            dma_make_ڈیٹا(src_spad_cur, src_ڈیٹا),
            src_صف_سائز_aligned, src_صف_سائز, src_صف_سائز, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    برائے (uint32_t ir = صف_آغاز; ir < صف_انتہا; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_صف(ir, ne1, ne2, &i1, &i2, &i3);
        ثابت dma_addr_t dst_ڈیٹا = dst->ڈیٹا + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        اگر (lp0 > 0) {
            uint8_t * dst_بایاں       = dst_spad_cur;
            ثابت uint8_t * src_بایاں = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_سائز;
            اگر ((uint32_t)lp0 < 32) {
                حافظہ_نقل(dst_بایاں, src_بایاں, (size_t)lp0 * type_سائز);
            } ورنہ {
                hvx_copy_f32_uu(dst_بایاں, src_بایاں, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_سائز;
            اگر ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } ورنہ {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        اگر (rp0 > 0) {
            uint8_t * dst_دایاں = dst_spad_cur + ((size_t)lp0 + ne00) * type_سائز;
            اگر ((uint32_t)rp0 < 32) {
                حافظہ_نقل(dst_دایاں, src_spad_cur, (size_t)rp0 * type_سائز);
            } ورنہ {
                اگر ((uintptr_t)dst_دایاں % VLEN == 0) {
                    hvx_copy_f32_aa(dst_دایاں, src_spad_cur, (uint32_t)rp0);
                } ورنہ {
                    hvx_copy_f32_ua(dst_دایاں, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_قطار_ڈالو(dma_q,
            dma_make_ڈیٹا(dst_ڈیٹا, dst_spad_cur),
            dst_صف_سائز, dst_صف_سائز_aligned, dst_صف_سائز, 1);

        ثابت uint32_t اگلا_صف = ir + 2;
        اگر (اگلا_صف < صف_انتہا) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_صف(اگلا_صف, ne1, ne2, &nri1, &nri2, &nri3);
            ثابت dma_addr_t اگلا_src_ڈیٹا = pad_circ_src_صف_ڈیٹا(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_قطار_ڈالو(dma_q,
                dma_make_ڈیٹا(src_spad_cur, اگلا_src_ڈیٹا),
                src_صف_سائز_aligned, src_صف_سائز, src_صف_سائز, 1);
        }
    }

    dma_قطار_flush(dma_q);

    FARF(بلند_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         صف_آغاز, صف_انتہا);
}

صحیح op_pad(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_سائز;
    انتخاب (src0->type) {
        حالت HTP_TYPE_F32: type_سائز = 4; توڑو;
        طےشدہ:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            واپس HTP_حالت_NO_SUPPORT;
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

    ثابت uint32_t کل_dst_صفیں = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت size_t dst_صف_سائز     = (size_t)ne0 * type_سائز;

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = کل_dst_صفیں;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, type_سائز, (uint32_t) dst_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_dst_صفیں, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    ثابت size_t src_صف_سائز         = (size_t)ne00 * type_سائز;
    ثابت size_t src_صف_سائز_aligned = hex_باری_اوپر(src_صف_سائز, VLEN);
    ثابت size_t dst_صف_سائز_aligned = hex_باری_اوپر(dst_صف_سائز, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    ثابت size_t vtcm_needed = (size_t)n_دھاگے * 2 * (src_صف_سائز_aligned + dst_صف_سائز_aligned);

    ثابت صحیح use_dma = (src0->nb[0] == (uint32_t)type_سائز) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_سائز >= vtcm_needed);

    اگر (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (use_dma) {
        octx->src0_spad.سائز_فی_thread = 2 * src_صف_سائز_aligned;
        octx->dst_spad.سائز_فی_thread  = 2 * dst_صف_سائز_aligned;
        octx->src0_spad.سائز = n_دھاگے * octx->src0_spad.سائز_فی_thread;
        octx->dst_spad.سائز  = n_دھاگے * octx->dst_spad.سائز_فی_thread;
        octx->src0_spad.ڈیٹا = octx->ctx->vtcm_بنیاد;
        octx->dst_spad.ڈیٹا  = octx->src0_spad.ڈیٹا + octx->src0_spad.سائز;
        octx->src0_spad.src  = تہی;
        octx->dst_spad.src   = تہی;
    }

    ساخت htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .کل_dst_صفیں   = nrows,
        .صف_آغاز        = صف_آغاز,
        .type_سائز        = type_سائز,
        .src_صف_سائز         = src_صف_سائز,
        .src_صف_سائز_aligned = src_صف_سائز_aligned,
        .dst_صف_سائز         = dst_صف_سائز,
        .dst_صف_سائز_aligned = dst_صف_سائز_aligned,
    };

    FARF(بلند_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    اگر      (circular && use_dma) { work_قطار_تسلسل(octx->ctx->work_قطار, pad_کام_فی_thread_hvx_circular_dma, &pctx, n_دھاگے); }
    ورنہ اگر (circular)            { work_قطار_تسلسل(octx->ctx->work_قطار, pad_کام_فی_thread_hvx_circular,     &pctx, n_دھاگے); }
    ورنہ اگر (use_dma)             { work_قطار_تسلسل(octx->ctx->work_قطار, pad_کام_فی_thread_hvx_dma,          &pctx, n_دھاگے); }
    ورنہ                          { work_قطار_تسلسل(octx->ctx->work_قطار, pad_کام_فی_thread_hvx,              &pctx, n_دھاگے); }

    واپس HTP_حالت_OK;
}
