#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#অন্তর্ভাব <string.h>

#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "hvx-utils.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"

/* Circular wrap: maps any integer x into [0, n) */
স্থির অন্তঃরেখ uint32_t wrap_around(int32_t x, uint32_t n) {
    ফেরত (uint32_t)(((x % (int32_t)n) + (int32_t)n) % (int32_t)n);
}

/* Decompose a flat dst row index into (i1, i2, i3) */
স্থির অন্তঃরেখ শূন্য pad_decompose_row(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     uint32_t *i1, uint32_t *i2, uint32_t *i3) {
    *i1 = ir % ne1;
    *i2 = (ir / ne1) % ne2;
    *i3 = ir / (ne1 * ne2);
}

/* Return non-zero if row (i1,i2,i3) falls in the non-padded interior */
স্থির অন্তঃরেখ পূর্ণসংখ্যা pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                   int32_t lp1, int32_t rp1, uint32_t ne1,
                                   int32_t lp2, int32_t rp2, uint32_t ne2,
                                   int32_t lp3, int32_t rp3, uint32_t ne3) {
    ফেরত ((int32_t)i1 >= lp1 && (int32_t)i1 < (int32_t)ne1 - rp1) &&
           ((int32_t)i2 >= lp2 && (int32_t)i2 < (int32_t)ne2 - rp2) &&
           ((int32_t)i3 >= lp3 && (int32_t)i3 < (int32_t)ne3 - rp3);
}

/* Compute the DDR src row pointer for a zero-pad interior row */
স্থির অন্তঃরেখ ধ্রুবক uint8_t * pad_src_row_ptr(ধ্রুবক কাঠামো htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    ফেরত (ধ্রুবক uint8_t *) src->তথ্য
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

স্থির অন্তঃরেখ dma_addr_t pad_src_row_তথ্য(ধ্রুবক কাঠামো htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    ফেরত src->তথ্য
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
স্থির অন্তঃরেখ ধ্রুবক uint8_t * pad_circ_src_row_ptr(ধ্রুবক কাঠামো htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    ফেরত (ধ্রুবক uint8_t *) src->তথ্য
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

স্থির অন্তঃরেখ dma_addr_t pad_circ_src_row_তথ্য(ধ্রুবক কাঠামো htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    ফেরত src->তথ্য
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

কাঠামো htp_pad_context {
    কাঠামো htp_ops_context * octx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_per_thread;
    uint32_t যোগ_dst_rows;
    uint32_t row_start;

    size_t   type_আকার;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_row_আকার;
    size_t   src_row_আকার_aligned;
    size_t   dst_row_আকার;
    size_t   dst_row_আকার_aligned;
};

#define htp_pad_preamble                                                             \
    ধ্রুবক কাঠামো htp_tensor * src = octx->src[0];                                    \
    ধ্রুবক কাঠামো htp_tensor * dst = octx->dst;                                       \
                                                                                     \
    ধ্রুবক uint32_t ne00 = src->ne[0];                                                \
    ধ্রুবক uint32_t nb00 = src->nb[0];                                                \
                                                                                     \
    ধ্রুবক uint32_t ne0 = dst->ne[0];                                                 \
    ধ্রুবক uint32_t ne1 = dst->ne[1];                                                 \
    ধ্রুবক uint32_t ne2 = dst->ne[2];                                                 \
    ধ্রুবক uint32_t ne3 = dst->ne[3];                                                 \
                                                                                     \
    ধ্রুবক uint32_t nb1 = dst->nb[1];                                                 \
    ধ্রুবক uint32_t nb2 = dst->nb[2];                                                 \
    ধ্রুবক uint32_t nb3 = dst->nb[3];                                                 \
                                                                                     \
    ধ্রুবক int32_t lp0 = pctx->lp0, rp0 = pctx->rp0;                                  \
    ধ্রুবক int32_t lp1 = pctx->lp1, rp1 = pctx->rp1;                                  \
    ধ্রুবক int32_t lp2 = pctx->lp2, rp2 = pctx->rp2;                                  \
    ধ্রুবক int32_t lp3 = pctx->lp3, rp3 = pctx->rp3;                                  \
                                                                                     \
    ধ্রুবক size_t type_আকার = pctx->type_আকার;                                        \
                                                                                     \
    ধ্রুবক uint32_t row_start = pctx->row_start + pctx->nrows_per_thread * ith;       \
    ধ্রুবক uint32_t row_end   = MIN(row_start + pctx->nrows_per_thread, pctx->row_start + pctx->যোগ_dst_rows);


#define htp_pad_dma_preamble                                                                \
    ধ্রুবক size_t src_row_আকার         = pctx->src_row_আকার;                                 \
    ধ্রুবক size_t src_row_আকার_aligned = pctx->src_row_আকার_aligned;                         \
    ধ্রুবক size_t dst_row_আকার         = pctx->dst_row_আকার;                                 \
    ধ্রুবক size_t dst_row_আকার_aligned = pctx->dst_row_আকার_aligned;                         \
                                                                                            \
    uint8_t * src_spad_base = octx->src0_spad.তথ্য + ith * octx->src0_spad.আকার_per_thread; \
    uint8_t * dst_spad_base = octx->dst_spad.তথ্য  + ith * octx->dst_spad.আকার_per_thread;  \
                                                                                            \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

স্থির শূন্য pad_job_per_thread_hvx(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    ধ্রুবক কাঠামো htp_pad_context * pctx = (ধ্রুবক কাঠামো htp_pad_context *) তথ্য;
    কাঠামো htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    জন্য (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        uint32_t i1, i2, i3;
        pad_decompose_row(dst_row, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->তথ্য + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ধ্রুবক পূর্ণসংখ্যা interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        যদি (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } অন্যথা {
            ধ্রুবক uint8_t * src_ptr = pad_src_row_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            যদি (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_row_start = dst_ptr + (size_t)lp0 * type_আকার;
            যদি (nb00 == type_আকার) {
                hvx_copy_f32_uu(dst_row_start, src_ptr, ne00);
            } অন্যথা {
                জন্য (uint32_t i = 0; i < ne00; i++) {
                    memcpy(dst_row_start + i * type_আকার,
                           src_ptr + (size_t)i * nb00,
                           type_আকার);
                }
            }

            যদি (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_আকার, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(উপরের_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

স্থির শূন্য pad_job_per_thread_hvx_dma(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    ধ্রুবক কাঠামো htp_pad_context * pctx = (ধ্রুবক কাঠামো htp_pad_context *) তথ্য;
    কাঠামো htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    জন্য (uint32_t ir = row_start, spad_idx = 0; ir < row_end && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_base + spad_idx * src_row_আকার_aligned;
        uint8_t * dst_spad_cur = dst_spad_base + spad_idx * dst_row_আকার_aligned;

        dma_queue_ঠেলো(dma_q,
            dma_make_তথ্য(dst->তথ্য, dst_spad_cur),
            dst_row_আকার, dst_row_আকার_aligned, dst_row_আকার, 0);

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);
        ধ্রুবক পূর্ণসংখ্যা interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ধ্রুবক dma_addr_t src_তথ্য = interior
            ? pad_src_row_তথ্য(src, i1, i2, i3, lp1, lp2, lp3) : src->তথ্য;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_queue_ঠেলো(dma_q,
            dma_make_তথ্য(src_spad_cur, src_তথ্য),
            src_row_আকার_aligned, src_row_আকার, src_row_আকার, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];

    জন্য (uint32_t ir = row_start; ir < row_end; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_queue_বের_করো(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);

        ধ্রুবক dma_addr_t dst_তথ্য = dst->তথ্য + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ধ্রুবক পূর্ণসংখ্যা interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        যদি (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } অন্যথা {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_আকার;

            যদি ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } অন্যথা {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_ঠেলো(dma_q,
            dma_make_তথ্য(dst_তথ্য, dst_spad_cur),
            dst_row_আকার, dst_row_আকার_aligned, dst_row_আকার, 1);

        ধ্রুবক uint32_t পরবর্তী_row = ir + 2;
        যদি (পরবর্তী_row < row_end) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_row(পরবর্তী_row, ne1, ne2, &ni1, &ni2, &ni3);
            ধ্রুবক পূর্ণসংখ্যা পরবর্তী_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            ধ্রুবক dma_addr_t পরবর্তী_src_তথ্য = পরবর্তী_interior
                ? pad_src_row_তথ্য(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->তথ্য;

            dma_queue_ঠেলো(dma_q,
                dma_make_তথ্য(src_spad_cur, পরবর্তী_src_তথ্য),
                src_row_আকার_aligned, src_row_আকার, src_row_আকার, পরবর্তী_interior ? 1 : 0);
        }
    }

    dma_queue_flush(dma_q);

    FARF(উপরের_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

স্থির শূন্য pad_job_per_thread_hvx_circular(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    ধ্রুবক কাঠামো htp_pad_context * pctx = (ধ্রুবক কাঠামো htp_pad_context *) তথ্য;
    কাঠামো htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    জন্য (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        uint32_t i1, i2, i3;
        pad_decompose_row(dst_row, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->তথ্য + i1 * nb1 + i2 * nb2 + i3 * nb3;
        ধ্রুবক uint8_t * src_row = pad_circ_src_row_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        যদি (nb00 == type_আকার) {

            যদি (lp0 > 0) {
                যদি ((uint32_t)lp0 < 32) {
                    memcpy(dst_ptr,
                           src_row + (size_t)(ne00 - (uint32_t)lp0) * type_আকার,
                           (size_t)lp0 * type_আকার);
                } অন্যথা {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_row + (size_t)(ne00 - (uint32_t)lp0) * type_আকার,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_আকার, src_row, ne00);
            যদি (rp0 > 0) {
                যদি ((uint32_t)rp0 < 32) {
                    memcpy(dst_ptr + ((size_t)lp0 + ne00) * type_আকার,
                           src_row,
                           (size_t)rp0 * type_আকার);
                } অন্যথা {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_আকার,
                                    src_row,
                                    (uint32_t)rp0);
                }
            }
        } অন্যথা {
            জন্য (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(দশমিক *)(dst_ptr + i * type_আকার) =
                    *(ধ্রুবক দশমিক *)(src_row + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            জন্য (uint32_t i = 0; i < ne00; i++) {
                *(দশমিক *)(dst_ptr + ((size_t)lp0 + i) * type_আকার) =
                    *(ধ্রুবক দশমিক *)(src_row + (size_t)i * nb00);
            }
            জন্য (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(দশমিক *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_আকার) =
                    *(ধ্রুবক দশমিক *)(src_row + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(উপরের_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

স্থির শূন্য pad_job_per_thread_hvx_circular_dma(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    ধ্রুবক কাঠামো htp_pad_context * pctx = (ধ্রুবক কাঠামো htp_pad_context *) তথ্য;
    কাঠামো htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    জন্য (uint32_t ir = row_start, spad_idx = 0; ir < row_end && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_base + spad_idx * src_row_আকার_aligned;
        uint8_t * dst_spad_cur = dst_spad_base + spad_idx * dst_row_আকার_aligned;

        dma_queue_ঠেলো(dma_q,
            dma_make_তথ্য(dst->তথ্য, dst_spad_cur),
            dst_row_আকার, dst_row_আকার_aligned, dst_row_আকার, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_row(ir, ne1, ne2, &pi1, &pi2, &pi3);
        ধ্রুবক dma_addr_t src_তথ্য = pad_circ_src_row_তথ্য(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_queue_ঠেলো(dma_q,
            dma_make_তথ্য(src_spad_cur, src_তথ্য),
            src_row_আকার_aligned, src_row_আকার, src_row_আকার, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];

    জন্য (uint32_t ir = row_start; ir < row_end; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_queue_বের_করো(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);
        ধ্রুবক dma_addr_t dst_তথ্য = dst->তথ্য + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        যদি (lp0 > 0) {
            uint8_t * dst_বাম       = dst_spad_cur;
            ধ্রুবক uint8_t * src_বাম = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_আকার;
            যদি ((uint32_t)lp0 < 32) {
                memcpy(dst_বাম, src_বাম, (size_t)lp0 * type_আকার);
            } অন্যথা {
                hvx_copy_f32_uu(dst_বাম, src_বাম, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_আকার;
            যদি ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } অন্যথা {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        যদি (rp0 > 0) {
            uint8_t * dst_ডান = dst_spad_cur + ((size_t)lp0 + ne00) * type_আকার;
            যদি ((uint32_t)rp0 < 32) {
                memcpy(dst_ডান, src_spad_cur, (size_t)rp0 * type_আকার);
            } অন্যথা {
                যদি ((uintptr_t)dst_ডান % VLEN == 0) {
                    hvx_copy_f32_aa(dst_ডান, src_spad_cur, (uint32_t)rp0);
                } অন্যথা {
                    hvx_copy_f32_ua(dst_ডান, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_ঠেলো(dma_q,
            dma_make_তথ্য(dst_তথ্য, dst_spad_cur),
            dst_row_আকার, dst_row_আকার_aligned, dst_row_আকার, 1);

        ধ্রুবক uint32_t পরবর্তী_row = ir + 2;
        যদি (পরবর্তী_row < row_end) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_row(পরবর্তী_row, ne1, ne2, &nri1, &nri2, &nri3);
            ধ্রুবক dma_addr_t পরবর্তী_src_তথ্য = pad_circ_src_row_তথ্য(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_queue_ঠেলো(dma_q,
                dma_make_তথ্য(src_spad_cur, পরবর্তী_src_তথ্য),
                src_row_আকার_aligned, src_row_আকার, src_row_আকার, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(উপরের_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

পূর্ণসংখ্যা op_pad(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_আকার;
    নির্বাচন (src0->type) {
        ক্ষেত্র HTP_TYPE_F32: type_আকার = 4; বিরতি;
        পূর্বনির্ধারিত:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক int32_t lp0 = octx->op_params[0];
    ধ্রুবক int32_t rp0 = octx->op_params[1];
    ধ্রুবক int32_t lp1 = octx->op_params[2];
    ধ্রুবক int32_t rp1 = octx->op_params[3];
    ধ্রুবক int32_t lp2 = octx->op_params[4];
    ধ্রুবক int32_t rp2 = octx->op_params[5];
    ধ্রুবক int32_t lp3 = octx->op_params[6];
    ধ্রুবক int32_t rp3 = octx->op_params[7];
    ধ্রুবক int32_t circular = octx->op_params[8];

    ধ্রুবক uint32_t ne0  = dst->ne[0];
    ধ্রুবক uint32_t ne00 = src0->ne[0];

    ধ্রুবক uint32_t যোগ_dst_rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ধ্রুবক size_t dst_row_আকার     = (size_t)ne0 * type_আকার;

    uint32_t row_start = 0;
    uint32_t nrows     = যোগ_dst_rows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_আকার, (uint32_t) dst_row_আকার, &rows_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(যোগ_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;

    ধ্রুবক size_t src_row_আকার         = (size_t)ne00 * type_আকার;
    ধ্রুবক size_t src_row_আকার_aligned = hex_round_up(src_row_আকার, VLEN);
    ধ্রুবক size_t dst_row_আকার_aligned = hex_round_up(dst_row_আকার, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    ধ্রুবক size_t vtcm_needed = (size_t)n_threads * 2 * (src_row_আকার_aligned + dst_row_আকার_aligned);

    ধ্রুবক পূর্ণসংখ্যা use_dma = (src0->nb[0] == (uint32_t)type_আকার) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_আকার >= vtcm_needed);

    যদি (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    যদি (use_dma) {
        octx->src0_spad.আকার_per_thread = 2 * src_row_আকার_aligned;
        octx->dst_spad.আকার_per_thread  = 2 * dst_row_আকার_aligned;
        octx->src0_spad.আকার_2 = n_threads * octx->src0_spad.আকার_per_thread;
        octx->dst_spad.আকার_2  = n_threads * octx->dst_spad.আকার_per_thread;
        octx->src0_spad.তথ্য = octx->ctx->vtcm_base;
        octx->dst_spad.তথ্য  = octx->src0_spad.তথ্য + octx->src0_spad.আকার_2;
        octx->src0_spad.src  = ফাঁকা;
        octx->dst_spad.src   = ফাঁকা;
    }

    কাঠামো htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .যোগ_dst_rows   = nrows,
        .row_start        = row_start,
        .type_আকার        = type_আকার,
        .src_row_আকার         = src_row_আকার,
        .src_row_আকার_aligned = src_row_আকার_aligned,
        .dst_row_আকার         = dst_row_আকার,
        .dst_row_আকার_aligned = dst_row_আকার_aligned,
    };

    FARF(উপরের_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    যদি      (circular && use_dma) { work_queue_run(octx->ctx->work_queue, pad_job_per_thread_hvx_circular_dma, &pctx, n_threads); }
    অন্যথা যদি (circular)            { work_queue_run(octx->ctx->work_queue, pad_job_per_thread_hvx_circular,     &pctx, n_threads); }
    অন্যথা যদি (use_dma)             { work_queue_run(octx->ctx->work_queue, pad_job_per_thread_hvx_dma,          &pctx, n_threads); }
    অন্যথা                          { work_queue_run(octx->ctx->work_queue, pad_job_per_thread_hvx,              &pctx, n_threads); }

    ফেরত HTP_STATUS_OK;
}
