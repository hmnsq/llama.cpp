#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#ଅନ୍ତର୍ଭାଵ <string.h>

#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"

/* Circular wrap: maps any integer x into [0, n) */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t wrap_around(int32_t x, uint32_t n) {
    ଫେରନ୍ତୁ (uint32_t)(((x % (int32_t)n) + (int32_t)n) % (int32_t)n);
}

/* Decompose a flat dst row index into (i1, i2, i3) */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ pad_decompose_row(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     uint32_t *i1, uint32_t *i2, uint32_t *i3) {
    *i1 = ir % ne1;
    *i2 = (ir / ne1) % ne2;
    *i3 = ir / (ne1 * ne2);
}

/* Return non-zero if row (i1,i2,i3) falls in the non-padded interior */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                   int32_t lp1, int32_t rp1, uint32_t ne1,
                                   int32_t lp2, int32_t rp2, uint32_t ne2,
                                   int32_t lp3, int32_t rp3, uint32_t ne3) {
    ଫେରନ୍ତୁ ((int32_t)i1 >= lp1 && (int32_t)i1 < (int32_t)ne1 - rp1) &&
           ((int32_t)i2 >= lp2 && (int32_t)i2 < (int32_t)ne2 - rp2) &&
           ((int32_t)i3 >= lp3 && (int32_t)i3 < (int32_t)ne3 - rp3);
}

/* Compute the DDR src row pointer for a zero-pad interior row */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ uint8_t * pad_src_row_ptr(ଅଚଳ ସଂରଚନା htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    ଫେରନ୍ତୁ (ଅଚଳ uint8_t *) src->data
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_addr_t pad_src_row_data(ଅଚଳ ସଂରଚନା htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    ଫେରନ୍ତୁ src->data
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ uint8_t * pad_circ_src_row_ptr(ଅଚଳ ସଂରଚନା htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    ଫେରନ୍ତୁ (ଅଚଳ uint8_t *) src->data
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_addr_t pad_circ_src_row_data(ଅଚଳ ସଂରଚନା htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    ଫେରନ୍ତୁ src->data
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

ସଂରଚନା htp_pad_context {
    ସଂରଚନା htp_ops_context * octx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_per_thread;
    uint32_t total_dst_rows;
    uint32_t row_start;

    size_t   type_size;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_row_size;
    size_t   src_row_size_aligned;
    size_t   dst_row_size;
    size_t   dst_row_size_aligned;
};

#define htp_pad_preamble                                                             \
    ଅଚଳ ସଂରଚନା htp_tensor * src = octx->src[0];                                    \
    ଅଚଳ ସଂରଚନା htp_tensor * dst = octx->dst;                                       \
                                                                                     \
    ଅଚଳ uint32_t ne00 = src->ne[0];                                                \
    ଅଚଳ uint32_t nb00 = src->nb[0];                                                \
                                                                                     \
    ଅଚଳ uint32_t ne0 = dst->ne[0];                                                 \
    ଅଚଳ uint32_t ne1 = dst->ne[1];                                                 \
    ଅଚଳ uint32_t ne2 = dst->ne[2];                                                 \
    ଅଚଳ uint32_t ne3 = dst->ne[3];                                                 \
                                                                                     \
    ଅଚଳ uint32_t nb1 = dst->nb[1];                                                 \
    ଅଚଳ uint32_t nb2 = dst->nb[2];                                                 \
    ଅଚଳ uint32_t nb3 = dst->nb[3];                                                 \
                                                                                     \
    ଅଚଳ int32_t lp0 = pctx->lp0, rp0 = pctx->rp0;                                  \
    ଅଚଳ int32_t lp1 = pctx->lp1, rp1 = pctx->rp1;                                  \
    ଅଚଳ int32_t lp2 = pctx->lp2, rp2 = pctx->rp2;                                  \
    ଅଚଳ int32_t lp3 = pctx->lp3, rp3 = pctx->rp3;                                  \
                                                                                     \
    ଅଚଳ size_t type_size = pctx->type_size;                                        \
                                                                                     \
    ଅଚଳ uint32_t row_start = pctx->row_start + pctx->nrows_per_thread * ith;       \
    ଅଚଳ uint32_t row_end   = MIN(row_start + pctx->nrows_per_thread, pctx->row_start + pctx->total_dst_rows);


#define htp_pad_dma_preamble                                                                \
    ଅଚଳ size_t src_row_size         = pctx->src_row_size;                                 \
    ଅଚଳ size_t src_row_size_aligned = pctx->src_row_size_aligned;                         \
    ଅଚଳ size_t dst_row_size         = pctx->dst_row_size;                                 \
    ଅଚଳ size_t dst_row_size_aligned = pctx->dst_row_size_aligned;                         \
                                                                                            \
    uint8_t * src_spad_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread; \
    uint8_t * dst_spad_base = octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread;  \
                                                                                            \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

ସ୍ଥିର ଶୂନ୍ୟ pad_job_per_thread_hvx(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ଅଚଳ ସଂରଚନା htp_pad_context * pctx = (ଅଚଳ ସଂରଚନା htp_pad_context *) data;
    ସଂରଚନା htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    ପାଇଁ (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        uint32_t i1, i2, i3;
        pad_decompose_row(dst_row, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ଯଦି (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } ଅନ୍ୟଥା {
            ଅଚଳ uint8_t * src_ptr = pad_src_row_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            ଯଦି (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_row_start = dst_ptr + (size_t)lp0 * type_size;
            ଯଦି (nb00 == type_size) {
                hvx_copy_f32_uu(dst_row_start, src_ptr, ne00);
            } ଅନ୍ୟଥା {
                ପାଇଁ (uint32_t i = 0; i < ne00; i++) {
                    memcpy(dst_row_start + i * type_size,
                           src_ptr + (size_t)i * nb00,
                           type_size);
                }
            }

            ଯଦି (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_size, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(ଉପର_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ସ୍ଥିର ଶୂନ୍ୟ pad_job_per_thread_hvx_dma(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ଅଚଳ ସଂରଚନା htp_pad_context * pctx = (ଅଚଳ ସଂରଚନା htp_pad_context *) data;
    ସଂରଚନା htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    ପାଇଁ (uint32_t ir = row_start, spad_idx = 0; ir < row_end && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_base + spad_idx * src_row_size_aligned;
        uint8_t * dst_spad_cur = dst_spad_base + spad_idx * dst_row_size_aligned;

        dma_queue_ଠେଲ(dma_q,
            dma_make_data(dst->data, dst_spad_cur),
            dst_row_size, dst_row_size_aligned, dst_row_size, 0);

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ଅଚଳ dma_addr_t src_data = interior
            ? pad_src_row_data(src, i1, i2, i3, lp1, lp2, lp3) : src->data;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_queue_ଠେଲ(dma_q,
            dma_make_data(src_spad_cur, src_data),
            src_row_size_aligned, src_row_size, src_row_size, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ପାଇଁ (uint32_t ir = row_start; ir < row_end; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_queue_କାଢ଼(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_queue_କାଢ଼(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);

        ଅଚଳ dma_addr_t dst_data = dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ଯଦି (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } ଅନ୍ୟଥା {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_size;

            ଯଦି ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } ଅନ୍ୟଥା {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_ଠେଲ(dma_q,
            dma_make_data(dst_data, dst_spad_cur),
            dst_row_size, dst_row_size_aligned, dst_row_size, 1);

        ଅଚଳ uint32_t next_row = ir + 2;
        ଯଦି (next_row < row_end) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_row(next_row, ne1, ne2, &ni1, &ni2, &ni3);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା next_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            ଅଚଳ dma_addr_t next_src_data = next_interior
                ? pad_src_row_data(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->data;

            dma_queue_ଠେଲ(dma_q,
                dma_make_data(src_spad_cur, next_src_data),
                src_row_size_aligned, src_row_size, src_row_size, next_interior ? 1 : 0);
        }
    }

    dma_queue_flush(dma_q);

    FARF(ଉପର_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

ସ୍ଥିର ଶୂନ୍ୟ pad_job_per_thread_hvx_circular(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ଅଚଳ ସଂରଚନା htp_pad_context * pctx = (ଅଚଳ ସଂରଚନା htp_pad_context *) data;
    ସଂରଚନା htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    ପାଇଁ (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        uint32_t i1, i2, i3;
        pad_decompose_row(dst_row, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;
        ଅଚଳ uint8_t * src_row = pad_circ_src_row_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        ଯଦି (nb00 == type_size) {

            ଯଦି (lp0 > 0) {
                ଯଦି ((uint32_t)lp0 < 32) {
                    memcpy(dst_ptr,
                           src_row + (size_t)(ne00 - (uint32_t)lp0) * type_size,
                           (size_t)lp0 * type_size);
                } ଅନ୍ୟଥା {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_row + (size_t)(ne00 - (uint32_t)lp0) * type_size,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_size, src_row, ne00);
            ଯଦି (rp0 > 0) {
                ଯଦି ((uint32_t)rp0 < 32) {
                    memcpy(dst_ptr + ((size_t)lp0 + ne00) * type_size,
                           src_row,
                           (size_t)rp0 * type_size);
                } ଅନ୍ୟଥା {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_size,
                                    src_row,
                                    (uint32_t)rp0);
                }
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(ଦଶମିକ *)(dst_ptr + i * type_size) =
                    *(ଅଚଳ ଦଶମିକ *)(src_row + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            ପାଇଁ (uint32_t i = 0; i < ne00; i++) {
                *(ଦଶମିକ *)(dst_ptr + ((size_t)lp0 + i) * type_size) =
                    *(ଅଚଳ ଦଶମିକ *)(src_row + (size_t)i * nb00);
            }
            ପାଇଁ (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(ଦଶମିକ *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_size) =
                    *(ଅଚଳ ଦଶମିକ *)(src_row + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(ଉପର_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

ସ୍ଥିର ଶୂନ୍ୟ pad_job_per_thread_hvx_circular_dma(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ଅଚଳ ସଂରଚନା htp_pad_context * pctx = (ଅଚଳ ସଂରଚନା htp_pad_context *) data;
    ସଂରଚନା htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    ପାଇଁ (uint32_t ir = row_start, spad_idx = 0; ir < row_end && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_base + spad_idx * src_row_size_aligned;
        uint8_t * dst_spad_cur = dst_spad_base + spad_idx * dst_row_size_aligned;

        dma_queue_ଠେଲ(dma_q,
            dma_make_data(dst->data, dst_spad_cur),
            dst_row_size, dst_row_size_aligned, dst_row_size, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_row(ir, ne1, ne2, &pi1, &pi2, &pi3);
        ଅଚଳ dma_addr_t src_data = pad_circ_src_row_data(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_queue_ଠେଲ(dma_q,
            dma_make_data(src_spad_cur, src_data),
            src_row_size_aligned, src_row_size, src_row_size, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ପାଇଁ (uint32_t ir = row_start; ir < row_end; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_queue_କାଢ଼(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_queue_କାଢ଼(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);
        ଅଚଳ dma_addr_t dst_data = dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        ଯଦି (lp0 > 0) {
            uint8_t * dst_ବାମ       = dst_spad_cur;
            ଅଚଳ uint8_t * src_ବାମ = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_size;
            ଯଦି ((uint32_t)lp0 < 32) {
                memcpy(dst_ବାମ, src_ବାମ, (size_t)lp0 * type_size);
            } ଅନ୍ୟଥା {
                hvx_copy_f32_uu(dst_ବାମ, src_ବାମ, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_size;
            ଯଦି ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } ଅନ୍ୟଥା {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        ଯଦି (rp0 > 0) {
            uint8_t * dst_ଡାହାଣ = dst_spad_cur + ((size_t)lp0 + ne00) * type_size;
            ଯଦି ((uint32_t)rp0 < 32) {
                memcpy(dst_ଡାହାଣ, src_spad_cur, (size_t)rp0 * type_size);
            } ଅନ୍ୟଥା {
                ଯଦି ((uintptr_t)dst_ଡାହାଣ % VLEN == 0) {
                    hvx_copy_f32_aa(dst_ଡାହାଣ, src_spad_cur, (uint32_t)rp0);
                } ଅନ୍ୟଥା {
                    hvx_copy_f32_ua(dst_ଡାହାଣ, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_ଠେଲ(dma_q,
            dma_make_data(dst_data, dst_spad_cur),
            dst_row_size, dst_row_size_aligned, dst_row_size, 1);

        ଅଚଳ uint32_t next_row = ir + 2;
        ଯଦି (next_row < row_end) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_row(next_row, ne1, ne2, &nri1, &nri2, &nri3);
            ଅଚଳ dma_addr_t next_src_data = pad_circ_src_row_data(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_queue_ଠେଲ(dma_q,
                dma_make_data(src_spad_cur, next_src_data),
                src_row_size_aligned, src_row_size, src_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(ଉପର_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_pad(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_size;
    ବାଛ (src0->type) {
        ସ୍ଥିତି HTP_TYPE_F32: type_size = 4; ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ int32_t lp0 = octx->op_params[0];
    ଅଚଳ int32_t rp0 = octx->op_params[1];
    ଅଚଳ int32_t lp1 = octx->op_params[2];
    ଅଚଳ int32_t rp1 = octx->op_params[3];
    ଅଚଳ int32_t lp2 = octx->op_params[4];
    ଅଚଳ int32_t rp2 = octx->op_params[5];
    ଅଚଳ int32_t lp3 = octx->op_params[6];
    ଅଚଳ int32_t rp3 = octx->op_params[7];
    ଅଚଳ int32_t circular = octx->op_params[8];

    ଅଚଳ uint32_t ne0  = dst->ne[0];
    ଅଚଳ uint32_t ne00 = src0->ne[0];

    ଅଚଳ uint32_t total_dst_rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ଅଚଳ size_t dst_row_size     = (size_t)ne0 * type_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_row_size, &rows_per_chunk);
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
    }

    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads = octx->n_threads;

    ଅଚଳ size_t src_row_size         = (size_t)ne00 * type_size;
    ଅଚଳ size_t src_row_size_aligned = hex_round_up(src_row_size, VLEN);
    ଅଚଳ size_t dst_row_size_aligned = hex_round_up(dst_row_size, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    ଅଚଳ size_t vtcm_needed = (size_t)n_threads * 2 * (src_row_size_aligned + dst_row_size_aligned);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା use_dma = (src0->nb[0] == (uint32_t)type_size) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_size >= vtcm_needed);

    ଯଦି (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (use_dma) {
        octx->src0_spad.size_per_thread = 2 * src_row_size_aligned;
        octx->dst_spad.size_per_thread  = 2 * dst_row_size_aligned;
        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
        octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;
        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size;
        octx->src0_spad.src  = ଖାଲି;
        octx->dst_spad.src   = ଖାଲି;
    }

    ସଂରଚନା htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_dst_rows   = nrows,
        .row_start        = row_start,
        .type_size        = type_size,
        .src_row_size         = src_row_size,
        .src_row_size_aligned = src_row_size_aligned,
        .dst_row_size         = dst_row_size,
        .dst_row_size_aligned = dst_row_size_aligned,
    };

    FARF(ଉପର_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    ଯଦି      (circular && use_dma) { work_queue_run(octx->ctx->work_queue, pad_job_per_thread_hvx_circular_dma, &pctx, n_threads); }
    ଅନ୍ୟଥା ଯଦି (circular)            { work_queue_run(octx->ctx->work_queue, pad_job_per_thread_hvx_circular,     &pctx, n_threads); }
    ଅନ୍ୟଥା ଯଦି (use_dma)             { work_queue_run(octx->ctx->work_queue, pad_job_per_thread_hvx_dma,          &pctx, n_threads); }
    ଅନ୍ୟଥା                          { work_queue_run(octx->ctx->work_queue, pad_job_per_thread_hvx,              &pctx, n_threads); }

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}
