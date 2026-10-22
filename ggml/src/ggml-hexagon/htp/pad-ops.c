#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <string.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hvx-utils.h"

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"

/* Circular wrap: maps any integer x into [0, n) */
स्थिर अन्तःस्थ uint32_t wrap_around(int32_t x, uint32_t n) {
    प्रत्यागम (uint32_t)(((x % (int32_t)n) + (int32_t)n) % (int32_t)n);
}

/* Decompose a flat dst row index into (i1, i2, i3) */
स्थिर अन्तःस्थ शून्य pad_decompose_row(uint32_t ir, uint32_t ne1, uint32_t ne2,
                                     uint32_t *i1, uint32_t *i2, uint32_t *i3) {
    *i1 = ir % ne1;
    *i2 = (ir / ne1) % ne2;
    *i3 = ir / (ne1 * ne2);
}

/* Return non-zero if row (i1,i2,i3) falls in the non-padded interior */
स्थिर अन्तःस्थ पूर्णाङ्क pad_is_interior(uint32_t i1, uint32_t i2, uint32_t i3,
                                   int32_t lp1, int32_t rp1, uint32_t ne1,
                                   int32_t lp2, int32_t rp2, uint32_t ne2,
                                   int32_t lp3, int32_t rp3, uint32_t ne3) {
    प्रत्यागम ((int32_t)i1 >= lp1 && (int32_t)i1 < (int32_t)ne1 - rp1) &&
           ((int32_t)i2 >= lp2 && (int32_t)i2 < (int32_t)ne2 - rp2) &&
           ((int32_t)i3 >= lp3 && (int32_t)i3 < (int32_t)ne3 - rp3);
}

/* Compute the DDR src row pointer for a zero-pad interior row */
स्थिर अन्तःस्थ ध्रुव uint8_t * pad_src_row_ptr(ध्रुव संरचना htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    प्रत्यागम (ध्रुव uint8_t *) src->दत्त
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

स्थिर अन्तःस्थ dma_addr_t pad_src_row_दत्त(ध्रुव संरचना htp_tensor * src,
                                          uint32_t i1, uint32_t i2, uint32_t i3,
                                          int32_t lp1, int32_t lp2, int32_t lp3) {
    प्रत्यागम src->दत्त
        + (i1 - (uint32_t)lp1) * src->nb[1]
        + (i2 - (uint32_t)lp2) * src->nb[2]
        + (i3 - (uint32_t)lp3) * src->nb[3];
}

/* Compute the DDR src row pointer for a circular row (wrap-around indexing) */
स्थिर अन्तःस्थ ध्रुव uint8_t * pad_circ_src_row_ptr(ध्रुव संरचना htp_tensor * src,
                                                    uint32_t i1, uint32_t i2, uint32_t i3,
                                                    int32_t lp1, int32_t lp2, int32_t lp3) {
    प्रत्यागम (ध्रुव uint8_t *) src->दत्त
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

स्थिर अन्तःस्थ dma_addr_t pad_circ_src_row_दत्त(ध्रुव संरचना htp_tensor * src,
                                               uint32_t i1, uint32_t i2, uint32_t i3,
                                               int32_t lp1, int32_t lp2, int32_t lp3) {
    प्रत्यागम src->दत्त
        + wrap_around((int32_t)i1 - lp1, src->ne[1]) * src->nb[1]
        + wrap_around((int32_t)i2 - lp2, src->ne[2]) * src->nb[2]
        + wrap_around((int32_t)i3 - lp3, src->ne[3]) * src->nb[3];
}

संरचना htp_pad_context {
    संरचना htp_ops_context * octx;

    int32_t  lp0, rp0;
    int32_t  lp1, rp1;
    int32_t  lp2, rp2;
    int32_t  lp3, rp3;

    uint32_t nrows_प्रत्येकम्_thread;
    uint32_t योग_dst_पङ्क्तिकाः;
    uint32_t row_start;

    size_t   type_आकार;

    // Row sizes for DMA kernel (populated when VTCM is available)
    size_t   src_row_आकार;
    size_t   src_row_आकार_aligned;
    size_t   dst_row_आकार;
    size_t   dst_row_आकार_aligned;
};

#परिभाषय htp_pad_preamble                                                             \
    ध्रुव संरचना htp_tensor * src = octx->src[0];                                    \
    ध्रुव संरचना htp_tensor * dst = octx->dst;                                       \
                                                                                     \
    ध्रुव uint32_t ne00 = src->ne[0];                                                \
    ध्रुव uint32_t nb00 = src->nb[0];                                                \
                                                                                     \
    ध्रुव uint32_t ne0 = dst->ne[0];                                                 \
    ध्रुव uint32_t ne1 = dst->ne[1];                                                 \
    ध्रुव uint32_t ne2 = dst->ne[2];                                                 \
    ध्रुव uint32_t ne3 = dst->ne[3];                                                 \
                                                                                     \
    ध्रुव uint32_t nb1 = dst->nb[1];                                                 \
    ध्रुव uint32_t nb2 = dst->nb[2];                                                 \
    ध्रुव uint32_t nb3 = dst->nb[3];                                                 \
                                                                                     \
    ध्रुव int32_t lp0 = pctx->lp0, rp0 = pctx->rp0;                                  \
    ध्रुव int32_t lp1 = pctx->lp1, rp1 = pctx->rp1;                                  \
    ध्रुव int32_t lp2 = pctx->lp2, rp2 = pctx->rp2;                                  \
    ध्रुव int32_t lp3 = pctx->lp3, rp3 = pctx->rp3;                                  \
                                                                                     \
    ध्रुव size_t type_आकार = pctx->type_आकार;                                        \
                                                                                     \
    ध्रुव uint32_t row_start = pctx->row_start + pctx->nrows_प्रत्येकम्_thread * ith;       \
    ध्रुव uint32_t row_अन्तः   = MIN(row_start + pctx->nrows_प्रत्येकम्_thread, pctx->row_start + pctx->योग_dst_पङ्क्तिकाः);


#परिभाषय htp_pad_dma_preamble                                                                \
    ध्रुव size_t src_row_आकार         = pctx->src_row_आकार;                                 \
    ध्रुव size_t src_row_आकार_aligned = pctx->src_row_आकार_aligned;                         \
    ध्रुव size_t dst_row_आकार         = pctx->dst_row_आकार;                                 \
    ध्रुव size_t dst_row_आकार_aligned = pctx->dst_row_आकार_aligned;                         \
                                                                                            \
    uint8_t * src_spad_आधारः = octx->src0_spad.दत्त + ith * octx->src0_spad.आकार_प्रत्येकम्_thread; \
    uint8_t * dst_spad_आधारः = octx->dst_spad.दत्त  + ith * octx->dst_spad.आकार_प्रत्येकम्_thread;  \
                                                                                            \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX vectorized PAD kernel
// ---------------------------------------------------------------------------

स्थिर शून्य pad_job_प्रत्येकम्_thread_hvx(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    ध्रुव संरचना htp_pad_context * pctx = (ध्रुव संरचना htp_pad_context *) दत्त;
    संरचना htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    प्रति (uint32_t dst_row = row_start; dst_row < row_अन्तः; dst_row++) {
        uint32_t i1, i2, i3;
        pad_decompose_row(dst_row, ne1, ne2, &i1, &i2, &i3);

        uint8_t * dst_ptr = (uint8_t *) dst->दत्त + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ध्रुव पूर्णाङ्क interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        यदि (!interior) {
            hvx_splat_f32_u(dst_ptr, 0.0f, ne0);
        } अन्यथा {
            ध्रुव uint8_t * src_ptr = pad_src_row_ptr(src, i1, i2, i3, lp1, lp2, lp3);

            यदि (lp0 > 0) {
                hvx_splat_f32_u(dst_ptr, 0.0f, (uint32_t)lp0);
            }

            uint8_t * dst_row_start = dst_ptr + (size_t)lp0 * type_आकार;
            यदि (nb00 == type_आकार) {
                hvx_copy_f32_uu(dst_row_start, src_ptr, ne00);
            } अन्यथा {
                प्रति (uint32_t i = 0; i < ne00; i++) {
                    स्मृतिं_प्रतिलिख(dst_row_start + i * type_आकार,
                           src_ptr + (size_t)i * nb00,
                           type_आकार);
                }
            }

            यदि (rp0 > 0) {
                hvx_splat_f32_u(dst_ptr + ((size_t)lp0 + ne00) * type_आकार, 0.0f, (uint32_t)rp0);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(उच्चः_2, "pad-hvx %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_अन्तः);
}

// ---------------------------------------------------------------------------
// HVX + DMA PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

स्थिर शून्य pad_job_प्रत्येकम्_thread_hvx_dma(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    ध्रुव संरचना htp_pad_context * pctx = (ध्रुव संरचना htp_pad_context *) दत्त;
    संरचना htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline before the main loop begins.
    // -----------------------------------------------------------------------
    प्रति (uint32_t ir = row_start, spad_idx = 0; ir < row_अन्तः && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_आधारः + spad_idx * src_row_आकार_aligned;
        uint8_t * dst_spad_cur = dst_spad_आधारः + spad_idx * dst_row_आकार_aligned;

        dma_queue_प्रक्षिप(dma_q,
            dma_make_दत्त(dst->दत्त, dst_spad_cur),
            dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 0);

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);
        ध्रुव पूर्णाङ्क interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        ध्रुव dma_addr_t src_दत्त = interior
            ? pad_src_row_दत्त(src, i1, i2, i3, lp1, lp2, lp3) : src->दत्त;

        // Interior row: real DMA (1 row) from DDR to VTCM.
        // Border row: null DMA (nrows=0)
        dma_queue_प्रक्षिप(dma_q,
            dma_make_दत्त(src_spad_cur, src_दत्त),
            src_row_आकार_aligned, src_row_आकार, src_row_आकार, interior ? 1 : 0);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, compute in VTCM with aligned HVX ops,
    // push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    प्रति (uint32_t ir = row_start; ir < row_अन्तः; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_queue_निष्कासय(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);

        ध्रुव dma_addr_t dst_दत्त = dst->दत्त + i1 * nb1 + i2 * nb2 + i3 * nb3;

        ध्रुव पूर्णाङ्क interior = pad_is_interior(i1, i2, i3,
                                             lp1, rp1, ne1,
                                             lp2, rp2, ne2,
                                             lp3, rp3, ne3);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        यदि (!interior) {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);
        } अन्यथा {
            hvx_splat_f32_a(dst_spad_cur, 0.0f, ne0);

            uint8_t * dst_interior = dst_spad_cur + (size_t)lp0 * type_आकार;

            यदि ((uintptr_t)dst_interior % VLEN == 0) {
                hvx_copy_f32_aa(dst_interior, src_spad_cur, ne00);
            } अन्यथा {
                hvx_copy_f32_ua(dst_interior, src_spad_cur, ne00);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_प्रक्षिप(dma_q,
            dma_make_दत्त(dst_दत्त, dst_spad_cur),
            dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 1);

        ध्रुव uint32_t अग्र_row = ir + 2;
        यदि (अग्र_row < row_अन्तः) {
            uint32_t ni1, ni2, ni3;
            pad_decompose_row(अग्र_row, ne1, ne2, &ni1, &ni2, &ni3);
            ध्रुव पूर्णाङ्क अग्र_interior = pad_is_interior(ni1, ni2, ni3,
                                                      lp1, rp1, ne1,
                                                      lp2, rp2, ne2,
                                                      lp3, rp3, ne3);
            ध्रुव dma_addr_t अग्र_src_दत्त = अग्र_interior
                ? pad_src_row_दत्त(src, ni1, ni2, ni3, lp1, lp2, lp3) : src->दत्त;

            dma_queue_प्रक्षिप(dma_q,
                dma_make_दत्त(src_spad_cur, अग्र_src_दत्त),
                src_row_आकार_aligned, src_row_आकार, src_row_आकार, अग्र_interior ? 1 : 0);
        }
    }

    dma_queue_flush(dma_q);

    FARF(उच्चः_2, "pad-hvx-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_अन्तः);
}

// ---------------------------------------------------------------------------
// HVX circular PAD kernel
// ---------------------------------------------------------------------------

स्थिर शून्य pad_job_प्रत्येकम्_thread_hvx_circular(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    ध्रुव संरचना htp_pad_context * pctx = (ध्रुव संरचना htp_pad_context *) दत्त;
    संरचना htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    प्रति (uint32_t dst_row = row_start; dst_row < row_अन्तः; dst_row++) {
        uint32_t i1, i2, i3;
        pad_decompose_row(dst_row, ne1, ne2, &i1, &i2, &i3);

        uint8_t       * dst_ptr = (uint8_t *) dst->दत्त + i1 * nb1 + i2 * nb2 + i3 * nb3;
        ध्रुव uint8_t * src_row = pad_circ_src_row_ptr(src, i1, i2, i3, lp1, lp2, lp3);

        यदि (nb00 == type_आकार) {

            यदि (lp0 > 0) {
                यदि ((uint32_t)lp0 < 32) {
                    स्मृतिं_प्रतिलिख(dst_ptr,
                           src_row + (size_t)(ne00 - (uint32_t)lp0) * type_आकार,
                           (size_t)lp0 * type_आकार);
                } अन्यथा {
                    hvx_copy_f32_uu(dst_ptr,
                                    src_row + (size_t)(ne00 - (uint32_t)lp0) * type_आकार,
                                    (uint32_t)lp0);
                }
            }
            hvx_copy_f32_uu(dst_ptr + (size_t)lp0 * type_आकार, src_row, ne00);
            यदि (rp0 > 0) {
                यदि ((uint32_t)rp0 < 32) {
                    स्मृतिं_प्रतिलिख(dst_ptr + ((size_t)lp0 + ne00) * type_आकार,
                           src_row,
                           (size_t)rp0 * type_आकार);
                } अन्यथा {
                    hvx_copy_f32_uu(dst_ptr + ((size_t)lp0 + ne00) * type_आकार,
                                    src_row,
                                    (uint32_t)rp0);
                }
            }
        } अन्यथा {
            प्रति (uint32_t i = 0; i < (uint32_t)lp0; i++) {
                *(भिन्न *)(dst_ptr + i * type_आकार) =
                    *(ध्रुव भिन्न *)(src_row + (size_t)(ne00 - (uint32_t)lp0 + i) * nb00);
            }
            प्रति (uint32_t i = 0; i < ne00; i++) {
                *(भिन्न *)(dst_ptr + ((size_t)lp0 + i) * type_आकार) =
                    *(ध्रुव भिन्न *)(src_row + (size_t)i * nb00);
            }
            प्रति (uint32_t i = 0; i < (uint32_t)rp0; i++) {
                *(भिन्न *)(dst_ptr + ((size_t)lp0 + ne00 + i) * type_आकार) =
                    *(ध्रुव भिन्न *)(src_row + (size_t)i * nb00);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(उच्चः_2, "pad-hvx-circ %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_अन्तः);
}

// ---------------------------------------------------------------------------
// HVX + DMA circular PAD kernel - aligned, double-buffered
// ---------------------------------------------------------------------------

स्थिर शून्य pad_job_प्रत्येकम्_thread_hvx_circular_dma(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    ध्रुव संरचना htp_pad_context * pctx = (ध्रुव संरचना htp_pad_context *) दत्त;
    संरचना htp_ops_context * octx = pctx->octx;
    htp_pad_preamble;
    htp_pad_dma_preamble;

    // -----------------------------------------------------------------------
    // Priming phase: push 2 pairs of (dummy_dst_DMA, src_DMA) to seed the
    // double-buffer pipeline.  Every row is a real src DMA (no null DMAs).
    // -----------------------------------------------------------------------
    प्रति (uint32_t ir = row_start, spad_idx = 0; ir < row_अन्तः && spad_idx < 2; ir++, spad_idx++) {
        uint8_t * src_spad_cur = src_spad_आधारः + spad_idx * src_row_आकार_aligned;
        uint8_t * dst_spad_cur = dst_spad_आधारः + spad_idx * dst_row_आकार_aligned;

        dma_queue_प्रक्षिप(dma_q,
            dma_make_दत्त(dst->दत्त, dst_spad_cur),
            dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 0);

        uint32_t pi1, pi2, pi3;
        pad_decompose_row(ir, ne1, ne2, &pi1, &pi2, &pi3);
        ध्रुव dma_addr_t src_दत्त = pad_circ_src_row_दत्त(src, pi1, pi2, pi3, lp1, lp2, lp3);
        dma_queue_प्रक्षिप(dma_q,
            dma_make_दत्त(src_spad_cur, src_दत्त),
            src_row_आकार_aligned, src_row_आकार, src_row_आकार, 1);
    }

    // -----------------------------------------------------------------------
    // Main loop: pop completed DMAs, assemble circular row in VTCM with
    // aligned HVX ops, push dst DMA and prefetch src for the next+1 row.
    // -----------------------------------------------------------------------
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    प्रति (uint32_t ir = row_start; ir < row_अन्तः; ir++) {
        uint8_t * dst_spad_cur = (uint8_t *) dma_queue_निष्कासय(dma_q).src;
        uint8_t * src_spad_cur = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

        uint32_t i1, i2, i3;
        pad_decompose_row(ir, ne1, ne2, &i1, &i2, &i3);
        ध्रुव dma_addr_t dst_दत्त = dst->दत्त + i1 * nb1 + i2 * nb2 + i3 * nb3;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        यदि (lp0 > 0) {
            uint8_t * dst_वामः       = dst_spad_cur;
            ध्रुव uint8_t * src_वामः = src_spad_cur + (size_t)(ne00 - (uint32_t)lp0) * type_आकार;
            यदि ((uint32_t)lp0 < 32) {
                स्मृतिं_प्रतिलिख(dst_वामः, src_वामः, (size_t)lp0 * type_आकार);
            } अन्यथा {
                hvx_copy_f32_uu(dst_वामः, src_वामः, (uint32_t)lp0);
            }
        }

        {
            uint8_t * dst_mid = dst_spad_cur + (size_t)lp0 * type_आकार;
            यदि ((uintptr_t)dst_mid % VLEN == 0) {
                hvx_copy_f32_aa(dst_mid, src_spad_cur, ne00);
            } अन्यथा {
                hvx_copy_f32_ua(dst_mid, src_spad_cur, ne00);
            }
        }

        यदि (rp0 > 0) {
            uint8_t * dst_दक्षिणः = dst_spad_cur + ((size_t)lp0 + ne00) * type_आकार;
            यदि ((uint32_t)rp0 < 32) {
                स्मृतिं_प्रतिलिख(dst_दक्षिणः, src_spad_cur, (size_t)rp0 * type_आकार);
            } अन्यथा {
                यदि ((uintptr_t)dst_दक्षिणः % VLEN == 0) {
                    hvx_copy_f32_aa(dst_दक्षिणः, src_spad_cur, (uint32_t)rp0);
                } अन्यथा {
                    hvx_copy_f32_ua(dst_दक्षिणः, src_spad_cur, (uint32_t)rp0);
                }
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_प्रक्षिप(dma_q,
            dma_make_दत्त(dst_दत्त, dst_spad_cur),
            dst_row_आकार, dst_row_आकार_aligned, dst_row_आकार, 1);

        ध्रुव uint32_t अग्र_row = ir + 2;
        यदि (अग्र_row < row_अन्तः) {
            uint32_t nri1, nri2, nri3;
            pad_decompose_row(अग्र_row, ne1, ne2, &nri1, &nri2, &nri3);
            ध्रुव dma_addr_t अग्र_src_दत्त = pad_circ_src_row_दत्त(src, nri1, nri2, nri3, lp1, lp2, lp3);
            dma_queue_प्रक्षिप(dma_q,
                dma_make_दत्त(src_spad_cur, अग्र_src_दत्त),
                src_row_आकार_aligned, src_row_आकार, src_row_आकार, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(उच्चः_2, "pad-hvx-circ-dma %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth,
         src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_अन्तः);
}

पूर्णाङ्क op_pad(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    // Only F32 supported
    size_t type_आकार;
    विकल्प (src0->type) {
        पक्ष HTP_TYPE_F32: type_आकार = 4; भङ्ग;
        स्वभाव:
            FARF(ERROR, "pad-hvx: unsupported type %u\n", src0->type);
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव int32_t lp0 = octx->op_params[0];
    ध्रुव int32_t rp0 = octx->op_params[1];
    ध्रुव int32_t lp1 = octx->op_params[2];
    ध्रुव int32_t rp1 = octx->op_params[3];
    ध्रुव int32_t lp2 = octx->op_params[4];
    ध्रुव int32_t rp2 = octx->op_params[5];
    ध्रुव int32_t lp3 = octx->op_params[6];
    ध्रुव int32_t rp3 = octx->op_params[7];
    ध्रुव int32_t circular = octx->op_params[8];

    ध्रुव uint32_t ne0  = dst->ne[0];
    ध्रुव uint32_t ne00 = src0->ne[0];

    ध्रुव uint32_t योग_dst_पङ्क्तिकाः = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ध्रुव size_t dst_row_आकार     = (size_t)ne0 * type_आकार;

    uint32_t row_start = 0;
    uint32_t nrows     = योग_dst_पङ्क्तिकाः;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = 0;
        htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(dst, type_आकार, (uint32_t) dst_row_आकार, &पङ्क्तिकाः_प्रत्येकम्_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_dst_पङ्क्तिकाः, पङ्क्तिकाः_प्रत्येकम्_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

    ध्रुव size_t src_row_आकार         = (size_t)ne00 * type_आकार;
    ध्रुव size_t src_row_आकार_aligned = hex_आवर्तनम्_ऊर्ध्वम्(src_row_आकार, VLEN);
    ध्रुव size_t dst_row_आकार_aligned = hex_आवर्तनम्_ऊर्ध्वम्(dst_row_आकार, VLEN);

    // Total VTCM needed: 2 buffers (ping+pong) for src and dst, per thread
    ध्रुव size_t vtcm_needed = (size_t)n_तन्तवः * 2 * (src_row_आकार_aligned + dst_row_आकार_aligned);

    ध्रुव पूर्णाङ्क use_dma = (src0->nb[0] == (uint32_t)type_आकार) && (ne00 >= 512) &&
                        (octx->ctx->vtcm_आकार >= vtcm_needed);

    यदि (!use_dma && (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst))) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (use_dma) {
        octx->src0_spad.आकार_प्रत्येकम्_thread = 2 * src_row_आकार_aligned;
        octx->dst_spad.आकार_प्रत्येकम्_thread  = 2 * dst_row_आकार_aligned;
        octx->src0_spad.आकार = n_तन्तवः * octx->src0_spad.आकार_प्रत्येकम्_thread;
        octx->dst_spad.आकार  = n_तन्तवः * octx->dst_spad.आकार_प्रत्येकम्_thread;
        octx->src0_spad.दत्त = octx->ctx->vtcm_आधारः;
        octx->dst_spad.दत्त  = octx->src0_spad.दत्त + octx->src0_spad.आकार;
        octx->src0_spad.src  = रिक्त;
        octx->dst_spad.src   = रिक्त;
    }

    संरचना htp_pad_context pctx = {
        .octx             = octx,
        .lp0 = lp0, .rp0 = rp0,
        .lp1 = lp1, .rp1 = rp1,
        .lp2 = lp2, .rp2 = rp2,
        .lp3 = lp3, .rp3 = rp3,
        .nrows_प्रत्येकम्_thread = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div),
        .योग_dst_पङ्क्तिकाः   = nrows,
        .row_start        = row_start,
        .type_आकार        = type_आकार,
        .src_row_आकार         = src_row_आकार,
        .src_row_आकार_aligned = src_row_आकार_aligned,
        .dst_row_आकार         = dst_row_आकार,
        .dst_row_आकार_aligned = dst_row_आकार_aligned,
    };

    FARF(उच्चः_2, "pad-hvx%s%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) pads=(%d,%d,%d,%d,%d,%d,%d,%d)\n",
         circular ? "-circ" : "",
         use_dma   ? "-dma"  : "",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    यदि      (circular && use_dma) { work_queue_सततम्(octx->ctx->work_queue, pad_job_प्रत्येकम्_thread_hvx_circular_dma, &pctx, n_तन्तवः); }
    अन्यथा यदि (circular)            { work_queue_सततम्(octx->ctx->work_queue, pad_job_प्रत्येकम्_thread_hvx_circular,     &pctx, n_तन्तवः); }
    अन्यथा यदि (use_dma)             { work_queue_सततम्(octx->ctx->work_queue, pad_job_प्रत्येकम्_thread_hvx_dma,          &pctx, n_तन्तवः); }
    अन्यथा                          { work_queue_सततम्(octx->ctx->work_queue, pad_job_प्रत्येकम्_thread_hvx,              &pctx, n_तन्तवः); }

    प्रत्यागम HTP_स्थितिः_OK;
}
