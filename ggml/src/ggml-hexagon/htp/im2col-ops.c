#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাৱ <HAP_farf.h>
#অন্তর্ভাৱ <hexagon_protos.h>
#অন্তর্ভাৱ <hexagon_types.h>
#অন্তর্ভাৱ <string.h>

#অন্তর্ভাৱ "hex-common.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাৱ "ggml-common.h"
#অন্তর্ভাৱ "htp-ctx.h"
#অন্তর্ভাৱ "htp-ops.h"
#অন্তর্ভাৱ "hvx-utils.h"
#অন্তর্ভাৱ "dma-queue.h"
#অন্তর্ভাৱ "hex-profile.h"
#অন্তর্ভাৱ "htp-vtcm.h"
#অন্তর্ভাৱ "htp-tensor.h"
#অন্তর্ভাৱ "im2col-ops.h"

সংরচনা htp_im2col_context {
    সংরচনা htp_ops_context * octx;

    uint32_t pe_row_base;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_rows_per_thread;                   // output rows per worker
    uint32_t pe_src_row_bytes;                     // one output row's source, rounded 256
    uint32_t pe_dst_row_bytes;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_size_per_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_size_per_thread;              // 2 * pe_dst_row_bytes

    uint32_t pe_owb;                               // output-col block size
    uint32_t pe_wb;                                // staged source window width
    uint32_t pe_nbpr;                              // blocks per row

    // Precomputed dimensions for worker threads
    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;

    // Per-thread initial coordinates (removes division from thread prologue)
    uint32_t thread_in[HTP_MAX_NTHREADS];
    uint32_t thread_iod[HTP_MAX_NTHREADS];
    uint32_t thread_ioh[HTP_MAX_NTHREADS];

    শূন্য * compute;
    uint32_t dst_elem;
};

প্রকারনির্দেশ শূন্য (*im2col_blocked_compute_fn_t)(
    ধ্রুৱ সংরচনা htp_im2col_context * ictx,
    শূন্য * restrict dstb, ধ্রুৱ ভিন্ন * restrict srcb,
    ভিন্ন * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

প্রকারনির্দেশ শূন্য (*im2col_patchembed_compute_fn_t)(
    ধ্রুৱ সংরচনা htp_im2col_context * ictx,
    শূন্য * restrict dstb, ধ্রুৱ ভিন্ন * restrict srcb, uint32_t ioh);

প্রকারনির্দেশ শূন্য (*im2col_3d_blocked_compute_fn_t)(
    ধ্রুৱ সংরচনা htp_im2col_context * ictx,
    শূন্য * restrict dstb, ধ্রুৱ ভিন্ন * restrict srcb,
    ভিন্ন * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

প্রকারনির্দেশ শূন্য (*im2col_3d_patchembed_compute_fn_t)(
    ধ্রুৱ সংরচনা htp_im2col_context * ictx,
    শূন্য * restrict dstb, ধ্রুৱ ভিন্ন * restrict srcb);

সংরচনা im2col_cursor {
    uint32_t r;
    uint32_t in;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

স্থির অন্তঃস্থ শূন্য im2col_step_row(সংরচনা im2col_cursor * c, ধ্রুৱ সংরচনা htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    যদি (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->in++;
    }
}

স্থির অন্তঃস্থ শূন্য im2col_step_block(সংরচনা im2col_cursor * c, ধ্রুৱ সংরচনা htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    যদি (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        যদি (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->in++;
        }
    }
}

স্থির শূন্য im2col_blocked_dma_stage(সংরচনা htp_im2col_context * ictx, dma_queue * dma_q,
                                     ভিন্ন * srcb, ধ্রুৱ সংরচনা im2col_cursor * c) {
    ধ্রুৱ uint32_t in = c->in, ioh = c->ioh, c0 = c->c0;
    ধ্রুৱ সংরচনা htp_ops_context * octx = ictx->octx;
    ধ্রুৱ সংরচনা htp_tensor * src1 = octx->src[1];
    ধ্রুৱ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    ধ্রুৱ int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    ধ্রুৱ int32_t d1 = octx->op_params[5];
    ধ্রুৱ bool is_2D = (octx->op_params[6] == 1);
    ধ্রুৱ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ধ্রুৱ uint32_t KH = ictx->KH;
    ধ্রুৱ uint32_t Wb = ictx->pe_wb;
    ধ্রুৱ dma_addr_t src_data = src1->data;
    ধ্রুৱ int32_t win0 = (int32_t) c0 * s0 - p0;
    ধ্রুৱ size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    ধ্রুৱ size_t batch_offset        = is_2D ? (size_t) in * src1->nb[3] : (size_t) in * src1->nb[2];

    বাবে (uint32_t ikh = 0; ikh < KH; ikh++) {
        ধ্রুৱ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        যদি (iih < 0 || iih >= (int32_t) IH) {
            dma_queue_ঠেলো(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            অনুৱর্ত;
        }
        ধ্রুৱ int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        যদি (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        যদি (hi <= lo) {
            dma_queue_ঠেলো(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            অনুৱর্ত;
        }
        ধ্রুৱ uint32_t cpw = (uint32_t) (hi - lo);
        ভিন্ন * vdst = srcb + (size_t) ikh * Wb + lo;
        ধ্রুৱ dma_addr_t vsrc = src_data + batch_offset + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_queue_ঠেলো(dma_q, dma_make_data(vdst, vsrc),
                       (size_t) KH * Wb * পরিমাণ(ভিন্ন), src_channel_stride,
                       cpw * পরিমাণ(ভিন্ন), IC);
    }
}

#define IM2COL_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
স্থির শূন্য FNAME(                                                                               \
    ধ্রুৱ সংরচনা htp_im2col_context * ictx,                                                      \
    শূন্য * restrict dstb_raw, ধ্রুৱ ভিন্ন * restrict srcb,                                       \
    ভিন্ন * restrict scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                          \
    ধ্রুৱ সংরচনা htp_ops_context * octx = ictx->octx;                                            \
    ধ্রুৱ int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    ধ্রুৱ int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    ধ্রুৱ int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    ধ্রুৱ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    ধ্রুৱ uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    ধ্রুৱ uint32_t Wb = ictx->pe_wb;                                                             \
    ধ্রুৱ uint32_t patch_stride = ictx->patch_stride;                                            \
    ধ্রুৱ int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    বাবে (uint32_t j = 0; j < nb; j++) {                                                          \
        ধ্রুৱ uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        ধ্রুৱ int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        বাবে (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            ধ্রুৱ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            ধ্রুৱ পূর্ণাঙ্ক     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            বাবে (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                যদি (!okh) {                                                                      \
                    SPLAT_FN(out_run, 0, KW);                                                    \
                    অনুৱর্ত;                                                                    \
                }                                                                                \
                ধ্রুৱ ভিন্ন * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                যদি (d0 == 1) {                                                                   \
                    ধ্রুৱ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    যদি (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    যদি (hi <= lo) {                                                              \
                        SPLAT_FN(out_run, 0, KW);                                                \
                    } অন্যথা {                                                                     \
                        যদি (lo > 0) {                                                            \
                            SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (out_run + lo),                                      \
                                (ধ্রুৱ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        যদি (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    অনুৱর্ত;                                                                    \
                }                                                                                \
                বাবে (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    ধ্রুৱ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    বাবে (uint32_t c = 0; c < chunk; c++) {                                       \
                        ধ্রুৱ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (out_run + ikw), (ধ্রুৱ uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, ভিন্ন,    hvx_copy_f32_uu,     hvx_splat_f32_u)

স্থির শূন্য im2col_blocked_dma_thread(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    (শূন্য) nth;
    সংরচনা htp_im2col_context * ictx      = (সংরচনা htp_im2col_context *) data;
    সংরচনা htp_ops_context *    octx      = ictx->octx;
    সংরচনা htp_thread_trace * restrict tr = &octx->ctx->trace[ith];
    ধ্রুৱ uint32_t OW                     = ictx->OW;
    ধ্রুৱ uint32_t KH                     = ictx->KH;
    ধ্রুৱ uint32_t owb                    = ictx->pe_owb;
    ধ্রুৱ uint32_t nbpr                   = ictx->pe_nbpr;
    ধ্রুৱ uint32_t patch_stride           = ictx->patch_stride;
    ধ্রুৱ uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    ধ্রুৱ dma_addr_t dst_data             = octx->dst->data;
    dma_queue *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_base              = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    ভিন্ন *        srcb2[2]               = { (ভিন্ন *) srcb_base, (ভিন্ন *) (srcb_base + ictx->pe_src_row_bytes) };
    শূন্য *         dstb2[2]               = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    ভিন্ন *        scratch                = (ভিন্ন *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    ধ্রুৱ uint32_t row_end_max            = ictx->pe_row_base + ictx->pe_nrows;
    ধ্রুৱ uint32_t row_start              = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ধ্রুৱ uint32_t row_end                = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    যদি (row_start >= row_end) {
        ঘূৰাই;
    }
    ধ্রুৱ uint32_t nrows_local  = row_end - row_start;
    ধ্রুৱ uint32_t total_blocks = nrows_local * nbpr;
    যদি (total_blocks == 0) {
        ঘূৰাই;
    }
    সংরচনা im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    সংরচনা im2col_cursor stage = cur;
    বাবে (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_ঠেলো(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_block(&stage, ictx);
    }
    বাবে (uint32_t bi = 0; bi < total_blocks; bi++) {
        ধ্রুৱ uint32_t buf = bi & 1u;
        ভিন্ন * srcb = srcb2[buf];
        শূন্য *  dstb = dstb2[buf];
        ধ্রুৱ uint32_t nb = MIN(owb, OW - cur.c0);
        dma_queue_বেৰ_কৰো(dma_q);
        বাবে (uint32_t k = 0; k < KH; k++) {
            dma_queue_বেৰ_কৰো(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ধ্রুৱ dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_ঠেলো(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        যদি (bi + 2 < total_blocks) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_block(&stage, ictx);
        }
        im2col_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

স্থির শূন্য im2col_patchembed_dma_stage(সংরচনা htp_im2col_context * ictx, dma_queue * dma_q,
                                        ভিন্ন * srcb, ধ্রুৱ সংরচনা im2col_cursor * c) {
    ধ্রুৱ সংরচনা htp_ops_context * octx = ictx->octx;
    ধ্রুৱ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ধ্রুৱ uint32_t KH = ictx->KH;
    ধ্রুৱ uint32_t in = c->in, ioh = c->ioh;
    ধ্রুৱ size_t src_channel_stride  = (size_t) IH * IW * পরিমাণ(ভিন্ন);
    ধ্রুৱ size_t vtcm_channel_stride = (size_t) KH * IW * পরিমাণ(ভিন্ন);
    ধ্রুৱ dma_addr_t src_data        = octx->src[1]->data;
    ধ্রুৱ dma_addr_t vsrc            = src_data + (size_t) (((in * IC) * IH + ioh * KH) * IW) * পরিমাণ(ভিন্ন);

    dma_queue_ঠেলো(dma_q, dma_make_data(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

স্থির শূন্য im2col_patchembed_compute_f16(
    ধ্রুৱ সংরচনা htp_im2col_context * ictx,
    শূন্য * restrict dstb_raw, ধ্রুৱ ভিন্ন * restrict srcb, uint32_t ioh) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    ধ্রুৱ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ধ্রুৱ uint32_t KH = ictx->KH, KW = ictx->KW;
    ধ্রুৱ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    বাবে (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        বাবে (uint32_t ikh = 0; ikh < KH; ikh++) {
            ধ্রুৱ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            বাবে (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                যদি (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(out_run, 0, KW);
                    অনুৱর্ত;
                }
                ধ্রুৱ ভিন্ন * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) out_run, (ধ্রুৱ uint8_t *) src_run, KW);
            }
        }
    }
}

স্থির শূন্য im2col_patchembed_compute_f32(
    ধ্রুৱ সংরচনা htp_im2col_context * ictx,
    শূন্য * restrict dstb_raw, ধ্রুৱ ভিন্ন * restrict srcb, uint32_t ioh) {
    ভিন্ন * restrict dstb = (ভিন্ন *) dstb_raw;
    ধ্রুৱ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ধ্রুৱ uint32_t KH = ictx->KH, KW = ictx->KW;
    ধ্রুৱ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    বাবে (uint32_t iow = 0; iow < OW; iow++) {
        ভিন্ন * dst_patch = dstb + (size_t) iow * patch_stride;
        বাবে (uint32_t ikh = 0; ikh < KH; ikh++) {
            ধ্রুৱ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            বাবে (uint32_t iic = 0; iic < IC; iic++) {
                ভিন্ন * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                যদি (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(out_run, 0, KW);
                    অনুৱর্ত;
                }
                ধ্রুৱ ভিন্ন * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) out_run, (ধ্রুৱ uint8_t *) src_run, KW);
            }
        }
    }
}

স্থির শূন্য im2col_patchembed_dma_thread(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    (শূন্য) nth;
    সংরচনা htp_im2col_context * ictx        = (সংরচনা htp_im2col_context *) data;
    সংরচনা htp_ops_context *    octx        = ictx->octx;
    সংরচনা htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    ধ্রুৱ uint32_t OW                       = ictx->OW;
    ধ্রুৱ uint32_t patch_stride             = ictx->patch_stride;
    ধ্রুৱ uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    ধ্রুৱ dma_addr_t dst_data               = octx->dst->data;
    dma_queue *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_base                = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    ভিন্ন *        srcb2[2]                 = { (ভিন্ন *) srcb_base, (ভিন্ন *) (srcb_base + ictx->pe_src_row_bytes) };
    শূন্য *         dstb2[2]                 = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    ধ্রুৱ uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;
    ধ্রুৱ uint32_t row_start                = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ধ্রুৱ uint32_t row_end                  = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    যদি (row_start >= row_end) {
        ঘূৰাই;
    }
    ধ্রুৱ uint32_t total_rows = row_end - row_start;
    যদি (total_rows == 0) {
        ঘূৰাই;
    }
    সংরচনা im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    সংরচনা im2col_cursor stage = cur;
    বাবে (uint32_t i = 0; i < total_rows && i < 2; i++) {
        dma_queue_ঠেলো(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_row(&stage, ictx);
    }
    বাবে (uint32_t ri = 0; ri < total_rows; ri++) {
        ধ্রুৱ uint32_t buf = ri & 1u;
        ভিন্ন * srcb = srcb2[buf];
        শূন্য *  dstb = dstb2[buf];
        dma_queue_বেৰ_কৰো(dma_q);
        dma_queue_বেৰ_কৰো(dma_q);
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ধ্রুৱ dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_ঠেলো(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        যদি (ri + 2 < total_rows) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_row(&stage, ictx);
        }
        im2col_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

স্থির অন্তঃস্থ শূন্য im2col_3d_step_row(সংরচনা im2col_cursor * c, ধ্রুৱ সংরচনা htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    যদি (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        যদি (c->iod >= ictx->OD) {
            c->iod = 0;
            c->in++;
        }
    }
}

স্থির অন্তঃস্থ শূন্য im2col_3d_step_block(সংরচনা im2col_cursor * c, ধ্রুৱ সংরচনা htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    যদি (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        যদি (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            যদি (c->iod >= ictx->OD) {
                c->iod = 0;
                c->in++;
            }
        }
    }
}

স্থির শূন্য im2col_3d_blocked_dma_stage(সংরচনা htp_im2col_context * ictx, dma_queue * dma_q,
                                        ভিন্ন * srcb, ধ্রুৱ সংরচনা im2col_cursor * c) {
    ধ্রুৱ uint32_t in = c->in, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    ধ্রুৱ সংরচনা htp_ops_context * octx = ictx->octx;
    ধ্রুৱ সংরচনা htp_tensor * src1 = octx->src[1];
    ধ্রুৱ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    ধ্রুৱ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    ধ্রুৱ int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    ধ্রুৱ uint32_t IC = ictx->IC;
    ধ্রুৱ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ধ্রুৱ uint32_t KD = ictx->KD, KH = ictx->KH;
    ধ্রুৱ uint32_t Wb = ictx->pe_wb;
    ধ্রুৱ int32_t win0 = (int32_t) c0 * s0 - p0;
    ধ্রুৱ dma_addr_t src_data = src1->data;
    ধ্রুৱ size_t src_channel_stride = src1->nb[3];
    ধ্রুৱ size_t vtcm_channel_stride = (size_t) KD * KH * Wb * পরিমাণ(ভিন্ন);
    ধ্রুৱ size_t batch_offset        = (size_t) (in * IC) * src1->nb[3];

    বাবে (uint32_t ikd = 0; ikd < KD; ikd++) {
        ধ্রুৱ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        ধ্রুৱ bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        বাবে (uint32_t ikh = 0; ikh < KH; ikh++) {
            ধ্রুৱ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            যদি (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_queue_ঠেলো(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                অনুৱর্ত;
            }
            ধ্রুৱ int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            যদি (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            যদি (hi <= lo) {
                dma_queue_ঠেলো(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                অনুৱর্ত;
            }
            ভিন্ন * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            ধ্রুৱ dma_addr_t vsrc = src_data + batch_offset +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_queue_ঠেলো(dma_q, dma_make_data(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * পরিমাণ(ভিন্ন), IC);
        }
    }
}

#define IM2COL_3D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
স্থির শূন্য FNAME(ধ্রুৱ সংরচনা htp_im2col_context * ictx,                                            \
                  শূন্য * restrict dstb_raw, ধ্রুৱ ভিন্ন * restrict srcb,                             \
                  ভিন্ন * restrict scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                              \
    ধ্রুৱ সংরচনা htp_ops_context * octx = ictx->octx;                                                \
    ধ্রুৱ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    ধ্রুৱ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    ধ্রুৱ int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    ধ্রুৱ uint32_t IC = ictx->IC;                                                                    \
    ধ্রুৱ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    ধ্রুৱ uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    ধ্রুৱ uint32_t patch_stride = ictx->patch_stride;                                                \
    ধ্রুৱ uint32_t Wb = ictx->pe_wb;                                                                 \
    ধ্রুৱ int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    বাবে (uint32_t j = 0; j < nb; j++) {                                                              \
        ধ্রুৱ uint32_t iow = c0 + j;                                                                 \
        ধ্রুৱ int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        বাবে (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            বাবে (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                ধ্রুৱ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                বাবে (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    ধ্রুৱ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * out_run = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    যদি (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(out_run, 0, KW);                                                    \
                        অনুৱর্ত;                                                                    \
                    }                                                                                \
                    ধ্রুৱ ভিন্ন * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    যদি (d0 == 1) {                                                                   \
                        ধ্রুৱ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        যদি (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        যদি (hi <= lo) {                                                              \
                            SPLAT_FN(out_run, 0, KW);                                                \
                        } অন্যথা {                                                                     \
                            যদি (lo > 0) {                                                            \
                                SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (out_run + lo),                                      \
                                    (ধ্রুৱ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            যদি (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        অনুৱর্ত;                                                                    \
                    }                                                                                \
                    বাবে (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        ধ্রুৱ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        বাবে (uint32_t c = 0; c < chunk; c++) {                                       \
                            ধ্রুৱ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (out_run + ikw), (ধ্রুৱ uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, ভিন্ন,    hvx_copy_f32_uu,     hvx_splat_f32_u)

স্থির শূন্য im2col_3d_blocked_dma_thread(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    (শূন্য) nth;
    সংরচনা htp_im2col_context * ictx        = (সংরচনা htp_im2col_context *) data;
    সংরচনা htp_ops_context *    octx        = ictx->octx;
    সংরচনা htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    ধ্রুৱ সংরচনা htp_tensor * restrict dst  = octx->dst;
    ধ্রুৱ uint32_t OW           = ictx->OW;
    ধ্রুৱ uint32_t KD = ictx->KD, KH = ictx->KH;
    ধ্রুৱ uint32_t kd_kh        = KD * KH;
    ধ্রুৱ uint32_t patch_stride = ictx->patch_stride;
    ধ্রুৱ uint32_t owb          = ictx->pe_owb;
    ধ্রুৱ uint32_t nbpr         = ictx->pe_nbpr;
    ধ্রুৱ uint32_t dst_elem     = ictx->dst_elem;
    ধ্রুৱ dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    ভিন্ন * srcb2[2]            = { (ভিন্ন *) srcb_base, (ভিন্ন *) (srcb_base + ictx->pe_src_row_bytes) };
    শূন্য * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    ভিন্ন * scratch             = (ভিন্ন *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    ধ্রুৱ uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    ধ্রুৱ uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ধ্রুৱ uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    যদি (row_start >= row_end) {
        ঘূৰাই;
    }
    ধ্রুৱ uint32_t nrows_local  = row_end - row_start;
    ধ্রুৱ uint32_t total_blocks = nrows_local * nbpr;
    যদি (total_blocks == 0) {
        ঘূৰাই;
    }
    সংরচনা im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    সংরচনা im2col_cursor stage = cur;
    বাবে (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_ঠেলো(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_block(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    বাবে (uint32_t bi = 0; bi < total_blocks; bi++) {
        ধ্রুৱ uint32_t buf = bi & 1u;
        ভিন্ন * srcb = srcb2[buf];
        শূন্য *  dstb = dstb2[buf];
        ধ্রুৱ uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_queue_বেৰ_কৰো(dma_q);
        বাবে (uint32_t p = 0; p < kd_kh; p++) {
            dma_queue_বেৰ_কৰো(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ধ্রুৱ dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_ঠেলো(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        যদি (bi + 2 < total_blocks) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_block(&stage, ictx);
        }
        im2col_3d_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}
স্থির শূন্য im2col_3d_patchembed_dma_stage(সংরচনা htp_im2col_context * ictx, dma_queue * dma_q,
                                           ভিন্ন * srcb, ধ্রুৱ সংরচনা im2col_cursor * c) {
    ধ্রুৱ সংরচনা htp_ops_context * octx = ictx->octx;
    ধ্রুৱ uint32_t IC = ictx->IC;
    ধ্রুৱ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ধ্রুৱ uint32_t KD = ictx->KD, KH = ictx->KH;
    ধ্রুৱ dma_addr_t src_data = octx->src[1]->data;
    ধ্রুৱ uint32_t in = c->in, iod = c->iod, ioh = c->ioh;
    ধ্রুৱ size_t src_channel_stride  = (size_t) ID * IH * IW * পরিমাণ(ভিন্ন);
    ধ্রুৱ size_t vtcm_channel_stride = (size_t) KD * KH * IW * পরিমাণ(ভিন্ন);
    ধ্রুৱ size_t row_size            = (size_t) KH * IW * পরিমাণ(ভিন্ন);

    বাবে (uint32_t ikd = 0; ikd < KD; ikd++) {
        ধ্রুৱ uint32_t iid = iod * KD + ikd;
        ভিন্ন * vdst = srcb + (size_t) ikd * KH * IW;
        ধ্রুৱ dma_addr_t vsrc = src_data + (size_t) (((in * IC) * ID + iid) * IH + ioh * KH) * IW * পরিমাণ(ভিন্ন);
        dma_queue_ঠেলো(dma_q, dma_make_data(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, row_size, IC);
    }
}
স্থির শূন্য im2col_3d_patchembed_compute_f16(
    ধ্রুৱ সংরচনা htp_im2col_context * ictx,
    শূন্য * restrict dstb_raw, ধ্রুৱ ভিন্ন * restrict srcb) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    ধ্রুৱ uint32_t OW           = ictx->OW;
    ধ্রুৱ uint32_t IC           = ictx->IC;
    ধ্রুৱ uint32_t KD           = ictx->KD;
    ধ্রুৱ uint32_t KH           = ictx->KH;
    ধ্রুৱ uint32_t KW           = ictx->KW;
    ধ্রুৱ uint32_t IW           = ictx->IW;
    ধ্রুৱ uint32_t patch_stride = ictx->patch_stride;

    বাবে (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        বাবে (uint32_t iic = 0; iic < IC; iic++) {
            বাবে (uint32_t ikd = 0; ikd < KD; ikd++) {
                বাবে (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ধ্রুৱ ভিন্ন * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) out_run, (ধ্রুৱ uint8_t *) src_run, KW);
                }
            }
        }
    }
}

স্থির শূন্য im2col_3d_patchembed_compute_f32(
    ধ্রুৱ সংরচনা htp_im2col_context * ictx,
    শূন্য * restrict dstb_raw, ধ্রুৱ ভিন্ন * restrict srcb) {
    ভিন্ন * restrict dstb = (ভিন্ন *) dstb_raw;
    ধ্রুৱ uint32_t OW           = ictx->OW;
    ধ্রুৱ uint32_t IC           = ictx->IC;
    ধ্রুৱ uint32_t KD           = ictx->KD;
    ধ্রুৱ uint32_t KH           = ictx->KH;
    ধ্রুৱ uint32_t KW           = ictx->KW;
    ধ্রুৱ uint32_t IW           = ictx->IW;
    ধ্রুৱ uint32_t patch_stride = ictx->patch_stride;

    বাবে (uint32_t iow = 0; iow < OW; iow++) {
        ভিন্ন * dst_patch = dstb + (size_t) iow * patch_stride;
        বাবে (uint32_t iic = 0; iic < IC; iic++) {
            বাবে (uint32_t ikd = 0; ikd < KD; ikd++) {
                বাবে (uint32_t ikh = 0; ikh < KH; ikh++) {
                    ভিন্ন * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ধ্রুৱ ভিন্ন * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) out_run, (ধ্রুৱ uint8_t *) src_run, KW);
                }
            }
        }
    }
}

স্থির শূন্য im2col_3d_patchembed_dma_thread(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    (শূন্য) nth;
    সংরচনা htp_im2col_context * ictx        = (সংরচনা htp_im2col_context *) data;
    সংরচনা htp_ops_context *    octx        = ictx->octx;
    সংরচনা htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    ধ্রুৱ সংরচনা htp_tensor * restrict dst  = octx->dst;
    ধ্রুৱ uint32_t KD           = ictx->KD;
    ধ্রুৱ uint32_t OW           = ictx->OW;
    ধ্রুৱ uint32_t patch_stride = ictx->patch_stride;
    ধ্রুৱ uint32_t dst_elem     = ictx->dst_elem;
    ধ্রুৱ dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    ধ্রুৱ uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    ধ্রুৱ uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ধ্রুৱ uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    যদি (row_start >= row_end) {
        ঘূৰাই;
    }

    ধ্রুৱ uint32_t nrows_local  = row_end - row_start;
    uint8_t * src_base          = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    ভিন্ন * srcb2[2]            = { (ভিন্ন *) src_base, (ভিন্ন *) (src_base + ictx->pe_src_row_bytes) };
    শূন্য * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    সংরচনা im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    সংরচনা im2col_cursor stage = cur;
    বাবে (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_queue_ঠেলো(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_row(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    বাবে (uint32_t idx = 0; idx < nrows_local; idx++) {
        ধ্রুৱ uint32_t buf = idx & 1u;
        ভিন্ন * srcb = srcb2[buf];
        শূন্য *  dstb = dstb2[buf];
        dma_queue_বেৰ_কৰো(dma_q);
        বাবে (uint32_t p = 0; p < KD; p++) {
            dma_queue_বেৰ_কৰো(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ধ্রুৱ dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_ঠেলো(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        যদি (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_row(&stage, ictx);
        }
        im2col_3d_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

পূর্ণাঙ্ক op_im2col(সংরচনা htp_ops_context * octx) {
    ধ্রুৱ সংরচনা htp_im2col_kernel_params * kparams = (ধ্রুৱ সংরচনা htp_im2col_kernel_params *) octx->kernel_params;
    যদি (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        ঘূৰাই HTP_STATUS_NO_SUPPORT;
    }

    যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ঘূৰাই HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুৱ সংরচনা htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুৱ uint32_t row_size = dst->nb[2];
        ধ্রুৱ uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_সাৰি_SIZE / hex_gcd_u32(row_size, HEX_L2_সাৰি_SIZE)) : 1;
        ধ্রুৱ সংরচনা htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_base = range.start;
        nrows    = range.গুনতি;
    }

    যদি (nrows == 0) {
        ঘূৰাই HTP_STATUS_OK;
    }

    সংরচনা htp_im2col_context ictx = { 0 };
    uint8_t * ধ্রুৱ base = octx->ctx->vtcm_base;

    ধ্রুৱ uint32_t pth = MIN(octx->n_threads, nrows);
    ictx.octx                   = octx;
    ictx.pe_row_base            = row_base;
    ictx.pe_nrows               = nrows;
    ictx.pe_rows_per_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_row_bytes       = kparams->src_row_bytes;
    ictx.pe_dst_row_bytes       = kparams->dst_row_bytes;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_dst);
    ictx.pe_src_size_per_thread = kparams->src_bytes_per_thread;
    ictx.pe_dst_size_per_thread = kparams->dst_bytes_per_thread;
    ictx.pe_owb                 = kparams->owb;
    ictx.pe_wb                  = kparams->wb;
    ictx.pe_nbpr                = kparams->nbpr;
    ictx.N                      = kparams->N;
    ictx.IC                     = kparams->IC;
    ictx.ID                     = kparams->ID;
    ictx.IH                     = kparams->IH;
    ictx.IW                     = kparams->IW;
    ictx.KD                     = kparams->KD;
    ictx.KH                     = kparams->KH;
    ictx.KW                     = kparams->KW;
    ictx.OD                     = kparams->OD;
    ictx.OH                     = kparams->OH;
    ictx.OW                     = kparams->OW;
    ictx.patch_stride           = kparams->patch_stride;

    বাবে (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_in[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? পরিমাণ(__fp16) : পরিমাণ(ভিন্ন);

    যদি (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (শূন্য *) im2col_patchembed_compute_f16
                                                 : (শূন্য *) im2col_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_dma_thread, &ictx, pth);
        ঘূৰাই HTP_STATUS_OK;
    } অন্যথা যদি (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (শূন্য *) im2col_blocked_compute_f16
                                                 : (শূন্য *) im2col_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_blocked_dma_thread, &ictx, pth);
        ঘূৰাই HTP_STATUS_OK;
    }

    ঘূৰাই HTP_STATUS_NO_SUPPORT;
}

পূর্ণাঙ্ক op_im2col_3d(সংরচনা htp_ops_context * octx) {
    ধ্রুৱ সংরচনা htp_im2col_kernel_params * kparams =
        (ধ্রুৱ সংরচনা htp_im2col_kernel_params *) octx->kernel_params;
    যদি (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        ঘূৰাই HTP_STATUS_NO_SUPPORT;
    }

    যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ঘূৰাই HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুৱ সংরচনা htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুৱ uint32_t row_size = dst->nb[2];
        ধ্রুৱ uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_সাৰি_SIZE / hex_gcd_u32(row_size, HEX_L2_সাৰি_SIZE)) : 1;
        ধ্রুৱ সংরচনা htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_base = range.start;
        nrows    = range.গুনতি;
    }

    যদি (nrows == 0) {
        ঘূৰাই HTP_STATUS_OK;
    }

    সংরচনা htp_im2col_context ictx = { 0 };
    uint8_t * ধ্রুৱ base = octx->ctx->vtcm_base;

    ধ্রুৱ uint32_t pth = MIN(octx->n_threads, nrows);
    ictx.octx                   = octx;
    ictx.pe_row_base            = row_base;
    ictx.pe_nrows               = nrows;
    ictx.pe_rows_per_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_row_bytes       = kparams->src_row_bytes;
    ictx.pe_dst_row_bytes       = kparams->dst_row_bytes;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_dst);
    ictx.pe_src_size_per_thread = kparams->src_bytes_per_thread;
    ictx.pe_dst_size_per_thread = kparams->dst_bytes_per_thread;
    ictx.pe_owb                 = kparams->owb;
    ictx.pe_wb                  = kparams->wb;
    ictx.pe_nbpr                = kparams->nbpr;
    ictx.N                      = kparams->N;
    ictx.IC                     = kparams->IC;
    ictx.ID                     = kparams->ID;
    ictx.IH                     = kparams->IH;
    ictx.IW                     = kparams->IW;
    ictx.KD                     = kparams->KD;
    ictx.KH                     = kparams->KH;
    ictx.KW                     = kparams->KW;
    ictx.OD                     = kparams->OD;
    ictx.OH                     = kparams->OH;
    ictx.OW                     = kparams->OW;
    ictx.patch_stride           = kparams->patch_stride;

    বাবে (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_in[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? পরিমাণ(__fp16) : পরিমাণ(ভিন্ন);

    যদি (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (শূন্য *) im2col_3d_patchembed_compute_f16
                                                 : (শূন্য *) im2col_3d_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_patchembed_dma_thread, &ictx, pth);
        ঘূৰাই HTP_STATUS_OK;
    } অন্যথা যদি (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (শূন্য *) im2col_3d_blocked_compute_f16
                                                 : (শূন্য *) im2col_3d_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_blocked_dma_thread, &ictx, pth);
        ঘূৰাই HTP_STATUS_OK;
    }

    ঘূৰাই HTP_STATUS_NO_SUPPORT;
}
