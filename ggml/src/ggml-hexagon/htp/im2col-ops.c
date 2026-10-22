#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <hexagon_protos.h>
#ಅನ್ತರ್ಭಾವ <hexagon_types.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "im2col-ops.h"

ಸಂರಚನಾ htp_im2col_context {
    ಸಂರಚನಾ htp_ops_context * octx;

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

    ಶೂನ್ಯ * compute;
    uint32_t dst_elem;
};

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*im2col_blocked_compute_fn_t)(
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,
    ಶೂನ್ಯ * restrict dstb, ಧ್ರುವ ಭಿನ್ನ * restrict srcb,
    ಭಿನ್ನ * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*im2col_patchembed_compute_fn_t)(
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,
    ಶೂನ್ಯ * restrict dstb, ಧ್ರುವ ಭಿನ್ನ * restrict srcb, uint32_t ioh);

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*im2col_3d_blocked_compute_fn_t)(
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,
    ಶೂನ್ಯ * restrict dstb, ಧ್ರುವ ಭಿನ್ನ * restrict srcb,
    ಭಿನ್ನ * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*im2col_3d_patchembed_compute_fn_t)(
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,
    ಶೂನ್ಯ * restrict dstb, ಧ್ರುವ ಭಿನ್ನ * restrict srcb);

ಸಂರಚನಾ im2col_cursor {
    uint32_t r;
    uint32_t in;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ im2col_step_row(ಸಂರಚನಾ im2col_cursor * c, ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ಯದಿ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->in++;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ im2col_step_block(ಸಂರಚನಾ im2col_cursor * c, ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ಯದಿ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ಯದಿ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->in++;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_blocked_dma_stage(ಸಂರಚನಾ htp_im2col_context * ictx, dma_queue * dma_q,
                                     ಭಿನ್ನ * srcb, ಧ್ರುವ ಸಂರಚನಾ im2col_cursor * c) {
    ಧ್ರುವ uint32_t in = c->in, ioh = c->ioh, c0 = c->c0;
    ಧ್ರುವ ಸಂರಚನಾ htp_ops_context * octx = ictx->octx;
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src1 = octx->src[1];
    ಧ್ರುವ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    ಧ್ರುವ int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    ಧ್ರುವ int32_t d1 = octx->op_params[5];
    ಧ್ರುವ bool is_2D = (octx->op_params[6] == 1);
    ಧ್ರುವ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ಧ್ರುವ uint32_t KH = ictx->KH;
    ಧ್ರುವ uint32_t Wb = ictx->pe_wb;
    ಧ್ರುವ dma_addr_t src_data = src1->data;
    ಧ್ರುವ int32_t win0 = (int32_t) c0 * s0 - p0;
    ಧ್ರುವ size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    ಧ್ರುವ size_t batch_offset        = is_2D ? (size_t) in * src1->nb[3] : (size_t) in * src1->nb[2];

    ಪ್ರತಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
        ಧ್ರುವ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        ಯದಿ (iih < 0 || iih >= (int32_t) IH) {
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            ಅನುವರ್ತ;
        }
        ಧ್ರುವ int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        ಯದಿ (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        ಯದಿ (hi <= lo) {
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            ಅನುವರ್ತ;
        }
        ಧ್ರುವ uint32_t cpw = (uint32_t) (hi - lo);
        ಭಿನ್ನ * vdst = srcb + (size_t) ikh * Wb + lo;
        ಧ್ರುವ dma_addr_t vsrc = src_data + batch_offset + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(vdst, vsrc),
                       (size_t) KH * Wb * ಪರಿಮಾಣ(ಭಿನ್ನ), src_channel_stride,
                       cpw * ಪರಿಮಾಣ(ಭಿನ್ನ), IC);
    }
}

#define IM2COL_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
ಸ್ಥಿರ ಶೂನ್ಯ FNAME(                                                                               \
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,                                                      \
    ಶೂನ್ಯ * restrict dstb_raw, ಧ್ರುವ ಭಿನ್ನ * restrict srcb,                                       \
    ಭಿನ್ನ * restrict scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                          \
    ಧ್ರುವ ಸಂರಚನಾ htp_ops_context * octx = ictx->octx;                                            \
    ಧ್ರುವ int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    ಧ್ರುವ int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    ಧ್ರುವ int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    ಧ್ರುವ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    ಧ್ರುವ uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    ಧ್ರುವ uint32_t Wb = ictx->pe_wb;                                                             \
    ಧ್ರುವ uint32_t patch_stride = ictx->patch_stride;                                            \
    ಧ್ರುವ int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    ಪ್ರತಿ (uint32_t j = 0; j < nb; j++) {                                                          \
        ಧ್ರುವ uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        ಧ್ರುವ int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        ಪ್ರತಿ (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            ಧ್ರುವ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            ಪ್ರತಿ (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                ಯದಿ (!okh) {                                                                      \
                    SPLAT_FN(out_run, 0, KW);                                                    \
                    ಅನುವರ್ತ;                                                                    \
                }                                                                                \
                ಧ್ರುವ ಭಿನ್ನ * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                ಯದಿ (d0 == 1) {                                                                   \
                    ಧ್ರುವ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    ಯದಿ (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    ಯದಿ (hi <= lo) {                                                              \
                        SPLAT_FN(out_run, 0, KW);                                                \
                    } ಅನ್ಯಥಾ {                                                                     \
                        ಯದಿ (lo > 0) {                                                            \
                            SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (out_run + lo),                                      \
                                (ಧ್ರುವ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        ಯದಿ (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    ಅನುವರ್ತ;                                                                    \
                }                                                                                \
                ಪ್ರತಿ (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    ಧ್ರುವ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    ಪ್ರತಿ (uint32_t c = 0; c < chunk; c++) {                                       \
                        ಧ್ರುವ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (out_run + ikw), (ಧ್ರುವ uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, ಭಿನ್ನ,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ಸ್ಥಿರ ಶೂನ್ಯ im2col_blocked_dma_thread(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    (ಶೂನ್ಯ) nth;
    ಸಂರಚನಾ htp_im2col_context * ictx      = (ಸಂರಚನಾ htp_im2col_context *) data;
    ಸಂರಚನಾ htp_ops_context *    octx      = ictx->octx;
    ಸಂರಚನಾ htp_thread_trace * restrict tr = &octx->ctx->trace[ith];
    ಧ್ರುವ uint32_t OW                     = ictx->OW;
    ಧ್ರುವ uint32_t KH                     = ictx->KH;
    ಧ್ರುವ uint32_t owb                    = ictx->pe_owb;
    ಧ್ರುವ uint32_t nbpr                   = ictx->pe_nbpr;
    ಧ್ರುವ uint32_t patch_stride           = ictx->patch_stride;
    ಧ್ರುವ uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    ಧ್ರುವ dma_addr_t dst_data             = octx->dst->data;
    dma_queue *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_base              = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    ಭಿನ್ನ *        srcb2[2]               = { (ಭಿನ್ನ *) srcb_base, (ಭಿನ್ನ *) (srcb_base + ictx->pe_src_row_bytes) };
    ಶೂನ್ಯ *         dstb2[2]               = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    ಭಿನ್ನ *        scratch                = (ಭಿನ್ನ *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    ಧ್ರುವ uint32_t row_end_max            = ictx->pe_row_base + ictx->pe_nrows;
    ಧ್ರುವ uint32_t row_start              = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ಧ್ರುವ uint32_t row_end                = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    ಯದಿ (row_start >= row_end) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಧ್ರುವ uint32_t nrows_local  = row_end - row_start;
    ಧ್ರುವ uint32_t total_blocks = nrows_local * nbpr;
    ಯದಿ (total_blocks == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಸಂರಚನಾ im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ಸಂರಚನಾ im2col_cursor stage = cur;
    ಪ್ರತಿ (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_block(&stage, ictx);
    }
    ಪ್ರತಿ (uint32_t bi = 0; bi < total_blocks; bi++) {
        ಧ್ರುವ uint32_t buf = bi & 1u;
        ಭಿನ್ನ * srcb = srcb2[buf];
        ಶೂನ್ಯ *  dstb = dstb2[buf];
        ಧ್ರುವ uint32_t nb = MIN(owb, OW - cur.c0);
        dma_queue_ತೆಗೆ(dma_q);
        ಪ್ರತಿ (uint32_t k = 0; k < KH; k++) {
            dma_queue_ತೆಗೆ(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ಧ್ರುವ dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ಯದಿ (bi + 2 < total_blocks) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_block(&stage, ictx);
        }
        im2col_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_patchembed_dma_stage(ಸಂರಚನಾ htp_im2col_context * ictx, dma_queue * dma_q,
                                        ಭಿನ್ನ * srcb, ಧ್ರುವ ಸಂರಚನಾ im2col_cursor * c) {
    ಧ್ರುವ ಸಂರಚನಾ htp_ops_context * octx = ictx->octx;
    ಧ್ರುವ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ಧ್ರುವ uint32_t KH = ictx->KH;
    ಧ್ರುವ uint32_t in = c->in, ioh = c->ioh;
    ಧ್ರುವ size_t src_channel_stride  = (size_t) IH * IW * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ size_t vtcm_channel_stride = (size_t) KH * IW * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ dma_addr_t src_data        = octx->src[1]->data;
    ಧ್ರುವ dma_addr_t vsrc            = src_data + (size_t) (((in * IC) * IH + ioh * KH) * IW) * ಪರಿಮಾಣ(ಭಿನ್ನ);

    dma_queue_ತಳ್ಳು(dma_q, dma_make_data(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_patchembed_compute_f16(
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,
    ಶೂನ್ಯ * restrict dstb_raw, ಧ್ರುವ ಭಿನ್ನ * restrict srcb, uint32_t ioh) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    ಧ್ರುವ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ಧ್ರುವ uint32_t KH = ictx->KH, KW = ictx->KW;
    ಧ್ರುವ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    ಪ್ರತಿ (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        ಪ್ರತಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
            ಧ್ರುವ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            ಪ್ರತಿ (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ಯದಿ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(out_run, 0, KW);
                    ಅನುವರ್ತ;
                }
                ಧ್ರುವ ಭಿನ್ನ * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) out_run, (ಧ್ರುವ uint8_t *) src_run, KW);
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_patchembed_compute_f32(
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,
    ಶೂನ್ಯ * restrict dstb_raw, ಧ್ರುವ ಭಿನ್ನ * restrict srcb, uint32_t ioh) {
    ಭಿನ್ನ * restrict dstb = (ಭಿನ್ನ *) dstb_raw;
    ಧ್ರುವ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ಧ್ರುವ uint32_t KH = ictx->KH, KW = ictx->KW;
    ಧ್ರುವ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    ಪ್ರತಿ (uint32_t iow = 0; iow < OW; iow++) {
        ಭಿನ್ನ * dst_patch = dstb + (size_t) iow * patch_stride;
        ಪ್ರತಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
            ಧ್ರುವ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            ಪ್ರತಿ (uint32_t iic = 0; iic < IC; iic++) {
                ಭಿನ್ನ * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ಯದಿ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(out_run, 0, KW);
                    ಅನುವರ್ತ;
                }
                ಧ್ರುವ ಭಿನ್ನ * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) out_run, (ಧ್ರುವ uint8_t *) src_run, KW);
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_patchembed_dma_thread(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    (ಶೂನ್ಯ) nth;
    ಸಂರಚನಾ htp_im2col_context * ictx        = (ಸಂರಚನಾ htp_im2col_context *) data;
    ಸಂರಚನಾ htp_ops_context *    octx        = ictx->octx;
    ಸಂರಚನಾ htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    ಧ್ರುವ uint32_t OW                       = ictx->OW;
    ಧ್ರುವ uint32_t patch_stride             = ictx->patch_stride;
    ಧ್ರುವ uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    ಧ್ರುವ dma_addr_t dst_data               = octx->dst->data;
    dma_queue *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_base                = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    ಭಿನ್ನ *        srcb2[2]                 = { (ಭಿನ್ನ *) srcb_base, (ಭಿನ್ನ *) (srcb_base + ictx->pe_src_row_bytes) };
    ಶೂನ್ಯ *         dstb2[2]                 = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    ಧ್ರುವ uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;
    ಧ್ರುವ uint32_t row_start                = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ಧ್ರುವ uint32_t row_end                  = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    ಯದಿ (row_start >= row_end) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಧ್ರುವ uint32_t total_rows = row_end - row_start;
    ಯದಿ (total_rows == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಸಂರಚನಾ im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ಸಂರಚನಾ im2col_cursor stage = cur;
    ಪ್ರತಿ (uint32_t i = 0; i < total_rows && i < 2; i++) {
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_row(&stage, ictx);
    }
    ಪ್ರತಿ (uint32_t ri = 0; ri < total_rows; ri++) {
        ಧ್ರುವ uint32_t buf = ri & 1u;
        ಭಿನ್ನ * srcb = srcb2[buf];
        ಶೂನ್ಯ *  dstb = dstb2[buf];
        dma_queue_ತೆಗೆ(dma_q);
        dma_queue_ತೆಗೆ(dma_q);
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ಧ್ರುವ dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ಯದಿ (ri + 2 < total_rows) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_row(&stage, ictx);
        }
        im2col_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ im2col_3d_step_row(ಸಂರಚನಾ im2col_cursor * c, ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ಯದಿ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        ಯದಿ (c->iod >= ictx->OD) {
            c->iod = 0;
            c->in++;
        }
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ im2col_3d_step_block(ಸಂರಚನಾ im2col_cursor * c, ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ಯದಿ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ಯದಿ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            ಯದಿ (c->iod >= ictx->OD) {
                c->iod = 0;
                c->in++;
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_blocked_dma_stage(ಸಂರಚನಾ htp_im2col_context * ictx, dma_queue * dma_q,
                                        ಭಿನ್ನ * srcb, ಧ್ರುವ ಸಂರಚನಾ im2col_cursor * c) {
    ಧ್ರುವ uint32_t in = c->in, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    ಧ್ರುವ ಸಂರಚನಾ htp_ops_context * octx = ictx->octx;
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src1 = octx->src[1];
    ಧ್ರುವ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    ಧ್ರುವ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    ಧ್ರುವ int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    ಧ್ರುವ uint32_t IC = ictx->IC;
    ಧ್ರುವ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ಧ್ರುವ uint32_t KD = ictx->KD, KH = ictx->KH;
    ಧ್ರುವ uint32_t Wb = ictx->pe_wb;
    ಧ್ರುವ int32_t win0 = (int32_t) c0 * s0 - p0;
    ಧ್ರುವ dma_addr_t src_data = src1->data;
    ಧ್ರುವ size_t src_channel_stride = src1->nb[3];
    ಧ್ರುವ size_t vtcm_channel_stride = (size_t) KD * KH * Wb * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ size_t batch_offset        = (size_t) (in * IC) * src1->nb[3];

    ಪ್ರತಿ (uint32_t ikd = 0; ikd < KD; ikd++) {
        ಧ್ರುವ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        ಧ್ರುವ bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        ಪ್ರತಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
            ಧ್ರುವ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            ಯದಿ (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_queue_ತಳ್ಳು(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                ಅನುವರ್ತ;
            }
            ಧ್ರುವ int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            ಯದಿ (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            ಯದಿ (hi <= lo) {
                dma_queue_ತಳ್ಳು(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                ಅನುವರ್ತ;
            }
            ಭಿನ್ನ * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            ಧ್ರುವ dma_addr_t vsrc = src_data + batch_offset +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * ಪರಿಮಾಣ(ಭಿನ್ನ), IC);
        }
    }
}

#define IM2COL_3D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
ಸ್ಥಿರ ಶೂನ್ಯ FNAME(ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,                                            \
                  ಶೂನ್ಯ * restrict dstb_raw, ಧ್ರುವ ಭಿನ್ನ * restrict srcb,                             \
                  ಭಿನ್ನ * restrict scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                              \
    ಧ್ರುವ ಸಂರಚನಾ htp_ops_context * octx = ictx->octx;                                                \
    ಧ್ರುವ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    ಧ್ರುವ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    ಧ್ರುವ int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    ಧ್ರುವ uint32_t IC = ictx->IC;                                                                    \
    ಧ್ರುವ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    ಧ್ರುವ uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    ಧ್ರುವ uint32_t patch_stride = ictx->patch_stride;                                                \
    ಧ್ರುವ uint32_t Wb = ictx->pe_wb;                                                                 \
    ಧ್ರುವ int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    ಪ್ರತಿ (uint32_t j = 0; j < nb; j++) {                                                              \
        ಧ್ರುವ uint32_t iow = c0 + j;                                                                 \
        ಧ್ರುವ int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        ಪ್ರತಿ (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            ಪ್ರತಿ (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                ಧ್ರುವ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                ಪ್ರತಿ (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    ಧ್ರುವ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * out_run = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    ಯದಿ (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(out_run, 0, KW);                                                    \
                        ಅನುವರ್ತ;                                                                    \
                    }                                                                                \
                    ಧ್ರುವ ಭಿನ್ನ * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    ಯದಿ (d0 == 1) {                                                                   \
                        ಧ್ರುವ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        ಯದಿ (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        ಯದಿ (hi <= lo) {                                                              \
                            SPLAT_FN(out_run, 0, KW);                                                \
                        } ಅನ್ಯಥಾ {                                                                     \
                            ಯದಿ (lo > 0) {                                                            \
                                SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (out_run + lo),                                      \
                                    (ಧ್ರುವ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            ಯದಿ (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        ಅನುವರ್ತ;                                                                    \
                    }                                                                                \
                    ಪ್ರತಿ (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        ಧ್ರುವ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        ಪ್ರತಿ (uint32_t c = 0; c < chunk; c++) {                                       \
                            ಧ್ರುವ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (out_run + ikw), (ಧ್ರುವ uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, ಭಿನ್ನ,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_blocked_dma_thread(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    (ಶೂನ್ಯ) nth;
    ಸಂರಚನಾ htp_im2col_context * ictx        = (ಸಂರಚನಾ htp_im2col_context *) data;
    ಸಂರಚನಾ htp_ops_context *    octx        = ictx->octx;
    ಸಂರಚನಾ htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * restrict dst  = octx->dst;
    ಧ್ರುವ uint32_t OW           = ictx->OW;
    ಧ್ರುವ uint32_t KD = ictx->KD, KH = ictx->KH;
    ಧ್ರುವ uint32_t kd_kh        = KD * KH;
    ಧ್ರುವ uint32_t patch_stride = ictx->patch_stride;
    ಧ್ರುವ uint32_t owb          = ictx->pe_owb;
    ಧ್ರುವ uint32_t nbpr         = ictx->pe_nbpr;
    ಧ್ರುವ uint32_t dst_elem     = ictx->dst_elem;
    ಧ್ರುವ dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    ಭಿನ್ನ * srcb2[2]            = { (ಭಿನ್ನ *) srcb_base, (ಭಿನ್ನ *) (srcb_base + ictx->pe_src_row_bytes) };
    ಶೂನ್ಯ * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    ಭಿನ್ನ * scratch             = (ಭಿನ್ನ *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    ಧ್ರುವ uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    ಧ್ರುವ uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ಧ್ರುವ uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    ಯದಿ (row_start >= row_end) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಧ್ರುವ uint32_t nrows_local  = row_end - row_start;
    ಧ್ರುವ uint32_t total_blocks = nrows_local * nbpr;
    ಯದಿ (total_blocks == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಸಂರಚನಾ im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ಸಂರಚನಾ im2col_cursor stage = cur;
    ಪ್ರತಿ (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_block(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    ಪ್ರತಿ (uint32_t bi = 0; bi < total_blocks; bi++) {
        ಧ್ರುವ uint32_t buf = bi & 1u;
        ಭಿನ್ನ * srcb = srcb2[buf];
        ಶೂನ್ಯ *  dstb = dstb2[buf];
        ಧ್ರುವ uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_queue_ತೆಗೆ(dma_q);
        ಪ್ರತಿ (uint32_t p = 0; p < kd_kh; p++) {
            dma_queue_ತೆಗೆ(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ಧ್ರುವ dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ಯದಿ (bi + 2 < total_blocks) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_block(&stage, ictx);
        }
        im2col_3d_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}
ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_patchembed_dma_stage(ಸಂರಚನಾ htp_im2col_context * ictx, dma_queue * dma_q,
                                           ಭಿನ್ನ * srcb, ಧ್ರುವ ಸಂರಚನಾ im2col_cursor * c) {
    ಧ್ರುವ ಸಂರಚನಾ htp_ops_context * octx = ictx->octx;
    ಧ್ರುವ uint32_t IC = ictx->IC;
    ಧ್ರುವ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ಧ್ರುವ uint32_t KD = ictx->KD, KH = ictx->KH;
    ಧ್ರುವ dma_addr_t src_data = octx->src[1]->data;
    ಧ್ರುವ uint32_t in = c->in, iod = c->iod, ioh = c->ioh;
    ಧ್ರುವ size_t src_channel_stride  = (size_t) ID * IH * IW * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ size_t vtcm_channel_stride = (size_t) KD * KH * IW * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ size_t row_size            = (size_t) KH * IW * ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಪ್ರತಿ (uint32_t ikd = 0; ikd < KD; ikd++) {
        ಧ್ರುವ uint32_t iid = iod * KD + ikd;
        ಭಿನ್ನ * vdst = srcb + (size_t) ikd * KH * IW;
        ಧ್ರುವ dma_addr_t vsrc = src_data + (size_t) (((in * IC) * ID + iid) * IH + ioh * KH) * IW * ಪರಿಮಾಣ(ಭಿನ್ನ);
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, row_size, IC);
    }
}
ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_patchembed_compute_f16(
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,
    ಶೂನ್ಯ * restrict dstb_raw, ಧ್ರುವ ಭಿನ್ನ * restrict srcb) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    ಧ್ರುವ uint32_t OW           = ictx->OW;
    ಧ್ರುವ uint32_t IC           = ictx->IC;
    ಧ್ರುವ uint32_t KD           = ictx->KD;
    ಧ್ರುವ uint32_t KH           = ictx->KH;
    ಧ್ರುವ uint32_t KW           = ictx->KW;
    ಧ್ರುವ uint32_t IW           = ictx->IW;
    ಧ್ರುವ uint32_t patch_stride = ictx->patch_stride;

    ಪ್ರತಿ (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        ಪ್ರತಿ (uint32_t iic = 0; iic < IC; iic++) {
            ಪ್ರತಿ (uint32_t ikd = 0; ikd < KD; ikd++) {
                ಪ್ರತಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ಧ್ರುವ ಭಿನ್ನ * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) out_run, (ಧ್ರುವ uint8_t *) src_run, KW);
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_patchembed_compute_f32(
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_context * ictx,
    ಶೂನ್ಯ * restrict dstb_raw, ಧ್ರುವ ಭಿನ್ನ * restrict srcb) {
    ಭಿನ್ನ * restrict dstb = (ಭಿನ್ನ *) dstb_raw;
    ಧ್ರುವ uint32_t OW           = ictx->OW;
    ಧ್ರುವ uint32_t IC           = ictx->IC;
    ಧ್ರುವ uint32_t KD           = ictx->KD;
    ಧ್ರುವ uint32_t KH           = ictx->KH;
    ಧ್ರುವ uint32_t KW           = ictx->KW;
    ಧ್ರುವ uint32_t IW           = ictx->IW;
    ಧ್ರುವ uint32_t patch_stride = ictx->patch_stride;

    ಪ್ರತಿ (uint32_t iow = 0; iow < OW; iow++) {
        ಭಿನ್ನ * dst_patch = dstb + (size_t) iow * patch_stride;
        ಪ್ರತಿ (uint32_t iic = 0; iic < IC; iic++) {
            ಪ್ರತಿ (uint32_t ikd = 0; ikd < KD; ikd++) {
                ಪ್ರತಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
                    ಭಿನ್ನ * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ಧ್ರುವ ಭಿನ್ನ * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) out_run, (ಧ್ರುವ uint8_t *) src_run, KW);
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_patchembed_dma_thread(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    (ಶೂನ್ಯ) nth;
    ಸಂರಚನಾ htp_im2col_context * ictx        = (ಸಂರಚನಾ htp_im2col_context *) data;
    ಸಂರಚನಾ htp_ops_context *    octx        = ictx->octx;
    ಸಂರಚನಾ htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * restrict dst  = octx->dst;
    ಧ್ರುವ uint32_t KD           = ictx->KD;
    ಧ್ರುವ uint32_t OW           = ictx->OW;
    ಧ್ರುವ uint32_t patch_stride = ictx->patch_stride;
    ಧ್ರುವ uint32_t dst_elem     = ictx->dst_elem;
    ಧ್ರುವ dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    ಧ್ರುವ uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    ಧ್ರುವ uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ಧ್ರುವ uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    ಯದಿ (row_start >= row_end) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ uint32_t nrows_local  = row_end - row_start;
    uint8_t * src_base          = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    ಭಿನ್ನ * srcb2[2]            = { (ಭಿನ್ನ *) src_base, (ಭಿನ್ನ *) (src_base + ictx->pe_src_row_bytes) };
    ಶೂನ್ಯ * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    ಸಂರಚನಾ im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ಸಂರಚನಾ im2col_cursor stage = cur;
    ಪ್ರತಿ (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_row(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    ಪ್ರತಿ (uint32_t idx = 0; idx < nrows_local; idx++) {
        ಧ್ರುವ uint32_t buf = idx & 1u;
        ಭಿನ್ನ * srcb = srcb2[buf];
        ಶೂನ್ಯ *  dstb = dstb2[buf];
        dma_queue_ತೆಗೆ(dma_q);
        ಪ್ರತಿ (uint32_t p = 0; p < KD; p++) {
            dma_queue_ತೆಗೆ(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ಧ್ರುವ dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_ತಳ್ಳು(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ಯದಿ (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_row(&stage, ictx);
        }
        im2col_3d_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

ಪೂರ್ಣಾಙ್ಕ op_im2col(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_kernel_params * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_im2col_kernel_params *) octx->kernel_params;
    ಯದಿ (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಯದಿ (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        ಧ್ರುವ uint32_t row_size = dst->nb[2];
        ಧ್ರುವ uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_ಸಾಲು_SIZE / hex_gcd_u32(row_size, HEX_L2_ಸಾಲು_SIZE)) : 1;
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        row_base = range.start;
        nrows    = range.ಎಣಿಕೆ;
    }

    ಯದಿ (nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಸಂರಚನಾ htp_im2col_context ictx = { 0 };
    uint8_t * ಧ್ರುವ base = octx->ctx->vtcm_base;

    ಧ್ರುವ uint32_t pth = MIN(octx->n_threads, nrows);
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

    ಪ್ರತಿ (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_in[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? ಪರಿಮಾಣ(__fp16) : ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಯದಿ (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ಶೂನ್ಯ *) im2col_patchembed_compute_f16
                                                 : (ಶೂನ್ಯ *) im2col_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_dma_thread, &ictx, pth);
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    } ಅನ್ಯಥಾ ಯದಿ (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ಶೂನ್ಯ *) im2col_blocked_compute_f16
                                                 : (ಶೂನ್ಯ *) im2col_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_blocked_dma_thread, &ictx, pth);
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
}

ಪೂರ್ಣಾಙ್ಕ op_im2col_3d(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_im2col_kernel_params * kparams =
        (ಧ್ರುವ ಸಂರಚನಾ htp_im2col_kernel_params *) octx->kernel_params;
    ಯದಿ (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಯದಿ (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        ಧ್ರುವ uint32_t row_size = dst->nb[2];
        ಧ್ರುವ uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_ಸಾಲು_SIZE / hex_gcd_u32(row_size, HEX_L2_ಸಾಲು_SIZE)) : 1;
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        row_base = range.start;
        nrows    = range.ಎಣಿಕೆ;
    }

    ಯದಿ (nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಸಂರಚನಾ htp_im2col_context ictx = { 0 };
    uint8_t * ಧ್ರುವ base = octx->ctx->vtcm_base;

    ಧ್ರುವ uint32_t pth = MIN(octx->n_threads, nrows);
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

    ಪ್ರತಿ (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_in[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? ಪರಿಮಾಣ(__fp16) : ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಯದಿ (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ಶೂನ್ಯ *) im2col_3d_patchembed_compute_f16
                                                 : (ಶೂನ್ಯ *) im2col_3d_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_patchembed_dma_thread, &ictx, pth);
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    } ಅನ್ಯಥಾ ಯದಿ (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ಶೂನ್ಯ *) im2col_3d_blocked_compute_f16
                                                 : (ಶೂನ್ಯ *) im2col_3d_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_blocked_dma_thread, &ictx, pth);
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
}
