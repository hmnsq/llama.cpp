#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <hexagon_protos.h>
#અન્તર્ભાવ <hexagon_types.h>
#અન્તર્ભાવ <string.h>

#અન્તર્ભાવ "hex-common.h"

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "hvx-utils.h"
#અન્તર્ભાવ "dma-queue.h"
#અન્તર્ભાવ "hex-profile.h"
#અન્તર્ભાવ "htp-vtcm.h"
#અન્તર્ભાવ "htp-tensor.h"
#અન્તર્ભાવ "im2col-ops.h"

સંરચના htp_im2col_context {
    સંરચના htp_ops_context * octx;

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

    શૂન્ય * compute;
    uint32_t dst_elem;
};

પ્રકારવ્યાખ્યા શૂન્ય (*im2col_blocked_compute_fn_t)(
    અચલ સંરચના htp_im2col_context * ictx,
    શૂન્ય * restrict dstb, અચલ દશાંશ * restrict srcb,
    દશાંશ * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

પ્રકારવ્યાખ્યા શૂન્ય (*im2col_patchembed_compute_fn_t)(
    અચલ સંરચના htp_im2col_context * ictx,
    શૂન્ય * restrict dstb, અચલ દશાંશ * restrict srcb, uint32_t ioh);

પ્રકારવ્યાખ્યા શૂન્ય (*im2col_3d_blocked_compute_fn_t)(
    અચલ સંરચના htp_im2col_context * ictx,
    શૂન્ય * restrict dstb, અચલ દશાંશ * restrict srcb,
    દશાંશ * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

પ્રકારવ્યાખ્યા શૂન્ય (*im2col_3d_patchembed_compute_fn_t)(
    અચલ સંરચના htp_im2col_context * ictx,
    શૂન્ય * restrict dstb, અચલ દશાંશ * restrict srcb);

સંરચના im2col_cursor {
    uint32_t r;
    uint32_t in;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

સ્થિર અંતઃરેખ શૂન્ય im2col_step_row(સંરચના im2col_cursor * c, અચલ સંરચના htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    જો (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->in++;
    }
}

સ્થિર અંતઃરેખ શૂન્ય im2col_step_block(સંરચના im2col_cursor * c, અચલ સંરચના htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    જો (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        જો (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->in++;
        }
    }
}

સ્થિર શૂન્ય im2col_blocked_dma_stage(સંરચના htp_im2col_context * ictx, dma_queue * dma_q,
                                     દશાંશ * srcb, અચલ સંરચના im2col_cursor * c) {
    અચલ uint32_t in = c->in, ioh = c->ioh, c0 = c->c0;
    અચલ સંરચના htp_ops_context * octx = ictx->octx;
    અચલ સંરચના htp_tensor * src1 = octx->src[1];
    અચલ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    અચલ int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    અચલ int32_t d1 = octx->op_params[5];
    અચલ bool is_2D = (octx->op_params[6] == 1);
    અચલ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    અચલ uint32_t KH = ictx->KH;
    અચલ uint32_t Wb = ictx->pe_wb;
    અચલ dma_addr_t src_data = src1->data;
    અચલ int32_t win0 = (int32_t) c0 * s0 - p0;
    અચલ size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    અચલ size_t batch_offset        = is_2D ? (size_t) in * src1->nb[3] : (size_t) in * src1->nb[2];

    માટે (uint32_t ikh = 0; ikh < KH; ikh++) {
        અચલ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        જો (iih < 0 || iih >= (int32_t) IH) {
            dma_queue_ધકેલો(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            ચાલુ;
        }
        અચલ int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        જો (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        જો (hi <= lo) {
            dma_queue_ધકેલો(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            ચાલુ;
        }
        અચલ uint32_t cpw = (uint32_t) (hi - lo);
        દશાંશ * vdst = srcb + (size_t) ikh * Wb + lo;
        અચલ dma_addr_t vsrc = src_data + batch_offset + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_queue_ધકેલો(dma_q, dma_make_data(vdst, vsrc),
                       (size_t) KH * Wb * કદ(દશાંશ), src_channel_stride,
                       cpw * કદ(દશાંશ), IC);
    }
}

#define IM2COL_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
સ્થિર શૂન્ય FNAME(                                                                               \
    અચલ સંરચના htp_im2col_context * ictx,                                                      \
    શૂન્ય * restrict dstb_raw, અચલ દશાંશ * restrict srcb,                                       \
    દશાંશ * restrict scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                          \
    અચલ સંરચના htp_ops_context * octx = ictx->octx;                                            \
    અચલ int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    અચલ int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    અચલ int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    અચલ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    અચલ uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    અચલ uint32_t Wb = ictx->pe_wb;                                                             \
    અચલ uint32_t patch_stride = ictx->patch_stride;                                            \
    અચલ int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    માટે (uint32_t j = 0; j < nb; j++) {                                                          \
        અચલ uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        અચલ int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        માટે (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            અચલ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            અચલ પૂર્ણાંક     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            માટે (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                જો (!okh) {                                                                      \
                    SPLAT_FN(out_run, 0, KW);                                                    \
                    ચાલુ;                                                                    \
                }                                                                                \
                અચલ દશાંશ * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                જો (d0 == 1) {                                                                   \
                    અચલ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    જો (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    જો (hi <= lo) {                                                              \
                        SPLAT_FN(out_run, 0, KW);                                                \
                    } નહીંતર {                                                                     \
                        જો (lo > 0) {                                                            \
                            SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (out_run + lo),                                      \
                                (અચલ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        જો (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    ચાલુ;                                                                    \
                }                                                                                \
                માટે (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    અચલ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    માટે (uint32_t c = 0; c < chunk; c++) {                                       \
                        અચલ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (out_run + ikw), (અચલ uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, દશાંશ,    hvx_copy_f32_uu,     hvx_splat_f32_u)

સ્થિર શૂન્ય im2col_blocked_dma_thread(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    (શૂન્ય) nth;
    સંરચના htp_im2col_context * ictx      = (સંરચના htp_im2col_context *) data;
    સંરચના htp_ops_context *    octx      = ictx->octx;
    સંરચના htp_thread_trace * restrict tr = &octx->ctx->trace[ith];
    અચલ uint32_t OW                     = ictx->OW;
    અચલ uint32_t KH                     = ictx->KH;
    અચલ uint32_t owb                    = ictx->pe_owb;
    અચલ uint32_t nbpr                   = ictx->pe_nbpr;
    અચલ uint32_t patch_stride           = ictx->patch_stride;
    અચલ uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    અચલ dma_addr_t dst_data             = octx->dst->data;
    dma_queue *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_base              = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    દશાંશ *        srcb2[2]               = { (દશાંશ *) srcb_base, (દશાંશ *) (srcb_base + ictx->pe_src_row_bytes) };
    શૂન્ય *         dstb2[2]               = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    દશાંશ *        scratch                = (દશાંશ *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    અચલ uint32_t row_end_max            = ictx->pe_row_base + ictx->pe_nrows;
    અચલ uint32_t row_start              = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    અચલ uint32_t row_end                = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    જો (row_start >= row_end) {
        પાછા;
    }
    અચલ uint32_t nrows_local  = row_end - row_start;
    અચલ uint32_t total_blocks = nrows_local * nbpr;
    જો (total_blocks == 0) {
        પાછા;
    }
    સંરચના im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    સંરચના im2col_cursor stage = cur;
    માટે (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_ધકેલો(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_block(&stage, ictx);
    }
    માટે (uint32_t bi = 0; bi < total_blocks; bi++) {
        અચલ uint32_t buf = bi & 1u;
        દશાંશ * srcb = srcb2[buf];
        શૂન્ય *  dstb = dstb2[buf];
        અચલ uint32_t nb = MIN(owb, OW - cur.c0);
        dma_queue_કાઢો(dma_q);
        માટે (uint32_t k = 0; k < KH; k++) {
            dma_queue_કાઢો(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        અચલ dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_ધકેલો(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        જો (bi + 2 < total_blocks) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_block(&stage, ictx);
        }
        im2col_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

સ્થિર શૂન્ય im2col_patchembed_dma_stage(સંરચના htp_im2col_context * ictx, dma_queue * dma_q,
                                        દશાંશ * srcb, અચલ સંરચના im2col_cursor * c) {
    અચલ સંરચના htp_ops_context * octx = ictx->octx;
    અચલ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    અચલ uint32_t KH = ictx->KH;
    અચલ uint32_t in = c->in, ioh = c->ioh;
    અચલ size_t src_channel_stride  = (size_t) IH * IW * કદ(દશાંશ);
    અચલ size_t vtcm_channel_stride = (size_t) KH * IW * કદ(દશાંશ);
    અચલ dma_addr_t src_data        = octx->src[1]->data;
    અચલ dma_addr_t vsrc            = src_data + (size_t) (((in * IC) * IH + ioh * KH) * IW) * કદ(દશાંશ);

    dma_queue_ધકેલો(dma_q, dma_make_data(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

સ્થિર શૂન્ય im2col_patchembed_compute_f16(
    અચલ સંરચના htp_im2col_context * ictx,
    શૂન્ય * restrict dstb_raw, અચલ દશાંશ * restrict srcb, uint32_t ioh) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    અચલ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    અચલ uint32_t KH = ictx->KH, KW = ictx->KW;
    અચલ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    માટે (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        માટે (uint32_t ikh = 0; ikh < KH; ikh++) {
            અચલ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            માટે (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                જો (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(out_run, 0, KW);
                    ચાલુ;
                }
                અચલ દશાંશ * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) out_run, (અચલ uint8_t *) src_run, KW);
            }
        }
    }
}

સ્થિર શૂન્ય im2col_patchembed_compute_f32(
    અચલ સંરચના htp_im2col_context * ictx,
    શૂન્ય * restrict dstb_raw, અચલ દશાંશ * restrict srcb, uint32_t ioh) {
    દશાંશ * restrict dstb = (દશાંશ *) dstb_raw;
    અચલ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    અચલ uint32_t KH = ictx->KH, KW = ictx->KW;
    અચલ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    માટે (uint32_t iow = 0; iow < OW; iow++) {
        દશાંશ * dst_patch = dstb + (size_t) iow * patch_stride;
        માટે (uint32_t ikh = 0; ikh < KH; ikh++) {
            અચલ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            માટે (uint32_t iic = 0; iic < IC; iic++) {
                દશાંશ * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                જો (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(out_run, 0, KW);
                    ચાલુ;
                }
                અચલ દશાંશ * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) out_run, (અચલ uint8_t *) src_run, KW);
            }
        }
    }
}

સ્થિર શૂન્ય im2col_patchembed_dma_thread(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    (શૂન્ય) nth;
    સંરચના htp_im2col_context * ictx        = (સંરચના htp_im2col_context *) data;
    સંરચના htp_ops_context *    octx        = ictx->octx;
    સંરચના htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    અચલ uint32_t OW                       = ictx->OW;
    અચલ uint32_t patch_stride             = ictx->patch_stride;
    અચલ uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    અચલ dma_addr_t dst_data               = octx->dst->data;
    dma_queue *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_base                = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    દશાંશ *        srcb2[2]                 = { (દશાંશ *) srcb_base, (દશાંશ *) (srcb_base + ictx->pe_src_row_bytes) };
    શૂન્ય *         dstb2[2]                 = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    અચલ uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;
    અચલ uint32_t row_start                = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    અચલ uint32_t row_end                  = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    જો (row_start >= row_end) {
        પાછા;
    }
    અચલ uint32_t total_rows = row_end - row_start;
    જો (total_rows == 0) {
        પાછા;
    }
    સંરચના im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    સંરચના im2col_cursor stage = cur;
    માટે (uint32_t i = 0; i < total_rows && i < 2; i++) {
        dma_queue_ધકેલો(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_row(&stage, ictx);
    }
    માટે (uint32_t ri = 0; ri < total_rows; ri++) {
        અચલ uint32_t buf = ri & 1u;
        દશાંશ * srcb = srcb2[buf];
        શૂન્ય *  dstb = dstb2[buf];
        dma_queue_કાઢો(dma_q);
        dma_queue_કાઢો(dma_q);
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        અચલ dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_ધકેલો(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        જો (ri + 2 < total_rows) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_row(&stage, ictx);
        }
        im2col_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

સ્થિર અંતઃરેખ શૂન્ય im2col_3d_step_row(સંરચના im2col_cursor * c, અચલ સંરચના htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    જો (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        જો (c->iod >= ictx->OD) {
            c->iod = 0;
            c->in++;
        }
    }
}

સ્થિર અંતઃરેખ શૂન્ય im2col_3d_step_block(સંરચના im2col_cursor * c, અચલ સંરચના htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    જો (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        જો (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            જો (c->iod >= ictx->OD) {
                c->iod = 0;
                c->in++;
            }
        }
    }
}

સ્થિર શૂન્ય im2col_3d_blocked_dma_stage(સંરચના htp_im2col_context * ictx, dma_queue * dma_q,
                                        દશાંશ * srcb, અચલ સંરચના im2col_cursor * c) {
    અચલ uint32_t in = c->in, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    અચલ સંરચના htp_ops_context * octx = ictx->octx;
    અચલ સંરચના htp_tensor * src1 = octx->src[1];
    અચલ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    અચલ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    અચલ int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    અચલ uint32_t IC = ictx->IC;
    અચલ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    અચલ uint32_t KD = ictx->KD, KH = ictx->KH;
    અચલ uint32_t Wb = ictx->pe_wb;
    અચલ int32_t win0 = (int32_t) c0 * s0 - p0;
    અચલ dma_addr_t src_data = src1->data;
    અચલ size_t src_channel_stride = src1->nb[3];
    અચલ size_t vtcm_channel_stride = (size_t) KD * KH * Wb * કદ(દશાંશ);
    અચલ size_t batch_offset        = (size_t) (in * IC) * src1->nb[3];

    માટે (uint32_t ikd = 0; ikd < KD; ikd++) {
        અચલ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        અચલ bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        માટે (uint32_t ikh = 0; ikh < KH; ikh++) {
            અચલ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            જો (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_queue_ધકેલો(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                ચાલુ;
            }
            અચલ int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            જો (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            જો (hi <= lo) {
                dma_queue_ધકેલો(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                ચાલુ;
            }
            દશાંશ * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            અચલ dma_addr_t vsrc = src_data + batch_offset +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_queue_ધકેલો(dma_q, dma_make_data(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * કદ(દશાંશ), IC);
        }
    }
}

#define IM2COL_3D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
સ્થિર શૂન્ય FNAME(અચલ સંરચના htp_im2col_context * ictx,                                            \
                  શૂન્ય * restrict dstb_raw, અચલ દશાંશ * restrict srcb,                             \
                  દશાંશ * restrict scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                              \
    અચલ સંરચના htp_ops_context * octx = ictx->octx;                                                \
    અચલ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    અચલ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    અચલ int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    અચલ uint32_t IC = ictx->IC;                                                                    \
    અચલ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    અચલ uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    અચલ uint32_t patch_stride = ictx->patch_stride;                                                \
    અચલ uint32_t Wb = ictx->pe_wb;                                                                 \
    અચલ int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    માટે (uint32_t j = 0; j < nb; j++) {                                                              \
        અચલ uint32_t iow = c0 + j;                                                                 \
        અચલ int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        માટે (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            માટે (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                અચલ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                માટે (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    અચલ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * out_run = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    જો (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(out_run, 0, KW);                                                    \
                        ચાલુ;                                                                    \
                    }                                                                                \
                    અચલ દશાંશ * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    જો (d0 == 1) {                                                                   \
                        અચલ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        જો (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        જો (hi <= lo) {                                                              \
                            SPLAT_FN(out_run, 0, KW);                                                \
                        } નહીંતર {                                                                     \
                            જો (lo > 0) {                                                            \
                                SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (out_run + lo),                                      \
                                    (અચલ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            જો (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        ચાલુ;                                                                    \
                    }                                                                                \
                    માટે (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        અચલ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        માટે (uint32_t c = 0; c < chunk; c++) {                                       \
                            અચલ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (out_run + ikw), (અચલ uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, દશાંશ,    hvx_copy_f32_uu,     hvx_splat_f32_u)

સ્થિર શૂન્ય im2col_3d_blocked_dma_thread(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    (શૂન્ય) nth;
    સંરચના htp_im2col_context * ictx        = (સંરચના htp_im2col_context *) data;
    સંરચના htp_ops_context *    octx        = ictx->octx;
    સંરચના htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    અચલ સંરચના htp_tensor * restrict dst  = octx->dst;
    અચલ uint32_t OW           = ictx->OW;
    અચલ uint32_t KD = ictx->KD, KH = ictx->KH;
    અચલ uint32_t kd_kh        = KD * KH;
    અચલ uint32_t patch_stride = ictx->patch_stride;
    અચલ uint32_t owb          = ictx->pe_owb;
    અચલ uint32_t nbpr         = ictx->pe_nbpr;
    અચલ uint32_t dst_elem     = ictx->dst_elem;
    અચલ dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    દશાંશ * srcb2[2]            = { (દશાંશ *) srcb_base, (દશાંશ *) (srcb_base + ictx->pe_src_row_bytes) };
    શૂન્ય * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    દશાંશ * scratch             = (દશાંશ *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    અચલ uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    અચલ uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    અચલ uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    જો (row_start >= row_end) {
        પાછા;
    }
    અચલ uint32_t nrows_local  = row_end - row_start;
    અચલ uint32_t total_blocks = nrows_local * nbpr;
    જો (total_blocks == 0) {
        પાછા;
    }
    સંરચના im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    સંરચના im2col_cursor stage = cur;
    માટે (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_ધકેલો(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_block(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    માટે (uint32_t bi = 0; bi < total_blocks; bi++) {
        અચલ uint32_t buf = bi & 1u;
        દશાંશ * srcb = srcb2[buf];
        શૂન્ય *  dstb = dstb2[buf];
        અચલ uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_queue_કાઢો(dma_q);
        માટે (uint32_t p = 0; p < kd_kh; p++) {
            dma_queue_કાઢો(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        અચલ dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_ધકેલો(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        જો (bi + 2 < total_blocks) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_block(&stage, ictx);
        }
        im2col_3d_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}
સ્થિર શૂન્ય im2col_3d_patchembed_dma_stage(સંરચના htp_im2col_context * ictx, dma_queue * dma_q,
                                           દશાંશ * srcb, અચલ સંરચના im2col_cursor * c) {
    અચલ સંરચના htp_ops_context * octx = ictx->octx;
    અચલ uint32_t IC = ictx->IC;
    અચલ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    અચલ uint32_t KD = ictx->KD, KH = ictx->KH;
    અચલ dma_addr_t src_data = octx->src[1]->data;
    અચલ uint32_t in = c->in, iod = c->iod, ioh = c->ioh;
    અચલ size_t src_channel_stride  = (size_t) ID * IH * IW * કદ(દશાંશ);
    અચલ size_t vtcm_channel_stride = (size_t) KD * KH * IW * કદ(દશાંશ);
    અચલ size_t row_size            = (size_t) KH * IW * કદ(દશાંશ);

    માટે (uint32_t ikd = 0; ikd < KD; ikd++) {
        અચલ uint32_t iid = iod * KD + ikd;
        દશાંશ * vdst = srcb + (size_t) ikd * KH * IW;
        અચલ dma_addr_t vsrc = src_data + (size_t) (((in * IC) * ID + iid) * IH + ioh * KH) * IW * કદ(દશાંશ);
        dma_queue_ધકેલો(dma_q, dma_make_data(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, row_size, IC);
    }
}
સ્થિર શૂન્ય im2col_3d_patchembed_compute_f16(
    અચલ સંરચના htp_im2col_context * ictx,
    શૂન્ય * restrict dstb_raw, અચલ દશાંશ * restrict srcb) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    અચલ uint32_t OW           = ictx->OW;
    અચલ uint32_t IC           = ictx->IC;
    અચલ uint32_t KD           = ictx->KD;
    અચલ uint32_t KH           = ictx->KH;
    અચલ uint32_t KW           = ictx->KW;
    અચલ uint32_t IW           = ictx->IW;
    અચલ uint32_t patch_stride = ictx->patch_stride;

    માટે (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        માટે (uint32_t iic = 0; iic < IC; iic++) {
            માટે (uint32_t ikd = 0; ikd < KD; ikd++) {
                માટે (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    અચલ દશાંશ * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) out_run, (અચલ uint8_t *) src_run, KW);
                }
            }
        }
    }
}

સ્થિર શૂન્ય im2col_3d_patchembed_compute_f32(
    અચલ સંરચના htp_im2col_context * ictx,
    શૂન્ય * restrict dstb_raw, અચલ દશાંશ * restrict srcb) {
    દશાંશ * restrict dstb = (દશાંશ *) dstb_raw;
    અચલ uint32_t OW           = ictx->OW;
    અચલ uint32_t IC           = ictx->IC;
    અચલ uint32_t KD           = ictx->KD;
    અચલ uint32_t KH           = ictx->KH;
    અચલ uint32_t KW           = ictx->KW;
    અચલ uint32_t IW           = ictx->IW;
    અચલ uint32_t patch_stride = ictx->patch_stride;

    માટે (uint32_t iow = 0; iow < OW; iow++) {
        દશાંશ * dst_patch = dstb + (size_t) iow * patch_stride;
        માટે (uint32_t iic = 0; iic < IC; iic++) {
            માટે (uint32_t ikd = 0; ikd < KD; ikd++) {
                માટે (uint32_t ikh = 0; ikh < KH; ikh++) {
                    દશાંશ * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    અચલ દશાંશ * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) out_run, (અચલ uint8_t *) src_run, KW);
                }
            }
        }
    }
}

સ્થિર શૂન્ય im2col_3d_patchembed_dma_thread(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    (શૂન્ય) nth;
    સંરચના htp_im2col_context * ictx        = (સંરચના htp_im2col_context *) data;
    સંરચના htp_ops_context *    octx        = ictx->octx;
    સંરચના htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    અચલ સંરચના htp_tensor * restrict dst  = octx->dst;
    અચલ uint32_t KD           = ictx->KD;
    અચલ uint32_t OW           = ictx->OW;
    અચલ uint32_t patch_stride = ictx->patch_stride;
    અચલ uint32_t dst_elem     = ictx->dst_elem;
    અચલ dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    અચલ uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    અચલ uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    અચલ uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    જો (row_start >= row_end) {
        પાછા;
    }

    અચલ uint32_t nrows_local  = row_end - row_start;
    uint8_t * src_base          = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    દશાંશ * srcb2[2]            = { (દશાંશ *) src_base, (દશાંશ *) (src_base + ictx->pe_src_row_bytes) };
    શૂન્ય * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    સંરચના im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    સંરચના im2col_cursor stage = cur;
    માટે (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_queue_ધકેલો(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_row(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    માટે (uint32_t idx = 0; idx < nrows_local; idx++) {
        અચલ uint32_t buf = idx & 1u;
        દશાંશ * srcb = srcb2[buf];
        શૂન્ય *  dstb = dstb2[buf];
        dma_queue_કાઢો(dma_q);
        માટે (uint32_t p = 0; p < KD; p++) {
            dma_queue_કાઢો(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        અચલ dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_ધકેલો(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        જો (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_row(&stage, ictx);
        }
        im2col_3d_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

પૂર્ણાંક op_im2col(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_im2col_kernel_params * kparams = (અચલ સંરચના htp_im2col_kernel_params *) octx->kernel_params;
    જો (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    જો (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }

    અચલ સંરચના htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    જો (octx->ctx->mdev.ગણતરી > 1) {
        અચલ uint32_t row_size = dst->nb[2];
        અચલ uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_લીટી_SIZE / hex_gcd_u32(row_size, HEX_L2_લીટી_SIZE)) : 1;
        અચલ સંરચના htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
        row_base = range.start;
        nrows    = range.ગણતરી;
    }

    જો (nrows == 0) {
        પાછા HTP_STATUS_OK;
    }

    સંરચના htp_im2col_context ictx = { 0 };
    uint8_t * અચલ base = octx->ctx->vtcm_base;

    અચલ uint32_t pth = MIN(octx->n_threads, nrows);
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

    માટે (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_in[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? કદ(__fp16) : કદ(દશાંશ);

    જો (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (શૂન્ય *) im2col_patchembed_compute_f16
                                                 : (શૂન્ય *) im2col_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_dma_thread, &ictx, pth);
        પાછા HTP_STATUS_OK;
    } નહીંતર જો (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (શૂન્ય *) im2col_blocked_compute_f16
                                                 : (શૂન્ય *) im2col_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_blocked_dma_thread, &ictx, pth);
        પાછા HTP_STATUS_OK;
    }

    પાછા HTP_STATUS_NO_SUPPORT;
}

પૂર્ણાંક op_im2col_3d(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_im2col_kernel_params * kparams =
        (અચલ સંરચના htp_im2col_kernel_params *) octx->kernel_params;
    જો (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    જો (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }

    અચલ સંરચના htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    જો (octx->ctx->mdev.ગણતરી > 1) {
        અચલ uint32_t row_size = dst->nb[2];
        અચલ uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_લીટી_SIZE / hex_gcd_u32(row_size, HEX_L2_લીટી_SIZE)) : 1;
        અચલ સંરચના htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
        row_base = range.start;
        nrows    = range.ગણતરી;
    }

    જો (nrows == 0) {
        પાછા HTP_STATUS_OK;
    }

    સંરચના htp_im2col_context ictx = { 0 };
    uint8_t * અચલ base = octx->ctx->vtcm_base;

    અચલ uint32_t pth = MIN(octx->n_threads, nrows);
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

    માટે (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_in[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? કદ(__fp16) : કદ(દશાંશ);

    જો (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (શૂન્ય *) im2col_3d_patchembed_compute_f16
                                                 : (શૂન્ય *) im2col_3d_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_patchembed_dma_thread, &ictx, pth);
        પાછા HTP_STATUS_OK;
    } નહીંતર જો (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (શૂન્ય *) im2col_3d_blocked_compute_f16
                                                 : (શૂન્ય *) im2col_3d_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_blocked_dma_thread, &ictx, pth);
        પાછા HTP_STATUS_OK;
    }

    પાછા HTP_STATUS_NO_SUPPORT;
}
