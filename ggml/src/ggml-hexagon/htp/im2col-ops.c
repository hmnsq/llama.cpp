#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <hexagon_protos.h>
#شامل <hexagon_types.h>
#شامل <string.h>

#شامل "hex-common.h"

#define GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "hvx-utils.h"
#شامل "dma-queue.h"
#شامل "hex-profile.h"
#شامل "htp-vtcm.h"
#شامل "htp-tensor.h"
#شامل "im2col-ops.h"

ساختار htp_im2col_context {
    ساختار htp_ops_context * octx;

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

    تهی * compute;
    uint32_t dst_elem;
};

تعریفنوع تهی (*im2col_blocked_compute_fn_t)(
    ثابت ساختار htp_im2col_context * ictx,
    تهی * restrict dstb, ثابت اعشاری * restrict srcb,
    اعشاری * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

تعریفنوع تهی (*im2col_patchembed_compute_fn_t)(
    ثابت ساختار htp_im2col_context * ictx,
    تهی * restrict dstb, ثابت اعشاری * restrict srcb, uint32_t ioh);

تعریفنوع تهی (*im2col_3d_blocked_compute_fn_t)(
    ثابت ساختار htp_im2col_context * ictx,
    تهی * restrict dstb, ثابت اعشاری * restrict srcb,
    اعشاری * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

تعریفنوع تهی (*im2col_3d_patchembed_compute_fn_t)(
    ثابت ساختار htp_im2col_context * ictx,
    تهی * restrict dstb, ثابت اعشاری * restrict srcb);

ساختار im2col_cursor {
    uint32_t r;
    uint32_t in;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

ایستا درونخطی تهی im2col_step_row(ساختار im2col_cursor * c, ثابت ساختار htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    اگر (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->in++;
    }
}

ایستا درونخطی تهی im2col_step_block(ساختار im2col_cursor * c, ثابت ساختار htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    اگر (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        اگر (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->in++;
        }
    }
}

ایستا تهی im2col_blocked_dma_stage(ساختار htp_im2col_context * ictx, dma_queue * dma_q,
                                     اعشاری * srcb, ثابت ساختار im2col_cursor * c) {
    ثابت uint32_t in = c->in, ioh = c->ioh, c0 = c->c0;
    ثابت ساختار htp_ops_context * octx = ictx->octx;
    ثابت ساختار htp_tensor * src1 = octx->src[1];
    ثابت int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    ثابت int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    ثابت int32_t d1 = octx->op_params[5];
    ثابت bool is_2D = (octx->op_params[6] == 1);
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH;
    ثابت uint32_t Wb = ictx->pe_wb;
    ثابت dma_addr_t src_data = src1->data;
    ثابت int32_t win0 = (int32_t) c0 * s0 - p0;
    ثابت size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    ثابت size_t batch_offset        = is_2D ? (size_t) in * src1->nb[3] : (size_t) in * src1->nb[2];

    برای (uint32_t ikh = 0; ikh < KH; ikh++) {
        ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        اگر (iih < 0 || iih >= (int32_t) IH) {
            dma_queue_افزودن(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            ادامه;
        }
        ثابت int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        اگر (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        اگر (hi <= lo) {
            dma_queue_افزودن(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            ادامه;
        }
        ثابت uint32_t cpw = (uint32_t) (hi - lo);
        اعشاری * vdst = srcb + (size_t) ikh * Wb + lo;
        ثابت dma_addr_t vsrc = src_data + batch_offset + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_queue_افزودن(dma_q, dma_make_data(vdst, vsrc),
                       (size_t) KH * Wb * اندازه(اعشاری), src_channel_stride,
                       cpw * اندازه(اعشاری), IC);
    }
}

#define IM2COL_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
ایستا تهی FNAME(                                                                               \
    ثابت ساختار htp_im2col_context * ictx,                                                      \
    تهی * restrict dstb_raw, ثابت اعشاری * restrict srcb,                                       \
    اعشاری * restrict scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                          \
    ثابت ساختار htp_ops_context * octx = ictx->octx;                                            \
    ثابت int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    ثابت int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    ثابت int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    ثابت uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    ثابت uint32_t Wb = ictx->pe_wb;                                                             \
    ثابت uint32_t patch_stride = ictx->patch_stride;                                            \
    ثابت int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    برای (uint32_t j = 0; j < nb; j++) {                                                          \
        ثابت uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        ثابت int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        برای (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            ثابت صحیح     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            برای (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                اگر (!okh) {                                                                      \
                    SPLAT_FN(out_run, 0, KW);                                                    \
                    ادامه;                                                                    \
                }                                                                                \
                ثابت اعشاری * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                اگر (d0 == 1) {                                                                   \
                    ثابت int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    اگر (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    اگر (hi <= lo) {                                                              \
                        SPLAT_FN(out_run, 0, KW);                                                \
                    } وگرنه {                                                                     \
                        اگر (lo > 0) {                                                            \
                            SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (out_run + lo),                                      \
                                (ثابت uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        اگر (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    ادامه;                                                                    \
                }                                                                                \
                برای (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    ثابت uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    برای (uint32_t c = 0; c < chunk; c++) {                                       \
                        ثابت int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (out_run + ikw), (ثابت uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, اعشاری,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ایستا تهی im2col_blocked_dma_thread(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    (تهی) nth;
    ساختار htp_im2col_context * ictx      = (ساختار htp_im2col_context *) data;
    ساختار htp_ops_context *    octx      = ictx->octx;
    ساختار htp_thread_trace * restrict tr = &octx->ctx->trace[ith];
    ثابت uint32_t OW                     = ictx->OW;
    ثابت uint32_t KH                     = ictx->KH;
    ثابت uint32_t owb                    = ictx->pe_owb;
    ثابت uint32_t nbpr                   = ictx->pe_nbpr;
    ثابت uint32_t patch_stride           = ictx->patch_stride;
    ثابت uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    ثابت dma_addr_t dst_data             = octx->dst->data;
    dma_queue *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_base              = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    اعشاری *        srcb2[2]               = { (اعشاری *) srcb_base, (اعشاری *) (srcb_base + ictx->pe_src_row_bytes) };
    تهی *         dstb2[2]               = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    اعشاری *        scratch                = (اعشاری *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    ثابت uint32_t row_end_max            = ictx->pe_row_base + ictx->pe_nrows;
    ثابت uint32_t row_start              = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ثابت uint32_t row_end                = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    اگر (row_start >= row_end) {
        بازگشت;
    }
    ثابت uint32_t nrows_local  = row_end - row_start;
    ثابت uint32_t total_blocks = nrows_local * nbpr;
    اگر (total_blocks == 0) {
        بازگشت;
    }
    ساختار im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ساختار im2col_cursor stage = cur;
    برای (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_افزودن(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_block(&stage, ictx);
    }
    برای (uint32_t bi = 0; bi < total_blocks; bi++) {
        ثابت uint32_t buf = bi & 1u;
        اعشاری * srcb = srcb2[buf];
        تهی *  dstb = dstb2[buf];
        ثابت uint32_t nb = MIN(owb, OW - cur.c0);
        dma_queue_برداشت(dma_q);
        برای (uint32_t k = 0; k < KH; k++) {
            dma_queue_برداشت(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_افزودن(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        اگر (bi + 2 < total_blocks) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_block(&stage, ictx);
        }
        im2col_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

ایستا تهی im2col_patchembed_dma_stage(ساختار htp_im2col_context * ictx, dma_queue * dma_q,
                                        اعشاری * srcb, ثابت ساختار im2col_cursor * c) {
    ثابت ساختار htp_ops_context * octx = ictx->octx;
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH;
    ثابت uint32_t in = c->in, ioh = c->ioh;
    ثابت size_t src_channel_stride  = (size_t) IH * IW * اندازه(اعشاری);
    ثابت size_t vtcm_channel_stride = (size_t) KH * IW * اندازه(اعشاری);
    ثابت dma_addr_t src_data        = octx->src[1]->data;
    ثابت dma_addr_t vsrc            = src_data + (size_t) (((in * IC) * IH + ioh * KH) * IW) * اندازه(اعشاری);

    dma_queue_افزودن(dma_q, dma_make_data(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

ایستا تهی im2col_patchembed_compute_f16(
    ثابت ساختار htp_im2col_context * ictx,
    تهی * restrict dstb_raw, ثابت اعشاری * restrict srcb, uint32_t ioh) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH, KW = ictx->KW;
    ثابت uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    برای (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        برای (uint32_t ikh = 0; ikh < KH; ikh++) {
            ثابت int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            برای (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                اگر (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(out_run, 0, KW);
                    ادامه;
                }
                ثابت اعشاری * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) out_run, (ثابت uint8_t *) src_run, KW);
            }
        }
    }
}

ایستا تهی im2col_patchembed_compute_f32(
    ثابت ساختار htp_im2col_context * ictx,
    تهی * restrict dstb_raw, ثابت اعشاری * restrict srcb, uint32_t ioh) {
    اعشاری * restrict dstb = (اعشاری *) dstb_raw;
    ثابت uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KH = ictx->KH, KW = ictx->KW;
    ثابت uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    برای (uint32_t iow = 0; iow < OW; iow++) {
        اعشاری * dst_patch = dstb + (size_t) iow * patch_stride;
        برای (uint32_t ikh = 0; ikh < KH; ikh++) {
            ثابت int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            برای (uint32_t iic = 0; iic < IC; iic++) {
                اعشاری * out_run = dst_patch + (size_t) (iic * KH + ikh) * KW;
                اگر (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(out_run, 0, KW);
                    ادامه;
                }
                ثابت اعشاری * src_run = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) out_run, (ثابت uint8_t *) src_run, KW);
            }
        }
    }
}

ایستا تهی im2col_patchembed_dma_thread(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    (تهی) nth;
    ساختار htp_im2col_context * ictx        = (ساختار htp_im2col_context *) data;
    ساختار htp_ops_context *    octx        = ictx->octx;
    ساختار htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    ثابت uint32_t OW                       = ictx->OW;
    ثابت uint32_t patch_stride             = ictx->patch_stride;
    ثابت uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    ثابت dma_addr_t dst_data               = octx->dst->data;
    dma_queue *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_base                = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    اعشاری *        srcb2[2]                 = { (اعشاری *) srcb_base, (اعشاری *) (srcb_base + ictx->pe_src_row_bytes) };
    تهی *         dstb2[2]                 = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    ثابت uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;
    ثابت uint32_t row_start                = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ثابت uint32_t row_end                  = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    اگر (row_start >= row_end) {
        بازگشت;
    }
    ثابت uint32_t total_rows = row_end - row_start;
    اگر (total_rows == 0) {
        بازگشت;
    }
    ساختار im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ساختار im2col_cursor stage = cur;
    برای (uint32_t i = 0; i < total_rows && i < 2; i++) {
        dma_queue_افزودن(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_row(&stage, ictx);
    }
    برای (uint32_t ri = 0; ri < total_rows; ri++) {
        ثابت uint32_t buf = ri & 1u;
        اعشاری * srcb = srcb2[buf];
        تهی *  dstb = dstb2[buf];
        dma_queue_برداشت(dma_q);
        dma_queue_برداشت(dma_q);
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_افزودن(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        اگر (ri + 2 < total_rows) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_row(&stage, ictx);
        }
        im2col_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

ایستا درونخطی تهی im2col_3d_step_row(ساختار im2col_cursor * c, ثابت ساختار htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    اگر (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        اگر (c->iod >= ictx->OD) {
            c->iod = 0;
            c->in++;
        }
    }
}

ایستا درونخطی تهی im2col_3d_step_block(ساختار im2col_cursor * c, ثابت ساختار htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    اگر (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        اگر (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            اگر (c->iod >= ictx->OD) {
                c->iod = 0;
                c->in++;
            }
        }
    }
}

ایستا تهی im2col_3d_blocked_dma_stage(ساختار htp_im2col_context * ictx, dma_queue * dma_q,
                                        اعشاری * srcb, ثابت ساختار im2col_cursor * c) {
    ثابت uint32_t in = c->in, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    ثابت ساختار htp_ops_context * octx = ictx->octx;
    ثابت ساختار htp_tensor * src1 = octx->src[1];
    ثابت int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    ثابت int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    ثابت int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    ثابت uint32_t IC = ictx->IC;
    ثابت uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KD = ictx->KD, KH = ictx->KH;
    ثابت uint32_t Wb = ictx->pe_wb;
    ثابت int32_t win0 = (int32_t) c0 * s0 - p0;
    ثابت dma_addr_t src_data = src1->data;
    ثابت size_t src_channel_stride = src1->nb[3];
    ثابت size_t vtcm_channel_stride = (size_t) KD * KH * Wb * اندازه(اعشاری);
    ثابت size_t batch_offset        = (size_t) (in * IC) * src1->nb[3];

    برای (uint32_t ikd = 0; ikd < KD; ikd++) {
        ثابت int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        ثابت bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        برای (uint32_t ikh = 0; ikh < KH; ikh++) {
            ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            اگر (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_queue_افزودن(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                ادامه;
            }
            ثابت int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            اگر (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            اگر (hi <= lo) {
                dma_queue_افزودن(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                ادامه;
            }
            اعشاری * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            ثابت dma_addr_t vsrc = src_data + batch_offset +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_queue_افزودن(dma_q, dma_make_data(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * اندازه(اعشاری), IC);
        }
    }
}

#define IM2COL_3D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
ایستا تهی FNAME(ثابت ساختار htp_im2col_context * ictx,                                            \
                  تهی * restrict dstb_raw, ثابت اعشاری * restrict srcb,                             \
                  اعشاری * restrict scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                              \
    ثابت ساختار htp_ops_context * octx = ictx->octx;                                                \
    ثابت int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    ثابت int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    ثابت int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    ثابت uint32_t IC = ictx->IC;                                                                    \
    ثابت uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    ثابت uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    ثابت uint32_t patch_stride = ictx->patch_stride;                                                \
    ثابت uint32_t Wb = ictx->pe_wb;                                                                 \
    ثابت int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    برای (uint32_t j = 0; j < nb; j++) {                                                              \
        ثابت uint32_t iow = c0 + j;                                                                 \
        ثابت int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        برای (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            برای (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                ثابت int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                برای (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    ثابت int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * out_run = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    اگر (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(out_run, 0, KW);                                                    \
                        ادامه;                                                                    \
                    }                                                                                \
                    ثابت اعشاری * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    اگر (d0 == 1) {                                                                   \
                        ثابت int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        اگر (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        اگر (hi <= lo) {                                                              \
                            SPLAT_FN(out_run, 0, KW);                                                \
                        } وگرنه {                                                                     \
                            اگر (lo > 0) {                                                            \
                                SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (out_run + lo),                                      \
                                    (ثابت uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            اگر (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        ادامه;                                                                    \
                    }                                                                                \
                    برای (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        ثابت uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        برای (uint32_t c = 0; c < chunk; c++) {                                       \
                            ثابت int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (out_run + ikw), (ثابت uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, اعشاری,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ایستا تهی im2col_3d_blocked_dma_thread(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    (تهی) nth;
    ساختار htp_im2col_context * ictx        = (ساختار htp_im2col_context *) data;
    ساختار htp_ops_context *    octx        = ictx->octx;
    ساختار htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    ثابت ساختار htp_tensor * restrict dst  = octx->dst;
    ثابت uint32_t OW           = ictx->OW;
    ثابت uint32_t KD = ictx->KD, KH = ictx->KH;
    ثابت uint32_t kd_kh        = KD * KH;
    ثابت uint32_t patch_stride = ictx->patch_stride;
    ثابت uint32_t owb          = ictx->pe_owb;
    ثابت uint32_t nbpr         = ictx->pe_nbpr;
    ثابت uint32_t dst_elem     = ictx->dst_elem;
    ثابت dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    اعشاری * srcb2[2]            = { (اعشاری *) srcb_base, (اعشاری *) (srcb_base + ictx->pe_src_row_bytes) };
    تهی * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    اعشاری * scratch             = (اعشاری *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    ثابت uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    ثابت uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ثابت uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    اگر (row_start >= row_end) {
        بازگشت;
    }
    ثابت uint32_t nrows_local  = row_end - row_start;
    ثابت uint32_t total_blocks = nrows_local * nbpr;
    اگر (total_blocks == 0) {
        بازگشت;
    }
    ساختار im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ساختار im2col_cursor stage = cur;
    برای (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_افزودن(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_block(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    برای (uint32_t bi = 0; bi < total_blocks; bi++) {
        ثابت uint32_t buf = bi & 1u;
        اعشاری * srcb = srcb2[buf];
        تهی *  dstb = dstb2[buf];
        ثابت uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_queue_برداشت(dma_q);
        برای (uint32_t p = 0; p < kd_kh; p++) {
            dma_queue_برداشت(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr = dst_data + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_افزودن(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        اگر (bi + 2 < total_blocks) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_block(&stage, ictx);
        }
        im2col_3d_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}
ایستا تهی im2col_3d_patchembed_dma_stage(ساختار htp_im2col_context * ictx, dma_queue * dma_q,
                                           اعشاری * srcb, ثابت ساختار im2col_cursor * c) {
    ثابت ساختار htp_ops_context * octx = ictx->octx;
    ثابت uint32_t IC = ictx->IC;
    ثابت uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ثابت uint32_t KD = ictx->KD, KH = ictx->KH;
    ثابت dma_addr_t src_data = octx->src[1]->data;
    ثابت uint32_t in = c->in, iod = c->iod, ioh = c->ioh;
    ثابت size_t src_channel_stride  = (size_t) ID * IH * IW * اندازه(اعشاری);
    ثابت size_t vtcm_channel_stride = (size_t) KD * KH * IW * اندازه(اعشاری);
    ثابت size_t row_size            = (size_t) KH * IW * اندازه(اعشاری);

    برای (uint32_t ikd = 0; ikd < KD; ikd++) {
        ثابت uint32_t iid = iod * KD + ikd;
        اعشاری * vdst = srcb + (size_t) ikd * KH * IW;
        ثابت dma_addr_t vsrc = src_data + (size_t) (((in * IC) * ID + iid) * IH + ioh * KH) * IW * اندازه(اعشاری);
        dma_queue_افزودن(dma_q, dma_make_data(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, row_size, IC);
    }
}
ایستا تهی im2col_3d_patchembed_compute_f16(
    ثابت ساختار htp_im2col_context * ictx,
    تهی * restrict dstb_raw, ثابت اعشاری * restrict srcb) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    ثابت uint32_t OW           = ictx->OW;
    ثابت uint32_t IC           = ictx->IC;
    ثابت uint32_t KD           = ictx->KD;
    ثابت uint32_t KH           = ictx->KH;
    ثابت uint32_t KW           = ictx->KW;
    ثابت uint32_t IW           = ictx->IW;
    ثابت uint32_t patch_stride = ictx->patch_stride;

    برای (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        برای (uint32_t iic = 0; iic < IC; iic++) {
            برای (uint32_t ikd = 0; ikd < KD; ikd++) {
                برای (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ثابت اعشاری * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) out_run, (ثابت uint8_t *) src_run, KW);
                }
            }
        }
    }
}

ایستا تهی im2col_3d_patchembed_compute_f32(
    ثابت ساختار htp_im2col_context * ictx,
    تهی * restrict dstb_raw, ثابت اعشاری * restrict srcb) {
    اعشاری * restrict dstb = (اعشاری *) dstb_raw;
    ثابت uint32_t OW           = ictx->OW;
    ثابت uint32_t IC           = ictx->IC;
    ثابت uint32_t KD           = ictx->KD;
    ثابت uint32_t KH           = ictx->KH;
    ثابت uint32_t KW           = ictx->KW;
    ثابت uint32_t IW           = ictx->IW;
    ثابت uint32_t patch_stride = ictx->patch_stride;

    برای (uint32_t iow = 0; iow < OW; iow++) {
        اعشاری * dst_patch = dstb + (size_t) iow * patch_stride;
        برای (uint32_t iic = 0; iic < IC; iic++) {
            برای (uint32_t ikd = 0; ikd < KD; ikd++) {
                برای (uint32_t ikh = 0; ikh < KH; ikh++) {
                    اعشاری * out_run = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ثابت اعشاری * src_run = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) out_run, (ثابت uint8_t *) src_run, KW);
                }
            }
        }
    }
}

ایستا تهی im2col_3d_patchembed_dma_thread(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    (تهی) nth;
    ساختار htp_im2col_context * ictx        = (ساختار htp_im2col_context *) data;
    ساختار htp_ops_context *    octx        = ictx->octx;
    ساختار htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    ثابت ساختار htp_tensor * restrict dst  = octx->dst;
    ثابت uint32_t KD           = ictx->KD;
    ثابت uint32_t OW           = ictx->OW;
    ثابت uint32_t patch_stride = ictx->patch_stride;
    ثابت uint32_t dst_elem     = ictx->dst_elem;
    ثابت dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    ثابت uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    ثابت uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    ثابت uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    اگر (row_start >= row_end) {
        بازگشت;
    }

    ثابت uint32_t nrows_local  = row_end - row_start;
    uint8_t * src_base          = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    اعشاری * srcb2[2]            = { (اعشاری *) src_base, (اعشاری *) (src_base + ictx->pe_src_row_bytes) };
    تهی * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    ساختار im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ساختار im2col_cursor stage = cur;
    برای (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_queue_افزودن(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_row(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    برای (uint32_t idx = 0; idx < nrows_local; idx++) {
        ثابت uint32_t buf = idx & 1u;
        اعشاری * srcb = srcb2[buf];
        تهی *  dstb = dstb2[buf];
        dma_queue_برداشت(dma_q);
        برای (uint32_t p = 0; p < KD; p++) {
            dma_queue_برداشت(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ثابت dma_addr_t ddr_row = dst_data + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_افزودن(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        اگر (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_row(&stage, ictx);
        }
        im2col_3d_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

صحیح op_im2col(ساختار htp_ops_context * octx) {
    ثابت ساختار htp_im2col_kernel_params * kparams = (ثابت ساختار htp_im2col_kernel_params *) octx->kernel_params;
    اگر (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    اگر (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        بازگشت HTP_STATUS_INVAL_PARAMS;
    }

    ثابت ساختار htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    اگر (octx->ctx->mdev.شمار > 1) {
        ثابت uint32_t row_size = dst->nb[2];
        ثابت uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_سطر_SIZE / hex_gcd_u32(row_size, HEX_L2_سطر_SIZE)) : 1;
        ثابت ساختار htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_base = range.start;
        nrows    = range.شمار;
    }

    اگر (nrows == 0) {
        بازگشت HTP_STATUS_OK;
    }

    ساختار htp_im2col_context ictx = { 0 };
    uint8_t * ثابت base = octx->ctx->vtcm_base;

    ثابت uint32_t pth = MIN(octx->n_threads, nrows);
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

    برای (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_in[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? اندازه(__fp16) : اندازه(اعشاری);

    اگر (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (تهی *) im2col_patchembed_compute_f16
                                                 : (تهی *) im2col_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_dma_thread, &ictx, pth);
        بازگشت HTP_STATUS_OK;
    } وگرنه اگر (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (تهی *) im2col_blocked_compute_f16
                                                 : (تهی *) im2col_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_blocked_dma_thread, &ictx, pth);
        بازگشت HTP_STATUS_OK;
    }

    بازگشت HTP_STATUS_NO_SUPPORT;
}

صحیح op_im2col_3d(ساختار htp_ops_context * octx) {
    ثابت ساختار htp_im2col_kernel_params * kparams =
        (ثابت ساختار htp_im2col_kernel_params *) octx->kernel_params;
    اگر (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    اگر (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        بازگشت HTP_STATUS_INVAL_PARAMS;
    }

    ثابت ساختار htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    اگر (octx->ctx->mdev.شمار > 1) {
        ثابت uint32_t row_size = dst->nb[2];
        ثابت uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_سطر_SIZE / hex_gcd_u32(row_size, HEX_L2_سطر_SIZE)) : 1;
        ثابت ساختار htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_base = range.start;
        nrows    = range.شمار;
    }

    اگر (nrows == 0) {
        بازگشت HTP_STATUS_OK;
    }

    ساختار htp_im2col_context ictx = { 0 };
    uint8_t * ثابت base = octx->ctx->vtcm_base;

    ثابت uint32_t pth = MIN(octx->n_threads, nrows);
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

    برای (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_in[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? اندازه(__fp16) : اندازه(اعشاری);

    اگر (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (تهی *) im2col_3d_patchembed_compute_f16
                                                 : (تهی *) im2col_3d_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_patchembed_dma_thread, &ictx, pth);
        بازگشت HTP_STATUS_OK;
    } وگرنه اگر (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (تهی *) im2col_3d_blocked_compute_f16
                                                 : (تهی *) im2col_3d_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_blocked_dma_thread, &ictx, pth);
        بازگشت HTP_STATUS_OK;
    }

    بازگشت HTP_STATUS_NO_SUPPORT;
}
