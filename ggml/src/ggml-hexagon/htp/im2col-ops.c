#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <hexagon_protos.h>
#अन्तर्भाव <hexagon_types.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-common.h"

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-vtcm.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "im2col-ops.h"

संरचना htp_im2col_context {
    संरचना htp_ops_context * octx;

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

    शून्य * compute;
    uint32_t dst_elem;
};

प्रकारनिर्देश शून्य (*im2col_blocked_compute_fn_t)(
    अचर संरचना htp_im2col_context * ictx,
    शून्य * restrict dstb, अचर भिन्न * restrict srcb,
    भिन्न * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

प्रकारनिर्देश शून्य (*im2col_patchembed_compute_fn_t)(
    अचर संरचना htp_im2col_context * ictx,
    शून्य * restrict dstb, अचर भिन्न * restrict srcb, uint32_t ioh);

प्रकारनिर्देश शून्य (*im2col_3d_blocked_compute_fn_t)(
    अचर संरचना htp_im2col_context * ictx,
    शून्य * restrict dstb, अचर भिन्न * restrict srcb,
    भिन्न * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

प्रकारनिर्देश शून्य (*im2col_3d_patchembed_compute_fn_t)(
    अचर संरचना htp_im2col_context * ictx,
    शून्य * restrict dstb, अचर भिन्न * restrict srcb);

संरचना im2col_cursor {
    uint32_t r;
    uint32_t in;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

स्थिर अन्तःस्थ शून्य im2col_step_row(संरचना im2col_cursor * c, अचर संरचना htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    यदि (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->in++;
    }
}

स्थिर अन्तःस्थ शून्य im2col_step_block(संरचना im2col_cursor * c, अचर संरचना htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    यदि (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        यदि (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->in++;
        }
    }
}

स्थिर शून्य im2col_blocked_dma_stage(संरचना htp_im2col_context * ictx, dma_queue * dma_q,
                                     भिन्न * srcb, अचर संरचना im2col_cursor * c) {
    अचर uint32_t in = c->in, ioh = c->ioh, c0 = c->c0;
    अचर संरचना htp_ops_context * octx = ictx->octx;
    अचर संरचना htp_tensor * src1 = octx->src[1];
    अचर int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    अचर int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    अचर int32_t d1 = octx->op_params[5];
    अचर bool is_2D = (octx->op_params[6] == 1);
    अचर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    अचर uint32_t KH = ictx->KH;
    अचर uint32_t Wb = ictx->pe_wb;
    अचर dma_addr_t src_data = src1->data;
    अचर int32_t win0 = (int32_t) c0 * s0 - p0;
    अचर माप_प्रकार src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    अचर माप_प्रकार batch_offset        = is_2D ? (माप_प्रकार) in * src1->nb[3] : (माप_प्रकार) in * src1->nb[2];

    लागि (uint32_t ikh = 0; ikh < KH; ikh++) {
        अचर int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        यदि (iih < 0 || iih >= (int32_t) IH) {
            dma_queue_थप(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            जारी;
        }
        अचर int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        यदि (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        यदि (hi <= lo) {
            dma_queue_थप(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
            जारी;
        }
        अचर uint32_t cpw = (uint32_t) (hi - lo);
        भिन्न * vdst = srcb + (माप_प्रकार) ikh * Wb + lo;
        अचर dma_addr_t vsrc = src_data + batch_offset + (माप_प्रकार) iih * src1->nb[1] + (माप_प्रकार) (win0 + lo) * src1->nb[0];
        dma_queue_थप(dma_q, dma_make_data(vdst, vsrc),
                       (माप_प्रकार) KH * Wb * परिमाण(भिन्न), src_channel_stride,
                       cpw * परिमाण(भिन्न), IC);
    }
}

#define IM2COL_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
स्थिर शून्य FNAME(                                                                               \
    अचर संरचना htp_im2col_context * ictx,                                                      \
    शून्य * restrict dstb_raw, अचर भिन्न * restrict srcb,                                       \
    भिन्न * restrict scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                          \
    अचर संरचना htp_ops_context * octx = ictx->octx;                                            \
    अचर int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    अचर int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    अचर int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    अचर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    अचर uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    अचर uint32_t Wb = ictx->pe_wb;                                                             \
    अचर uint32_t patch_stride = ictx->patch_stride;                                            \
    अचर int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    लागि (uint32_t j = 0; j < nb; j++) {                                                          \
        अचर uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (माप_प्रकार) j * patch_stride;                             \
        अचर int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        लागि (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            अचर int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            अचर पूर्णाङ्क     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            लागि (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * out_run = dst_patch + (माप_प्रकार) (iic * KH + ikh) * KW;                \
                यदि (!okh) {                                                                      \
                    SPLAT_FN(out_run, 0, KW);                                                    \
                    जारी;                                                                    \
                }                                                                                \
                अचर भिन्न * vrow = srcb + (माप_प्रकार) (iic * KH + ikh) * Wb;                      \
                यदि (d0 == 1) {                                                                   \
                    अचर int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    यदि (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    यदि (hi <= lo) {                                                              \
                        SPLAT_FN(out_run, 0, KW);                                                \
                    } अन्यथा {                                                                     \
                        यदि (lo > 0) {                                                            \
                            SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (out_run + lo),                                      \
                                (अचर uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        यदि (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    जारी;                                                                    \
                }                                                                                \
                लागि (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    अचर uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    लागि (uint32_t c = 0; c < chunk; c++) {                                       \
                        अचर int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (out_run + ikw), (अचर uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, भिन्न,    hvx_copy_f32_uu,     hvx_splat_f32_u)

स्थिर शून्य im2col_blocked_dma_thread(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना htp_im2col_context * ictx      = (संरचना htp_im2col_context *) data;
    संरचना htp_ops_context *    octx      = ictx->octx;
    संरचना htp_thread_trace * restrict tr = &octx->ctx->trace[ith];
    अचर uint32_t OW                     = ictx->OW;
    अचर uint32_t KH                     = ictx->KH;
    अचर uint32_t owb                    = ictx->pe_owb;
    अचर uint32_t nbpr                   = ictx->pe_nbpr;
    अचर uint32_t patch_stride           = ictx->patch_stride;
    अचर uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    अचर dma_addr_t dst_data             = octx->dst->data;
    dma_queue *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_base              = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    भिन्न *        srcb2[2]               = { (भिन्न *) srcb_base, (भिन्न *) (srcb_base + ictx->pe_src_row_bytes) };
    शून्य *         dstb2[2]               = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    भिन्न *        scratch                = (भिन्न *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    अचर uint32_t row_end_max            = ictx->pe_row_base + ictx->pe_nrows;
    अचर uint32_t row_start              = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    अचर uint32_t row_end                = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    यदि (row_start >= row_end) {
        फर्काऊ;
    }
    अचर uint32_t nrows_local  = row_end - row_start;
    अचर uint32_t total_blocks = nrows_local * nbpr;
    यदि (total_blocks == 0) {
        फर्काऊ;
    }
    संरचना im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    संरचना im2col_cursor stage = cur;
    लागि (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_थप(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_block(&stage, ictx);
    }
    लागि (uint32_t bi = 0; bi < total_blocks; bi++) {
        अचर uint32_t buf = bi & 1u;
        भिन्न * srcb = srcb2[buf];
        शून्य *  dstb = dstb2[buf];
        अचर uint32_t nb = MIN(owb, OW - cur.c0);
        dma_queue_झिक(dma_q);
        लागि (uint32_t k = 0; k < KH; k++) {
            dma_queue_झिक(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        अचर dma_addr_t ddr = dst_data + (माप_प्रकार) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_थप(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        यदि (bi + 2 < total_blocks) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_block(&stage, ictx);
        }
        im2col_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

स्थिर शून्य im2col_patchembed_dma_stage(संरचना htp_im2col_context * ictx, dma_queue * dma_q,
                                        भिन्न * srcb, अचर संरचना im2col_cursor * c) {
    अचर संरचना htp_ops_context * octx = ictx->octx;
    अचर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    अचर uint32_t KH = ictx->KH;
    अचर uint32_t in = c->in, ioh = c->ioh;
    अचर माप_प्रकार src_channel_stride  = (माप_प्रकार) IH * IW * परिमाण(भिन्न);
    अचर माप_प्रकार vtcm_channel_stride = (माप_प्रकार) KH * IW * परिमाण(भिन्न);
    अचर dma_addr_t src_data        = octx->src[1]->data;
    अचर dma_addr_t vsrc            = src_data + (माप_प्रकार) (((in * IC) * IH + ioh * KH) * IW) * परिमाण(भिन्न);

    dma_queue_थप(dma_q, dma_make_data(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

स्थिर शून्य im2col_patchembed_compute_f16(
    अचर संरचना htp_im2col_context * ictx,
    शून्य * restrict dstb_raw, अचर भिन्न * restrict srcb, uint32_t ioh) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    अचर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    अचर uint32_t KH = ictx->KH, KW = ictx->KW;
    अचर uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    लागि (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (माप_प्रकार) iow * patch_stride;
        लागि (uint32_t ikh = 0; ikh < KH; ikh++) {
            अचर int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            लागि (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * out_run = dst_patch + (माप_प्रकार) (iic * KH + ikh) * KW;
                यदि (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(out_run, 0, KW);
                    जारी;
                }
                अचर भिन्न * src_run = srcb + (माप_प्रकार) (iic * KH + ikh) * IW + (माप_प्रकार) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) out_run, (अचर uint8_t *) src_run, KW);
            }
        }
    }
}

स्थिर शून्य im2col_patchembed_compute_f32(
    अचर संरचना htp_im2col_context * ictx,
    शून्य * restrict dstb_raw, अचर भिन्न * restrict srcb, uint32_t ioh) {
    भिन्न * restrict dstb = (भिन्न *) dstb_raw;
    अचर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    अचर uint32_t KH = ictx->KH, KW = ictx->KW;
    अचर uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    लागि (uint32_t iow = 0; iow < OW; iow++) {
        भिन्न * dst_patch = dstb + (माप_प्रकार) iow * patch_stride;
        लागि (uint32_t ikh = 0; ikh < KH; ikh++) {
            अचर int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            लागि (uint32_t iic = 0; iic < IC; iic++) {
                भिन्न * out_run = dst_patch + (माप_प्रकार) (iic * KH + ikh) * KW;
                यदि (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(out_run, 0, KW);
                    जारी;
                }
                अचर भिन्न * src_run = srcb + (माप_प्रकार) (iic * KH + ikh) * IW + (माप_प्रकार) iow * KW;
                hvx_copy_f32_uu((uint8_t *) out_run, (अचर uint8_t *) src_run, KW);
            }
        }
    }
}

स्थिर शून्य im2col_patchembed_dma_thread(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना htp_im2col_context * ictx        = (संरचना htp_im2col_context *) data;
    संरचना htp_ops_context *    octx        = ictx->octx;
    संरचना htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    अचर uint32_t OW                       = ictx->OW;
    अचर uint32_t patch_stride             = ictx->patch_stride;
    अचर uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    अचर dma_addr_t dst_data               = octx->dst->data;
    dma_queue *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_base                = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t *      dstb_base                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    भिन्न *        srcb2[2]                 = { (भिन्न *) srcb_base, (भिन्न *) (srcb_base + ictx->pe_src_row_bytes) };
    शून्य *         dstb2[2]                 = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    अचर uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;
    अचर uint32_t row_start                = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    अचर uint32_t row_end                  = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    यदि (row_start >= row_end) {
        फर्काऊ;
    }
    अचर uint32_t total_rows = row_end - row_start;
    यदि (total_rows == 0) {
        फर्काऊ;
    }
    संरचना im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    संरचना im2col_cursor stage = cur;
    लागि (uint32_t i = 0; i < total_rows && i < 2; i++) {
        dma_queue_थप(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_row(&stage, ictx);
    }
    लागि (uint32_t ri = 0; ri < total_rows; ri++) {
        अचर uint32_t buf = ri & 1u;
        भिन्न * srcb = srcb2[buf];
        शून्य *  dstb = dstb2[buf];
        dma_queue_झिक(dma_q);
        dma_queue_झिक(dma_q);
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        अचर dma_addr_t ddr_row = dst_data + (माप_प्रकार) cur.r * OW * patch_stride * dst_elem;
        dma_queue_थप(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        यदि (ri + 2 < total_rows) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_row(&stage, ictx);
        }
        im2col_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

स्थिर अन्तःस्थ शून्य im2col_3d_step_row(संरचना im2col_cursor * c, अचर संरचना htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    यदि (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        यदि (c->iod >= ictx->OD) {
            c->iod = 0;
            c->in++;
        }
    }
}

स्थिर अन्तःस्थ शून्य im2col_3d_step_block(संरचना im2col_cursor * c, अचर संरचना htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    यदि (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        यदि (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            यदि (c->iod >= ictx->OD) {
                c->iod = 0;
                c->in++;
            }
        }
    }
}

स्थिर शून्य im2col_3d_blocked_dma_stage(संरचना htp_im2col_context * ictx, dma_queue * dma_q,
                                        भिन्न * srcb, अचर संरचना im2col_cursor * c) {
    अचर uint32_t in = c->in, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    अचर संरचना htp_ops_context * octx = ictx->octx;
    अचर संरचना htp_tensor * src1 = octx->src[1];
    अचर int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    अचर int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    अचर int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    अचर uint32_t IC = ictx->IC;
    अचर uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    अचर uint32_t KD = ictx->KD, KH = ictx->KH;
    अचर uint32_t Wb = ictx->pe_wb;
    अचर int32_t win0 = (int32_t) c0 * s0 - p0;
    अचर dma_addr_t src_data = src1->data;
    अचर माप_प्रकार src_channel_stride = src1->nb[3];
    अचर माप_प्रकार vtcm_channel_stride = (माप_प्रकार) KD * KH * Wb * परिमाण(भिन्न);
    अचर माप_प्रकार batch_offset        = (माप_प्रकार) (in * IC) * src1->nb[3];

    लागि (uint32_t ikd = 0; ikd < KD; ikd++) {
        अचर int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        अचर bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        लागि (uint32_t ikh = 0; ikh < KH; ikh++) {
            अचर int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            यदि (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_queue_थप(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                जारी;
            }
            अचर int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            यदि (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            यदि (hi <= lo) {
                dma_queue_थप(dma_q, dma_make_data(srcb, 0), 0, 0, 0, 0);
                जारी;
            }
            भिन्न * vdst = srcb + (माप_प्रकार) (ikd * KH + ikh) * Wb + lo;
            अचर dma_addr_t vsrc = src_data + batch_offset +
                (माप_प्रकार) iid * src1->nb[2] +
                (माप_प्रकार) iih * src1->nb[1] +
                (माप_प्रकार) (win0 + lo) * src1->nb[0];
            dma_queue_थप(dma_q, dma_make_data(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (माप_प्रकार) (hi - lo) * परिमाण(भिन्न), IC);
        }
    }
}

#define IM2COL_3D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
स्थिर शून्य FNAME(अचर संरचना htp_im2col_context * ictx,                                            \
                  शून्य * restrict dstb_raw, अचर भिन्न * restrict srcb,                             \
                  भिन्न * restrict scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                              \
    अचर संरचना htp_ops_context * octx = ictx->octx;                                                \
    अचर int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    अचर int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    अचर int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    अचर uint32_t IC = ictx->IC;                                                                    \
    अचर uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    अचर uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    अचर uint32_t patch_stride = ictx->patch_stride;                                                \
    अचर uint32_t Wb = ictx->pe_wb;                                                                 \
    अचर int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    लागि (uint32_t j = 0; j < nb; j++) {                                                              \
        अचर uint32_t iow = c0 + j;                                                                 \
        अचर int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (माप_प्रकार) j * patch_stride;                                    \
        लागि (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            लागि (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                अचर int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                लागि (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    अचर int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * out_run = dst_patch +                                                \
                        (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    यदि (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(out_run, 0, KW);                                                    \
                        जारी;                                                                    \
                    }                                                                                \
                    अचर भिन्न * vrow = srcb + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    यदि (d0 == 1) {                                                                   \
                        अचर int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        यदि (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        यदि (hi <= lo) {                                                              \
                            SPLAT_FN(out_run, 0, KW);                                                \
                        } अन्यथा {                                                                     \
                            यदि (lo > 0) {                                                            \
                                SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (out_run + lo),                                      \
                                    (अचर uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            यदि (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        जारी;                                                                    \
                    }                                                                                \
                    लागि (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        अचर uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        लागि (uint32_t c = 0; c < chunk; c++) {                                       \
                            अचर int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (out_run + ikw), (अचर uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, भिन्न,    hvx_copy_f32_uu,     hvx_splat_f32_u)

स्थिर शून्य im2col_3d_blocked_dma_thread(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना htp_im2col_context * ictx        = (संरचना htp_im2col_context *) data;
    संरचना htp_ops_context *    octx        = ictx->octx;
    संरचना htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    अचर संरचना htp_tensor * restrict dst  = octx->dst;
    अचर uint32_t OW           = ictx->OW;
    अचर uint32_t KD = ictx->KD, KH = ictx->KH;
    अचर uint32_t kd_kh        = KD * KH;
    अचर uint32_t patch_stride = ictx->patch_stride;
    अचर uint32_t owb          = ictx->pe_owb;
    अचर uint32_t nbpr         = ictx->pe_nbpr;
    अचर uint32_t dst_elem     = ictx->dst_elem;
    अचर dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    भिन्न * srcb2[2]            = { (भिन्न *) srcb_base, (भिन्न *) (srcb_base + ictx->pe_src_row_bytes) };
    शून्य * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    भिन्न * scratch             = (भिन्न *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    अचर uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    अचर uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    अचर uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    यदि (row_start >= row_end) {
        फर्काऊ;
    }
    अचर uint32_t nrows_local  = row_end - row_start;
    अचर uint32_t total_blocks = nrows_local * nbpr;
    यदि (total_blocks == 0) {
        फर्काऊ;
    }
    संरचना im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    संरचना im2col_cursor stage = cur;
    लागि (uint32_t i = 0; i < total_blocks && i < 2; i++) {
        dma_queue_थप(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_block(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    लागि (uint32_t bi = 0; bi < total_blocks; bi++) {
        अचर uint32_t buf = bi & 1u;
        भिन्न * srcb = srcb2[buf];
        शून्य *  dstb = dstb2[buf];
        अचर uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_queue_झिक(dma_q);
        लागि (uint32_t p = 0; p < kd_kh; p++) {
            dma_queue_झिक(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        अचर dma_addr_t ddr = dst_data + (माप_प्रकार) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_थप(dma_q, dma_make_data(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        यदि (bi + 2 < total_blocks) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_block(&stage, ictx);
        }
        im2col_3d_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}
स्थिर शून्य im2col_3d_patchembed_dma_stage(संरचना htp_im2col_context * ictx, dma_queue * dma_q,
                                           भिन्न * srcb, अचर संरचना im2col_cursor * c) {
    अचर संरचना htp_ops_context * octx = ictx->octx;
    अचर uint32_t IC = ictx->IC;
    अचर uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    अचर uint32_t KD = ictx->KD, KH = ictx->KH;
    अचर dma_addr_t src_data = octx->src[1]->data;
    अचर uint32_t in = c->in, iod = c->iod, ioh = c->ioh;
    अचर माप_प्रकार src_channel_stride  = (माप_प्रकार) ID * IH * IW * परिमाण(भिन्न);
    अचर माप_प्रकार vtcm_channel_stride = (माप_प्रकार) KD * KH * IW * परिमाण(भिन्न);
    अचर माप_प्रकार row_size            = (माप_प्रकार) KH * IW * परिमाण(भिन्न);

    लागि (uint32_t ikd = 0; ikd < KD; ikd++) {
        अचर uint32_t iid = iod * KD + ikd;
        भिन्न * vdst = srcb + (माप_प्रकार) ikd * KH * IW;
        अचर dma_addr_t vsrc = src_data + (माप_प्रकार) (((in * IC) * ID + iid) * IH + ioh * KH) * IW * परिमाण(भिन्न);
        dma_queue_थप(dma_q, dma_make_data(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, row_size, IC);
    }
}
स्थिर शून्य im2col_3d_patchembed_compute_f16(
    अचर संरचना htp_im2col_context * ictx,
    शून्य * restrict dstb_raw, अचर भिन्न * restrict srcb) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    अचर uint32_t OW           = ictx->OW;
    अचर uint32_t IC           = ictx->IC;
    अचर uint32_t KD           = ictx->KD;
    अचर uint32_t KH           = ictx->KH;
    अचर uint32_t KW           = ictx->KW;
    अचर uint32_t IW           = ictx->IW;
    अचर uint32_t patch_stride = ictx->patch_stride;

    लागि (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (माप_प्रकार) iow * patch_stride;
        लागि (uint32_t iic = 0; iic < IC; iic++) {
            लागि (uint32_t ikd = 0; ikd < KD; ikd++) {
                लागि (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * out_run = dst_patch + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * KW;
                    अचर भिन्न * src_run = srcb + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * IW + (माप_प्रकार) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) out_run, (अचर uint8_t *) src_run, KW);
                }
            }
        }
    }
}

स्थिर शून्य im2col_3d_patchembed_compute_f32(
    अचर संरचना htp_im2col_context * ictx,
    शून्य * restrict dstb_raw, अचर भिन्न * restrict srcb) {
    भिन्न * restrict dstb = (भिन्न *) dstb_raw;
    अचर uint32_t OW           = ictx->OW;
    अचर uint32_t IC           = ictx->IC;
    अचर uint32_t KD           = ictx->KD;
    अचर uint32_t KH           = ictx->KH;
    अचर uint32_t KW           = ictx->KW;
    अचर uint32_t IW           = ictx->IW;
    अचर uint32_t patch_stride = ictx->patch_stride;

    लागि (uint32_t iow = 0; iow < OW; iow++) {
        भिन्न * dst_patch = dstb + (माप_प्रकार) iow * patch_stride;
        लागि (uint32_t iic = 0; iic < IC; iic++) {
            लागि (uint32_t ikd = 0; ikd < KD; ikd++) {
                लागि (uint32_t ikh = 0; ikh < KH; ikh++) {
                    भिन्न * out_run = dst_patch + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * KW;
                    अचर भिन्न * src_run = srcb + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * IW + (माप_प्रकार) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) out_run, (अचर uint8_t *) src_run, KW);
                }
            }
        }
    }
}

स्थिर शून्य im2col_3d_patchembed_dma_thread(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना htp_im2col_context * ictx        = (संरचना htp_im2col_context *) data;
    संरचना htp_ops_context *    octx        = ictx->octx;
    संरचना htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    अचर संरचना htp_tensor * restrict dst  = octx->dst;
    अचर uint32_t KD           = ictx->KD;
    अचर uint32_t OW           = ictx->OW;
    अचर uint32_t patch_stride = ictx->patch_stride;
    अचर uint32_t dst_elem     = ictx->dst_elem;
    अचर dma_addr_t dst_data   = dst->data;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    अचर uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    अचर uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    अचर uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    यदि (row_start >= row_end) {
        फर्काऊ;
    }

    अचर uint32_t nrows_local  = row_end - row_start;
    uint8_t * src_base          = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;
    भिन्न * srcb2[2]            = { (भिन्न *) src_base, (भिन्न *) (src_base + ictx->pe_src_row_bytes) };
    शून्य * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    संरचना im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    संरचना im2col_cursor stage = cur;
    लागि (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_queue_थप(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_row(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    लागि (uint32_t idx = 0; idx < nrows_local; idx++) {
        अचर uint32_t buf = idx & 1u;
        भिन्न * srcb = srcb2[buf];
        शून्य *  dstb = dstb2[buf];
        dma_queue_झिक(dma_q);
        लागि (uint32_t p = 0; p < KD; p++) {
            dma_queue_झिक(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        अचर dma_addr_t ddr_row = dst_data + (माप_प्रकार) cur.r * OW * patch_stride * dst_elem;
        dma_queue_थप(dma_q, dma_make_data(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        यदि (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_row(&stage, ictx);
        }
        im2col_3d_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

पूर्णाङ्क op_im2col(संरचना htp_ops_context * octx) {
    अचर संरचना htp_im2col_kernel_params * kparams = (अचर संरचना htp_im2col_kernel_params *) octx->kernel_params;
    यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    यदि (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        फर्काऊ HTP_STATUS_INVAL_PARAMS;
    }

    अचर संरचना htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    यदि (octx->ctx->mdev.गन्ती > 1) {
        अचर uint32_t row_size = dst->nb[2];
        अचर uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_पङ्क्ति_SIZE / hex_gcd_u32(row_size, HEX_L2_पङ्क्ति_SIZE)) : 1;
        अचर संरचना htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.गन्ती, &octx->ctx->mdev.गन्ती_div);
        row_base = range.start;
        nrows    = range.गन्ती;
    }

    यदि (nrows == 0) {
        फर्काऊ HTP_STATUS_OK;
    }

    संरचना htp_im2col_context ictx = { 0 };
    uint8_t * अचर base = octx->ctx->vtcm_base;

    अचर uint32_t pth = MIN(octx->n_threads, nrows);
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

    लागि (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_in[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? परिमाण(__fp16) : परिमाण(भिन्न);

    यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (शून्य *) im2col_patchembed_compute_f16
                                                 : (शून्य *) im2col_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_dma_thread, &ictx, pth);
        फर्काऊ HTP_STATUS_OK;
    } अन्यथा यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (शून्य *) im2col_blocked_compute_f16
                                                 : (शून्य *) im2col_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_blocked_dma_thread, &ictx, pth);
        फर्काऊ HTP_STATUS_OK;
    }

    फर्काऊ HTP_STATUS_NO_SUPPORT;
}

पूर्णाङ्क op_im2col_3d(संरचना htp_ops_context * octx) {
    अचर संरचना htp_im2col_kernel_params * kparams =
        (अचर संरचना htp_im2col_kernel_params *) octx->kernel_params;
    यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    यदि (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        फर्काऊ HTP_STATUS_INVAL_PARAMS;
    }

    अचर संरचना htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->total_rows;
    यदि (octx->ctx->mdev.गन्ती > 1) {
        अचर uint32_t row_size = dst->nb[2];
        अचर uint32_t rows_per_chunk = (row_size > 0) ? (HEX_L2_पङ्क्ति_SIZE / hex_gcd_u32(row_size, HEX_L2_पङ्क्ति_SIZE)) : 1;
        अचर संरचना htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.गन्ती, &octx->ctx->mdev.गन्ती_div);
        row_base = range.start;
        nrows    = range.गन्ती;
    }

    यदि (nrows == 0) {
        फर्काऊ HTP_STATUS_OK;
    }

    संरचना htp_im2col_context ictx = { 0 };
    uint8_t * अचर base = octx->ctx->vtcm_base;

    अचर uint32_t pth = MIN(octx->n_threads, nrows);
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

    लागि (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_in[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? परिमाण(__fp16) : परिमाण(भिन्न);

    यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (शून्य *) im2col_3d_patchembed_compute_f16
                                                 : (शून्य *) im2col_3d_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_patchembed_dma_thread, &ictx, pth);
        फर्काऊ HTP_STATUS_OK;
    } अन्यथा यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (शून्य *) im2col_3d_blocked_compute_f16
                                                 : (शून्य *) im2col_3d_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_blocked_dma_thread, &ictx, pth);
        फर्काऊ HTP_STATUS_OK;
    }

    फर्काऊ HTP_STATUS_NO_SUPPORT;
}
