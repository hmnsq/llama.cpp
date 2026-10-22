#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <hexagon_protos.h>
#समावेश <hexagon_types.h>
#समावेश <string.h>

#समावेश "hex-common.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "hvx-utils.h"
#समावेश "dma-queue.h"
#समावेश "hex-profile.h"
#समावेश "htp-vtcm.h"
#समावेश "htp-tensor.h"
#समावेश "im2col-ops.h"

काष्ठा htp_im2col_context {
    काष्ठा htp_ops_context * octx;

    uint32_t pe_row_base;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_rows_per_thread;                   // output rows per worker
    uint32_t pe_src_row_bytes;                     // one output row's source, rounded 256
    uint32_t pe_dst_row_bytes;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_आकार_per_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_आकार_per_thread;              // 2 * pe_dst_row_bytes

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

    व्योम * compute;
    uint32_t dst_elem;
};

प्रकार व्योम (*im2col_blocked_compute_fn_t)(
    स्थिर काष्ठा htp_im2col_context * ictx,
    व्योम * restrict dstb, स्थिर भग्न * restrict srcb,
    भग्न * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

प्रकार व्योम (*im2col_patchembed_compute_fn_t)(
    स्थिर काष्ठा htp_im2col_context * ictx,
    व्योम * restrict dstb, स्थिर भग्न * restrict srcb, uint32_t ioh);

प्रकार व्योम (*im2col_3d_blocked_compute_fn_t)(
    स्थिर काष्ठा htp_im2col_context * ictx,
    व्योम * restrict dstb, स्थिर भग्न * restrict srcb,
    भग्न * restrict scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

प्रकार व्योम (*im2col_3d_patchembed_compute_fn_t)(
    स्थिर काष्ठा htp_im2col_context * ictx,
    व्योम * restrict dstb, स्थिर भग्न * restrict srcb);

काष्ठा im2col_cursor {
    uint32_t r;
    uint32_t in;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

अटल अंतरभूत व्योम im2col_step_row(काष्ठा im2col_cursor * c, स्थिर काष्ठा htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    अगर (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->in++;
    }
}

अटल अंतरभूत व्योम im2col_step_block(काष्ठा im2col_cursor * c, स्थिर काष्ठा htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    अगर (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        अगर (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->in++;
        }
    }
}

अटल व्योम im2col_blocked_dma_stage(काष्ठा htp_im2col_context * ictx, dma_queue * dma_q,
                                     भग्न * srcb, स्थिर काष्ठा im2col_cursor * c) {
    स्थिर uint32_t in = c->in, ioh = c->ioh, c0 = c->c0;
    स्थिर काष्ठा htp_ops_context * octx = ictx->octx;
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    स्थिर int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    स्थिर int32_t d1 = octx->op_params[5];
    स्थिर bool is_2D = (octx->op_params[6] == 1);
    स्थिर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    स्थिर uint32_t KH = ictx->KH;
    स्थिर uint32_t Wb = ictx->pe_wb;
    स्थिर dma_addr_t src_तथ्य = src1->तथ्य;
    स्थिर int32_t win0 = (int32_t) c0 * s0 - p0;
    स्थिर माप_प्रकार src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    स्थिर माप_प्रकार batch_offset        = is_2D ? (माप_प्रकार) in * src1->nb[3] : (माप_प्रकार) in * src1->nb[2];

    क्रम (uint32_t ikh = 0; ikh < KH; ikh++) {
        स्थिर int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        अगर (iih < 0 || iih >= (int32_t) IH) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(srcb, 0), 0, 0, 0, 0);
            जारी;
        }
        स्थिर int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        अगर (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        अगर (hi <= lo) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(srcb, 0), 0, 0, 0, 0);
            जारी;
        }
        स्थिर uint32_t cpw = (uint32_t) (hi - lo);
        भग्न * vdst = srcb + (माप_प्रकार) ikh * Wb + lo;
        स्थिर dma_addr_t vsrc = src_तथ्य + batch_offset + (माप_प्रकार) iih * src1->nb[1] + (माप_प्रकार) (win0 + lo) * src1->nb[0];
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(vdst, vsrc),
                       (माप_प्रकार) KH * Wb * माप(भग्न), src_channel_stride,
                       cpw * माप(भग्न), IC);
    }
}

#घोषणा IM2COL_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
अटल व्योम FNAME(                                                                               \
    स्थिर काष्ठा htp_im2col_context * ictx,                                                      \
    व्योम * restrict dstb_raw, स्थिर भग्न * restrict srcb,                                       \
    भग्न * restrict scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                          \
    स्थिर काष्ठा htp_ops_context * octx = ictx->octx;                                            \
    स्थिर int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    स्थिर int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    स्थिर int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    स्थिर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    स्थिर uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    स्थिर uint32_t Wb = ictx->pe_wb;                                                             \
    स्थिर uint32_t patch_stride = ictx->patch_stride;                                            \
    स्थिर int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    क्रम (uint32_t j = 0; j < nb; j++) {                                                          \
        स्थिर uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (माप_प्रकार) j * patch_stride;                             \
        स्थिर int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        क्रम (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            स्थिर int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            स्थिर पूर्णांक     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            क्रम (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * out_run = dst_patch + (माप_प्रकार) (iic * KH + ikh) * KW;                \
                अगर (!okh) {                                                                      \
                    SPLAT_FN(out_run, 0, KW);                                                    \
                    जारी;                                                                    \
                }                                                                                \
                स्थिर भग्न * vrow = srcb + (माप_प्रकार) (iic * KH + ikh) * Wb;                      \
                अगर (d0 == 1) {                                                                   \
                    स्थिर int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    अगर (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    अगर (hi <= lo) {                                                              \
                        SPLAT_FN(out_run, 0, KW);                                                \
                    } अन्यथा {                                                                     \
                        अगर (lo > 0) {                                                            \
                            SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (out_run + lo),                                      \
                                (स्थिर uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        अगर (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    जारी;                                                                    \
                }                                                                                \
                क्रम (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    स्थिर uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    क्रम (uint32_t c = 0; c < chunk; c++) {                                       \
                        स्थिर int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (out_run + ikw), (स्थिर uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, भग्न,    hvx_copy_f32_uu,     hvx_splat_f32_u)

अटल व्योम im2col_blocked_dma_thread(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा htp_im2col_context * ictx      = (काष्ठा htp_im2col_context *) तथ्य;
    काष्ठा htp_ops_context *    octx      = ictx->octx;
    काष्ठा htp_thread_trace * restrict tr = &octx->ctx->trace[ith];
    स्थिर uint32_t OW                     = ictx->OW;
    स्थिर uint32_t KH                     = ictx->KH;
    स्थिर uint32_t owb                    = ictx->pe_owb;
    स्थिर uint32_t nbpr                   = ictx->pe_nbpr;
    स्थिर uint32_t patch_stride           = ictx->patch_stride;
    स्थिर uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    स्थिर dma_addr_t dst_तथ्य             = octx->dst->तथ्य;
    dma_queue *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_base              = ictx->pe_vtcm_src + ith * ictx->pe_src_आकार_per_thread;
    uint8_t *      dstb_base              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_आकार_per_thread;
    भग्न *        srcb2[2]               = { (भग्न *) srcb_base, (भग्न *) (srcb_base + ictx->pe_src_row_bytes) };
    व्योम *         dstb2[2]               = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    भग्न *        scratch                = (भग्न *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    स्थिर uint32_t row_end_max            = ictx->pe_row_base + ictx->pe_nrows;
    स्थिर uint32_t row_start              = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    स्थिर uint32_t row_end                = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    अगर (row_start >= row_end) {
        वापस;
    }
    स्थिर uint32_t nrows_local  = row_end - row_start;
    स्थिर uint32_t योग_blocks = nrows_local * nbpr;
    अगर (योग_blocks == 0) {
        वापस;
    }
    काष्ठा im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    काष्ठा im2col_cursor stage = cur;
    क्रम (uint32_t i = 0; i < योग_blocks && i < 2; i++) {
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_तथ्य, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_block(&stage, ictx);
    }
    क्रम (uint32_t bi = 0; bi < योग_blocks; bi++) {
        स्थिर uint32_t buf = bi & 1u;
        भग्न * srcb = srcb2[buf];
        व्योम *  dstb = dstb2[buf];
        स्थिर uint32_t nb = MIN(owb, OW - cur.c0);
        dma_queue_निकालो(dma_q);
        क्रम (uint32_t k = 0; k < KH; k++) {
            dma_queue_निकालो(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        स्थिर dma_addr_t ddr = dst_तथ्य + (माप_प्रकार) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        अगर (bi + 2 < योग_blocks) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_block(&stage, ictx);
        }
        im2col_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

अटल व्योम im2col_patchembed_dma_stage(काष्ठा htp_im2col_context * ictx, dma_queue * dma_q,
                                        भग्न * srcb, स्थिर काष्ठा im2col_cursor * c) {
    स्थिर काष्ठा htp_ops_context * octx = ictx->octx;
    स्थिर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    स्थिर uint32_t KH = ictx->KH;
    स्थिर uint32_t in = c->in, ioh = c->ioh;
    स्थिर माप_प्रकार src_channel_stride  = (माप_प्रकार) IH * IW * माप(भग्न);
    स्थिर माप_प्रकार vtcm_channel_stride = (माप_प्रकार) KH * IW * माप(भग्न);
    स्थिर dma_addr_t src_तथ्य        = octx->src[1]->तथ्य;
    स्थिर dma_addr_t vsrc            = src_तथ्य + (माप_प्रकार) (((in * IC) * IH + ioh * KH) * IW) * माप(भग्न);

    dma_queue_धकेलो(dma_q, dma_make_तथ्य(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

अटल व्योम im2col_patchembed_compute_f16(
    स्थिर काष्ठा htp_im2col_context * ictx,
    व्योम * restrict dstb_raw, स्थिर भग्न * restrict srcb, uint32_t ioh) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    स्थिर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    स्थिर uint32_t KH = ictx->KH, KW = ictx->KW;
    स्थिर uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    क्रम (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (माप_प्रकार) iow * patch_stride;
        क्रम (uint32_t ikh = 0; ikh < KH; ikh++) {
            स्थिर int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            क्रम (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * out_run = dst_patch + (माप_प्रकार) (iic * KH + ikh) * KW;
                अगर (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(out_run, 0, KW);
                    जारी;
                }
                स्थिर भग्न * src_run = srcb + (माप_प्रकार) (iic * KH + ikh) * IW + (माप_प्रकार) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) out_run, (स्थिर uint8_t *) src_run, KW);
            }
        }
    }
}

अटल व्योम im2col_patchembed_compute_f32(
    स्थिर काष्ठा htp_im2col_context * ictx,
    व्योम * restrict dstb_raw, स्थिर भग्न * restrict srcb, uint32_t ioh) {
    भग्न * restrict dstb = (भग्न *) dstb_raw;
    स्थिर uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    स्थिर uint32_t KH = ictx->KH, KW = ictx->KW;
    स्थिर uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    क्रम (uint32_t iow = 0; iow < OW; iow++) {
        भग्न * dst_patch = dstb + (माप_प्रकार) iow * patch_stride;
        क्रम (uint32_t ikh = 0; ikh < KH; ikh++) {
            स्थिर int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            क्रम (uint32_t iic = 0; iic < IC; iic++) {
                भग्न * out_run = dst_patch + (माप_प्रकार) (iic * KH + ikh) * KW;
                अगर (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(out_run, 0, KW);
                    जारी;
                }
                स्थिर भग्न * src_run = srcb + (माप_प्रकार) (iic * KH + ikh) * IW + (माप_प्रकार) iow * KW;
                hvx_copy_f32_uu((uint8_t *) out_run, (स्थिर uint8_t *) src_run, KW);
            }
        }
    }
}

अटल व्योम im2col_patchembed_dma_thread(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा htp_im2col_context * ictx        = (काष्ठा htp_im2col_context *) तथ्य;
    काष्ठा htp_ops_context *    octx        = ictx->octx;
    काष्ठा htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    स्थिर uint32_t OW                       = ictx->OW;
    स्थिर uint32_t patch_stride             = ictx->patch_stride;
    स्थिर uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    स्थिर dma_addr_t dst_तथ्य               = octx->dst->तथ्य;
    dma_queue *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_base                = ictx->pe_vtcm_src + ith * ictx->pe_src_आकार_per_thread;
    uint8_t *      dstb_base                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_आकार_per_thread;
    भग्न *        srcb2[2]                 = { (भग्न *) srcb_base, (भग्न *) (srcb_base + ictx->pe_src_row_bytes) };
    व्योम *         dstb2[2]                 = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    स्थिर uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;
    स्थिर uint32_t row_start                = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    स्थिर uint32_t row_end                  = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    अगर (row_start >= row_end) {
        वापस;
    }
    स्थिर uint32_t योग_rows = row_end - row_start;
    अगर (योग_rows == 0) {
        वापस;
    }
    काष्ठा im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    काष्ठा im2col_cursor stage = cur;
    क्रम (uint32_t i = 0; i < योग_rows && i < 2; i++) {
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_तथ्य, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_step_row(&stage, ictx);
    }
    क्रम (uint32_t ri = 0; ri < योग_rows; ri++) {
        स्थिर uint32_t buf = ri & 1u;
        भग्न * srcb = srcb2[buf];
        व्योम *  dstb = dstb2[buf];
        dma_queue_निकालो(dma_q);
        dma_queue_निकालो(dma_q);
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        स्थिर dma_addr_t ddr_row = dst_तथ्य + (माप_प्रकार) cur.r * OW * patch_stride * dst_elem;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        अगर (ri + 2 < योग_rows) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_step_row(&stage, ictx);
        }
        im2col_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

अटल अंतरभूत व्योम im2col_3d_step_row(काष्ठा im2col_cursor * c, स्थिर काष्ठा htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    अगर (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        अगर (c->iod >= ictx->OD) {
            c->iod = 0;
            c->in++;
        }
    }
}

अटल अंतरभूत व्योम im2col_3d_step_block(काष्ठा im2col_cursor * c, स्थिर काष्ठा htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    अगर (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        अगर (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            अगर (c->iod >= ictx->OD) {
                c->iod = 0;
                c->in++;
            }
        }
    }
}

अटल व्योम im2col_3d_blocked_dma_stage(काष्ठा htp_im2col_context * ictx, dma_queue * dma_q,
                                        भग्न * srcb, स्थिर काष्ठा im2col_cursor * c) {
    स्थिर uint32_t in = c->in, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    स्थिर काष्ठा htp_ops_context * octx = ictx->octx;
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    स्थिर int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    स्थिर int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    स्थिर uint32_t IC = ictx->IC;
    स्थिर uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    स्थिर uint32_t KD = ictx->KD, KH = ictx->KH;
    स्थिर uint32_t Wb = ictx->pe_wb;
    स्थिर int32_t win0 = (int32_t) c0 * s0 - p0;
    स्थिर dma_addr_t src_तथ्य = src1->तथ्य;
    स्थिर माप_प्रकार src_channel_stride = src1->nb[3];
    स्थिर माप_प्रकार vtcm_channel_stride = (माप_प्रकार) KD * KH * Wb * माप(भग्न);
    स्थिर माप_प्रकार batch_offset        = (माप_प्रकार) (in * IC) * src1->nb[3];

    क्रम (uint32_t ikd = 0; ikd < KD; ikd++) {
        स्थिर int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        स्थिर bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        क्रम (uint32_t ikh = 0; ikh < KH; ikh++) {
            स्थिर int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            अगर (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(srcb, 0), 0, 0, 0, 0);
                जारी;
            }
            स्थिर int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            अगर (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            अगर (hi <= lo) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(srcb, 0), 0, 0, 0, 0);
                जारी;
            }
            भग्न * vdst = srcb + (माप_प्रकार) (ikd * KH + ikh) * Wb + lo;
            स्थिर dma_addr_t vsrc = src_तथ्य + batch_offset +
                (माप_प्रकार) iid * src1->nb[2] +
                (माप_प्रकार) iih * src1->nb[1] +
                (माप_प्रकार) (win0 + lo) * src1->nb[0];
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (माप_प्रकार) (hi - lo) * माप(भग्न), IC);
        }
    }
}

#घोषणा IM2COL_3D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
अटल व्योम FNAME(स्थिर काष्ठा htp_im2col_context * ictx,                                            \
                  व्योम * restrict dstb_raw, स्थिर भग्न * restrict srcb,                             \
                  भग्न * restrict scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * restrict dstb = (DST_CTYPE *) dstb_raw;                                              \
    स्थिर काष्ठा htp_ops_context * octx = ictx->octx;                                                \
    स्थिर int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    स्थिर int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    स्थिर int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    स्थिर uint32_t IC = ictx->IC;                                                                    \
    स्थिर uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    स्थिर uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    स्थिर uint32_t patch_stride = ictx->patch_stride;                                                \
    स्थिर uint32_t Wb = ictx->pe_wb;                                                                 \
    स्थिर int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    क्रम (uint32_t j = 0; j < nb; j++) {                                                              \
        स्थिर uint32_t iow = c0 + j;                                                                 \
        स्थिर int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (माप_प्रकार) j * patch_stride;                                    \
        क्रम (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            क्रम (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                स्थिर int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                क्रम (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    स्थिर int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * out_run = dst_patch +                                                \
                        (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    अगर (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(out_run, 0, KW);                                                    \
                        जारी;                                                                    \
                    }                                                                                \
                    स्थिर भग्न * vrow = srcb + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    अगर (d0 == 1) {                                                                   \
                        स्थिर int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        अगर (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        अगर (hi <= lo) {                                                              \
                            SPLAT_FN(out_run, 0, KW);                                                \
                        } अन्यथा {                                                                     \
                            अगर (lo > 0) {                                                            \
                                SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (out_run + lo),                                      \
                                    (स्थिर uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            अगर (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        जारी;                                                                    \
                    }                                                                                \
                    क्रम (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        स्थिर uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        क्रम (uint32_t c = 0; c < chunk; c++) {                                       \
                            स्थिर int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (out_run + ikw), (स्थिर uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, भग्न,    hvx_copy_f32_uu,     hvx_splat_f32_u)

अटल व्योम im2col_3d_blocked_dma_thread(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा htp_im2col_context * ictx        = (काष्ठा htp_im2col_context *) तथ्य;
    काष्ठा htp_ops_context *    octx        = ictx->octx;
    काष्ठा htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    स्थिर काष्ठा htp_tensor * restrict dst  = octx->dst;
    स्थिर uint32_t OW           = ictx->OW;
    स्थिर uint32_t KD = ictx->KD, KH = ictx->KH;
    स्थिर uint32_t kd_kh        = KD * KH;
    स्थिर uint32_t patch_stride = ictx->patch_stride;
    स्थिर uint32_t owb          = ictx->pe_owb;
    स्थिर uint32_t nbpr         = ictx->pe_nbpr;
    स्थिर uint32_t dst_elem     = ictx->dst_elem;
    स्थिर dma_addr_t dst_तथ्य   = dst->तथ्य;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_आकार_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_आकार_per_thread;
    भग्न * srcb2[2]            = { (भग्न *) srcb_base, (भग्न *) (srcb_base + ictx->pe_src_row_bytes) };
    व्योम * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    भग्न * scratch             = (भग्न *) (srcb_base + 2 * ictx->pe_src_row_bytes);
    स्थिर uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    स्थिर uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    स्थिर uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    अगर (row_start >= row_end) {
        वापस;
    }
    स्थिर uint32_t nrows_local  = row_end - row_start;
    स्थिर uint32_t योग_blocks = nrows_local * nbpr;
    अगर (योग_blocks == 0) {
        वापस;
    }
    काष्ठा im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    काष्ठा im2col_cursor stage = cur;
    क्रम (uint32_t i = 0; i < योग_blocks && i < 2; i++) {
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_तथ्य, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_block(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    क्रम (uint32_t bi = 0; bi < योग_blocks; bi++) {
        स्थिर uint32_t buf = bi & 1u;
        भग्न * srcb = srcb2[buf];
        व्योम *  dstb = dstb2[buf];
        स्थिर uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_queue_निकालो(dma_q);
        क्रम (uint32_t p = 0; p < kd_kh; p++) {
            dma_queue_निकालो(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        स्थिर dma_addr_t ddr = dst_तथ्य + (माप_प्रकार) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        अगर (bi + 2 < योग_blocks) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_block(&stage, ictx);
        }
        im2col_3d_step_block(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}
अटल व्योम im2col_3d_patchembed_dma_stage(काष्ठा htp_im2col_context * ictx, dma_queue * dma_q,
                                           भग्न * srcb, स्थिर काष्ठा im2col_cursor * c) {
    स्थिर काष्ठा htp_ops_context * octx = ictx->octx;
    स्थिर uint32_t IC = ictx->IC;
    स्थिर uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    स्थिर uint32_t KD = ictx->KD, KH = ictx->KH;
    स्थिर dma_addr_t src_तथ्य = octx->src[1]->तथ्य;
    स्थिर uint32_t in = c->in, iod = c->iod, ioh = c->ioh;
    स्थिर माप_प्रकार src_channel_stride  = (माप_प्रकार) ID * IH * IW * माप(भग्न);
    स्थिर माप_प्रकार vtcm_channel_stride = (माप_प्रकार) KD * KH * IW * माप(भग्न);
    स्थिर माप_प्रकार row_आकार            = (माप_प्रकार) KH * IW * माप(भग्न);

    क्रम (uint32_t ikd = 0; ikd < KD; ikd++) {
        स्थिर uint32_t iid = iod * KD + ikd;
        भग्न * vdst = srcb + (माप_प्रकार) ikd * KH * IW;
        स्थिर dma_addr_t vsrc = src_तथ्य + (माप_प्रकार) (((in * IC) * ID + iid) * IH + ioh * KH) * IW * माप(भग्न);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, row_आकार, IC);
    }
}
अटल व्योम im2col_3d_patchembed_compute_f16(
    स्थिर काष्ठा htp_im2col_context * ictx,
    व्योम * restrict dstb_raw, स्थिर भग्न * restrict srcb) {
    uint16_t * restrict dstb = (uint16_t *) dstb_raw;
    स्थिर uint32_t OW           = ictx->OW;
    स्थिर uint32_t IC           = ictx->IC;
    स्थिर uint32_t KD           = ictx->KD;
    स्थिर uint32_t KH           = ictx->KH;
    स्थिर uint32_t KW           = ictx->KW;
    स्थिर uint32_t IW           = ictx->IW;
    स्थिर uint32_t patch_stride = ictx->patch_stride;

    क्रम (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (माप_प्रकार) iow * patch_stride;
        क्रम (uint32_t iic = 0; iic < IC; iic++) {
            क्रम (uint32_t ikd = 0; ikd < KD; ikd++) {
                क्रम (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * out_run = dst_patch + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * KW;
                    स्थिर भग्न * src_run = srcb + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * IW + (माप_प्रकार) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) out_run, (स्थिर uint8_t *) src_run, KW);
                }
            }
        }
    }
}

अटल व्योम im2col_3d_patchembed_compute_f32(
    स्थिर काष्ठा htp_im2col_context * ictx,
    व्योम * restrict dstb_raw, स्थिर भग्न * restrict srcb) {
    भग्न * restrict dstb = (भग्न *) dstb_raw;
    स्थिर uint32_t OW           = ictx->OW;
    स्थिर uint32_t IC           = ictx->IC;
    स्थिर uint32_t KD           = ictx->KD;
    स्थिर uint32_t KH           = ictx->KH;
    स्थिर uint32_t KW           = ictx->KW;
    स्थिर uint32_t IW           = ictx->IW;
    स्थिर uint32_t patch_stride = ictx->patch_stride;

    क्रम (uint32_t iow = 0; iow < OW; iow++) {
        भग्न * dst_patch = dstb + (माप_प्रकार) iow * patch_stride;
        क्रम (uint32_t iic = 0; iic < IC; iic++) {
            क्रम (uint32_t ikd = 0; ikd < KD; ikd++) {
                क्रम (uint32_t ikh = 0; ikh < KH; ikh++) {
                    भग्न * out_run = dst_patch + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * KW;
                    स्थिर भग्न * src_run = srcb + (माप_प्रकार) (iic * KD * KH + ikd * KH + ikh) * IW + (माप_प्रकार) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) out_run, (स्थिर uint8_t *) src_run, KW);
                }
            }
        }
    }
}

अटल व्योम im2col_3d_patchembed_dma_thread(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा htp_im2col_context * ictx        = (काष्ठा htp_im2col_context *) तथ्य;
    काष्ठा htp_ops_context *    octx        = ictx->octx;
    काष्ठा htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];
    स्थिर काष्ठा htp_tensor * restrict dst  = octx->dst;
    स्थिर uint32_t KD           = ictx->KD;
    स्थिर uint32_t OW           = ictx->OW;
    स्थिर uint32_t patch_stride = ictx->patch_stride;
    स्थिर uint32_t dst_elem     = ictx->dst_elem;
    स्थिर dma_addr_t dst_तथ्य   = dst->तथ्य;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    स्थिर uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;
    स्थिर uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;
    स्थिर uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);
    अगर (row_start >= row_end) {
        वापस;
    }

    स्थिर uint32_t nrows_local  = row_end - row_start;
    uint8_t * src_base          = ictx->pe_vtcm_src + ith * ictx->pe_src_आकार_per_thread;
    uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_आकार_per_thread;
    भग्न * srcb2[2]            = { (भग्न *) src_base, (भग्न *) (src_base + ictx->pe_src_row_bytes) };
    व्योम * dstb2[2]             = { dstb_base, dstb_base + ictx->pe_dst_row_bytes };
    काष्ठा im2col_cursor cur = {
        .r   = row_start,
        .in  = ictx->thread_in[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    काष्ठा im2col_cursor stage = cur;
    क्रम (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_तथ्य, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_step_row(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    क्रम (uint32_t idx = 0; idx < nrows_local; idx++) {
        स्थिर uint32_t buf = idx & 1u;
        भग्न * srcb = srcb2[buf];
        व्योम *  dstb = dstb2[buf];
        dma_queue_निकालो(dma_q);
        क्रम (uint32_t p = 0; p < KD; p++) {
            dma_queue_निकालो(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        स्थिर dma_addr_t ddr_row = dst_तथ्य + (माप_प्रकार) cur.r * OW * patch_stride * dst_elem;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        अगर (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_step_row(&stage, ictx);
        }
        im2col_3d_step_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

पूर्णांक op_im2col(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_im2col_kernel_params * kparams = (स्थिर काष्ठा htp_im2col_kernel_params *) octx->kernel_params;
    अगर (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर काष्ठा htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->योग_rows;
    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर uint32_t row_आकार = dst->nb[2];
        स्थिर uint32_t rows_per_chunk = (row_आकार > 0) ? (HEX_L2_पंक्ति_आकार / hex_gcd_u32(row_आकार, HEX_L2_पंक्ति_आकार)) : 1;
        स्थिर काष्ठा htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->योग_rows, htp_tensor_mdev_तथ्य_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_base = range.start;
        nrows    = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    काष्ठा htp_im2col_context ictx = { 0 };
    uint8_t * स्थिर base = octx->ctx->vtcm_base;

    स्थिर uint32_t pth = MIN(octx->n_threads, nrows);
    ictx.octx                   = octx;
    ictx.pe_row_base            = row_base;
    ictx.pe_nrows               = nrows;
    ictx.pe_rows_per_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_row_bytes       = kparams->src_row_bytes;
    ictx.pe_dst_row_bytes       = kparams->dst_row_bytes;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_dst);
    ictx.pe_src_आकार_per_thread = kparams->src_bytes_per_thread;
    ictx.pe_dst_आकार_per_thread = kparams->dst_bytes_per_thread;
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

    क्रम (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_in[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? माप(__fp16) : माप(भग्न);

    अगर (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (व्योम *) im2col_patchembed_compute_f16
                                                 : (व्योम *) im2col_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_dma_thread, &ictx, pth);
        वापस HTP_STATUS_OK;
    } अन्यथा अगर (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (व्योम *) im2col_blocked_compute_f16
                                                 : (व्योम *) im2col_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_blocked_dma_thread, &ictx, pth);
        वापस HTP_STATUS_OK;
    }

    वापस HTP_STATUS_NO_SUPPORT;
}

पूर्णांक op_im2col_3d(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_im2col_kernel_params * kparams =
        (स्थिर काष्ठा htp_im2col_kernel_params *) octx->kernel_params;
    अगर (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर काष्ठा htp_tensor * dst = octx->dst;

    uint32_t row_base = 0;
    uint32_t nrows    = kparams->योग_rows;
    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर uint32_t row_आकार = dst->nb[2];
        स्थिर uint32_t rows_per_chunk = (row_आकार > 0) ? (HEX_L2_पंक्ति_आकार / hex_gcd_u32(row_आकार, HEX_L2_पंक्ति_आकार)) : 1;
        स्थिर काष्ठा htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->योग_rows, htp_tensor_mdev_तथ्य_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_base = range.start;
        nrows    = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    काष्ठा htp_im2col_context ictx = { 0 };
    uint8_t * स्थिर base = octx->ctx->vtcm_base;

    स्थिर uint32_t pth = MIN(octx->n_threads, nrows);
    ictx.octx                   = octx;
    ictx.pe_row_base            = row_base;
    ictx.pe_nrows               = nrows;
    ictx.pe_rows_per_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_row_bytes       = kparams->src_row_bytes;
    ictx.pe_dst_row_bytes       = kparams->dst_row_bytes;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, base, kparams->off_dst);
    ictx.pe_src_आकार_per_thread = kparams->src_bytes_per_thread;
    ictx.pe_dst_आकार_per_thread = kparams->dst_bytes_per_thread;
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

    क्रम (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_in[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? माप(__fp16) : माप(भग्न);

    अगर (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (व्योम *) im2col_3d_patchembed_compute_f16
                                                 : (व्योम *) im2col_3d_patchembed_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_patchembed_dma_thread, &ictx, pth);
        वापस HTP_STATUS_OK;
    } अन्यथा अगर (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (व्योम *) im2col_3d_blocked_compute_f16
                                                 : (व्योम *) im2col_3d_blocked_compute_f32;
        work_queue_run(octx->ctx->work_queue, im2col_3d_blocked_dma_thread, &ictx, pth);
        वापस HTP_STATUS_OK;
    }

    वापस HTP_STATUS_NO_SUPPORT;
}
