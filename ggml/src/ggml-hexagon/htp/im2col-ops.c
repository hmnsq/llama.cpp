#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <hexagon_protos.h>
#अन्तर्भाव <hexagon_types.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-common.h"

#परिभाषय GGML_COMMON_DECL_C
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

    uint32_t pe_row_आधारः;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_पङ्क्तिकाः_प्रत्येकम्_thread;                   // output rows per worker
    uint32_t pe_src_row_अष्टकानि;                     // one output row's source, rounded 256
    uint32_t pe_dst_row_अष्टकानि;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_आकार_प्रत्येकम्_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_आकार_प्रत्येकम्_thread;              // 2 * pe_dst_row_bytes

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
    uint32_t thread_अन्तर्[HTP_MAX_NTHREADS];
    uint32_t thread_iod[HTP_MAX_NTHREADS];
    uint32_t thread_ioh[HTP_MAX_NTHREADS];

    शून्य * compute;
    uint32_t dst_elem;
};

प्रकारनिर्देश शून्य (*im2col_blocked_compute_fn_t)(
    ध्रुव संरचना htp_im2col_context * ictx,
    शून्य * सीमित dstb, ध्रुव भिन्न * सीमित srcb,
    भिन्न * सीमित scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

प्रकारनिर्देश शून्य (*im2col_patchembed_compute_fn_t)(
    ध्रुव संरचना htp_im2col_context * ictx,
    शून्य * सीमित dstb, ध्रुव भिन्न * सीमित srcb, uint32_t ioh);

प्रकारनिर्देश शून्य (*im2col_3d_blocked_compute_fn_t)(
    ध्रुव संरचना htp_im2col_context * ictx,
    शून्य * सीमित dstb, ध्रुव भिन्न * सीमित srcb,
    भिन्न * सीमित scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

प्रकारनिर्देश शून्य (*im2col_3d_patchembed_compute_fn_t)(
    ध्रुव संरचना htp_im2col_context * ictx,
    शून्य * सीमित dstb, ध्रुव भिन्न * सीमित srcb);

संरचना im2col_cursor {
    uint32_t r;
    uint32_t अन्तर्;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

स्थिर अन्तःस्थ शून्य im2col_चरणः_row(संरचना im2col_cursor * c, ध्रुव संरचना htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    यदि (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->अन्तर्++;
    }
}

स्थिर अन्तःस्थ शून्य im2col_चरणः_खण्डम्(संरचना im2col_cursor * c, ध्रुव संरचना htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    यदि (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        यदि (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->अन्तर्++;
        }
    }
}

स्थिर शून्य im2col_blocked_dma_stage(संरचना htp_im2col_context * ictx, dma_queue * dma_q,
                                     भिन्न * srcb, ध्रुव संरचना im2col_cursor * c) {
    ध्रुव uint32_t अन्तर् = c->अन्तर्, ioh = c->ioh, c0 = c->c0;
    ध्रुव संरचना htp_ops_context * octx = ictx->octx;
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    ध्रुव int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    ध्रुव int32_t d1 = octx->op_params[5];
    ध्रुव bool is_2D = (octx->op_params[6] == 1);
    ध्रुव uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ध्रुव uint32_t KH = ictx->KH;
    ध्रुव uint32_t Wb = ictx->pe_wb;
    ध्रुव dma_addr_t src_दत्त = src1->दत्त;
    ध्रुव int32_t win0 = (int32_t) c0 * s0 - p0;
    ध्रुव size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    ध्रुव size_t batch_विस्थापनम्        = is_2D ? (size_t) अन्तर् * src1->nb[3] : (size_t) अन्तर् * src1->nb[2];

    प्रति (uint32_t ikh = 0; ikh < KH; ikh++) {
        ध्रुव int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        यदि (iih < 0 || iih >= (int32_t) IH) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(srcb, 0), 0, 0, 0, 0);
            अनुवर्त;
        }
        ध्रुव int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        यदि (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        यदि (hi <= lo) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(srcb, 0), 0, 0, 0, 0);
            अनुवर्त;
        }
        ध्रुव uint32_t cpw = (uint32_t) (hi - lo);
        भिन्न * vdst = srcb + (size_t) ikh * Wb + lo;
        ध्रुव dma_addr_t vsrc = src_दत्त + batch_विस्थापनम् + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vdst, vsrc),
                       (size_t) KH * Wb * परिमाण(भिन्न), src_channel_stride,
                       cpw * परिमाण(भिन्न), IC);
    }
}

#परिभाषय IM2COL_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
स्थिर शून्य FNAME(                                                                               \
    ध्रुव संरचना htp_im2col_context * ictx,                                                      \
    शून्य * सीमित dstb_raw, ध्रुव भिन्न * सीमित srcb,                                       \
    भिन्न * सीमित scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * सीमित dstb = (DST_CTYPE *) dstb_raw;                                          \
    ध्रुव संरचना htp_ops_context * octx = ictx->octx;                                            \
    ध्रुव int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    ध्रुव int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    ध्रुव int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    ध्रुव uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    ध्रुव uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    ध्रुव uint32_t Wb = ictx->pe_wb;                                                             \
    ध्रुव uint32_t patch_stride = ictx->patch_stride;                                            \
    ध्रुव int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    प्रति (uint32_t j = 0; j < nb; j++) {                                                          \
        ध्रुव uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        ध्रुव int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        प्रति (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            ध्रुव int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            ध्रुव पूर्णाङ्क     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            प्रति (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * बहिर्गमनम्_सततम् = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                यदि (!okh) {                                                                      \
                    SPLAT_FN(बहिर्गमनम्_सततम्, 0, KW);                                                    \
                    अनुवर्त;                                                                    \
                }                                                                                \
                ध्रुव भिन्न * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                यदि (d0 == 1) {                                                                   \
                    ध्रुव int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    यदि (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    यदि (hi <= lo) {                                                              \
                        SPLAT_FN(बहिर्गमनम्_सततम्, 0, KW);                                                \
                    } अन्यथा {                                                                     \
                        यदि (lo > 0) {                                                            \
                            SPLAT_FN(बहिर्गमनम्_सततम्, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (बहिर्गमनम्_सततम् + lo),                                      \
                                (ध्रुव uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        यदि (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(बहिर्गमनम्_सततम् + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    अनुवर्त;                                                                    \
                }                                                                                \
                प्रति (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    ध्रुव uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    प्रति (uint32_t c = 0; c < chunk; c++) {                                       \
                        ध्रुव int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (बहिर्गमनम्_सततम् + ikw), (ध्रुव uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, भिन्न,    hvx_copy_f32_uu,     hvx_splat_f32_u)

स्थिर शून्य im2col_blocked_dma_thread(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना htp_im2col_context * ictx      = (संरचना htp_im2col_context *) दत्त;
    संरचना htp_ops_context *    octx      = ictx->octx;
    संरचना htp_thread_trace * सीमित tr = &octx->ctx->trace[ith];
    ध्रुव uint32_t OW                     = ictx->OW;
    ध्रुव uint32_t KH                     = ictx->KH;
    ध्रुव uint32_t owb                    = ictx->pe_owb;
    ध्रुव uint32_t nbpr                   = ictx->pe_nbpr;
    ध्रुव uint32_t patch_stride           = ictx->patch_stride;
    ध्रुव uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    ध्रुव dma_addr_t dst_दत्त             = octx->dst->दत्त;
    dma_queue *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_आधारः              = ictx->pe_vtcm_src + ith * ictx->pe_src_आकार_प्रत्येकम्_thread;
    uint8_t *      dstb_आधारः              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_आकार_प्रत्येकम्_thread;
    भिन्न *        srcb2[2]               = { (भिन्न *) srcb_आधारः, (भिन्न *) (srcb_आधारः + ictx->pe_src_row_अष्टकानि) };
    शून्य *         dstb2[2]               = { dstb_आधारः, dstb_आधारः + ictx->pe_dst_row_अष्टकानि };
    भिन्न *        scratch                = (भिन्न *) (srcb_आधारः + 2 * ictx->pe_src_row_अष्टकानि);
    ध्रुव uint32_t row_अन्तः_max            = ictx->pe_row_आधारः + ictx->pe_nrows;
    ध्रुव uint32_t row_start              = ictx->pe_row_आधारः + ictx->pe_पङ्क्तिकाः_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t row_अन्तः                = MIN(row_start + ictx->pe_पङ्क्तिकाः_प्रत्येकम्_thread, row_अन्तः_max);
    यदि (row_start >= row_अन्तः) {
        प्रत्यागम;
    }
    ध्रुव uint32_t nrows_local  = row_अन्तः - row_start;
    ध्रुव uint32_t योग_खण्डानि = nrows_local * nbpr;
    यदि (योग_खण्डानि == 0) {
        प्रत्यागम;
    }
    संरचना im2col_cursor cur = {
        .r   = row_start,
        .अन्तर्  = ictx->thread_अन्तर्[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    संरचना im2col_cursor stage = cur;
    प्रति (uint32_t i = 0; i < योग_खण्डानि && i < 2; i++) {
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_दत्त, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_चरणः_खण्डम्(&stage, ictx);
    }
    प्रति (uint32_t bi = 0; bi < योग_खण्डानि; bi++) {
        ध्रुव uint32_t buf = bi & 1u;
        भिन्न * srcb = srcb2[buf];
        शून्य *  dstb = dstb2[buf];
        ध्रुव uint32_t nb = MIN(owb, OW - cur.c0);
        dma_queue_निष्कासय(dma_q);
        प्रति (uint32_t k = 0; k < KH; k++) {
            dma_queue_निष्कासय(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ध्रुव dma_addr_t ddr = dst_दत्त + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        यदि (bi + 2 < योग_खण्डानि) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_चरणः_खण्डम्(&stage, ictx);
        }
        im2col_चरणः_खण्डम्(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

स्थिर शून्य im2col_patchembed_dma_stage(संरचना htp_im2col_context * ictx, dma_queue * dma_q,
                                        भिन्न * srcb, ध्रुव संरचना im2col_cursor * c) {
    ध्रुव संरचना htp_ops_context * octx = ictx->octx;
    ध्रुव uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ध्रुव uint32_t KH = ictx->KH;
    ध्रुव uint32_t अन्तर् = c->अन्तर्, ioh = c->ioh;
    ध्रुव size_t src_channel_stride  = (size_t) IH * IW * परिमाण(भिन्न);
    ध्रुव size_t vtcm_channel_stride = (size_t) KH * IW * परिमाण(भिन्न);
    ध्रुव dma_addr_t src_दत्त        = octx->src[1]->दत्त;
    ध्रुव dma_addr_t vsrc            = src_दत्त + (size_t) (((अन्तर् * IC) * IH + ioh * KH) * IW) * परिमाण(भिन्न);

    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

स्थिर शून्य im2col_patchembed_compute_f16(
    ध्रुव संरचना htp_im2col_context * ictx,
    शून्य * सीमित dstb_raw, ध्रुव भिन्न * सीमित srcb, uint32_t ioh) {
    uint16_t * सीमित dstb = (uint16_t *) dstb_raw;
    ध्रुव uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ध्रुव uint32_t KH = ictx->KH, KW = ictx->KW;
    ध्रुव uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    प्रति (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        प्रति (uint32_t ikh = 0; ikh < KH; ikh++) {
            ध्रुव int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            प्रति (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * बहिर्गमनम्_सततम् = dst_patch + (size_t) (iic * KH + ikh) * KW;
                यदि (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(बहिर्गमनम्_सततम्, 0, KW);
                    अनुवर्त;
                }
                ध्रुव भिन्न * src_सततम् = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) बहिर्गमनम्_सततम्, (ध्रुव uint8_t *) src_सततम्, KW);
            }
        }
    }
}

स्थिर शून्य im2col_patchembed_compute_f32(
    ध्रुव संरचना htp_im2col_context * ictx,
    शून्य * सीमित dstb_raw, ध्रुव भिन्न * सीमित srcb, uint32_t ioh) {
    भिन्न * सीमित dstb = (भिन्न *) dstb_raw;
    ध्रुव uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ध्रुव uint32_t KH = ictx->KH, KW = ictx->KW;
    ध्रुव uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    प्रति (uint32_t iow = 0; iow < OW; iow++) {
        भिन्न * dst_patch = dstb + (size_t) iow * patch_stride;
        प्रति (uint32_t ikh = 0; ikh < KH; ikh++) {
            ध्रुव int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            प्रति (uint32_t iic = 0; iic < IC; iic++) {
                भिन्न * बहिर्गमनम्_सततम् = dst_patch + (size_t) (iic * KH + ikh) * KW;
                यदि (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(बहिर्गमनम्_सततम्, 0, KW);
                    अनुवर्त;
                }
                ध्रुव भिन्न * src_सततम् = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) बहिर्गमनम्_सततम्, (ध्रुव uint8_t *) src_सततम्, KW);
            }
        }
    }
}

स्थिर शून्य im2col_patchembed_dma_thread(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना htp_im2col_context * ictx        = (संरचना htp_im2col_context *) दत्त;
    संरचना htp_ops_context *    octx        = ictx->octx;
    संरचना htp_thread_trace * सीमित tr   = &octx->ctx->trace[ith];
    ध्रुव uint32_t OW                       = ictx->OW;
    ध्रुव uint32_t patch_stride             = ictx->patch_stride;
    ध्रुव uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    ध्रुव dma_addr_t dst_दत्त               = octx->dst->दत्त;
    dma_queue *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_आधारः                = ictx->pe_vtcm_src + ith * ictx->pe_src_आकार_प्रत्येकम्_thread;
    uint8_t *      dstb_आधारः                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_आकार_प्रत्येकम्_thread;
    भिन्न *        srcb2[2]                 = { (भिन्न *) srcb_आधारः, (भिन्न *) (srcb_आधारः + ictx->pe_src_row_अष्टकानि) };
    शून्य *         dstb2[2]                 = { dstb_आधारः, dstb_आधारः + ictx->pe_dst_row_अष्टकानि };
    ध्रुव uint32_t row_अन्तः_max              = ictx->pe_row_आधारः + ictx->pe_nrows;
    ध्रुव uint32_t row_start                = ictx->pe_row_आधारः + ictx->pe_पङ्क्तिकाः_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t row_अन्तः                  = MIN(row_start + ictx->pe_पङ्क्तिकाः_प्रत्येकम्_thread, row_अन्तः_max);
    यदि (row_start >= row_अन्तः) {
        प्रत्यागम;
    }
    ध्रुव uint32_t योग_पङ्क्तिकाः = row_अन्तः - row_start;
    यदि (योग_पङ्क्तिकाः == 0) {
        प्रत्यागम;
    }
    संरचना im2col_cursor cur = {
        .r   = row_start,
        .अन्तर्  = ictx->thread_अन्तर्[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    संरचना im2col_cursor stage = cur;
    प्रति (uint32_t i = 0; i < योग_पङ्क्तिकाः && i < 2; i++) {
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_दत्त, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_चरणः_row(&stage, ictx);
    }
    प्रति (uint32_t ri = 0; ri < योग_पङ्क्तिकाः; ri++) {
        ध्रुव uint32_t buf = ri & 1u;
        भिन्न * srcb = srcb2[buf];
        शून्य *  dstb = dstb2[buf];
        dma_queue_निष्कासय(dma_q);
        dma_queue_निष्कासय(dma_q);
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ध्रुव dma_addr_t ddr_row = dst_दत्त + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        यदि (ri + 2 < योग_पङ्क्तिकाः) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_चरणः_row(&stage, ictx);
        }
        im2col_चरणः_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

स्थिर अन्तःस्थ शून्य im2col_3d_चरणः_row(संरचना im2col_cursor * c, ध्रुव संरचना htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    यदि (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        यदि (c->iod >= ictx->OD) {
            c->iod = 0;
            c->अन्तर्++;
        }
    }
}

स्थिर अन्तःस्थ शून्य im2col_3d_चरणः_खण्डम्(संरचना im2col_cursor * c, ध्रुव संरचना htp_im2col_context * ictx) {
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
                c->अन्तर्++;
            }
        }
    }
}

स्थिर शून्य im2col_3d_blocked_dma_stage(संरचना htp_im2col_context * ictx, dma_queue * dma_q,
                                        भिन्न * srcb, ध्रुव संरचना im2col_cursor * c) {
    ध्रुव uint32_t अन्तर् = c->अन्तर्, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    ध्रुव संरचना htp_ops_context * octx = ictx->octx;
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    ध्रुव int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    ध्रुव int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    ध्रुव uint32_t IC = ictx->IC;
    ध्रुव uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ध्रुव uint32_t KD = ictx->KD, KH = ictx->KH;
    ध्रुव uint32_t Wb = ictx->pe_wb;
    ध्रुव int32_t win0 = (int32_t) c0 * s0 - p0;
    ध्रुव dma_addr_t src_दत्त = src1->दत्त;
    ध्रुव size_t src_channel_stride = src1->nb[3];
    ध्रुव size_t vtcm_channel_stride = (size_t) KD * KH * Wb * परिमाण(भिन्न);
    ध्रुव size_t batch_विस्थापनम्        = (size_t) (अन्तर् * IC) * src1->nb[3];

    प्रति (uint32_t ikd = 0; ikd < KD; ikd++) {
        ध्रुव int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        ध्रुव bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        प्रति (uint32_t ikh = 0; ikh < KH; ikh++) {
            ध्रुव int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            यदि (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(srcb, 0), 0, 0, 0, 0);
                अनुवर्त;
            }
            ध्रुव int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            यदि (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            यदि (hi <= lo) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(srcb, 0), 0, 0, 0, 0);
                अनुवर्त;
            }
            भिन्न * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            ध्रुव dma_addr_t vsrc = src_दत्त + batch_विस्थापनम् +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * परिमाण(भिन्न), IC);
        }
    }
}

#परिभाषय IM2COL_3D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
स्थिर शून्य FNAME(ध्रुव संरचना htp_im2col_context * ictx,                                            \
                  शून्य * सीमित dstb_raw, ध्रुव भिन्न * सीमित srcb,                             \
                  भिन्न * सीमित scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * सीमित dstb = (DST_CTYPE *) dstb_raw;                                              \
    ध्रुव संरचना htp_ops_context * octx = ictx->octx;                                                \
    ध्रुव int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    ध्रुव int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    ध्रुव int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    ध्रुव uint32_t IC = ictx->IC;                                                                    \
    ध्रुव uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    ध्रुव uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    ध्रुव uint32_t patch_stride = ictx->patch_stride;                                                \
    ध्रुव uint32_t Wb = ictx->pe_wb;                                                                 \
    ध्रुव int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    प्रति (uint32_t j = 0; j < nb; j++) {                                                              \
        ध्रुव uint32_t iow = c0 + j;                                                                 \
        ध्रुव int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        प्रति (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            प्रति (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                ध्रुव int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                प्रति (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    ध्रुव int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * बहिर्गमनम्_सततम् = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    यदि (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(बहिर्गमनम्_सततम्, 0, KW);                                                    \
                        अनुवर्त;                                                                    \
                    }                                                                                \
                    ध्रुव भिन्न * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    यदि (d0 == 1) {                                                                   \
                        ध्रुव int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        यदि (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        यदि (hi <= lo) {                                                              \
                            SPLAT_FN(बहिर्गमनम्_सततम्, 0, KW);                                                \
                        } अन्यथा {                                                                     \
                            यदि (lo > 0) {                                                            \
                                SPLAT_FN(बहिर्गमनम्_सततम्, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (बहिर्गमनम्_सततम् + lo),                                      \
                                    (ध्रुव uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            यदि (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(बहिर्गमनम्_सततम् + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        अनुवर्त;                                                                    \
                    }                                                                                \
                    प्रति (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        ध्रुव uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        प्रति (uint32_t c = 0; c < chunk; c++) {                                       \
                            ध्रुव int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (बहिर्गमनम्_सततम् + ikw), (ध्रुव uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, भिन्न,    hvx_copy_f32_uu,     hvx_splat_f32_u)

स्थिर शून्य im2col_3d_blocked_dma_thread(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना htp_im2col_context * ictx        = (संरचना htp_im2col_context *) दत्त;
    संरचना htp_ops_context *    octx        = ictx->octx;
    संरचना htp_thread_trace * सीमित tr   = &octx->ctx->trace[ith];
    ध्रुव संरचना htp_tensor * सीमित dst  = octx->dst;
    ध्रुव uint32_t OW           = ictx->OW;
    ध्रुव uint32_t KD = ictx->KD, KH = ictx->KH;
    ध्रुव uint32_t kd_kh        = KD * KH;
    ध्रुव uint32_t patch_stride = ictx->patch_stride;
    ध्रुव uint32_t owb          = ictx->pe_owb;
    ध्रुव uint32_t nbpr         = ictx->pe_nbpr;
    ध्रुव uint32_t dst_elem     = ictx->dst_elem;
    ध्रुव dma_addr_t dst_दत्त   = dst->दत्त;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_आधारः         = ictx->pe_vtcm_src + ith * ictx->pe_src_आकार_प्रत्येकम्_thread;
    uint8_t * dstb_आधारः         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_आकार_प्रत्येकम्_thread;
    भिन्न * srcb2[2]            = { (भिन्न *) srcb_आधारः, (भिन्न *) (srcb_आधारः + ictx->pe_src_row_अष्टकानि) };
    शून्य * dstb2[2]             = { dstb_आधारः, dstb_आधारः + ictx->pe_dst_row_अष्टकानि };
    भिन्न * scratch             = (भिन्न *) (srcb_आधारः + 2 * ictx->pe_src_row_अष्टकानि);
    ध्रुव uint32_t row_अन्तः_max  = ictx->pe_row_आधारः + ictx->pe_nrows;
    ध्रुव uint32_t row_start    = ictx->pe_row_आधारः + ictx->pe_पङ्क्तिकाः_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t row_अन्तः      = MIN(row_start + ictx->pe_पङ्क्तिकाः_प्रत्येकम्_thread, row_अन्तः_max);
    यदि (row_start >= row_अन्तः) {
        प्रत्यागम;
    }
    ध्रुव uint32_t nrows_local  = row_अन्तः - row_start;
    ध्रुव uint32_t योग_खण्डानि = nrows_local * nbpr;
    यदि (योग_खण्डानि == 0) {
        प्रत्यागम;
    }
    संरचना im2col_cursor cur = {
        .r   = row_start,
        .अन्तर्  = ictx->thread_अन्तर्[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    संरचना im2col_cursor stage = cur;
    प्रति (uint32_t i = 0; i < योग_खण्डानि && i < 2; i++) {
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_दत्त, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_चरणः_खण्डम्(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    प्रति (uint32_t bi = 0; bi < योग_खण्डानि; bi++) {
        ध्रुव uint32_t buf = bi & 1u;
        भिन्न * srcb = srcb2[buf];
        शून्य *  dstb = dstb2[buf];
        ध्रुव uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_queue_निष्कासय(dma_q);
        प्रति (uint32_t p = 0; p < kd_kh; p++) {
            dma_queue_निष्कासय(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ध्रुव dma_addr_t ddr = dst_दत्त + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        यदि (bi + 2 < योग_खण्डानि) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_चरणः_खण्डम्(&stage, ictx);
        }
        im2col_3d_चरणः_खण्डम्(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}
स्थिर शून्य im2col_3d_patchembed_dma_stage(संरचना htp_im2col_context * ictx, dma_queue * dma_q,
                                           भिन्न * srcb, ध्रुव संरचना im2col_cursor * c) {
    ध्रुव संरचना htp_ops_context * octx = ictx->octx;
    ध्रुव uint32_t IC = ictx->IC;
    ध्रुव uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ध्रुव uint32_t KD = ictx->KD, KH = ictx->KH;
    ध्रुव dma_addr_t src_दत्त = octx->src[1]->दत्त;
    ध्रुव uint32_t अन्तर् = c->अन्तर्, iod = c->iod, ioh = c->ioh;
    ध्रुव size_t src_channel_stride  = (size_t) ID * IH * IW * परिमाण(भिन्न);
    ध्रुव size_t vtcm_channel_stride = (size_t) KD * KH * IW * परिमाण(भिन्न);
    ध्रुव size_t row_आकार            = (size_t) KH * IW * परिमाण(भिन्न);

    प्रति (uint32_t ikd = 0; ikd < KD; ikd++) {
        ध्रुव uint32_t iid = iod * KD + ikd;
        भिन्न * vdst = srcb + (size_t) ikd * KH * IW;
        ध्रुव dma_addr_t vsrc = src_दत्त + (size_t) (((अन्तर् * IC) * ID + iid) * IH + ioh * KH) * IW * परिमाण(भिन्न);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, row_आकार, IC);
    }
}
स्थिर शून्य im2col_3d_patchembed_compute_f16(
    ध्रुव संरचना htp_im2col_context * ictx,
    शून्य * सीमित dstb_raw, ध्रुव भिन्न * सीमित srcb) {
    uint16_t * सीमित dstb = (uint16_t *) dstb_raw;
    ध्रुव uint32_t OW           = ictx->OW;
    ध्रुव uint32_t IC           = ictx->IC;
    ध्रुव uint32_t KD           = ictx->KD;
    ध्रुव uint32_t KH           = ictx->KH;
    ध्रुव uint32_t KW           = ictx->KW;
    ध्रुव uint32_t IW           = ictx->IW;
    ध्रुव uint32_t patch_stride = ictx->patch_stride;

    प्रति (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        प्रति (uint32_t iic = 0; iic < IC; iic++) {
            प्रति (uint32_t ikd = 0; ikd < KD; ikd++) {
                प्रति (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * बहिर्गमनम्_सततम् = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ध्रुव भिन्न * src_सततम् = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) बहिर्गमनम्_सततम्, (ध्रुव uint8_t *) src_सततम्, KW);
                }
            }
        }
    }
}

स्थिर शून्य im2col_3d_patchembed_compute_f32(
    ध्रुव संरचना htp_im2col_context * ictx,
    शून्य * सीमित dstb_raw, ध्रुव भिन्न * सीमित srcb) {
    भिन्न * सीमित dstb = (भिन्न *) dstb_raw;
    ध्रुव uint32_t OW           = ictx->OW;
    ध्रुव uint32_t IC           = ictx->IC;
    ध्रुव uint32_t KD           = ictx->KD;
    ध्रुव uint32_t KH           = ictx->KH;
    ध्रुव uint32_t KW           = ictx->KW;
    ध्रुव uint32_t IW           = ictx->IW;
    ध्रुव uint32_t patch_stride = ictx->patch_stride;

    प्रति (uint32_t iow = 0; iow < OW; iow++) {
        भिन्न * dst_patch = dstb + (size_t) iow * patch_stride;
        प्रति (uint32_t iic = 0; iic < IC; iic++) {
            प्रति (uint32_t ikd = 0; ikd < KD; ikd++) {
                प्रति (uint32_t ikh = 0; ikh < KH; ikh++) {
                    भिन्न * बहिर्गमनम्_सततम् = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ध्रुव भिन्न * src_सततम् = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) बहिर्गमनम्_सततम्, (ध्रुव uint8_t *) src_सततम्, KW);
                }
            }
        }
    }
}

स्थिर शून्य im2col_3d_patchembed_dma_thread(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना htp_im2col_context * ictx        = (संरचना htp_im2col_context *) दत्त;
    संरचना htp_ops_context *    octx        = ictx->octx;
    संरचना htp_thread_trace * सीमित tr   = &octx->ctx->trace[ith];
    ध्रुव संरचना htp_tensor * सीमित dst  = octx->dst;
    ध्रुव uint32_t KD           = ictx->KD;
    ध्रुव uint32_t OW           = ictx->OW;
    ध्रुव uint32_t patch_stride = ictx->patch_stride;
    ध्रुव uint32_t dst_elem     = ictx->dst_elem;
    ध्रुव dma_addr_t dst_दत्त   = dst->दत्त;
    dma_queue * dma_q           = octx->ctx->dma[ith];
    ध्रुव uint32_t row_अन्तः_max  = ictx->pe_row_आधारः + ictx->pe_nrows;
    ध्रुव uint32_t row_start    = ictx->pe_row_आधारः + ictx->pe_पङ्क्तिकाः_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t row_अन्तः      = MIN(row_start + ictx->pe_पङ्क्तिकाः_प्रत्येकम्_thread, row_अन्तः_max);
    यदि (row_start >= row_अन्तः) {
        प्रत्यागम;
    }

    ध्रुव uint32_t nrows_local  = row_अन्तः - row_start;
    uint8_t * src_आधारः          = ictx->pe_vtcm_src + ith * ictx->pe_src_आकार_प्रत्येकम्_thread;
    uint8_t * dstb_आधारः         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_आकार_प्रत्येकम्_thread;
    भिन्न * srcb2[2]            = { (भिन्न *) src_आधारः, (भिन्न *) (src_आधारः + ictx->pe_src_row_अष्टकानि) };
    शून्य * dstb2[2]             = { dstb_आधारः, dstb_आधारः + ictx->pe_dst_row_अष्टकानि };
    संरचना im2col_cursor cur = {
        .r   = row_start,
        .अन्तर्  = ictx->thread_अन्तर्[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    संरचना im2col_cursor stage = cur;
    प्रति (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_दत्त, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_चरणः_row(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    प्रति (uint32_t idx = 0; idx < nrows_local; idx++) {
        ध्रुव uint32_t buf = idx & 1u;
        भिन्न * srcb = srcb2[buf];
        शून्य *  dstb = dstb2[buf];
        dma_queue_निष्कासय(dma_q);
        प्रति (uint32_t p = 0; p < KD; p++) {
            dma_queue_निष्कासय(dma_q);
        }
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ध्रुव dma_addr_t ddr_row = dst_दत्त + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(ddr_row, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        यदि (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_चरणः_row(&stage, ictx);
        }
        im2col_3d_चरणः_row(&cur, ictx);
    }
    dma_queue_flush(dma_q);
}

पूर्णाङ्क op_im2col(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_im2col_kernel_params * kparams = (ध्रुव संरचना htp_im2col_kernel_params *) octx->kernel_params;
    यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    ध्रुव संरचना htp_tensor * dst = octx->dst;

    uint32_t row_आधारः = 0;
    uint32_t nrows    = kparams->योग_पङ्क्तिकाः;
    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव uint32_t row_आकार = dst->nb[2];
        ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = (row_आकार > 0) ? (HEX_L2_पङ्क्तिः_आकार / hex_gcd_u32(row_आकार, HEX_L2_पङ्क्तिः_आकार)) : 1;
        ध्रुव संरचना htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->योग_पङ्क्तिकाः, htp_tensor_mdev_दत्त_aligned(dst) ? पङ्क्तिकाः_प्रत्येकम्_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_आधारः = range.start;
        nrows    = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    संरचना htp_im2col_context ictx = { 0 };
    uint8_t * ध्रुव आधारः = octx->ctx->vtcm_आधारः;

    ध्रुव uint32_t pth = MIN(octx->n_तन्तवः, nrows);
    ictx.octx                   = octx;
    ictx.pe_row_आधारः            = row_आधारः;
    ictx.pe_nrows               = nrows;
    ictx.pe_पङ्क्तिकाः_प्रत्येकम्_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_row_अष्टकानि       = kparams->src_row_अष्टकानि;
    ictx.pe_dst_row_अष्टकानि       = kparams->dst_row_अष्टकानि;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, आधारः, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, आधारः, kparams->off_dst);
    ictx.pe_src_आकार_प्रत्येकम्_thread = kparams->src_अष्टकानि_प्रत्येकम्_thread;
    ictx.pe_dst_आकार_प्रत्येकम्_thread = kparams->dst_अष्टकानि_प्रत्येकम्_thread;
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

    प्रति (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = row_आधारः + ictx.pe_पङ्क्तिकाः_प्रत्येकम्_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_अन्तर्[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? परिमाण(__fp16) : परिमाण(भिन्न);

    यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (शून्य *) im2col_patchembed_compute_f16
                                                 : (शून्य *) im2col_patchembed_compute_f32;
        work_queue_सततम्(octx->ctx->work_queue, im2col_patchembed_dma_thread, &ictx, pth);
        प्रत्यागम HTP_स्थितिः_OK;
    } अन्यथा यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (शून्य *) im2col_blocked_compute_f16
                                                 : (शून्य *) im2col_blocked_compute_f32;
        work_queue_सततम्(octx->ctx->work_queue, im2col_blocked_dma_thread, &ictx, pth);
        प्रत्यागम HTP_स्थितिः_OK;
    }

    प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
}

पूर्णाङ्क op_im2col_3d(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_im2col_kernel_params * kparams =
        (ध्रुव संरचना htp_im2col_kernel_params *) octx->kernel_params;
    यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_UNSUPPORTED) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    ध्रुव संरचना htp_tensor * dst = octx->dst;

    uint32_t row_आधारः = 0;
    uint32_t nrows    = kparams->योग_पङ्क्तिकाः;
    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव uint32_t row_आकार = dst->nb[2];
        ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = (row_आकार > 0) ? (HEX_L2_पङ्क्तिः_आकार / hex_gcd_u32(row_आकार, HEX_L2_पङ्क्तिः_आकार)) : 1;
        ध्रुव संरचना htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(kparams->योग_पङ्क्तिकाः, htp_tensor_mdev_दत्त_aligned(dst) ? पङ्क्तिकाः_प्रत्येकम्_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_आधारः = range.start;
        nrows    = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    संरचना htp_im2col_context ictx = { 0 };
    uint8_t * ध्रुव आधारः = octx->ctx->vtcm_आधारः;

    ध्रुव uint32_t pth = MIN(octx->n_तन्तवः, nrows);
    ictx.octx                   = octx;
    ictx.pe_row_आधारः            = row_आधारः;
    ictx.pe_nrows               = nrows;
    ictx.pe_पङ्क्तिकाः_प्रत्येकम्_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_row_अष्टकानि       = kparams->src_row_अष्टकानि;
    ictx.pe_dst_row_अष्टकानि       = kparams->dst_row_अष्टकानि;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, आधारः, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, आधारः, kparams->off_dst);
    ictx.pe_src_आकार_प्रत्येकम्_thread = kparams->src_अष्टकानि_प्रत्येकम्_thread;
    ictx.pe_dst_आकार_प्रत्येकम्_thread = kparams->dst_अष्टकानि_प्रत्येकम्_thread;
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

    प्रति (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = row_आधारः + ictx.pe_पङ्क्तिकाः_प्रत्येकम्_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_अन्तर्[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? परिमाण(__fp16) : परिमाण(भिन्न);

    यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (शून्य *) im2col_3d_patchembed_compute_f16
                                                 : (शून्य *) im2col_3d_patchembed_compute_f32;
        work_queue_सततम्(octx->ctx->work_queue, im2col_3d_patchembed_dma_thread, &ictx, pth);
        प्रत्यागम HTP_स्थितिः_OK;
    } अन्यथा यदि (kparams->kernel_type == HTP_IM2COL_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (शून्य *) im2col_3d_blocked_compute_f16
                                                 : (शून्य *) im2col_3d_blocked_compute_f32;
        work_queue_सततम्(octx->ctx->work_queue, im2col_3d_blocked_dma_thread, &ictx, pth);
        प्रत्यागम HTP_स्थितिः_OK;
    }

    प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
}
