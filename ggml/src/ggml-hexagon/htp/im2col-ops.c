#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <hexagon_protos.h>
#ಅನ್ತರ್ಭಾವ <hexagon_types.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "im2col-ops.h"

ರಚನೆ htp_im2col_context {
    ರಚನೆ htp_ops_context * octx;

    uint32_t pe_ಅಡ್ಡಸಾಲು_ತಳ;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;                   // output rows per worker
    uint32_t pe_src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;                     // one output row's source, rounded 256
    uint32_t pe_dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_ಗಾತ್ರ_ಪ್ರತಿ_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;              // 2 * pe_dst_row_bytes

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
    uint32_t thread_ಒಳ[HTP_MAX_NTHREADS];
    uint32_t thread_iod[HTP_MAX_NTHREADS];
    uint32_t thread_ioh[HTP_MAX_NTHREADS];

    ಶೂನ್ಯ * compute;
    uint32_t dst_elem;
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*im2col_blocked_compute_fn_t)(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,
    ಶೂನ್ಯ * ಸೀಮಿತ dstb, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb,
    ದಶಮಾಂಶ * ಸೀಮಿತ scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*im2col_patchembed_compute_fn_t)(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,
    ಶೂನ್ಯ * ಸೀಮಿತ dstb, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb, uint32_t ioh);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*im2col_3d_blocked_compute_fn_t)(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,
    ಶೂನ್ಯ * ಸೀಮಿತ dstb, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb,
    ದಶಮಾಂಶ * ಸೀಮಿತ scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*im2col_3d_patchembed_compute_fn_t)(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,
    ಶೂನ್ಯ * ಸೀಮಿತ dstb, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb);

ರಚನೆ im2col_cursor {
    uint32_t r;
    uint32_t ಒಳ;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ im2col_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು(ರಚನೆ im2col_cursor * c, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ಆದರೆ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->ಒಳ++;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ im2col_ಹೆಜ್ಜೆ_ಖಂಡ(ರಚನೆ im2col_cursor * c, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ಆದರೆ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ಆದರೆ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->ಒಳ++;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_blocked_dma_stage(ರಚನೆ htp_im2col_context * ictx, dma_ಸರತಿ * dma_q,
                                     ದಶಮಾಂಶ * srcb, ಸ್ಥಿರಾಂಕ ರಚನೆ im2col_cursor * c) {
    ಸ್ಥಿರಾಂಕ uint32_t ಒಳ = c->ಒಳ, ioh = c->ioh, c0 = c->c0;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx = ictx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    ಸ್ಥಿರಾಂಕ int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    ಸ್ಥಿರಾಂಕ int32_t d1 = octx->op_params[5];
    ಸ್ಥಿರಾಂಕ bool is_2D = (octx->op_params[6] == 1);
    ಸ್ಥಿರಾಂಕ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ಸ್ಥಿರಾಂಕ uint32_t KH = ictx->KH;
    ಸ್ಥಿರಾಂಕ uint32_t Wb = ictx->pe_wb;
    ಸ್ಥಿರಾಂಕ dma_addr_t src_ದತ್ತಾಂಶ = src1->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int32_t win0 = (int32_t) c0 * s0 - p0;
    ಸ್ಥಿರಾಂಕ size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    ಸ್ಥಿರಾಂಕ size_t batch_ಸ್ಥಾನಾಂತರ        = is_2D ? (size_t) ಒಳ * src1->nb[3] : (size_t) ಒಳ * src1->nb[2];

    ಗಾಗಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
        ಸ್ಥಿರಾಂಕ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        ಆದರೆ (iih < 0 || iih >= (int32_t) IH) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(srcb, 0), 0, 0, 0, 0);
            ಮುಂದುವರಿ;
        }
        ಸ್ಥಿರಾಂಕ int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        ಆದರೆ (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        ಆದರೆ (hi <= lo) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(srcb, 0), 0, 0, 0, 0);
            ಮುಂದುವರಿ;
        }
        ಸ್ಥಿರಾಂಕ uint32_t cpw = (uint32_t) (hi - lo);
        ದಶಮಾಂಶ * vdst = srcb + (size_t) ikh * Wb + lo;
        ಸ್ಥಿರಾಂಕ dma_addr_t vsrc = src_ದತ್ತಾಂಶ + batch_ಸ್ಥಾನಾಂತರ + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vdst, vsrc),
                       (size_t) KH * Wb * ಗಾತ್ರ(ದಶಮಾಂಶ), src_channel_stride,
                       cpw * ಗಾತ್ರ(ದಶಮಾಂಶ), IC);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು IM2_ಲಂಬಸಾಲು_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
ಸ್ಥಿರ ಶೂನ್ಯ FNAME(                                                                               \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,                                                      \
    ಶೂನ್ಯ * ಸೀಮಿತ dstb_raw, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb,                                       \
    ದಶಮಾಂಶ * ಸೀಮಿತ scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * ಸೀಮಿತ dstb = (DST_CTYPE *) dstb_raw;                                          \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx = ictx->octx;                                            \
    ಸ್ಥಿರಾಂಕ int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    ಸ್ಥಿರಾಂಕ int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    ಸ್ಥಿರಾಂಕ int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    ಸ್ಥಿರಾಂಕ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    ಸ್ಥಿರಾಂಕ uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    ಸ್ಥಿರಾಂಕ uint32_t Wb = ictx->pe_wb;                                                             \
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride = ictx->patch_stride;                                            \
    ಸ್ಥಿರಾಂಕ int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    ಗಾಗಿ (uint32_t j = 0; j < nb; j++) {                                                          \
        ಸ್ಥಿರಾಂಕ uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        ಸ್ಥಿರಾಂಕ int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        ಗಾಗಿ (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            ಸ್ಥಿರಾಂಕ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            ಗಾಗಿ (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * ಹೊರಹೋಗು_ಸತತ = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                ಆದರೆ (!okh) {                                                                      \
                    SPLAT_FN(ಹೊರಹೋಗು_ಸತತ, 0, KW);                                                    \
                    ಮುಂದುವರಿ;                                                                    \
                }                                                                                \
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                ಆದರೆ (d0 == 1) {                                                                   \
                    ಸ್ಥಿರಾಂಕ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    ಆದರೆ (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    ಆದರೆ (hi <= lo) {                                                              \
                        SPLAT_FN(ಹೊರಹೋಗು_ಸತತ, 0, KW);                                                \
                    } ಇಲ್ಲದಿದ್ದರೆ {                                                                     \
                        ಆದರೆ (lo > 0) {                                                            \
                            SPLAT_FN(ಹೊರಹೋಗು_ಸತತ, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (ಹೊರಹೋಗು_ಸತತ + lo),                                      \
                                (ಸ್ಥಿರಾಂಕ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        ಆದರೆ (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(ಹೊರಹೋಗು_ಸತತ + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    ಮುಂದುವರಿ;                                                                    \
                }                                                                                \
                ಗಾಗಿ (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    ಸ್ಥಿರಾಂಕ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    ಗಾಗಿ (uint32_t c = 0; c < chunk; c++) {                                       \
                        ಸ್ಥಿರಾಂಕ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (ಹೊರಹೋಗು_ಸತತ + ikw), (ಸ್ಥಿರಾಂಕ uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2_ಲಂಬಸಾಲು_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_ಲಂಬಸಾಲು_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, ದಶಮಾಂಶ,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ಸ್ಥಿರ ಶೂನ್ಯ im2col_blocked_dma_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ htp_im2col_context * ictx      = (ರಚನೆ htp_im2col_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context *    octx      = ictx->octx;
    ರಚನೆ htp_thread_trace * ಸೀಮಿತ tr = &octx->ctx->trace[ith];
    ಸ್ಥಿರಾಂಕ uint32_t OW                     = ictx->OW;
    ಸ್ಥಿರಾಂಕ uint32_t KH                     = ictx->KH;
    ಸ್ಥಿರಾಂಕ uint32_t owb                    = ictx->pe_owb;
    ಸ್ಥಿರಾಂಕ uint32_t nbpr                   = ictx->pe_nbpr;
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride           = ictx->patch_stride;
    ಸ್ಥಿರಾಂಕ uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ             = octx->dst->ದತ್ತಾಂಶ;
    dma_ಸರತಿ *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_ತಳ              = ictx->pe_vtcm_src + ith * ictx->pe_src_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t *      dstb_ತಳ              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    ದಶಮಾಂಶ *        srcb2[2]               = { (ದಶಮಾಂಶ *) srcb_ತಳ, (ದಶಮಾಂಶ *) (srcb_ತಳ + ictx->pe_src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು) };
    ಶೂನ್ಯ *         dstb2[2]               = { dstb_ತಳ, dstb_ತಳ + ictx->pe_dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು };
    ದಶಮಾಂಶ *        scratch                = (ದಶಮಾಂಶ *) (srcb_ತಳ + 2 * ictx->pe_src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು);
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ_max            = ictx->pe_ಅಡ್ಡಸಾಲು_ತಳ + ictx->pe_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ              = ictx->pe_ಅಡ್ಡಸಾಲು_ತಳ + ictx->pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ                = MIN(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ictx->pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, ಅಡ್ಡಸಾಲು_ಕೊನೆ_max);
    ಆದರೆ (ಅಡ್ಡಸಾಲು_ಆರಂಭ >= ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ uint32_t nrows_local  = ಅಡ್ಡಸಾಲು_ಕೊನೆ - ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಖಂಡಗಳು = nrows_local * nbpr;
    ಆದರೆ (ಒಟ್ಟು_ಖಂಡಗಳು == 0) {
        ಹಿಂತಿರುಗಿ;
    }
    ರಚನೆ im2col_cursor cur = {
        .r   = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .ಒಳ  = ictx->thread_ಒಳ[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ರಚನೆ im2col_cursor stage = cur;
    ಗಾಗಿ (uint32_t i = 0; i < ಒಟ್ಟು_ಖಂಡಗಳು && i < 2; i++) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_ಹೆಜ್ಜೆ_ಖಂಡ(&stage, ictx);
    }
    ಗಾಗಿ (uint32_t bi = 0; bi < ಒಟ್ಟು_ಖಂಡಗಳು; bi++) {
        ಸ್ಥಿರಾಂಕ uint32_t buf = bi & 1u;
        ದಶಮಾಂಶ * srcb = srcb2[buf];
        ಶೂನ್ಯ *  dstb = dstb2[buf];
        ಸ್ಥಿರಾಂಕ uint32_t nb = MIN(owb, OW - cur.c0);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
        ಗಾಗಿ (uint32_t k = 0; k < KH; k++) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q);
        }
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ಸ್ಥಿರಾಂಕ dma_addr_t ddr = dst_ದತ್ತಾಂಶ + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ಆದರೆ (bi + 2 < ಒಟ್ಟು_ಖಂಡಗಳು) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_ಹೆಜ್ಜೆ_ಖಂಡ(&stage, ictx);
        }
        im2col_ಹೆಜ್ಜೆ_ಖಂಡ(&cur, ictx);
    }
    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_patchembed_dma_stage(ರಚನೆ htp_im2col_context * ictx, dma_ಸರತಿ * dma_q,
                                        ದಶಮಾಂಶ * srcb, ಸ್ಥಿರಾಂಕ ರಚನೆ im2col_cursor * c) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx = ictx->octx;
    ಸ್ಥಿರಾಂಕ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ಸ್ಥಿರಾಂಕ uint32_t KH = ictx->KH;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಳ = c->ಒಳ, ioh = c->ioh;
    ಸ್ಥಿರಾಂಕ size_t src_channel_stride  = (size_t) IH * IW * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t vtcm_channel_stride = (size_t) KH * IW * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ dma_addr_t src_ದತ್ತಾಂಶ        = octx->src[1]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ dma_addr_t vsrc            = src_ದತ್ತಾಂಶ + (size_t) (((ಒಳ * IC) * IH + ioh * KH) * IW) * ಗಾತ್ರ(ದಶಮಾಂಶ);

    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_patchembed_compute_f16(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,
    ಶೂನ್ಯ * ಸೀಮಿತ dstb_raw, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb, uint32_t ioh) {
    uint16_t * ಸೀಮಿತ dstb = (uint16_t *) dstb_raw;
    ಸ್ಥಿರಾಂಕ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ಸ್ಥಿರಾಂಕ uint32_t KH = ictx->KH, KW = ictx->KW;
    ಸ್ಥಿರಾಂಕ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    ಗಾಗಿ (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        ಗಾಗಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
            ಸ್ಥಿರಾಂಕ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            ಗಾಗಿ (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * ಹೊರಹೋಗು_ಸತತ = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ಆದರೆ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(ಹೊರಹೋಗು_ಸತತ, 0, KW);
                    ಮುಂದುವರಿ;
                }
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಸತತ = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) ಹೊರಹೋಗು_ಸತತ, (ಸ್ಥಿರಾಂಕ uint8_t *) src_ಸತತ, KW);
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_patchembed_compute_f32(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,
    ಶೂನ್ಯ * ಸೀಮಿತ dstb_raw, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb, uint32_t ioh) {
    ದಶಮಾಂಶ * ಸೀಮಿತ dstb = (ದಶಮಾಂಶ *) dstb_raw;
    ಸ್ಥಿರಾಂಕ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ಸ್ಥಿರಾಂಕ uint32_t KH = ictx->KH, KW = ictx->KW;
    ಸ್ಥಿರಾಂಕ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    ಗಾಗಿ (uint32_t iow = 0; iow < OW; iow++) {
        ದಶಮಾಂಶ * dst_patch = dstb + (size_t) iow * patch_stride;
        ಗಾಗಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
            ಸ್ಥಿರಾಂಕ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            ಗಾಗಿ (uint32_t iic = 0; iic < IC; iic++) {
                ದಶಮಾಂಶ * ಹೊರಹೋಗು_ಸತತ = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ಆದರೆ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(ಹೊರಹೋಗು_ಸತತ, 0, KW);
                    ಮುಂದುವರಿ;
                }
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಸತತ = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) ಹೊರಹೋಗು_ಸತತ, (ಸ್ಥಿರಾಂಕ uint8_t *) src_ಸತತ, KW);
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_patchembed_dma_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ htp_im2col_context * ictx        = (ರಚನೆ htp_im2col_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context *    octx        = ictx->octx;
    ರಚನೆ htp_thread_trace * ಸೀಮಿತ tr   = &octx->ctx->trace[ith];
    ಸ್ಥಿರಾಂಕ uint32_t OW                       = ictx->OW;
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride             = ictx->patch_stride;
    ಸ್ಥಿರಾಂಕ uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ               = octx->dst->ದತ್ತಾಂಶ;
    dma_ಸರತಿ *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_ತಳ                = ictx->pe_vtcm_src + ith * ictx->pe_src_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t *      dstb_ತಳ                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    ದಶಮಾಂಶ *        srcb2[2]                 = { (ದಶಮಾಂಶ *) srcb_ತಳ, (ದಶಮಾಂಶ *) (srcb_ತಳ + ictx->pe_src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು) };
    ಶೂನ್ಯ *         dstb2[2]                 = { dstb_ತಳ, dstb_ತಳ + ictx->pe_dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು };
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ_max              = ictx->pe_ಅಡ್ಡಸಾಲು_ತಳ + ictx->pe_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ                = ictx->pe_ಅಡ್ಡಸಾಲು_ತಳ + ictx->pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ                  = MIN(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ictx->pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, ಅಡ್ಡಸಾಲು_ಕೊನೆ_max);
    ಆದರೆ (ಅಡ್ಡಸಾಲು_ಆರಂಭ >= ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ಅಡ್ಡಸಾಲು_ಕೊನೆ - ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ಆದರೆ (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು == 0) {
        ಹಿಂತಿರುಗಿ;
    }
    ರಚನೆ im2col_cursor cur = {
        .r   = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .ಒಳ  = ictx->thread_ಒಳ[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ರಚನೆ im2col_cursor stage = cur;
    ಗಾಗಿ (uint32_t i = 0; i < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು && i < 2; i++) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು(&stage, ictx);
    }
    ಗಾಗಿ (uint32_t ri = 0; ri < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ri++) {
        ಸ್ಥಿರಾಂಕ uint32_t buf = ri & 1u;
        ದಶಮಾಂಶ * srcb = srcb2[buf];
        ಶೂನ್ಯ *  dstb = dstb2[buf];
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ಸ್ಥಿರಾಂಕ dma_addr_t ddr_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ddr_ಅಡ್ಡಸಾಲು, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ಆದರೆ (ri + 2 < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು(&stage, ictx);
        }
        im2col_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು(&cur, ictx);
    }
    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ im2col_3d_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು(ರಚನೆ im2col_cursor * c, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ಆದರೆ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        ಆದರೆ (c->iod >= ictx->OD) {
            c->iod = 0;
            c->ಒಳ++;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ im2col_3d_ಹೆಜ್ಜೆ_ಖಂಡ(ರಚನೆ im2col_cursor * c, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ಆದರೆ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ಆದರೆ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            ಆದರೆ (c->iod >= ictx->OD) {
                c->iod = 0;
                c->ಒಳ++;
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_blocked_dma_stage(ರಚನೆ htp_im2col_context * ictx, dma_ಸರತಿ * dma_q,
                                        ದಶಮಾಂಶ * srcb, ಸ್ಥಿರಾಂಕ ರಚನೆ im2col_cursor * c) {
    ಸ್ಥಿರಾಂಕ uint32_t ಒಳ = c->ಒಳ, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx = ictx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    ಸ್ಥಿರಾಂಕ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    ಸ್ಥಿರಾಂಕ int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    ಸ್ಥಿರಾಂಕ uint32_t IC = ictx->IC;
    ಸ್ಥಿರಾಂಕ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ಸ್ಥಿರಾಂಕ uint32_t KD = ictx->KD, KH = ictx->KH;
    ಸ್ಥಿರಾಂಕ uint32_t Wb = ictx->pe_wb;
    ಸ್ಥಿರಾಂಕ int32_t win0 = (int32_t) c0 * s0 - p0;
    ಸ್ಥಿರಾಂಕ dma_addr_t src_ದತ್ತಾಂಶ = src1->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ size_t src_channel_stride = src1->nb[3];
    ಸ್ಥಿರಾಂಕ size_t vtcm_channel_stride = (size_t) KD * KH * Wb * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t batch_ಸ್ಥಾನಾಂತರ        = (size_t) (ಒಳ * IC) * src1->nb[3];

    ಗಾಗಿ (uint32_t ikd = 0; ikd < KD; ikd++) {
        ಸ್ಥಿರಾಂಕ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        ಸ್ಥಿರಾಂಕ bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        ಗಾಗಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
            ಸ್ಥಿರಾಂಕ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            ಆದರೆ (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(srcb, 0), 0, 0, 0, 0);
                ಮುಂದುವರಿ;
            }
            ಸ್ಥಿರಾಂಕ int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            ಆದರೆ (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            ಆದರೆ (hi <= lo) {
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(srcb, 0), 0, 0, 0, 0);
                ಮುಂದುವರಿ;
            }
            ದಶಮಾಂಶ * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            ಸ್ಥಿರಾಂಕ dma_addr_t vsrc = src_ದತ್ತಾಂಶ + batch_ಸ್ಥಾನಾಂತರ +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * ಗಾತ್ರ(ದಶಮಾಂಶ), IC);
        }
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು IM2_ಲಂಬಸಾಲು_3_D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
ಸ್ಥಿರ ಶೂನ್ಯ FNAME(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,                                            \
                  ಶೂನ್ಯ * ಸೀಮಿತ dstb_raw, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb,                             \
                  ದಶಮಾಂಶ * ಸೀಮಿತ scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * ಸೀಮಿತ dstb = (DST_CTYPE *) dstb_raw;                                              \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx = ictx->octx;                                                \
    ಸ್ಥಿರಾಂಕ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    ಸ್ಥಿರಾಂಕ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    ಸ್ಥಿರಾಂಕ int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    ಸ್ಥಿರಾಂಕ uint32_t IC = ictx->IC;                                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    ಸ್ಥಿರಾಂಕ uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride = ictx->patch_stride;                                                \
    ಸ್ಥಿರಾಂಕ uint32_t Wb = ictx->pe_wb;                                                                 \
    ಸ್ಥಿರಾಂಕ int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    ಗಾಗಿ (uint32_t j = 0; j < nb; j++) {                                                              \
        ಸ್ಥಿರಾಂಕ uint32_t iow = c0 + j;                                                                 \
        ಸ್ಥಿರಾಂಕ int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        ಗಾಗಿ (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            ಗಾಗಿ (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                ಸ್ಥಿರಾಂಕ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                ಗಾಗಿ (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    ಸ್ಥಿರಾಂಕ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * ಹೊರಹೋಗು_ಸತತ = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    ಆದರೆ (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(ಹೊರಹೋಗು_ಸತತ, 0, KW);                                                    \
                        ಮುಂದುವರಿ;                                                                    \
                    }                                                                                \
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    ಆದರೆ (d0 == 1) {                                                                   \
                        ಸ್ಥಿರಾಂಕ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        ಆದರೆ (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        ಆದರೆ (hi <= lo) {                                                              \
                            SPLAT_FN(ಹೊರಹೋಗು_ಸತತ, 0, KW);                                                \
                        } ಇಲ್ಲದಿದ್ದರೆ {                                                                     \
                            ಆದರೆ (lo > 0) {                                                            \
                                SPLAT_FN(ಹೊರಹೋಗು_ಸತತ, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (ಹೊರಹೋಗು_ಸತತ + lo),                                      \
                                    (ಸ್ಥಿರಾಂಕ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            ಆದರೆ (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(ಹೊರಹೋಗು_ಸತತ + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        ಮುಂದುವರಿ;                                                                    \
                    }                                                                                \
                    ಗಾಗಿ (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        ಸ್ಥಿರಾಂಕ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        ಗಾಗಿ (uint32_t c = 0; c < chunk; c++) {                                       \
                            ಸ್ಥಿರಾಂಕ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (ಹೊರಹೋಗು_ಸತತ + ikw), (ಸ್ಥಿರಾಂಕ uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2_ಲಂಬಸಾಲು_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_ಲಂಬಸಾಲು_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, ದಶಮಾಂಶ,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_blocked_dma_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ htp_im2col_context * ictx        = (ರಚನೆ htp_im2col_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context *    octx        = ictx->octx;
    ರಚನೆ htp_thread_trace * ಸೀಮಿತ tr   = &octx->ctx->trace[ith];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst  = octx->dst;
    ಸ್ಥಿರಾಂಕ uint32_t OW           = ictx->OW;
    ಸ್ಥಿರಾಂಕ uint32_t KD = ictx->KD, KH = ictx->KH;
    ಸ್ಥಿರಾಂಕ uint32_t kd_kh        = KD * KH;
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride = ictx->patch_stride;
    ಸ್ಥಿರಾಂಕ uint32_t owb          = ictx->pe_owb;
    ಸ್ಥಿರಾಂಕ uint32_t nbpr         = ictx->pe_nbpr;
    ಸ್ಥಿರಾಂಕ uint32_t dst_elem     = ictx->dst_elem;
    ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ   = dst->ದತ್ತಾಂಶ;
    dma_ಸರತಿ * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_ತಳ         = ictx->pe_vtcm_src + ith * ictx->pe_src_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t * dstb_ತಳ         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    ದಶಮಾಂಶ * srcb2[2]            = { (ದಶಮಾಂಶ *) srcb_ತಳ, (ದಶಮಾಂಶ *) (srcb_ತಳ + ictx->pe_src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು) };
    ಶೂನ್ಯ * dstb2[2]             = { dstb_ತಳ, dstb_ತಳ + ictx->pe_dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು };
    ದಶಮಾಂಶ * scratch             = (ದಶಮಾಂಶ *) (srcb_ತಳ + 2 * ictx->pe_src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು);
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ_max  = ictx->pe_ಅಡ್ಡಸಾಲು_ತಳ + ictx->pe_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ    = ictx->pe_ಅಡ್ಡಸಾಲು_ತಳ + ictx->pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ      = MIN(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ictx->pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, ಅಡ್ಡಸಾಲು_ಕೊನೆ_max);
    ಆದರೆ (ಅಡ್ಡಸಾಲು_ಆರಂಭ >= ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ uint32_t nrows_local  = ಅಡ್ಡಸಾಲು_ಕೊನೆ - ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಖಂಡಗಳು = nrows_local * nbpr;
    ಆದರೆ (ಒಟ್ಟು_ಖಂಡಗಳು == 0) {
        ಹಿಂತಿರುಗಿ;
    }
    ರಚನೆ im2col_cursor cur = {
        .r   = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .ಒಳ  = ictx->thread_ಒಳ[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ರಚನೆ im2col_cursor stage = cur;
    ಗಾಗಿ (uint32_t i = 0; i < ಒಟ್ಟು_ಖಂಡಗಳು && i < 2; i++) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_ಹೆಜ್ಜೆ_ಖಂಡ(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    ಗಾಗಿ (uint32_t bi = 0; bi < ಒಟ್ಟು_ಖಂಡಗಳು; bi++) {
        ಸ್ಥಿರಾಂಕ uint32_t buf = bi & 1u;
        ದಶಮಾಂಶ * srcb = srcb2[buf];
        ಶೂನ್ಯ *  dstb = dstb2[buf];
        ಸ್ಥಿರಾಂಕ uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
        ಗಾಗಿ (uint32_t p = 0; p < kd_kh; p++) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q);
        }
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ಸ್ಥಿರಾಂಕ dma_addr_t ddr = dst_ದತ್ತಾಂಶ + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ಆದರೆ (bi + 2 < ಒಟ್ಟು_ಖಂಡಗಳು) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_ಹೆಜ್ಜೆ_ಖಂಡ(&stage, ictx);
        }
        im2col_3d_ಹೆಜ್ಜೆ_ಖಂಡ(&cur, ictx);
    }
    dma_ಸರತಿ_flush(dma_q);
}
ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_patchembed_dma_stage(ರಚನೆ htp_im2col_context * ictx, dma_ಸರತಿ * dma_q,
                                           ದಶಮಾಂಶ * srcb, ಸ್ಥಿರಾಂಕ ರಚನೆ im2col_cursor * c) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx = ictx->octx;
    ಸ್ಥಿರಾಂಕ uint32_t IC = ictx->IC;
    ಸ್ಥಿರಾಂಕ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ಸ್ಥಿರಾಂಕ uint32_t KD = ictx->KD, KH = ictx->KH;
    ಸ್ಥಿರಾಂಕ dma_addr_t src_ದತ್ತಾಂಶ = octx->src[1]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಳ = c->ಒಳ, iod = c->iod, ioh = c->ioh;
    ಸ್ಥಿರಾಂಕ size_t src_channel_stride  = (size_t) ID * IH * IW * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t vtcm_channel_stride = (size_t) KD * KH * IW * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ            = (size_t) KH * IW * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಗಾಗಿ (uint32_t ikd = 0; ikd < KD; ikd++) {
        ಸ್ಥಿರಾಂಕ uint32_t iid = iod * KD + ikd;
        ದಶಮಾಂಶ * vdst = srcb + (size_t) ikd * KH * IW;
        ಸ್ಥಿರಾಂಕ dma_addr_t vsrc = src_ದತ್ತಾಂಶ + (size_t) (((ಒಳ * IC) * ID + iid) * IH + ioh * KH) * IW * ಗಾತ್ರ(ದಶಮಾಂಶ);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, IC);
    }
}
ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_patchembed_compute_f16(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,
    ಶೂನ್ಯ * ಸೀಮಿತ dstb_raw, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb) {
    uint16_t * ಸೀಮಿತ dstb = (uint16_t *) dstb_raw;
    ಸ್ಥಿರಾಂಕ uint32_t OW           = ictx->OW;
    ಸ್ಥಿರಾಂಕ uint32_t IC           = ictx->IC;
    ಸ್ಥಿರಾಂಕ uint32_t KD           = ictx->KD;
    ಸ್ಥಿರಾಂಕ uint32_t KH           = ictx->KH;
    ಸ್ಥಿರಾಂಕ uint32_t KW           = ictx->KW;
    ಸ್ಥಿರಾಂಕ uint32_t IW           = ictx->IW;
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride = ictx->patch_stride;

    ಗಾಗಿ (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        ಗಾಗಿ (uint32_t iic = 0; iic < IC; iic++) {
            ಗಾಗಿ (uint32_t ikd = 0; ikd < KD; ikd++) {
                ಗಾಗಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * ಹೊರಹೋಗು_ಸತತ = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಸತತ = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) ಹೊರಹೋಗು_ಸತತ, (ಸ್ಥಿರಾಂಕ uint8_t *) src_ಸತತ, KW);
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_patchembed_compute_f32(
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_context * ictx,
    ಶೂನ್ಯ * ಸೀಮಿತ dstb_raw, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ srcb) {
    ದಶಮಾಂಶ * ಸೀಮಿತ dstb = (ದಶಮಾಂಶ *) dstb_raw;
    ಸ್ಥಿರಾಂಕ uint32_t OW           = ictx->OW;
    ಸ್ಥಿರಾಂಕ uint32_t IC           = ictx->IC;
    ಸ್ಥಿರಾಂಕ uint32_t KD           = ictx->KD;
    ಸ್ಥಿರಾಂಕ uint32_t KH           = ictx->KH;
    ಸ್ಥಿರಾಂಕ uint32_t KW           = ictx->KW;
    ಸ್ಥಿರಾಂಕ uint32_t IW           = ictx->IW;
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride = ictx->patch_stride;

    ಗಾಗಿ (uint32_t iow = 0; iow < OW; iow++) {
        ದಶಮಾಂಶ * dst_patch = dstb + (size_t) iow * patch_stride;
        ಗಾಗಿ (uint32_t iic = 0; iic < IC; iic++) {
            ಗಾಗಿ (uint32_t ikd = 0; ikd < KD; ikd++) {
                ಗಾಗಿ (uint32_t ikh = 0; ikh < KH; ikh++) {
                    ದಶಮಾಂಶ * ಹೊರಹೋಗು_ಸತತ = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಸತತ = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) ಹೊರಹೋಗು_ಸತತ, (ಸ್ಥಿರಾಂಕ uint8_t *) src_ಸತತ, KW);
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_patchembed_dma_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ htp_im2col_context * ictx        = (ರಚನೆ htp_im2col_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context *    octx        = ictx->octx;
    ರಚನೆ htp_thread_trace * ಸೀಮಿತ tr   = &octx->ctx->trace[ith];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst  = octx->dst;
    ಸ್ಥಿರಾಂಕ uint32_t KD           = ictx->KD;
    ಸ್ಥಿರಾಂಕ uint32_t OW           = ictx->OW;
    ಸ್ಥಿರಾಂಕ uint32_t patch_stride = ictx->patch_stride;
    ಸ್ಥಿರಾಂಕ uint32_t dst_elem     = ictx->dst_elem;
    ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ   = dst->ದತ್ತಾಂಶ;
    dma_ಸರತಿ * dma_q           = octx->ctx->dma[ith];
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ_max  = ictx->pe_ಅಡ್ಡಸಾಲು_ತಳ + ictx->pe_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ    = ictx->pe_ಅಡ್ಡಸಾಲು_ತಳ + ictx->pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ      = MIN(ಅಡ್ಡಸಾಲು_ಆರಂಭ + ictx->pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, ಅಡ್ಡಸಾಲು_ಕೊನೆ_max);
    ಆದರೆ (ಅಡ್ಡಸಾಲು_ಆರಂಭ >= ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t nrows_local  = ಅಡ್ಡಸಾಲು_ಕೊನೆ - ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint8_t * src_ತಳ          = ictx->pe_vtcm_src + ith * ictx->pe_src_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t * dstb_ತಳ         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    ದಶಮಾಂಶ * srcb2[2]            = { (ದಶಮಾಂಶ *) src_ತಳ, (ದಶಮಾಂಶ *) (src_ತಳ + ictx->pe_src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು) };
    ಶೂನ್ಯ * dstb2[2]             = { dstb_ತಳ, dstb_ತಳ + ictx->pe_dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು };
    ರಚನೆ im2col_cursor cur = {
        .r   = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .ಒಳ  = ictx->thread_ಒಳ[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ರಚನೆ im2col_cursor stage = cur;
    ಗಾಗಿ (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    ಗಾಗಿ (uint32_t idx = 0; idx < nrows_local; idx++) {
        ಸ್ಥಿರಾಂಕ uint32_t buf = idx & 1u;
        ದಶಮಾಂಶ * srcb = srcb2[buf];
        ಶೂನ್ಯ *  dstb = dstb2[buf];
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
        ಗಾಗಿ (uint32_t p = 0; p < KD; p++) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q);
        }
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ಸ್ಥಿರಾಂಕ dma_addr_t ddr_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ddr_ಅಡ್ಡಸಾಲು, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ಆದರೆ (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು(&stage, ictx);
        }
        im2col_3d_ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು(&cur, ictx);
    }
    dma_ಸರತಿ_flush(dma_q);
}

ಪೂರ್ಣಾಂಕ op_im2col(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_kernel_params *) octx->kernel_params;
    ಆದರೆ (kparams->kernel_type == HTP_IM2_ಲಂಬಸಾಲು_KERNEL_UNSUPPORTED) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;

    uint32_t ಅಡ್ಡಸಾಲು_ತಳ = 0;
    uint32_t nrows    = kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->nb[2];
        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = (ಅಡ್ಡಸಾಲು_ಗಾತ್ರ > 0) ? (HEX_L2_ಸಾಲು_ಗಾತ್ರ / hex_ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_u32(ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, HEX_L2_ಸಾಲು_ಗಾತ್ರ)) : 1;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range =
            htp_tensor_mdev_ವಿಭಜನೆ(kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) ? ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ತಳ = range.ಆರಂಭ;
        nrows    = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ರಚನೆ htp_im2col_context ictx = { 0 };
    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = octx->ctx->vtcm_ತಳ;

    ಸ್ಥಿರಾಂಕ uint32_t pth = MIN(octx->n_ಎಳೆಗಳು, nrows);
    ictx.octx                   = octx;
    ictx.pe_ಅಡ್ಡಸಾಲು_ತಳ            = ಅಡ್ಡಸಾಲು_ತಳ;
    ictx.pe_nrows               = nrows;
    ictx.pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು       = kparams->src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
    ictx.pe_dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು       = kparams->dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, ತಳ, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, ತಳ, kparams->off_dst);
    ictx.pe_src_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    ictx.pe_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
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

    ಗಾಗಿ (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = ಅಡ್ಡಸಾಲು_ತಳ + ictx.pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_ಒಳ[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? ಗಾತ್ರ(__fp16) : ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಆದರೆ (kparams->kernel_type == HTP_IM2_ಲಂಬಸಾಲು_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ಶೂನ್ಯ *) im2col_patchembed_compute_f16
                                                 : (ಶೂನ್ಯ *) im2col_patchembed_compute_f32;
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, im2col_patchembed_dma_thread, &ictx, pth);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kparams->kernel_type == HTP_IM2_ಲಂಬಸಾಲು_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ಶೂನ್ಯ *) im2col_blocked_compute_f16
                                                 : (ಶೂನ್ಯ *) im2col_blocked_compute_f32;
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, im2col_blocked_dma_thread, &ictx, pth);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
}

ಪೂರ್ಣಾಂಕ op_im2col_3d(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_kernel_params * kparams =
        (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_im2col_kernel_params *) octx->kernel_params;
    ಆದರೆ (kparams->kernel_type == HTP_IM2_ಲಂಬಸಾಲು_KERNEL_UNSUPPORTED) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;

    uint32_t ಅಡ್ಡಸಾಲು_ತಳ = 0;
    uint32_t nrows    = kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->nb[2];
        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = (ಅಡ್ಡಸಾಲು_ಗಾತ್ರ > 0) ? (HEX_L2_ಸಾಲು_ಗಾತ್ರ / hex_ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_u32(ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, HEX_L2_ಸಾಲು_ಗಾತ್ರ)) : 1;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range =
            htp_tensor_mdev_ವಿಭಜನೆ(kparams->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) ? ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ತಳ = range.ಆರಂಭ;
        nrows    = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ರಚನೆ htp_im2col_context ictx = { 0 };
    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = octx->ctx->vtcm_ತಳ;

    ಸ್ಥಿರಾಂಕ uint32_t pth = MIN(octx->n_ಎಳೆಗಳು, nrows);
    ictx.octx                   = octx;
    ictx.pe_ಅಡ್ಡಸಾಲು_ತಳ            = ಅಡ್ಡಸಾಲು_ತಳ;
    ictx.pe_nrows               = nrows;
    ictx.pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು       = kparams->src_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
    ictx.pe_dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು       = kparams->dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, ತಳ, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, ತಳ, kparams->off_dst);
    ictx.pe_src_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    ictx.pe_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
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

    ಗಾಗಿ (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = ಅಡ್ಡಸಾಲು_ತಳ + ictx.pe_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_ಒಳ[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? ಗಾತ್ರ(__fp16) : ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಆದರೆ (kparams->kernel_type == HTP_IM2_ಲಂಬಸಾಲು_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ಶೂನ್ಯ *) im2col_3d_patchembed_compute_f16
                                                 : (ಶೂನ್ಯ *) im2col_3d_patchembed_compute_f32;
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, im2col_3d_patchembed_dma_thread, &ictx, pth);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kparams->kernel_type == HTP_IM2_ಲಂಬಸಾಲು_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ಶೂನ್ಯ *) im2col_3d_blocked_compute_f16
                                                 : (ಶೂನ್ಯ *) im2col_3d_blocked_compute_f32;
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, im2col_3d_blocked_dma_thread, &ictx, pth);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
}
