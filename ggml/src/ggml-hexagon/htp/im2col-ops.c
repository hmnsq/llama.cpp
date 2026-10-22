#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <hexagon_protos.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <hexagon_types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "im2col-ops.h"

ਬਣਤਰ htp_im2col_context {
    ਬਣਤਰ htp_ops_context * octx;

    uint32_t pe_ਪੰਕਤੀ_ਅਧਾਰ;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;                   // output rows per worker
    uint32_t pe_src_ਪੰਕਤੀ_ਬਾਈਟਾਂ;                     // one output row's source, rounded 256
    uint32_t pe_dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_ਮਾਪ_ਪ੍ਰਤੀ_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;              // 2 * pe_dst_row_bytes

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
    uint32_t thread_ਵਿੱਚ[HTP_MAX_NTHREADS];
    uint32_t thread_iod[HTP_MAX_NTHREADS];
    uint32_t thread_ioh[HTP_MAX_NTHREADS];

    ਖਾਲੀ * compute;
    uint32_t dst_elem;
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*im2col_blocked_compute_fn_t)(
    ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,
    ਖਾਲੀ * ਸੀਮਿਤ dstb, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb,
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ scratch,
    uint32_t nb, uint32_t c0, uint32_t ioh);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*im2col_patchembed_compute_fn_t)(
    ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,
    ਖਾਲੀ * ਸੀਮਿਤ dstb, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb, uint32_t ioh);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*im2col_3d_blocked_compute_fn_t)(
    ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,
    ਖਾਲੀ * ਸੀਮਿਤ dstb, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb,
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ scratch,
    uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*im2col_3d_patchembed_compute_fn_t)(
    ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,
    ਖਾਲੀ * ਸੀਮਿਤ dstb, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb);

ਬਣਤਰ im2col_cursor {
    uint32_t r;
    uint32_t ਵਿੱਚ;
    uint32_t iod;
    uint32_t ioh;
    uint32_t c0;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ im2col_ਕਦਮ_ਪੰਕਤੀ(ਬਣਤਰ im2col_cursor * c, ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ਜੇ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->ਵਿੱਚ++;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ im2col_ਕਦਮ_ਖੰਡ(ਬਣਤਰ im2col_cursor * c, ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ਜੇ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ਜੇ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->ਵਿੱਚ++;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ im2col_blocked_dma_stage(ਬਣਤਰ htp_im2col_context * ictx, dma_ਕਤਾਰ * dma_q,
                                     ਦਸ਼ਮਲਵ * srcb, ਅਚਲ ਬਣਤਰ im2col_cursor * c) {
    ਅਚਲ uint32_t ਵਿੱਚ = c->ਵਿੱਚ, ioh = c->ioh, c0 = c->c0;
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx = ictx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    ਅਚਲ int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    ਅਚਲ int32_t d1 = octx->op_params[5];
    ਅਚਲ bool is_2D = (octx->op_params[6] == 1);
    ਅਚਲ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ਅਚਲ uint32_t KH = ictx->KH;
    ਅਚਲ uint32_t Wb = ictx->pe_wb;
    ਅਚਲ dma_addr_t src_ਅੰਕੜੇ = src1->ਅੰਕੜੇ;
    ਅਚਲ int32_t win0 = (int32_t) c0 * s0 - p0;
    ਅਚਲ size_t src_channel_stride = is_2D ? src1->nb[2] : src1->nb[1];
    ਅਚਲ size_t batch_ਵਿੱਥ        = is_2D ? (size_t) ਵਿੱਚ * src1->nb[3] : (size_t) ਵਿੱਚ * src1->nb[2];

    ਲਈ (uint32_t ikh = 0; ikh < KH; ikh++) {
        ਅਚਲ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        ਜੇ (iih < 0 || iih >= (int32_t) IH) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(srcb, 0), 0, 0, 0, 0);
            ਜਾਰੀ;
        }
        ਅਚਲ int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        ਜੇ (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        ਜੇ (hi <= lo) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(srcb, 0), 0, 0, 0, 0);
            ਜਾਰੀ;
        }
        ਅਚਲ uint32_t cpw = (uint32_t) (hi - lo);
        ਦਸ਼ਮਲਵ * vdst = srcb + (size_t) ikh * Wb + lo;
        ਅਚਲ dma_addr_t vsrc = src_ਅੰਕੜੇ + batch_ਵਿੱਥ + (size_t) iih * src1->nb[1] + (size_t) (win0 + lo) * src1->nb[0];
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vdst, vsrc),
                       (size_t) KH * Wb * ਆਕਾਰ(ਦਸ਼ਮਲਵ), src_channel_stride,
                       cpw * ਆਕਾਰ(ਦਸ਼ਮਲਵ), IC);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IM2_ਖਾਨਾ_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                         \
ਸਥਿਰ ਖਾਲੀ FNAME(                                                                               \
    ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,                                                      \
    ਖਾਲੀ * ਸੀਮਿਤ dstb_raw, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb,                                       \
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ scratch,                                                                    \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                    \
    DST_CTYPE * ਸੀਮਿਤ dstb = (DST_CTYPE *) dstb_raw;                                          \
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx = ictx->octx;                                            \
    ਅਚਲ int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                             \
    ਅਚਲ int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                             \
    ਅਚਲ int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                             \
    ਅਚਲ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                  \
    ਅਚਲ uint32_t KH = ictx->KH, KW = ictx->KW;                                                 \
    ਅਚਲ uint32_t Wb = ictx->pe_wb;                                                             \
    ਅਚਲ uint32_t patch_stride = ictx->patch_stride;                                            \
    ਅਚਲ int32_t  win0 = (int32_t) c0 * s0 - p0;                                                \
                                                                                                 \
    ਲਈ (uint32_t j = 0; j < nb; j++) {                                                          \
        ਅਚਲ uint32_t iow       = c0 + j;                                                       \
        DST_CTYPE *    dst_patch = dstb + (size_t) j * patch_stride;                             \
        ਅਚਲ int32_t  iiw0      = (int32_t) iow * s0 - p0;                                      \
        ਲਈ (uint32_t ikh = 0; ikh < KH; ikh++) {                                                \
            ਅਚਲ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                    \
            ਅਚਲ ਪੂਰਨ     okh = (iih >= 0 && iih < (int32_t) IH);                                \
            ਲਈ (uint32_t iic = 0; iic < IC; iic++) {                                            \
                DST_CTYPE * ਬਾਹਰਲਾ_ਲੜੀਵਾਰ = dst_patch + (size_t) (iic * KH + ikh) * KW;                \
                ਜੇ (!okh) {                                                                      \
                    SPLAT_FN(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, 0, KW);                                                    \
                    ਜਾਰੀ;                                                                    \
                }                                                                                \
                ਅਚਲ ਦਸ਼ਮਲਵ * vrow = srcb + (size_t) (iic * KH + ikh) * Wb;                      \
                ਜੇ (d0 == 1) {                                                                   \
                    ਅਚਲ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                    int32_t       hi = (int32_t) IW - iiw0;                                      \
                    ਜੇ (hi > (int32_t) KW) {                                                     \
                        hi = (int32_t) KW;                                                       \
                    }                                                                            \
                    ਜੇ (hi <= lo) {                                                              \
                        SPLAT_FN(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, 0, KW);                                                \
                    } ਨਹੀਂਤਾਂ {                                                                     \
                        ਜੇ (lo > 0) {                                                            \
                            SPLAT_FN(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, 0, (uint32_t) lo);                                 \
                        }                                                                        \
                        COPY_FN((uint8_t *) (ਬਾਹਰਲਾ_ਲੜੀਵਾਰ + lo),                                      \
                                (ਅਚਲ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                (uint32_t) (hi - lo));                                           \
                        ਜੇ (hi < (int32_t) KW) {                                                 \
                            SPLAT_FN(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ + hi, 0, (uint32_t) (KW - hi));                     \
                        }                                                                        \
                    }                                                                            \
                    ਜਾਰੀ;                                                                    \
                }                                                                                \
                ਲਈ (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                    ਅਚਲ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                    ਲਈ (uint32_t c = 0; c < chunk; c++) {                                       \
                        ਅਚਲ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                        scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                    }                                                                            \
                    COPY_FN((uint8_t *) (ਬਾਹਰਲਾ_ਲੜੀਵਾਰ + ikw), (ਅਚਲ uint8_t *) scratch, chunk);      \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

IM2_ਖਾਨਾ_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_ਖਾਨਾ_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, ਦਸ਼ਮਲਵ,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ਸਥਿਰ ਖਾਲੀ im2col_blocked_dma_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ htp_im2col_context * ictx      = (ਬਣਤਰ htp_im2col_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context *    octx      = ictx->octx;
    ਬਣਤਰ htp_thread_trace * ਸੀਮਿਤ tr = &octx->ctx->trace[ith];
    ਅਚਲ uint32_t OW                     = ictx->OW;
    ਅਚਲ uint32_t KH                     = ictx->KH;
    ਅਚਲ uint32_t owb                    = ictx->pe_owb;
    ਅਚਲ uint32_t nbpr                   = ictx->pe_nbpr;
    ਅਚਲ uint32_t patch_stride           = ictx->patch_stride;
    ਅਚਲ uint32_t dst_elem               = ictx->dst_elem;
    im2col_blocked_compute_fn_t compute   = (im2col_blocked_compute_fn_t) ictx->compute;
    ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ             = octx->dst->ਅੰਕੜੇ;
    dma_ਕਤਾਰ *    dma_q                  = octx->ctx->dma[ith];
    uint8_t *      srcb_ਅਧਾਰ              = ictx->pe_vtcm_src + ith * ictx->pe_src_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t *      dstb_ਅਧਾਰ              = ictx->pe_vtcm_dst + ith * ictx->pe_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    ਦਸ਼ਮਲਵ *        srcb2[2]               = { (ਦਸ਼ਮਲਵ *) srcb_ਅਧਾਰ, (ਦਸ਼ਮਲਵ *) (srcb_ਅਧਾਰ + ictx->pe_src_ਪੰਕਤੀ_ਬਾਈਟਾਂ) };
    ਖਾਲੀ *         dstb2[2]               = { dstb_ਅਧਾਰ, dstb_ਅਧਾਰ + ictx->pe_dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ };
    ਦਸ਼ਮਲਵ *        scratch                = (ਦਸ਼ਮਲਵ *) (srcb_ਅਧਾਰ + 2 * ictx->pe_src_ਪੰਕਤੀ_ਬਾਈਟਾਂ);
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ_max            = ictx->pe_ਪੰਕਤੀ_ਅਧਾਰ + ictx->pe_nrows;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ              = ictx->pe_ਪੰਕਤੀ_ਅਧਾਰ + ictx->pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ                = MIN(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ictx->pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, ਪੰਕਤੀ_ਸਿਰਾ_max);
    ਜੇ (ਪੰਕਤੀ_ਸ਼ੁਰੂ >= ਪੰਕਤੀ_ਸਿਰਾ) {
        ਵਾਪਸ;
    }
    ਅਚਲ uint32_t nrows_local  = ਪੰਕਤੀ_ਸਿਰਾ - ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ਅਚਲ uint32_t ਕੁੱਲ_ਖੰਡਾਂ = nrows_local * nbpr;
    ਜੇ (ਕੁੱਲ_ਖੰਡਾਂ == 0) {
        ਵਾਪਸ;
    }
    ਬਣਤਰ im2col_cursor cur = {
        .r   = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .ਵਿੱਚ  = ictx->thread_ਵਿੱਚ[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ਬਣਤਰ im2col_cursor stage = cur;
    ਲਈ (uint32_t i = 0; i < ਕੁੱਲ_ਖੰਡਾਂ && i < 2; i++) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, dstb2[i]), 0, 0, 0, 0);
        im2col_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_ਕਦਮ_ਖੰਡ(&stage, ictx);
    }
    ਲਈ (uint32_t bi = 0; bi < ਕੁੱਲ_ਖੰਡਾਂ; bi++) {
        ਅਚਲ uint32_t buf = bi & 1u;
        ਦਸ਼ਮਲਵ * srcb = srcb2[buf];
        ਖਾਲੀ *  dstb = dstb2[buf];
        ਅਚਲ uint32_t nb = MIN(owb, OW - cur.c0);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        ਲਈ (uint32_t k = 0; k < KH; k++) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        }
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ਅਚਲ dma_addr_t ddr = dst_ਅੰਕੜੇ + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ਜੇ (bi + 2 < ਕੁੱਲ_ਖੰਡਾਂ) {
            im2col_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_ਕਦਮ_ਖੰਡ(&stage, ictx);
        }
        im2col_ਕਦਮ_ਖੰਡ(&cur, ictx);
    }
    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਖਾਲੀ im2col_patchembed_dma_stage(ਬਣਤਰ htp_im2col_context * ictx, dma_ਕਤਾਰ * dma_q,
                                        ਦਸ਼ਮਲਵ * srcb, ਅਚਲ ਬਣਤਰ im2col_cursor * c) {
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx = ictx->octx;
    ਅਚਲ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ਅਚਲ uint32_t KH = ictx->KH;
    ਅਚਲ uint32_t ਵਿੱਚ = c->ਵਿੱਚ, ioh = c->ioh;
    ਅਚਲ size_t src_channel_stride  = (size_t) IH * IW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t vtcm_channel_stride = (size_t) KH * IW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ dma_addr_t src_ਅੰਕੜੇ        = octx->src[1]->ਅੰਕੜੇ;
    ਅਚਲ dma_addr_t vsrc            = src_ਅੰਕੜੇ + (size_t) (((ਵਿੱਚ * IC) * IH + ioh * KH) * IW) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(srcb, vsrc),
                   vtcm_channel_stride, src_channel_stride,
                   vtcm_channel_stride, IC);
}

ਸਥਿਰ ਖਾਲੀ im2col_patchembed_compute_f16(
    ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,
    ਖਾਲੀ * ਸੀਮਿਤ dstb_raw, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb, uint32_t ioh) {
    uint16_t * ਸੀਮਿਤ dstb = (uint16_t *) dstb_raw;
    ਅਚਲ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ਅਚਲ uint32_t KH = ictx->KH, KW = ictx->KW;
    ਅਚਲ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    ਲਈ (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        ਲਈ (uint32_t ikh = 0; ikh < KH; ikh++) {
            ਅਚਲ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            ਲਈ (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * ਬਾਹਰਲਾ_ਲੜੀਵਾਰ = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ਜੇ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, 0, KW);
                    ਜਾਰੀ;
                }
                ਅਚਲ ਦਸ਼ਮਲਵ * src_ਲੜੀਵਾਰ = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, (ਅਚਲ uint8_t *) src_ਲੜੀਵਾਰ, KW);
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ im2col_patchembed_compute_f32(
    ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,
    ਖਾਲੀ * ਸੀਮਿਤ dstb_raw, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb, uint32_t ioh) {
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dstb = (ਦਸ਼ਮਲਵ *) dstb_raw;
    ਅਚਲ uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    ਅਚਲ uint32_t KH = ictx->KH, KW = ictx->KW;
    ਅਚਲ uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    ਲਈ (uint32_t iow = 0; iow < OW; iow++) {
        ਦਸ਼ਮਲਵ * dst_patch = dstb + (size_t) iow * patch_stride;
        ਲਈ (uint32_t ikh = 0; ikh < KH; ikh++) {
            ਅਚਲ int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            ਲਈ (uint32_t iic = 0; iic < IC; iic++) {
                ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ_ਲੜੀਵਾਰ = dst_patch + (size_t) (iic * KH + ikh) * KW;
                ਜੇ (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, 0, KW);
                    ਜਾਰੀ;
                }
                ਅਚਲ ਦਸ਼ਮਲਵ * src_ਲੜੀਵਾਰ = srcb + (size_t) (iic * KH + ikh) * IW + (size_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, (ਅਚਲ uint8_t *) src_ਲੜੀਵਾਰ, KW);
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ im2col_patchembed_dma_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ htp_im2col_context * ictx        = (ਬਣਤਰ htp_im2col_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context *    octx        = ictx->octx;
    ਬਣਤਰ htp_thread_trace * ਸੀਮਿਤ tr   = &octx->ctx->trace[ith];
    ਅਚਲ uint32_t OW                       = ictx->OW;
    ਅਚਲ uint32_t patch_stride             = ictx->patch_stride;
    ਅਚਲ uint32_t dst_elem                 = ictx->dst_elem;
    im2col_patchembed_compute_fn_t compute  = (im2col_patchembed_compute_fn_t) ictx->compute;
    ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ               = octx->dst->ਅੰਕੜੇ;
    dma_ਕਤਾਰ *    dma_q                    = octx->ctx->dma[ith];
    uint8_t *      srcb_ਅਧਾਰ                = ictx->pe_vtcm_src + ith * ictx->pe_src_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t *      dstb_ਅਧਾਰ                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    ਦਸ਼ਮਲਵ *        srcb2[2]                 = { (ਦਸ਼ਮਲਵ *) srcb_ਅਧਾਰ, (ਦਸ਼ਮਲਵ *) (srcb_ਅਧਾਰ + ictx->pe_src_ਪੰਕਤੀ_ਬਾਈਟਾਂ) };
    ਖਾਲੀ *         dstb2[2]                 = { dstb_ਅਧਾਰ, dstb_ਅਧਾਰ + ictx->pe_dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ };
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ_max              = ictx->pe_ਪੰਕਤੀ_ਅਧਾਰ + ictx->pe_nrows;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ                = ictx->pe_ਪੰਕਤੀ_ਅਧਾਰ + ictx->pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ                  = MIN(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ictx->pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, ਪੰਕਤੀ_ਸਿਰਾ_max);
    ਜੇ (ਪੰਕਤੀ_ਸ਼ੁਰੂ >= ਪੰਕਤੀ_ਸਿਰਾ) {
        ਵਾਪਸ;
    }
    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ਪੰਕਤੀ_ਸਿਰਾ - ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ਜੇ (ਕੁੱਲ_ਪੰਕਤੀਆਂ == 0) {
        ਵਾਪਸ;
    }
    ਬਣਤਰ im2col_cursor cur = {
        .r   = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .ਵਿੱਚ  = ictx->thread_ਵਿੱਚ[ith],
        .iod = 0,
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ਬਣਤਰ im2col_cursor stage = cur;
    ਲਈ (uint32_t i = 0; i < ਕੁੱਲ_ਪੰਕਤੀਆਂ && i < 2; i++) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, dstb2[i]), 0, 0, 0, 0);
        im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_ਕਦਮ_ਪੰਕਤੀ(&stage, ictx);
    }
    ਲਈ (uint32_t ri = 0; ri < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ri++) {
        ਅਚਲ uint32_t buf = ri & 1u;
        ਦਸ਼ਮਲਵ * srcb = srcb2[buf];
        ਖਾਲੀ *  dstb = dstb2[buf];
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ਅਚਲ dma_addr_t ddr_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ddr_ਪੰਕਤੀ, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ਜੇ (ri + 2 < ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
            im2col_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_ਕਦਮ_ਪੰਕਤੀ(&stage, ictx);
        }
        im2col_ਕਦਮ_ਪੰਕਤੀ(&cur, ictx);
    }
    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ im2col_3d_ਕਦਮ_ਪੰਕਤੀ(ਬਣਤਰ im2col_cursor * c, ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx) {
    c->r++;
    c->ioh++;
    ਜੇ (c->ioh >= ictx->OH) {
        c->ioh = 0;
        c->iod++;
        ਜੇ (c->iod >= ictx->OD) {
            c->iod = 0;
            c->ਵਿੱਚ++;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ im2col_3d_ਕਦਮ_ਖੰਡ(ਬਣਤਰ im2col_cursor * c, ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx) {
    c->c0 += ictx->pe_owb;
    ਜੇ (c->c0 >= ictx->OW) {
        c->c0 = 0;
        c->r++;
        c->ioh++;
        ਜੇ (c->ioh >= ictx->OH) {
            c->ioh = 0;
            c->iod++;
            ਜੇ (c->iod >= ictx->OD) {
                c->iod = 0;
                c->ਵਿੱਚ++;
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ im2col_3d_blocked_dma_stage(ਬਣਤਰ htp_im2col_context * ictx, dma_ਕਤਾਰ * dma_q,
                                        ਦਸ਼ਮਲਵ * srcb, ਅਚਲ ਬਣਤਰ im2col_cursor * c) {
    ਅਚਲ uint32_t ਵਿੱਚ = c->ਵਿੱਚ, iod = c->iod, ioh = c->ioh, c0 = c->c0;
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx = ictx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    ਅਚਲ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    ਅਚਲ int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    ਅਚਲ uint32_t IC = ictx->IC;
    ਅਚਲ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ਅਚਲ uint32_t KD = ictx->KD, KH = ictx->KH;
    ਅਚਲ uint32_t Wb = ictx->pe_wb;
    ਅਚਲ int32_t win0 = (int32_t) c0 * s0 - p0;
    ਅਚਲ dma_addr_t src_ਅੰਕੜੇ = src1->ਅੰਕੜੇ;
    ਅਚਲ size_t src_channel_stride = src1->nb[3];
    ਅਚਲ size_t vtcm_channel_stride = (size_t) KD * KH * Wb * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t batch_ਵਿੱਥ        = (size_t) (ਵਿੱਚ * IC) * src1->nb[3];

    ਲਈ (uint32_t ikd = 0; ikd < KD; ikd++) {
        ਅਚਲ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;
        ਅਚਲ bool valid_d = (iid >= 0 && iid < (int32_t) ID);
        ਲਈ (uint32_t ikh = 0; ikh < KH; ikh++) {
            ਅਚਲ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
            ਜੇ (!valid_d || iih < 0 || iih >= (int32_t) IH) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(srcb, 0), 0, 0, 0, 0);
                ਜਾਰੀ;
            }
            ਅਚਲ int32_t lo = win0 < 0 ? -win0 : 0;
            int32_t hi = (int32_t) IW - win0;
            ਜੇ (hi > (int32_t) Wb) {
                hi = (int32_t) Wb;
            }
            ਜੇ (hi <= lo) {
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(srcb, 0), 0, 0, 0, 0);
                ਜਾਰੀ;
            }
            ਦਸ਼ਮਲਵ * vdst = srcb + (size_t) (ikd * KH + ikh) * Wb + lo;
            ਅਚਲ dma_addr_t vsrc = src_ਅੰਕੜੇ + batch_ਵਿੱਥ +
                (size_t) iid * src1->nb[2] +
                (size_t) iih * src1->nb[1] +
                (size_t) (win0 + lo) * src1->nb[0];
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * ਆਕਾਰ(ਦਸ਼ਮਲਵ), IC);
        }
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ IM2_ਖਾਨਾ_3_D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
ਸਥਿਰ ਖਾਲੀ FNAME(ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,                                            \
                  ਖਾਲੀ * ਸੀਮਿਤ dstb_raw, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb,                             \
                  ਦਸ਼ਮਲਵ * ਸੀਮਿਤ scratch,                                                          \
                  uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {                            \
    DST_CTYPE * ਸੀਮਿਤ dstb = (DST_CTYPE *) dstb_raw;                                              \
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx = ictx->octx;                                                \
    ਅਚਲ int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    ਅਚਲ int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    ਅਚਲ int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    ਅਚਲ uint32_t IC = ictx->IC;                                                                    \
    ਅਚਲ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    ਅਚਲ uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    ਅਚਲ uint32_t patch_stride = ictx->patch_stride;                                                \
    ਅਚਲ uint32_t Wb = ictx->pe_wb;                                                                 \
    ਅਚਲ int32_t win0 = (int32_t) c0 * s0 - p0;                                                     \
    ਲਈ (uint32_t j = 0; j < nb; j++) {                                                              \
        ਅਚਲ uint32_t iow = c0 + j;                                                                 \
        ਅਚਲ int32_t iiw0 = (int32_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (size_t) j * patch_stride;                                    \
        ਲਈ (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            ਲਈ (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                ਅਚਲ int32_t iid = (int32_t) iod * s2 + (int32_t) ikd * d2 - p2;                    \
                ਲਈ (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    ਅਚਲ int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;                \
                    DST_CTYPE * ਬਾਹਰਲਾ_ਲੜੀਵਾਰ = dst_patch +                                                \
                        (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;                              \
                    ਜੇ (iid < 0 || iid >= (int32_t) ID || iih < 0 || iih >= (int32_t) IH) {          \
                        SPLAT_FN(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, 0, KW);                                                    \
                        ਜਾਰੀ;                                                                    \
                    }                                                                                \
                    ਅਚਲ ਦਸ਼ਮਲਵ * vrow = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * Wb;      \
                    ਜੇ (d0 == 1) {                                                                   \
                        ਅਚਲ int32_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int32_t hi = (int32_t) IW - iiw0;                                            \
                        ਜੇ (hi > (int32_t) KW) {                                                     \
                            hi = (int32_t) KW;                                                       \
                        }                                                                            \
                        ਜੇ (hi <= lo) {                                                              \
                            SPLAT_FN(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, 0, KW);                                                \
                        } ਨਹੀਂਤਾਂ {                                                                     \
                            ਜੇ (lo > 0) {                                                            \
                                SPLAT_FN(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (ਬਾਹਰਲਾ_ਲੜੀਵਾਰ + lo),                                      \
                                    (ਅਚਲ uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            ਜੇ (hi < (int32_t) KW) {                                                 \
                                SPLAT_FN(ਬਾਹਰਲਾ_ਲੜੀਵਾਰ + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        ਜਾਰੀ;                                                                    \
                    }                                                                                \
                    ਲਈ (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        ਅਚਲ uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        ਲਈ (uint32_t c = 0; c < chunk; c++) {                                       \
                            ਅਚਲ int32_t iiw = iiw0 + (int32_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int32_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (ਬਾਹਰਲਾ_ਲੜੀਵਾਰ + ikw), (ਅਚਲ uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2_ਖਾਨਾ_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2_ਖਾਨਾ_3_D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, ਦਸ਼ਮਲਵ,    hvx_copy_f32_uu,     hvx_splat_f32_u)

ਸਥਿਰ ਖਾਲੀ im2col_3d_blocked_dma_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ htp_im2col_context * ictx        = (ਬਣਤਰ htp_im2col_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context *    octx        = ictx->octx;
    ਬਣਤਰ htp_thread_trace * ਸੀਮਿਤ tr   = &octx->ctx->trace[ith];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst  = octx->dst;
    ਅਚਲ uint32_t OW           = ictx->OW;
    ਅਚਲ uint32_t KD = ictx->KD, KH = ictx->KH;
    ਅਚਲ uint32_t kd_kh        = KD * KH;
    ਅਚਲ uint32_t patch_stride = ictx->patch_stride;
    ਅਚਲ uint32_t owb          = ictx->pe_owb;
    ਅਚਲ uint32_t nbpr         = ictx->pe_nbpr;
    ਅਚਲ uint32_t dst_elem     = ictx->dst_elem;
    ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ   = dst->ਅੰਕੜੇ;
    dma_ਕਤਾਰ * dma_q           = octx->ctx->dma[ith];
    uint8_t * srcb_ਅਧਾਰ         = ictx->pe_vtcm_src + ith * ictx->pe_src_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t * dstb_ਅਧਾਰ         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    ਦਸ਼ਮਲਵ * srcb2[2]            = { (ਦਸ਼ਮਲਵ *) srcb_ਅਧਾਰ, (ਦਸ਼ਮਲਵ *) (srcb_ਅਧਾਰ + ictx->pe_src_ਪੰਕਤੀ_ਬਾਈਟਾਂ) };
    ਖਾਲੀ * dstb2[2]             = { dstb_ਅਧਾਰ, dstb_ਅਧਾਰ + ictx->pe_dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ };
    ਦਸ਼ਮਲਵ * scratch             = (ਦਸ਼ਮਲਵ *) (srcb_ਅਧਾਰ + 2 * ictx->pe_src_ਪੰਕਤੀ_ਬਾਈਟਾਂ);
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ_max  = ictx->pe_ਪੰਕਤੀ_ਅਧਾਰ + ictx->pe_nrows;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ    = ictx->pe_ਪੰਕਤੀ_ਅਧਾਰ + ictx->pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ      = MIN(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ictx->pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, ਪੰਕਤੀ_ਸਿਰਾ_max);
    ਜੇ (ਪੰਕਤੀ_ਸ਼ੁਰੂ >= ਪੰਕਤੀ_ਸਿਰਾ) {
        ਵਾਪਸ;
    }
    ਅਚਲ uint32_t nrows_local  = ਪੰਕਤੀ_ਸਿਰਾ - ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ਅਚਲ uint32_t ਕੁੱਲ_ਖੰਡਾਂ = nrows_local * nbpr;
    ਜੇ (ਕੁੱਲ_ਖੰਡਾਂ == 0) {
        ਵਾਪਸ;
    }
    ਬਣਤਰ im2col_cursor cur = {
        .r   = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .ਵਿੱਚ  = ictx->thread_ਵਿੱਚ[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ਬਣਤਰ im2col_cursor stage = cur;
    ਲਈ (uint32_t i = 0; i < ਕੁੱਲ_ਖੰਡਾਂ && i < 2; i++) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_ਕਦਮ_ਖੰਡ(&stage, ictx);
    }

    im2col_3d_blocked_compute_fn_t compute = (im2col_3d_blocked_compute_fn_t) ictx->compute;
    ਲਈ (uint32_t bi = 0; bi < ਕੁੱਲ_ਖੰਡਾਂ; bi++) {
        ਅਚਲ uint32_t buf = bi & 1u;
        ਦਸ਼ਮਲਵ * srcb = srcb2[buf];
        ਖਾਲੀ *  dstb = dstb2[buf];
        ਅਚਲ uint32_t nb  = MIN(owb, OW - cur.c0);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        ਲਈ (uint32_t p = 0; p < kd_kh; p++) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        }
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb, scratch, nb, cur.c0, cur.iod, cur.ioh);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ਅਚਲ dma_addr_t ddr = dst_ਅੰਕੜੇ + (size_t) (cur.r * OW + cur.c0) * patch_stride * dst_elem;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ddr, dstb),
                       nb * patch_stride * dst_elem, nb * patch_stride * dst_elem,
                       nb * patch_stride * dst_elem, 1);
        ਜੇ (bi + 2 < ਕੁੱਲ_ਖੰਡਾਂ) {
            im2col_3d_blocked_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_ਕਦਮ_ਖੰਡ(&stage, ictx);
        }
        im2col_3d_ਕਦਮ_ਖੰਡ(&cur, ictx);
    }
    dma_ਕਤਾਰ_flush(dma_q);
}
ਸਥਿਰ ਖਾਲੀ im2col_3d_patchembed_dma_stage(ਬਣਤਰ htp_im2col_context * ictx, dma_ਕਤਾਰ * dma_q,
                                           ਦਸ਼ਮਲਵ * srcb, ਅਚਲ ਬਣਤਰ im2col_cursor * c) {
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx = ictx->octx;
    ਅਚਲ uint32_t IC = ictx->IC;
    ਅਚਲ uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    ਅਚਲ uint32_t KD = ictx->KD, KH = ictx->KH;
    ਅਚਲ dma_addr_t src_ਅੰਕੜੇ = octx->src[1]->ਅੰਕੜੇ;
    ਅਚਲ uint32_t ਵਿੱਚ = c->ਵਿੱਚ, iod = c->iod, ioh = c->ioh;
    ਅਚਲ size_t src_channel_stride  = (size_t) ID * IH * IW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t vtcm_channel_stride = (size_t) KD * KH * IW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ            = (size_t) KH * IW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਲਈ (uint32_t ikd = 0; ikd < KD; ikd++) {
        ਅਚਲ uint32_t iid = iod * KD + ikd;
        ਦਸ਼ਮਲਵ * vdst = srcb + (size_t) ikd * KH * IW;
        ਅਚਲ dma_addr_t vsrc = src_ਅੰਕੜੇ + (size_t) (((ਵਿੱਚ * IC) * ID + iid) * IH + ioh * KH) * IW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, ਪੰਕਤੀ_ਮਾਪ, IC);
    }
}
ਸਥਿਰ ਖਾਲੀ im2col_3d_patchembed_compute_f16(
    ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,
    ਖਾਲੀ * ਸੀਮਿਤ dstb_raw, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb) {
    uint16_t * ਸੀਮਿਤ dstb = (uint16_t *) dstb_raw;
    ਅਚਲ uint32_t OW           = ictx->OW;
    ਅਚਲ uint32_t IC           = ictx->IC;
    ਅਚਲ uint32_t KD           = ictx->KD;
    ਅਚਲ uint32_t KH           = ictx->KH;
    ਅਚਲ uint32_t KW           = ictx->KW;
    ਅਚਲ uint32_t IW           = ictx->IW;
    ਅਚਲ uint32_t patch_stride = ictx->patch_stride;

    ਲਈ (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (size_t) iow * patch_stride;
        ਲਈ (uint32_t iic = 0; iic < IC; iic++) {
            ਲਈ (uint32_t ikd = 0; ikd < KD; ikd++) {
                ਲਈ (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * ਬਾਹਰਲਾ_ਲੜੀਵਾਰ = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ਅਚਲ ਦਸ਼ਮਲਵ * src_ਲੜੀਵਾਰ = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, (ਅਚਲ uint8_t *) src_ਲੜੀਵਾਰ, KW);
                }
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ im2col_3d_patchembed_compute_f32(
    ਅਚਲ ਬਣਤਰ htp_im2col_context * ictx,
    ਖਾਲੀ * ਸੀਮਿਤ dstb_raw, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ srcb) {
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dstb = (ਦਸ਼ਮਲਵ *) dstb_raw;
    ਅਚਲ uint32_t OW           = ictx->OW;
    ਅਚਲ uint32_t IC           = ictx->IC;
    ਅਚਲ uint32_t KD           = ictx->KD;
    ਅਚਲ uint32_t KH           = ictx->KH;
    ਅਚਲ uint32_t KW           = ictx->KW;
    ਅਚਲ uint32_t IW           = ictx->IW;
    ਅਚਲ uint32_t patch_stride = ictx->patch_stride;

    ਲਈ (uint32_t iow = 0; iow < OW; iow++) {
        ਦਸ਼ਮਲਵ * dst_patch = dstb + (size_t) iow * patch_stride;
        ਲਈ (uint32_t iic = 0; iic < IC; iic++) {
            ਲਈ (uint32_t ikd = 0; ikd < KD; ikd++) {
                ਲਈ (uint32_t ikh = 0; ikh < KH; ikh++) {
                    ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ_ਲੜੀਵਾਰ = dst_patch + (size_t) (iic * KD * KH + ikd * KH + ikh) * KW;
                    ਅਚਲ ਦਸ਼ਮਲਵ * src_ਲੜੀਵਾਰ = srcb + (size_t) (iic * KD * KH + ikd * KH + ikh) * IW + (size_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) ਬਾਹਰਲਾ_ਲੜੀਵਾਰ, (ਅਚਲ uint8_t *) src_ਲੜੀਵਾਰ, KW);
                }
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ im2col_3d_patchembed_dma_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ htp_im2col_context * ictx        = (ਬਣਤਰ htp_im2col_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context *    octx        = ictx->octx;
    ਬਣਤਰ htp_thread_trace * ਸੀਮਿਤ tr   = &octx->ctx->trace[ith];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst  = octx->dst;
    ਅਚਲ uint32_t KD           = ictx->KD;
    ਅਚਲ uint32_t OW           = ictx->OW;
    ਅਚਲ uint32_t patch_stride = ictx->patch_stride;
    ਅਚਲ uint32_t dst_elem     = ictx->dst_elem;
    ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ   = dst->ਅੰਕੜੇ;
    dma_ਕਤਾਰ * dma_q           = octx->ctx->dma[ith];
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ_max  = ictx->pe_ਪੰਕਤੀ_ਅਧਾਰ + ictx->pe_nrows;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ    = ictx->pe_ਪੰਕਤੀ_ਅਧਾਰ + ictx->pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ      = MIN(ਪੰਕਤੀ_ਸ਼ੁਰੂ + ictx->pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, ਪੰਕਤੀ_ਸਿਰਾ_max);
    ਜੇ (ਪੰਕਤੀ_ਸ਼ੁਰੂ >= ਪੰਕਤੀ_ਸਿਰਾ) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t nrows_local  = ਪੰਕਤੀ_ਸਿਰਾ - ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint8_t * src_ਅਧਾਰ          = ictx->pe_vtcm_src + ith * ictx->pe_src_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t * dstb_ਅਧਾਰ         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    ਦਸ਼ਮਲਵ * srcb2[2]            = { (ਦਸ਼ਮਲਵ *) src_ਅਧਾਰ, (ਦਸ਼ਮਲਵ *) (src_ਅਧਾਰ + ictx->pe_src_ਪੰਕਤੀ_ਬਾਈਟਾਂ) };
    ਖਾਲੀ * dstb2[2]             = { dstb_ਅਧਾਰ, dstb_ਅਧਾਰ + ictx->pe_dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ };
    ਬਣਤਰ im2col_cursor cur = {
        .r   = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .ਵਿੱਚ  = ictx->thread_ਵਿੱਚ[ith],
        .iod = ictx->thread_iod[ith],
        .ioh = ictx->thread_ioh[ith],
        .c0  = 0,
    };
    ਬਣਤਰ im2col_cursor stage = cur;
    ਲਈ (uint32_t i = 0; i < nrows_local && i < 2; i++) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, dstb2[i]), 0, 0, 0, 0);
        im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb2[i], &stage);
        im2col_3d_ਕਦਮ_ਪੰਕਤੀ(&stage, ictx);
    }

    im2col_3d_patchembed_compute_fn_t compute = (im2col_3d_patchembed_compute_fn_t) ictx->compute;
    ਲਈ (uint32_t idx = 0; idx < nrows_local; idx++) {
        ਅਚਲ uint32_t buf = idx & 1u;
        ਦਸ਼ਮਲਵ * srcb = srcb2[buf];
        ਖਾਲੀ *  dstb = dstb2[buf];
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        ਲਈ (uint32_t p = 0; p < KD; p++) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        }
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        compute(ictx, dstb, srcb);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur.r);
        ਅਚਲ dma_addr_t ddr_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + (size_t) cur.r * OW * patch_stride * dst_elem;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ddr_ਪੰਕਤੀ, dstb),
                       OW * patch_stride * dst_elem, OW * patch_stride * dst_elem,
                       OW * patch_stride * dst_elem, 1);
        ਜੇ (idx + 2 < nrows_local) {
            im2col_3d_patchembed_dma_stage(ictx, dma_q, srcb, &stage);
            im2col_3d_ਕਦਮ_ਪੰਕਤੀ(&stage, ictx);
        }
        im2col_3d_ਕਦਮ_ਪੰਕਤੀ(&cur, ictx);
    }
    dma_ਕਤਾਰ_flush(dma_q);
}

ਪੂਰਨ op_im2col(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_im2col_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_im2col_kernel_params *) octx->kernel_params;
    ਜੇ (kparams->kernel_type == HTP_IM2_ਖਾਨਾ_KERNEL_UNSUPPORTED) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;

    uint32_t ਪੰਕਤੀ_ਅਧਾਰ = 0;
    uint32_t nrows    = kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ = dst->nb[2];
        ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = (ਪੰਕਤੀ_ਮਾਪ > 0) ? (HEX_L2_ਸਤਰ_ਮਾਪ / hex_ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_u32(ਪੰਕਤੀ_ਮਾਪ, HEX_L2_ਸਤਰ_ਮਾਪ)) : 1;
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range =
            htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ, htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) ? ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਅਧਾਰ = range.ਸ਼ੁਰੂ;
        nrows    = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਬਣਤਰ htp_im2col_context ictx = { 0 };
    uint8_t * ਅਚਲ ਅਧਾਰ = octx->ctx->vtcm_ਅਧਾਰ;

    ਅਚਲ uint32_t pth = MIN(octx->n_ਧਾਗੇ, nrows);
    ictx.octx                   = octx;
    ictx.pe_ਪੰਕਤੀ_ਅਧਾਰ            = ਪੰਕਤੀ_ਅਧਾਰ;
    ictx.pe_nrows               = nrows;
    ictx.pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_ਪੰਕਤੀ_ਬਾਈਟਾਂ       = kparams->src_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
    ictx.pe_dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ       = kparams->dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, kparams->off_dst);
    ictx.pe_src_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    ictx.pe_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
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

    ਲਈ (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = ਪੰਕਤੀ_ਅਧਾਰ + ictx.pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
        ictx.thread_ioh[ith] = r % kparams->OH;
        ictx.thread_ਵਿੱਚ[ith]  = r / kparams->OH;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? ਆਕਾਰ(__fp16) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਜੇ (kparams->kernel_type == HTP_IM2_ਖਾਨਾ_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ਖਾਲੀ *) im2col_patchembed_compute_f16
                                                 : (ਖਾਲੀ *) im2col_patchembed_compute_f32;
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, im2col_patchembed_dma_thread, &ictx, pth);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    } ਨਹੀਂਤਾਂ ਜੇ (kparams->kernel_type == HTP_IM2_ਖਾਨਾ_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ਖਾਲੀ *) im2col_blocked_compute_f16
                                                 : (ਖਾਲੀ *) im2col_blocked_compute_f32;
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, im2col_blocked_dma_thread, &ictx, pth);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
}

ਪੂਰਨ op_im2col_3d(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_im2col_kernel_params * kparams =
        (ਅਚਲ ਬਣਤਰ htp_im2col_kernel_params *) octx->kernel_params;
    ਜੇ (kparams->kernel_type == HTP_IM2_ਖਾਨਾ_KERNEL_UNSUPPORTED) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;

    uint32_t ਪੰਕਤੀ_ਅਧਾਰ = 0;
    uint32_t nrows    = kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ = dst->nb[2];
        ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = (ਪੰਕਤੀ_ਮਾਪ > 0) ? (HEX_L2_ਸਤਰ_ਮਾਪ / hex_ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_u32(ਪੰਕਤੀ_ਮਾਪ, HEX_L2_ਸਤਰ_ਮਾਪ)) : 1;
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range =
            htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ, htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) ? ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਅਧਾਰ = range.ਸ਼ੁਰੂ;
        nrows    = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਬਣਤਰ htp_im2col_context ictx = { 0 };
    uint8_t * ਅਚਲ ਅਧਾਰ = octx->ctx->vtcm_ਅਧਾਰ;

    ਅਚਲ uint32_t pth = MIN(octx->n_ਧਾਗੇ, nrows);
    ictx.octx                   = octx;
    ictx.pe_ਪੰਕਤੀ_ਅਧਾਰ            = ਪੰਕਤੀ_ਅਧਾਰ;
    ictx.pe_nrows               = nrows;
    ictx.pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread     = (nrows + pth - 1) / pth;
    ictx.pe_src_ਪੰਕਤੀ_ਬਾਈਟਾਂ       = kparams->src_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
    ictx.pe_dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ       = kparams->dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
    ictx.pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, kparams->off_src);
    ictx.pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, kparams->off_dst);
    ictx.pe_src_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    ictx.pe_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
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

    ਲਈ (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = ਪੰਕਤੀ_ਅਧਾਰ + ictx.pe_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
        ictx.thread_ioh[ith] = rem % kparams->OH;
        rem /= kparams->OH;
        ictx.thread_iod[ith] = rem % kparams->OD;
        ictx.thread_ਵਿੱਚ[ith]  = rem / kparams->OD;
    }

    ictx.dst_elem = dst->type == HTP_TYPE_F16 ? ਆਕਾਰ(__fp16) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਜੇ (kparams->kernel_type == HTP_IM2_ਖਾਨਾ_KERNEL_PATCHEMBED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ਖਾਲੀ *) im2col_3d_patchembed_compute_f16
                                                 : (ਖਾਲੀ *) im2col_3d_patchembed_compute_f32;
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, im2col_3d_patchembed_dma_thread, &ictx, pth);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    } ਨਹੀਂਤਾਂ ਜੇ (kparams->kernel_type == HTP_IM2_ਖਾਨਾ_KERNEL_BLOCKED_DMA) {
        ictx.compute = dst->type == HTP_TYPE_F16 ? (ਖਾਲੀ *) im2col_3d_blocked_compute_f16
                                                 : (ਖਾਲੀ *) im2col_3d_blocked_compute_f32;
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, im2col_3d_blocked_dma_thread, &ictx, pth);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
}
