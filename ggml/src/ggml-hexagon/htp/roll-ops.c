#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

ರಚನೆ htp_roll_context {
    ರಚನೆ htp_ops_context * octx;

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t nrows;
    uint32_t nrows_ಪ್ರತಿ_thread;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne1;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne2_ne1;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t htp_roll_wrap(int32_t i, uint32_t ne) {
    ಆದರೆ (i < 0) {
        ಹಿಂತಿರುಗಿ (uint32_t) (i + (int32_t) ne);
    }
    ಆದರೆ ((uint32_t) i >= ne) {
        ಹಿಂತಿರುಗಿ (uint32_t) i - ne;
    }
    ಹಿಂತಿರುಗಿ (uint32_t) i;
}

#ವ್ಯಾಖ್ಯಾನಿಸು htp_roll_preamble                             \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];    \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;       \
                                                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];                  \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];                  \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];                  \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];                  \
                                                      \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1];                \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];                \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];                \
                                                      \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];                  \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];                  \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];                  \
                                                      \
    ಸ್ಥಿರಾಂಕ int32_t s0 = octx->op_params[0];            \
    ಸ್ಥಿರಾಂಕ int32_t s1 = octx->op_params[1];            \
    ಸ್ಥಿರಾಂಕ int32_t s2 = octx->op_params[2];            \
    ಸ್ಥಿರಾಂಕ int32_t s3 = octx->op_params[3];            \
                                                      \
    ಸ್ಥಿರಾಂಕ uint32_t i0_src0 = htp_roll_wrap(-s0, ne0); \
    ಸ್ಥಿರಾಂಕ uint32_t n0      = ne0 - i0_src0;

#ವ್ಯಾಖ್ಯಾನಿಸು htp_roll_dma_preamble dma_ಸರತಿ * q = octx->ctx->dma[0];

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ roll_dma_ತಳ್ಳು(dma_ಸರತಿ * q,
                                 dma_addr_t  dst,
                                 dma_addr_t  src,
                                 uint32_t    dst_stride,
                                 uint32_t    src_stride,
                                 uint32_t    ಬೈಟುಗಳು,
                                 uint32_t    nrows) {
    ಆದರೆ (ಬೈಟುಗಳು == 0 || nrows == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (!dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(dst, src), dst_stride, src_stride, ಬೈಟುಗಳು, nrows)) {
        dma_ಸರತಿ_flush(q);
        dma_ಸರತಿ_ತಳ್ಳು(q, dma_make_ದತ್ತಾಂಶ(dst, src),
                       dst_stride, src_stride, ಬೈಟುಗಳು, nrows);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ roll_dma_ತಳ್ಳು_ಅಡ್ಡಸಾಲುಗಳು(dma_ಸರತಿ *               q,
                                      ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst,
                                      ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0,
                                      uint32_t                  dst_ಅಡ್ಡಸಾಲು,
                                      uint32_t                  src_ಅಡ್ಡಸಾಲು,
                                      uint32_t                  nrows,
                                      uint32_t                  ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
                                      uint32_t                  i0_src0) {
    ಸ್ಥಿರಾಂಕ dma_addr_t dst_ತಳ = dst->ದತ್ತಾಂಶ + (size_t) dst_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ dma_addr_t src_ತಳ = src0->ದತ್ತಾಂಶ + (size_t) src_ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t   n0       = src0->ne[0] - i0_src0;

    roll_dma_ತಳ್ಳು(q, dst_ತಳ, src_ತಳ + (size_t) i0_src0 * ಗಾತ್ರ(ದಶಮಾಂಶ),
                  ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, n0 * ಗಾತ್ರ(ದಶಮಾಂಶ), nrows);
    roll_dma_ತಳ್ಳು(q, dst_ತಳ + (size_t) n0 * ಗಾತ್ರ(ದಶಮಾಂಶ), src_ತಳ,
                  ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, i0_src0 * ಗಾತ್ರ(ದಶಮಾಂಶ), nrows);
}

// Same row-wrap split as roll_dma_push_rows, but addressed with explicit byte strides so it
// also works for a src0 that is row-contiguous only (e.g. a permuted view) rather than fully packed.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ roll_dma_ತಳ್ಳು_range(dma_ಸರತಿ * q,
                                       dma_addr_t  dst_ಅಡ್ಡಸಾಲು,
                                       dma_addr_t  src_ಅಡ್ಡಸಾಲು,
                                       uint32_t    dst_stride,
                                       uint32_t    src_stride,
                                       uint32_t    nrows,
                                       uint32_t    i0_src0,
                                       uint32_t    n0) {
    roll_dma_ತಳ್ಳು(q, dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು + (size_t) i0_src0 * ಗಾತ್ರ(ದಶಮಾಂಶ),
                  dst_stride, src_stride, n0 * ಗಾತ್ರ(ದಶಮಾಂಶ), nrows);
    roll_dma_ತಳ್ಳು(q, dst_ಅಡ್ಡಸಾಲು + (size_t) n0 * ಗಾತ್ರ(ದಶಮಾಂಶ), src_ಅಡ್ಡಸಾಲು,
                  dst_stride, src_stride, i0_src0 * ಗಾತ್ರ(ದಶಮಾಂಶ), nrows);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ roll_dma_f32_contiguous(ರಚನೆ htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಆದರೆ (s1 == 0 && s2 == 0 && s3 == 0) {
        roll_dma_ತಳ್ಳು_ಅಡ್ಡಸಾಲುಗಳು(q, dst, src0, 0, 0, ne1 * ne2 * ne3, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, i0_src0);
        dma_ಸರತಿ_flush(q);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (s1 == 0) {
        ಸ್ಥಿರಾಂಕ uint32_t i2_src0 = htp_roll_wrap(-s2, ne2);
        ಗಾಗಿ (uint32_t i3 = 0; i3 < ne3; i3++) {
            ಸ್ಥಿರಾಂಕ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
            ಸ್ಥಿರಾಂಕ uint32_t dst_row0 = i3 * ne2 * ne1;
            ಸ್ಥಿರಾಂಕ uint32_t src_row0 = (i03 * ne2 + i2_src0) * ne1;
            ಸ್ಥಿರಾಂಕ uint32_t n2_ಮೊದಲ = ne2 - i2_src0;

            roll_dma_ತಳ್ಳು_ಅಡ್ಡಸಾಲುಗಳು(q, dst, src0, dst_row0, src_row0, n2_ಮೊದಲ * ne1,
                               ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, i0_src0);
            roll_dma_ತಳ್ಳು_ಅಡ್ಡಸಾಲುಗಳು(q, dst, src0, dst_row0 + n2_ಮೊದಲ * ne1, i03 * ne2 * ne1,
                               i2_src0 * ne1, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, i0_src0);
        }

        dma_ಸರತಿ_flush(q);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t i1_src0 = htp_roll_wrap(-s1, ne1);
    ಸ್ಥಿರಾಂಕ uint32_t n1_ಮೊದಲ = ne1 - i1_src0;

    ಗಾಗಿ (uint32_t i3 = 0; i3 < ne3; i3++) {
        ಸ್ಥಿರಾಂಕ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        ಗಾಗಿ (uint32_t i2 = 0; i2 < ne2; i2++) {
            ಸ್ಥಿರಾಂಕ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
            ಸ್ಥಿರಾಂಕ uint32_t dst_row0 = (i3 * ne2 + i2) * ne1;
            ಸ್ಥಿರಾಂಕ uint32_t src_row0 = (i03 * ne2 + i02) * ne1;

            roll_dma_ತಳ್ಳು_ಅಡ್ಡಸಾಲುಗಳು(q, dst, src0, dst_row0, src_row0 + i1_src0,
                               n1_ಮೊದಲ, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, i0_src0);
            roll_dma_ತಳ್ಳು_ಅಡ್ಡಸಾಲುಗಳು(q, dst, src0, dst_row0 + n1_ಮೊದಲ, src_row0,
                               i1_src0, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, i0_src0);
        }
    }

    dma_ಸರತಿ_flush(q);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

// DMA path for a row-contiguous but otherwise arbitrarily strided src0 (e.g. a permuted view).
// Same row-wrap split as above, one DMA push per (i2,i3), addressed via the real nb01/nb02/nb03
// instead of assuming a packed layout.
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ roll_dma_f32_strided(ರಚನೆ htp_ops_context * octx) {
    htp_roll_preamble;
    htp_roll_dma_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t i1_src0  = htp_roll_wrap(-s1, ne1);
    ಸ್ಥಿರಾಂಕ uint32_t n1_ಮೊದಲ = ne1 - i1_src0;

    ಗಾಗಿ (uint32_t i3 = 0; i3 < ne3; i3++) {
        ಸ್ಥಿರಾಂಕ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);
        ಗಾಗಿ (uint32_t i2 = 0; i2 < ne2; i2++) {
            ಸ್ಥಿರಾಂಕ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);

            ಸ್ಥಿರಾಂಕ dma_addr_t dst_row0 = dst->ದತ್ತಾಂಶ  + (size_t) i2  * nb2  + (size_t) i3  * nb3;
            ಸ್ಥಿರಾಂಕ dma_addr_t src_row0 = src0->ದತ್ತಾಂಶ + (size_t) i02 * nb02 + (size_t) i03 * nb03;

            roll_dma_ತಳ್ಳು_range(q, dst_row0, src_row0 + (size_t) i1_src0 * nb01,
                                nb1, nb01, n1_ಮೊದಲ, i0_src0, n0);
            roll_dma_ತಳ್ಳು_range(q, dst_row0 + (size_t) n1_ಮೊದಲ * nb1, src_row0,
                                nb1, nb01, i1_src0, i0_src0, n0);
        }
    }

    dma_ಸರತಿ_flush(q);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಶೂನ್ಯ roll_thread_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_roll_context * rctx = (ರಚನೆ htp_roll_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = rctx->octx;

    htp_roll_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = rctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + rctx->nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ   = MIN(ಅಡ್ಡಸಾಲು_ಆರಂಭ + rctx->nrows_ಪ್ರತಿ_thread, rctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + rctx->nrows);
    ಆದರೆ (ಅಡ್ಡಸಾಲು_ಆರಂಭ >= ಅಡ್ಡಸಾಲು_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ಅಡ್ಡಸಾಲು_ಆರಂಭ);

    ಗಾಗಿ (uint32_t ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ಅಡ್ಡಸಾಲು++) {
        ಸ್ಥಿರಾಂಕ uint32_t i3  = fastdiv(ಅಡ್ಡಸಾಲು, &rctx->div_ne2_ne1);
        ಸ್ಥಿರಾಂಕ uint32_t rem = ಅಡ್ಡಸಾಲು - i3 * ne2 * ne1;
        ಸ್ಥಿರಾಂಕ uint32_t i2  = fastdiv(rem, &rctx->div_ne1);
        ಸ್ಥಿರಾಂಕ uint32_t i1  = rem - i2 * ne1;

        ಸ್ಥಿರಾಂಕ uint32_t i01 = htp_roll_wrap((int32_t) i1 - s1, ne1);
        ಸ್ಥಿರಾಂಕ uint32_t i02 = htp_roll_wrap((int32_t) i2 - s2, ne2);
        ಸ್ಥಿರಾಂಕ uint32_t i03 = htp_roll_wrap((int32_t) i3 - s3, ne3);

        ಸ್ಥಿರಾಂಕ uint8_t * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) src0->ದತ್ತಾಂಶ + i01*nb01 + i02*nb02 + i03*nb03;
        uint8_t * dst_ಅಡ್ಡಸಾಲು = (uint8_t *) (uintptr_t) dst->ದತ್ತಾಂಶ + i1*nb1 + i2*nb2 + i3*nb3;

        hex_l2fetch(src_ಅಡ್ಡಸಾಲು + i0_src0 * ಗಾತ್ರ(ದಶಮಾಂಶ), n0 * ಗಾತ್ರ(ದಶಮಾಂಶ), ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ), 1);
        hvx_copy_uu(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು + i0_src0 * ಗಾತ್ರ(ದಶಮಾಂಶ), n0, ಗಾತ್ರ(ದಶಮಾಂಶ));

        ಆದರೆ (i0_src0 != 0) {
            hex_l2fetch(src_ಅಡ್ಡಸಾಲು, i0_src0 * ಗಾತ್ರ(ದಶಮಾಂಶ), ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ), 1);
            hvx_copy_uu(dst_ಅಡ್ಡಸಾಲು + n0 * ಗಾತ್ರ(ದಶಮಾಂಶ), src_ಅಡ್ಡಸಾಲು, i0_src0, ಗಾತ್ರ(ದಶಮಾಂಶ));
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ಅಡ್ಡಸಾಲು_ಆರಂಭ);

    FARF(ಮೇಲಿನ_2, "roll %d/%d: (%ux%ux%ux%u) rows %u:%u shift=(%d,%d,%d,%d)\n",
         ith, nth, ne0, ne1, ne2, ne3,
         ಅಡ್ಡಸಾಲು_ಆರಂಭ, ಅಡ್ಡಸಾಲು_ಕೊನೆ, s0, s1, s2, s3);
}

ಪೂರ್ಣಾಂಕ execute_op_roll_f32(ರಚನೆ htp_ops_context * octx) {
    htp_roll_preamble;

    ಆದರೆ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "roll: unsupported type %u -> %u\n", src0->type, dst->type);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (src0->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ) || dst->nb[0] != ಗಾತ್ರ(ದಶಮಾಂಶ)) {
        FARF(ERROR, "roll: unsupported nb0 %u -> %u\n", src0->nb[0], dst->nb[0]);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (src0->ne[0] != ne0 || src0->ne[1] != ne1 ||
        src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        FARF(ERROR, "roll: shape mismatch\n");
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ne1 * ne2 * ne3;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ);

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, ಗಾತ್ರ(ದಶಮಾಂಶ), (uint32_t) dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 <= 1) {
        ಆದರೆ (htp_tensor_is_contiguous(src0, ಗಾತ್ರ(ದಶಮಾಂಶ)) && htp_tensor_is_contiguous(dst, ಗಾತ್ರ(ದಶಮಾಂಶ))) {
            ಹಿಂತಿರುಗಿ roll_dma_f32_contiguous(octx);
        }
        ಹಿಂತಿರುಗಿ roll_dma_f32_strided(octx);
    }

    ಆದರೆ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;
    ರಚನೆ htp_roll_context rctx = {
        .octx             = octx,
        .ಅಡ್ಡಸಾಲು_ಆರಂಭ        = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .nrows            = nrows,
        .nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
        .div_ne1          = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[1]),
        .div_ne2_ne1      = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[2] * dst->ne[1]),
    };

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, roll_thread_f32, &rctx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_roll(ರಚನೆ htp_ops_context * octx) {
    ಆಯ್ಕೆ (octx->src[0]->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:
            ಹಿಂತಿರುಗಿ execute_op_roll_f32(octx);

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
}
