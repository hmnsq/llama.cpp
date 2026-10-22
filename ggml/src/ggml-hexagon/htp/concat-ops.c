#ಅನ್ತರ್ಭಾವ "concat-ops.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "dma-copy.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "hexagon_protos.h"
#ಅನ್ತರ್ಭಾವ "hexagon_types.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-fence.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx_hexagon_protos.h"
#ಅನ್ತರ್ಭಾವ "gather-rows.h"

#ಅನ್ತರ್ಭಾವ <string.h>

ರಚನೆ htp_concat_context {
    ರಚನೆ htp_ops_context * octx;
    uint8_t * spad0_ತಳ;
    uint8_t * spad1_ತಳ;
    uint32_t  spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t  spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t  ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t  nrows;
    uint32_t  nrows_ಪ್ರತಿ_thread;
    uint32_t  nplanes;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne2;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dma_addr_t concat_plane_addr(ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * t, uint32_t p, ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne2, uint32_t ne2) {
    ಸ್ಥಿರಾಂಕ uint32_t i3 = fastdiv(p, div_ne2);
    ಸ್ಥಿರಾಂಕ uint32_t i2 = p - i3 * ne2;
    ಹಿಂತಿರುಗಿ t->ದತ್ತಾಂಶ + i2 * t->nb[2] + i3 * t->nb[3];
}

ಸ್ಥಿರ ಶೂನ್ಯ concat_2d_f32_transposed(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_concat_context * cctx = (ರಚನೆ htp_concat_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = cctx->octx;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ne0 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t src1_ne0 = src1->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ = cctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + cctx->nrows;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_i = cctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + ith * cctx->nrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_i   = (ಆರಂಭ_i + cctx->nrows_ಪ್ರತಿ_thread < ಅಡ್ಡಸಾಲು_ಕೊನೆ) ? (ಆರಂಭ_i + cctx->nrows_ಪ್ರತಿ_thread) : ಅಡ್ಡಸಾಲು_ಕೊನೆ;
    ಆದರೆ (ಆರಂಭ_i >= ಕೊನೆ_i) ಹಿಂತಿರುಗಿ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_ತಳ = cctx->spad0_ತಳ + ith * cctx->spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t * spad1_ತಳ = cctx->spad1_ತಳ + ith * cctx->spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread;

    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_i = 32;
    ಸ್ಥಿರಾಂಕ uint32_t spad1_stride = ಖಂಡ_i * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ಸ್ಥಿರಾಂಕ uint32_t src1_ne0_padded = hex_ಸುತ್ತು_ಮೇಲೆ(src1_ne0, 32);
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು  = src0_ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, VLEN);
    ಸ್ಥಿರಾಂಕ uint32_t src0_pre        = src0_ಅಡ್ಡಸಾಲು_padded - src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = src0_ಅಡ್ಡಸಾಲು_padded + src1_ne0_padded * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne2 = &cctx->div_ne2;
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = ಆರಂಭ_i;

    ಸ್ಥಿರಾಂಕ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(spad1_ತಳ, src1_addr), spad1_stride, src1->nb[0], MIN(ಕೊನೆ_i - i, ಖಂಡ_i) * ಗಾತ್ರ(ದಶಮಾಂಶ), src1_ne0);

    ಸ್ಥಿರಾಂಕ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(spad0_ತಳ + src0_pre, src0_addr), spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src0->nb[1], src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, MIN(ಕೊನೆ_i - i, ಖಂಡ_i));

    dma_ಸರತಿ_ತೆಗೆ(dma_q); // src1

    ಆಗ (p < cctx->nplanes) {
        ಸ್ಥಿರಾಂಕ uint32_t current_ಖಂಡ_i = MIN(ಕೊನೆ_i - i, ಖಂಡ_i);

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ಗಾಗಿ (uint32_t j = 0; j < src1_ne0; j += 32) {
            ಸ್ಥಿರಾಂಕ uint8_t * src_ptr = spad1_ತಳ + j * spad1_stride;
            uint8_t * dst_ptr = spad0_ತಳ + src0_ಅಡ್ಡಸಾಲು_padded + j * ಗಾತ್ರ(ದಶಮಾಂಶ);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src_ptr, spad1_stride, offsets, current_ಖಂಡ_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_ತಳ + src0_ಅಡ್ಡಸಾಲು_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + ಖಂಡ_i;
        ಆದರೆ (ni >= ಕೊನೆ_i) {
            ni = ಆರಂಭ_i;
            np++;
        }
        ಸ್ಥಿರಾಂಕ bool has_ಮುಂದಿನ = np < cctx->nplanes;
        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಖಂಡ_i = MIN(ಕೊನೆ_i - ni, ಖಂಡ_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ಆದರೆ (has_ಮುಂದಿನ) {
            ಸ್ಥಿರಾಂಕ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(spad1_ತಳ, nsrc1_addr), spad1_stride, src1->nb[0], ಮುಂದಿನ_ಖಂಡ_i * ಗಾತ್ರ(ದಶಮಾಂಶ), src1_ne0);
        }

        dma_ಸರತಿ_ತೆಗೆ(dma_q); // src0

        ಸ್ಥಿರಾಂಕ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_addr, spad0_ತಳ + src0_pre), dst->nb[1], spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, (src0_ne0 + src1_ne0) * ಗಾತ್ರ(ದಶಮಾಂಶ), current_ಖಂಡ_i);

        ಆದರೆ (has_ಮುಂದಿನ) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q); // next src1
        }
        dma_ಸರತಿ_ತೆಗೆ(dma_q); // dst

        // spad0 is free after the dst write
        ಆದರೆ (has_ಮುಂದಿನ) {
            ಸ್ಥಿರಾಂಕ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(spad0_ತಳ + src0_pre, nsrc0_addr), spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src0->nb[1], src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ಮುಂದಿನ_ಖಂಡ_i);
        }

        p = np;
        i = ni;
    }
    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಶೂನ್ಯ concat_2d_f16_transposed(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_concat_context * cctx = (ರಚನೆ htp_concat_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = cctx->octx;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ne0 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t src1_ne0 = src1->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ = cctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + cctx->nrows;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_i = cctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + ith * cctx->nrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_i   = (ಆರಂಭ_i + cctx->nrows_ಪ್ರತಿ_thread < ಅಡ್ಡಸಾಲು_ಕೊನೆ) ? (ಆರಂಭ_i + cctx->nrows_ಪ್ರತಿ_thread) : ಅಡ್ಡಸಾಲು_ಕೊನೆ;
    ಆದರೆ (ಆರಂಭ_i >= ಕೊನೆ_i) ಹಿಂತಿರುಗಿ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_ತಳ = cctx->spad0_ತಳ + ith * cctx->spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint8_t * spad1_ತಳ = cctx->spad1_ತಳ + ith * cctx->spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread;

    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_i = 64;
    ಸ್ಥಿರಾಂಕ uint32_t spad1_stride = ಖಂಡ_i * ಗಾತ್ರ(__fp16);

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ಸ್ಥಿರಾಂಕ uint32_t src1_ne0_padded = hex_ಸುತ್ತು_ಮೇಲೆ(src1_ne0, 64);
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು  = src0_ne0 * ಗಾತ್ರ(__fp16);
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, VLEN);
    ಸ್ಥಿರಾಂಕ uint32_t src0_pre        = src0_ಅಡ್ಡಸಾಲು_padded - src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = src0_ಅಡ್ಡಸಾಲು_padded + src1_ne0_padded * ಗಾತ್ರ(__fp16);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne2 = &cctx->div_ne2;
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = ಆರಂಭ_i;

    ಸ್ಥಿರಾಂಕ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(spad1_ತಳ, src1_addr), spad1_stride, src1->nb[0], MIN(ಕೊನೆ_i - i, ಖಂಡ_i) * ಗಾತ್ರ(__fp16), src1_ne0);

    ಸ್ಥಿರಾಂಕ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(spad0_ತಳ + src0_pre, src0_addr), spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src0->nb[1], src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, MIN(ಕೊನೆ_i - i, ಖಂಡ_i));

    dma_ಸರತಿ_ತೆಗೆ(dma_q); // src1

    ಆಗ (p < cctx->nplanes) {
        ಸ್ಥಿರಾಂಕ uint32_t current_ಖಂಡ_i = MIN(ಕೊನೆ_i - i, ಖಂಡ_i);

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ಗಾಗಿ (uint32_t j = 0; j < src1_ne0; j += 64) {
            ಸ್ಥಿರಾಂಕ uint8_t * src_ptr = spad1_ತಳ + j * spad1_stride;
            uint8_t * dst_ptr = spad0_ತಳ + src0_ಅಡ್ಡಸಾಲು_padded + j * ಗಾತ್ರ(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src_ptr, spad1_stride, offsets, current_ಖಂಡ_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_ತಳ + src0_ಅಡ್ಡಸಾಲು_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + ಖಂಡ_i;
        ಆದರೆ (ni >= ಕೊನೆ_i) {
            ni = ಆರಂಭ_i;
            np++;
        }
        ಸ್ಥಿರಾಂಕ bool has_ಮುಂದಿನ = np < cctx->nplanes;
        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಖಂಡ_i = MIN(ಕೊನೆ_i - ni, ಖಂಡ_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ಆದರೆ (has_ಮುಂದಿನ) {
            ಸ್ಥಿರಾಂಕ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(spad1_ತಳ, nsrc1_addr), spad1_stride, src1->nb[0], ಮುಂದಿನ_ಖಂಡ_i * ಗಾತ್ರ(__fp16), src1_ne0);
        }

        dma_ಸರತಿ_ತೆಗೆ(dma_q); // src0

        ಸ್ಥಿರಾಂಕ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_addr, spad0_ತಳ + src0_pre), dst->nb[1], spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, (src0_ne0 + src1_ne0) * ಗಾತ್ರ(__fp16), current_ಖಂಡ_i);

        ಆದರೆ (has_ಮುಂದಿನ) {
            dma_ಸರತಿ_ತೆಗೆ(dma_q); // next src1
        }
        dma_ಸರತಿ_ತೆಗೆ(dma_q); // dst

        // spad0 is free after the dst write
        ಆದರೆ (has_ಮುಂದಿನ) {
            ಸ್ಥಿರಾಂಕ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(spad0_ತಳ + src0_pre, nsrc0_addr), spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src0->nb[1], src0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ಮುಂದಿನ_ಖಂಡ_i);
        }

        p = np;
        i = ni;
    }
    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ concat_regular(ರಚನೆ htp_ops_context * octx, ಪೂರ್ಣಾಂಕ dim, uint32_t type_ಗಾತ್ರ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ರಚನೆ htp_tensor view0 = *dst;
    ರಚನೆ htp_tensor view1 = *dst;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.ದತ್ತಾಂಶ += src0->ne[dim] * dst->nb[dim];

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_0;
    uint32_t rstart1 = 0, nrows1 = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_1;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range0 = htp_tensor_mdev_ವಿಭಜನೆ(
            ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        rstart0 = range0.ಆರಂಭ;
        nrows0  = range0.ಎಣಿಕೆ_2;

        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range1 = htp_tensor_mdev_ವಿಭಜನೆ(
            ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        rstart1 = range1.ಆರಂಭ;
        nrows1  = range1.ಎಣಿಕೆ_2;
    }

    dma_ಸರತಿ * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_ಗಾತ್ರ, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_ಗಾತ್ರ, rstart1, nrows1);
    dma_ಸರತಿ_flush(q);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ concat_transposed(ರಚನೆ htp_ops_context * octx, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_concat_kernel_params * kparams, uint32_t type_ಗಾತ್ರ) {
    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = dst->ne[1];
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (kparams->vtcm_ಗಾತ್ರ > octx->ctx->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;

    ರಚನೆ htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_ತಳ            = vtcm_ತಳ;
    cctx.spad1_ತಳ            = vtcm_ತಳ + n_ಎಳೆಗಳು * kparams->u.transposed.spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    cctx.spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->u.transposed.spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    cctx.spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->u.transposed.spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    cctx.ಅಡ್ಡಸಾಲು_ಆರಂಭ             = ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[2]);
    cctx.nrows_ಪ್ರತಿ_thread      = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);

    work_ಸರತಿ_func_t ಕೆಲಸಗಾರ_func = (type_ಗಾತ್ರ == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, ಕೆಲಸಗಾರ_func, &cctx, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_concat(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_concat_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_concat_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = HTP_ಸ್ಥಿತಿ_OK;
    ಸ್ಥಿರಾಂಕ uint32_t type_ಗಾತ್ರ = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    ಆಯ್ಕೆ (kparams->kernel_type) {
        ಸ್ಥಿತಿ HTP_CONCAT_KERNEL_REGULAR:
            ಸ್ಥಿತಿ_2 = concat_regular(octx, kparams->dim, type_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_CONCAT_KERNEL_TRANSPOSED:
            ಸ್ಥಿತಿ_2 = concat_transposed(octx, kparams, type_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;

        ಸ್ಥಿತಿ HTP_CONCAT_KERNEL_GATHER_ಅಡ್ಡಸಾಲುಗಳು:
            ಸ್ಥಿತಿ_2 = htp_gather_ಅಡ್ಡಸಾಲುಗಳು(octx, src0, src1, &kparams->u.gather, kparams->n_ಎಳೆಗಳು);
            ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಸ್ಥಿತಿ_2 = HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            ನಿಲ್ಲಿಸು;
    }

    htp_ops_context_ಗಣ_ಸ್ಥಿತಿ(octx, ಸ್ಥಿತಿ_2);

    ಹಿಂತಿರುಗಿ octx->ಸ್ಥಿತಿ_2;
}
