#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hvx-types.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"

#ವ್ಯಾಖ್ಯಾನಿಸು htp_cumsum_tensors_preamble                         \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0]; \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst  = octx->dst;    \
                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1];                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];                      \
                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];                        \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];                        \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];                        \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];                        \
                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t nb00 = src0->nb[0];                      \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1];                      \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];                      \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t nb0 = dst->nb[0];                        \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];                        \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];                        \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

ರಚನೆ htp_cumsum_context {
    ರಚನೆ htp_ops_context * octx;
    size_t          src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t          dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t          src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t          dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t        ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
    uint32_t        ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t        ಅಡ್ಡಸಾಲು_ಆರಂಭ;
};

#ವ್ಯಾಖ್ಯಾನಿಸು htp_cumsum_preamble                                                \
    ರಚನೆ htp_cumsum_context * cctx = (ರಚನೆ htp_cumsum_context *) ದತ್ತಾಂಶ; \
    ರಚನೆ htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#ಆದರೆ __HVX_ARCH__ > 75
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_cumsum_vadd(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b) {
    ಹಿಂತಿರುಗಿ Q6_Vsf_vadd_VsfVsf(a, b);
}
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_cumsum_vadd(HVX_ಸದಿಶ a, HVX_ಸದಿಶ b) {
    ಹಿಂತಿರುಗಿ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#ಆದರೆ_ಅಂತ್ಯ  // __HVX_ARCH__ > 75

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_ಪೂರ್ವಪ್ರತ್ಯಯ_ಪರಿಶೀಲನೆ_f32(HVX_ಸದಿಶ v, HVX_ಸದಿಶ ವಾಹಕ_ಒಳ) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, ವಾಹಕ_ಒಳ);

    ಹಿಂತಿರುಗಿ v;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_splat_ಕೊನೆಯ_f32(HVX_ಸದಿಶ v) {
    ಹಿಂತಿರುಗಿ hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_cumsum_ಅಡ್ಡಸಾಲು_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src, ದಶಮಾಂಶ * ಸೀಮಿತ dst, uint32_t n) {
    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP32;

    HVX_ಸದಿಶ ವಾಹಕ = Q6_V_vsplat_R(0);

    ಗಾಗಿ (uint32_t i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v = *((ಸ್ಥಿರಾಂಕ HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_ಪೂರ್ವಪ್ರತ್ಯಯ_ಪರಿಶೀಲನೆ_f32(v, ವಾಹಕ);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        ವಾಹಕ = hvx_splat_ಕೊನೆಯ_f32(v);
    }

    ಆದರೆ (nloe) {
        ದಶಮಾಂಶ acc = hvx_vec_get_f32(ವಾಹಕ);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಬಾಲ = src + nvec * VLEN_FP32;
        ದಶಮಾಂಶ       * dst_ಬಾಲ = dst + nvec * VLEN_FP32;
        ಗಾಗಿ (uint32_t i = 0; i < nloe; i++) {
            acc        += src_ಬಾಲ[i];
            dst_ಬಾಲ[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ cumsum_thread_f32_dma(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_cumsum_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t ir0 = cctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + cctx->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + cctx->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, cctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + cctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);

    ಆದರೆ (ir0 >= ir1) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = cctx->src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = cctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = cctx->src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = cctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    ಸ್ಥಿರಾಂಕ dma_addr_t src_ದತ್ತಾಂಶ = src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ = dst->ದತ್ತಾಂಶ;

    uint8_t * src_spad = octx->src0_spad.ದತ್ತಾಂಶ + (ith * src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.ದತ್ತಾಂಶ  + (ith * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * 2);

    ಗಾಗಿ (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, dst_spad + (spad_idx * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned)),
                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(src_spad + (spad_idx * src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned),
                                     src_ದತ್ತಾಂಶ + (ir * src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ)),
                       src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಗಾಗಿ (uint32_t ir = ir0; ir < ir1; ir++) {
        ದಶಮಾಂಶ * dst_spad_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        ದಶಮಾಂಶ * src_spad_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_ಅಡ್ಡಸಾಲು_f32(src_spad_ಅಡ್ಡಸಾಲು, dst_spad_ಅಡ್ಡಸಾಲು, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ), dst_spad_ಅಡ್ಡಸಾಲು),
                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಅಡ್ಡಸಾಲು = ir + 2;
        ಆದರೆ (ಮುಂದಿನ_ಅಡ್ಡಸಾಲು < ir1) {
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                           dma_make_ದತ್ತಾಂಶ(src_spad_ಅಡ್ಡಸಾಲು, src_ದತ್ತಾಂಶ + (ಮುಂದಿನ_ಅಡ್ಡಸಾಲು * src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ)),
                           src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
        }
    }

    dma_ಸರತಿ_flush(dma_q);

    FARF(ಮೇಲಿನ_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ cumsum_thread_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_cumsum_preamble;

    ಸ್ಥಿರಾಂಕ uint8_t * src_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ uint8_t *) src0->ದತ್ತಾಂಶ;
    uint8_t *       dst_ದತ್ತಾಂಶ = (uint8_t *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ uint32_t ir0 = cctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + cctx->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + cctx->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread, cctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + cctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    ಗಾಗಿ (uint32_t ir = ir0; ir < ir1; ir++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (src_ದತ್ತಾಂಶ + ir * cctx->src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
        ದಶಮಾಂಶ       * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *)       (dst_ದತ್ತಾಂಶ + ir * cctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
        hvx_cumsum_ಅಡ್ಡಸಾಲು_f32(src_ಅಡ್ಡಸಾಲು, dst_ಅಡ್ಡಸಾಲು, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(ಮೇಲಿನ_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

ಪೂರ್ಣಾಂಕ op_cumsum_f32(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ size_t dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, ಗಾತ್ರ(ದಶಮಾಂಶ), (uint32_t) dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = dst->nb[1];
    ಸ್ಥಿರಾಂಕ size_t src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ಸ್ಥಿರಾಂಕ size_t spad_ಪ್ರತಿ_thread = 2 * (src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread = src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * 2;
    octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * 2;

    octx->src0_spad.ಗಾತ್ರ_2  = n_ಎಳೆಗಳು * octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;
    octx->dst_spad.ಗಾತ್ರ_2   = n_ಎಳೆಗಳು * octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;

    octx->src0_spad.ದತ್ತಾಂಶ  = octx->ctx->vtcm_ತಳ;                        octx->src0_spad.src = ಖಾಲಿ;
    octx->dst_spad.ದತ್ತಾಂಶ   = octx->src0_spad.ದತ್ತಾಂಶ + octx->src0_spad.ಗಾತ್ರ_2; octx->dst_spad.src  = ಖಾಲಿ;

    ರಚನೆ htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
        .dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
        .src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
        .dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
        .ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread      = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
        .ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು           = nrows,
        .ಅಡ್ಡಸಾಲು_ಆರಂಭ            = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
    };

    ಆದರೆ (octx->ctx->vtcm_ಗಾತ್ರ < spad_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು) {
        ಆದರೆ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, cumsum_thread_f32, &cctx, n_ಎಳೆಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, cumsum_thread_f32_dma, &cctx, n_ಎಳೆಗಳು);
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_cumsum(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;

    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:
            ಹಿಂತಿರುಗಿ op_cumsum_f32(octx);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
}
