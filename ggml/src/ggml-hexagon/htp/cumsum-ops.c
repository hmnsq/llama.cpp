#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-types.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_cumsum_tensors_preamble                         \
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0]; \
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst  = octx->dst;    \
                                                            \
    ਅਚਲ uint32_t ne00 = src0->ne[0];                      \
    ਅਚਲ uint32_t ne01 = src0->ne[1];                      \
    ਅਚਲ uint32_t ne02 = src0->ne[2];                      \
    ਅਚਲ uint32_t ne03 = src0->ne[3];                      \
                                                            \
    ਅਚਲ uint32_t ne0 = dst->ne[0];                        \
    ਅਚਲ uint32_t ne1 = dst->ne[1];                        \
    ਅਚਲ uint32_t ne2 = dst->ne[2];                        \
    ਅਚਲ uint32_t ne3 = dst->ne[3];                        \
                                                            \
    ਅਚਲ uint32_t nb00 = src0->nb[0];                      \
    ਅਚਲ uint32_t nb01 = src0->nb[1];                      \
    ਅਚਲ uint32_t nb02 = src0->nb[2];                      \
    ਅਚਲ uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ਅਚਲ uint32_t nb0 = dst->nb[0];                        \
    ਅਚਲ uint32_t nb1 = dst->nb[1];                        \
    ਅਚਲ uint32_t nb2 = dst->nb[2];                        \
    ਅਚਲ uint32_t nb3 = dst->nb[3];

ਬਣਤਰ htp_cumsum_context {
    ਬਣਤਰ htp_ops_context * octx;
    size_t          src_ਪੰਕਤੀ_ਮਾਪ;
    size_t          dst_ਪੰਕਤੀ_ਮਾਪ;
    size_t          src_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t          dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t        ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
    uint32_t        ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    uint32_t        ਪੰਕਤੀ_ਸ਼ੁਰੂ;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_cumsum_preamble                                                \
    ਬਣਤਰ htp_cumsum_context * cctx = (ਬਣਤਰ htp_cumsum_context *) ਅੰਕੜੇ; \
    ਬਣਤਰ htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#ਜੇ __HVX_ARCH__ > 75
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_cumsum_vadd(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b) {
    ਵਾਪਸ Q6_Vsf_vadd_VsfVsf(a, b);
}
#ਨਹੀਂਤਾਂ
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_cumsum_vadd(HVX_ਸਦਿਸ਼ a, HVX_ਸਦਿਸ਼ b) {
    ਵਾਪਸ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#ਜੇ_ਅੰਤ  // __HVX_ARCH__ > 75

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_ਅਗੇਤਰ_ਜਾਂਚ_f32(HVX_ਸਦਿਸ਼ v, HVX_ਸਦਿਸ਼ ਹਾਸਲ_ਵਿੱਚ) {
    ਅਚਲ HVX_ਸਦਿਸ਼ zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, ਹਾਸਲ_ਵਿੱਚ);

    ਵਾਪਸ v;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_splat_ਆਖਰੀ_f32(HVX_ਸਦਿਸ਼ v) {
    ਵਾਪਸ hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_cumsum_ਪੰਕਤੀ_f32(ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst, uint32_t n) {
    ਅਚਲ uint32_t nvec = n / VLEN_FP32;
    ਅਚਲ uint32_t nloe = n % VLEN_FP32;

    HVX_ਸਦਿਸ਼ ਹਾਸਲ = Q6_V_vsplat_R(0);

    ਲਈ (uint32_t i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v = *((ਅਚਲ HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_ਅਗੇਤਰ_ਜਾਂਚ_f32(v, ਹਾਸਲ);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        ਹਾਸਲ = hvx_splat_ਆਖਰੀ_f32(v);
    }

    ਜੇ (nloe) {
        ਦਸ਼ਮਲਵ acc = hvx_vec_get_f32(ਹਾਸਲ);
        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੂਛ = src + nvec * VLEN_FP32;
        ਦਸ਼ਮਲਵ       * dst_ਪੂਛ = dst + nvec * VLEN_FP32;
        ਲਈ (uint32_t i = 0; i < nloe; i++) {
            acc        += src_ਪੂਛ[i];
            dst_ਪੂਛ[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

ਸਥਿਰ ਖਾਲੀ cumsum_thread_f32_dma(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_cumsum_preamble;

    ਅਚਲ uint32_t ir0 = cctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + cctx->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ir1 = MIN(ir0 + cctx->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, cctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + cctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ);

    ਜੇ (ir0 >= ir1) {
        ਵਾਪਸ;
    }

    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ         = cctx->src_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ         = cctx->dst_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ_aligned = cctx->src_ਪੰਕਤੀ_ਮਾਪ_aligned;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned = cctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    ਅਚਲ dma_addr_t src_ਅੰਕੜੇ = src0->ਅੰਕੜੇ;
    ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ = dst->ਅੰਕੜੇ;

    uint8_t * src_spad = octx->src0_spad.ਅੰਕੜੇ + (ith * src_ਪੰਕਤੀ_ਮਾਪ_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.ਅੰਕੜੇ  + (ith * dst_ਪੰਕਤੀ_ਮਾਪ_aligned * 2);

    ਲਈ (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, dst_spad + (spad_idx * dst_ਪੰਕਤੀ_ਮਾਪ_aligned)),
                       dst_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 0);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(src_spad + (spad_idx * src_ਪੰਕਤੀ_ਮਾਪ_aligned),
                                     src_ਅੰਕੜੇ + (ir * src_ਪੰਕਤੀ_ਮਾਪ)),
                       src_ਪੰਕਤੀ_ਮਾਪ_aligned, src_ਪੰਕਤੀ_ਮਾਪ, src_ਪੰਕਤੀ_ਮਾਪ, 1);
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਲਈ (uint32_t ir = ir0; ir < ir1; ir++) {
        ਦਸ਼ਮਲਵ * dst_spad_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        ਦਸ਼ਮਲਵ * src_spad_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_ਪੰਕਤੀ_f32(src_spad_ਪੰਕਤੀ, dst_spad_ਪੰਕਤੀ, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ + (ir * dst_ਪੰਕਤੀ_ਮਾਪ), dst_spad_ਪੰਕਤੀ),
                       dst_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 1);

        ਅਚਲ uint32_t ਅਗਲਾ_ਪੰਕਤੀ = ir + 2;
        ਜੇ (ਅਗਲਾ_ਪੰਕਤੀ < ir1) {
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                           dma_make_ਅੰਕੜੇ(src_spad_ਪੰਕਤੀ, src_ਅੰਕੜੇ + (ਅਗਲਾ_ਪੰਕਤੀ * src_ਪੰਕਤੀ_ਮਾਪ)),
                           src_ਪੰਕਤੀ_ਮਾਪ_aligned, src_ਪੰਕਤੀ_ਮਾਪ, src_ਪੰਕਤੀ_ਮਾਪ, 1);
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);

    FARF(ਉੱਚਾ_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ਸਥਿਰ ਖਾਲੀ cumsum_thread_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_cumsum_preamble;

    ਅਚਲ uint8_t * src_ਅੰਕੜੇ = (ਅਚਲ uint8_t *) src0->ਅੰਕੜੇ;
    uint8_t *       dst_ਅੰਕੜੇ = (uint8_t *) dst->ਅੰਕੜੇ;

    ਅਚਲ uint32_t ir0 = cctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + cctx->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ir1 = MIN(ir0 + cctx->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread, cctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + cctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    ਲਈ (uint32_t ir = ir0; ir < ir1; ir++) {
        ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) (src_ਅੰਕੜੇ + ir * cctx->src_ਪੰਕਤੀ_ਮਾਪ);
        ਦਸ਼ਮਲਵ       * ਸੀਮਿਤ dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *)       (dst_ਅੰਕੜੇ + ir * cctx->dst_ਪੰਕਤੀ_ਮਾਪ);
        hvx_cumsum_ਪੰਕਤੀ_f32(src_ਪੰਕਤੀ, dst_ਪੰਕਤੀ, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(ਉੱਚਾ_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

ਪੂਰਨ op_cumsum_f32(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ size_t dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = dst->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = ਕੁੱਲ_ਪੰਕਤੀਆਂ;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ), (uint32_t) dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_ਪੰਕਤੀਆਂ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ         = src0->nb[1];
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ         = dst->nb[1];
    ਅਚਲ size_t src_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src_ਪੰਕਤੀ_ਮਾਪ, VLEN);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(dst_ਪੰਕਤੀ_ਮਾਪ, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ਅਚਲ size_t spad_ਪ੍ਰਤੀ_thread = 2 * (src_ਪੰਕਤੀ_ਮਾਪ_aligned + dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

    octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread = src_ਪੰਕਤੀ_ਮਾਪ_aligned * 2;
    octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned * 2;

    octx->src0_spad.ਮਾਪ  = n_ਧਾਗੇ * octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;
    octx->dst_spad.ਮਾਪ   = n_ਧਾਗੇ * octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;

    octx->src0_spad.ਅੰਕੜੇ  = octx->ctx->vtcm_ਅਧਾਰ;                        octx->src0_spad.src = ਨਿਰਾ;
    octx->dst_spad.ਅੰਕੜੇ   = octx->src0_spad.ਅੰਕੜੇ + octx->src0_spad.ਮਾਪ; octx->dst_spad.src  = ਨਿਰਾ;

    ਬਣਤਰ htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_ਪੰਕਤੀ_ਮਾਪ         = src_ਪੰਕਤੀ_ਮਾਪ,
        .dst_ਪੰਕਤੀ_ਮਾਪ         = dst_ਪੰਕਤੀ_ਮਾਪ,
        .src_ਪੰਕਤੀ_ਮਾਪ_aligned = src_ਪੰਕਤੀ_ਮਾਪ_aligned,
        .dst_ਪੰਕਤੀ_ਮਾਪ_aligned = dst_ਪੰਕਤੀ_ਮਾਪ_aligned,
        .ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread      = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
        .ਕੁੱਲ_ਪੰਕਤੀਆਂ           = nrows,
        .ਪੰਕਤੀ_ਸ਼ੁਰੂ            = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
    };

    ਜੇ (octx->ctx->vtcm_ਮਾਪ < spad_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ) {
        ਜੇ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, cumsum_thread_f32, &cctx, n_ਧਾਗੇ);
    } ਨਹੀਂਤਾਂ {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, cumsum_thread_f32_dma, &cctx, n_ਧਾਗੇ);
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_cumsum(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;

    ਚੋਣ (dst->type) {
        ਹਾਲਤ HTP_TYPE_F32:
            ਵਾਪਸ op_cumsum_f32(octx);
        ਮੂਲ:
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
}
