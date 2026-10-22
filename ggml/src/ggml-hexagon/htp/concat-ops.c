#ਅਨ੍ਤਰ੍ਭਾਵ "concat-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-copy.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hexagon_protos.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hexagon_types.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-fence.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx_hexagon_protos.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "gather-rows.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

ਬਣਤਰ htp_concat_context {
    ਬਣਤਰ htp_ops_context * octx;
    uint8_t * spad0_ਅਧਾਰ;
    uint8_t * spad1_ਅਧਾਰ;
    uint32_t  spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t  spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t  ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t  nrows;
    uint32_t  nrows_ਪ੍ਰਤੀ_thread;
    uint32_t  nplanes;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne2;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ dma_addr_t concat_plane_addr(ਅਚਲ ਬਣਤਰ htp_tensor * t, uint32_t p, ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne2, uint32_t ne2) {
    ਅਚਲ uint32_t i3 = fastdiv(p, div_ne2);
    ਅਚਲ uint32_t i2 = p - i3 * ne2;
    ਵਾਪਸ t->ਅੰਕੜੇ + i2 * t->nb[2] + i3 * t->nb[3];
}

ਸਥਿਰ ਖਾਲੀ concat_2d_f32_transposed(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_concat_context * cctx = (ਬਣਤਰ htp_concat_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = cctx->octx;

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਅਚਲ uint32_t src0_ne0 = src0->ne[0];
    ਅਚਲ uint32_t src1_ne0 = src1->ne[0];

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ = cctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + cctx->nrows;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ_i = cctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + ith * cctx->nrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ਸਿਰਾ_i   = (ਸ਼ੁਰੂ_i + cctx->nrows_ਪ੍ਰਤੀ_thread < ਪੰਕਤੀ_ਸਿਰਾ) ? (ਸ਼ੁਰੂ_i + cctx->nrows_ਪ੍ਰਤੀ_thread) : ਪੰਕਤੀ_ਸਿਰਾ;
    ਜੇ (ਸ਼ੁਰੂ_i >= ਸਿਰਾ_i) ਵਾਪਸ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_ਅਧਾਰ = cctx->spad0_ਅਧਾਰ + ith * cctx->spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t * spad1_ਅਧਾਰ = cctx->spad1_ਅਧਾਰ + ith * cctx->spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread;

    ਅਚਲ uint32_t ਖੰਡ_i = 32;
    ਅਚਲ uint32_t spad1_stride = ਖੰਡ_i * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ HVX_ਸਦਿਸ਼ offsets = hvx_vec_gather_offsets_w(spad1_stride);
    ਅਚਲ uint32_t src1_ne0_padded = hex_ਗੇੜਾ_ਉੱਪਰ(src1_ne0, 32);
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ  = src0_ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, VLEN);
    ਅਚਲ uint32_t src0_pre        = src0_ਪੰਕਤੀ_padded - src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
    ਅਚਲ uint32_t spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ = src0_ਪੰਕਤੀ_padded + src1_ne0_padded * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne2 = &cctx->div_ne2;
    ਅਚਲ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = ਸ਼ੁਰੂ_i;

    ਅਚਲ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(spad1_ਅਧਾਰ, src1_addr), spad1_stride, src1->nb[0], MIN(ਸਿਰਾ_i - i, ਖੰਡ_i) * ਆਕਾਰ(ਦਸ਼ਮਲਵ), src1_ne0);

    ਅਚਲ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(spad0_ਅਧਾਰ + src0_pre, src0_addr), spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, src0->nb[1], src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, MIN(ਸਿਰਾ_i - i, ਖੰਡ_i));

    dma_ਕਤਾਰ_ਕੱਢੋ(dma_q); // src1

    ਜਦੋਂ (p < cctx->nplanes) {
        ਅਚਲ uint32_t current_ਖੰਡ_i = MIN(ਸਿਰਾ_i - i, ਖੰਡ_i);

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ਲਈ (uint32_t j = 0; j < src1_ne0; j += 32) {
            ਅਚਲ uint8_t * src_ptr = spad1_ਅਧਾਰ + j * spad1_stride;
            uint8_t * dst_ptr = spad0_ਅਧਾਰ + src0_ਪੰਕਤੀ_padded + j * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            hvx_transpose_32x32_w_gather(dst_ptr, spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, src_ptr, spad1_stride, offsets, current_ਖੰਡ_i, MIN(src1_ne0 - j, 32));
        }
        hvx_gather_sync(spad0_ਅਧਾਰ + src0_ਪੰਕਤੀ_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + ਖੰਡ_i;
        ਜੇ (ni >= ਸਿਰਾ_i) {
            ni = ਸ਼ੁਰੂ_i;
            np++;
        }
        ਅਚਲ bool has_ਅਗਲਾ = np < cctx->nplanes;
        ਅਚਲ uint32_t ਅਗਲਾ_ਖੰਡ_i = MIN(ਸਿਰਾ_i - ni, ਖੰਡ_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ਜੇ (has_ਅਗਲਾ) {
            ਅਚਲ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(spad1_ਅਧਾਰ, nsrc1_addr), spad1_stride, src1->nb[0], ਅਗਲਾ_ਖੰਡ_i * ਆਕਾਰ(ਦਸ਼ਮਲਵ), src1_ne0);
        }

        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q); // src0

        ਅਚਲ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_addr, spad0_ਅਧਾਰ + src0_pre), dst->nb[1], spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, (src0_ne0 + src1_ne0) * ਆਕਾਰ(ਦਸ਼ਮਲਵ), current_ਖੰਡ_i);

        ਜੇ (has_ਅਗਲਾ) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q); // next src1
        }
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q); // dst

        // spad0 is free after the dst write
        ਜੇ (has_ਅਗਲਾ) {
            ਅਚਲ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(spad0_ਅਧਾਰ + src0_pre, nsrc0_addr), spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, src0->nb[1], src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਖੰਡ_i);
        }

        p = np;
        i = ni;
    }
    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਖਾਲੀ concat_2d_f16_transposed(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_concat_context * cctx = (ਬਣਤਰ htp_concat_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = cctx->octx;

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਅਚਲ uint32_t src0_ne0 = src0->ne[0];
    ਅਚਲ uint32_t src1_ne0 = src1->ne[0];

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਸਿਰਾ = cctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + cctx->nrows;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ_i = cctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + ith * cctx->nrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ਸਿਰਾ_i   = (ਸ਼ੁਰੂ_i + cctx->nrows_ਪ੍ਰਤੀ_thread < ਪੰਕਤੀ_ਸਿਰਾ) ? (ਸ਼ੁਰੂ_i + cctx->nrows_ਪ੍ਰਤੀ_thread) : ਪੰਕਤੀ_ਸਿਰਾ;
    ਜੇ (ਸ਼ੁਰੂ_i >= ਸਿਰਾ_i) ਵਾਪਸ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_ਅਧਾਰ = cctx->spad0_ਅਧਾਰ + ith * cctx->spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint8_t * spad1_ਅਧਾਰ = cctx->spad1_ਅਧਾਰ + ith * cctx->spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread;

    ਅਚਲ uint32_t ਖੰਡ_i = 64;
    ਅਚਲ uint32_t spad1_stride = ਖੰਡ_i * ਆਕਾਰ(__fp16);

    ਅਚਲ HVX_ਸਦਿਸ਼ offsets = hvx_vec_gather_offsets_h(spad1_stride);
    ਅਚਲ uint32_t src1_ne0_padded = hex_ਗੇੜਾ_ਉੱਪਰ(src1_ne0, 64);
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ  = src0_ne0 * ਆਕਾਰ(__fp16);
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, VLEN);
    ਅਚਲ uint32_t src0_pre        = src0_ਪੰਕਤੀ_padded - src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
    ਅਚਲ uint32_t spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ = src0_ਪੰਕਤੀ_padded + src1_ne0_padded * ਆਕਾਰ(__fp16);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne2 = &cctx->div_ne2;
    ਅਚਲ uint32_t ne2 = dst->ne[2];

    uint32_t p = 0;
    uint32_t i = ਸ਼ੁਰੂ_i;

    ਅਚਲ dma_addr_t src1_addr = concat_plane_addr(src1, p, div_ne2, ne2) + i * src1->nb[1];
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(spad1_ਅਧਾਰ, src1_addr), spad1_stride, src1->nb[0], MIN(ਸਿਰਾ_i - i, ਖੰਡ_i) * ਆਕਾਰ(__fp16), src1_ne0);

    ਅਚਲ dma_addr_t src0_addr = concat_plane_addr(src0, p, div_ne2, ne2) + i * src0->nb[1];
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(spad0_ਅਧਾਰ + src0_pre, src0_addr), spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, src0->nb[1], src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, MIN(ਸਿਰਾ_i - i, ਖੰਡ_i));

    dma_ਕਤਾਰ_ਕੱਢੋ(dma_q); // src1

    ਜਦੋਂ (p < cctx->nplanes) {
        ਅਚਲ uint32_t current_ਖੰਡ_i = MIN(ਸਿਰਾ_i - i, ਖੰਡ_i);

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ਲਈ (uint32_t j = 0; j < src1_ne0; j += 64) {
            ਅਚਲ uint8_t * src_ptr = spad1_ਅਧਾਰ + j * spad1_stride;
            uint8_t * dst_ptr = spad0_ਅਧਾਰ + src0_ਪੰਕਤੀ_padded + j * ਆਕਾਰ(__fp16);
            hvx_transpose_64x64_h_gather(dst_ptr, spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, src_ptr, spad1_stride, offsets, current_ਖੰਡ_i, MIN(src1_ne0 - j, 64));
        }
        hvx_gather_sync(spad0_ਅਧਾਰ + src0_ਪੰਕਤੀ_padded);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

        uint32_t np = p;
        uint32_t ni = i + ਖੰਡ_i;
        ਜੇ (ni >= ਸਿਰਾ_i) {
            ni = ਸ਼ੁਰੂ_i;
            np++;
        }
        ਅਚਲ bool has_ਅਗਲਾ = np < cctx->nplanes;
        ਅਚਲ uint32_t ਅਗਲਾ_ਖੰਡ_i = MIN(ਸਿਰਾ_i - ni, ਖੰਡ_i);

        // spad1 is free after the gather sync, prefetch next src1 ahead of the dst write
        ਜੇ (has_ਅਗਲਾ) {
            ਅਚਲ dma_addr_t nsrc1_addr = concat_plane_addr(src1, np, div_ne2, ne2) + ni * src1->nb[1];
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(spad1_ਅਧਾਰ, nsrc1_addr), spad1_stride, src1->nb[0], ਅਗਲਾ_ਖੰਡ_i * ਆਕਾਰ(__fp16), src1_ne0);
        }

        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q); // src0

        ਅਚਲ dma_addr_t dst_addr = concat_plane_addr(dst, p, div_ne2, ne2) + i * dst->nb[1];
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_addr, spad0_ਅਧਾਰ + src0_pre), dst->nb[1], spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, (src0_ne0 + src1_ne0) * ਆਕਾਰ(__fp16), current_ਖੰਡ_i);

        ਜੇ (has_ਅਗਲਾ) {
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q); // next src1
        }
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q); // dst

        // spad0 is free after the dst write
        ਜੇ (has_ਅਗਲਾ) {
            ਅਚਲ dma_addr_t nsrc0_addr = concat_plane_addr(src0, np, div_ne2, ne2) + ni * src0->nb[1];
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(spad0_ਅਧਾਰ + src0_pre, nsrc0_addr), spad0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, src0->nb[1], src0_ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਖੰਡ_i);
        }

        p = np;
        i = ni;
    }
    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਪੂਰਨ concat_regular(ਬਣਤਰ htp_ops_context * octx, ਪੂਰਨ dim, uint32_t type_ਮਾਪ) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਬਣਤਰ htp_tensor view0 = *dst;
    ਬਣਤਰ htp_tensor view1 = *dst;
    ਲਈ (ਪੂਰਨ d = 0; d < HTP_OP_MAX_DIMS; d++) {
        view0.ne[d] = src0->ne[d];
        view1.ne[d] = src1->ne[d];
    }
    view1.ਅੰਕੜੇ += src0->ne[dim] * dst->nb[dim];

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ_0 = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ_1 = src1->ne[1] * src1->ne[2] * src1->ne[3];

    uint32_t rstart0 = 0, nrows0 = ਕੁੱਲ_ਪੰਕਤੀਆਂ_0;
    uint32_t rstart1 = 0, nrows1 = ਕੁੱਲ_ਪੰਕਤੀਆਂ_1;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range0 = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            ਕੁੱਲ_ਪੰਕਤੀਆਂ_0, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        rstart0 = range0.ਸ਼ੁਰੂ;
        nrows0  = range0.ਗਿਣਤੀ_2;

        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range1 = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            ਕੁੱਲ_ਪੰਕਤੀਆਂ_1, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        rstart1 = range1.ਸ਼ੁਰੂ;
        nrows1  = range1.ਗਿਣਤੀ_2;
    }

    dma_ਕਤਾਰ * q = octx->ctx->dma[0];

    dma_cpy_sametype_sameshape_range(q, &view0, src0, type_ਮਾਪ, rstart0, nrows0);
    dma_cpy_sametype_sameshape_range(q, &view1, src1, type_ਮਾਪ, rstart1, nrows1);
    dma_ਕਤਾਰ_flush(q);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਸਥਿਰ ਪੂਰਨ concat_transposed(ਬਣਤਰ htp_ops_context * octx, ਅਚਲ ਬਣਤਰ htp_concat_kernel_params * kparams, uint32_t type_ਮਾਪ) {
    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = dst->ne[1];
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_ਪੰਕਤੀਆਂ, 1, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0 || dst->ne[2] == 0 || dst->ne[3] == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (kparams->vtcm_ਮਾਪ > octx->ctx->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    // layout precomputed on host; kept for reference:
    // struct htp_concat_transposed_vtcm_layout layout;
    // htp_concat_transposed_vtcm_layout_build(&layout, octx->src[0]->ne[0], octx->src[1]->ne[0], type_size, n_threads);

    uint8_t * vtcm_ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;

    ਬਣਤਰ htp_concat_context cctx;
    cctx.octx                  = octx;
    cctx.spad0_ਅਧਾਰ            = vtcm_ਅਧਾਰ;
    cctx.spad1_ਅਧਾਰ            = vtcm_ਅਧਾਰ + n_ਧਾਗੇ * kparams->u.transposed.spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    cctx.spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->u.transposed.spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    cctx.spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->u.transposed.spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    cctx.ਪੰਕਤੀ_ਸ਼ੁਰੂ             = ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    cctx.nrows                 = nrows;
    cctx.nplanes               = dst->ne[2] * dst->ne[3];
    cctx.div_ne2               = init_fastdiv_ਮੁੱਲਾਂ(dst->ne[2]);
    cctx.nrows_ਪ੍ਰਤੀ_thread      = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);

    work_ਕਤਾਰ_func_t ਕਾਮਾ_func = (type_ਮਾਪ == 4) ? concat_2d_f32_transposed : concat_2d_f16_transposed;
    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, ਕਾਮਾ_func, &cctx, n_ਧਾਗੇ);
    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_concat(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_concat_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_concat_kernel_params *) octx->kernel_params;
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਪੂਰਨ ਸਥਿਤੀ = HTP_ਸਥਿਤੀ_OK;
    ਅਚਲ uint32_t type_ਮਾਪ = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    ਚੋਣ (kparams->kernel_type) {
        ਹਾਲਤ HTP_CONCAT_KERNEL_REGULAR:
            ਸਥਿਤੀ = concat_regular(octx, kparams->dim, type_ਮਾਪ);
            ਰੋਕੋ;

        ਹਾਲਤ HTP_CONCAT_KERNEL_TRANSPOSED:
            ਸਥਿਤੀ = concat_transposed(octx, kparams, type_ਮਾਪ);
            ਰੋਕੋ;

        ਹਾਲਤ HTP_CONCAT_KERNEL_GATHER_ਪੰਕਤੀਆਂ:
            ਸਥਿਤੀ = htp_gather_ਪੰਕਤੀਆਂ(octx, src0, src1, &kparams->u.gather, kparams->n_ਧਾਗੇ);
            ਰੋਕੋ;

        ਮੂਲ:
            ਸਥਿਤੀ = HTP_ਸਥਿਤੀ_NO_SUPPORT;
            ਰੋਕੋ;
    }

    htp_ops_context_ਸੈੱਟ_ਸਥਿਤੀ(octx, ਸਥਿਤੀ);

    ਵਾਪਸ octx->ਸਥਿਤੀ;
}
