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
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-copy.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_diag_tensors_preamble                           \
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ src0 = octx->src[0]; \
    ਅਚਲ ਬਣਤਰ htp_tensor * ਸੀਮਿਤ dst  = octx->dst;    \
                                                            \
    ਅਚਲ uint32_t ne02 = src0->ne[2];                      \
                                                            \
    ਅਚਲ uint32_t ne0 = dst->ne[0];                        \
    ਅਚਲ uint32_t ne1 = dst->ne[1];                        \
                                                            \
    ਅਚਲ uint32_t nb02 = src0->nb[2];                      \
    ਅਚਲ uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ਅਚਲ uint32_t nb1 = dst->nb[1];                        \
    ਅਚਲ uint32_t nb2 = dst->nb[2];                        \
    ਅਚਲ uint32_t nb3 = dst->nb[3];

ਬਣਤਰ htp_diag_context {
    ਬਣਤਰ htp_ops_context * octx;
    size_t          src_batch_ਮਾਪ;
    size_t          dst_ਪੰਕਤੀ_ਮਾਪ;
    size_t          src_batch_ਮਾਪ_aligned;
    size_t          dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t        batches_ਪ੍ਰਤੀ_thread;
    uint32_t        ਕੁੱਲ_batches;
    uint32_t        batch_ਸ਼ੁਰੂ;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_diag_preamble                                              \
    ਬਣਤਰ htp_diag_context * dctx = (ਬਣਤਰ htp_diag_context *) ਅੰਕੜੇ; \
    ਬਣਤਰ htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_diag_ਪੰਕਤੀ_f32(ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src, ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst,
                                    uint32_t ਪੰਕਤੀ_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[ਪੰਕਤੀ_idx] = src[ਪੰਕਤੀ_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

ਸਥਿਰ ਖਾਲੀ diag_thread_f32_dma(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_diag_preamble;
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਅਚਲ uint32_t ib0 = dctx->batch_ਸ਼ੁਰੂ + dctx->batches_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ib1 = MIN(ib0 + dctx->batches_ਪ੍ਰਤੀ_thread, dctx->batch_ਸ਼ੁਰੂ + dctx->ਕੁੱਲ_batches);

    ਜੇ (ib0 >= ib1) {
        ਵਾਪਸ;
    }

    ਅਚਲ size_t src_batch_ਮਾਪ         = dctx->src_batch_ਮਾਪ;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ           = dctx->dst_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t src_batch_ਮਾਪ_aligned = dctx->src_batch_ਮਾਪ_aligned;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned   = dctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    ਅਚਲ dma_addr_t src_ਅੰਕੜੇ = src0->ਅੰਕੜੇ;
    ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ = dst->ਅੰਕੜੇ;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.ਅੰਕੜੇ + (ith * src_batch_ਮਾਪ_aligned);
    uint8_t * dst_spad = octx->dst_spad.ਅੰਕੜੇ  + (ith * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਲਈ (uint32_t ib = ib0; ib < ib1; ib++) {
        ਅਚਲ uint32_t i3 = ib / ne02;
        ਅਚਲ uint32_t i2 = ib % ne02;

        ਅਚਲ dma_addr_t src_batch = src_ਅੰਕੜੇ + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                       dma_make_ਅੰਕੜੇ(src_spad, src_batch),
                       src_batch_ਮਾਪ_aligned, src_batch_ਮਾਪ, src_batch_ਮਾਪ, 1);
        dma_ਕਤਾਰ_flush(dma_q);

        ਅਚਲ ਦਸ਼ਮਲਵ * src_spad_f32 = (ਅਚਲ ਦਸ਼ਮਲਵ *) src_spad;
        ਦਸ਼ਮਲਵ       * dst_spad_f32 = (ਦਸ਼ਮਲਵ *) dst_spad;

        ਲਈ (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_ਪੰਕਤੀ_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ਅਚਲ dma_addr_t dst_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                           dma_make_ਅੰਕੜੇ(dst_ਪੰਕਤੀ, dst_spad),
                           dst_ਪੰਕਤੀ_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 1);
            dma_ਕਤਾਰ_flush(dma_q);
        }
    }

    FARF(ਉੱਚਾ_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ਸਥਿਰ ਖਾਲੀ diag_thread_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    htp_diag_preamble;

    ਅਚਲ uint8_t * src_ਅੰਕੜੇ = (ਅਚਲ uint8_t *) src0->ਅੰਕੜੇ;
    uint8_t *       dst_ਅੰਕੜੇ = (uint8_t *) dst->ਅੰਕੜੇ;

    ਅਚਲ uint32_t ib0 = dctx->batch_ਸ਼ੁਰੂ + dctx->batches_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ib1 = MIN(ib0 + dctx->batches_ਪ੍ਰਤੀ_thread, dctx->batch_ਸ਼ੁਰੂ + dctx->ਕੁੱਲ_batches);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    ਲਈ (uint32_t ib = ib0; ib < ib1; ib++) {
        ਅਚਲ uint32_t i3 = ib / ne02;
        ਅਚਲ uint32_t i2 = ib % ne02;

        ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src_batch = (ਅਚਲ ਦਸ਼ਮਲਵ *)(src_ਅੰਕੜੇ + i3 * nb03 + i2 * nb02);

        ਲਈ (uint32_t i1 = 0; i1 < ne1; i1++) {
            ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *)(dst_ਅੰਕੜੇ + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_ਪੰਕਤੀ_f32(src_batch, dst_ਪੰਕਤੀ, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(ਉੱਚਾ_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

ਪੂਰਨ op_diag_f32(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਅਚਲ uint32_t ਕੁੱਲ_batches = src0->ne[2] * src0->ne[3];
    ਅਚਲ size_t dst_batch_ਮਾਪ  = dst->ne[1] * dst->nb[1];

    uint32_t batch_ਸ਼ੁਰੂ = 0;
    uint32_t nbatches    = ਕੁੱਲ_batches;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        bool can_ਵੰਡੋ = htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_ਪ੍ਰਤੀ_chunk = 1;
        ਜੇ (can_ਵੰਡੋ) {
            ਜੇ (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) == 0)) {
                batches_ਪ੍ਰਤੀ_chunk = 1;
            } ਨਹੀਂਤਾਂ ਜੇ (dst->nb[2] == dst_batch_ਮਾਪ &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_ਪ੍ਰਤੀ_chunk = (dst_batch_ਮਾਪ > 0) ? (HEX_L2_ਸਤਰ_ਮਾਪ / hex_ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_u32(dst_batch_ਮਾਪ, HEX_L2_ਸਤਰ_ਮਾਪ)) : 1;
            } ਨਹੀਂਤਾਂ {
                can_ਵੰਡੋ = false;
            }
        }

        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_batches, can_ਵੰਡੋ ? batches_ਪ੍ਰਤੀ_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        batch_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nbatches    = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nbatches == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ     = octx->n_ਧਾਗੇ;

    ਅਚਲ size_t src_batch_ਮਾਪ         = src0->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ           = dst->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t src_batch_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src_batch_ਮਾਪ, VLEN);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned   = hex_ਗੇੜਾ_ਉੱਪਰ(dst_ਪੰਕਤੀ_ਮਾਪ, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ਅਚਲ size_t spad_ਪ੍ਰਤੀ_thread = src_batch_ਮਾਪ_aligned + dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread = src_batch_ਮਾਪ_aligned;
    octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    octx->src0_spad.ਮਾਪ = n_ਧਾਗੇ * octx->src0_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;
    octx->dst_spad.ਮਾਪ  = n_ਧਾਗੇ * octx->dst_spad.ਮਾਪ_ਪ੍ਰਤੀ_thread;

    octx->src0_spad.ਅੰਕੜੇ = octx->ctx->vtcm_ਅਧਾਰ;                        octx->src0_spad.src = ਨਿਰਾ;
    octx->dst_spad.ਅੰਕੜੇ  = octx->src0_spad.ਅੰਕੜੇ + octx->src0_spad.ਮਾਪ; octx->dst_spad.src  = ਨਿਰਾ;

    ਬਣਤਰ htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_ਮਾਪ         = src_batch_ਮਾਪ,
        .dst_ਪੰਕਤੀ_ਮਾਪ           = dst_ਪੰਕਤੀ_ਮਾਪ,
        .src_batch_ਮਾਪ_aligned = src_batch_ਮਾਪ_aligned,
        .dst_ਪੰਕਤੀ_ਮਾਪ_aligned   = dst_ਪੰਕਤੀ_ਮਾਪ_aligned,
        .batches_ਪ੍ਰਤੀ_thread     = fastdiv(nbatches + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
        .ਕੁੱਲ_batches          = nbatches,
        .batch_ਸ਼ੁਰੂ            = batch_ਸ਼ੁਰੂ,
    };

    ਜੇ (octx->ctx->vtcm_ਮਾਪ < spad_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ) {
        ਜੇ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, diag_thread_f32, &dctx, n_ਧਾਗੇ);
    } ਨਹੀਂਤਾਂ {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, diag_thread_f32_dma, &dctx, n_ਧਾਗੇ);
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_diag(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;

    ਪੂਰਨ err = HTP_ਸਥਿਤੀ_OK;

    ਚੋਣ (dst->type) {
        ਹਾਲਤ HTP_TYPE_F32:
            err = op_diag_f32(octx);
            ਰੋਕੋ;
        ਮੂਲ:
            err = HTP_ਸਥਿਤੀ_NO_SUPPORT;
            ਰੋਕੋ;
    }

    ਵਾਪਸ err;
}
