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
#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-copy.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"

#ವ್ಯಾಖ್ಯಾನಿಸು htp_diag_tensors_preamble                           \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ src0 = octx->src[0]; \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಸೀಮಿತ dst  = octx->dst;    \
                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];                      \
                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];                        \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];                        \
                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];                      \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];                        \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];                        \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

ರಚನೆ htp_diag_context {
    ರಚನೆ htp_ops_context * octx;
    size_t          src_batch_ಗಾತ್ರ;
    size_t          dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t          src_batch_ಗಾತ್ರ_aligned;
    size_t          dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t        batches_ಪ್ರತಿ_thread;
    uint32_t        ಒಟ್ಟು_batches;
    uint32_t        batch_ಆರಂಭ;
};

#ವ್ಯಾಖ್ಯಾನಿಸು htp_diag_preamble                                              \
    ರಚನೆ htp_diag_context * dctx = (ರಚನೆ htp_diag_context *) ದತ್ತಾಂಶ; \
    ರಚನೆ htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_diag_ಅಡ್ಡಸಾಲು_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src, ದಶಮಾಂಶ * ಸೀಮಿತ dst,
                                    uint32_t ಅಡ್ಡಸಾಲು_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[ಅಡ್ಡಸಾಲು_idx] = src[ಅಡ್ಡಸಾಲು_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ diag_thread_f32_dma(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_diag_preamble;
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಸ್ಥಿರಾಂಕ uint32_t ib0 = dctx->batch_ಆರಂಭ + dctx->batches_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ib1 = MIN(ib0 + dctx->batches_ಪ್ರತಿ_thread, dctx->batch_ಆರಂಭ + dctx->ಒಟ್ಟು_batches);

    ಆದರೆ (ib0 >= ib1) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t src_batch_ಗಾತ್ರ         = dctx->src_batch_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ           = dctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src_batch_ಗಾತ್ರ_aligned = dctx->src_batch_ಗಾತ್ರ_aligned;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned   = dctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    ಸ್ಥಿರಾಂಕ dma_addr_t src_ದತ್ತಾಂಶ = src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ = dst->ದತ್ತಾಂಶ;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.ದತ್ತಾಂಶ + (ith * src_batch_ಗಾತ್ರ_aligned);
    uint8_t * dst_spad = octx->dst_spad.ದತ್ತಾಂಶ  + (ith * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಗಾಗಿ (uint32_t ib = ib0; ib < ib1; ib++) {
        ಸ್ಥಿರಾಂಕ uint32_t i3 = ib / ne02;
        ಸ್ಥಿರಾಂಕ uint32_t i2 = ib % ne02;

        ಸ್ಥಿರಾಂಕ dma_addr_t src_batch = src_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                       dma_make_ದತ್ತಾಂಶ(src_spad, src_batch),
                       src_batch_ಗಾತ್ರ_aligned, src_batch_ಗಾತ್ರ, src_batch_ಗಾತ್ರ, 1);
        dma_ಸರತಿ_flush(dma_q);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_spad_f32 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_spad;
        ದಶಮಾಂಶ       * dst_spad_f32 = (ದಶಮಾಂಶ *) dst_spad;

        ಗಾಗಿ (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_ಅಡ್ಡಸಾಲು_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ಸ್ಥಿರಾಂಕ dma_addr_t dst_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                           dma_make_ದತ್ತಾಂಶ(dst_ಅಡ್ಡಸಾಲು, dst_spad),
                           dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
            dma_ಸರತಿ_flush(dma_q);
        }
    }

    FARF(ಮೇಲಿನ_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ಸ್ಥಿರ ಶೂನ್ಯ diag_thread_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    htp_diag_preamble;

    ಸ್ಥಿರಾಂಕ uint8_t * src_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ uint8_t *) src0->ದತ್ತಾಂಶ;
    uint8_t *       dst_ದತ್ತಾಂಶ = (uint8_t *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ uint32_t ib0 = dctx->batch_ಆರಂಭ + dctx->batches_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ib1 = MIN(ib0 + dctx->batches_ಪ್ರತಿ_thread, dctx->batch_ಆರಂಭ + dctx->ಒಟ್ಟು_batches);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    ಗಾಗಿ (uint32_t ib = ib0; ib < ib1; ib++) {
        ಸ್ಥಿರಾಂಕ uint32_t i3 = ib / ne02;
        ಸ್ಥಿರಾಂಕ uint32_t i2 = ib % ne02;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src_batch = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(src_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02);

        ಗಾಗಿ (uint32_t i1 = 0; i1 < ne1; i1++) {
            ದಶಮಾಂಶ * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *)(dst_ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_ಅಡ್ಡಸಾಲು_f32(src_batch, dst_ಅಡ್ಡಸಾಲು, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(ಮೇಲಿನ_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

ಪೂರ್ಣಾಂಕ op_diag_f32(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_batches = src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ size_t dst_batch_ಗಾತ್ರ  = dst->ne[1] * dst->nb[1];

    uint32_t batch_ಆರಂಭ = 0;
    uint32_t nbatches    = ಒಟ್ಟು_batches;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        bool can_ವಿಭಜಿಸು = htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_ಪ್ರತಿ_chunk = 1;
        ಆದರೆ (can_ವಿಭಜಿಸು) {
            ಆದರೆ (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) == 0)) {
                batches_ಪ್ರತಿ_chunk = 1;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->nb[2] == dst_batch_ಗಾತ್ರ &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_ಪ್ರತಿ_chunk = (dst_batch_ಗಾತ್ರ > 0) ? (HEX_L2_ಸಾಲು_ಗಾತ್ರ / hex_ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_u32(dst_batch_ಗಾತ್ರ, HEX_L2_ಸಾಲು_ಗಾತ್ರ)) : 1;
            } ಇಲ್ಲದಿದ್ದರೆ {
                can_ವಿಭಜಿಸು = false;
            }
        }

        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_batches, can_ವಿಭಜಿಸು ? batches_ಪ್ರತಿ_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        batch_ಆರಂಭ = range.ಆರಂಭ;
        nbatches    = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nbatches == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು     = octx->n_ಎಳೆಗಳು;

    ಸ್ಥಿರಾಂಕ size_t src_batch_ಗಾತ್ರ         = src0->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ           = dst->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src_batch_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src_batch_ಗಾತ್ರ, VLEN);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned   = hex_ಸುತ್ತು_ಮೇಲೆ(dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ಸ್ಥಿರಾಂಕ size_t spad_ಪ್ರತಿ_thread = src_batch_ಗಾತ್ರ_aligned + dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread = src_batch_ಗಾತ್ರ_aligned;
    octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    octx->src0_spad.ಗಾತ್ರ_2 = n_ಎಳೆಗಳು * octx->src0_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;
    octx->dst_spad.ಗಾತ್ರ_2  = n_ಎಳೆಗಳು * octx->dst_spad.ಗಾತ್ರ_ಪ್ರತಿ_thread;

    octx->src0_spad.ದತ್ತಾಂಶ = octx->ctx->vtcm_ತಳ;                        octx->src0_spad.src = ಖಾಲಿ;
    octx->dst_spad.ದತ್ತಾಂಶ  = octx->src0_spad.ದತ್ತಾಂಶ + octx->src0_spad.ಗಾತ್ರ_2; octx->dst_spad.src  = ಖಾಲಿ;

    ರಚನೆ htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_ಗಾತ್ರ         = src_batch_ಗಾತ್ರ,
        .dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ           = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
        .src_batch_ಗಾತ್ರ_aligned = src_batch_ಗಾತ್ರ_aligned,
        .dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned   = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
        .batches_ಪ್ರತಿ_thread     = fastdiv(nbatches + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
        .ಒಟ್ಟು_batches          = nbatches,
        .batch_ಆರಂಭ            = batch_ಆರಂಭ,
    };

    ಆದರೆ (octx->ctx->vtcm_ಗಾತ್ರ < spad_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು) {
        ಆದರೆ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, diag_thread_f32, &dctx, n_ಎಳೆಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, diag_thread_f32_dma, &dctx, n_ಎಳೆಗಳು);
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_diag(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;

    ಪೂರ್ಣಾಂಕ err = HTP_ಸ್ಥಿತಿ_OK;

    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:
            err = op_diag_f32(octx);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            err = HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            ನಿಲ್ಲಿಸು;
    }

    ಹಿಂತಿರುಗಿ err;
}
