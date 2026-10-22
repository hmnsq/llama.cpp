#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-tensor.h"
#അന്തര്ഭാവ "hvx-types.h"
#അന്തര്ഭാവ "hex-utils.h"
#അന്തര്ഭാവ "hvx-copy.h"
#അന്തര്ഭാവ "dma-queue.h"

#define htp_diag_tensors_preamble                           \
    സ്ഥിരം ഘടന htp_tensor * restrict src0 = octx->src[0]; \
    സ്ഥിരം ഘടന htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    സ്ഥിരം uint32_t ne02 = src0->ne[2];                      \
                                                            \
    സ്ഥിരം uint32_t ne0 = dst->ne[0];                        \
    സ്ഥിരം uint32_t ne1 = dst->ne[1];                        \
                                                            \
    സ്ഥിരം uint32_t nb02 = src0->nb[2];                      \
    സ്ഥിരം uint32_t nb03 = src0->nb[3];                      \
                                                            \
    സ്ഥിരം uint32_t nb1 = dst->nb[1];                        \
    സ്ഥിരം uint32_t nb2 = dst->nb[2];                        \
    സ്ഥിരം uint32_t nb3 = dst->nb[3];

ഘടന htp_diag_context {
    ഘടന htp_ops_context * octx;
    size_t          src_batch_size;
    size_t          dst_row_size;
    size_t          src_batch_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        batches_per_thread;
    uint32_t        total_batches;
    uint32_t        batch_start;
};

#define htp_diag_preamble                                              \
    ഘടന htp_diag_context * dctx = (ഘടന htp_diag_context *) data; \
    ഘടന htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_diag_row_f32(സ്ഥിരം ദശാംശം * restrict src, ദശാംശം * restrict dst,
                                    uint32_t row_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[row_idx] = src[row_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

സ്ഥിതം ശൂന്യം diag_thread_f32_dma(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    htp_diag_preamble;
    dma_queue * dma_q = octx->ctx->dma[ith];

    സ്ഥിരം uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    സ്ഥിരം uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    എങ്കിൽ (ib0 >= ib1) {
        തിരികെ;
    }

    സ്ഥിരം size_t src_batch_size         = dctx->src_batch_size;
    സ്ഥിരം size_t dst_row_size           = dctx->dst_row_size;
    സ്ഥിരം size_t src_batch_size_aligned = dctx->src_batch_size_aligned;
    സ്ഥിരം size_t dst_row_size_aligned   = dctx->dst_row_size_aligned;

    സ്ഥിരം dma_addr_t src_data = src0->data;
    സ്ഥിരം dma_addr_t dst_data = dst->data;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.data + (ith * src_batch_size_aligned);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned);

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];

    വേണ്ടി (uint32_t ib = ib0; ib < ib1; ib++) {
        സ്ഥിരം uint32_t i3 = ib / ne02;
        സ്ഥിരം uint32_t i2 = ib % ne02;

        സ്ഥിരം dma_addr_t src_batch = src_data + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_queue_തള്ളുക(dma_q,
                       dma_make_data(src_spad, src_batch),
                       src_batch_size_aligned, src_batch_size, src_batch_size, 1);
        dma_queue_flush(dma_q);

        സ്ഥിരം ദശാംശം * src_spad_f32 = (സ്ഥിരം ദശാംശം *) src_spad;
        ദശാംശം       * dst_spad_f32 = (ദശാംശം *) dst_spad;

        വേണ്ടി (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_row_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            സ്ഥിരം dma_addr_t dst_row = dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_queue_തള്ളുക(dma_q,
                           dma_make_data(dst_row, dst_spad),
                           dst_row_size, dst_row_size_aligned, dst_row_size, 1);
            dma_queue_flush(dma_q);
        }
    }

    FARF(മുകളിലെ_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

സ്ഥിതം ശൂന്യം diag_thread_f32(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    htp_diag_preamble;

    സ്ഥിരം uint8_t * src_data = (സ്ഥിരം uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    സ്ഥിരം uint32_t ib0 = dctx->batch_start + dctx->batches_per_thread * ith;
    സ്ഥിരം uint32_t ib1 = MIN(ib0 + dctx->batches_per_thread, dctx->batch_start + dctx->total_batches);

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    വേണ്ടി (uint32_t ib = ib0; ib < ib1; ib++) {
        സ്ഥിരം uint32_t i3 = ib / ne02;
        സ്ഥിരം uint32_t i2 = ib % ne02;

        സ്ഥിരം ദശാംശം * restrict src_batch = (സ്ഥിരം ദശാംശം *)(src_data + i3 * nb03 + i2 * nb02);

        വേണ്ടി (uint32_t i1 = 0; i1 < ne1; i1++) {
            ദശാംശം * restrict dst_row = (ദശാംശം *)(dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_row_f32(src_batch, dst_row, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(മുകളിലെ_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

പൂർണ്ണസംഖ്യ op_diag_f32(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;

    സ്ഥിരം uint32_t total_batches = src0->ne[2] * src0->ne[3];
    സ്ഥിരം size_t dst_batch_size  = dst->ne[1] * dst->nb[1];

    uint32_t batch_start = 0;
    uint32_t nbatches    = total_batches;

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == വലുപ്പം(ദശാംശം)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_per_chunk = 1;
        എങ്കിൽ (can_split) {
            എങ്കിൽ (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) == 0)) {
                batches_per_chunk = 1;
            } അല്ലെങ്കിൽ എങ്കിൽ (dst->nb[2] == dst_batch_size &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_per_chunk = (dst_batch_size > 0) ? (HEX_L2_വരി_SIZE / hex_gcd_u32(dst_batch_size, HEX_L2_വരി_SIZE)) : 1;
            } അല്ലെങ്കിൽ {
                can_split = false;
            }
        }

        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, can_split ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        batch_start = range.start;
        nbatches    = range.എണ്ണം;
    }

    എങ്കിൽ (nbatches == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t n_threads     = octx->n_threads;

    സ്ഥിരം size_t src_batch_size         = src0->ne[0] * വലുപ്പം(ദശാംശം);
    സ്ഥിരം size_t dst_row_size           = dst->ne[0] * വലുപ്പം(ദശാംശം);
    സ്ഥിരം size_t src_batch_size_aligned = hex_round_up(src_batch_size, VLEN);
    സ്ഥിരം size_t dst_row_size_aligned   = hex_round_up(dst_row_size, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    സ്ഥിരം size_t spad_per_thread = src_batch_size_aligned + dst_row_size_aligned;

    octx->src0_spad.size_per_thread = src_batch_size_aligned;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = ശൂന്യ;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = ശൂന്യ;

    ഘടന htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_size         = src_batch_size,
        .dst_row_size           = dst_row_size,
        .src_batch_size_aligned = src_batch_size_aligned,
        .dst_row_size_aligned   = dst_row_size_aligned,
        .batches_per_thread     = fastdiv(nbatches + n_threads - 1, &octx->n_threads_div),
        .total_batches          = nbatches,
        .batch_start            = batch_start,
    };

    എങ്കിൽ (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        എങ്കിൽ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            തിരികെ HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, diag_thread_f32, &dctx, n_threads);
    } അല്ലെങ്കിൽ {
        work_queue_run(octx->ctx->work_queue, diag_thread_f32_dma, &dctx, n_threads);
    }

    തിരികെ HTP_STATUS_OK;
}

പൂർണ്ണസംഖ്യ op_diag(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * dst = octx->dst;

    പൂർണ്ണസംഖ്യ err = HTP_STATUS_OK;

    തിരഞ്ഞെടുപ്പ് (dst->type) {
        അവസ്ഥ HTP_TYPE_F32:
            err = op_diag_f32(octx);
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            err = HTP_STATUS_NO_SUPPORT;
            നിർത്തുക;
    }

    തിരികെ err;
}
