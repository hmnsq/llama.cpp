#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
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
#അന്തര്ഭാവ "hvx-utils.h"
#അന്തര്ഭാവ "dma-queue.h"

#define htp_cumsum_tensors_preamble                         \
    സ്ഥിരം ഘടന htp_tensor * restrict src0 = octx->src[0]; \
    സ്ഥിരം ഘടന htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    സ്ഥിരം uint32_t ne00 = src0->ne[0];                      \
    സ്ഥിരം uint32_t ne01 = src0->ne[1];                      \
    സ്ഥിരം uint32_t ne02 = src0->ne[2];                      \
    സ്ഥിരം uint32_t ne03 = src0->ne[3];                      \
                                                            \
    സ്ഥിരം uint32_t ne0 = dst->ne[0];                        \
    സ്ഥിരം uint32_t ne1 = dst->ne[1];                        \
    സ്ഥിരം uint32_t ne2 = dst->ne[2];                        \
    സ്ഥിരം uint32_t ne3 = dst->ne[3];                        \
                                                            \
    സ്ഥിരം uint32_t nb00 = src0->nb[0];                      \
    സ്ഥിരം uint32_t nb01 = src0->nb[1];                      \
    സ്ഥിരം uint32_t nb02 = src0->nb[2];                      \
    സ്ഥിരം uint32_t nb03 = src0->nb[3];                      \
                                                            \
    സ്ഥിരം uint32_t nb0 = dst->nb[0];                        \
    സ്ഥിരം uint32_t nb1 = dst->nb[1];                        \
    സ്ഥിരം uint32_t nb2 = dst->nb[2];                        \
    സ്ഥിരം uint32_t nb3 = dst->nb[3];

ഘടന htp_cumsum_context {
    ഘടന htp_ops_context * octx;
    size_t          src_row_size;
    size_t          dst_row_size;
    size_t          src_row_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    uint32_t        row_start;
};

#define htp_cumsum_preamble                                                \
    ഘടന htp_cumsum_context * cctx = (ഘടന htp_cumsum_context *) data; \
    ഘടന htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#എങ്കിൽ __HVX_ARCH__ > 75
സ്ഥിതം വരിയിൽ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    തിരികെ Q6_Vsf_vadd_VsfVsf(a, b);
}
#അല്ലെങ്കിൽ
സ്ഥിതം വരിയിൽ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    തിരികെ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#endif  // __HVX_ARCH__ > 75

സ്ഥിതം വരിയിൽ HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    സ്ഥിരം HVX_Vector zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_in);

    തിരികെ v;
}

സ്ഥിതം വരിയിൽ HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    തിരികെ hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_cumsum_row_f32(സ്ഥിരം ദശാംശം * restrict src, ദശാംശം * restrict dst, uint32_t n) {
    സ്ഥിരം uint32_t nvec = n / VLEN_FP32;
    സ്ഥിരം uint32_t nloe = n % VLEN_FP32;

    HVX_Vector carry = Q6_V_vsplat_R(0);

    വേണ്ടി (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = *((സ്ഥിരം HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        carry = hvx_splat_last_f32(v);
    }

    എങ്കിൽ (nloe) {
        ദശാംശം acc = hvx_vec_get_f32(carry);
        സ്ഥിരം ദശാംശം * src_tail = src + nvec * VLEN_FP32;
        ദശാംശം       * dst_tail = dst + nvec * VLEN_FP32;
        വേണ്ടി (uint32_t i = 0; i < nloe; i++) {
            acc        += src_tail[i];
            dst_tail[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

സ്ഥിതം ശൂന്യം cumsum_thread_f32_dma(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    htp_cumsum_preamble;

    സ്ഥിരം uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    സ്ഥിരം uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    എങ്കിൽ (ir0 >= ir1) {
        തിരികെ;
    }

    സ്ഥിരം size_t src_row_size         = cctx->src_row_size;
    സ്ഥിരം size_t dst_row_size         = cctx->dst_row_size;
    സ്ഥിരം size_t src_row_size_aligned = cctx->src_row_size_aligned;
    സ്ഥിരം size_t dst_row_size_aligned = cctx->dst_row_size_aligned;

    സ്ഥിരം dma_addr_t src_data = src0->data;
    സ്ഥിരം dma_addr_t dst_data = dst->data;

    uint8_t * src_spad = octx->src0_spad.data + (ith * src_row_size_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned * 2);

    വേണ്ടി (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_തള്ളുക(dma_q,
                       dma_make_data(dst_data, dst_spad + (spad_idx * dst_row_size_aligned)),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 0);

        dma_queue_തള്ളുക(dma_q,
                       dma_make_data(src_spad + (spad_idx * src_row_size_aligned),
                                     src_data + (ir * src_row_size)),
                       src_row_size_aligned, src_row_size, src_row_size, 1);
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];

    വേണ്ടി (uint32_t ir = ir0; ir < ir1; ir++) {
        ദശാംശം * dst_spad_row = (ദശാംശം *) dma_queue_എടുക്കുക(dma_q).src;
        ദശാംശം * src_spad_row = (ദശാംശം *) dma_queue_എടുക്കുക(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_row_f32(src_spad_row, dst_spad_row, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_തള്ളുക(dma_q,
                       dma_make_data(dst_data + (ir * dst_row_size), dst_spad_row),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 1);

        സ്ഥിരം uint32_t next_row = ir + 2;
        എങ്കിൽ (next_row < ir1) {
            dma_queue_തള്ളുക(dma_q,
                           dma_make_data(src_spad_row, src_data + (next_row * src_row_size)),
                           src_row_size_aligned, src_row_size, src_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(മുകളിലെ_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

സ്ഥിതം ശൂന്യം cumsum_thread_f32(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    htp_cumsum_preamble;

    സ്ഥിരം uint8_t * src_data = (സ്ഥിരം uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    സ്ഥിരം uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    സ്ഥിരം uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    വേണ്ടി (uint32_t ir = ir0; ir < ir1; ir++) {
        സ്ഥിരം ദശാംശം * restrict src_row = (സ്ഥിരം ദശാംശം *) (src_data + ir * cctx->src_row_size);
        ദശാംശം       * restrict dst_row = (ദശാംശം *)       (dst_data + ir * cctx->dst_row_size);
        hvx_cumsum_row_f32(src_row, dst_row, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(മുകളിലെ_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

പൂർണ്ണസംഖ്യ op_cumsum_f32(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;

    സ്ഥിരം uint32_t total_rows      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    സ്ഥിരം size_t dst_data_row_size = dst->ne[0] * വലുപ്പം(ദശാംശം);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, വലുപ്പം(ദശാംശം), (uint32_t) dst_data_row_size, &rows_per_chunk);
        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        row_start = range.start;
        nrows     = range.എണ്ണം;
    }

    എങ്കിൽ (nrows == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t n_threads = octx->n_threads;

    സ്ഥിരം size_t src_row_size         = src0->nb[1];
    സ്ഥിരം size_t dst_row_size         = dst->nb[1];
    സ്ഥിരം size_t src_row_size_aligned = hex_round_up(src_row_size, VLEN);
    സ്ഥിരം size_t dst_row_size_aligned = hex_round_up(dst_row_size, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    സ്ഥിരം size_t spad_per_thread = 2 * (src_row_size_aligned + dst_row_size_aligned);

    octx->src0_spad.size_per_thread = src_row_size_aligned * 2;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned * 2;

    octx->src0_spad.size  = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size   = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data  = octx->ctx->vtcm_base;                        octx->src0_spad.src = ശൂന്യ;
    octx->dst_spad.data   = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = ശൂന്യ;

    ഘടന htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_row_size         = src_row_size,
        .dst_row_size         = dst_row_size,
        .src_row_size_aligned = src_row_size_aligned,
        .dst_row_size_aligned = dst_row_size_aligned,
        .rows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows           = nrows,
        .row_start            = row_start,
    };

    എങ്കിൽ (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        എങ്കിൽ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            തിരികെ HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32, &cctx, n_threads);
    } അല്ലെങ്കിൽ {
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32_dma, &cctx, n_threads);
    }

    തിരികെ HTP_STATUS_OK;
}

പൂർണ്ണസംഖ്യ op_cumsum(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * dst = octx->dst;

    തിരഞ്ഞെടുപ്പ് (dst->type) {
        അവസ്ഥ HTP_TYPE_F32:
            തിരികെ op_cumsum_f32(octx);
        സ്ഥിരസ്ഥിതി:
            തിരികെ HTP_STATUS_NO_SUPPORT;
    }
}
