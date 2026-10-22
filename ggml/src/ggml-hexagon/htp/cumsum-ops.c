#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "hvx-types.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"
#ଅନ୍ତର୍ଭାଵ "dma-queue.h"

#define htp_cumsum_tensors_preamble                         \
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0]; \
    ଅଚଳ ସଂରଚନା htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ଅଚଳ uint32_t ne00 = src0->ne[0];                      \
    ଅଚଳ uint32_t ne01 = src0->ne[1];                      \
    ଅଚଳ uint32_t ne02 = src0->ne[2];                      \
    ଅଚଳ uint32_t ne03 = src0->ne[3];                      \
                                                            \
    ଅଚଳ uint32_t ne0 = dst->ne[0];                        \
    ଅଚଳ uint32_t ne1 = dst->ne[1];                        \
    ଅଚଳ uint32_t ne2 = dst->ne[2];                        \
    ଅଚଳ uint32_t ne3 = dst->ne[3];                        \
                                                            \
    ଅଚଳ uint32_t nb00 = src0->nb[0];                      \
    ଅଚଳ uint32_t nb01 = src0->nb[1];                      \
    ଅଚଳ uint32_t nb02 = src0->nb[2];                      \
    ଅଚଳ uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ଅଚଳ uint32_t nb0 = dst->nb[0];                        \
    ଅଚଳ uint32_t nb1 = dst->nb[1];                        \
    ଅଚଳ uint32_t nb2 = dst->nb[2];                        \
    ଅଚଳ uint32_t nb3 = dst->nb[3];

ସଂରଚନା htp_cumsum_context {
    ସଂରଚନା htp_ops_context * octx;
    size_t          src_row_size;
    size_t          dst_row_size;
    size_t          src_row_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    uint32_t        row_start;
};

#define htp_cumsum_preamble                                                \
    ସଂରଚନା htp_cumsum_context * cctx = (ସଂରଚନା htp_cumsum_context *) data; \
    ସଂରଚନା htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#ଯଦି __HVX_ARCH__ > 75
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    ଫେରନ୍ତୁ Q6_Vsf_vadd_VsfVsf(a, b);
}
#ଅନ୍ୟଥା
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    ଫେରନ୍ତୁ Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#endif  // __HVX_ARCH__ > 75

ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    ଅଚଳ HVX_Vector zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_in);

    ଫେରନ୍ତୁ v;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    ଫେରନ୍ତୁ hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_cumsum_row_f32(ଅଚଳ ଦଶମିକ * restrict src, ଦଶମିକ * restrict dst, uint32_t n) {
    ଅଚଳ uint32_t nvec = n / VLEN_FP32;
    ଅଚଳ uint32_t nloe = n % VLEN_FP32;

    HVX_Vector carry = Q6_V_vsplat_R(0);

    ପାଇଁ (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = *((ଅଚଳ HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        carry = hvx_splat_last_f32(v);
    }

    ଯଦି (nloe) {
        ଦଶମିକ acc = hvx_vec_get_f32(carry);
        ଅଚଳ ଦଶମିକ * src_tail = src + nvec * VLEN_FP32;
        ଦଶମିକ       * dst_tail = dst + nvec * VLEN_FP32;
        ପାଇଁ (uint32_t i = 0; i < nloe; i++) {
            acc        += src_tail[i];
            dst_tail[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

ସ୍ଥିର ଶୂନ୍ୟ cumsum_thread_f32_dma(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_cumsum_preamble;

    ଅଚଳ uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ଅଚଳ uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    ଯଦି (ir0 >= ir1) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ size_t src_row_size         = cctx->src_row_size;
    ଅଚଳ size_t dst_row_size         = cctx->dst_row_size;
    ଅଚଳ size_t src_row_size_aligned = cctx->src_row_size_aligned;
    ଅଚଳ size_t dst_row_size_aligned = cctx->dst_row_size_aligned;

    ଅଚଳ dma_addr_t src_data = src0->data;
    ଅଚଳ dma_addr_t dst_data = dst->data;

    uint8_t * src_spad = octx->src0_spad.data + (ith * src_row_size_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned * 2);

    ପାଇଁ (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_ଠେଲ(dma_q,
                       dma_make_data(dst_data, dst_spad + (spad_idx * dst_row_size_aligned)),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 0);

        dma_queue_ଠେଲ(dma_q,
                       dma_make_data(src_spad + (spad_idx * src_row_size_aligned),
                                     src_data + (ir * src_row_size)),
                       src_row_size_aligned, src_row_size, src_row_size, 1);
    }

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ପାଇଁ (uint32_t ir = ir0; ir < ir1; ir++) {
        ଦଶମିକ * dst_spad_row = (ଦଶମିକ *) dma_queue_କାଢ଼(dma_q).src;
        ଦଶମିକ * src_spad_row = (ଦଶମିକ *) dma_queue_କାଢ଼(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_row_f32(src_spad_row, dst_spad_row, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_ଠେଲ(dma_q,
                       dma_make_data(dst_data + (ir * dst_row_size), dst_spad_row),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 1);

        ଅଚଳ uint32_t next_row = ir + 2;
        ଯଦି (next_row < ir1) {
            dma_queue_ଠେଲ(dma_q,
                           dma_make_data(src_spad_row, src_data + (next_row * src_row_size)),
                           src_row_size_aligned, src_row_size, src_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(ଉପର_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ସ୍ଥିର ଶୂନ୍ୟ cumsum_thread_f32(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_cumsum_preamble;

    ଅଚଳ uint8_t * src_data = (ଅଚଳ uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    ଅଚଳ uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ଅଚଳ uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    ପାଇଁ (uint32_t ir = ir0; ir < ir1; ir++) {
        ଅଚଳ ଦଶମିକ * restrict src_row = (ଅଚଳ ଦଶମିକ *) (src_data + ir * cctx->src_row_size);
        ଦଶମିକ       * restrict dst_row = (ଦଶମିକ *)       (dst_data + ir * cctx->dst_row_size);
        hvx_cumsum_row_f32(src_row, dst_row, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(ଉପର_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_cumsum_f32(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ଅଚଳ uint32_t total_rows      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ size_t dst_data_row_size = dst->ne[0] * ଆକାର(ଦଶମିକ);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, ଆକାର(ଦଶମିକ), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
    }

    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads = octx->n_threads;

    ଅଚଳ size_t src_row_size         = src0->nb[1];
    ଅଚଳ size_t dst_row_size         = dst->nb[1];
    ଅଚଳ size_t src_row_size_aligned = hex_round_up(src_row_size, VLEN);
    ଅଚଳ size_t dst_row_size_aligned = hex_round_up(dst_row_size, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ଅଚଳ size_t spad_per_thread = 2 * (src_row_size_aligned + dst_row_size_aligned);

    octx->src0_spad.size_per_thread = src_row_size_aligned * 2;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned * 2;

    octx->src0_spad.size  = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size   = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data  = octx->ctx->vtcm_base;                        octx->src0_spad.src = ଖାଲି;
    octx->dst_spad.data   = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = ଖାଲି;

    ସଂରଚନା htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_row_size         = src_row_size,
        .dst_row_size         = dst_row_size,
        .src_row_size_aligned = src_row_size_aligned,
        .dst_row_size_aligned = dst_row_size_aligned,
        .rows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows           = nrows,
        .row_start            = row_start,
    };

    ଯଦି (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        ଯଦି (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32, &cctx, n_threads);
    } ଅନ୍ୟଥା {
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32_dma, &cctx, n_threads);
    }

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_cumsum(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * dst = octx->dst;

    ବାଛ (dst->type) {
        ସ୍ଥିତି HTP_TYPE_F32:
            ଫେରନ୍ତୁ op_cumsum_f32(octx);
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }
}
