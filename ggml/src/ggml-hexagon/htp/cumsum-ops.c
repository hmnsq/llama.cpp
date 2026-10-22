#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#define GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-types.h"
#شامل "hvx-utils.h"
#شامل "dma-queue.h"

#define htp_cumsum_tensors_preamble                         \
    ثابت ساختار htp_tensor * restrict src0 = octx->src[0]; \
    ثابت ساختار htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ثابت uint32_t ne00 = src0->ne[0];                      \
    ثابت uint32_t ne01 = src0->ne[1];                      \
    ثابت uint32_t ne02 = src0->ne[2];                      \
    ثابت uint32_t ne03 = src0->ne[3];                      \
                                                            \
    ثابت uint32_t ne0 = dst->ne[0];                        \
    ثابت uint32_t ne1 = dst->ne[1];                        \
    ثابت uint32_t ne2 = dst->ne[2];                        \
    ثابت uint32_t ne3 = dst->ne[3];                        \
                                                            \
    ثابت uint32_t nb00 = src0->nb[0];                      \
    ثابت uint32_t nb01 = src0->nb[1];                      \
    ثابت uint32_t nb02 = src0->nb[2];                      \
    ثابت uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ثابت uint32_t nb0 = dst->nb[0];                        \
    ثابت uint32_t nb1 = dst->nb[1];                        \
    ثابت uint32_t nb2 = dst->nb[2];                        \
    ثابت uint32_t nb3 = dst->nb[3];

ساختار htp_cumsum_context {
    ساختار htp_ops_context * octx;
    size_t          src_row_size;
    size_t          dst_row_size;
    size_t          src_row_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    uint32_t        row_start;
};

#define htp_cumsum_preamble                                                \
    ساختار htp_cumsum_context * cctx = (ساختار htp_cumsum_context *) data; \
    ساختار htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#اگر __HVX_ARCH__ > 75
ایستا درونخطی HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    بازگشت Q6_Vsf_vadd_VsfVsf(a, b);
}
#وگرنه
ایستا درونخطی HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    بازگشت Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#endif  // __HVX_ARCH__ > 75

ایستا درونخطی HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    ثابت HVX_Vector zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_in);

    بازگشت v;
}

ایستا درونخطی HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    بازگشت hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

ایستا درونخطی تهی hvx_cumsum_row_f32(ثابت اعشاری * restrict src, اعشاری * restrict dst, uint32_t n) {
    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    HVX_Vector carry = Q6_V_vsplat_R(0);

    برای (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = *((ثابت HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        carry = hvx_splat_last_f32(v);
    }

    اگر (nloe) {
        اعشاری acc = hvx_vec_get_f32(carry);
        ثابت اعشاری * src_tail = src + nvec * VLEN_FP32;
        اعشاری       * dst_tail = dst + nvec * VLEN_FP32;
        برای (uint32_t i = 0; i < nloe; i++) {
            acc        += src_tail[i];
            dst_tail[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

ایستا تهی cumsum_thread_f32_dma(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    htp_cumsum_preamble;

    ثابت uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    اگر (ir0 >= ir1) {
        بازگشت;
    }

    ثابت size_t src_row_size         = cctx->src_row_size;
    ثابت size_t dst_row_size         = cctx->dst_row_size;
    ثابت size_t src_row_size_aligned = cctx->src_row_size_aligned;
    ثابت size_t dst_row_size_aligned = cctx->dst_row_size_aligned;

    ثابت dma_addr_t src_data = src0->data;
    ثابت dma_addr_t dst_data = dst->data;

    uint8_t * src_spad = octx->src0_spad.data + (ith * src_row_size_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned * 2);

    برای (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_افزودن(dma_q,
                       dma_make_data(dst_data, dst_spad + (spad_idx * dst_row_size_aligned)),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 0);

        dma_queue_افزودن(dma_q,
                       dma_make_data(src_spad + (spad_idx * src_row_size_aligned),
                                     src_data + (ir * src_row_size)),
                       src_row_size_aligned, src_row_size, src_row_size, 1);
    }

    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];

    برای (uint32_t ir = ir0; ir < ir1; ir++) {
        اعشاری * dst_spad_row = (اعشاری *) dma_queue_برداشت(dma_q).src;
        اعشاری * src_spad_row = (اعشاری *) dma_queue_برداشت(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_row_f32(src_spad_row, dst_spad_row, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_افزودن(dma_q,
                       dma_make_data(dst_data + (ir * dst_row_size), dst_spad_row),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 1);

        ثابت uint32_t next_row = ir + 2;
        اگر (next_row < ir1) {
            dma_queue_افزودن(dma_q,
                           dma_make_data(src_spad_row, src_data + (next_row * src_row_size)),
                           src_row_size_aligned, src_row_size, src_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(بالا_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ایستا تهی cumsum_thread_f32(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    htp_cumsum_preamble;

    ثابت uint8_t * src_data = (ثابت uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    ثابت uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ثابت uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    برای (uint32_t ir = ir0; ir < ir1; ir++) {
        ثابت اعشاری * restrict src_row = (ثابت اعشاری *) (src_data + ir * cctx->src_row_size);
        اعشاری       * restrict dst_row = (اعشاری *)       (dst_data + ir * cctx->dst_row_size);
        hvx_cumsum_row_f32(src_row, dst_row, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(بالا_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحیح op_cumsum_f32(ساختار htp_ops_context * octx) {
    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    ثابت uint32_t total_rows      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_data_row_size = dst->ne[0] * اندازه(اعشاری);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    اگر (octx->ctx->mdev.شمار > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, اندازه(اعشاری), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_start = range.start;
        nrows     = range.شمار;
    }

    اگر (nrows == 0) {
        بازگشت HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads = octx->n_threads;

    ثابت size_t src_row_size         = src0->nb[1];
    ثابت size_t dst_row_size         = dst->nb[1];
    ثابت size_t src_row_size_aligned = hex_round_up(src_row_size, VLEN);
    ثابت size_t dst_row_size_aligned = hex_round_up(dst_row_size, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ثابت size_t spad_per_thread = 2 * (src_row_size_aligned + dst_row_size_aligned);

    octx->src0_spad.size_per_thread = src_row_size_aligned * 2;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned * 2;

    octx->src0_spad.size  = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size   = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data  = octx->ctx->vtcm_base;                        octx->src0_spad.src = پوچ;
    octx->dst_spad.data   = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = پوچ;

    ساختار htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_row_size         = src_row_size,
        .dst_row_size         = dst_row_size,
        .src_row_size_aligned = src_row_size_aligned,
        .dst_row_size_aligned = dst_row_size_aligned,
        .rows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows           = nrows,
        .row_start            = row_start,
    };

    اگر (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            بازگشت HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32, &cctx, n_threads);
    } وگرنه {
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32_dma, &cctx, n_threads);
    }

    بازگشت HTP_STATUS_OK;
}

صحیح op_cumsum(ساختار htp_ops_context * octx) {
    ثابت ساختار htp_tensor * dst = octx->dst;

    انتخاب (dst->type) {
        حالت HTP_TYPE_F32:
            بازگشت op_cumsum_f32(octx);
        پیشفرض:
            بازگشت HTP_STATUS_NO_SUPPORT;
    }
}
