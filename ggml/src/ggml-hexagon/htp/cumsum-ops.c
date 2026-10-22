#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাৱ <HAP_farf.h>
#অন্তর্ভাৱ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#অন্তর্ভাৱ "ggml-common.h"
#অন্তর্ভাৱ "hex-common.h"
#অন্তর্ভাৱ "hex-profile.h"
#অন্তর্ভাৱ "htp-ctx.h"
#অন্তর্ভাৱ "htp-ops.h"
#অন্তর্ভাৱ "htp-tensor.h"
#অন্তর্ভাৱ "hvx-types.h"
#অন্তর্ভাৱ "hvx-utils.h"
#অন্তর্ভাৱ "dma-queue.h"

#define htp_cumsum_tensors_preamble                         \
    ধ্রুৱ সংরচনা htp_tensor * restrict src0 = octx->src[0]; \
    ধ্রুৱ সংরচনা htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ধ্রুৱ uint32_t ne00 = src0->ne[0];                      \
    ধ্রুৱ uint32_t ne01 = src0->ne[1];                      \
    ধ্রুৱ uint32_t ne02 = src0->ne[2];                      \
    ধ্রুৱ uint32_t ne03 = src0->ne[3];                      \
                                                            \
    ধ্রুৱ uint32_t ne0 = dst->ne[0];                        \
    ধ্রুৱ uint32_t ne1 = dst->ne[1];                        \
    ধ্রুৱ uint32_t ne2 = dst->ne[2];                        \
    ধ্রুৱ uint32_t ne3 = dst->ne[3];                        \
                                                            \
    ধ্রুৱ uint32_t nb00 = src0->nb[0];                      \
    ধ্রুৱ uint32_t nb01 = src0->nb[1];                      \
    ধ্রুৱ uint32_t nb02 = src0->nb[2];                      \
    ধ্রুৱ uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ধ্রুৱ uint32_t nb0 = dst->nb[0];                        \
    ধ্রুৱ uint32_t nb1 = dst->nb[1];                        \
    ধ্রুৱ uint32_t nb2 = dst->nb[2];                        \
    ধ্রুৱ uint32_t nb3 = dst->nb[3];

সংরচনা htp_cumsum_context {
    সংরচনা htp_ops_context * octx;
    size_t          src_row_size;
    size_t          dst_row_size;
    size_t          src_row_size_aligned;
    size_t          dst_row_size_aligned;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    uint32_t        row_start;
};

#define htp_cumsum_preamble                                                \
    সংরচনা htp_cumsum_context * cctx = (সংরচনা htp_cumsum_context *) data; \
    সংরচনা htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#যদি __HVX_ARCH__ > 75
স্থির অন্তঃস্থ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    ঘূৰাই Q6_Vsf_vadd_VsfVsf(a, b);
}
#অন্যথা
স্থির অন্তঃস্থ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    ঘূৰাই Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#endif  // __HVX_ARCH__ > 75

স্থির অন্তঃস্থ HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    ধ্রুৱ HVX_Vector zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_in);

    ঘূৰাই v;
}

স্থির অন্তঃস্থ HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    ঘূৰাই hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

স্থির অন্তঃস্থ শূন্য hvx_cumsum_row_f32(ধ্রুৱ ভিন্ন * restrict src, ভিন্ন * restrict dst, uint32_t n) {
    ধ্রুৱ uint32_t nvec = n / VLEN_FP32;
    ধ্রুৱ uint32_t nloe = n % VLEN_FP32;

    HVX_Vector carry = Q6_V_vsplat_R(0);

    বাবে (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = *((ধ্রুৱ HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        carry = hvx_splat_last_f32(v);
    }

    যদি (nloe) {
        ভিন্ন acc = hvx_vec_get_f32(carry);
        ধ্রুৱ ভিন্ন * src_tail = src + nvec * VLEN_FP32;
        ভিন্ন       * dst_tail = dst + nvec * VLEN_FP32;
        বাবে (uint32_t i = 0; i < nloe; i++) {
            acc        += src_tail[i];
            dst_tail[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

স্থির শূন্য cumsum_thread_f32_dma(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_cumsum_preamble;

    ধ্রুৱ uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ধ্রুৱ uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    যদি (ir0 >= ir1) {
        ঘূৰাই;
    }

    ধ্রুৱ size_t src_row_size         = cctx->src_row_size;
    ধ্রুৱ size_t dst_row_size         = cctx->dst_row_size;
    ধ্রুৱ size_t src_row_size_aligned = cctx->src_row_size_aligned;
    ধ্রুৱ size_t dst_row_size_aligned = cctx->dst_row_size_aligned;

    ধ্রুৱ dma_addr_t src_data = src0->data;
    ধ্রুৱ dma_addr_t dst_data = dst->data;

    uint8_t * src_spad = octx->src0_spad.data + (ith * src_row_size_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned * 2);

    বাবে (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_ঠেলো(dma_q,
                       dma_make_data(dst_data, dst_spad + (spad_idx * dst_row_size_aligned)),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 0);

        dma_queue_ঠেলো(dma_q,
                       dma_make_data(src_spad + (spad_idx * src_row_size_aligned),
                                     src_data + (ir * src_row_size)),
                       src_row_size_aligned, src_row_size, src_row_size, 1);
    }

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    বাবে (uint32_t ir = ir0; ir < ir1; ir++) {
        ভিন্ন * dst_spad_row = (ভিন্ন *) dma_queue_বেৰ_কৰো(dma_q).src;
        ভিন্ন * src_spad_row = (ভিন্ন *) dma_queue_বেৰ_কৰো(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_row_f32(src_spad_row, dst_spad_row, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_ঠেলো(dma_q,
                       dma_make_data(dst_data + (ir * dst_row_size), dst_spad_row),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 1);

        ধ্রুৱ uint32_t next_row = ir + 2;
        যদি (next_row < ir1) {
            dma_queue_ঠেলো(dma_q,
                           dma_make_data(src_spad_row, src_data + (next_row * src_row_size)),
                           src_row_size_aligned, src_row_size, src_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(উপৰেৰ_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

স্থির শূন্য cumsum_thread_f32(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_cumsum_preamble;

    ধ্রুৱ uint8_t * src_data = (ধ্রুৱ uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    ধ্রুৱ uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ধ্রুৱ uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    বাবে (uint32_t ir = ir0; ir < ir1; ir++) {
        ধ্রুৱ ভিন্ন * restrict src_row = (ধ্রুৱ ভিন্ন *) (src_data + ir * cctx->src_row_size);
        ভিন্ন       * restrict dst_row = (ভিন্ন *)       (dst_data + ir * cctx->dst_row_size);
        hvx_cumsum_row_f32(src_row, dst_row, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(উপৰেৰ_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

পূর্ণাঙ্ক op_cumsum_f32(সংরচনা htp_ops_context * octx) {
    ধ্রুৱ সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * dst  = octx->dst;

    ধ্রুৱ uint32_t total_rows      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুৱ size_t dst_data_row_size = dst->ne[0] * পরিমাণ(ভিন্ন);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, পরিমাণ(ভিন্ন), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ধ্রুৱ সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ঘূৰাই HTP_STATUS_OK;
    }

    ধ্রুৱ uint32_t n_threads = octx->n_threads;

    ধ্রুৱ size_t src_row_size         = src0->nb[1];
    ধ্রুৱ size_t dst_row_size         = dst->nb[1];
    ধ্রুৱ size_t src_row_size_aligned = hex_round_up(src_row_size, VLEN);
    ধ্রুৱ size_t dst_row_size_aligned = hex_round_up(dst_row_size, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ধ্রুৱ size_t spad_per_thread = 2 * (src_row_size_aligned + dst_row_size_aligned);

    octx->src0_spad.size_per_thread = src_row_size_aligned * 2;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned * 2;

    octx->src0_spad.size  = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size   = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data  = octx->ctx->vtcm_base;                        octx->src0_spad.src = রিক্ত;
    octx->dst_spad.data   = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = রিক্ত;

    সংরচনা htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_row_size         = src_row_size,
        .dst_row_size         = dst_row_size,
        .src_row_size_aligned = src_row_size_aligned,
        .dst_row_size_aligned = dst_row_size_aligned,
        .rows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows           = nrows,
        .row_start            = row_start,
    };

    যদি (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ঘূৰাই HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32, &cctx, n_threads);
    } অন্যথা {
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32_dma, &cctx, n_threads);
    }

    ঘূৰাই HTP_STATUS_OK;
}

পূর্ণাঙ্ক op_cumsum(সংরচনা htp_ops_context * octx) {
    ধ্রুৱ সংরচনা htp_tensor * dst = octx->dst;

    ৱিকল্প (dst->type) {
        পক্ষ HTP_TYPE_F32:
            ঘূৰাই op_cumsum_f32(octx);
        স্ৱভাৱ:
            ঘূৰাই HTP_STATUS_NO_SUPPORT;
    }
}
