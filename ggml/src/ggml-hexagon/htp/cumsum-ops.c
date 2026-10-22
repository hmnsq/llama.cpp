#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "hvx-types.h"
#অন্তর্ভাব "hvx-utils.h"
#অন্তর্ভাব "dma-queue.h"

#define htp_cumsum_tensors_preamble                         \
    ধ্রুবক কাঠামো htp_tensor * restrict src0 = octx->src[0]; \
    ধ্রুবক কাঠামো htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ধ্রুবক uint32_t ne00 = src0->ne[0];                      \
    ধ্রুবক uint32_t ne01 = src0->ne[1];                      \
    ধ্রুবক uint32_t ne02 = src0->ne[2];                      \
    ধ্রুবক uint32_t ne03 = src0->ne[3];                      \
                                                            \
    ধ্রুবক uint32_t ne0 = dst->ne[0];                        \
    ধ্রুবক uint32_t ne1 = dst->ne[1];                        \
    ধ্রুবক uint32_t ne2 = dst->ne[2];                        \
    ধ্রুবক uint32_t ne3 = dst->ne[3];                        \
                                                            \
    ধ্রুবক uint32_t nb00 = src0->nb[0];                      \
    ধ্রুবক uint32_t nb01 = src0->nb[1];                      \
    ধ্রুবক uint32_t nb02 = src0->nb[2];                      \
    ধ্রুবক uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ধ্রুবক uint32_t nb0 = dst->nb[0];                        \
    ধ্রুবক uint32_t nb1 = dst->nb[1];                        \
    ধ্রুবক uint32_t nb2 = dst->nb[2];                        \
    ধ্রুবক uint32_t nb3 = dst->nb[3];

কাঠামো htp_cumsum_context {
    কাঠামো htp_ops_context * octx;
    size_t          src_row_আকার;
    size_t          dst_row_আকার;
    size_t          src_row_আকার_aligned;
    size_t          dst_row_আকার_aligned;
    uint32_t        rows_per_thread;
    uint32_t        যোগ_rows;
    uint32_t        row_start;
};

#define htp_cumsum_preamble                                                \
    কাঠামো htp_cumsum_context * cctx = (কাঠামো htp_cumsum_context *) তথ্য; \
    কাঠামো htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#যদি __HVX_ARCH__ > 75
স্থির অন্তঃরেখ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    ফেরত Q6_Vsf_vadd_VsfVsf(a, b);
}
#অন্যথা
স্থির অন্তঃরেখ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    ফেরত Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#endif  // __HVX_ARCH__ > 75

স্থির অন্তঃরেখ HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    ধ্রুবক HVX_Vector zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_in);

    ফেরত v;
}

স্থির অন্তঃরেখ HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    ফেরত hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

স্থির অন্তঃরেখ শূন্য hvx_cumsum_row_f32(ধ্রুবক দশমিক * restrict src, দশমিক * restrict dst, uint32_t n) {
    ধ্রুবক uint32_t nvec = n / VLEN_FP32;
    ধ্রুবক uint32_t nloe = n % VLEN_FP32;

    HVX_Vector carry = Q6_V_vsplat_R(0);

    জন্য (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = *((ধ্রুবক HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        carry = hvx_splat_last_f32(v);
    }

    যদি (nloe) {
        দশমিক acc = hvx_vec_get_f32(carry);
        ধ্রুবক দশমিক * src_লেজ = src + nvec * VLEN_FP32;
        দশমিক       * dst_লেজ = dst + nvec * VLEN_FP32;
        জন্য (uint32_t i = 0; i < nloe; i++) {
            acc        += src_লেজ[i];
            dst_লেজ[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

স্থির শূন্য cumsum_thread_f32_dma(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    htp_cumsum_preamble;

    ধ্রুবক uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ধ্রুবক uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->যোগ_rows);

    যদি (ir0 >= ir1) {
        ফেরত;
    }

    ধ্রুবক size_t src_row_আকার         = cctx->src_row_আকার;
    ধ্রুবক size_t dst_row_আকার         = cctx->dst_row_আকার;
    ধ্রুবক size_t src_row_আকার_aligned = cctx->src_row_আকার_aligned;
    ধ্রুবক size_t dst_row_আকার_aligned = cctx->dst_row_আকার_aligned;

    ধ্রুবক dma_addr_t src_তথ্য = src0->তথ্য;
    ধ্রুবক dma_addr_t dst_তথ্য = dst->তথ্য;

    uint8_t * src_spad = octx->src0_spad.তথ্য + (ith * src_row_আকার_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.তথ্য  + (ith * dst_row_আকার_aligned * 2);

    জন্য (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(dst_তথ্য, dst_spad + (spad_idx * dst_row_আকার_aligned)),
                       dst_row_আকার, dst_row_আকার_aligned, dst_row_আকার, 0);

        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(src_spad + (spad_idx * src_row_আকার_aligned),
                                     src_তথ্য + (ir * src_row_আকার)),
                       src_row_আকার_aligned, src_row_আকার, src_row_আকার, 1);
    }

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];

    জন্য (uint32_t ir = ir0; ir < ir1; ir++) {
        দশমিক * dst_spad_row = (দশমিক *) dma_queue_বের_করো(dma_q).src;
        দশমিক * src_spad_row = (দশমিক *) dma_queue_বের_করো(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_row_f32(src_spad_row, dst_spad_row, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_ঠেলো(dma_q,
                       dma_make_তথ্য(dst_তথ্য + (ir * dst_row_আকার), dst_spad_row),
                       dst_row_আকার, dst_row_আকার_aligned, dst_row_আকার, 1);

        ধ্রুবক uint32_t পরবর্তী_row = ir + 2;
        যদি (পরবর্তী_row < ir1) {
            dma_queue_ঠেলো(dma_q,
                           dma_make_তথ্য(src_spad_row, src_তথ্য + (পরবর্তী_row * src_row_আকার)),
                           src_row_আকার_aligned, src_row_আকার, src_row_আকার, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(উপরের_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

স্থির শূন্য cumsum_thread_f32(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    htp_cumsum_preamble;

    ধ্রুবক uint8_t * src_তথ্য = (ধ্রুবক uint8_t *) src0->তথ্য;
    uint8_t *       dst_তথ্য = (uint8_t *) dst->তথ্য;

    ধ্রুবক uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ধ্রুবক uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->যোগ_rows);

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    জন্য (uint32_t ir = ir0; ir < ir1; ir++) {
        ধ্রুবক দশমিক * restrict src_row = (ধ্রুবক দশমিক *) (src_তথ্য + ir * cctx->src_row_আকার);
        দশমিক       * restrict dst_row = (দশমিক *)       (dst_তথ্য + ir * cctx->dst_row_আকার);
        hvx_cumsum_row_f32(src_row, dst_row, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(উপরের_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

পূর্ণসংখ্যা op_cumsum_f32(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    ধ্রুবক uint32_t যোগ_rows      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুবক size_t dst_তথ্য_row_আকার = dst->ne[0] * আকার(দশমিক);

    uint32_t row_start = 0;
    uint32_t nrows     = যোগ_rows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, আকার(দশমিক), (uint32_t) dst_তথ্য_row_আকার, &rows_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(যোগ_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;

    ধ্রুবক size_t src_row_আকার         = src0->nb[1];
    ধ্রুবক size_t dst_row_আকার         = dst->nb[1];
    ধ্রুবক size_t src_row_আকার_aligned = hex_round_up(src_row_আকার, VLEN);
    ধ্রুবক size_t dst_row_আকার_aligned = hex_round_up(dst_row_আকার, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ধ্রুবক size_t spad_per_thread = 2 * (src_row_আকার_aligned + dst_row_আকার_aligned);

    octx->src0_spad.আকার_per_thread = src_row_আকার_aligned * 2;
    octx->dst_spad.আকার_per_thread  = dst_row_আকার_aligned * 2;

    octx->src0_spad.আকার_2  = n_threads * octx->src0_spad.আকার_per_thread;
    octx->dst_spad.আকার_2   = n_threads * octx->dst_spad.আকার_per_thread;

    octx->src0_spad.তথ্য  = octx->ctx->vtcm_base;                        octx->src0_spad.src = ফাঁকা;
    octx->dst_spad.তথ্য   = octx->src0_spad.তথ্য + octx->src0_spad.আকার_2; octx->dst_spad.src  = ফাঁকা;

    কাঠামো htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_row_আকার         = src_row_আকার,
        .dst_row_আকার         = dst_row_আকার,
        .src_row_আকার_aligned = src_row_আকার_aligned,
        .dst_row_আকার_aligned = dst_row_আকার_aligned,
        .rows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .যোগ_rows           = nrows,
        .row_start            = row_start,
    };

    যদি (octx->ctx->vtcm_আকার < spad_per_thread * n_threads) {
        যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            ফেরত HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32, &cctx, n_threads);
    } অন্যথা {
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32_dma, &cctx, n_threads);
    }

    ফেরত HTP_STATUS_OK;
}

পূর্ণসংখ্যা op_cumsum(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * dst = octx->dst;

    নির্বাচন (dst->type) {
        ক্ষেত্র HTP_TYPE_F32:
            ফেরত op_cumsum_f32(octx);
        পূর্বনির্ধারিত:
            ফেরত HTP_STATUS_NO_SUPPORT;
    }
}
