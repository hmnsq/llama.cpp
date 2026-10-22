#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-types.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "dma-queue.h"

#define htp_cumsum_tensors_preamble                         \
    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0]; \
    ध्रुव संरचना htp_tensor * restrict dst  = octx->dst;    \
                                                            \
    ध्रुव uint32_t ne00 = src0->ne[0];                      \
    ध्रुव uint32_t ne01 = src0->ne[1];                      \
    ध्रुव uint32_t ne02 = src0->ne[2];                      \
    ध्रुव uint32_t ne03 = src0->ne[3];                      \
                                                            \
    ध्रुव uint32_t ne0 = dst->ne[0];                        \
    ध्रुव uint32_t ne1 = dst->ne[1];                        \
    ध्रुव uint32_t ne2 = dst->ne[2];                        \
    ध्रुव uint32_t ne3 = dst->ne[3];                        \
                                                            \
    ध्रुव uint32_t nb00 = src0->nb[0];                      \
    ध्रुव uint32_t nb01 = src0->nb[1];                      \
    ध्रुव uint32_t nb02 = src0->nb[2];                      \
    ध्रुव uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ध्रुव uint32_t nb0 = dst->nb[0];                        \
    ध्रुव uint32_t nb1 = dst->nb[1];                        \
    ध्रुव uint32_t nb2 = dst->nb[2];                        \
    ध्रुव uint32_t nb3 = dst->nb[3];

संरचना htp_cumsum_context {
    संरचना htp_ops_context * octx;
    माप_प्रकार          src_row_size;
    माप_प्रकार          dst_row_size;
    माप_प्रकार          src_row_size_aligned;
    माप_प्रकार          dst_row_size_aligned;
    uint32_t        rows_per_thread;
    uint32_t        total_rows;
    uint32_t        row_start;
};

#define htp_cumsum_preamble                                                \
    संरचना htp_cumsum_context * cctx = (संरचना htp_cumsum_context *) data; \
    संरचना htp_ops_context *    octx = cctx->octx;                         \
    htp_cumsum_tensors_preamble;                                           \
    dma_queue * dma_q = octx->ctx->dma[ith];

// ---------------------------------------------------------------------------
// HVX prefix scan helpers
// ---------------------------------------------------------------------------

#यदि __HVX_ARCH__ > 75
स्थिर अन्तःस्थ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    प्रत्यागम Q6_Vsf_vadd_VsfVsf(a, b);
}
#अन्यथा
स्थिर अन्तःस्थ HVX_Vector hvx_cumsum_vadd(HVX_Vector a, HVX_Vector b) {
    प्रत्यागम Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
#endif  // __HVX_ARCH__ > 75

स्थिर अन्तःस्थ HVX_Vector hvx_prefix_scan_f32(HVX_Vector v, HVX_Vector carry_in) {
    ध्रुव HVX_Vector zero = Q6_V_vsplat_R(0);

    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  4));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero,  8));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = hvx_cumsum_vadd(v, Q6_V_vlalign_VVR(v, zero, 64));
    v = hvx_cumsum_vadd(v, carry_in);

    प्रत्यागम v;
}

स्थिर अन्तःस्थ HVX_Vector hvx_splat_last_f32(HVX_Vector v) {
    प्रत्यागम hvx_vec_repl4(Q6_V_vror_VR(v, 124));
}

स्थिर अन्तःस्थ शून्य hvx_cumsum_row_f32(ध्रुव भिन्न * restrict src, भिन्न * restrict dst, uint32_t n) {
    ध्रुव uint32_t nvec = n / VLEN_FP32;
    ध्रुव uint32_t nloe = n % VLEN_FP32;

    HVX_Vector carry = Q6_V_vsplat_R(0);

    प्रति (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v = *((ध्रुव HVX_UVector *) (src + i * VLEN_FP32));
        v = hvx_prefix_scan_f32(v, carry);
        hvx_vec_store_u(dst + i * VLEN_FP32, VLEN, v);
        carry = hvx_splat_last_f32(v);
    }

    यदि (nloe) {
        भिन्न acc = hvx_vec_get_f32(carry);
        ध्रुव भिन्न * src_tail = src + nvec * VLEN_FP32;
        भिन्न       * dst_tail = dst + nvec * VLEN_FP32;
        प्रति (uint32_t i = 0; i < nloe; i++) {
            acc        += src_tail[i];
            dst_tail[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Per thread worker: Double-buffered DMA
// ---------------------------------------------------------------------------

स्थिर शून्य cumsum_thread_f32_dma(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_cumsum_preamble;

    ध्रुव uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ध्रुव uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    यदि (ir0 >= ir1) {
        प्रत्यागम;
    }

    ध्रुव माप_प्रकार src_row_size         = cctx->src_row_size;
    ध्रुव माप_प्रकार dst_row_size         = cctx->dst_row_size;
    ध्रुव माप_प्रकार src_row_size_aligned = cctx->src_row_size_aligned;
    ध्रुव माप_प्रकार dst_row_size_aligned = cctx->dst_row_size_aligned;

    ध्रुव dma_addr_t src_data = src0->data;
    ध्रुव dma_addr_t dst_data = dst->data;

    uint8_t * src_spad = octx->src0_spad.data + (ith * src_row_size_aligned * 2);
    uint8_t * dst_spad = octx->dst_spad.data  + (ith * dst_row_size_aligned * 2);

    प्रति (uint32_t ir = ir0, spad_idx = 0; ir < ir1 && spad_idx < 2; ir++, spad_idx++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_धकेलो(dma_q,
                       dma_make_data(dst_data, dst_spad + (spad_idx * dst_row_size_aligned)),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 0);

        dma_queue_धकेलो(dma_q,
                       dma_make_data(src_spad + (spad_idx * src_row_size_aligned),
                                     src_data + (ir * src_row_size)),
                       src_row_size_aligned, src_row_size, src_row_size, 1);
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    प्रति (uint32_t ir = ir0; ir < ir1; ir++) {
        भिन्न * dst_spad_row = (भिन्न *) dma_queue_निकालो(dma_q).src;
        भिन्न * src_spad_row = (भिन्न *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        hvx_cumsum_row_f32(src_spad_row, dst_spad_row, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_queue_धकेलो(dma_q,
                       dma_make_data(dst_data + (ir * dst_row_size), dst_spad_row),
                       dst_row_size, dst_row_size_aligned, dst_row_size, 1);

        ध्रुव uint32_t next_row = ir + 2;
        यदि (next_row < ir1) {
            dma_queue_धकेलो(dma_q,
                           dma_make_data(src_spad_row, src_data + (next_row * src_row_size)),
                           src_row_size_aligned, src_row_size, src_row_size, 1);
        }
    }

    dma_queue_flush(dma_q);

    FARF(ऊपरी_2, "cumsum-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

स्थिर शून्य cumsum_thread_f32(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_cumsum_preamble;

    ध्रुव uint8_t * src_data = (ध्रुव uint8_t *) src0->data;
    uint8_t *       dst_data = (uint8_t *) dst->data;

    ध्रुव uint32_t ir0 = cctx->row_start + cctx->rows_per_thread * ith;
    ध्रुव uint32_t ir1 = MIN(ir0 + cctx->rows_per_thread, cctx->row_start + cctx->total_rows);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    प्रति (uint32_t ir = ir0; ir < ir1; ir++) {
        ध्रुव भिन्न * restrict src_row = (ध्रुव भिन्न *) (src_data + ir * cctx->src_row_size);
        भिन्न       * restrict dst_row = (भिन्न *)       (dst_data + ir * cctx->dst_row_size);
        hvx_cumsum_row_f32(src_row, dst_row, ne00);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    FARF(ऊपरी_2, "cumsum-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ir0, ir1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

पूर्णाङ्क op_cumsum_f32(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    ध्रुव uint32_t total_rows      = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव माप_प्रकार dst_data_row_size = dst->ne[0] * परिमाण(भिन्न);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    यदि (octx->ctx->mdev.गिनती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, परिमाण(भिन्न), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_STATUS_OK;
    }

    ध्रुव uint32_t n_threads = octx->n_threads;

    ध्रुव माप_प्रकार src_row_size         = src0->nb[1];
    ध्रुव माप_प्रकार dst_row_size         = dst->nb[1];
    ध्रुव माप_प्रकार src_row_size_aligned = hex_round_up(src_row_size, VLEN);
    ध्रुव माप_प्रकार dst_row_size_aligned = hex_round_up(dst_row_size, VLEN);

    // 2 ping-pong buffers per thread for src and dst
    ध्रुव माप_प्रकार spad_per_thread = 2 * (src_row_size_aligned + dst_row_size_aligned);

    octx->src0_spad.size_per_thread = src_row_size_aligned * 2;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned * 2;

    octx->src0_spad.size  = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size   = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data  = octx->ctx->vtcm_base;                        octx->src0_spad.src = रिक्त;
    octx->dst_spad.data   = octx->src0_spad.data + octx->src0_spad.size; octx->dst_spad.src  = रिक्त;

    संरचना htp_cumsum_context cctx = {
        .octx                 = octx,
        .src_row_size         = src_row_size,
        .dst_row_size         = dst_row_size,
        .src_row_size_aligned = src_row_size_aligned,
        .dst_row_size_aligned = dst_row_size_aligned,
        .rows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_rows           = nrows,
        .row_start            = row_start,
    };

    यदि (octx->ctx->vtcm_size < spad_per_thread * n_threads) {
        यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            प्रत्यागम HTP_STATUS_NO_SUPPORT;
        }
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32, &cctx, n_threads);
    } अन्यथा {
        work_queue_run(octx->ctx->work_queue, cumsum_thread_f32_dma, &cctx, n_threads);
    }

    प्रत्यागम HTP_STATUS_OK;
}

पूर्णाङ्क op_cumsum(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * dst = octx->dst;

    विकल्प (dst->type) {
        पक्ष HTP_TYPE_F32:
            प्रत्यागम op_cumsum_f32(octx);
        स्वभाव:
            प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }
}
