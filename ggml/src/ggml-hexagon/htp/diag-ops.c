#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-types.h"
#شامل "hex-utils.h"
#شامل "hvx-copy.h"
#شامل "dma-queue.h"

#تعریف_کرو htp_diag_tensors_preamble                           \
    ثابت ساخت htp_tensor * محدود src0 = octx->src[0]; \
    ثابت ساخت htp_tensor * محدود dst  = octx->dst;    \
                                                            \
    ثابت uint32_t ne02 = src0->ne[2];                      \
                                                            \
    ثابت uint32_t ne0 = dst->ne[0];                        \
    ثابت uint32_t ne1 = dst->ne[1];                        \
                                                            \
    ثابت uint32_t nb02 = src0->nb[2];                      \
    ثابت uint32_t nb03 = src0->nb[3];                      \
                                                            \
    ثابت uint32_t nb1 = dst->nb[1];                        \
    ثابت uint32_t nb2 = dst->nb[2];                        \
    ثابت uint32_t nb3 = dst->nb[3];

ساخت htp_diag_context {
    ساخت htp_ops_context * octx;
    size_t          src_batch_سائز;
    size_t          dst_صف_سائز;
    size_t          src_batch_سائز_aligned;
    size_t          dst_صف_سائز_aligned;
    uint32_t        batches_فی_thread;
    uint32_t        کل_batches;
    uint32_t        batch_آغاز;
};

#تعریف_کرو htp_diag_preamble                                              \
    ساخت htp_diag_context * dctx = (ساخت htp_diag_context *) ڈیٹا; \
    ساخت htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

ساکن اندرونی خالی hvx_diag_صف_f32(ثابت اعشاری * محدود src, اعشاری * محدود dst,
                                    uint32_t صف_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[صف_idx] = src[صف_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

ساکن خالی diag_thread_f32_dma(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_diag_preamble;
    dma_قطار * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t ib0 = dctx->batch_آغاز + dctx->batches_فی_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_فی_thread, dctx->batch_آغاز + dctx->کل_batches);

    اگر (ib0 >= ib1) {
        واپس;
    }

    ثابت size_t src_batch_سائز         = dctx->src_batch_سائز;
    ثابت size_t dst_صف_سائز           = dctx->dst_صف_سائز;
    ثابت size_t src_batch_سائز_aligned = dctx->src_batch_سائز_aligned;
    ثابت size_t dst_صف_سائز_aligned   = dctx->dst_صف_سائز_aligned;

    ثابت dma_addr_t src_ڈیٹا = src0->ڈیٹا;
    ثابت dma_addr_t dst_ڈیٹا = dst->ڈیٹا;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.ڈیٹا + (ith * src_batch_سائز_aligned);
    uint8_t * dst_spad = octx->dst_spad.ڈیٹا  + (ith * dst_صف_سائز_aligned);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    برائے (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت dma_addr_t src_batch = src_ڈیٹا + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_قطار_ڈالو(dma_q,
                       dma_make_ڈیٹا(src_spad, src_batch),
                       src_batch_سائز_aligned, src_batch_سائز, src_batch_سائز, 1);
        dma_قطار_flush(dma_q);

        ثابت اعشاری * src_spad_f32 = (ثابت اعشاری *) src_spad;
        اعشاری       * dst_spad_f32 = (اعشاری *) dst_spad;

        برائے (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_صف_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ثابت dma_addr_t dst_صف = dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_قطار_ڈالو(dma_q,
                           dma_make_ڈیٹا(dst_صف, dst_spad),
                           dst_صف_سائز, dst_صف_سائز_aligned, dst_صف_سائز, 1);
            dma_قطار_flush(dma_q);
        }
    }

    FARF(بلند_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ساکن خالی diag_thread_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    htp_diag_preamble;

    ثابت uint8_t * src_ڈیٹا = (ثابت uint8_t *) src0->ڈیٹا;
    uint8_t *       dst_ڈیٹا = (uint8_t *) dst->ڈیٹا;

    ثابت uint32_t ib0 = dctx->batch_آغاز + dctx->batches_فی_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_فی_thread, dctx->batch_آغاز + dctx->کل_batches);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    برائے (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت اعشاری * محدود src_batch = (ثابت اعشاری *)(src_ڈیٹا + i3 * nb03 + i2 * nb02);

        برائے (uint32_t i1 = 0; i1 < ne1; i1++) {
            اعشاری * محدود dst_صف = (اعشاری *)(dst_ڈیٹا + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_صف_f32(src_batch, dst_صف, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(بلند_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحیح op_diag_f32(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    ثابت uint32_t کل_batches = src0->ne[2] * src0->ne[3];
    ثابت size_t dst_batch_سائز  = dst->ne[1] * dst->nb[1];

    uint32_t batch_آغاز = 0;
    uint32_t nbatches    = کل_batches;

    اگر (octx->ctx->mdev.گنتی > 1) {
        bool can_بانٹو = htp_tensor_mdev_ڈیٹا_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == حجم(اعشاری)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_فی_chunk = 1;
        اگر (can_بانٹو) {
            اگر (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) == 0)) {
                batches_فی_chunk = 1;
            } ورنہ اگر (dst->nb[2] == dst_batch_سائز &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_فی_chunk = (dst_batch_سائز > 0) ? (HEX_L2_سطر_سائز / hex_عاد_اعظم_u32(dst_batch_سائز, HEX_L2_سطر_سائز)) : 1;
            } ورنہ {
                can_بانٹو = false;
            }
        }

        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_batches, can_بانٹو ? batches_فی_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        batch_آغاز = range.آغاز;
        nbatches    = range.گنتی;
    }

    اگر (nbatches == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے     = octx->n_دھاگے;

    ثابت size_t src_batch_سائز         = src0->ne[0] * حجم(اعشاری);
    ثابت size_t dst_صف_سائز           = dst->ne[0] * حجم(اعشاری);
    ثابت size_t src_batch_سائز_aligned = hex_باری_اوپر(src_batch_سائز, VLEN);
    ثابت size_t dst_صف_سائز_aligned   = hex_باری_اوپر(dst_صف_سائز, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ثابت size_t spad_فی_thread = src_batch_سائز_aligned + dst_صف_سائز_aligned;

    octx->src0_spad.سائز_فی_thread = src_batch_سائز_aligned;
    octx->dst_spad.سائز_فی_thread  = dst_صف_سائز_aligned;

    octx->src0_spad.سائز = n_دھاگے * octx->src0_spad.سائز_فی_thread;
    octx->dst_spad.سائز  = n_دھاگے * octx->dst_spad.سائز_فی_thread;

    octx->src0_spad.ڈیٹا = octx->ctx->vtcm_بنیاد;                        octx->src0_spad.src = تہی;
    octx->dst_spad.ڈیٹا  = octx->src0_spad.ڈیٹا + octx->src0_spad.سائز; octx->dst_spad.src  = تہی;

    ساخت htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_سائز         = src_batch_سائز,
        .dst_صف_سائز           = dst_صف_سائز,
        .src_batch_سائز_aligned = src_batch_سائز_aligned,
        .dst_صف_سائز_aligned   = dst_صف_سائز_aligned,
        .batches_فی_thread     = fastdiv(nbatches + n_دھاگے - 1, &octx->n_دھاگے_div),
        .کل_batches          = nbatches,
        .batch_آغاز            = batch_آغاز,
    };

    اگر (octx->ctx->vtcm_سائز < spad_فی_thread * n_دھاگے) {
        اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            واپس HTP_حالت_NO_SUPPORT;
        }
        work_قطار_تسلسل(octx->ctx->work_قطار, diag_thread_f32, &dctx, n_دھاگے);
    } ورنہ {
        work_قطار_تسلسل(octx->ctx->work_قطار, diag_thread_f32_dma, &dctx, n_دھاگے);
    }

    واپس HTP_حالت_OK;
}

صحیح op_diag(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * dst = octx->dst;

    صحیح err = HTP_حالت_OK;

    انتخاب (dst->type) {
        حالت HTP_TYPE_F32:
            err = op_diag_f32(octx);
            توڑو;
        طےشدہ:
            err = HTP_حالت_NO_SUPPORT;
            توڑو;
    }

    واپس err;
}
