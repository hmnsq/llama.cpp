#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hex-utils.h"
#ضمن "hvx-copy.h"
#ضمن "dma-queue.h"

#تعریف_کرو htp_diag_tensors_preamble                           \
    ثابت بنية htp_tensor * محدود src0 = octx->src[0]; \
    ثابت بنية htp_tensor * محدود dst  = octx->dst;    \
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

بنية htp_diag_context {
    بنية htp_ops_context * octx;
    size_t          src_batch_ناپ;
    size_t          dst_صف_ناپ;
    size_t          src_batch_ناپ_aligned;
    size_t          dst_صف_ناپ_aligned;
    uint32_t        batches_فی_thread;
    uint32_t        کل_batches;
    uint32_t        batch_شروع;
};

#تعریف_کرو htp_diag_preamble                                              \
    بنية htp_diag_context * dctx = (بنية htp_diag_context *) انکڑے; \
    بنية htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

ساكن مضمن فراغ hvx_diag_صف_f32(ثابت عائم * محدود src, عائم * محدود dst,
                                    uint32_t صف_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[صف_idx] = src[صف_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

ساكن فراغ diag_thread_f32_dma(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_diag_preamble;
    dma_قطار * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t ib0 = dctx->batch_شروع + dctx->batches_فی_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_فی_thread, dctx->batch_شروع + dctx->کل_batches);

    إذا (ib0 >= ib1) {
        أرجع;
    }

    ثابت size_t src_batch_ناپ         = dctx->src_batch_ناپ;
    ثابت size_t dst_صف_ناپ           = dctx->dst_صف_ناپ;
    ثابت size_t src_batch_ناپ_aligned = dctx->src_batch_ناپ_aligned;
    ثابت size_t dst_صف_ناپ_aligned   = dctx->dst_صف_ناپ_aligned;

    ثابت dma_addr_t src_انکڑے = src0->انکڑے;
    ثابت dma_addr_t dst_انکڑے = dst->انکڑے;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.انکڑے + (ith * src_batch_ناپ_aligned);
    uint8_t * dst_spad = octx->dst_spad.انکڑے  + (ith * dst_صف_ناپ_aligned);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت dma_addr_t src_batch = src_انکڑے + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_قطار_ڈالو(dma_q,
                       dma_make_انکڑے(src_spad, src_batch),
                       src_batch_ناپ_aligned, src_batch_ناپ, src_batch_ناپ, 1);
        dma_قطار_flush(dma_q);

        ثابت عائم * src_spad_f32 = (ثابت عائم *) src_spad;
        عائم       * dst_spad_f32 = (عائم *) dst_spad;

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_صف_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ثابت dma_addr_t dst_صف = dst_انکڑے + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_قطار_ڈالو(dma_q,
                           dma_make_انکڑے(dst_صف, dst_spad),
                           dst_صف_ناپ, dst_صف_ناپ_aligned, dst_صف_ناپ, 1);
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

ساكن فراغ diag_thread_f32(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    htp_diag_preamble;

    ثابت uint8_t * src_انکڑے = (ثابت uint8_t *) src0->انکڑے;
    uint8_t *       dst_انکڑے = (uint8_t *) dst->انکڑے;

    ثابت uint32_t ib0 = dctx->batch_شروع + dctx->batches_فی_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_فی_thread, dctx->batch_شروع + dctx->کل_batches);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت عائم * محدود src_batch = (ثابت عائم *)(src_انکڑے + i3 * nb03 + i2 * nb02);

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            عائم * محدود dst_صف = (عائم *)(dst_انکڑے + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_صف_f32(src_batch, dst_صف, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(بلند_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحيح op_diag_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t کل_batches = src0->ne[2] * src0->ne[3];
    ثابت size_t dst_batch_ناپ  = dst->ne[1] * dst->nb[1];

    uint32_t batch_شروع = 0;
    uint32_t nbatches    = کل_batches;

    إذا (octx->ctx->mdev.گنتی > 1) {
        bool can_ونڈو = htp_tensor_mdev_انکڑے_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == حجم(عائم)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_فی_chunk = 1;
        إذا (can_ونڈو) {
            إذا (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) == 0)) {
                batches_فی_chunk = 1;
            } وإلا إذا (dst->nb[2] == dst_batch_ناپ &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_فی_chunk = (dst_batch_ناپ > 0) ? (HEX_L2_سطر_ناپ / hex_عاد_اعظم_u32(dst_batch_ناپ, HEX_L2_سطر_ناپ)) : 1;
            } وإلا {
                can_ونڈو = false;
            }
        }

        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(کل_batches, can_ونڈو ? batches_فی_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        batch_شروع = range.شروع;
        nbatches    = range.گنتی;
    }

    إذا (nbatches == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے     = octx->n_دھاگے;

    ثابت size_t src_batch_ناپ         = src0->ne[0] * حجم(عائم);
    ثابت size_t dst_صف_ناپ           = dst->ne[0] * حجم(عائم);
    ثابت size_t src_batch_ناپ_aligned = hex_گیڑا_اتے(src_batch_ناپ, VLEN);
    ثابت size_t dst_صف_ناپ_aligned   = hex_گیڑا_اتے(dst_صف_ناپ, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ثابت size_t spad_فی_thread = src_batch_ناپ_aligned + dst_صف_ناپ_aligned;

    octx->src0_spad.ناپ_فی_thread = src_batch_ناپ_aligned;
    octx->dst_spad.ناپ_فی_thread  = dst_صف_ناپ_aligned;

    octx->src0_spad.ناپ = n_دھاگے * octx->src0_spad.ناپ_فی_thread;
    octx->dst_spad.ناپ  = n_دھاگے * octx->dst_spad.ناپ_فی_thread;

    octx->src0_spad.انکڑے = octx->ctx->vtcm_ادھار;                        octx->src0_spad.src = عدم;
    octx->dst_spad.انکڑے  = octx->src0_spad.انکڑے + octx->src0_spad.ناپ; octx->dst_spad.src  = عدم;

    بنية htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_ناپ         = src_batch_ناپ,
        .dst_صف_ناپ           = dst_صف_ناپ,
        .src_batch_ناپ_aligned = src_batch_ناپ_aligned,
        .dst_صف_ناپ_aligned   = dst_صف_ناپ_aligned,
        .batches_فی_thread     = fastdiv(nbatches + n_دھاگے - 1, &octx->n_دھاگے_div),
        .کل_batches          = nbatches,
        .batch_شروع            = batch_شروع,
    };

    إذا (octx->ctx->vtcm_ناپ < spad_فی_thread * n_دھاگے) {
        إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
        work_قطار_لگاتار(octx->ctx->work_قطار, diag_thread_f32, &dctx, n_دھاگے);
    } وإلا {
        work_قطار_لگاتار(octx->ctx->work_قطار, diag_thread_f32_dma, &dctx, n_دھاگے);
    }

    أرجع HTP_حالت_OK;
}

صحيح op_diag(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * dst = octx->dst;

    صحيح err = HTP_حالت_OK;

    بدل (dst->type) {
        حالة HTP_TYPE_F32:
            err = op_diag_f32(octx);
            اكسر;
        افتراضي:
            err = HTP_حالت_NO_SUPPORT;
            اكسر;
    }

    أرجع err;
}
