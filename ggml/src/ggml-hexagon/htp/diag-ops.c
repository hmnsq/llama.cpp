#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#وصف_ڪريو GGML_COMMON_DECL_C
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

#وصف_ڪريو htp_diag_tensors_preamble                           \
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
    size_t          src_batch_سائيز;
    size_t          dst_صف_سائيز;
    size_t          src_batch_سائيز_aligned;
    size_t          dst_صف_سائيز_aligned;
    uint32_t        batches_في_thread;
    uint32_t        ڪل_batches;
    uint32_t        batch_شروعات;
};

#وصف_ڪريو htp_diag_preamble                                              \
    بنية htp_diag_context * dctx = (بنية htp_diag_context *) معلومات; \
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

ساكن فراغ diag_thread_f32_dma(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    htp_diag_preamble;
    dma_قطار * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t ib0 = dctx->batch_شروعات + dctx->batches_في_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_في_thread, dctx->batch_شروعات + dctx->ڪل_batches);

    إذا (ib0 >= ib1) {
        أرجع;
    }

    ثابت size_t src_batch_سائيز         = dctx->src_batch_سائيز;
    ثابت size_t dst_صف_سائيز           = dctx->dst_صف_سائيز;
    ثابت size_t src_batch_سائيز_aligned = dctx->src_batch_سائيز_aligned;
    ثابت size_t dst_صف_سائيز_aligned   = dctx->dst_صف_سائيز_aligned;

    ثابت dma_addr_t src_معلومات = src0->معلومات;
    ثابت dma_addr_t dst_معلومات = dst->معلومات;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.معلومات + (ith * src_batch_سائيز_aligned);
    uint8_t * dst_spad = octx->dst_spad.معلومات  + (ith * dst_صف_سائيز_aligned);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت dma_addr_t src_batch = src_معلومات + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_قطار_ڌڪو(dma_q,
                       dma_make_معلومات(src_spad, src_batch),
                       src_batch_سائيز_aligned, src_batch_سائيز, src_batch_سائيز, 1);
        dma_قطار_flush(dma_q);

        ثابت عائم * src_spad_f32 = (ثابت عائم *) src_spad;
        عائم       * dst_spad_f32 = (عائم *) dst_spad;

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_صف_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ثابت dma_addr_t dst_صف = dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_قطار_ڌڪو(dma_q,
                           dma_make_معلومات(dst_صف, dst_spad),
                           dst_صف_سائيز, dst_صف_سائيز_aligned, dst_صف_سائيز, 1);
            dma_قطار_flush(dma_q);
        }
    }

    FARF(مٿاهون_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ساكن فراغ diag_thread_f32(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    htp_diag_preamble;

    ثابت uint8_t * src_معلومات = (ثابت uint8_t *) src0->معلومات;
    uint8_t *       dst_معلومات = (uint8_t *) dst->معلومات;

    ثابت uint32_t ib0 = dctx->batch_شروعات + dctx->batches_في_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_في_thread, dctx->batch_شروعات + dctx->ڪل_batches);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت عائم * محدود src_batch = (ثابت عائم *)(src_معلومات + i3 * nb03 + i2 * nb02);

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            عائم * محدود dst_صف = (عائم *)(dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_صف_f32(src_batch, dst_صف, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(مٿاهون_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحيح op_diag_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t ڪل_batches = src0->ne[2] * src0->ne[3];
    ثابت size_t dst_batch_سائيز  = dst->ne[1] * dst->nb[1];

    uint32_t batch_شروعات = 0;
    uint32_t nbatches    = ڪل_batches;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        bool can_ورهايو = htp_tensor_mdev_معلومات_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == حجم(عائم)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_في_chunk = 1;
        إذا (can_ورهايو) {
            إذا (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) == 0)) {
                batches_في_chunk = 1;
            } وإلا إذا (dst->nb[2] == dst_batch_سائيز &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_في_chunk = (dst_batch_سائيز > 0) ? (HEX_L2_سٽ_سائيز / hex_وڏو_گڏيل_ونڊيندڙ_u32(dst_batch_سائيز, HEX_L2_سٽ_سائيز)) : 1;
            } وإلا {
                can_ورهايو = false;
            }
        }

        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(ڪل_batches, can_ورهايو ? batches_في_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        batch_شروعات = range.شروعات;
        nbatches    = range.ڳڻپ;
    }

    إذا (nbatches == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا     = octx->n_ڌاڳا;

    ثابت size_t src_batch_سائيز         = src0->ne[0] * حجم(عائم);
    ثابت size_t dst_صف_سائيز           = dst->ne[0] * حجم(عائم);
    ثابت size_t src_batch_سائيز_aligned = hex_ڦيرو_مٿي(src_batch_سائيز, VLEN);
    ثابت size_t dst_صف_سائيز_aligned   = hex_ڦيرو_مٿي(dst_صف_سائيز, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ثابت size_t spad_في_thread = src_batch_سائيز_aligned + dst_صف_سائيز_aligned;

    octx->src0_spad.سائيز_في_thread = src_batch_سائيز_aligned;
    octx->dst_spad.سائيز_في_thread  = dst_صف_سائيز_aligned;

    octx->src0_spad.سائيز = n_ڌاڳا * octx->src0_spad.سائيز_في_thread;
    octx->dst_spad.سائيز  = n_ڌاڳا * octx->dst_spad.سائيز_في_thread;

    octx->src0_spad.معلومات = octx->ctx->vtcm_بنياد;                        octx->src0_spad.src = عدم;
    octx->dst_spad.معلومات  = octx->src0_spad.معلومات + octx->src0_spad.سائيز; octx->dst_spad.src  = عدم;

    بنية htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_سائيز         = src_batch_سائيز,
        .dst_صف_سائيز           = dst_صف_سائيز,
        .src_batch_سائيز_aligned = src_batch_سائيز_aligned,
        .dst_صف_سائيز_aligned   = dst_صف_سائيز_aligned,
        .batches_في_thread     = fastdiv(nbatches + n_ڌاڳا - 1, &octx->n_ڌاڳا_div),
        .ڪل_batches          = nbatches,
        .batch_شروعات            = batch_شروعات,
    };

    إذا (octx->ctx->vtcm_سائيز < spad_في_thread * n_ڌاڳا) {
        إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
        work_قطار_لاڳيتو(octx->ctx->work_قطار, diag_thread_f32, &dctx, n_ڌاڳا);
    } وإلا {
        work_قطار_لاڳيتو(octx->ctx->work_قطار, diag_thread_f32_dma, &dctx, n_ڌاڳا);
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
