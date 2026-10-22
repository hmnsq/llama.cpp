#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#تعریف_کن GGML_COMMON_DECL_C
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

#تعریف_کن htp_diag_tensors_preamble                           \
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
    size_t          src_batch_اندازه;
    size_t          dst_ردیف_اندازه;
    size_t          src_batch_اندازه_aligned;
    size_t          dst_ردیف_اندازه_aligned;
    uint32_t        batches_در_هر_thread;
    uint32_t        مجموع_batches;
    uint32_t        batch_شروع;
};

#تعریف_کن htp_diag_preamble                                              \
    بنية htp_diag_context * dctx = (بنية htp_diag_context *) داده; \
    بنية htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

ساكن مضمن فراغ hvx_diag_ردیف_f32(ثابت عائم * محدود src, عائم * محدود dst,
                                    uint32_t ردیف_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[ردیف_idx] = src[ردیف_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

ساكن فراغ diag_thread_f32_dma(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_diag_preamble;
    dma_صف * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t ib0 = dctx->batch_شروع + dctx->batches_در_هر_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_در_هر_thread, dctx->batch_شروع + dctx->مجموع_batches);

    إذا (ib0 >= ib1) {
        أرجع;
    }

    ثابت size_t src_batch_اندازه         = dctx->src_batch_اندازه;
    ثابت size_t dst_ردیف_اندازه           = dctx->dst_ردیف_اندازه;
    ثابت size_t src_batch_اندازه_aligned = dctx->src_batch_اندازه_aligned;
    ثابت size_t dst_ردیف_اندازه_aligned   = dctx->dst_ردیف_اندازه_aligned;

    ثابت dma_addr_t src_داده = src0->داده;
    ثابت dma_addr_t dst_داده = dst->داده;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.داده + (ith * src_batch_اندازه_aligned);
    uint8_t * dst_spad = octx->dst_spad.داده  + (ith * dst_ردیف_اندازه_aligned);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت dma_addr_t src_batch = src_داده + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_صف_افزودن(dma_q,
                       dma_make_داده(src_spad, src_batch),
                       src_batch_اندازه_aligned, src_batch_اندازه, src_batch_اندازه, 1);
        dma_صف_flush(dma_q);

        ثابت عائم * src_spad_f32 = (ثابت عائم *) src_spad;
        عائم       * dst_spad_f32 = (عائم *) dst_spad;

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_ردیف_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ثابت dma_addr_t dst_ردیف = dst_داده + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_صف_افزودن(dma_q,
                           dma_make_داده(dst_ردیف, dst_spad),
                           dst_ردیف_اندازه, dst_ردیف_اندازه_aligned, dst_ردیف_اندازه, 1);
            dma_صف_flush(dma_q);
        }
    }

    FARF(بالا_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ساكن فراغ diag_thread_f32(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    htp_diag_preamble;

    ثابت uint8_t * src_داده = (ثابت uint8_t *) src0->داده;
    uint8_t *       dst_داده = (uint8_t *) dst->داده;

    ثابت uint32_t ib0 = dctx->batch_شروع + dctx->batches_در_هر_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_در_هر_thread, dctx->batch_شروع + dctx->مجموع_batches);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت عائم * محدود src_batch = (ثابت عائم *)(src_داده + i3 * nb03 + i2 * nb02);

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            عائم * محدود dst_ردیف = (عائم *)(dst_داده + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_ردیف_f32(src_batch, dst_ردیف, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(بالا_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحيح op_diag_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t مجموع_batches = src0->ne[2] * src0->ne[3];
    ثابت size_t dst_batch_اندازه  = dst->ne[1] * dst->nb[1];

    uint32_t batch_شروع = 0;
    uint32_t nbatches    = مجموع_batches;

    إذا (octx->ctx->mdev.شمار > 1) {
        bool can_تقسیم = htp_tensor_mdev_داده_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == حجم(عائم)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_در_هر_chunk = 1;
        إذا (can_تقسیم) {
            إذا (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) == 0)) {
                batches_در_هر_chunk = 1;
            } وإلا إذا (dst->nb[2] == dst_batch_اندازه &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_در_هر_chunk = (dst_batch_اندازه > 0) ? (HEX_L2_سطر_اندازه / hex_بزرگترین_مقسوم_علیه_u32(dst_batch_اندازه, HEX_L2_سطر_اندازه)) : 1;
            } وإلا {
                can_تقسیم = false;
            }
        }

        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_batches, can_تقسیم ? batches_در_هر_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        batch_شروع = range.شروع;
        nbatches    = range.شمار;
    }

    إذا (nbatches == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها     = octx->n_نخها;

    ثابت size_t src_batch_اندازه         = src0->ne[0] * حجم(عائم);
    ثابت size_t dst_ردیف_اندازه           = dst->ne[0] * حجم(عائم);
    ثابت size_t src_batch_اندازه_aligned = hex_دور_به_بالا(src_batch_اندازه, VLEN);
    ثابت size_t dst_ردیف_اندازه_aligned   = hex_دور_به_بالا(dst_ردیف_اندازه, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ثابت size_t spad_در_هر_thread = src_batch_اندازه_aligned + dst_ردیف_اندازه_aligned;

    octx->src0_spad.اندازه_در_هر_thread = src_batch_اندازه_aligned;
    octx->dst_spad.اندازه_در_هر_thread  = dst_ردیف_اندازه_aligned;

    octx->src0_spad.اندازه = n_نخها * octx->src0_spad.اندازه_در_هر_thread;
    octx->dst_spad.اندازه  = n_نخها * octx->dst_spad.اندازه_در_هر_thread;

    octx->src0_spad.داده = octx->ctx->vtcm_پایه;                        octx->src0_spad.src = عدم;
    octx->dst_spad.داده  = octx->src0_spad.داده + octx->src0_spad.اندازه; octx->dst_spad.src  = عدم;

    بنية htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_اندازه         = src_batch_اندازه,
        .dst_ردیف_اندازه           = dst_ردیف_اندازه,
        .src_batch_اندازه_aligned = src_batch_اندازه_aligned,
        .dst_ردیف_اندازه_aligned   = dst_ردیف_اندازه_aligned,
        .batches_در_هر_thread     = fastdiv(nbatches + n_نخها - 1, &octx->n_نخها_div),
        .مجموع_batches          = nbatches,
        .batch_شروع            = batch_شروع,
    };

    إذا (octx->ctx->vtcm_اندازه < spad_در_هر_thread * n_نخها) {
        إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            أرجع HTP_وضعیت_NO_SUPPORT;
        }
        work_صف_پیاپی(octx->ctx->work_صف, diag_thread_f32, &dctx, n_نخها);
    } وإلا {
        work_صف_پیاپی(octx->ctx->work_صف, diag_thread_f32_dma, &dctx, n_نخها);
    }

    أرجع HTP_وضعیت_OK;
}

صحيح op_diag(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * dst = octx->dst;

    صحيح err = HTP_وضعیت_OK;

    بدل (dst->type) {
        حالة HTP_TYPE_F32:
            err = op_diag_f32(octx);
            اكسر;
        افتراضي:
            err = HTP_وضعیت_NO_SUPPORT;
            اكسر;
    }

    أرجع err;
}
