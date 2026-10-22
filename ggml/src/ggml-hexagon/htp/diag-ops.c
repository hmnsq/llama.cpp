#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#تعریف_کړه GGML_COMMON_DECL_C
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

#تعریف_کړه htp_diag_tensors_preamble                           \
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
    size_t          dst_لیکه_اندازه;
    size_t          src_batch_اندازه_aligned;
    size_t          dst_لیکه_اندازه_aligned;
    uint32_t        batches_فی_thread;
    uint32_t        ټول_batches;
    uint32_t        batch_پیل;
};

#تعریف_کړه htp_diag_preamble                                              \
    بنية htp_diag_context * dctx = (بنية htp_diag_context *) معلومات; \
    بنية htp_ops_context *  octx = dctx->octx;                       \
    htp_diag_tensors_preamble;

ساكن مضمن فراغ hvx_diag_لیکه_f32(ثابت عائم * محدود src, عائم * محدود dst,
                                    uint32_t لیکه_idx, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0.0f, n);
    dst[لیکه_idx] = src[لیکه_idx];
}

// ---------------------------------------------------------------------------
// Per thread worker: DMA src fetch, compute in VTCM, DMA dst writeback
// ---------------------------------------------------------------------------

ساكن فراغ diag_thread_f32_dma(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_diag_preamble;
    dma_کتار * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t ib0 = dctx->batch_پیل + dctx->batches_فی_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_فی_thread, dctx->batch_پیل + dctx->ټول_batches);

    إذا (ib0 >= ib1) {
        أرجع;
    }

    ثابت size_t src_batch_اندازه         = dctx->src_batch_اندازه;
    ثابت size_t dst_لیکه_اندازه           = dctx->dst_لیکه_اندازه;
    ثابت size_t src_batch_اندازه_aligned = dctx->src_batch_اندازه_aligned;
    ثابت size_t dst_لیکه_اندازه_aligned   = dctx->dst_لیکه_اندازه_aligned;

    ثابت dma_addr_t src_معلومات = src0->معلومات;
    ثابت dma_addr_t dst_معلومات = dst->معلومات;

    // 1 src buffer + 1 dst row buffer per thread in VTCM
    uint8_t * src_spad = octx->src0_spad.معلومات + (ith * src_batch_اندازه_aligned);
    uint8_t * dst_spad = octx->dst_spad.معلومات  + (ith * dst_لیکه_اندازه_aligned);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت dma_addr_t src_batch = src_معلومات + i3 * nb03 + i2 * nb02;

        // Fetch source vector into VTCM
        dma_کتار_ټېل(dma_q,
                       dma_make_معلومات(src_spad, src_batch),
                       src_batch_اندازه_aligned, src_batch_اندازه, src_batch_اندازه, 1);
        dma_کتار_flush(dma_q);

        ثابت عائم * src_spad_f32 = (ثابت عائم *) src_spad;
        عائم       * dst_spad_f32 = (عائم *) dst_spad;

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            // Compute row in VTCM
            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));
            hvx_diag_لیکه_f32(src_spad_f32, dst_spad_f32, i1, ne0);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) (ib * ne1 + i1));

            // Write completed row back to DDR
            ثابت dma_addr_t dst_لیکه = dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dma_کتار_ټېل(dma_q,
                           dma_make_معلومات(dst_لیکه, dst_spad),
                           dst_لیکه_اندازه, dst_لیکه_اندازه_aligned, dst_لیکه_اندازه, 1);
            dma_کتار_flush(dma_q);
        }
    }

    FARF(لوړ_2, "diag-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

// ---------------------------------------------------------------------------
// Per thread worker: Direct HVX (no DMA)
// ---------------------------------------------------------------------------

ساكن فراغ diag_thread_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    htp_diag_preamble;

    ثابت uint8_t * src_معلومات = (ثابت uint8_t *) src0->معلومات;
    uint8_t *       dst_معلومات = (uint8_t *) dst->معلومات;

    ثابت uint32_t ib0 = dctx->batch_پیل + dctx->batches_فی_thread * ith;
    ثابت uint32_t ib1 = MIN(ib0 + dctx->batches_فی_thread, dctx->batch_پیل + dctx->ټول_batches);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    لكل (uint32_t ib = ib0; ib < ib1; ib++) {
        ثابت uint32_t i3 = ib / ne02;
        ثابت uint32_t i2 = ib % ne02;

        ثابت عائم * محدود src_batch = (ثابت عائم *)(src_معلومات + i3 * nb03 + i2 * nb02);

        لكل (uint32_t i1 = 0; i1 < ne1; i1++) {
            عائم * محدود dst_لیکه = (عائم *)(dst_معلومات + i3 * nb3 + i2 * nb2 + i1 * nb1);
            hvx_diag_لیکه_f32(src_batch, dst_لیکه, i1, ne0);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ib0);

    FARF(لوړ_2, "diag-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], ib0, ib1,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

صحيح op_diag_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t ټول_batches = src0->ne[2] * src0->ne[3];
    ثابت size_t dst_batch_اندازه  = dst->ne[1] * dst->nb[1];

    uint32_t batch_پیل = 0;
    uint32_t nbatches    = ټول_batches;

    إذا (octx->ctx->mdev.شمېر > 1) {
        bool can_وېشل = htp_tensor_mdev_معلومات_aligned(dst) && (dst->ne[0] == 1 || dst->nb[0] == حجم(عائم)) && !htp_tensor_is_permuted(dst);
        uint32_t batches_فی_chunk = 1;
        إذا (can_وېشل) {
            إذا (dst->ne[2] > 1 && (dst->nb[2] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) == 0 &&
                (dst->ne[3] <= 1 || (dst->nb[3] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) == 0)) {
                batches_فی_chunk = 1;
            } وإلا إذا (dst->nb[2] == dst_batch_اندازه &&
                       (dst->ne[3] <= 1 || dst->nb[3] == dst->nb[2] * dst->ne[2])) {
                batches_فی_chunk = (dst_batch_اندازه > 0) ? (HEX_L2_کرښه_اندازه / hex_لوی_ګډ_وېشونکی_u32(dst_batch_اندازه, HEX_L2_کرښه_اندازه)) : 1;
            } وإلا {
                can_وېشل = false;
            }
        }

        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ټول_batches, can_وېشل ? batches_فی_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        batch_پیل = range.پیل;
        nbatches    = range.شمېر;
    }

    إذا (nbatches == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه     = octx->n_تارونه;

    ثابت size_t src_batch_اندازه         = src0->ne[0] * حجم(عائم);
    ثابت size_t dst_لیکه_اندازه           = dst->ne[0] * حجم(عائم);
    ثابت size_t src_batch_اندازه_aligned = hex_پړاو_پورته(src_batch_اندازه, VLEN);
    ثابت size_t dst_لیکه_اندازه_aligned   = hex_پړاو_پورته(dst_لیکه_اندازه, VLEN);

    // 1 src buffer + 1 dst row buffer per thread
    ثابت size_t spad_فی_thread = src_batch_اندازه_aligned + dst_لیکه_اندازه_aligned;

    octx->src0_spad.اندازه_فی_thread = src_batch_اندازه_aligned;
    octx->dst_spad.اندازه_فی_thread  = dst_لیکه_اندازه_aligned;

    octx->src0_spad.اندازه = n_تارونه * octx->src0_spad.اندازه_فی_thread;
    octx->dst_spad.اندازه  = n_تارونه * octx->dst_spad.اندازه_فی_thread;

    octx->src0_spad.معلومات = octx->ctx->vtcm_بنسټ;                        octx->src0_spad.src = عدم;
    octx->dst_spad.معلومات  = octx->src0_spad.معلومات + octx->src0_spad.اندازه; octx->dst_spad.src  = عدم;

    بنية htp_diag_context dctx = {
        .octx                   = octx,
        .src_batch_اندازه         = src_batch_اندازه,
        .dst_لیکه_اندازه           = dst_لیکه_اندازه,
        .src_batch_اندازه_aligned = src_batch_اندازه_aligned,
        .dst_لیکه_اندازه_aligned   = dst_لیکه_اندازه_aligned,
        .batches_فی_thread     = fastdiv(nbatches + n_تارونه - 1, &octx->n_تارونه_div),
        .ټول_batches          = nbatches,
        .batch_پیل            = batch_پیل,
    };

    إذا (octx->ctx->vtcm_اندازه < spad_فی_thread * n_تارونه) {
        إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
            أرجع HTP_حالت_NO_SUPPORT;
        }
        work_کتار_پرلپسې(octx->ctx->work_کتار, diag_thread_f32, &dctx, n_تارونه);
    } وإلا {
        work_کتار_پرلپسې(octx->ctx->work_کتار, diag_thread_f32_dma, &dctx, n_تارونه);
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
