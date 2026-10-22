#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#অন্তর্ভাব <string.h>

#অন্তর্ভাব "hvx-utils.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"

কাঠামো htp_repeat_context {
    কাঠামো htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_per_thread;
    uint32_t যোগ_dst_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_আকার;
};

স্থির শূন্য repeat_job_per_thread(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    ধ্রুবক কাঠামো htp_repeat_context * rctx = (ধ্রুবক কাঠামো htp_repeat_context *) তথ্য;
    কাঠামো htp_ops_context * octx = rctx->octx;
    ধ্রুবক কাঠামো htp_tensor * src = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst = octx->dst;

    ধ্রুবক uint32_t ne00 = src->ne[0];
    ধ্রুবক uint32_t ne01 = src->ne[1];
    ধ্রুবক uint32_t ne02 = src->ne[2];
    ধ্রুবক uint32_t ne03 = src->ne[3];

    ধ্রুবক uint32_t nb00 = src->nb[0];
    ধ্রুবক uint32_t nb01 = src->nb[1];
    ধ্রুবক uint32_t nb02 = src->nb[2];
    ধ্রুবক uint32_t nb03 = src->nb[3];

    ধ্রুবক uint32_t ne0 = dst->ne[0];
    ধ্রুবক uint32_t ne1 = dst->ne[1];
    ধ্রুবক uint32_t ne2 = dst->ne[2];
    ধ্রুবক uint32_t ne3 = dst->ne[3];

    ধ্রুবক uint32_t nb0 = dst->nb[0];
    ধ্রুবক uint32_t nb1 = dst->nb[1];
    ধ্রুবক uint32_t nb2 = dst->nb[2];
    ধ্রুবক uint32_t nb3 = dst->nb[3];

    ধ্রুবক uint32_t nr0 = rctx->nr0;
    ধ্রুবক uint32_t nr1 = rctx->nr1;
    ধ্রুবক uint32_t nr2 = rctx->nr2;
    ধ্রুবক uint32_t nr3 = rctx->nr3;

    ধ্রুবক size_t row_bytes = ne00 * rctx->type_আকার;

    ধ্রুবক uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    ধ্রুবক uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->যোগ_dst_rows);

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    জন্য (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ধ্রুবক uint32_t i1 = dst_row % ne1;
        ধ্রুবক uint32_t i2 = (dst_row / ne1) % ne2;
        ধ্রুবক uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        ধ্রুবক uint32_t k1 = i1 % ne01;
        ধ্রুবক uint32_t k2 = i2 % ne02;
        ধ্রুবক uint32_t k3 = i3 % ne03;

        ধ্রুবক uint8_t * src_row = (ধ্রুবক uint8_t *) src->তথ্য + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) dst->তথ্য + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        জন্য (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(উপরের_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

পূর্ণসংখ্যা op_repeat(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    যদি (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    size_t type_আকার;
    নির্বাচন (src0->type) {
        ক্ষেত্র HTP_TYPE_F32: type_আকার = 4; বিরতি;
        ক্ষেত্র HTP_TYPE_F16: type_আকার = 2; বিরতি;
        পূর্বনির্ধারিত:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            ফেরত HTP_STATUS_NO_SUPPORT;
    }

    যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক uint32_t যোগ_dst_rows  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ধ্রুবক size_t dst_row_আকার = dst->ne[0] * type_আকার;

    uint32_t row_start = 0;
    uint32_t nrows     = যোগ_dst_rows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_আকার, (uint32_t) dst_row_আকার, &rows_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(যোগ_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;

    কাঠামো htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .যোগ_dst_rows   = nrows,
        .row_start        = row_start,
        .type_আকার        = type_আকার,
    };

    FARF(উপরের_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_queue_run(octx->ctx->work_queue, repeat_job_per_thread, &rctx, n_threads);

    ফেরত HTP_STATUS_OK;
}
