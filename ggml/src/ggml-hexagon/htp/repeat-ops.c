#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <string.h>

#ضمن "hvx-utils.h"

#define GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

بنية htp_repeat_context {
    بنية htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_per_thread;
    uint32_t total_dst_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_size;
};

ساكن فراغ repeat_job_per_thread(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    ثابت بنية htp_repeat_context * rctx = (ثابت بنية htp_repeat_context *) data;
    بنية htp_ops_context * octx = rctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;

    ثابت uint32_t ne00 = src->ne[0];
    ثابت uint32_t ne01 = src->ne[1];
    ثابت uint32_t ne02 = src->ne[2];
    ثابت uint32_t ne03 = src->ne[3];

    ثابت uint32_t nb00 = src->nb[0];
    ثابت uint32_t nb01 = src->nb[1];
    ثابت uint32_t nb02 = src->nb[2];
    ثابت uint32_t nb03 = src->nb[3];

    ثابت uint32_t ne0 = dst->ne[0];
    ثابت uint32_t ne1 = dst->ne[1];
    ثابت uint32_t ne2 = dst->ne[2];
    ثابت uint32_t ne3 = dst->ne[3];

    ثابت uint32_t nb0 = dst->nb[0];
    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    ثابت uint32_t nr0 = rctx->nr0;
    ثابت uint32_t nr1 = rctx->nr1;
    ثابت uint32_t nr2 = rctx->nr2;
    ثابت uint32_t nr3 = rctx->nr3;

    ثابت size_t row_bytes = ne00 * rctx->type_size;

    ثابت uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    ثابت uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->total_dst_rows);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    لكل (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ثابت uint32_t i1 = dst_row % ne1;
        ثابت uint32_t i2 = (dst_row / ne1) % ne2;
        ثابت uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        ثابت uint32_t k1 = i1 % ne01;
        ثابت uint32_t k2 = i2 % ne02;
        ثابت uint32_t k3 = i3 % ne03;

        ثابت uint8_t * src_row = (ثابت uint8_t *) src->data + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        لكل (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(بەرز_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

صحيح op_repeat(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    إذا (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        أرجع HTP_STATUS_INVAL_PARAMS;
    }

    size_t type_size;
    بدل (src0->type) {
        حالة HTP_TYPE_F32: type_size = 4; اكسر;
        حالة HTP_TYPE_F16: type_size = 2; اكسر;
        افتراضي:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            أرجع HTP_STATUS_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_STATUS_NO_SUPPORT;
    }

    ثابت uint32_t total_dst_rows  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت size_t dst_row_size = dst->ne[0] * type_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    إذا (octx->ctx->mdev.ژماردن > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_row_size, &rows_per_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ژماردن, &octx->ctx->mdev.ژماردن_div);
        row_start = range.start;
        nrows     = range.ژماردن;
    }

    إذا (nrows == 0) {
        أرجع HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads = octx->n_threads;

    بنية htp_repeat_context rctx = {
        .octx             = octx,
        .nr0              = dst->ne[0] / src0->ne[0],
        .nr1              = dst->ne[1] / src0->ne[1],
        .nr2              = dst->ne[2] / src0->ne[2],
        .nr3              = dst->ne[3] / src0->ne[3],
        .nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .total_dst_rows   = nrows,
        .row_start        = row_start,
        .type_size        = type_size,
    };

    FARF(بەرز_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_queue_run(octx->ctx->work_queue, repeat_job_per_thread, &rctx, n_threads);

    أرجع HTP_STATUS_OK;
}
