#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <HAP_perf.h>

#அந்தர்பாவ <string.h>

#அந்தர்பாவ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"

கட்டமைப்பு htp_repeat_context {
    கட்டமைப்பு htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_per_thread;
    uint32_t total_dst_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_size;
};

நிலையான வெறுமை repeat_job_per_thread(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    மாறிலி கட்டமைப்பு htp_repeat_context * rctx = (மாறிலி கட்டமைப்பு htp_repeat_context *) data;
    கட்டமைப்பு htp_ops_context * octx = rctx->octx;
    மாறிலி கட்டமைப்பு htp_tensor * src = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst = octx->dst;

    மாறிலி uint32_t ne00 = src->ne[0];
    மாறிலி uint32_t ne01 = src->ne[1];
    மாறிலி uint32_t ne02 = src->ne[2];
    மாறிலி uint32_t ne03 = src->ne[3];

    மாறிலி uint32_t nb00 = src->nb[0];
    மாறிலி uint32_t nb01 = src->nb[1];
    மாறிலி uint32_t nb02 = src->nb[2];
    மாறிலி uint32_t nb03 = src->nb[3];

    மாறிலி uint32_t ne0 = dst->ne[0];
    மாறிலி uint32_t ne1 = dst->ne[1];
    மாறிலி uint32_t ne2 = dst->ne[2];
    மாறிலி uint32_t ne3 = dst->ne[3];

    மாறிலி uint32_t nb0 = dst->nb[0];
    மாறிலி uint32_t nb1 = dst->nb[1];
    மாறிலி uint32_t nb2 = dst->nb[2];
    மாறிலி uint32_t nb3 = dst->nb[3];

    மாறிலி uint32_t nr0 = rctx->nr0;
    மாறிலி uint32_t nr1 = rctx->nr1;
    மாறிலி uint32_t nr2 = rctx->nr2;
    மாறிலி uint32_t nr3 = rctx->nr3;

    மாறிலி size_t row_bytes = ne00 * rctx->type_size;

    மாறிலி uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    மாறிலி uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->total_dst_rows);

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    க்கு (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        மாறிலி uint32_t i1 = dst_row % ne1;
        மாறிலி uint32_t i2 = (dst_row / ne1) % ne2;
        மாறிலி uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        மாறிலி uint32_t k1 = i1 % ne01;
        மாறிலி uint32_t k2 = i2 % ne02;
        மாறிலி uint32_t k3 = i3 % ne03;

        மாறிலி uint8_t * src_row = (மாறிலி uint8_t *) src->data + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        க்கு (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(உயர்வு_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

முழு op_repeat(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    எனில் (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }

    size_t type_size;
    தேர்வு (src0->type) {
        நிலை HTP_TYPE_F32: type_size = 4; நிறுத்து;
        நிலை HTP_TYPE_F16: type_size = 2; நிறுத்து;
        இயல்பு:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    மாறிலி uint32_t total_dst_rows  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    மாறிலி size_t dst_row_size = dst->ne[0] * type_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_row_size, &rows_per_chunk);
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        row_start = range.start;
        nrows     = range.மொத்தம்;
    }

    எனில் (nrows == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    மாறிலி uint32_t n_threads = octx->n_threads;

    கட்டமைப்பு htp_repeat_context rctx = {
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

    FARF(உயர்வு_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_queue_run(octx->ctx->work_queue, repeat_job_per_thread, &rctx, n_threads);

    திரும்பு HTP_STATUS_OK;
}
