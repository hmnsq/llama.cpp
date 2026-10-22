#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#ଅନ୍ତର୍ଭାଵ <string.h>

#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"

ସଂରଚନା htp_repeat_context {
    ସଂରଚନା htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_per_thread;
    uint32_t total_dst_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_size;
};

ସ୍ଥିର ଶୂନ୍ୟ repeat_job_per_thread(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ଅଚଳ ସଂରଚନା htp_repeat_context * rctx = (ଅଚଳ ସଂରଚନା htp_repeat_context *) data;
    ସଂରଚନା htp_ops_context * octx = rctx->octx;
    ଅଚଳ ସଂରଚନା htp_tensor * src = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst = octx->dst;

    ଅଚଳ uint32_t ne00 = src->ne[0];
    ଅଚଳ uint32_t ne01 = src->ne[1];
    ଅଚଳ uint32_t ne02 = src->ne[2];
    ଅଚଳ uint32_t ne03 = src->ne[3];

    ଅଚଳ uint32_t nb00 = src->nb[0];
    ଅଚଳ uint32_t nb01 = src->nb[1];
    ଅଚଳ uint32_t nb02 = src->nb[2];
    ଅଚଳ uint32_t nb03 = src->nb[3];

    ଅଚଳ uint32_t ne0 = dst->ne[0];
    ଅଚଳ uint32_t ne1 = dst->ne[1];
    ଅଚଳ uint32_t ne2 = dst->ne[2];
    ଅଚଳ uint32_t ne3 = dst->ne[3];

    ଅଚଳ uint32_t nb0 = dst->nb[0];
    ଅଚଳ uint32_t nb1 = dst->nb[1];
    ଅଚଳ uint32_t nb2 = dst->nb[2];
    ଅଚଳ uint32_t nb3 = dst->nb[3];

    ଅଚଳ uint32_t nr0 = rctx->nr0;
    ଅଚଳ uint32_t nr1 = rctx->nr1;
    ଅଚଳ uint32_t nr2 = rctx->nr2;
    ଅଚଳ uint32_t nr3 = rctx->nr3;

    ଅଚଳ size_t row_bytes = ne00 * rctx->type_size;

    ଅଚଳ uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    ଅଚଳ uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->total_dst_rows);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    ପାଇଁ (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ଅଚଳ uint32_t i1 = dst_row % ne1;
        ଅଚଳ uint32_t i2 = (dst_row / ne1) % ne2;
        ଅଚଳ uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        ଅଚଳ uint32_t k1 = i1 % ne01;
        ଅଚଳ uint32_t k2 = i2 % ne02;
        ଅଚଳ uint32_t k3 = i3 % ne03;

        ଅଚଳ uint8_t * src_row = (ଅଚଳ uint8_t *) src->data + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        ପାଇଁ (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(ଉପର_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_repeat(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    ଯଦି (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    size_t type_size;
    ବାଛ (src0->type) {
        ସ୍ଥିତି HTP_TYPE_F32: type_size = 4; ଭାଙ୍ଗ;
        ସ୍ଥିତି HTP_TYPE_F16: type_size = 2; ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ uint32_t total_dst_rows  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ଅଚଳ size_t dst_row_size = dst->ne[0] * type_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_row_size, &rows_per_chunk);
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
    }

    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads = octx->n_threads;

    ସଂରଚନା htp_repeat_context rctx = {
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

    FARF(ଉପର_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_queue_run(octx->ctx->work_queue, repeat_job_per_thread, &rctx, n_threads);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}
