#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <HAP_perf.h>

#അന്തര്ഭാവ <string.h>

#അന്തര്ഭാവ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-tensor.h"

ഘടന htp_repeat_context {
    ഘടന htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_per_thread;
    uint32_t total_dst_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_size;
};

സ്ഥിതം ശൂന്യം repeat_job_per_thread(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    സ്ഥിരം ഘടന htp_repeat_context * rctx = (സ്ഥിരം ഘടന htp_repeat_context *) data;
    ഘടന htp_ops_context * octx = rctx->octx;
    സ്ഥിരം ഘടന htp_tensor * src = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst = octx->dst;

    സ്ഥിരം uint32_t ne00 = src->ne[0];
    സ്ഥിരം uint32_t ne01 = src->ne[1];
    സ്ഥിരം uint32_t ne02 = src->ne[2];
    സ്ഥിരം uint32_t ne03 = src->ne[3];

    സ്ഥിരം uint32_t nb00 = src->nb[0];
    സ്ഥിരം uint32_t nb01 = src->nb[1];
    സ്ഥിരം uint32_t nb02 = src->nb[2];
    സ്ഥിരം uint32_t nb03 = src->nb[3];

    സ്ഥിരം uint32_t ne0 = dst->ne[0];
    സ്ഥിരം uint32_t ne1 = dst->ne[1];
    സ്ഥിരം uint32_t ne2 = dst->ne[2];
    സ്ഥിരം uint32_t ne3 = dst->ne[3];

    സ്ഥിരം uint32_t nb0 = dst->nb[0];
    സ്ഥിരം uint32_t nb1 = dst->nb[1];
    സ്ഥിരം uint32_t nb2 = dst->nb[2];
    സ്ഥിരം uint32_t nb3 = dst->nb[3];

    സ്ഥിരം uint32_t nr0 = rctx->nr0;
    സ്ഥിരം uint32_t nr1 = rctx->nr1;
    സ്ഥിരം uint32_t nr2 = rctx->nr2;
    സ്ഥിരം uint32_t nr3 = rctx->nr3;

    സ്ഥിരം size_t row_bytes = ne00 * rctx->type_size;

    സ്ഥിരം uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    സ്ഥിരം uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->total_dst_rows);

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    വേണ്ടി (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        സ്ഥിരം uint32_t i1 = dst_row % ne1;
        സ്ഥിരം uint32_t i2 = (dst_row / ne1) % ne2;
        സ്ഥിരം uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        സ്ഥിരം uint32_t k1 = i1 % ne01;
        സ്ഥിരം uint32_t k2 = i2 % ne02;
        സ്ഥിരം uint32_t k3 = i3 % ne03;

        സ്ഥിരം uint8_t * src_row = (സ്ഥിരം uint8_t *) src->data + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        വേണ്ടി (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(മുകളിലെ_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

പൂർണ്ണസംഖ്യ op_repeat(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    എങ്കിൽ (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }

    size_t type_size;
    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ HTP_TYPE_F32: type_size = 4; നിർത്തുക;
        അവസ്ഥ HTP_TYPE_F16: type_size = 2; നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    എങ്കിൽ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    സ്ഥിരം uint32_t total_dst_rows  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    സ്ഥിരം size_t dst_row_size = dst->ne[0] * type_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_row_size, &rows_per_chunk);
        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        row_start = range.start;
        nrows     = range.എണ്ണം;
    }

    എങ്കിൽ (nrows == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t n_threads = octx->n_threads;

    ഘടന htp_repeat_context rctx = {
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

    FARF(മുകളിലെ_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_queue_run(octx->ctx->work_queue, repeat_job_per_thread, &rctx, n_threads);

    തിരികെ HTP_STATUS_OK;
}
