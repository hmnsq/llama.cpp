#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

ಸಂರಚನಾ htp_repeat_context {
    ಸಂರಚನಾ htp_ops_context * octx;

    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    uint32_t nrows_per_thread;
    uint32_t total_dst_rows;  // ne1 * ne2 * ne3
    uint32_t row_start;

    size_t   type_size;
};

ಸ್ಥಿರ ಶೂನ್ಯ repeat_job_per_thread(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಧ್ರುವ ಸಂರಚನಾ htp_repeat_context * rctx = (ಧ್ರುವ ಸಂರಚನಾ htp_repeat_context *) data;
    ಸಂರಚನಾ htp_ops_context * octx = rctx->octx;
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst = octx->dst;

    ಧ್ರುವ uint32_t ne00 = src->ne[0];
    ಧ್ರುವ uint32_t ne01 = src->ne[1];
    ಧ್ರುವ uint32_t ne02 = src->ne[2];
    ಧ್ರುವ uint32_t ne03 = src->ne[3];

    ಧ್ರುವ uint32_t nb00 = src->nb[0];
    ಧ್ರುವ uint32_t nb01 = src->nb[1];
    ಧ್ರುವ uint32_t nb02 = src->nb[2];
    ಧ್ರುವ uint32_t nb03 = src->nb[3];

    ಧ್ರುವ uint32_t ne0 = dst->ne[0];
    ಧ್ರುವ uint32_t ne1 = dst->ne[1];
    ಧ್ರುವ uint32_t ne2 = dst->ne[2];
    ಧ್ರುವ uint32_t ne3 = dst->ne[3];

    ಧ್ರುವ uint32_t nb0 = dst->nb[0];
    ಧ್ರುವ uint32_t nb1 = dst->nb[1];
    ಧ್ರುವ uint32_t nb2 = dst->nb[2];
    ಧ್ರುವ uint32_t nb3 = dst->nb[3];

    ಧ್ರುವ uint32_t nr0 = rctx->nr0;
    ಧ್ರುವ uint32_t nr1 = rctx->nr1;
    ಧ್ರುವ uint32_t nr2 = rctx->nr2;
    ಧ್ರುವ uint32_t nr3 = rctx->nr3;

    ಧ್ರುವ size_t row_bytes = ne00 * rctx->type_size;

    ಧ್ರುವ uint32_t row_start = rctx->row_start + rctx->nrows_per_thread * ith;
    ಧ್ರುವ uint32_t row_end   = MIN(row_start + rctx->nrows_per_thread, rctx->row_start + rctx->total_dst_rows);

    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    ಪ್ರತಿ (uint32_t dst_row = row_start; dst_row < row_end; dst_row++) {
        // Decompose flat dst row index into (i1, i2, i3)
        ಧ್ರುವ uint32_t i1 = dst_row % ne1;
        ಧ್ರುವ uint32_t i2 = (dst_row / ne1) % ne2;
        ಧ್ರುವ uint32_t i3 = dst_row / (ne1 * ne2);

        // Map to source indices (tiling)
        ಧ್ರುವ uint32_t k1 = i1 % ne01;
        ಧ್ರುವ uint32_t k2 = i2 % ne02;
        ಧ್ರುವ uint32_t k3 = i3 % ne03;

        ಧ್ರುವ uint8_t * src_row = (ಧ್ರುವ uint8_t *) src->data + k1 * nb01 + k2 * nb02 + k3 * nb03;
        uint8_t * dst_base      = (uint8_t *) dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3;

        // Tile along dimension 0
        ಪ್ರತಿ (uint32_t i0 = 0; i0 < nr0; i0++) {
            uint8_t * dst_ptr = dst_base + i0 * ne00 * nb0;
            memcpy(dst_ptr, src_row, row_bytes);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    FARF(ಮೇಲಿನ_2, "repeat %d/%d: (%ux%ux%ux%u) -> (%ux%ux%ux%u) rows %u:%u\n",
         ith, nth, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         row_start, row_end);
}

ಪೂರ್ಣಾಙ್ಕ op_repeat(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    // Validate that dst dims are multiples of src dims
    ಯದಿ (dst->ne[0] % src0->ne[0] != 0 ||
        dst->ne[1] % src0->ne[1] != 0 ||
        dst->ne[2] % src0->ne[2] != 0 ||
        dst->ne[3] % src0->ne[3] != 0) {
        FARF(ERROR, "repeat: dst dims must be multiples of src dims\n");
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }

    size_t type_size;
    ವಿಕಲ್ಪ (src0->type) {
        ಪಕ್ಷ HTP_TYPE_F32: type_size = 4; ಭಙ್ಗ;
        ಪಕ್ಷ HTP_TYPE_F16: type_size = 2; ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            FARF(ERROR, "repeat: unsupported type %u\n", src0->type);
            ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಯದಿ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಧ್ರುವ uint32_t total_dst_rows  = dst->ne[1] * dst->ne[2] * dst->ne[3];
    ಧ್ರುವ size_t dst_row_size = dst->ne[0] * type_size;

    uint32_t row_start = 0;
    uint32_t nrows     = total_dst_rows;

    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_row_size, &rows_per_chunk);
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_dst_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        row_start = range.start;
        nrows     = range.ಎಣಿಕೆ;
    }

    ಯದಿ (nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಧ್ರುವ uint32_t n_threads = octx->n_threads;

    ಸಂರಚನಾ htp_repeat_context rctx = {
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

    FARF(ಮೇಲಿನ_2, "repeat: (%ux%ux%ux%u) -> (%ux%ux%ux%u) nr=(%u,%u,%u,%u)\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         rctx.nr0, rctx.nr1, rctx.nr2, rctx.nr3);

    work_queue_run(octx->ctx->work_queue, repeat_job_per_thread, &rctx, n_threads);

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}
